"""``stop`` and ``restart`` pass through ``STOPPING`` while the process is being terminated.

Before 2026-11-08 ``STOPPING`` was declared but never entered: a row stayed
``running`` for the whole SIGTERM/SIGKILL wait. Now every reader sees the
stop in flight, and a stop that cannot run puts the row back where it was.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session

from llmctl.db import (
    RuntimeName,
    SessionKind,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import AdapterStatus, HealthState
from llmctl.services.sessions import SessionService


class _WatchingAdapter:
    """Records the status the session carried when ``stop`` was asked."""

    def __init__(self, *, stopped: bool = True, fail: bool = False) -> None:
        self.runtime = RuntimeName.VLLM
        self.stopped = stopped
        self.fail = fail
        self.seen: list[SessionStatus] = []

    async def stop(self, session) -> AdapterStatus:
        self.seen.append(session.status)
        if self.fail:
            raise RuntimeError("adapter exploded")
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.OK if self.stopped else HealthState.DEGRADED,
            message="terminated" if self.stopped else "did not terminate cleanly",
            details={"pid": session.pid, "stopped": self.stopped},
        )


class _Supervisor:
    def __init__(self, *, alive: bool = False) -> None:
        self.alive = alive

    def is_running(self, pid: int | None) -> bool:
        return self.alive


class _Router:
    def __init__(self, adapter: _WatchingAdapter, *, pid_alive: bool = False) -> None:
        self.adapter = adapter
        self.supervisor = _Supervisor(alive=pid_alive)

    def get_adapter(self, runtime: RuntimeName) -> _WatchingAdapter:
        return self.adapter


class _Systemctl:
    """A systemctl whose ``stop`` raises, as a timeout or a missing binary would."""

    def is_user_unit(self, unit: str | None) -> bool:
        return True

    def stop(self, unit: str, *, user: bool = False):
        raise OSError("systemctl: command not found")


def _service(
    tmp_path: Path,
    adapter: _WatchingAdapter,
    *,
    pid_alive: bool = False,
    endpoint_alive: bool = False,
    systemctl=None,
) -> tuple[Session, SessionService]:
    url = f"sqlite:///{tmp_path / 'stopping.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    service = SessionService(
        db,
        router=_Router(adapter, pid_alive=pid_alive),
        probe=lambda u, t: ["m"] if endpoint_alive else None,
        systemctl=systemctl,
    )
    return db, service


def _running_row() -> SessionRecord:
    return SessionRecord(
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
        kind=SessionKind.OWNED,
        endpoint_url="http://127.0.0.1:8003",
        port=8003,
        pid=4242,
    )


def test_stop_marks_the_row_stopping_while_the_adapter_works(tmp_path: Path) -> None:
    adapter = _WatchingAdapter()
    db, service = _service(tmp_path, adapter)
    try:
        row = _running_row()
        db.add(row)
        db.commit()

        session = service.stop(row.id)

        assert adapter.seen == [SessionStatus.STOPPING]
        assert session is not None and session.status is SessionStatus.STOPPED
        # The trail records both moves with their reasons.
        assert [(c.old, c.new, c.why) for c in session.status_history] == [
            (SessionStatus.RUNNING, SessionStatus.STOPPING, "stop requested"),
            (SessionStatus.STOPPING, SessionStatus.STOPPED, "stop"),
        ]
    finally:
        db.close()


def test_survivor_goes_from_stopping_to_degraded(tmp_path: Path) -> None:
    adapter = _WatchingAdapter(stopped=False)
    db, service = _service(tmp_path, adapter)
    try:
        row = _running_row()
        db.add(row)
        db.commit()

        session = service.stop(row.id)

        assert adapter.seen == [SessionStatus.STOPPING]
        assert session is not None and session.status is SessionStatus.DEGRADED
        assert session.pid == 4242
    finally:
        db.close()


def test_stop_that_cannot_run_restores_the_previous_status(tmp_path: Path) -> None:
    adapter = _WatchingAdapter(fail=True)
    db, service = _service(tmp_path, adapter)
    try:
        row = _running_row()
        db.add(row)
        db.commit()

        with pytest.raises(RuntimeError, match="adapter exploded"):
            service.stop(row.id)

        db.refresh(row)
        assert row.status is SessionStatus.RUNNING, "a failed stop must not leave a row stopping"
    finally:
        db.close()


def test_reconcile_settles_a_stopping_row_whose_process_died(tmp_path: Path) -> None:
    """The stop's own process died after committing STOPPING; the pid is gone."""
    db, service = _service(tmp_path, _WatchingAdapter())
    try:
        row = _running_row()
        row.status = SessionStatus.STOPPING
        db.add(row)
        db.commit()

        assert service.reconcile() == 1
        db.refresh(row)
        assert row.status is SessionStatus.STOPPED
        assert row.error is None, "a stop landing is not an unexpected exit"
        assert row.status_history[-1]["why"] == "stop finished"
    finally:
        db.close()


def test_reconcile_returns_a_stopping_row_that_still_answers_to_running(tmp_path: Path) -> None:
    db, service = _service(tmp_path, _WatchingAdapter(), pid_alive=True, endpoint_alive=True)
    try:
        row = _running_row()
        row.status = SessionStatus.STOPPING
        db.add(row)
        db.commit()

        assert service.reconcile() == 1
        db.refresh(row)
        assert row.status is SessionStatus.RUNNING
    finally:
        db.close()


def test_adopted_stop_restores_status_when_systemctl_raises(tmp_path: Path) -> None:
    db, service = _service(tmp_path, _WatchingAdapter(), systemctl=_Systemctl())
    try:
        row = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.RUNNING,
            kind=SessionKind.ADOPTED,
            endpoint_url="http://127.0.0.1:8003",
            systemd_unit="vllm-tp.service",
        )
        db.add(row)
        db.commit()

        with pytest.raises(OSError, match="systemctl"):
            service.stop(row.id, stop_unit=True)

        db.refresh(row)
        assert row.status is SessionStatus.RUNNING
        assert [e["why"] for e in row.status_history] == [
            "systemctl stop requested",
            "systemctl stop could not run",
        ]
    finally:
        db.close()


def test_restart_passes_through_stopping_then_replans(tmp_path: Path) -> None:
    adapter = _WatchingAdapter()
    db, service = _service(tmp_path, adapter)
    try:
        row = _running_row()  # no launch plan: restart stops at PLANNED
        db.add(row)
        db.commit()

        session = service.restart(row.id)

        assert adapter.seen == [SessionStatus.STOPPING]
        assert session is not None and session.status is SessionStatus.PLANNED
    finally:
        db.close()
