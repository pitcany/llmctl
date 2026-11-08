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
    def is_running(self, pid: int | None) -> bool:
        return False


class _Router:
    def __init__(self, adapter: _WatchingAdapter) -> None:
        self.adapter = adapter
        self.supervisor = _Supervisor()

    def get_adapter(self, runtime: RuntimeName) -> _WatchingAdapter:
        return self.adapter


def _service(tmp_path: Path, adapter: _WatchingAdapter) -> tuple[Session, SessionService]:
    url = f"sqlite:///{tmp_path / 'stopping.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    service = SessionService(db, router=_Router(adapter), probe=lambda u, t: None)
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
