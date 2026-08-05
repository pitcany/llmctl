"""A process that survives SIGTERM *and* SIGKILL must not vanish from tracking.

``ProcessSupervisor.terminate`` returns ``False`` when the pid is still alive
after both signals — routine for a vLLM worker wedged in an uninterruptible
driver call. ``stop()`` used to record ``STOPPED`` and erase the pid anyway.

That is the worst possible outcome: ``reconcile()`` finds dead OWNED sessions
*through* ``record.pid``, so a row with no pid is never revisited. The survivor
keeps its ~40 GB of VRAM and its port while being invisible to every llmctl
surface, and ``restart()`` on it would launch a second copy alongside the first.
"""

from __future__ import annotations

from pathlib import Path

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


class _StubAdapter:
    """Adapter whose stop reports whether the process actually died."""

    def __init__(self, runtime: RuntimeName, *, stopped: bool) -> None:
        self.runtime = runtime
        self._stopped = stopped
        self.launches = 0

    async def stop(self, session) -> AdapterStatus:
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.OK if self._stopped else HealthState.DEGRADED,
            message=(
                f"vLLM process {session.pid} "
                f"{'terminated' if self._stopped else 'did not terminate cleanly'}."
            ),
            details={"pid": session.pid, "stopped": self._stopped},
        )

    async def start(self, plan):  # pragma: no cover - must never be reached
        self.launches += 1
        raise AssertionError("must not relaunch while the old process is alive")


class _StubSupervisor:
    """Reports pid liveness; reconcile consults this to settle OWNED rows."""

    def __init__(self, *, alive: bool) -> None:
        self.alive = alive

    def is_running(self, pid: int | None) -> bool:
        return self.alive


class _StubRouter:
    def __init__(self, adapter: _StubAdapter, *, pid_alive: bool = False) -> None:
        self.adapter = adapter
        self.supervisor = _StubSupervisor(alive=pid_alive)

    def get_adapter(self, runtime: RuntimeName) -> _StubAdapter:
        return self.adapter


def _service(tmp_path: Path, *, stopped: bool) -> tuple[Session, SessionService, _StubAdapter]:
    url = f"sqlite:///{tmp_path / 'survivor.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    adapter = _StubAdapter(RuntimeName.VLLM, stopped=stopped)
    service = SessionService(db, router=_StubRouter(adapter), probe=lambda u, t: None)
    return db, service, adapter


def _running_row() -> SessionRecord:
    return SessionRecord(
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
        kind=SessionKind.OWNED,
        endpoint_url="http://127.0.0.1:8003",
        port=8003,
        pid=4242,
    )


def test_survivor_keeps_its_pid_and_is_marked_degraded(tmp_path: Path) -> None:
    db, service, _ = _service(tmp_path, stopped=False)
    try:
        row = _running_row()
        db.add(row)
        db.commit()

        session = service.stop(row.id)

        assert session is not None
        assert session.status == SessionStatus.DEGRADED, "a survivor is not 'stopped'"
        assert session.pid == 4242, "erasing the pid hides it from reconcile forever"
        assert session.error and "did not terminate" in session.error
    finally:
        db.close()


def test_survivor_stays_visible_to_reconcile(tmp_path: Path) -> None:
    """DEGRADED is in _ACTIVE_STATES, so the next pass re-checks the pid."""
    db, service, _ = _service(tmp_path, stopped=False)
    try:
        row = _running_row()
        db.add(row)
        db.commit()
        service.stop(row.id)

        # pid 4242 is not a live process on this machine, so reconcile should
        # now be able to see the row and settle it.
        changed = service.reconcile()

        db.refresh(row)
        assert changed == 1, "reconcile could not see the degraded row"
        assert row.status == SessionStatus.STOPPED
    finally:
        db.close()


def test_clean_stop_is_unchanged(tmp_path: Path) -> None:
    """The normal path must still clear the pid and report STOPPED."""
    db, service, _ = _service(tmp_path, stopped=True)
    try:
        row = _running_row()
        db.add(row)
        db.commit()

        session = service.stop(row.id)

        assert session is not None
        assert session.status == SessionStatus.STOPPED
        assert session.pid is None
        assert session.error is None
    finally:
        db.close()


def test_restart_refuses_to_launch_a_second_copy(tmp_path: Path) -> None:
    """Relaunching over a live process is how you get two servers on one port."""
    db, service, adapter = _service(tmp_path, stopped=False)
    try:
        row = _running_row()
        row.launch_plan = {
            "model_id": "m",
            "runtime": "vllm",
            "gpu_ids": [0],
            "command_preview": "vllm serve m",
        }
        db.add(row)
        db.commit()

        session = service.restart(row.id)

        assert session is not None
        assert session.status == SessionStatus.DEGRADED
        assert session.pid == 4242
        assert adapter.launches == 0
    finally:
        db.close()
