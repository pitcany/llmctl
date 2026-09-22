"""A failed termination is an *event*, not a *state* — and must survive reconcile.

PR #168 recorded a process that outlived SIGTERM and SIGKILL by marking the row
``DEGRADED`` and keeping its pid, so ``reconcile`` could still see it. That fixed
the invisibility, but overloaded ``DEGRADED``, which already meant something
else: "process alive, endpoint dead", a condition that is *supposed* to
auto-recover. ``_reconcile_owned`` promotes ``DEGRADED`` → ``RUNNING`` and clears
``error`` as soon as the endpoint answers, so the record that a stop had failed
was erased by the next `llmctl sessions`.

The split: ``status`` describes the world right now (and a survivor that is
serving really is running — reconcile is right about that), while
``terminate_failed_at`` records that an operation was attempted and did not take.
Reconcile owns the first and never touches the second. A successful stop, or a
fresh launch, clears it.
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
    def __init__(self, runtime: RuntimeName, *, stopped: bool) -> None:
        self.runtime = runtime
        self._stopped = stopped

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


class _StubSupervisor:
    def __init__(self, *, alive: bool) -> None:
        self.alive = alive

    def is_running(self, pid: int | None) -> bool:
        return self.alive


class _StubRouter:
    def __init__(self, adapter: _StubAdapter, *, pid_alive: bool = True) -> None:
        self.adapter = adapter
        self.supervisor = _StubSupervisor(alive=pid_alive)

    def get_adapter(self, runtime: RuntimeName) -> _StubAdapter:
        return self.adapter


def _service(
    tmp_path: Path,
    *,
    stopped: bool,
    pid_alive: bool = True,
    served: list[str] | None = None,
):
    url = f"sqlite:///{tmp_path / 'term.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    router = _StubRouter(_StubAdapter(RuntimeName.VLLM, stopped=stopped), pid_alive=pid_alive)
    service = SessionService(db, router=router, probe=lambda u, t: served)
    return db, service


def _row() -> SessionRecord:
    return SessionRecord(
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
        kind=SessionKind.OWNED,
        endpoint_url="http://127.0.0.1:8003",
        port=8003,
        pid=4242,
    )


def test_a_failed_stop_is_recorded_out_of_band(tmp_path: Path) -> None:
    db, service = _service(tmp_path, stopped=False)
    try:
        row = _row()
        db.add(row)
        db.commit()

        session = service.stop(row.id)

        assert session is not None
        assert session.terminate_failed_at is not None
        assert session.pid == 4242
    finally:
        db.close()


def test_reconcile_does_not_erase_it_when_the_process_recovers(tmp_path: Path) -> None:
    """The whole point: the survivor keeps serving, so reconcile says RUNNING.

    That is truthful about the process. It must not also mean "the stop worked".
    """
    db, service = _service(tmp_path, stopped=False, pid_alive=True, served=["qwen"])
    try:
        row = _row()
        db.add(row)
        db.commit()
        service.stop(row.id)

        service.reconcile()

        db.refresh(row)
        assert row.status == SessionStatus.RUNNING, "reconcile should report reality"
        assert row.terminate_failed_at is not None, (
            "reconcile erased the record that the stop failed"
        )
    finally:
        db.close()


def test_a_clean_stop_records_nothing(tmp_path: Path) -> None:
    db, service = _service(tmp_path, stopped=True)
    try:
        row = _row()
        db.add(row)
        db.commit()

        session = service.stop(row.id)

        assert session is not None
        assert session.terminate_failed_at is None
        assert session.status == SessionStatus.STOPPED
    finally:
        db.close()


def test_a_later_successful_stop_clears_the_marker(tmp_path: Path) -> None:
    """Once the process really dies, the failure is history, not current."""
    db, service = _service(tmp_path, stopped=False)
    try:
        row = _row()
        db.add(row)
        db.commit()
        service.stop(row.id)
        db.refresh(row)
        assert row.terminate_failed_at is not None

        # Second attempt, this time the process goes away.
        service.router.adapter._stopped = True
        session = service.stop(row.id)

        assert session is not None
        assert session.terminate_failed_at is None
        assert session.status == SessionStatus.STOPPED
    finally:
        db.close()


def test_degraded_still_means_endpoint_down_for_an_ordinary_session(tmp_path: Path) -> None:
    """The original meaning is untouched: alive pid + dead endpoint → DEGRADED.

    And with no failed termination, nothing is marked.
    """
    db, service = _service(tmp_path, stopped=True, pid_alive=True, served=None)
    try:
        row = _row()
        db.add(row)
        db.commit()

        service.reconcile()

        db.refresh(row)
        assert row.status == SessionStatus.DEGRADED
        assert row.terminate_failed_at is None
    finally:
        db.close()


def test_a_fresh_launch_clears_the_marker(tmp_path: Path) -> None:
    """A new process supersedes the one that would not die."""
    db, service = _service(tmp_path, stopped=False)
    try:
        row = _row()
        row.launch_plan = {
            "model_id": "m",
            "runtime": "vllm",
            "gpu_ids": [0],
            "command_preview": "vllm serve m",
            "dry_run": True,  # keep the test off the process supervisor
        }
        db.add(row)
        db.commit()
        service.stop(row.id)
        db.refresh(row)
        assert row.terminate_failed_at is not None

        service.router.adapter._stopped = True
        service.restart(row.id)

        db.refresh(row)
        assert row.terminate_failed_at is None
    finally:
        db.close()


def test_the_cli_surfaces_it_next_to_the_status(tmp_path: Path, monkeypatch) -> None:
    """Reconcile says RUNNING; the operator must still see the stop failed."""
    from typer.testing import CliRunner

    from llmctl import cli
    from llmctl.schemas import Session as SessionSchema

    stuck = SessionSchema(
        id="sess-1",
        model_id="m1",
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
        pid=4242,
        terminate_failed_at=__import__("datetime").datetime(2026, 8, 6, 3, 0, 0),
    )

    class _Svc:
        def __init__(self, db, *a, **k) -> None:
            pass

        def reconcile(self) -> int:
            return 0

        def list_sessions(self) -> list:
            return [stuck]

    class _NullCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    class _NullRegistry:
        def list_models(self) -> list:
            return []

    monkeypatch.setattr(cli, "_session", lambda: _NullCtx())
    monkeypatch.setattr(cli, "SessionService", _Svc)
    # The Model column resolves a model_id through the registry; this test's
    # db is a null stub, so the registry gets one too.
    monkeypatch.setattr(cli, "RegistryService", lambda db: _NullRegistry())

    out = CliRunner().invoke(cli.app, ["sessions"])

    assert out.exit_code == 0, out.output
    assert "running" in out.output
    assert "stop failed" in out.output, out.output


def test_an_existing_database_gains_the_column(tmp_path: Path) -> None:
    """New columns must land on databases created before they existed.

    `SQLModel.create_all` never alters an existing table; `apply_migrations`
    does. Without it, every query against an older registry would fail with
    "no such column" -- a much worse outcome than the bug being fixed.
    """
    import sqlalchemy

    url = f"sqlite:///{tmp_path / 'old.sqlite3'}"
    engine = get_engine(url)
    # A pre-existing sessions table that predates the column.
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text(
                "CREATE TABLE sessions ("
                "id VARCHAR PRIMARY KEY, runtime VARCHAR, status VARCHAR, "
                "created_at DATETIME, updated_at DATETIME)"
            )
        )
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO sessions (id, runtime, status, created_at, updated_at) "
                "VALUES ('legacy', 'VLLM', 'STOPPED', '2026-01-01', '2026-01-01')"
            )
        )

    init_db(url)

    with Session(get_engine(url)) as db:
        row = db.get(SessionRecord, "legacy")
        assert row is not None, "the pre-existing row did not survive migration"
        assert row.terminate_failed_at is None
