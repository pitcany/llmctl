"""Tests for start readiness gating, owned-session reconcile, and DB pragmas.

Covers the control-plane correctness fixes:

* a non-dry-run start of a server runtime is only ``RUNNING`` once the
  endpoint answers (spawn alone is not success);
* reconcile promotes ``STARTING`` → ``RUNNING`` and demotes ``RUNNING`` →
  ``DEGRADED`` from real endpoint probes, and marks startup deaths ``FAILED``;
* an adopted endpoint that is reachable but momentarily serves an empty model
  list is not flapped to ``STOPPED``;
* file-backed SQLite engines run in WAL mode with a busy timeout.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import text
from sqlmodel import Session

from llmctl.adapters.llama_cpp import LlamaCppAdapter
from llmctl.config import RuntimeConfig
from llmctl.db import (
    RuntimeName,
    SessionKind,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import LaunchPlan
from llmctl.services.sessions import SessionService
from llmctl.telemetry.process import ProcessSupervisor


class _FakeSupervisor(ProcessSupervisor):
    """Supervisor stub with a scripted PID liveness answer."""

    def __init__(self, alive: bool = True) -> None:
        super().__init__()
        self.alive = alive
        self.launched: list[list[str]] = []

    def launch(self, command, env=None, cwd=None, log_name=None):  # type: ignore[override]
        from llmctl.db import utcnow
        from llmctl.telemetry.process import LaunchResult

        self.launched.append([str(part) for part in command])
        return LaunchResult(pid=4242, command=list(command), log_path=None, started_at=utcnow())

    def is_running(self, pid):  # type: ignore[override]
        return self.alive


def _gated_adapter(alive: bool, endpoint_alive: bool) -> LlamaCppAdapter:
    config = RuntimeConfig(readiness_timeout_s=0.2, readiness_poll_interval_s=0.05)
    adapter = LlamaCppAdapter(config, _FakeSupervisor(alive=alive))

    async def fake_alive(_endpoint):
        return endpoint_alive

    adapter._endpoint_alive = fake_alive  # type: ignore[method-assign]
    return adapter


def _plan(**overrides) -> LaunchPlan:
    defaults = dict(
        runtime=RuntimeName.LLAMA_CPP,
        command=["llama-server", "-m", "model.gguf"],
        endpoint_url="http://127.0.0.1:9999",
        dry_run=False,
    )
    defaults.update(overrides)
    return LaunchPlan(**defaults)


def test_start_running_only_after_endpoint_answers() -> None:
    adapter = _gated_adapter(alive=True, endpoint_alive=True)
    session = asyncio.run(adapter.start(_plan()))
    assert session.status == SessionStatus.RUNNING
    assert session.error is None


def test_start_returns_starting_when_endpoint_not_ready() -> None:
    adapter = _gated_adapter(alive=True, endpoint_alive=False)
    session = asyncio.run(adapter.start(_plan()))
    assert session.status == SessionStatus.STARTING
    assert session.error is None


def test_start_fails_when_process_dies_before_ready(tmp_path: Path) -> None:
    adapter = _gated_adapter(alive=False, endpoint_alive=False)
    log = tmp_path / "boot.log"
    log.write_text("loading weights\nCUDA out of memory\n")

    async def run() -> None:
        session = await adapter.start(_plan())
        assert session.status == SessionStatus.FAILED
        assert "exited before becoming ready" in (session.error or "")

    asyncio.run(run())


def test_startup_failure_message_includes_log_tail(tmp_path: Path) -> None:
    from llmctl.adapters._common import _tail_log

    log = tmp_path / "boot.log"
    log.write_text("line1\nline2\nCUDA out of memory\n")
    tail = _tail_log(str(log))
    assert tail is not None
    assert "CUDA out of memory" in tail
    assert _tail_log(str(tmp_path / "missing.log")) is None


def test_python_script_start_is_not_readiness_gated() -> None:
    from llmctl.adapters.python_script import PythonScriptAdapter

    adapter = PythonScriptAdapter(RuntimeConfig(), _FakeSupervisor(alive=True))
    plan = _plan(runtime=RuntimeName.PYTHON_SCRIPT, command=["python3", "job.py"])
    session = asyncio.run(adapter.start(plan))
    assert session.status == SessionStatus.RUNNING


def _service_with(tmp_path: Path, *, probe, pid_alive: bool) -> tuple[SessionService, Session]:
    url = f"sqlite:///{tmp_path / 'reconcile.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    service = SessionService(db, probe=probe)
    service.router.supervisor = _FakeSupervisor(alive=pid_alive)
    return service, db


def _owned_record(
    status: SessionStatus, endpoint: str | None = "http://127.0.0.1:9999"
) -> SessionRecord:
    return SessionRecord(
        runtime=RuntimeName.LLAMA_CPP,
        status=status,
        kind=SessionKind.OWNED,
        pid=4242,
        endpoint_url=endpoint,
    )


def test_reconcile_promotes_starting_to_running(tmp_path: Path) -> None:
    service, db = _service_with(tmp_path, probe=lambda url, t: ["m"], pid_alive=True)
    record = _owned_record(SessionStatus.STARTING)
    db.add(record)
    db.commit()
    assert service.reconcile() == 1
    db.refresh(record)
    assert record.status == SessionStatus.RUNNING
    assert record.error is None


def test_reconcile_demotes_running_to_degraded(tmp_path: Path) -> None:
    service, db = _service_with(tmp_path, probe=lambda url, t: None, pid_alive=True)
    record = _owned_record(SessionStatus.RUNNING)
    db.add(record)
    db.commit()
    assert service.reconcile() == 1
    db.refresh(record)
    assert record.status == SessionStatus.DEGRADED
    assert "not responding" in (record.error or "")


def test_reconcile_recovers_degraded_to_running(tmp_path: Path) -> None:
    service, db = _service_with(tmp_path, probe=lambda url, t: ["m"], pid_alive=True)
    record = _owned_record(SessionStatus.DEGRADED)
    db.add(record)
    db.commit()
    assert service.reconcile() == 1
    db.refresh(record)
    assert record.status == SessionStatus.RUNNING


def test_reconcile_leaves_starting_alone_while_loading(tmp_path: Path) -> None:
    service, db = _service_with(tmp_path, probe=lambda url, t: None, pid_alive=True)
    record = _owned_record(SessionStatus.STARTING)
    db.add(record)
    db.commit()
    assert service.reconcile() == 0
    db.refresh(record)
    assert record.status == SessionStatus.STARTING


def test_reconcile_marks_starting_death_failed(tmp_path: Path) -> None:
    service, db = _service_with(tmp_path, probe=lambda url, t: None, pid_alive=False)
    record = _owned_record(SessionStatus.STARTING)
    db.add(record)
    db.commit()
    assert service.reconcile() == 1
    db.refresh(record)
    assert record.status == SessionStatus.FAILED
    assert "before becoming ready" in (record.error or "")


def test_adopted_empty_model_list_stays_running(tmp_path: Path) -> None:
    service, db = _service_with(tmp_path, probe=lambda url, t: [], pid_alive=True)
    record = SessionRecord(
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
        kind=SessionKind.ADOPTED,
        endpoint_url="http://127.0.0.1:8003",
    )
    db.add(record)
    db.commit()
    service.reconcile()
    db.refresh(record)
    assert record.status == SessionStatus.RUNNING


def test_file_backed_sqlite_uses_wal_and_busy_timeout(tmp_path: Path) -> None:
    engine = get_engine(f"sqlite:///{tmp_path / 'wal.sqlite3'}")
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 30000
