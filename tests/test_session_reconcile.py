"""Test session reconciliation (marking dead sessions) and log tailing."""

from __future__ import annotations

import time
from pathlib import Path

from sqlmodel import Session

from llmctl.db import ModelRecord, RuntimeName, SessionStatus, get_engine, init_db
from llmctl.schemas import SessionStartRequest
from llmctl.services.router import RuntimeRouter
from llmctl.services.sessions import SessionService
from llmctl.telemetry.process import ProcessSupervisor


def _wait_dead(supervisor: ProcessSupervisor, pid: int | None) -> None:
    """Wait until ``pid`` is no longer running (bounded)."""
    for _ in range(80):
        if not supervisor.is_running(pid):
            return
        time.sleep(0.05)


def test_reconcile_marks_dead_session(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'rec.sqlite3'}"
    init_db(url)
    router = RuntimeRouter(supervisor=ProcessSupervisor(log_dir=tmp_path / "logs"))
    script = tmp_path / "quick.py"
    script.write_text("pass\n")
    with Session(get_engine(url)) as db:
        model = ModelRecord(name="quick", runtime=RuntimeName.PYTHON_SCRIPT, path=str(script))
        db.add(model)
        db.commit()
        db.refresh(model)
        service = SessionService(db, router=router)
        started = service.start(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, dry_run=False)
        )
        assert started.status == SessionStatus.RUNNING

        _wait_dead(router.supervisor, started.pid)
        assert service.reconcile() >= 1
        refreshed = service.get_session(started.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.STOPPED


def test_tail_log_returns_content(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'log.sqlite3'}"
    init_db(url)
    router = RuntimeRouter(supervisor=ProcessSupervisor(log_dir=tmp_path / "logs"))
    printer = tmp_path / "printer.py"
    printer.write_text("print('hello-from-session', flush=True)\n")
    with Session(get_engine(url)) as db:
        model = ModelRecord(name="printer", runtime=RuntimeName.PYTHON_SCRIPT, path=str(printer))
        db.add(model)
        db.commit()
        db.refresh(model)
        service = SessionService(db, router=router)
        started = service.start(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, dry_run=False)
        )
        assert started.status == SessionStatus.RUNNING
        _wait_dead(router.supervisor, started.pid)
        content = service.tail_log(started.id, lines=10)
        assert content is not None
        assert "hello-from-session" in content
