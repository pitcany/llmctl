"""Integration test for real session start/stop control."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from llmctl.db import (
    ModelRecord,
    RuntimeName,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import SessionStartRequest
from llmctl.services.events import list_events
from llmctl.services.router import RuntimeRouter
from llmctl.services.sessions import SessionService
from llmctl.telemetry.process import ProcessSupervisor


def test_real_python_session_start_and_stop(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'sess.sqlite3'}"
    init_db(url)
    router = RuntimeRouter(supervisor=ProcessSupervisor(log_dir=tmp_path / "logs"))
    with Session(get_engine(url)) as db:
        model = ModelRecord(
            name="sleeper",
            runtime=RuntimeName.PYTHON_SCRIPT,
            path=str(_write_sleep_script(tmp_path)),
        )
        db.add(model)
        db.commit()
        db.refresh(model)

        service = SessionService(db, router=router)
        started = service.start(
            SessionStartRequest(
                model_id=model.id,
                runtime=RuntimeName.PYTHON_SCRIPT,
                dry_run=False,
            )
        )
        try:
            assert started.status == SessionStatus.RUNNING
            assert started.pid is not None
            assert router.supervisor.is_running(started.pid)
        finally:
            stopped = service.stop(started.id)
        assert stopped is not None
        assert stopped.status == SessionStatus.STOPPED
        assert not router.supervisor.is_running(started.pid)

        events = list_events(db)
        assert any("stopped" in event.message.lower() for event in events)


def test_dry_run_session_does_not_launch(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'dry.sqlite3'}"
    init_db(url)
    with Session(get_engine(url)) as db:
        service = SessionService(db)
        session = service.start(
            SessionStartRequest(model_id="x", runtime=RuntimeName.VLLM, dry_run=True)
        )
        assert session.status == SessionStatus.PLANNED
        assert session.pid is None


def _write_sleep_script(tmp_path: Path) -> Path:
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(30)\n")
    return script
