"""Tests for backend detection and session cleanup."""

from __future__ import annotations

import time
from pathlib import Path

from sqlmodel import Session

from llmctl.db import ModelRecord, RuntimeName, SessionStatus, get_engine, init_db
from llmctl.schemas import SessionStartRequest
from llmctl.services.backends import detect_backends, missing_backends
from llmctl.services.router import RuntimeRouter
from llmctl.services.sessions import SessionService
from llmctl.telemetry.process import ProcessSupervisor


def test_detect_backends_reports_python_available() -> None:
    backends = {b["backend"]: b for b in detect_backends()}
    assert backends["python"]["available"] is True
    # In CI no LLM runtimes are installed.
    assert "vllm" in missing_backends()


def _wait_dead(supervisor: ProcessSupervisor, pid: int | None) -> None:
    for _ in range(80):
        if not supervisor.is_running(pid):
            return
        time.sleep(0.05)


def test_cleanup_marks_dead_and_reports(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'cleanup.sqlite3'}"
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
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, dry_run=False, force=True
            )
        )
        assert started.status == SessionStatus.RUNNING
        assert started.port is not None

        _wait_dead(router.supervisor, started.pid)
        report = service.cleanup(remove_stale=False)
        assert report["dead_marked"] >= 1
        assert report["active_remaining"] == 0

        # Stale removal deletes the terminal session record.
        report2 = service.cleanup(remove_stale=True)
        assert report2["stale_removed"] >= 1
        assert service.get_session(started.id) is None
