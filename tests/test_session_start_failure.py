"""A launch that raises must not leave the session row in STARTING.

Reconcile only fails a STARTING row whose recorded pid is dead. A row with no
pid at all would stay STARTING forever and keep its GPU claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session as DBSession
from sqlmodel import select

from llmctl.adapters.python_script import PythonScriptAdapter
from llmctl.db import (
    ModelRecord,
    RuntimeName,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import SessionStartRequest
from llmctl.services.router import RuntimeRouter
from llmctl.services.sessions import SessionService
from llmctl.telemetry.process import ProcessSupervisor


def _service(tmp_path: Path) -> tuple[SessionService, DBSession, SessionStartRequest]:
    url = f"sqlite:///{tmp_path / 'start.sqlite3'}"
    init_db(url)
    router = RuntimeRouter(supervisor=ProcessSupervisor(log_dir=tmp_path / "logs"))
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n")
    db = DBSession(get_engine(url))
    model = ModelRecord(name="sleeper", runtime=RuntimeName.PYTHON_SCRIPT, path=str(script))
    db.add(model)
    db.commit()
    db.refresh(model)
    request = SessionStartRequest(
        model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, dry_run=False
    )
    return SessionService(db, router=router), db, request


def test_start_that_raises_before_spawn_marks_the_row_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(self, plan, on_spawn=None):  # noqa: ANN001, ANN202
        raise OSError("exec format error")

    monkeypatch.setattr(PythonScriptAdapter, "start", _boom)
    service, db, request = _service(tmp_path)
    try:
        with pytest.raises(OSError, match="exec format error"):
            service.start(request)
        db.rollback()
        rows = db.exec(select(SessionRecord)).all()
        assert len(rows) == 1
        assert rows[0].status == SessionStatus.FAILED
        assert "exec format error" in (rows[0].error or "")
    finally:
        db.close()


def test_start_that_raises_after_spawn_leaves_the_row_for_reconcile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a pid on the row, reconcile decides: the child may still be alive."""

    async def _boom(self, plan, on_spawn=None):  # noqa: ANN001, ANN202
        if on_spawn is not None:
            on_spawn(2**22 + 1, None)  # above pid_max: never a live process
        raise RuntimeError("readiness probe crashed")

    monkeypatch.setattr(PythonScriptAdapter, "start", _boom)
    service, db, request = _service(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="readiness probe crashed"):
            service.start(request)
        db.rollback()
        row = db.exec(select(SessionRecord)).one()
        assert row.status == SessionStatus.STARTING
        assert row.pid == 2**22 + 1
        service.reconcile()
        db.refresh(row)
        assert row.status == SessionStatus.FAILED
    finally:
        db.close()
