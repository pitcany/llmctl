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


def test_interrupt_before_spawn_marks_the_row_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C is a BaseException; with no child the row must not stay STARTING."""

    async def _interrupted(self, plan, on_spawn=None):  # noqa: ANN001, ANN202
        raise KeyboardInterrupt

    monkeypatch.setattr(PythonScriptAdapter, "start", _interrupted)
    service, db, request = _service(tmp_path)
    try:
        with pytest.raises(KeyboardInterrupt):
            service.start(request)
        db.rollback()
        row = db.exec(select(SessionRecord)).one()
        assert row.status == SessionStatus.FAILED
        assert "KeyboardInterrupt" in (row.error or "")
    finally:
        db.close()


def _fail_commits(db: DBSession, armed: list[int]) -> None:
    """Make ``db.commit`` raise while ``armed[0]`` is above zero (one per call)."""
    real_commit = db.commit

    def _commit() -> None:
        if armed[0] > 0:
            armed[0] -= 1
            raise RuntimeError("database is locked")
        real_commit()

    db.commit = _commit  # type: ignore[method-assign]


def test_failed_pid_commit_still_records_the_running_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rollback drops the pid; a FAILED row would orphan the child on its GPU."""
    armed = [0]

    async def _spawn(self, plan, on_spawn=None):  # noqa: ANN001, ANN202
        armed[0] = 1  # the commit inside on_spawn fails once
        on_spawn(2**22 + 1, "/tmp/child.log")
        raise AssertionError("on_spawn should have raised")

    monkeypatch.setattr(PythonScriptAdapter, "start", _spawn)
    service, db, request = _service(tmp_path)
    _fail_commits(db, armed)
    try:
        with pytest.raises(RuntimeError, match="database is locked"):
            service.start(request)
        db.rollback()
        row = db.exec(select(SessionRecord)).one()
        assert row.status == SessionStatus.STARTING
        assert row.pid == 2**22 + 1
        assert row.log_path == "/tmp/child.log"
    finally:
        db.close()


def test_child_that_cannot_be_recorded_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    armed = [0]
    stopped: list[int | None] = []

    async def _spawn(self, plan, on_spawn=None):  # noqa: ANN001, ANN202
        armed[0] = 2  # the commit in on_spawn and the retry both fail
        on_spawn(2**22 + 1, None)

    def _terminate(self, pid, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        stopped.append(pid)
        return True

    monkeypatch.setattr(PythonScriptAdapter, "start", _spawn)
    monkeypatch.setattr(ProcessSupervisor, "terminate", _terminate)
    service, db, request = _service(tmp_path)
    _fail_commits(db, armed)
    try:
        with pytest.raises(RuntimeError, match="database is locked"):
            service.start(request)
        assert stopped == [2**22 + 1]
        db.rollback()
        row = db.exec(select(SessionRecord)).one()
        assert row.pid is None
        assert row.status == SessionStatus.FAILED
    finally:
        db.close()
