"""A stored PID is only ours while its /proc start time still matches.

PIDs are reused. A session row can outlive its process (crash, reboot), and
the number it kept may later belong to an unrelated process. Liveness alone
would then report the session alive and ``stop`` would signal a stranger.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

from sqlmodel import Session as DBSession

from llmctl.adapters.python_script import PythonScriptAdapter
from llmctl.config import RuntimeConfig
from llmctl.db import (
    ModelRecord,
    RuntimeName,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import LaunchPlan, SessionStartRequest
from llmctl.services.router import RuntimeRouter
from llmctl.services.sessions import SessionService
from llmctl.telemetry.process import ProcessSupervisor, pid_start_time

_SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_launch_records_the_process_start_time(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(log_dir=tmp_path)
    adapter = PythonScriptAdapter(RuntimeConfig(), supervisor)
    plan = LaunchPlan(runtime=RuntimeName.PYTHON_SCRIPT, command=_SLEEP, dry_run=False)
    session = asyncio.run(adapter.start(plan))
    try:
        assert session.pid_start_time is not None
        assert session.pid_start_time == pid_start_time(session.pid)
    finally:
        supervisor.terminate(session.pid)


def test_adapter_stop_does_not_signal_a_reused_pid(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(log_dir=tmp_path)
    adapter = PythonScriptAdapter(RuntimeConfig(), supervisor)
    plan = LaunchPlan(runtime=RuntimeName.PYTHON_SCRIPT, command=_SLEEP, dry_run=False)
    session = asyncio.run(adapter.start(plan))
    # The row remembers an older process that had this number.
    stale = session.model_copy(update={"pid_start_time": "1"})
    try:
        status = asyncio.run(adapter.stop(stale))
        assert "not running" in status.message
        assert supervisor.is_running(session.pid), "stop signalled a process it does not own"
    finally:
        supervisor.terminate(session.pid)


def _service_with_stranger(tmp_path: Path):
    """A RUNNING row whose pid now belongs to an unrelated live process."""
    url = f"sqlite:///{tmp_path / 'pid.sqlite3'}"
    init_db(url)
    router = RuntimeRouter(supervisor=ProcessSupervisor(log_dir=tmp_path / "logs"))
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(30)\n")
    db = DBSession(get_engine(url))
    model = ModelRecord(name="sleeper", runtime=RuntimeName.PYTHON_SCRIPT, path=str(script))
    db.add(model)
    db.commit()
    db.refresh(model)
    service = SessionService(db, router=router)
    started = service.start(
        SessionStartRequest(model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, dry_run=False)
    )
    assert started.status == SessionStatus.RUNNING
    router.supervisor.terminate(started.pid)
    stranger = subprocess.Popen(_SLEEP, start_new_session=True)  # noqa: S603
    record = db.get(SessionRecord, started.id)
    assert record is not None
    assert record.pid_start_time is not None
    record.pid = stranger.pid  # the kernel handed the number to someone else
    db.add(record)
    db.commit()
    return service, db, started.id, stranger


def test_reconcile_treats_a_reused_pid_as_exited(tmp_path: Path) -> None:
    service, db, session_id, stranger = _service_with_stranger(tmp_path)
    try:
        service.reconcile()
        refreshed = service.get_session(session_id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.STOPPED
        assert refreshed.pid is None
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait()
        db.close()


def test_service_stop_does_not_signal_a_reused_pid(tmp_path: Path) -> None:
    service, db, session_id, stranger = _service_with_stranger(tmp_path)
    try:
        service.stop(session_id)
        assert stranger.poll() is None, "stop signalled a process it does not own"
    finally:
        stranger.kill()
        stranger.wait()
        db.close()
