"""A stored PID is only ours while its /proc start time still matches.

PIDs are reused. A session row can outlive its process (crash, reboot), and
the number it kept may later belong to an unrelated process. Liveness alone
would then report the session alive and ``stop`` would signal a stranger.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
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
from llmctl.telemetry import process as process_module
from llmctl.telemetry.process import ProcessSupervisor, is_same_process, pid_start_time

# The identity comes from /proc; without it there is nothing to compare.
pytestmark = pytest.mark.skipif(
    pid_start_time(os.getpid()) is None, reason="needs a readable /proc/<pid>/stat"
)

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
    # Two processes can start in the same clock tick; make the mismatch certain.
    record.pid_start_time = f"not-{pid_start_time(stranger.pid)}"
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


def test_identity_includes_the_boot_id() -> None:
    """Start time is ticks since boot; the same value after a reboot is another process."""
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    token = pid_start_time(os.getpid())
    assert token is not None
    assert token.startswith(f"{boot_id}:")
    ticks = token.split(":", 1)[1]
    assert not is_same_process(os.getpid(), f"00000000-0000-0000-0000-000000000000:{ticks}")


def test_terminate_does_not_escalate_to_a_reused_pid(tmp_path: Path, monkeypatch) -> None:
    """If the pid changes hands while terminate waits, SIGKILL must not follow."""
    code = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(30)"
    )
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", code], stdout=subprocess.PIPE, start_new_session=True
    )
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == b"ready"
        answers = iter([True])  # ours at the first check, someone else's afterwards
        monkeypatch.setattr(
            process_module, "is_same_process", lambda pid, start: next(answers, False)
        )
        supervisor = ProcessSupervisor(log_dir=tmp_path)
        assert supervisor.terminate(child.pid, timeout=0.3, start_time="recorded") is True
        assert child.poll() is None, "terminate sent SIGKILL to a process it no longer owns"
    finally:
        child.kill()
        child.wait()
