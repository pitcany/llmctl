"""Tests for the process supervisor and process telemetry."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from llmctl.telemetry.process import (
    ProcessSupervisor,
    get_process_snapshot,
    is_pid_running,
)

SLEEP_CMD = [sys.executable, "-c", "import time; time.sleep(30)"]


def test_launch_and_terminate(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(log_dir=tmp_path)
    result = supervisor.launch(SLEEP_CMD, log_name="sleeper")
    try:
        assert result.pid > 0
        assert supervisor.is_running(result.pid)
        assert result.log_path is not None
        assert Path(result.log_path).exists()
    finally:
        assert supervisor.terminate(result.pid, timeout=5.0) is True
    assert not supervisor.is_running(result.pid)


def test_terminate_already_exited() -> None:
    supervisor = ProcessSupervisor()
    result = supervisor.launch([sys.executable, "-c", "pass"])
    # Give the short-lived process time to exit.
    for _ in range(50):
        if not supervisor.is_running(result.pid):
            break
        time.sleep(0.05)
    assert supervisor.terminate(result.pid) is True


def test_launch_empty_command_raises() -> None:
    supervisor = ProcessSupervisor()
    with pytest.raises(ValueError):
        supervisor.launch([])


def test_launch_missing_binary_raises() -> None:
    supervisor = ProcessSupervisor()
    with pytest.raises(FileNotFoundError):
        supervisor.launch(["this-binary-does-not-exist-xyz"])


def test_is_pid_running_false_for_bogus_pid() -> None:
    assert is_pid_running(None) is False
    assert is_pid_running(2_147_483_000) is False


def test_process_snapshot_for_current_process() -> None:
    import os

    snapshot = get_process_snapshot(os.getpid())
    assert snapshot is not None
    assert snapshot["pid"] == os.getpid()
