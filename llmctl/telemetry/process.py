"""Process telemetry and supervision helpers.

This module provides two responsibilities:

* Lightweight, read-only process *telemetry* (snapshots, candidate discovery).
* A :class:`ProcessSupervisor` that performs real process *control* — launching
  detached child processes with captured logs and terminating them gracefully
  (SIGTERM then SIGKILL on the whole process group).

All control operations are explicit; callers (services/adapters) decide when to
invoke them based on ``dry_run``/``safe_mode`` policy.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from llmctl.db import utcnow

_DEAD_STATES = {psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD}


def get_process_snapshot(pid: int) -> dict[str, Any] | None:
    """Return a safe process snapshot, or ``None`` if the process is unavailable."""
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            return {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cmdline": process.cmdline(),
                "cpu_percent": process.cpu_percent(interval=None),
                "memory_rss_bytes": process.memory_info().rss,
                "create_time": process.create_time(),
            }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def find_candidate_runtime_processes() -> list[dict[str, Any]]:
    """Return candidate local LLM runtime processes using command heuristics."""
    keywords = ("vllm", "llama", "ollama", "lmstudio", "lms ", "python")
    results: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "status"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            name = proc.info.get("name") or ""
            haystack = f"{name} {cmdline}".lower()
            if any(keyword in haystack for keyword in keywords):
                results.append(dict(proc.info))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return results


def is_pid_running(pid: int | None) -> bool:
    """Return True when ``pid`` refers to a live, non-zombie process."""
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() not in _DEAD_STATES
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


@dataclass(slots=True)
class LaunchResult:
    """Result of a successful process launch."""

    pid: int
    command: list[str]
    log_path: str | None
    started_at: datetime


class ProcessSupervisor:
    """Launch and supervise detached child processes with log capture.

    The supervisor launches each command in its own session/process group so the
    full child tree can be signalled on termination. Standard output and error
    are redirected to a per-launch log file under ``log_dir``.
    """

    def __init__(self, log_dir: Path | str | None = None) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else None

    def _resolve_log_path(self, log_name: str | None) -> Path | None:
        """Return the log file path for a launch, creating parent dirs."""
        if self.log_dir is None or log_name is None:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / f"{log_name}.log"

    def launch(
        self,
        command: Sequence[str],
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        log_name: str | None = None,
    ) -> LaunchResult:
        """Launch ``command`` as a detached child process.

        Args:
            command: Argument vector; ``command[0]`` must be an executable.
            env: Extra environment variables merged over the current environment.
            cwd: Working directory for the child process.
            log_name: Base name for the captured log file (when a log dir is set).

        Returns:
            A :class:`LaunchResult` describing the started process.

        Raises:
            ValueError: If ``command`` is empty.
            FileNotFoundError: If the executable cannot be found.
        """
        argv = [str(part) for part in command]
        if not argv:
            raise ValueError("Cannot launch an empty command.")

        full_env = dict(os.environ)
        if env:
            full_env.update({str(k): str(v) for k, v in env.items()})

        log_path = self._resolve_log_path(log_name)
        stdout: Any = subprocess.DEVNULL
        log_handle = None
        if log_path is not None:
            log_handle = log_path.open("ab")
            stdout = log_handle

        try:
            process = subprocess.Popen(  # noqa: S603 - explicit, validated argv
                argv,
                env=full_env,
                cwd=str(cwd) if cwd else None,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            if log_handle is not None:
                log_handle.close()

        return LaunchResult(
            pid=process.pid,
            command=argv,
            log_path=str(log_path) if log_path else None,
            started_at=utcnow(),
        )

    def is_running(self, pid: int | None) -> bool:
        """Return True when the supervised pid is alive."""
        return is_pid_running(pid)

    def terminate(
        self,
        pid: int | None,
        timeout: float = 10.0,
        kill_group: bool = True,
    ) -> bool:
        """Gracefully terminate a process, escalating to SIGKILL on timeout.

        Args:
            pid: Process id to terminate.
            timeout: Seconds to wait after SIGTERM before sending SIGKILL.
            kill_group: When True, signal the entire process group.

        Returns:
            True if the process is no longer running after the call.
        """
        if not is_pid_running(pid):
            return True
        assert pid is not None  # for type-checkers; guarded above

        children = self._live_children(pid)
        self._signal(pid, signal.SIGTERM, kill_group)
        if self._wait_for_exit(pid, timeout):
            self._reap_orphans(children)
            return True

        self._signal(pid, signal.SIGKILL, kill_group)
        stopped = self._wait_for_exit(pid, timeout=5.0)
        self._reap_orphans(children)
        return stopped

    @staticmethod
    def _live_children(pid: int) -> list[psutil.Process]:
        """Snapshot the child tree before signalling, for orphan cleanup."""
        try:
            return psutil.Process(pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    @staticmethod
    def _reap_orphans(children: list[psutil.Process], timeout: float = 5.0) -> None:
        """Kill children that survived the group signal (e.g. after setsid).

        Group signalling covers the normal case; this catches workers that
        detached into their own process group and would otherwise be orphaned
        holding GPU memory.
        """
        survivors = [child for child in children if child.is_running()]
        if not survivors:
            return
        for child in survivors:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _, alive = psutil.wait_procs(survivors, timeout=timeout)
        for child in alive:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _, still_alive = psutil.wait_procs(alive, timeout=2.0)
        for child in still_alive:
            logging.getLogger(__name__).warning(
                "Orphaned child process %s survived SIGTERM/SIGKILL; "
                "it may still hold GPU memory.",
                child.pid,
            )

    @staticmethod
    def _signal(pid: int, sig: signal.Signals, kill_group: bool) -> None:
        """Send ``sig`` to a pid or its process group, ignoring missing targets."""
        try:
            if kill_group:
                try:
                    os.killpg(os.getpgid(pid), sig)
                    return
                except (ProcessLookupError, PermissionError):
                    pass
            os.kill(pid, sig)
        except ProcessLookupError:
            return

    @staticmethod
    def _wait_for_exit(pid: int, timeout: float) -> bool:
        """Poll until the pid exits or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not is_pid_running(pid):
                return True
            time.sleep(0.05)
        return not is_pid_running(pid)
