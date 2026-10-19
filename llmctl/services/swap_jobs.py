"""Background swap jobs for the dashboard/API.

A swap runs ``llmctl run <key> --yes`` in a subprocess rather than calling the
swap code in-process: that reuses the CLI swap path, including its rollback,
so the API and the CLI behave identically.

Jobs live in memory only and are lost when the process restarts.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import llmctl.services.swap_jobs as swap_jobs  # self-reference so patches apply

MAX_OUTPUT_LINES = 5000


def run_argv(argv: list[str], on_line: Callable[[str], None]) -> int:
    """Run ``argv``, calling ``on_line`` per output line; return the exit code."""
    env = {**os.environ, "NO_COLOR": "1", "COLUMNS": "200"}
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    try:
        with proc.stdout:
            for line in proc.stdout:
                on_line(line.rstrip("\r\n"))
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def build_run_argv(key: str) -> list[str]:
    return [sys.executable, "-m", "llmctl.main", "run", key, "--yes"]


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class SwapJob:
    name: str
    model: str
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"
    exit_code: int | None = None
    output: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "model": self.model,
            "status": self.status,
            "exit_code": self.exit_code,
            "output": list(self.output),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class JobBusy(Exception):
    """Raised when a swap job is already running."""

    def __init__(self, job: SwapJob) -> None:
        super().__init__(f"swap job {job.job_id} is already running")
        self.job = job


class SwapJobs:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, SwapJob] = {}
        self._threads: dict[str, threading.Thread] = {}

    def start(self, name: str, model: str, argv: list[str]) -> SwapJob:
        with self._lock:
            for existing in self._jobs.values():
                if existing.status == "running":
                    raise JobBusy(existing)
            job = SwapJob(name=name, model=model)
            self._jobs[job.job_id] = job
            thread = threading.Thread(target=self._run, args=(job, argv), daemon=True)
            self._threads[job.job_id] = thread
        thread.start()
        return job

    def _run(self, job: SwapJob, argv: list[str]) -> None:
        def on_line(line: str) -> None:
            with self._lock:
                if len(job.output) < MAX_OUTPUT_LINES:
                    job.output.append(line)

        code: int | None
        try:
            code = swap_jobs.run_argv(argv, on_line)
        except Exception as exc:  # noqa: BLE001 - surfaced in job output
            code = None
            on_line(f"error: {exc}")
        with self._lock:
            job.exit_code = code
            job.status = "succeeded" if code == 0 else "failed"
            job.finished_at = _now()

    def get(self, job_id: str) -> SwapJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def running(self) -> SwapJob | None:
        with self._lock:
            for job in self._jobs.values():
                if job.status == "running":
                    return job
        return None

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
