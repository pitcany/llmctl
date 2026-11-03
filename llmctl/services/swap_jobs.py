"""Background swap jobs for the dashboard/API.

A swap runs ``llmctl run <key> --yes`` in a subprocess rather than calling the
swap code in-process: that reuses the CLI swap path, including its rollback,
so the API and the CLI behave identically.

Jobs used to live in memory only and were lost when the process restarted.
Since FIXPLAN 2026-11-03 §2B/§2C every job also has a persistent operation
record (see ``services/operations.py``): intent is written before the child
starts, the CLI child adopts the record and drives its phases, and the
record survives API restarts so recovery can reconcile unfinished work. The
child -- not the API -- owns the host-wide GPU operation lock (§2A); this
registry never acquires it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import llmctl.services.swap_jobs as swap_jobs  # self-reference so patches apply
from llmctl.services.operations import (
    OP_DB_ENV,
    OP_ID_ENV,
    TERMINAL_PHASES,
    IdempotencyConflict,  # noqa: F401 - re-exported for route handlers
    OperationPhase,
    OperationRecord,
    OperationStore,
    ReconciledOperation,
)

MAX_OUTPUT_LINES = 5000


def run_argv(
    argv: list[str], on_line: Callable[[str], None], extra_env: dict[str, str] | None = None
) -> int:
    """Run ``argv``, calling ``on_line`` per output line; return the exit code."""
    env = {**os.environ, "NO_COLOR": "1", "COLUMNS": "200", **(extra_env or {})}
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
    # The tail, not the head: a swap's error and rollback lines come last.
    output: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_OUTPUT_LINES))
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    #: Persistent operation record id (§2B); None when no store is wired
    #: (legacy in-memory-only usage, e.g. unit tests).
    operation_id: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "operation_id": self.operation_id,
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
            "operation_id": self.operation_id,
        }


class JobBusy(Exception):
    """Raised when a swap job is already running."""

    def __init__(self, job: SwapJob) -> None:
        super().__init__(f"swap job {job.job_id} is already running")
        self.job = job


class StopInProgress(Exception):
    """Raised when a swap is asked for while a stop is changing what serves."""


class OperationConflict(Exception):
    """A persistent unfinished operation (another process) blocks this swap."""

    def __init__(self, record: OperationRecord) -> None:
        owner = record.owner or {}
        super().__init__(
            f"operation {record.id} ({record.kind} {record.target}, phase "
            f"{record.phase}) is still running under pid {owner.get('pid')}"
        )
        self.record = record


@dataclass
class StartOutcome:
    """Result of :meth:`SwapJobs.start`."""

    job: SwapJob | None
    operation: OperationRecord | None
    #: True when an idempotent replay returned the existing operation
    #: without starting a second switch.
    replayed: bool = False

    @property
    def job_id(self) -> str | None:
        """The started job's id (None on idempotent replay)."""
        return self.job.job_id if self.job is not None else None


class SwapJobs:
    def __init__(self, database_url: str | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, SwapJob] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stopping = 0
        self._store: OperationStore | None = (
            OperationStore(database_url) if database_url else None
        )
        #: §2C: reconciliation results from construction time (API restart).
        self.reconciliation: list[ReconciledOperation] = []
        if self._store is not None:
            self.reconciliation = self._store.reconcile()

    @property
    def store(self) -> OperationStore | None:
        return self._store

    def start(
        self,
        name: str,
        model: str,
        argv: list[str],
    ) -> SwapJob:
        """Start a swap job (historical contract: returns the job)."""
        return self.start_with_operation(name, model, argv).job

    def start_with_operation(
        self,
        name: str,
        model: str,
        argv: list[str],
        idempotency_key: str | None = None,
        previous_state: dict[str, Any] | None = None,
    ) -> StartOutcome:
        """Start a swap job with its persistent operation record (§2B)."""
        store = self._store
        record: OperationRecord | None = None
        if store is not None and idempotency_key:
            # Idempotency wins over busy: repeating a key with the same
            # request must return the existing operation, not a 409.
            record, created = store.begin(
                kind="swap",
                target=name,
                interface="api",
                purpose=f"swap to {model}",
                request={"name": name, "model": model, "argv": argv},
                previous_state=previous_state,
                idempotency_key=idempotency_key,
            )
            if not created:
                return StartOutcome(job=None, operation=record, replayed=True)
        try:
            with self._lock:
                for existing in self._jobs.values():
                    if existing.status == "running":
                        raise JobBusy(existing)
                if self._stopping:
                    raise StopInProgress("a stop is in progress")
                if store is not None:
                    blocked = self._unfinished_elsewhere(
                        exclude_id=record.id if record else None
                    )
                    if blocked is not None:
                        raise OperationConflict(blocked)
                    if record is None:
                        record, _ = store.begin(
                            kind="swap",
                            target=name,
                            interface="api",
                            purpose=f"swap to {model}",
                            request={"name": name, "model": model, "argv": argv},
                            previous_state=previous_state,
                        )
                job = SwapJob(
                    name=name, model=model, operation_id=record.id if record else None
                )
                self._jobs[job.job_id] = job
                thread = threading.Thread(target=self._run, args=(job, argv), daemon=True)
                self._threads[job.job_id] = thread
            try:
                thread.start()
            except BaseException as exc:
                # Left "running", the job would refuse every later swap until restart.
                self._finish(job, None, f"error: could not start the swap thread: {exc}")
                if record is not None and store is not None:
                    store.finish(
                        record.id or "",
                        OperationPhase.FAILED,
                        error=f"could not start the swap thread: {exc}",
                    )
                raise
        except BaseException:
            # The record was created but nothing will run: finalize it so it
            # cannot wedge future swaps as a phantom unfinished operation.
            if record is not None and store is not None:
                store.finish(
                    record.id or "", OperationPhase.FAILED, error="swap was not started"
                )
            raise
        return StartOutcome(job=job, operation=record)

    def _unfinished_elsewhere(self, exclude_id: str | None = None) -> OperationRecord | None:
        """An unfinished operation not owned by one of our live jobs."""
        assert self._store is not None
        own = {
            job.operation_id for job in self._jobs.values() if job.status == "running"
        }
        for record in self._store.unfinished():
            if record.id in own or record.id == exclude_id:
                continue
            return record
        return None

    @contextmanager
    def stopping(self) -> Iterator[None]:
        """Hold off new swaps while a stop runs; raise JobBusy if one is running.

        The check and the hold happen under one lock, so a swap cannot start
        between "nothing is running" and the stop itself.
        """
        with self._lock:
            for existing in self._jobs.values():
                if existing.status == "running":
                    raise JobBusy(existing)
            self._stopping += 1
        try:
            yield
        finally:
            with self._lock:
                self._stopping -= 1

    def _run(self, job: SwapJob, argv: list[str]) -> None:
        def on_line(line: str) -> None:
            with self._lock:
                job.output.append(line)
            if self._store is not None and job.operation_id:
                self._store.append_output(job.operation_id, line)

        extra_env: dict[str, str] = {}
        if self._store is not None and job.operation_id:
            # Hand the record to the child: it owns the lock and the phases.
            extra_env[OP_ID_ENV] = job.operation_id
            extra_env[OP_DB_ENV] = self._store.database_url

        code: int | None = None
        error: str | None = None
        try:
            # Injected runners (tests) may predate extra_env; detect once
            # instead of retrying and risking a double child run.
            import inspect

            params = inspect.signature(swap_jobs.run_argv).parameters
            supports_env = "extra_env" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if extra_env and supports_env:
                code = swap_jobs.run_argv(argv, on_line, extra_env=extra_env)
            else:
                code = swap_jobs.run_argv(argv, on_line)
        except BaseException as exc:  # noqa: BLE001 - surfaced in job output
            error = f"error: {exc}"
        finally:
            self._finish(job, code, error)
            if self._store is not None and job.operation_id:
                record = self._store.get(job.operation_id)
                if record is None or record.phase not in {
                    phase.value for phase in TERMINAL_PHASES
                }:
                    # The child sets richer outcomes itself; this is the
                    # fallback when it died before finishing the record.
                    self._store.finish(
                        job.operation_id,
                        OperationPhase.FAILED if code != 0 else OperationPhase.COMPLETED,
                        error=error,
                    )

    def _finish(self, job: SwapJob, code: int | None, error: str | None) -> None:
        with self._lock:
            if error:
                job.output.append(error)
            job.exit_code = code
            job.finished_at = _now()
            # Last, so a reader that sees a final status also sees finished_at.
            job.status = "succeeded" if code == 0 else "failed"

    def get(self, job_id: str) -> SwapJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def running(self) -> SwapJob | None:
        with self._lock:
            for job in self._jobs.values():
                if job.status == "running":
                    return job
        return None

    def operation_for(self, job: SwapJob) -> OperationRecord | None:
        """The persistent record behind ``job`` (§2D), when a store is wired."""
        if self._store is None or not job.operation_id:
            return None
        return self._store.get(job.operation_id)

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
