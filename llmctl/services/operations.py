"""Persistent operation records for GPU lifecycle changes (FIXPLAN §2B/§2C).

An operation is one managed mutation of what serves: a swap, a stop, a
start. Its record is written *before* side effects (intent) and updated as
the executor moves through phases, so a crash, an API restart, or a child
that outlives its parent always leaves a discoverable, honest trail.

Guarantees encoded here:

* Phases are explicit (preparing/stopping/loading/checking_readiness/
  restoring) and terminal states are distinct: ``completed``, ``failed``,
  ``interrupted`` (we could not tell), and ``needs_attention`` (the world
  was left different from where it started -- e.g. a failed swap whose
  rollback also failed). A model serving again after restoration is
  recorded in ``recovery``, never by relabeling the failure.
* Idempotency: repeating a key with the same request returns the existing
  operation; reusing the key for a different request is refused.
* Owner identity is recorded (pid, hostname, interface, pid start time) so
  recovery can distinguish a surviving child from a recycled PID.
* Output is bounded and sanitized: no credentials, no unbounded logs.
* Recovery is conservative: it verifies liveness (proc exists, start time
  matches, cmdline looks like llmctl) before declaring anything. It never
  replays stop/start side effects and never infers an end from a missing
  lock or a dead parent alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlmodel import Session, select
from typer import Exit as TyperExit

from llmctl.db import OperationRecord, ensure_tables, get_engine, new_id

#: Environment variables used to hand an operation from a parent (the API)
#: to the CLI child that actually executes it.
OP_ID_ENV = "LLMCTL_OPERATION_ID"
OP_DB_ENV = "LLMCTL_OPERATION_DB_URL"

MAX_OUTPUT_LINES = 200
MAX_LINE_CHARS = 400

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|password|secret)\s*[=:]\s*\S+"),
)


class OperationPhase(StrEnum):
    """Explicit phases of a mutating operation."""

    PREPARING = "preparing"
    STOPPING = "stopping"
    LOADING = "loading"
    CHECKING_READINESS = "checking_readiness"
    RESTORING = "restoring"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    NEEDS_ATTENTION = "needs_attention"


#: Phases that mean the operation is over (successfully or not).
TERMINAL_PHASES = frozenset(
    {
        OperationPhase.COMPLETED,
        OperationPhase.FAILED,
        OperationPhase.INTERRUPTED,
        OperationPhase.NEEDS_ATTENTION,
    }
)


class IdempotencyConflict(Exception):
    """An idempotency key was reused for a different request."""


@dataclass(frozen=True)
class ReconciledOperation:
    """Outcome of reconciling one unfinished operation at startup."""

    operation_id: str
    outcome: str  # "reattached" | "interrupted" | "needs_attention"
    detail: str


def utcnow() -> datetime:
    return datetime.now(UTC)


def sanitize_line(line: str) -> str:
    """Bound length and scrub credential-shaped content from output."""
    safe = line[:MAX_LINE_CHARS]
    for pattern in _SECRET_PATTERNS:
        safe = pattern.sub(r"\1=<redacted>", safe)
    return safe


def fingerprint(request: Any) -> str:
    """Stable hash of the request payload, for idempotency comparison."""
    blob = json.dumps(request, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _pid_start_time(pid: int) -> str | None:
    """/proc start time (field 22 of stat) — the PID-reuse guard."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            stat = fh.read()
        # comm may contain spaces/parens; take everything after the last ')'.
        rest = stat[stat.rindex(b")") + 2 :].split()
        return rest[19].decode()
    except (OSError, IndexError, ValueError):
        return None


def _proc_cmdline(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return [part.decode(errors="replace") for part in fh.read().split(b"\0") if part]
    except OSError:
        return []


def owner_identity(interface: str, purpose: str) -> dict[str, Any]:
    """Record who is executing: pid, host, interface, and pid start time."""
    pid = os.getpid()
    return {
        "pid": pid,
        "hostname": socket.gethostname(),
        "interface": interface,
        "purpose": purpose,
        "pid_start_time": _pid_start_time(pid),
        "started_cmdline": _proc_cmdline(pid)[:6],
    }


def _owner_alive(owner: dict[str, Any]) -> str:
    """Classify the owner process: 'alive', 'dead', or 'unknown'."""
    pid = owner.get("pid")
    if not isinstance(pid, int):
        return "unknown"
    start = _pid_start_time(pid)
    if start is None:
        return "dead"
    recorded = owner.get("pid_start_time")
    if recorded and str(recorded) != start:
        # Same number, different process: the original is gone and the PID
        # was recycled. Treating it as alive would attribute someone else's
        # process to our operation.
        return "dead"
    cmdline = _proc_cmdline(pid)
    joined = " ".join(cmdline)
    if cmdline and "llmctl" not in joined and "python" not in joined:
        return "recycled"
    return "alive"


class OperationStore:
    """Create, update, reconcile, and read operation records."""

    def __init__(self, database_url: str) -> None:
        # One engine for the store's lifetime: every get_engine() call can
        # be a distinct database (in-memory URLs always are), so opening a
        # fresh engine per _session() would read tableless copies.
        self.database_url = database_url
        self._engine = get_engine(database_url)
        ensure_tables(self._engine, database_url)

    def _session(self) -> Session:
        return Session(self._engine)

    def begin(
        self,
        *,
        kind: str,
        target: str,
        interface: str,
        purpose: str,
        request: Any,
        previous_state: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        job_id: str | None = None,
    ) -> tuple[OperationRecord, bool]:
        """Persist intent before any side effect.

        Returns ``(record, created)``; ``created`` is False when an
        idempotent replay returned the existing record unchanged.
        """
        fp = fingerprint(request)
        with self._session() as db:
            if idempotency_key:
                existing = db.exec(
                    select(OperationRecord).where(
                        OperationRecord.idempotency_key == idempotency_key
                    )
                ).first()
                if existing is not None:
                    if existing.request_fingerprint != fp:
                        raise IdempotencyConflict(
                            f"idempotency key {idempotency_key!r} was already used "
                            f"for a different request ({existing.kind} {existing.target})"
                        )
                    return existing, False
            record = OperationRecord(
                id=new_id(),
                kind=kind,
                target=target,
                phase=OperationPhase.PREPARING.value,
                idempotency_key=idempotency_key,
                job_id=job_id,
                request_fingerprint=fp,
                request=request if isinstance(request, dict) else {"value": str(request)},
                previous_state=previous_state or {},
                owner=owner_identity(interface, purpose),
                output=[],
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            return record, True

    def adopt(self, operation_id: str, *, interface: str, purpose: str) -> OperationRecord | None:
        """Take ownership of an unfinished record (parent handed it to us)."""
        with self._session() as db:
            record = db.get(OperationRecord, operation_id)
            if record is None or OperationPhase(record.phase) in TERMINAL_PHASES:
                return None
            record.owner = owner_identity(interface, purpose)
            record.updated_at = utcnow()
            db.add(record)
            db.commit()
            db.refresh(record)
            return record

    def set_phase(self, operation_id: str, phase: OperationPhase) -> None:
        with self._session() as db:
            record = db.get(OperationRecord, operation_id)
            if record is None:
                return
            record.phase = phase.value
            record.updated_at = utcnow()
            db.add(record)
            db.commit()

    def append_output(self, operation_id: str, line: str) -> None:
        with self._session() as db:
            record = db.get(OperationRecord, operation_id)
            if record is None:
                return
            output = list(record.output or [])
            output.append(sanitize_line(line))
            record.output = output[-MAX_OUTPUT_LINES:]
            db.add(record)
            db.commit()

    def finish(
        self,
        operation_id: str,
        phase: OperationPhase,
        *,
        error: str | None = None,
        recovery: dict[str, Any] | None = None,
    ) -> None:
        if phase not in TERMINAL_PHASES:
            raise ValueError(f"{phase} is not a terminal phase")
        with self._session() as db:
            record = db.get(OperationRecord, operation_id)
            if record is None:
                return
            record.phase = phase.value
            record.error = error
            record.recovery = recovery or {}
            record.finished_at = utcnow()
            record.updated_at = utcnow()
            db.add(record)
            db.commit()

    def get(self, operation_id: str) -> OperationRecord | None:
        with self._session() as db:
            return db.get(OperationRecord, operation_id)

    def get_job(self, job_id: str) -> OperationRecord | None:
        """Return the operation belonging to a persisted browser job ID."""
        with self._session() as db:
            return db.exec(
                select(OperationRecord).where(OperationRecord.job_id == job_id)
            ).first()

    def record_job_exit(self, operation_id: str, exit_code: int | None) -> None:
        """Save the child exit code without overwriting its recovery outcome."""
        with self._session() as db:
            record = db.get(OperationRecord, operation_id)
            if record is not None:
                record.job_exit_code = exit_code
                db.add(record)
                db.commit()

    def unfinished(self) -> list[OperationRecord]:
        with self._session() as db:
            records = db.exec(select(OperationRecord)).all()
            return [
                r
                for r in records
                if r.phase not in {p.value for p in TERMINAL_PHASES}
            ]

    def reconcile(self) -> list[ReconciledOperation]:
        """Classify unfinished records against live processes (§2C).

        * owner verified alive -> ``reattached``: the operation is still
          running somewhere; conflicting work must be refused, not started.
        * owner dead/recycled -> ``interrupted`` with concrete guidance.
          A record whose phase was ``restoring`` becomes ``needs_attention``:
          we were putting the previous model back and cannot claim it did.
        Nothing is replayed; the lock alone proves nothing about liveness.
        """
        results: list[ReconciledOperation] = []
        for record in self.unfinished():
            owner = dict(record.owner or {})
            state = _owner_alive(owner)
            phase = OperationPhase(record.phase)
            if state == "alive":
                results.append(
                    ReconciledOperation(
                        operation_id=record.id or "",
                        outcome="reattached",
                        detail=(
                            f"owner pid {owner.get('pid')} still running "
                            f"({record.kind} {record.target}); refusing conflicting work"
                        ),
                    )
                )
                continue
            guidance = (
                f"owner pid {owner.get('pid')} is {state}; the {record.kind} of "
                f"{record.target} stopped at phase {phase.value}. Check `llmctl status` "
                "and the serving endpoint before retrying; nothing was auto-replayed."
            )
            terminal = OperationPhase.INTERRUPTED
            recovery = {"state": state, "guidance": guidance}
            if phase == OperationPhase.RESTORING:
                terminal = OperationPhase.NEEDS_ATTENTION
                recovery["guidance"] = (
                    "the operation failed AND rollback was in flight when it was "
                    "interrupted; verify what is serving (`llmctl status`) before "
                    "starting anything else."
                )
            self.finish(record.id or "", terminal, error=record.error, recovery=recovery)
            results.append(
                ReconciledOperation(
                    operation_id=record.id or "",
                    outcome=terminal.value,
                    detail=recovery["guidance"],
                )
            )
        return results


# -- executor-side helpers (used by the CLI run path) ------------------------
#
# A thread-local "current operation" lets cli_swap helpers mark phases
# without threading the record through every signature. Without a bound
# operation every helper is a no-op, so unchanged call sites stay safe.

_current = threading.local()


def bind_current(store: OperationStore, operation_id: str) -> None:
    """Make ``operation_id`` the current operation for this thread."""
    _current.store = store
    _current.operation_id = operation_id
    _current.restore_failed = []


def clear_current() -> None:
    _current.__dict__.clear()


def current_operation_id() -> str | None:
    return getattr(_current, "operation_id", None)


def mark_phase(phase: OperationPhase) -> None:
    """Update the current operation's phase (no-op when none is bound)."""
    store = getattr(_current, "store", None)
    op_id = getattr(_current, "operation_id", None)
    if store is None or op_id is None:
        return
    store.set_phase(op_id, phase)


def note_restore_failure(labels: list[str] | tuple[str, ...]) -> None:
    """Record that rollback could not restart ``labels``."""
    if labels:
        _current.restore_failed = list(getattr(_current, "restore_failed", [])) + list(
            labels
        )


def restore_failures() -> list[str]:
    return list(getattr(_current, "restore_failed", []))


def finish_current(
    ok_phase: OperationPhase = OperationPhase.COMPLETED,
    failed_phase: OperationPhase = OperationPhase.FAILED,
    *,
    error: str | None = None,
) -> None:
    """Finalize the current operation (failed-with-bad-rollback upgrades to
    ``needs_attention`` so failed restoration never reads as a clean fail)."""
    store = getattr(_current, "store", None)
    op_id = getattr(_current, "operation_id", None)
    if store is None or op_id is None:
        return
    if failed_phase == OperationPhase.FAILED and restore_failures():
        failed_phase = OperationPhase.NEEDS_ATTENTION
    recovery = {}
    if restore_failures():
        recovery["not_restored"] = restore_failures()
    store.finish(
        op_id, ok_phase if error is None else failed_phase, error=error, recovery=recovery
    )


def interrupt_current(error: str) -> None:
    """Finalize the bound operation after an exception, retaining recovery facts."""
    store = getattr(_current, "store", None)
    op_id = current_operation_id()
    if store is None or op_id is None:
        return
    record = store.get(op_id)
    restoring = record is not None and record.phase == OperationPhase.RESTORING.value
    failed = restore_failures()
    recovery: dict[str, Any] = {
        "guidance": "The runner was interrupted; check `llmctl status` and the serving "
        "endpoint before retrying. Nothing was auto-replayed.",
    }
    if failed:
        recovery["not_restored"] = failed
    store.finish(
        op_id,
        OperationPhase.NEEDS_ATTENTION if failed or restoring else OperationPhase.INTERRUPTED,
        error=error,
        recovery=recovery,
    )


@contextmanager
def cli_operation_guard(
    *,
    db_url: str,
    kind: str,
    target: str,
    model: str,
) -> Iterator[OperationRecord]:
    """Bind one persistent operation to a CLI mutation (``llmctl run``).

    Adopts the record a parent API handed over via environment when present,
    otherwise creates one. The record is finalized on every exit path:
    SystemExit(0) completes, SystemExit(non-zero) fails (upgraded to
    needs_attention when rollback could not restore), a crash or Ctrl-C is
    recorded as interrupted with recovery guidance — never silently dropped.
    """
    store = OperationStore(db_url)
    handed_over = os.environ.get(OP_ID_ENV)
    record: OperationRecord | None = None
    if handed_over:
        record = store.adopt(handed_over, interface="cli", purpose=f"run {model}")
    if record is None:
        record, _ = store.begin(
            kind=kind,
            target=target,
            interface="cli",
            purpose=f"run {model}",
            request={"target": target, "model": model},
        )
    bind_current(store, record.id or "")
    try:
        yield record
    except KeyboardInterrupt:
        interrupt_current("interrupted (Ctrl-C)")
        raise
    except SystemExit as exc:
        if exc.code in (None, 0):
            finish_current(OperationPhase.COMPLETED)
        else:
            finish_current(OperationPhase.FAILED, error=f"exit {exc.code}")
        raise
    except TyperExit as exc:  # typer command exit: same semantics as SystemExit
        code = getattr(exc, "exit_code", 1)
        if code in (None, 0):
            finish_current(OperationPhase.COMPLETED)
        else:
            finish_current(OperationPhase.FAILED, error=f"exit {code}")
        raise
    except BaseException as exc:  # noqa: BLE001 - crash: record, then propagate
        interrupt_current(f"{type(exc).__name__}: {exc}")
        raise
    else:
        finish_current(OperationPhase.COMPLETED)
    finally:
        clear_current()
