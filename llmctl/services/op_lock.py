"""Host-wide operation lock for managed GPU lifecycle changes (FIXPLAN §2A).

llmctl mutates the same GPUs from several processes: the CLI, the TUI's
worker threads, the API server, and swap jobs the API spawns as CLI child
processes. A threading lock in one of them sees none of the others. This
module is the single boundary: an ``flock`` on a host-wide lock file that
every mutating path must hold while it changes what is serving.

Design points (from the fix plan):

* Conservative host-wide scope: one lock covers every managed GPU lifecycle
  change (start, stop, restart, swap). Read-only status never takes it.
* Reentrant per (process, thread): the CLI holds the lock across the whole
  swap while lower-level helpers it calls (lane swap, session stop) also
  request it. flock would deadlock a second acquire, so the wrapper
  reference-counts per thread and only flocks once.
* Cross-thread still conflicts: two threads in one process fighting over
  the same GPUs is a real conflict, so a different thread gets ``LockBusy``
  after ``timeout`` rather than silently sharing.
* The lock file records who holds it (pid, host, interface, purpose) so a
  refusal can say *what* is running, not just "busy".
* flock is released by the kernel if the holder dies, so a crashed process
  never wedges the host; stale file content is simply overwritten by the
  next holder. External tools (systemd, manual ``systemctl``) are outside
  llmctl's lock by design -- mutating paths must still probe actual state
  before acting.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

#: Environment override for the lock path (used by tests and embedders).
LOCK_PATH_ENV = "LLMCTL_OP_LOCK"

#: Default wait before refusing: long enough for a fast conflicting mutation
#: to finish, short enough that an operator is not left hanging.
DEFAULT_TIMEOUT_S = 5.0


class LockBusy(Exception):
    """Another holder owns the operation lock."""

    def __init__(self, holder: dict[str, Any], timeout: float) -> None:
        self.holder = holder
        who = describe_holder(holder)
        super().__init__(f"another GPU operation is active ({who}); waited {timeout:.0f}s")


def describe_holder(holder: dict[str, Any] | None) -> str:
    """One-line description of a lock holder for messages and the TUI."""
    if not holder:
        return "unknown holder"
    pid = holder.get("pid")
    interface = holder.get("interface") or "?"
    purpose = holder.get("purpose") or "?"
    host = holder.get("hostname") or "?"
    age = ""
    acquired = holder.get("acquired_at")
    if isinstance(acquired, (int, float)) and acquired > 0:
        age = f", {max(0, int(time.time() - acquired))}s ago"
    return f"pid {pid} on {host} via {interface} ({purpose}{age})"


def lock_path(settings: Any = None) -> Path:
    """Resolve the host-wide lock file path.

    ``LLMCTL_OP_LOCK`` wins, then the settings ``paths.data_dir``, then the
    platform data dir (e.g. ``~/.local/share/llmctl``).
    """
    env = os.environ.get(LOCK_PATH_ENV)
    if env:
        return Path(env).expanduser()
    if settings is None:
        from llmctl.config import load_settings

        settings = load_settings()
    data_dir = getattr(getattr(settings, "paths", None), "data_dir", None) if settings else None
    if data_dir:
        return Path(data_dir).expanduser() / "operations.lock"
    return Path(user_data_dir("llmctl")) / "operations.lock"


@dataclass
class _Held:
    """One process-wide hold of the lock file."""

    fd: int
    thread_id: int
    depth: int
    holder: dict[str, Any]


_guard = threading.Lock()
_held: dict[str, _Held] = {}


def _read_holder(path: Path) -> dict[str, Any] | None:
    """Best-effort read of the current holder description."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _flock_owner_pid(path: Path, proc_locks: Path = Path("/proc/locks")) -> int | None:
    """Pid holding the exclusive flock on ``path`` per ``/proc/locks`` (None if unknown)."""
    try:
        st = os.stat(path)
        lines = proc_locks.read_text().splitlines()
    except OSError:
        return None
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    for line in lines:
        parts = line.split()  # blocked waiters have "->" in parts[1] and are skipped
        if len(parts) < 6 or parts[1] != "FLOCK" or parts[3] != "WRITE" or parts[5] != key:
            continue
        try:
            return int(parts[4])
        except ValueError:
            return None
    return None


def _current_holder(path: Path, proc_locks: Path = Path("/proc/locks")) -> dict[str, Any] | None:
    """The holder record, checked against the process that really holds the flock.

    The file keeps whatever the last cooperating holder wrote, so a refusal
    could name an exited llmctl while a plain ``flock`` held the lock (live
    acceptance 2026-11-04, finding 2). Same logic as llm-router's
    ``op_lock.current_holder``.
    """
    recorded = _read_holder(path)
    real = _flock_owner_pid(path, proc_locks)
    if real is None or (recorded is not None and recorded.get("pid") == real):
        return recorded
    note = "not recorded"
    if recorded is not None and recorded.get("pid") is not None:
        note += f"; lock file names pid {recorded.get('pid')}, which does not hold it"
    try:
        comm = Path(f"/proc/{real}/comm").read_text().strip() or "?"
    except OSError:
        comm = "?"
    return {"pid": real, "hostname": socket.gethostname(), "interface": comm, "purpose": note}


@contextmanager
def gpu_operation_lock(
    *,
    purpose: str,
    interface: str,
    timeout: float = DEFAULT_TIMEOUT_S,
    settings: Any = None,
) -> Iterator[dict[str, Any]]:
    """Hold the host-wide GPU operation lock for this thread.

    Raises :class:`LockBusy` when another process (or another thread of this
    process) holds it and does not release within ``timeout``. Reentrant for
    the same thread; cross-thread acquires conflict.
    """
    path = lock_path(settings)
    key = str(path)
    me = threading.get_ident()
    deadline = time.monotonic() + max(0.0, timeout)

    with _guard:
        held = _held.get(key)
        reentrant: _Held | None = None
        if held is not None:
            if held.thread_id != me:
                # A different thread of this process holds it: real conflict.
                holder = _read_holder(path) or held.holder
                raise LockBusy(holder, timeout)
            held.depth += 1
            reentrant = held
    if reentrant is not None:
        # NOTE: _guard is released before yielding so a further nested
        # acquire (same thread) can update the refcount without deadlock.
        try:
            yield reentrant.holder
        finally:
            with _guard:
                reentrant.depth -= 1
                if reentrant.depth <= 0:
                    _release(key, reentrant)
        return

    # Not held in-process: take the flock, retrying until the deadline.
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    registered = False
    try:
        os.chmod(path, 0o600)  # umask may have loosened O_CREAT's mode
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise LockBusy(_current_holder(path), timeout) from exc
                time.sleep(0.05)
        holder = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "interface": interface,
            "purpose": purpose,
            "acquired_at": time.time(),
            "thread": me,
        }
        os.lseek(fd, 0, os.SEEK_SET)
        os.truncate(fd, 0)
        os.write(fd, json.dumps(holder).encode("utf-8"))
        os.fsync(fd)
        with _guard:
            existing = _held.get(key)
            if existing is not None:
                # Lost a race to another thread of this process.
                os.lseek(fd, 0, os.SEEK_SET)
                os.truncate(fd, 0)
                os.write(fd, json.dumps(existing.holder).encode("utf-8"))
                raise LockBusy(existing.holder, timeout)
            record = _Held(fd=fd, thread_id=me, depth=1, holder=holder)
            _held[key] = record
        registered = True
        try:
            yield holder
        finally:
            with _guard:
                record.depth -= 1
                if record.depth <= 0:
                    _release(key, record)
    finally:
        if not registered:
            # We flocked but never registered (busy race) or raised first.
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def _release(key: str, held: _Held) -> None:
    """Drop the process-wide hold (idempotent, called under ``_guard``)."""
    if _held.get(key) is not held:
        return
    del _held[key]
    try:
        fcntl.flock(held.fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(held.fd)
    except OSError:
        pass


@contextmanager
def handed_off(*, timeout: float = 30.0, settings: Any = None) -> Iterator[None]:
    """Hand this thread's hold to a cooperating owner for the block.

    llm-router takes this same flock around its switch (and rollback). A CLI
    that kept holding it across ``POST /admin/switch`` would wait on a router
    that waits on it, so the delegated call runs inside this block: the flock
    is released (the reentrancy depth is kept), the router owns the GPUs for
    its whole mutation, and the hold is taken back -- waiting up to
    ``timeout`` -- for the caller's remaining work (restoring paused lanes,
    recording the outcome). If someone else still owns it then,
    :class:`LockBusy` is raised and nothing stays registered: the caller must
    not continue without ownership.

    A no-op when this thread holds nothing. No token or bypass is involved:
    whoever owns the GPUs during the call holds the flock like anyone else.
    """
    path = lock_path(settings)
    key = str(path)
    me = threading.get_ident()
    with _guard:
        held = _held.get(key)
        if held is None or held.thread_id != me:
            held = None
        else:
            del _held[key]
            try:
                fcntl.flock(held.fd, fcntl.LOCK_UN)
            except OSError:
                pass
    if held is None:
        yield
        return
    try:
        yield
    finally:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(held.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    try:
                        os.close(held.fd)
                    except OSError:
                        pass
                    raise LockBusy(_current_holder(path), timeout) from exc
                time.sleep(0.05)
        held.holder = {**held.holder, "acquired_at": time.time()}
        os.lseek(held.fd, 0, os.SEEK_SET)
        os.truncate(held.fd, 0)
        os.write(held.fd, json.dumps(held.holder).encode("utf-8"))
        os.fsync(held.fd)
        with _guard:
            _held[key] = held


def held_by_us(settings: Any = None) -> bool:
    """True when the current thread already holds the lock (test helper)."""
    path = str(lock_path(settings))
    with _guard:
        held = _held.get(path)
        return held is not None and held.thread_id == threading.get_ident()
