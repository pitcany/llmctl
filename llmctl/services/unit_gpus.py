"""Best-effort GPU-pinning introspection for externally-managed systemd units.

An ADOPTED session's process lives in systemd, not llmctl's process tree, so we
can't read its launch plan to learn which GPUs it occupies. But the unit's
``MainPID`` environment still carries the ``CUDA_VISIBLE_DEVICES`` it was started
with — enough to label which GPUs an adopted vLLM unit is pinned to.

Everything here is best-effort: a missing unit, no systemd, no ``/proc`` (e.g.
macOS CI), an unset variable, or UUID-form device ids all yield an empty list,
leaving the caller exactly as it was before. This keeps ``adopt``/``reconcile``
pure on hosts where the introspection can't run.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

#: The environment variable systemd applies (via EnvironmentFile/Environment)
#: that pins a vLLM process to specific GPU indices.
_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"

#: Short, bounded timeout for the ``systemctl show`` call. Reconcile runs on a
#: timer; we never want a stuck systemctl to wedge a reconcile pass.
_SYSTEMCTL_TIMEOUT_S = 2.0

#: Injection seams so the logic is testable without a live systemd or ``/proc``.
RunFn = Callable[..., "subprocess.CompletedProcess[str]"]
ReadEnvironFn = Callable[[int], "str | None"]


def parse_cuda_visible_devices(raw: str) -> list[int]:
    """Parse a ``CUDA_VISIBLE_DEVICES`` value into integer GPU indices.

    Only plain integer indices are kept, preserving written order. Blank
    entries, UUID-form ids (``GPU-...``), MIG ids, and the ``-1`` disable
    sentinel (non-numeric after the sign) are dropped — we can't map those to a
    bare index, so we'd rather report nothing than something wrong.

    :param raw: The raw env value, e.g. ``"0,1"``.
    :returns: GPU indices, e.g. ``[0, 1]``; ``[]`` when none parse.
    """
    indices: list[int] = []
    for token in raw.split(","):
        cleaned = token.strip()
        if cleaned.isdigit():
            indices.append(int(cleaned))
    return indices


def _systemctl_main_pid(unit_name: str, *, run: RunFn) -> int | None:
    """Return a running unit's ``MainPID``, or ``None`` if not resolvable.

    systemd reports ``MainPID=0`` for a unit that isn't running; that maps to
    ``None`` here. Any failure to invoke ``systemctl`` (absent, timeout,
    non-zero) is swallowed and also yields ``None``.
    """
    try:
        proc = run(
            ["systemctl", "show", unit_name, "--property=MainPID", "--value"],
            capture_output=True,
            text=True,
            timeout=_SYSTEMCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = (proc.stdout or "").strip()
    if not raw.isdigit():
        return None
    pid = int(raw)
    return pid or None


def _read_proc_environ(pid: int) -> str | None:
    """Return the raw, NUL-delimited environ of ``pid``, or ``None`` on failure."""
    try:
        return Path(f"/proc/{pid}/environ").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None


def unit_gpu_ids(
    unit_name: str | None,
    *,
    run: RunFn = subprocess.run,
    read_environ: ReadEnvironFn = _read_proc_environ,
) -> list[int]:
    """Best-effort GPU indices a systemd unit is pinned to via ``CUDA_VISIBLE_DEVICES``.

    Resolves the unit's ``MainPID`` and reads ``CUDA_VISIBLE_DEVICES`` from that
    process's environ. Returns ``[]`` on any failure (no unit name, unit not
    running, no ``/proc``, variable unset, only non-integer device ids) so
    callers degrade to today's behavior rather than raising.

    :param unit_name: systemd unit, e.g. ``"vllm-tp.service"``.
    :param run: ``subprocess.run``-compatible callable (injected in tests).
    :param read_environ: maps a pid to its raw environ string (injected in tests).
    :returns: GPU indices, e.g. ``[0, 1]``; ``[]`` when undeterminable.
    """
    if not unit_name:
        return []
    pid = _systemctl_main_pid(unit_name, run=run)
    if pid is None:
        return []
    raw_environ = read_environ(pid)
    if raw_environ is None:
        return []
    for entry in raw_environ.split("\0"):
        key, sep, value = entry.partition("=")
        if sep and key == _CUDA_VISIBLE_DEVICES:
            return parse_cuda_visible_devices(value)
    return []
