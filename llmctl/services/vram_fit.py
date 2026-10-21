"""Run two models at once when they fit: per-card VRAM need vs free VRAM.

A swap used to stop everything sharing a card with the new target, even when
the card had room for both. Now each lane's VRAM, per card, is measured once it
serves and remembered next to its load time (``load-times.json``,
``"vram_mib": {"0": 25400}``). A later swap to that lane keeps what is running
when every card it uses has that much free, plus :data:`HEADROOM_MIB`.

Fails closed everywhere: a lane never measured, a card NVML cannot read, or a
systemd ``Conflicts=`` between the two units means the old behaviour (stop
what overlaps).

The measure is what the lane held after a successful start. Engines that size
their KV cache from free memory (vLLM's ``--gpu-memory-utilization``) measure
as nearly the whole card when started alone, so they never "fit" beside
anything; that is the safe direction.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

#: Free VRAM a card must keep after the new model loads (CUDA context, spikes).
HEADROOM_MIB = 1024

#: Per-card MiB: ``{0: 25400}``.
CardMib = dict[int, int]


def fits(need: Mapping[int, int] | None, free: Mapping[int, int] | None) -> bool:
    """True when every card in ``need`` has that much free plus headroom.

    ``None`` or empty on either side (never measured / unreadable) -> False.
    """
    if not need or not free:
        return False
    return all(c in free and free[c] >= mib + HEADROOM_MIB for c, mib in need.items())


def free_mib_now(gpus: list[Any] | None = None) -> CardMib | None:
    """Free MiB per card from NVML, or None when it cannot be read."""
    if gpus is None:
        from llmctl.telemetry.gpu import get_gpu_info

        gpus = get_gpu_info()
    out = {g.index: g.memory_free_mb for g in gpus if g.memory_free_mb is not None}
    return out or None


# -- measuring a lane -----------------------------------------------------------------


def _unit_pids(unit: str, user: bool = True) -> set[int]:
    """Every PID in ``unit``'s cgroup (an engine forks workers that hold VRAM)."""
    argv = ["systemctl", *(["--user"] if user else []), "show", "-p", "ControlGroup", "--value"]
    try:
        cgroup = subprocess.run(  # noqa: S603 - fixed argv
            [*argv, unit], check=False, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if not cgroup:
        return set()
    pids: set[int] = set()
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    try:
        for procs in root.rglob("cgroup.procs"):
            pids |= {int(p) for p in procs.read_text().split() if p.isdigit()}
    except OSError:
        return set()
    return pids


def pids_vram_mib(pids: set[int], gpus: list[Any] | None = None) -> CardMib:
    """MiB each card holds for ``pids`` (NVML compute processes)."""
    if gpus is None:
        from llmctl.telemetry.gpu import get_gpu_info

        gpus = get_gpu_info()
    out: CardMib = {}
    for gpu in gpus:
        mib = sum(int(p.get("used_memory_mb") or 0) for p in gpu.processes if p.get("pid") in pids)
        if mib:
            out[gpu.index] = mib
    return out


def ollama_model_vram_mib(upstream: str, model: str, timeout: float = 3.0) -> int | None:
    """``size_vram`` of ``model`` per ``/api/ps``, in MiB.

    One Ollama unit can hold several lanes' models; its cgroup total would
    charge each lane for its siblings.
    """
    import urllib.request

    url = f"{upstream.rstrip('/')}/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local upstream
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    for m in payload.get("models", []) if isinstance(payload, dict) else []:
        if isinstance(m, dict) and model in (m.get("name"), m.get("model")):
            size = m.get("size_vram")
            return int(size) // 1048576 if isinstance(size, int) and size > 0 else None
    return None


def measure_lane(lane: Any) -> CardMib | None:
    """VRAM ``lane`` holds now, per card, or None when it cannot be told apart.

    An Ollama lane pinned to one card is charged its model's ``size_vram``;
    anything else is charged its unit's processes.
    """
    if lane.runtime == "ollama":
        if lane.gpus is None or len(lane.gpus) != 1:
            return None
        mib = ollama_model_vram_mib(lane.upstream, lane.served_model)
        return {next(iter(lane.gpus)): mib} if mib else None
    if not lane.unit:
        return None
    pids = _unit_pids(lane.unit)
    return (pids_vram_mib(pids) or None) if pids else None


# -- memory -----------------------------------------------------------------------------


def last_vram_mib(key: str, path: Path | None = None) -> CardMib | None:
    """The per-card VRAM last measured for ``key``, or None if never measured."""
    from llmctl.services.lanes import load_times_path

    try:
        data = json.loads((path or load_times_path()).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    entry = data.get(key) if isinstance(data, dict) else None
    raw = entry.get("vram_mib") if isinstance(entry, dict) else None
    if not isinstance(raw, dict):
        return None
    try:
        out = {int(c): int(m) for c, m in raw.items()}
    except (TypeError, ValueError):
        return None
    return out if out and all(m > 0 for m in out.values()) else None


def record_vram(key: str, per_card: Mapping[int, int], path: Path | None = None) -> None:
    """Remember ``key``'s per-card VRAM next to its load time. Best effort."""
    from llmctl.services.lanes import load_times_path

    if not per_card:
        return
    path = path or load_times_path()
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}
    entry = data.get(key) if isinstance(data.get(key), dict) else {}
    entry["vram_mib"] = {str(c): int(m) for c, m in sorted(per_card.items())}
    entry["vram_at"] = int(time.time())
    data[key] = entry
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(path)
    except OSError:
        pass


def remember_lane_vram(
    key: str,
    lane: Any,
    *,
    measure: Callable[[Any], CardMib | None] = measure_lane,
    path: Path | None = None,
) -> CardMib | None:
    """Measure a serving lane and record it. Returns what was recorded."""
    try:
        per_card = measure(lane)
    except Exception:  # noqa: BLE001 - measuring is advisory
        return None
    if per_card:
        record_vram(key, per_card, path)
    return per_card


def unit_conflicts(unit: str) -> set[str] | None:
    """Units systemd stops when ``unit`` starts (its ``Conflicts=``).

    None when the query fails: the caller must not assume "no conflicts".
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv
            ["systemctl", "--user", "show", "-p", "Conflicts", "--value", unit],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return set(result.stdout.split())


def format_mib(per_card: Mapping[int, int] | None) -> str:
    """``{0: 25400}`` -> ``24.8 GiB on GPU 0``."""
    if not per_card:
        return "not measured yet"
    return ", ".join(f"{m / 1024:.1f} GiB on GPU {c}" for c, m in sorted(per_card.items()))
