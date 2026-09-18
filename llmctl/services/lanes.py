"""One list of everything ``llmctl swap`` can serve: vLLM presets plus lanes.

A *lane* is a self-contained server with its own systemd unit and start script
(FreeToken, the llama.cpp servers, ...). The authoritative list of lanes is
llm-router's ``config/router.yaml``: every backend there with a ``start_cmd``.
llmctl reads that file instead of keeping a second copy, so a lane added for
the router shows up in ``llmctl swap`` too (the FreeToken lane was invisible to
llmctl for exactly that reason).

The lane's start script owns the GPU hand-over (``--take-gpus`` stops vllm-tp
and evicts Ollama models); llmctl runs it, waits for readiness, and rolls back.
Lane -> systemd unit comes from ``managed_units`` in llmctl's settings, matched
by port.

The router's ``profile_env`` backends are NOT read: vLLM rows come from llmctl's
own presets, the canonical vLLM schema.

Each row has a GPU *footprint*: the cards it occupies. A lane declares it as
``gpus: [0]`` in router.yaml (the one place llm-router reads too); a lane
without one, and every vLLM preset (``vllm-tp`` is TP=2), occupies every card.
A swap stops only what overlaps the new target's footprint.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from llmctl.config import ManagedUnitsConfig

#: Override for the router config location (tests, other machines).
ROUTER_CONFIG_ENV = "LLMCTL_ROUTER_CONFIG"
DEFAULT_ROUTER_CONFIG = Path.home() / "AI" / "llm-router" / "config" / "router.yaml"

#: Health wait when router.yaml gives none; the router's own default.
DEFAULT_LANE_TIMEOUT_S = 300.0

_ENGINE_LABELS = {
    "vllm": "vLLM",
    "freetoken": "FreeToken",
    "llama_cpp": "llama.cpp",
    "sglang": "SGLang",
    "openai": "OpenAI-compat",
    "ollama": "Ollama",
}


def engine_label(runtime: str | None) -> str:
    """Human name for a runtime id (``llama_cpp`` -> ``llama.cpp``)."""
    if not runtime:
        return "?"
    return _ENGINE_LABELS.get(runtime, runtime)


def router_config_path() -> Path:
    """Where router.yaml lives: ``$LLMCTL_ROUTER_CONFIG`` or the ~/AI default."""
    override = os.environ.get(ROUTER_CONFIG_ENV)
    return Path(override).expanduser() if override else DEFAULT_ROUTER_CONFIG


#: A GPU footprint: the card indices a target occupies. ``None`` = every card.
Gpus = frozenset[int] | None


def gpus_overlap(a: Gpus, b: Gpus) -> bool:
    """True if two footprints share a card. ``None`` (every card) overlaps all."""
    if a is None or b is None:
        return True
    return bool(a & b)


def format_gpus(gpus: Gpus) -> str:
    """``frozenset({0})`` -> ``0``; ``{0, 1}`` -> ``0+1``; ``None`` -> ``all``."""
    if gpus is None:
        return "all"
    return "+".join(str(i) for i in sorted(gpus))


def _parse_gpus(raw: Any) -> Gpus:
    """router.yaml ``gpus:`` -> footprint. Missing or malformed -> every card.

    Fails closed: an unreadable footprint overlaps everything, so a swap stops
    it rather than loading a second engine onto its card.
    """
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in raw):
        return None
    return frozenset(raw)


def _port_of(url: str) -> int | None:
    try:
        return urlparse(url).port
    except ValueError:
        return None


@dataclass(frozen=True)
class Lane:
    """A command-launched backend from router.yaml."""

    name: str
    served_model: str
    upstream: str
    start_cmd: tuple[str, ...]
    health_timeout_s: float = DEFAULT_LANE_TIMEOUT_S
    aliases: tuple[str, ...] = ()
    #: systemd unit (user scope) serving this lane, from llmctl managed_units.
    unit: str | None = None
    #: llmctl runtime id of that unit (``freetoken``, ``llama_cpp``, ...).
    runtime: str | None = None
    #: Cards the lane occupies, from router.yaml ``gpus:``; None = every card.
    gpus: Gpus = None


def load_lanes(
    path: Path | None = None,
    managed_units: ManagedUnitsConfig | None = None,
) -> list[Lane]:
    """Return router.yaml's command-launched backends, in file order.

    A missing or unreadable file yields no lanes: llmctl keeps working with
    its vLLM presets alone (e.g. on a machine without llm-router).
    """
    path = path or router_config_path()
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return []
    backends = raw.get("backends") if isinstance(raw, dict) else None
    if not isinstance(backends, list):
        return []
    default_timeout = DEFAULT_LANE_TIMEOUT_S
    switch = raw.get("switch")
    if isinstance(switch, dict) and isinstance(switch.get("health_timeout_s"), int | float):
        default_timeout = float(switch["health_timeout_s"])

    by_port: dict[int, tuple[str, str]] = {}
    if managed_units is not None:
        for unit in managed_units.roles().values():
            by_port[unit.default_port] = (unit.unit_name, unit.runtime)

    lanes: list[Lane] = []
    for entry in backends:
        if not isinstance(entry, dict):
            continue
        cmd = entry.get("start_cmd")
        name = entry.get("name")
        served = entry.get("served_model")
        upstream = entry.get("upstream")
        if not cmd or not name or not served or not upstream:
            continue  # profile_env (vLLM) backend, or malformed
        timeout = entry.get("health_timeout_s")
        unit, runtime = by_port.get(_port_of(str(upstream)) or -1, (None, None))
        if unit is not None and not unit.endswith(".service"):
            unit = f"{unit}.service"
        lanes.append(
            Lane(
                name=str(name),
                served_model=str(served),
                upstream=str(upstream).rstrip("/"),
                start_cmd=tuple(str(part) for part in cmd),
                health_timeout_s=float(timeout)
                if isinstance(timeout, int | float)
                else default_timeout,
                aliases=tuple(str(a) for a in entry.get("aliases") or ()),
                unit=unit,
                runtime=runtime,
                gpus=_parse_gpus(entry.get("gpus")),
            )
        )
    return lanes


@dataclass(frozen=True)
class SwapTarget:
    """One row of the swap list: a vLLM preset or a lane."""

    key: str
    """What you type: the preset alias or the router backend name."""
    kind: str
    """``"vllm"`` or ``"lane"``."""
    model: str
    """Served model name (what clients ask for)."""
    engine: str
    upstream: str
    lane: Lane | None = None

    @property
    def gpus(self) -> Gpus:
        """Cards this target occupies. vLLM presets (TP=2) take every card."""
        return self.lane.gpus if self.lane is not None else None

    def matches(self, wanted: str) -> bool:
        """Case-insensitive match on key, served model, or lane alias."""
        target = wanted.strip().lower()
        if not target:
            return False
        names = {self.key.lower(), self.model.lower()}
        if self.lane is not None:
            names.update(a.lower() for a in self.lane.aliases)
        return target in names


def build_catalog(
    presets: dict[str, Any],
    lanes: Iterable[Lane],
    *,
    vllm_port: int,
) -> list[SwapTarget]:
    """vLLM presets first (sorted), then lanes in router.yaml order.

    ``presets`` maps alias -> anything with a ``served_name`` attribute
    (:class:`VLLMLaunchSpec` in production).
    """
    rows: list[SwapTarget] = []
    for alias in sorted(presets):
        served = getattr(presets[alias], "served_name", None) or alias
        rows.append(
            SwapTarget(
                key=alias,
                kind="vllm",
                model=str(served),
                engine=engine_label("vllm"),
                upstream=f"http://127.0.0.1:{vllm_port}",
            )
        )
    for lane in lanes:
        rows.append(
            SwapTarget(
                key=lane.name,
                kind="lane",
                model=lane.served_model,
                engine=engine_label(lane.runtime),
                upstream=lane.upstream,
                lane=lane,
            )
        )
    return rows


def resolve(wanted: str, catalog: list[SwapTarget]) -> SwapTarget | None:
    """Find a row by key first, then by served model / alias."""
    for row in catalog:
        if row.key == wanted:
            return row
    for row in catalog:
        if row.matches(wanted):
            return row
    return None


Prober = Callable[[str], list[str] | None]


def serving_keys(catalog: list[SwapTarget], probe: Prober) -> set[str]:
    """Keys of rows whose upstream lists their served model right now.

    Each upstream is probed once, however many rows share it (every vLLM
    preset shares :8003).
    """
    served_by: dict[str, list[str] | None] = {}
    keys: set[str] = set()
    for row in catalog:
        if row.upstream not in served_by:
            served_by[row.upstream] = probe(row.upstream)
        ids = served_by[row.upstream] or []
        if row.model in ids:
            keys.add(row.key)
    return keys


# -- load-time memory ---------------------------------------------------------


def load_times_path() -> Path:
    """``$XDG_STATE_HOME/llmctl/load-times.json`` (``~/.local/state`` default)."""
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "llmctl" / "load-times.json"


def last_load_s(key: str, path: Path | None = None) -> float | None:
    """Seconds the last successful swap to ``key`` took, or None if never seen."""
    try:
        data = json.loads((path or load_times_path()).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    entry = data.get(key) if isinstance(data, dict) else None
    if isinstance(entry, dict) and isinstance(entry.get("seconds"), int | float):
        return float(entry["seconds"])
    return None


def record_load(key: str, seconds: float, path: Path | None = None) -> None:
    """Remember how long a successful swap to ``key`` took. Best effort."""
    path = path or load_times_path()
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data[key] = {"seconds": round(seconds, 1), "at": int(time.time())}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(path)
    except OSError:
        pass


def format_duration(seconds: float | None) -> str:
    """``130.4`` -> ``2m 10s``; None -> ``never measured``."""
    if seconds is None:
        return "never measured"
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


# -- preview --------------------------------------------------------------------


@dataclass
class SwapPreview:
    """What a swap will do, shown before the confirmation prompt."""

    target: SwapTarget
    already_serving: bool
    stops: list[str] = field(default_factory=list)
    starts: str = ""
    last_load_s: float | None = None

    def lines(self) -> list[str]:
        """Plain-text preview lines (the CLI and the TUI both render these)."""
        if self.already_serving and self.target.kind == "lane":
            return [f"{self.target.model} ({self.target.engine}) is already serving."]
        out = [f"stops:  {', '.join(self.stops) if self.stops else 'nothing is running'}"]
        out.append(f"starts: {self.starts}")
        out.append(f"last load took {format_duration(self.last_load_s)}")
        return out


def preview_swap(
    target: SwapTarget,
    catalog: list[SwapTarget],
    *,
    serving: set[str],
    active_units: set[str],
    vllm_unit: str,
    vllm_active: bool,
    load_times: Path | None = None,
) -> SwapPreview:
    """Describe the swap to ``target`` given what is running now.

    ``active_units`` are lane units whose systemd state is active (serving or
    still loading); ``serving`` are catalog keys answering right now.
    """
    preview = SwapPreview(
        target=target,
        already_serving=target.key in serving,
        last_load_s=last_load_s(target.key, load_times),
    )
    stops: list[str] = []
    if vllm_active:
        current = next((r.model for r in catalog if r.kind == "vllm" and r.key in serving), None)
        label = f"{vllm_unit} ({current})" if current else vllm_unit
        if target.kind == "lane":
            stops.append(label)
        else:
            stops.append(f"{label}, restarted on preset {target.key}")
    for row in catalog:
        if row.kind != "lane" or row.lane is None or row.lane.unit is None:
            continue
        if row.key == target.key or row.lane.unit not in active_units:
            continue
        if not gpus_overlap(row.gpus, target.gpus):
            continue
        stops.append(f"{row.lane.unit} ({row.model})")
    if target.kind == "vllm":
        stops.append("Ollama/Harbor (paused while vLLM loads, started again after)")
        preview.starts = f"{vllm_unit} on preset {target.key}"
    else:
        lane = target.lane
        assert lane is not None
        if "--take-gpus" in lane.start_cmd:
            stops.append("Ollama models on its GPUs (start script, --take-gpus)")
        unit = lane.unit or "its unit"
        preview.starts = f"{unit} via {' '.join(lane.start_cmd)}"
    preview.stops = stops
    return preview


# -- live state -------------------------------------------------------------------


@dataclass
class SwapState:
    """The swap list plus what is running right now."""

    catalog: list[SwapTarget]
    serving: set[str]
    active_units: set[str]
    vllm_unit: str
    vllm_active: bool

    def state_of(self, row: SwapTarget) -> str:
        """``serving`` / ``loading`` / ``""`` for the picker's State column."""
        if row.key in self.serving:
            return "serving"
        if row.lane is not None and row.lane.unit in self.active_units:
            return "loading"
        return ""

    def active_lanes(self, exclude: str | None = None) -> list[Lane]:
        """Lanes whose unit is active, except ``exclude`` (a catalog key)."""
        return [
            row.lane
            for row in self.catalog
            if row.lane is not None and row.lane.unit in self.active_units and row.key != exclude
        ]

    def lanes_in_the_way(self, target: SwapTarget) -> list[Lane]:
        """Active lanes a swap to ``target`` must stop: those sharing a card."""
        return [
            row.lane
            for row in self.catalog
            if row.lane is not None
            and row.lane.unit in self.active_units
            and row.key != target.key
            and gpus_overlap(row.gpus, target.gpus)
        ]

    def preview(self, target: SwapTarget, load_times: Path | None = None) -> SwapPreview:
        """:func:`preview_swap` against this state."""
        return preview_swap(
            target,
            self.catalog,
            serving=self.serving,
            active_units=self.active_units,
            vllm_unit=self.vllm_unit,
            vllm_active=self.vllm_active,
            load_times=load_times,
        )


def gather_state(
    settings: Any,
    *,
    presets: dict[str, Any] | None = None,
    lanes: list[Lane] | None = None,
    probe: Prober | None = None,
    is_active: Callable[[str, bool], bool] | None = None,
) -> SwapState:
    """Build the catalog and probe it. Read-only; every seam is injectable.

    ``is_active(unit, user_scope)`` answers systemd's is-active.
    """
    if presets is None:
        from llmctl.services.preset_loader import load_presets

        presets = load_presets(defaults=settings.vllm.defaults)
    if lanes is None:
        lanes = load_lanes(managed_units=settings.managed_units)
    if probe is None:
        from llmctl.services.backends import probe_openai_v1_models

        probe = probe_openai_v1_models
    if is_active is None:
        from llmctl.integrations.systemctl import SystemctlRunner

        runner = SystemctlRunner()

        def is_active(unit: str, user: bool) -> bool:
            return runner.is_active(unit, user=user)

    vllm = settings.managed_units.vllm_tp
    catalog = build_catalog(presets, lanes, vllm_port=vllm.default_port)
    active_units = {
        lane.unit for lane in lanes if lane.unit is not None and is_active(lane.unit, True)
    }
    return SwapState(
        catalog=catalog,
        serving=serving_keys(catalog, probe),
        active_units=active_units,
        vllm_unit=vllm.unit_name,
        vllm_active=is_active(vllm.unit_name, False),
    )
