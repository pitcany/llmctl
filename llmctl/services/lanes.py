"""One list of everything ``llmctl swap`` can serve: vLLM presets plus lanes.

A *lane* is a self-contained server with its own systemd unit and start script
(FreeToken, the llama.cpp servers, ...). The authoritative list of lanes is
llm-router's ``config/router.yaml``: every backend there with a ``start_cmd``.
llmctl reads that file instead of keeping a second copy, so a lane added for
the router shows up in ``llmctl swap`` too (the FreeToken lane was invisible to
llmctl for exactly that reason).

The lane's start script owns the GPU hand-over (``--take-gpus`` stops vllm-tp
and evicts Ollama models); llmctl runs it, waits for readiness, and rolls back.
Lane -> systemd unit comes from ``managed_units`` in llmctl's settings when a
row there shares the port, else from the lane's own ``unit:`` key in
router.yaml (llm-router ignores it). A lane therefore needs no settings row:
``units_from_lanes`` derives one so ``llmctl status`` sees it too.

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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig

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


def cards_label(row: Any) -> str:
    """The swap list's Cards cell. A session row's cards are chosen at start."""
    placement = getattr(row, "placement", None)
    if getattr(row, "kind", "") == "session" and placement is not None:
        return f"any {placement.cards}" if placement.cards == 1 else f"{placement.cards} (split)"
    return format_gpus(row.gpus)


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
    #: systemd unit (user scope) serving this lane. Taken from llmctl's
    #: managed_units when a row there shares the port, else from router.yaml's
    #: own ``unit:`` key — a lane needs no settings row to be complete.
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
        if unit is None and entry.get("unit"):
            # No managed_units row: router.yaml carries the unit itself, which
            # is what keeps a new lane from needing a hand-written settings
            # entry repeating the same unit name and port.
            unit = str(entry["unit"])
            runtime = str(entry["runtime"]) if entry.get("runtime") else None
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
    """``"vllm"``, ``"lane"``, or ``"session"`` (a llama.cpp model llmctl starts)."""
    model: str
    """Served model name (what clients ask for)."""
    engine: str
    upstream: str
    lane: Lane | None = None
    #: ``session`` rows: the registry row, and the fit decided before any stop.
    model_id: str | None = None
    placement: Any = None
    #: ``session`` rows: the port the model's default profile pins, if any.
    port: int | None = None
    #: ``session`` rows: a systemd unit that serves the same model file.
    unit: str | None = None

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


def user_unit_text(unit: str) -> str:
    """A user unit file's text, ``%h`` expanded; ``""`` when it cannot be read."""
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    try:
        text = (base / "systemd" / "user" / unit).read_text()
    except OSError:
        return ""
    return text.replace("%h", str(Path.home()))


def lane_for_model(
    name: str,
    path: str | None,
    catalog: list[SwapTarget],
    *,
    unit_text: Callable[[str], str] = user_unit_text,
) -> SwapTarget | None:
    """The lane row that already serves a registered model, if one does.

    Matched by name, or by the model file the lane's unit loads: registry
    names (``deepseek-v4-flash-0731-mxfp4``) rarely match lane names (``dsv4``).
    Swapping to the lane keeps one server per model instead of starting a
    second, untracked llama-server on the same weights.
    """
    wanted = str(Path(path).expanduser()) if path else ""
    for row in catalog:
        if row.lane is None:
            continue
        if row.matches(name):
            return row
        if wanted and row.lane.unit and wanted in unit_text(row.lane.unit):
            return row
    return None


def unit_for_model(
    path: str | None,
    units: Iterable[str],
    *,
    unit_text: Callable[[str], str] = user_unit_text,
) -> str | None:
    """The first user unit whose file loads the model at ``path``, if any."""
    if not path:
        return None
    wanted = str(Path(path).expanduser())
    for unit in units:
        if wanted in unit_text(unit):
            return unit
    return None


def llama_cpp_units(managed_units: ManagedUnitsConfig) -> list[str]:
    """Unit files of the llama.cpp servers in ``managed_units`` (gpt-oss, ...)."""
    return [
        cfg.unit_name if "." in cfg.unit_name else f"{cfg.unit_name}.service"
        for cfg in managed_units.units.values()
        if cfg.runtime == "llama_cpp"
    ]


Prober = Callable[[str], list[str] | None]


def serving_keys(catalog: list[SwapTarget], probe: Prober) -> set[str]:
    """Keys of rows whose upstream lists their served model right now.

    Each upstream is probed once, however many rows share it (every vLLM
    preset shares :8003).
    """
    served_by: dict[str, list[str] | None] = {}
    keys: set[str] = set()
    for row in catalog:
        if not row.upstream:
            continue  # a session row: its port is picked at start
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
        if self.already_serving and self.target.kind == "session":
            how = f" as {self.target.unit}" if self.target.unit else ""
            where = f" on port {self.target.port}" if self.target.port else ""
            return [
                f"{self.target.model} ({self.target.engine}) is already running{how}{where}; "
                "nothing to do."
            ]
        out = [f"stops:  {', '.join(self.stops) if self.stops else 'nothing is running'}"]
        out.append(f"starts: {self.starts}")
        out.append(f"last load took {format_duration(self.last_load_s)}")
        return out


def session_is_running(
    row: SwapTarget, active_units: set[str], gpu_sessions: Sequence[Any]
) -> bool:
    """A session row's model is up: llmctl started it, or its unit is active."""
    if row.kind != "session":
        return False
    if row.unit is not None and row.unit in active_units:
        return True
    return any(s.name == row.key for s in gpu_sessions)


def preview_swap(
    target: SwapTarget,
    catalog: list[SwapTarget],
    *,
    serving: set[str],
    active_units: set[str],
    vllm_unit: str,
    vllm_active: bool,
    load_times: Path | None = None,
    gpu_sessions: Sequence[Any] = (),
) -> SwapPreview:
    """Describe the swap to ``target`` given what is running now.

    ``active_units`` are lane units whose systemd state is active (serving or
    still loading); ``serving`` are catalog keys answering right now.
    """
    preview = SwapPreview(
        target=target,
        already_serving=target.key in serving
        or session_is_running(target, active_units, gpu_sessions),
        last_load_s=last_load_s(target.key, load_times),
    )
    stops: list[str] = []
    if vllm_active:
        current = next((r.model for r in catalog if r.kind == "vllm" and r.key in serving), None)
        label = f"{vllm_unit} ({current})" if current else vllm_unit
        if target.kind == "vllm":
            stops.append(f"{label}, restarted on preset {target.key}")
        else:
            stops.append(label)
    for row in catalog:
        if row.kind != "lane" or row.lane is None or row.lane.unit is None:
            continue
        if row.key == target.key or row.lane.unit not in active_units:
            continue
        if not gpus_overlap(row.gpus, target.gpus):
            continue
        stops.append(f"{row.lane.unit} ({row.model})")
    for session in gpu_sessions:
        if target.kind == "session" and session.name == target.key:
            continue
        if gpus_overlap(frozenset(session.gpu_ids), target.gpus):
            stops.append(session.label())
    if target.kind == "session":
        placement = target.placement
        cards = getattr(placement, "cards", 1) or 1
        where = "one GPU" if cards == 1 else f"{cards} GPUs (split)"
        port = f"port {target.port}" if target.port else "llmctl picks the port"
        preview.starts = f"llama-server for {target.key} on {where}; {port}"
        note = getattr(placement, "note", None)
        if note:
            stops.append(note)
    elif target.kind == "vllm":
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
    #: Processes llmctl started that hold GPU memory (llama.cpp models).
    gpu_sessions: list[Any] = field(default_factory=list)

    def state_of(self, row: SwapTarget) -> str:
        """``serving`` / ``loading`` / ``""`` for the picker's State column."""
        if row.key in self.serving:
            return "serving"
        if session_is_running(row, self.active_units, self.gpu_sessions):
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
            gpu_sessions=self.gpu_sessions,
        )

    def sessions_in_the_way(self, target: SwapTarget) -> list[Any]:
        """llmctl-started GPU processes a swap to ``target`` must stop."""
        return [
            s
            for s in self.gpu_sessions
            if gpus_overlap(frozenset(s.gpu_ids), target.gpus)
            and not (target.kind == "session" and s.name == target.key)
        ]


def gather_state(
    settings: Any,
    *,
    presets: dict[str, Any] | None = None,
    lanes: list[Lane] | None = None,
    probe: Prober | None = None,
    is_active: Callable[[str, bool], bool] | None = None,
    gpu_sessions: list[Any] | None = None,
    session_rows: Callable[[list[SwapTarget]], list[SwapTarget]] | None = None,
) -> SwapState:
    """Build the catalog and probe it. Read-only; every seam is injectable.

    ``session_rows(catalog)`` returns the llama.cpp registry models to list
    after the lanes (default: :func:`load_session_rows`).

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
    if session_rows is None:

        def session_rows(listed: list[SwapTarget]) -> list[SwapTarget]:
            return load_session_rows(settings, listed)

    catalog += session_rows(catalog)
    units = {lane.unit for lane in lanes if lane.unit is not None}
    units |= {row.unit for row in catalog if row.kind == "session" and row.unit}
    active_units = {unit for unit in units if is_active(unit, True)}
    if gpu_sessions is None:
        gpu_sessions = _load_gpu_sessions(settings)
    return SwapState(
        catalog=catalog,
        serving=serving_keys(catalog, probe),
        active_units=active_units,
        vllm_unit=vllm.unit_name,
        vllm_active=is_active(vllm.unit_name, False),
        gpu_sessions=gpu_sessions,
    )


def session_rows_for(
    models: Iterable[Any],
    catalog: list[SwapTarget],
    *,
    target_for: Callable[[Any], SwapTarget],
    is_local: Callable[[Any], bool],
    unit_text: Callable[[str], str] = user_unit_text,
) -> list[SwapTarget]:
    """Swap-list rows for llama.cpp models llmctl starts itself, sorted by name.

    Skipped: other engines, models on another machine, models a listed row
    already serves (same name, or a lane whose unit loads the same file), and
    models that fit on no combination of cards.
    """
    rows: list[SwapTarget] = []
    for model in models:
        if getattr(model.runtime, "value", model.runtime) != "llama_cpp" or not is_local(model):
            continue
        if resolve(model.name, catalog) is not None:
            continue
        if lane_for_model(model.name, model.path, catalog, unit_text=unit_text) is not None:
            continue
        row = target_for(model)
        if row.placement is not None and row.placement.problem:
            continue
        rows.append(row)
    return sorted(rows, key=lambda row: row.key)


def load_session_rows(settings: Any, catalog: list[SwapTarget]) -> list[SwapTarget]:
    """:func:`session_rows_for` over the real registry. [] if it is unreadable."""
    try:
        from sqlmodel import Session

        from llmctl.db import get_engine, init_db
        from llmctl.services.registry import RegistryService
        from llmctl.services.serve import effective_host, local_host
        from llmctl.services.session_swap import pinned_port, session_target
        from llmctl.telemetry.gpu import get_gpu_info

        url = settings.database_url
        init_db(url)
        with Session(get_engine(url)) as db:
            models = RegistryService(db).list_models()
            ports = {m.id: pinned_port(db, m) for m in models if m.runtime.value == "llama_cpp"}
        gpus = get_gpu_info()
        margin = settings.scheduler.safety_margin_gb
        units = llama_cpp_units(settings.managed_units)
        here = local_host()
        return session_rows_for(
            models,
            catalog,
            target_for=lambda m: session_target(
                m,
                margin_gb=margin,
                gpus=gpus,
                port=ports.get(m.id),
                unit=unit_for_model(m.path, units),
            ),
            is_local=lambda m: effective_host(m.host) == here,
        )
    except Exception:  # noqa: BLE001 - the picker must render without the DB
        return []


def _load_gpu_sessions(settings: Any) -> list[Any]:
    """GPU processes llmctl started, from the registry. [] if it is unreadable."""
    try:
        from sqlmodel import Session

        from llmctl.db import get_engine, init_db
        from llmctl.services.session_swap import load_gpu_sessions

        url = settings.database_url
        init_db(url)
        with Session(get_engine(url)) as db:
            return load_gpu_sessions(db)
    except Exception:  # noqa: BLE001 - the picker must render without the DB
        return []


def units_from_lanes(
    lanes: Iterable[Lane],
    managed_units: ManagedUnitsConfig,
) -> dict[str, ManagedUnitConfig]:
    """Managed-unit rows for lanes that declare their unit in router.yaml.

    A lane already covered by a ``managed_units`` row (matched by port) is
    skipped, so a hand-written row always wins. Everything else becomes a row
    derived from router.yaml, which is what lets ``llmctl status`` see a lane
    that was never added to settings.yaml — the gap that left FreeToken and
    GLM-5.3 invisible while they were serving.
    """
    taken_ports = {unit.default_port for unit in managed_units.roles().values()}
    derived: dict[str, ManagedUnitConfig] = {}
    for lane in lanes:
        port = _port_of(lane.upstream)
        if lane.unit is None or port is None or port in taken_ports:
            continue
        derived[lane.name] = ManagedUnitConfig(
            unit_name=lane.unit.removesuffix(".service"),
            default_port=port,
            runtime=lane.runtime or "openai",
        )
    return derived
