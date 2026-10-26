"""One list of everything ``llmctl run`` can serve: vLLM presets plus lanes.

A *lane* is a self-contained server with its own systemd unit and start script
(FreeToken, the llama.cpp servers, ...). Lanes come from plugins: every
:data:`~llmctl.plugins.LaneSource` registered in the plugin registry
contributes rows (the ``llmctl-stack`` plugin reads llm-router's
``router.yaml``). With no plugin installed there are no lanes and
``llmctl run`` lists vLLM presets and registered models alone.

The lane's start script owns the GPU hand-over (``--take-gpus`` stops vllm-tp
and evicts Ollama models); llmctl runs it, waits for readiness, and rolls back.
A lane that names its own unit needs no ``managed_units`` settings row:
``units_from_lanes`` derives one so ``llmctl status`` sees it too.

Each row has a GPU *footprint*: the cards it occupies. A lane declares it
(``Lane.gpus``); a lane without one, and every vLLM preset (``vllm-tp`` is
TP=2), occupies every card. A swap stops only what overlaps the new target's
footprint.
"""

from __future__ import annotations

import difflib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig

#: Health wait when a lane source gives none.
DEFAULT_LANE_TIMEOUT_S = 300.0

#: Runtime ids of the engines llmctl starts itself for a registered model
#: (as a tracked session), as opposed to a unit or lane something else runs.
SELF_STARTED_RUNTIMES = frozenset({"llama_cpp", "sglang", "freetoken", "strata"})

_ENGINE_LABELS = {
    "vllm": "vLLM",
    "freetoken": "FreeToken",
    "strata": "Strata",
    "llama_cpp": "llama.cpp",
    "sglang": "SGLang",
    "openai": "OpenAI-compat",
    "ollama": "Ollama",
    "llama_swap": "llama-swap",
}


#: Engine label -> what its server process is called in messages.
_SERVER_NAMES = {
    "llama.cpp": "llama-server",
    "SGLang": "sglang server",
    "FreeToken": "ft serve",
    "Strata": "strata server",
}


def server_name(engine_label_text: str) -> str:
    """``llama.cpp`` -> ``llama-server``; other engines name their server plainly."""
    return _SERVER_NAMES.get(engine_label_text, f"{engine_label_text} server")


def engine_label(runtime: str | None) -> str:
    """Human name for a runtime id (``llama_cpp`` -> ``llama.cpp``)."""
    if not runtime:
        return "?"
    return _ENGINE_LABELS.get(runtime, runtime)


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
    if getattr(row, "kind", "") == "vllm" and getattr(row, "tp", None) == 1:
        return "any 1"
    return format_gpus(row.gpus)


def parse_gpus(raw: Any) -> Gpus:
    """A declared ``gpus:`` list -> footprint. Missing or malformed -> every card.

    Fails closed: an unreadable footprint overlaps everything, so a swap stops
    it rather than loading a second engine onto its card.
    """
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in raw):
        return None
    return frozenset(raw)


def port_of(url: str) -> int | None:
    """The port in ``url``, or None when it has none or is malformed."""
    try:
        return urlparse(url).port
    except ValueError:
        return None


@dataclass(frozen=True)
class Lane:
    """A command-launched server a plugin lane source described."""

    name: str
    served_model: str
    upstream: str
    start_cmd: tuple[str, ...]
    health_timeout_s: float = DEFAULT_LANE_TIMEOUT_S
    aliases: tuple[str, ...] = ()
    #: systemd unit (user scope) serving this lane. A source takes it from
    #: llmctl's managed_units when a row there shares the port, else from its
    #: own config — a lane needs no settings row to be complete.
    unit: str | None = None
    #: llmctl runtime id of that unit (``freetoken``, ``llama_cpp``, ...).
    runtime: str | None = None
    #: Cards the lane occupies; None = every card.
    gpus: Gpus = None
    #: How to stop a lane that has no ``unit`` (a router that runs its own
    #: backends, such as llama-swap). Empty: ``llmctl stop`` cannot stop it.
    stop_cmd: tuple[str, ...] = ()
    #: What starting this lane stops, one line each, for lanes whose router
    #: decides that (llama-swap evicts by group). None: llmctl works it out
    #: from units and cards.
    plan: Callable[[], list[str]] | None = None


def load_lanes(
    managed_units: ManagedUnitsConfig | None = None,
    *,
    sources: Sequence[Callable[..., list[Lane]]] | None = None,
) -> list[Lane]:
    """Return every lane the registered lane sources describe, in order.

    ``sources`` pins the list (tests); the default is the plugin registry. A
    source that raises is skipped: one broken config file must not take the
    swap list down with it.
    """
    if sources is None:
        from llmctl.plugins import registry

        sources = registry().lane_sources
    lanes: list[Lane] = []
    for source in sources:
        try:
            lanes += list(source(managed_units))
        except Exception:  # noqa: BLE001 - a lane source is advisory
            continue
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
    #: ``vllm`` rows: the preset's tensor-parallel size (cards it needs).
    tp: int | None = None
    #: ``vllm`` rows once placed (:func:`~llmctl.services.vllm_slots.place`):
    #: the cards it will take and the engine unit that will serve it.
    cards: Gpus = None
    engine_unit: str | None = None

    @property
    def gpus(self) -> Gpus:
        """Cards this target occupies. An unplaced vLLM preset takes every card."""
        return self.lane.gpus if self.lane is not None else self.cards

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
    """vLLM presets first (sorted), then lanes in source order.

    ``presets`` maps alias -> anything with a ``served_name`` attribute
    (:class:`VLLMLaunchSpec` in production).
    """
    rows: list[SwapTarget] = []
    for alias in sorted(presets):
        served = getattr(presets[alias], "served_name", None) or alias
        tp = getattr(presets[alias], "tensor_parallel", None)
        rows.append(
            SwapTarget(
                key=alias,
                kind="vllm",
                model=str(served),
                engine=engine_label("vllm"),
                upstream=f"http://127.0.0.1:{vllm_port}",
                tp=tp if isinstance(tp, int) else None,
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


def close_matches(wanted: str, catalog: list[SwapTarget], n: int = 3) -> list[str]:
    """Up to ``n`` catalog keys or served models that look like ``wanted``, best first."""
    names = list(dict.fromkeys(name for row in catalog for name in (row.key, row.model) if name))
    return difflib.get_close_matches(wanted, names, n=n)


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


def serving_keys(
    catalog: list[SwapTarget],
    probe: Prober,
    loaded_probes: Mapping[str, Prober] | None = None,
) -> set[str]:
    """Keys of rows whose upstream lists their served model right now.

    Each upstream is probed once, however many rows share it (every vLLM
    preset shares :8003). A lane whose runtime has an entry in
    ``loaded_probes`` uses it instead (Ollama's ``/api/ps``, llama-swap's
    ``/running``): those servers' ``/v1/models`` lists every model they could
    serve, so every lane on one server would read serving.
    """
    loaded_probes = loaded_probes or {}
    served_by: dict[tuple[str, str | None], list[str] | None] = {}
    keys: set[str] = set()
    for row in catalog:
        if not row.upstream:
            continue  # a session row: its port is picked at start
        runtime = row.lane.runtime if row.lane is not None else None
        loaded_probe = loaded_probes.get(runtime) if runtime else None
        cache_key = (row.upstream, runtime if loaded_probe else None)
        if cache_key not in served_by:
            served_by[cache_key] = (loaded_probe or probe)(row.upstream)
        ids = served_by[cache_key] or []
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
    entry = data.get(key) if isinstance(data.get(key), dict) else {}
    # Merge: the entry also holds the lane's measured VRAM (vram_fit.record_vram).
    data[key] = {**entry, "seconds": round(seconds, 1), "at": int(time.time())}
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
    #: Why nothing is stopped (the target fits beside what runs), if so.
    note: str = ""
    #: ``stops`` came from the lane's own plan: empty means nothing stops,
    #: not that nothing runs.
    planned: bool = False

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
        if self.note:
            keeps = ", ".join(self.stops) if self.stops else "nothing; everything running stays up"
            out = [self.note, f"stops:  {keeps}"]
        else:
            empty = "nothing" if self.planned else "nothing is running"
            out = [f"stops:  {', '.join(self.stops) if self.stops else empty}"]
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
    alongside: Fit | None = None,
    engine_lanes: Sequence[Lane] = (),
) -> SwapPreview:
    """Describe the swap to ``target`` given what is running now.

    ``active_units`` are lane units whose systemd state is active (serving or
    still loading); ``serving`` are catalog keys answering right now.
    ``alongside`` (a fit) means ``target`` starts beside what runs: only the
    units its ``Conflicts=`` names are stopped.
    """
    if alongside is not None and target.lane is not None:
        return _preview_alongside(
            target, catalog, serving, active_units, load_times, gpu_sessions, alongside
        )
    preview = SwapPreview(
        target=target,
        already_serving=target.key in serving
        or session_is_running(target, active_units, gpu_sessions),
        last_load_s=last_load_s(target.key, load_times),
    )
    stops: list[str] = []
    on_engine = target.engine_unit or vllm_unit
    if vllm_active:
        current = next((r.model for r in catalog if r.kind == "vllm" and r.key in serving), None)
        label = f"{vllm_unit} ({current})" if current else vllm_unit
        if target.kind == "vllm" and on_engine == vllm_unit:
            stops.append(f"{label}, restarted on preset {target.key}")
        else:
            stops.append(label)
    for lane in engine_lanes:
        stops.append(f"{lane.unit} ({lane.served_model})")
    for row in catalog:
        if row.kind != "lane" or row.lane is None or row.lane.unit is None:
            continue
        if row.key == target.key or not lane_holds_unit(row, catalog, serving, active_units):
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
        preview.starts = f"{server_name(target.engine)} for {target.key} on {where}; {port}"
        note = getattr(placement, "note", None)
        if note:
            stops.append(note)
    elif target.kind == "vllm":
        stops.append("Ollama and plugin companions (paused while vLLM loads, started again after)")
        where = f" on GPU {format_gpus(target.cards)}" if target.cards is not None else ""
        preview.starts = f"{on_engine} on preset {target.key}{where}"
    else:
        lane = target.lane
        assert lane is not None
        if "--take-gpus" in lane.start_cmd:
            stops.append("Ollama models on its GPUs (start script, --take-gpus)")
        if lane.plan is not None:
            preview.planned = True
            try:
                stops += lane.plan()
            except Exception as exc:  # noqa: BLE001 - a preview must not fail the run
                stops.append(f"unknown ({exc}); the start command checks again")
        command = " ".join(lane.start_cmd)
        preview.starts = f"{lane.unit} via {command}" if lane.unit else command
    preview.stops = stops
    return preview


@dataclass(frozen=True)
class Fit:
    """A lane fits beside what is running: its last measured need vs free now."""

    need: dict[int, int]
    free: dict[int, int]
    #: Units the target's ``Conflicts=`` stops anyway.
    conflicts: frozenset[str] = frozenset()

    def describe(self) -> str:
        from llmctl.services.vram_fit import format_mib

        free = ", ".join(
            f"{self.free[c] / 1024:.1f} GiB free on GPU {c}" for c in sorted(self.need)
        )
        return f"fits beside what is running: needs {format_mib(self.need)}; {free}"


def share_start_cmd(start_cmd: Sequence[str]) -> tuple[str, ...]:
    """The start command with ``--take-gpus`` swapped for ``--share-gpus``.

    ``--share-gpus`` tells a lane script llmctl already checked the fit: start
    without stopping vLLM, Ollama or other lanes, and without its busy-card refusal.
    """
    return tuple("--share-gpus" if a == "--take-gpus" else a for a in start_cmd)


def _preview_alongside(
    target: SwapTarget,
    catalog: list[SwapTarget],
    serving: set[str],
    active_units: set[str],
    load_times: Path | None,
    gpu_sessions: Sequence[Any],
    fit: Fit,
) -> SwapPreview:
    lane = target.lane
    assert lane is not None
    preview = SwapPreview(
        target=target,
        already_serving=target.key in serving
        or session_is_running(target, active_units, gpu_sessions),
        last_load_s=last_load_s(target.key, load_times),
    )
    stops = [
        f"{row.lane.unit} ({row.model}, its unit Conflicts= with {lane.unit})"
        for row in catalog
        if row.lane is not None
        and row.lane.unit in fit.conflicts
        and row.key != target.key
        and lane_holds_unit(row, catalog, serving, active_units)
    ]
    preview.stops = stops
    unit = lane.unit or "its unit"
    preview.starts = f"{unit} via {' '.join(share_start_cmd(lane.start_cmd))}"
    preview.note = fit.describe()
    return preview


def lane_holds_unit(
    row: SwapTarget, catalog: list[SwapTarget], serving: set[str], active_units: set[str]
) -> bool:
    """The lane's unit is up and running this lane, not a sibling's model.

    Lanes can share a unit (``coder1`` and ``gemma1`` are two models on
    ollama-gpu1). While a sibling on the unit is serving, this lane is not
    the one up; with none serving, every lane on it may be loading.
    """
    if row.lane is None or row.lane.unit not in active_units:
        return False
    if row.key in serving:
        return True
    return not any(
        other.key in serving
        for other in catalog
        if other.lane is not None and other.lane.unit == row.lane.unit
    )


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
    #: Free MiB per card now. None (unread) = nothing fits alongside.
    free_mib: dict[int, int] | None = None
    #: catalog key -> per-card MiB last measured, or None if never.
    vram_need: Callable[[str], dict[int, int] | None] | None = None
    #: unit -> the units its systemd ``Conflicts=`` stops.
    conflicts_of: Callable[[str], set[str] | None] | None = None
    #: Cards the vLLM engine unit holds (its env's CUDA_VISIBLE_DEVICES).
    vllm_cards: Gpus = None
    #: Running second-engine units, as lanes: stopped and restored like one.
    engine_lanes: list[Lane] = field(default_factory=list)
    #: The second engine's config, when one is set up.
    engine2: Any = None
    #: vLLM preset keys the second engine is serving (not the main one).
    engine2_serving: set[str] = field(default_factory=set)
    #: GPUs on this host (placement needs it); None = unknown.
    card_count: int | None = None

    def state_of(self, row: SwapTarget) -> str:
        """``serving`` / ``loading`` / ``""`` for the picker's State column."""
        if row.key in self.serving:
            return "serving"
        if session_is_running(row, self.active_units, self.gpu_sessions):
            return "serving"
        if self._holds_unit(row):
            return "loading"
        return ""

    def _holds_unit(self, row: SwapTarget) -> bool:
        return lane_holds_unit(row, self.catalog, self.serving, self.active_units)

    def active_lanes(self, exclude: str | None = None) -> list[Lane]:
        """Lanes whose unit is active, except ``exclude`` (a catalog key).

        Includes a running second vLLM engine.
        """
        return [
            row.lane
            for row in self.catalog
            if row.lane is not None and self._holds_unit(row) and row.key != exclude
        ] + list(self.engine_lanes)

    def vllm_in_the_way(self, target: SwapTarget) -> bool:
        """The vLLM engine unit is up and the swap stops or restarts it.

        It does when it holds a card ``target`` needs, or when ``target`` is a
        preset that will be started on it (a restart replaces its model).
        """
        if not self.vllm_active:
            return False
        if target.kind == "vllm" and (target.engine_unit or self.vllm_unit) == self.vllm_unit:
            return True
        return gpus_overlap(self.vllm_cards, target.gpus)

    def fits_alongside(self, target: SwapTarget) -> Fit | None:
        """A :class:`Fit` when lane ``target`` can start beside what runs, else None.

        Only a lane whose script takes ``--take-gpus`` (and so knows
        ``--share-gpus``) and whose VRAM was measured before qualifies.
        """
        from llmctl.services.vram_fit import fits

        lane = target.lane
        if lane is None or lane.unit is None or "--take-gpus" not in lane.start_cmd:
            return None
        if self.free_mib is None or self.vram_need is None:
            return None
        need = self.vram_need(target.key)
        if not need or (lane.gpus is not None and not set(need) <= lane.gpus):
            return None
        if not fits(need, self.free_mib):
            return None
        conflicts = self.conflicts_of(lane.unit) if self.conflicts_of else set()
        if conflicts is None:
            return None  # Conflicts= unreadable: stop what overlaps, as before
        return Fit(need=dict(need), free=dict(self.free_mib), conflicts=frozenset(conflicts))

    def lanes_in_the_way(self, target: SwapTarget, fit: Fit | None = None) -> list[Lane]:
        """Active lanes a swap to ``target`` must stop: those sharing a card.

        With ``fit`` (see :meth:`fits_alongside`), only those its unit conflicts with.
        """
        lanes = [
            row.lane
            for row in self.catalog
            if row.lane is not None
            and self._holds_unit(row)
            and row.key != target.key
            and (
                row.lane.unit in fit.conflicts
                if fit is not None
                else gpus_overlap(row.gpus, target.gpus)
            )
        ]
        # A running second engine: in the way where its cards overlap, unless
        # it is the engine the target will be (re)started on.
        lanes += [
            lane
            for lane in self.engine_lanes
            if lane.unit != target.engine_unit
            and (
                lane.unit in fit.conflicts
                if fit is not None
                else gpus_overlap(lane.gpus, target.gpus)
            )
        ]
        return lanes

    def preview(self, target: SwapTarget, load_times: Path | None = None) -> SwapPreview:
        """:func:`preview_swap` against this state (a lane that fits stops nothing)."""
        return preview_swap(
            target,
            self.catalog,
            serving=self.serving,
            active_units=self.active_units,
            vllm_unit=self.vllm_unit,
            vllm_active=self.vllm_in_the_way(target),
            engine_lanes=[
                lane
                for lane in self.engine_lanes
                if lane.unit != target.engine_unit and gpus_overlap(lane.gpus, target.gpus)
            ],
            load_times=load_times,
            gpu_sessions=self.gpu_sessions,
            alongside=self.fits_alongside(target),
        )

    def sessions_in_the_way(self, target: SwapTarget, fit: Fit | None = None) -> list[Any]:
        """llmctl-started GPU processes a swap to ``target`` must stop (none if it fits)."""
        if fit is not None:
            return []
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
    loaded_probes: Mapping[str, Prober] | None = None,
    free_mib: dict[int, int] | None = None,
    vram_need: Callable[[str], dict[int, int] | None] | None = None,
    conflicts_of: Callable[[str], set[str] | None] | None = None,
    live_fit: bool = True,
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
        from llmctl.services.backends import LOADED_PROBES, probe_openai_v1_models

        probe = probe_openai_v1_models
        if loaded_probes is None:
            loaded_probes = LOADED_PROBES
    if is_active is None:
        from llmctl.integrations.systemctl import SystemctlRunner

        runner = SystemctlRunner()

        def is_active(unit: str, user: bool) -> bool:
            return runner.is_active(unit, user=user)

    vllm = settings.managed_units.vllm_engine
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
    if live_fit:
        from llmctl.services import vram_fit

        free_mib = free_mib if free_mib is not None else vram_fit.free_mib_now()
        vram_need = vram_need or vram_fit.last_vram_mib
        conflicts_of = conflicts_of or vram_fit.unit_conflicts
    from llmctl.services import vllm_slots

    serving = serving_keys(catalog, probe, loaded_probes)
    engine2 = getattr(settings.managed_units, "vllm_engine2", None)
    engine_lanes: list[Lane] = []
    engine2_serving: set[str] = set()
    if engine2 is not None and is_active(engine2.unit_name, engine2.user_scope):
        served = probe(f"http://127.0.0.1:{engine2.default_port}") or []
        engine2_serving = {r.key for r in catalog if r.kind == "vllm" and r.model in served}
        serving |= engine2_serving
        engine_lanes.append(vllm_slots.engine_lane(engine2, served[0] if served else ""))
    return SwapState(
        catalog=catalog,
        serving=serving,
        engine_lanes=engine_lanes,
        engine2=engine2,
        engine2_serving=engine2_serving,
        vllm_cards=vllm_slots.engine_cards(vllm),
        card_count=len(free_mib) if free_mib else None,
        active_units=active_units,
        vllm_unit=vllm.unit_name,
        vllm_active=is_active(vllm.unit_name, False),
        gpu_sessions=gpu_sessions,
        free_mib=free_mib,
        vram_need=vram_need,
        conflicts_of=conflicts_of,
    )


def session_rows_for(
    models: Iterable[Any],
    catalog: list[SwapTarget],
    *,
    target_for: Callable[[Any], SwapTarget],
    is_local: Callable[[Any], bool],
    unit_text: Callable[[str], str] = user_unit_text,
) -> list[SwapTarget]:
    """Swap-list rows for the models llmctl starts itself, sorted by name.

    That is every llama.cpp, SGLang and FreeToken model (see
    :data:`SELF_STARTED_RUNTIMES`).

    Skipped: other engines, models on another machine, models a listed row
    already serves (same name, or a lane whose unit loads the same file), and
    models that fit on no combination of cards.
    """
    rows: list[SwapTarget] = []
    for model in models:
        if getattr(model.runtime, "value", model.runtime) not in SELF_STARTED_RUNTIMES:
            continue
        if not is_local(model):
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
            ports = {
                m.id: pinned_port(db, m) for m in models if m.runtime.value in SELF_STARTED_RUNTIMES
            }
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
    """Managed-unit rows for lanes that declare their own unit.

    A lane already covered by a ``managed_units`` row (matched by port) is
    skipped, so a hand-written row always wins. Everything else becomes a row
    derived from the lane, which is what lets ``llmctl status`` see a lane
    that was never added to settings.yaml — the gap that left FreeToken and
    GLM-5.3 invisible while they were serving.
    """
    taken_ports = {unit.default_port for unit in managed_units.roles().values()}
    derived: dict[str, ManagedUnitConfig] = {}
    for lane in lanes:
        port = port_of(lane.upstream)
        if lane.unit is None or port is None or port in taken_ports:
            continue
        derived[lane.name] = ManagedUnitConfig(
            unit_name=lane.unit.removesuffix(".service"),
            default_port=port,
            runtime=lane.runtime or "openai",
        )
    return derived
