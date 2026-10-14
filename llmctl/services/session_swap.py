"""Swap the GPUs over to a model that llmctl starts itself (llama.cpp, SGLang, FreeToken).

Such a model in the registry has no systemd unit: llmctl runs the engine's
server (``llama-server``, ``python -m sglang.launch_server``, ``ft serve``)
as a background process and keeps a record of it (a
"session": pid, port, status; ``llmctl sessions`` lists them, ``llmctl stop``
ends one). Before this module, ``swap`` started that process *next to*
whatever held the GPUs, which only worked when the cards happened to be empty.
Now it is a real swap, in the same order as a lane swap:

1. **Fit check first, nothing stopped yet.** One card when the model fits on
   one, both when it only fits split across two, refused when it fits on
   neither. vLLM holds both cards, so free memory right now says nothing;
   the check is against each card's total size.
2. Stop other llama.cpp processes llmctl started, the active lanes, and
   ``vllm-tp``.
3. Start ``llama-server`` and wait until it answers.
4. On failure: stop what was half-started, then start vLLM (on its previous
   preset), the lanes and the stopped processes again.

The reverse matters as much: :func:`pause_gpu_sessions` lets a swap *to*
vLLM or a lane stop these processes first (and restart them on failure),
because vLLM will not load next to them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from llmctl.services.lane_swap import (
    LaneSwapDeps,
    LaneSwapResult,
    restore_previous,
    stop_user_unit,
)
from llmctl.services.lanes import Gpus, Lane, gpus_overlap

#: GiB kept free per card on top of the model's own estimate.
DEFAULT_MARGIN_GB = 1.0
#: Freed VRAM can lag the unit stop by a moment; retry a refused start.
_VRAM_RETRIES = 4
_VRAM_RETRY_S = 3.0


@dataclass(frozen=True)
class GpuSession:
    """A process llmctl started that holds GPU memory."""

    id: str
    name: str
    runtime: str
    port: int | None
    gpu_ids: tuple[int, ...]

    def label(self) -> str:
        what = "llama-server" if self.runtime == "llama_cpp" else f"{self.runtime} process"
        where = f", port {self.port}" if self.port else ""
        return f"{what} for {self.name}{where}"


def load_gpu_sessions(db: Any) -> list[GpuSession]:
    """Running processes llmctl started that sit on a GPU."""
    from sqlmodel import select

    from llmctl.db import ModelRecord, SessionKind, SessionRecord, SessionStatus

    live = (SessionStatus.RUNNING, SessionStatus.STARTING, SessionStatus.DEGRADED)
    rows = db.exec(
        select(SessionRecord).where(SessionRecord.status.in_(live))  # type: ignore[attr-defined]
    ).all()
    out = []
    for row in rows:
        if row.kind == SessionKind.ADOPTED or not row.gpu_ids:
            continue
        model = db.get(ModelRecord, row.model_id) if row.model_id else None
        out.append(
            GpuSession(
                id=row.id,
                name=model.name if model else (row.model_id or row.id[:8]),
                runtime=row.runtime.value,
                port=row.port,
                gpu_ids=tuple(row.gpu_ids),
            )
        )
    return out


def sessions_in_the_way(sessions: Sequence[GpuSession], gpus: Gpus) -> list[GpuSession]:
    """Sessions on a card that ``gpus`` (None = every card) needs."""
    return [s for s in sessions if gpus_overlap(frozenset(s.gpu_ids), gpus)]


# -- fit --------------------------------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """How many cards the model gets, or why it cannot be placed."""

    cards: int
    problem: str | None = None
    note: str | None = None


def place(
    name: str,
    estimated_gb: float | None,
    card_totals_gb: Sequence[float],
    *,
    margin_gb: float = DEFAULT_MARGIN_GB,
) -> Placement:
    """Fewest cards that hold ``estimated_gb``, judged against card *totals*."""
    if estimated_gb is None:
        return Placement(
            cards=1,
            note=f"{name} has no VRAM estimate, so the fit is not checked up front; "
            "llama-server-guarded checks free memory when it starts.",
        )
    totals = sorted(card_totals_gb, reverse=True)
    if not totals:
        return Placement(cards=0, problem="No NVIDIA GPUs detected.")
    for cards in range(1, len(totals) + 1):
        if estimated_gb / cards + margin_gb <= totals[cards - 1]:
            return Placement(cards=cards)
    return Placement(
        cards=0,
        problem=f"{name} needs about {estimated_gb:.0f} GB and all {len(totals)} GPUs "
        f"together hold {sum(totals):.0f} GB. Nothing was stopped.",
    )


# -- the swap -------------------------------------------------------------------


@dataclass
class SessionSwapResult:
    """What happened; ``ok`` means llama-server answers."""

    ok: bool
    seconds: float = 0.0
    error: str | None = None
    session: Any = None
    stopped: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    restore_failed: list[str] = field(default_factory=list)


def pause_gpu_sessions(
    sessions: Sequence[GpuSession],
    stop: Callable[[str], object],
    log: Callable[[str], None],
) -> tuple[list[GpuSession], str | None]:
    """Stop ``sessions``; returns (stopped, error). Stops at the first failure."""
    stopped: list[GpuSession] = []
    for session in sessions:
        log(f"stopping {session.label()} ...")
        try:
            stop(session.id)
        except Exception as exc:  # noqa: BLE001 - reported, then rolled back
            return stopped, f"could not stop {session.label()}: {exc}"
        stopped.append(session)
    return stopped, None


def resume_gpu_sessions(
    sessions: Sequence[GpuSession],
    restart: Callable[[str], object],
    log: Callable[[str], None],
    result: SessionSwapResult | LaneSwapResult,
) -> None:
    """Start ``sessions`` again after a failed swap."""
    for session in sessions:
        try:
            restart(session.id)
        except Exception:  # noqa: BLE001 - reported below
            result.restore_failed.append(session.label())
            log(f"  rollback: {session.label()} did not come back; run: llmctl restart "
                f"{session.id[:8]}")
            continue
        result.restored.append(session.label())
        log(f"  rollback: started {session.label()} again")


def swap_to_session(
    name: str,
    *,
    placement: Placement,
    start: Callable[[int], Any],
    stop: Callable[[str], object],
    restart: Callable[[str], object],
    gpu_sessions: Sequence[GpuSession],
    active_lanes: Sequence[Lane],
    vllm_active: bool,
    vllm_unit: str,
    deps: LaneSwapDeps | None = None,
    engine: str = "llama-server",
) -> SessionSwapResult:
    """Free the GPUs, start ``name`` via ``start(cards)``, roll back on failure.

    ``start(cards)`` launches and returns the session (it blocks until the
    server answers or fails); it may raise when the scheduler refuses.
    """
    deps = deps or LaneSwapDeps()
    result = SessionSwapResult(ok=False)
    if placement.problem:
        result.error = placement.problem
        return result
    began = deps.clock()

    stopped_sessions, error = pause_gpu_sessions(gpu_sessions, stop, deps.log)
    result.stopped += [s.label() for s in stopped_sessions]
    stopped_lanes: list[Lane] = []
    if error is None:
        for lane in active_lanes:
            if lane.unit is None:
                continue
            deps.log(f"stopping {lane.unit} ...")
            if not stop_user_unit(lane.unit, deps):
                error = f"could not stop {lane.unit}"
                break
            stopped_lanes.append(lane)
            result.stopped.append(lane.unit)
    stopped_vllm = False
    if error is None and vllm_active:
        deps.log(f"stopping {vllm_unit} ...")
        if deps.run_quiet(["sudo", "-n", "systemctl", "stop", vllm_unit]) == 0:
            stopped_vllm = True
            result.stopped.append(vllm_unit)
        else:
            error = f"could not stop {vllm_unit} (sudo -n systemctl stop {vllm_unit})"

    if error is None:
        where = "one GPU" if placement.cards == 1 else f"{placement.cards} GPUs"
        deps.log(f"starting {engine} for {name} on {where} (this can take minutes) ...")
        for attempt in range(_VRAM_RETRIES + 1):
            try:
                session = start(placement.cards)
            except Exception as exc:  # noqa: BLE001 - scheduler refusal, retried
                message = str(exc)
                if "exceeds free" in message and attempt < _VRAM_RETRIES:
                    deps.sleep(_VRAM_RETRY_S)  # memory from the stopped units lags
                    continue
                error = message
                break
            result.session = session
            status = getattr(getattr(session, "status", None), "value", "")
            if status == "running":
                result.ok = True
            else:
                detail = getattr(session, "error", None) or f"status {status or 'unknown'}"
                error = f"{engine} did not come up: {detail}"
            break

    result.seconds = deps.clock() - began
    if result.ok:
        deps.record(name, result.seconds)
        return result

    result.error = error
    deps.log(f"swap failed: {error}; restoring the previous state")
    session = result.session
    if session is not None and getattr(session, "id", None):
        try:
            stop(session.id)
            deps.log(f"  rollback: stopped the half-started {engine}")
        except Exception:  # noqa: BLE001 - best effort; reported by status
            deps.log(f"  rollback: could not stop it; run: llmctl stop {session.id[:8]}")
    lane_result = LaneSwapResult(ok=False)
    restore_previous(
        stopped_lanes,
        restart_vllm=stopped_vllm,
        vllm_unit=vllm_unit,
        deps=deps,
        result=lane_result,
    )
    result.restored += lane_result.restored
    result.restore_failed += lane_result.restore_failed
    resume_gpu_sessions(stopped_sessions, restart, deps.log, result)
    return result


def pinned_port(db: Any, model: Any) -> int | None:
    """The port ``model``'s default profile pins, or None (llmctl picks one)."""
    profile_id = getattr(model, "default_profile_id", None)
    if not profile_id:
        return None
    from llmctl.db import ProfileRecord
    from llmctl.services.profiles import effective_parameters
    from llmctl.services.scheduler import SchedulerService

    profile = db.get(ProfileRecord, profile_id)
    return SchedulerService._pinned_port(effective_parameters(profile)) if profile else None


def session_target(
    model: Any,
    *,
    margin_gb: float | None = None,
    gpus: Any = None,
    port: int | None = None,
    unit: str | None = None,
) -> Any:
    """The swap-list row for a llama.cpp registry model, fit decided now."""
    from llmctl.services.lanes import SwapTarget, engine_label

    if gpus is None:
        from llmctl.telemetry.gpu import get_gpu_info

        gpus = get_gpu_info()
    if margin_gb is None:
        from llmctl.config import load_settings

        margin_gb = load_settings().scheduler.safety_margin_gb
    totals = [(g.memory_total_mb or 0) / 1024 for g in gpus]
    placement = place(model.name, model.estimated_vram_gb, totals, margin_gb=margin_gb)
    return SwapTarget(
        key=model.name,
        kind="session",
        model=model.name,
        engine=engine_label(model.runtime.value),
        upstream="",
        model_id=model.id,
        placement=placement,
        port=port,
        unit=unit,
    )


def start_request_fields(record: Any, cards: int) -> dict[str, Any]:
    """The session start request a swap makes for ``record`` on ``cards`` GPUs.

    The model's default profile rides along, so per-model launch settings
    (a pinned port, extra flags) apply to swaps too. The placement's card
    count overrides any tensor_parallel_size the profile carries.
    """
    return {
        "model_id": record.id,
        "runtime": record.runtime,
        "profile_id": record.default_profile_id or None,
        "gpus_auto": True,
        "dry_run": False,
        "parameters": {"tensor_parallel_size": cards},
    }


def run_session_swap(
    state: Any, target: Any, *, settings: Any, deps: LaneSwapDeps | None = None
) -> SessionSwapResult:
    """:func:`swap_to_session` against the real registry and systemd."""
    from sqlmodel import Session

    from llmctl.db import ModelRecord, get_engine, init_db
    from llmctl.schemas import SessionStartRequest
    from llmctl.services.sessions import SessionService

    url = settings.database_url
    init_db(url)
    with Session(get_engine(url)) as db:
        service = SessionService(db, settings)
        record = db.get(ModelRecord, target.model_id)
        runtime = record.runtime if record is not None else None

        def start(cards: int) -> Any:
            if record is None or runtime is None:
                raise ValueError(f"model {target.key} is no longer registered")
            return service.start(SessionStartRequest(**start_request_fields(record, cards)))

        return swap_to_session(
            target.key,
            placement=target.placement,
            start=start,
            stop=lambda sid: service.stop(sid),
            restart=lambda sid: service.restart(sid),
            gpu_sessions=state.sessions_in_the_way(target),
            active_lanes=state.active_lanes(),
            vllm_active=state.vllm_active,
            vllm_unit=state.vllm_unit,
            deps=deps,
            engine=_server_name(runtime),
        )


def _server_name(runtime: Any) -> str:
    """What the started process is called in messages (``llama-server``, ``ft serve``)."""
    value = getattr(runtime, "value", runtime)
    names = {
        "llama_cpp": "llama-server",
        "freetoken": "ft serve",
        "sglang": "sglang",
        "strata": "strata server",
    }
    return names.get(
        str(value), str(value)
    )


def pause_for_swap(
    state: Any, target: Any, *, settings: Any, log: Callable[[str], None]
) -> tuple[list[GpuSession], str | None]:
    """Before a vLLM or lane swap: stop llmctl's GPU processes in its way."""
    in_the_way = state.sessions_in_the_way(target)
    if not in_the_way:
        return [], None
    with _service(settings) as service:
        return pause_gpu_sessions(in_the_way, lambda sid: service.stop(sid), log)


def resume_after_failed_swap(
    sessions: Sequence[GpuSession], *, settings: Any, log: Callable[[str], None]
) -> SessionSwapResult:
    """After a failed vLLM or lane swap: start the paused processes again."""
    result = SessionSwapResult(ok=False)
    if sessions:
        with _service(settings) as service:
            resume_gpu_sessions(sessions, lambda sid: service.restart(sid), log, result)
    return result


class _service:
    """``with _service(settings) as svc`` -> a SessionService on its own DB session."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._db: Any = None

    def __enter__(self) -> Any:
        from sqlmodel import Session

        from llmctl.db import get_engine, init_db
        from llmctl.services.sessions import SessionService

        url = self._settings.database_url
        init_db(url)
        self._db = Session(get_engine(url))
        return SessionService(self._db, self._settings)

    def __exit__(self, *exc: object) -> None:
        self._db.close()


__all__ = [
    "GpuSession",
    "Placement",
    "SessionSwapResult",
    "load_gpu_sessions",
    "pause_for_swap",
    "resume_after_failed_swap",
    "run_session_swap",
    "session_target",
    "pause_gpu_sessions",
    "place",
    "resume_gpu_sessions",
    "sessions_in_the_way",
    "swap_to_session",
]
