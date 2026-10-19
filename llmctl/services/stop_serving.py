"""``llmctl stop`` without a session ID: stop what is serving.

``run`` starts one of three things: the vLLM unit on a preset, a router lane
(a systemd user unit), or a process llmctl starts itself (llama.cpp and the
other session engines). ``stop`` with no ID, or with a name from ``llmctl
ls``, stops the same three, so the operator never has to look up an ID.
Ollama and LM Studio models are left alone: their servers unload on their own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llmctl.services.lanes import Gpus, format_gpus


@dataclass(frozen=True)
class Serving:
    """One thing that is serving now and can be stopped."""

    kind: str  # "vllm" | "lane" | "session"
    label: str
    #: Names that pick it on the command line (lane name, served model, ...).
    names: frozenset[str]
    #: Unit name (vllm, lane) or session id (session).
    ref: str
    #: Cards stopping it frees; ``None`` = every card.
    gpus: Gpus = None


def preview_line(item: Serving) -> str:
    """The stop preview's line for ``item``, the cards it frees last."""
    return f"  {item.label} — frees cards {format_gpus(item.gpus)}"


def serving_now(state: Any) -> list[Serving]:
    """Everything ``state`` shows as serving that llmctl knows how to stop."""
    items: list[Serving] = []
    if state.vllm_active:
        presets = [r for r in state.catalog if r.kind == "vllm" and r.key in state.serving]
        model = presets[0].key if presets else "?"
        items.append(
            Serving(
                kind="vllm",
                label=f"vLLM unit {state.vllm_unit} (serving {model})",
                names=frozenset(
                    {
                        state.vllm_unit,
                        "vllm",
                        *(r.key for r in presets),
                        *(r.model for r in presets),
                    }
                ),
                ref=state.vllm_unit,
            )
        )
    for lane in state.active_lanes():
        assert lane.unit is not None
        items.append(
            Serving(
                kind="lane",
                label=f"lane {lane.name} ({lane.unit}, {lane.served_model})",
                names=frozenset({lane.name, lane.unit, lane.served_model}),
                ref=lane.unit,
                gpus=lane.gpus,
            )
        )
    for session in state.gpu_sessions:
        items.append(
            Serving(
                kind="session",
                label=f"{session.label()} (session {session.id[:8]})",
                names=frozenset({session.name, session.id}),
                ref=session.id,
                gpus=frozenset(session.gpu_ids) if session.gpu_ids else None,
            )
        )
    return items


def pick(items: list[Serving], name: str) -> list[Serving]:
    """The items ``name`` selects: an exact name, or an unambiguous session-id prefix."""
    exact = [item for item in items if name in item.names]
    if exact:
        return exact
    # An id prefix picks a session only when exactly one fits: `stop 0` must
    # never stop several sessions that happen to share a first character.
    prefixed = [i for i in items if i.kind == "session" and i.ref.startswith(name)]
    return prefixed if len(prefixed) == 1 else []


@dataclass
class StopDeps:
    """Side-effect seams; the defaults act on the real machine."""

    stop_vllm: Callable[[str], bool]
    stop_lane: Callable[[str], bool]
    stop_session: Callable[[str], bool]


def stop_items(items: list[Serving], deps: StopDeps) -> list[tuple[Serving, bool]]:
    """Stop each item; returns ``(item, stopped)`` pairs in order."""
    results = []
    for item in items:
        fn = {"vllm": deps.stop_vllm, "lane": deps.stop_lane, "session": deps.stop_session}[
            item.kind
        ]
        try:
            ok = bool(fn(item.ref))
        except Exception:  # noqa: BLE001 - reported as not stopped
            ok = False
        results.append((item, ok))
    return results


def real_deps(settings: Any) -> StopDeps:
    """Stops through systemd and the session service."""
    from llmctl.adapters.vllm_systemd import VLLMSystemdAdapter
    from llmctl.services.lane_swap import LaneSwapDeps, stop_user_unit

    def stop_vllm(unit: str) -> bool:
        adapter = VLLMSystemdAdapter(config=settings.managed_units.vllm_engine)
        adapter.stop()
        return not adapter.is_active()

    def stop_session(session_id: str) -> bool:
        from sqlmodel import Session

        from llmctl.db import get_engine, init_db
        from llmctl.services.sessions import SessionService

        init_db(settings.database_url)
        with Session(get_engine(settings.database_url)) as db:
            session = SessionService(db, settings).stop(session_id)
        return session is not None and session.status.value == "stopped"

    return StopDeps(
        stop_vllm=stop_vllm,
        stop_lane=lambda unit: stop_user_unit(unit, LaneSwapDeps()),
        stop_session=stop_session,
    )
