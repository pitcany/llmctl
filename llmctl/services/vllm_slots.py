"""Where a vLLM preset runs: which cards, and on which engine unit.

A preset needs ``tp`` cards. One that needs every card (TP=2 on this host)
takes the main engine (``vllm-engine``) and stops everything, as before. A
one-card preset (``tp: 1``, or ``llmctl run NAME --tp 1``) takes one card:

* the card is the one nothing holds, else the one with the most free VRAM;
* it runs on ``vllm-engine`` unless that engine is up on the *other* card,
  in which case it runs on the second engine (``managed_units.vllm_engine2``,
  a user unit) so both keep serving.

Only what holds the chosen card is stopped. ``CUDA_VISIBLE_DEVICES`` is the
chosen card, so vLLM sizes its KV cache from that card alone.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from llmctl.services.lanes import Gpus, Lane, SwapState, SwapTarget, gpus_overlap

#: Readiness budget when a failed swap starts the second engine again.
_ENGINE_TIMEOUT_S = 900.0


def parse_cards(value: str | None) -> Gpus:
    """``"1"`` -> ``{1}``; ``"0,1"`` -> ``{0, 1}``; unset or unreadable -> every card."""
    if not value:
        return None
    try:
        cards = frozenset(int(part) for part in value.split(",") if part.strip())
    except ValueError:
        return None
    return cards or None


def engine_cards(unit: Any) -> Gpus:
    """Cards an engine unit's env file pins (``CUDA_VISIBLE_DEVICES``)."""
    try:
        text = unit.resolve_env_file().read_text()
    except (OSError, AttributeError):
        return None
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "CUDA_VISIBLE_DEVICES":
            return parse_cards(value.strip().strip("\"'"))
    return None


def unit_service(name: str) -> str:
    """``vllm-engine2`` -> ``vllm-engine2.service``."""
    return name if name.endswith(".service") else f"{name}.service"


def engine_lane(unit: Any, served_model: str) -> Lane:
    """A running second engine as a :class:`Lane`: stop it, restart it, wait for it.

    Its env file is left alone on a stop, so starting the unit again brings
    back the same model.
    """
    service = unit_service(unit.unit_name)
    return Lane(
        name=unit.unit_name,
        served_model=served_model,
        upstream=f"http://127.0.0.1:{unit.default_port}",
        start_cmd=("systemctl", "--user", "start", service),
        health_timeout_s=_ENGINE_TIMEOUT_S,
        unit=service,
        runtime="vllm",
        gpus=engine_cards(unit),
    )


def _held(state: SwapState, card: int) -> bool:
    """Something holds ``card``: the main engine, a lane, or a session."""
    one = frozenset({card})
    if state.vllm_active and gpus_overlap(state.vllm_cards, one):
        return True
    if any(gpus_overlap(lane.gpus, one) for lane in state.active_lanes()):
        return True
    return any(gpus_overlap(frozenset(s.gpu_ids), one) for s in state.gpu_sessions)


def place(state: SwapState, target: SwapTarget, tp: int | None) -> SwapTarget:
    """``target`` with its cards and engine set (a copy). Non-vLLM rows pass through.

    Left unplaced (every card, the main engine: the old behaviour) when
    ``tp`` is unknown or covers every card, or the card count is unknown.
    """
    if target.kind != "vllm":
        return target
    count = state.card_count
    if not tp or count is None or tp >= count or tp != 1:
        return target
    free = state.free_mib or {}
    card = min(range(count), key=lambda c: (_held(state, c), -free.get(c, 0), c))
    engine = state.vllm_unit
    main_elsewhere = (
        state.vllm_active and state.vllm_cards is not None and card not in state.vllm_cards
    )
    if main_elsewhere and state.engine2 is not None:
        engine = unit_service(state.engine2.unit_name)
    return dataclasses.replace(target, cards=frozenset({card}), engine_unit=engine)


def engine_config(settings: Any, target: SwapTarget) -> Any:
    """The :class:`ManagedUnitConfig` that serves placed ``target``."""
    engine2 = getattr(settings.managed_units, "vllm_engine2", None)
    if (
        engine2 is not None
        and target.engine_unit is not None
        and target.engine_unit == unit_service(engine2.unit_name)
    ):
        return engine2
    return settings.managed_units.vllm_engine


def cuda_devices(target: SwapTarget) -> str | None:
    """``CUDA_VISIBLE_DEVICES`` for a placed target; None = the preset's own."""
    if target.cards is None:
        return None
    return ",".join(str(c) for c in sorted(target.cards))
