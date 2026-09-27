"""How to serve a registered model, whatever its engine.

``llmctl swap NAME`` is the one serve verb. Presets and router.yaml lanes are
its own list; :func:`route` answers for every other registered model so the
operator never has to know that llama.cpp rows start as sessions, that Ollama
loads on demand, or that a vLLM checkpoint needs a preset first:

========== ==================================================================
kind       what ``swap`` does
========== ==================================================================
preset     swap the vLLM unit to the preset that serves this checkpoint
no-preset  vLLM checkpoint with no preset: offer to write a starter preset
session    llama.cpp row: free the GPUs, start llama-server, roll back on
           failure (:mod:`llmctl.services.session_swap`)
start      python_script row: run it next to everything else, no swap
on-demand  Ollama / LM Studio: nothing to start, the first request loads it
cannot     SGLang, adopted endpoints, other machines, a model too
           big for every GPU: says why, changes nothing
========== ==================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlmodel import Session

from llmctl.db import RuntimeName
from llmctl.schemas import Model
from llmctl.services.lanes import SwapTarget
from llmctl.services.registry import RegistryService, effective_host, local_host


@dataclass(frozen=True)
class ServeRoute:
    """Where ``swap NAME`` goes for a registered model."""

    kind: str
    model: Model
    message: str
    #: ``preset`` / ``session`` / ``lane``: the swap-list row to swap to.
    target: SwapTarget | None = None


class AmbiguousModel(ValueError):
    """NAME matches several local rows; the message lists them."""


def find_model(db: Session, name: str) -> Model | None:
    """A registry row by id, id prefix, name, or source; local rows first.

    Names repeat across hosts (the same LM Studio model on three Macs), so a
    row on this machine wins over rows elsewhere. Two different local rows
    with the same name are ambiguous and raise, naming both ids.
    """
    wanted = name.strip()
    if not wanted:
        return None
    models = RegistryService(db).list_models()
    lowered = wanted.lower()
    hits = [m for m in models if m.id == wanted]
    if not hits and len(wanted) >= 6:
        hits = [m for m in models if (m.id or "").startswith(wanted)]
    if not hits:
        hits = [
            m
            for m in models
            if lowered in {(m.name or "").lower(), (m.source or "").lower()}
        ]
    if not hits:
        return None
    here = local_host()
    local = [m for m in hits if effective_host(m.host) == here]
    pool = local or hits
    if len(pool) > 1 and local:
        listed = ", ".join(f"{m.id[:8]} ({m.runtime.value})" for m in pool if m.id)
        raise AmbiguousModel(f"'{name}' matches several models here: {listed}. Use the id.")
    return pool[0]


def route(db: Session, model: Model, catalog: list[SwapTarget]) -> ServeRoute:
    """Decide how ``model`` is served. Reads presets and GPU sizes; writes nothing.

    Messages are for the operator, so they say what happens to the GPUs and
    what to type next, not which llmctl subsystem is involved.
    """
    host = effective_host(model.host)
    if host != local_host():
        return ServeRoute(
            "cannot",
            model,
            f"{model.name} is on {host}, not this machine. Nothing here was changed; "
            "serve it from that machine.",
        )
    engine = model.runtime
    if engine == RuntimeName.OLLAMA:
        return ServeRoute(
            "on-demand",
            model,
            f"Nothing to start: Ollama is already running and loads {model.name} into "
            "memory the first time something asks for it, for example "
            f'`ollama run {model.name} "hi"`. The GPUs were left as they are.',
        )
    if engine == RuntimeName.LMSTUDIO:
        return ServeRoute(
            "on-demand",
            model,
            f"Nothing to start: LM Studio's server (port 1234) loads {model.name} the "
            f"first time a request names it, or now with `lms load {model.source or model.name}`. "
            "The GPUs were left as they are.",
        )
    if engine == RuntimeName.VLLM:
        alias = _serving_preset(db, model)
        target = next((t for t in catalog if t.kind == "vllm" and t.key == alias), None)
        if target is not None:
            return ServeRoute(
                "preset",
                model,
                f"{model.name} runs on vLLM with the settings in preset {alias}.",
                target,
            )
        return ServeRoute(
            "no-preset",
            model,
            f"vLLM needs a preset (a small settings file: which GPUs, how much context) "
            f"before it can run {model.name}, and none exists yet.",
        )
    if engine == RuntimeName.LLAMA_CPP:
        from llmctl.services.lanes import lane_for_model
        from llmctl.services.session_swap import pinned_port, session_target

        lane = lane_for_model(model.name, model.path, catalog)
        if lane is not None:
            return ServeRoute(
                "lane",
                model,
                f"{model.name} is served by the {lane.key} lane (its own service); "
                f"swapping to that.",
                lane,
            )
        from llmctl.config import load_settings
        from llmctl.services.lanes import llama_cpp_units, unit_for_model

        target = session_target(
            model,
            port=pinned_port(db, model),
            unit=unit_for_model(model.path, llama_cpp_units(load_settings().managed_units)),
        )
        placement = target.placement
        if placement.problem:
            return ServeRoute("cannot", model, placement.problem)
        return ServeRoute(
            "session",
            model,
            f"{model.name} runs as a llama-server that llmctl starts and tracks "
            "(`llmctl sessions` lists it, `llmctl stop ID` ends it). The swap frees "
            "the GPUs for it first.",
            target,
        )
    if engine == RuntimeName.SGLANG:
        return ServeRoute(
            "cannot",
            model,
            "llmctl cannot start SGLang models yet: SGLang runs from a start script "
            "listed in llm-router's router.yaml. Nothing was changed. Serve the same "
            f"checkpoint with vLLM instead: llmctl model add {model.source or model.name}",
        )
    if engine == RuntimeName.PYTHON_SCRIPT:
        return ServeRoute(
            "start",
            model,
            f"{model.name} is a script: llmctl runs it as a background process it "
            "tracks (`llmctl sessions` lists it, `llmctl stop ID` ends it). Nothing "
            "else is stopped.",
        )
    return ServeRoute(
        "cannot",
        model,
        f"{model.name} is served by something outside llmctl ({engine.value}); llmctl "
        "only tracks it. Nothing was changed.",
    )


def _serving_preset(db: Session, model: Model) -> str | None:
    """The alias of the preset that serves ``model``, if any."""
    from llmctl.presets import load_all_records
    from llmctl.services.preset_loader import resolve_preset_link
    from llmctl.services.preset_scaffold import STARTER_KEY, _model_id_for

    try:
        records = load_all_records()
    except Exception:
        return None
    starter = (model.metadata or {}).get(STARTER_KEY)
    if starter in records:
        return str(starter)
    models = RegistryService(db).list_models()
    model_id = _model_id_for(model)
    for alias, record in sorted(records.items()):
        linked, _state = resolve_preset_link(record.model, models)
        if (linked is not None and linked.id == model.id) or record.model.model_id in {
            model_id,
            model.source,
            model.name,
        }:
            return alias
    return None


__all__ = ["AmbiguousModel", "ServeRoute", "find_model", "route"]
