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
start      llama.cpp / python_script row: start it as a session
on-demand  Ollama / LM Studio: nothing to start, the first request loads it
cannot     SGLang, FreeToken, adopted endpoints, other machines: says why
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
    #: ``preset`` only: the swap-list row to swap to.
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
    """Decide how ``model`` is served. Reads presets; writes nothing."""
    host = effective_host(model.host)
    if host != local_host():
        return ServeRoute(
            "cannot", model, f"{model.name} lives on {host}; serve it there."
        )
    engine = model.runtime
    if engine in (RuntimeName.OLLAMA, RuntimeName.LMSTUDIO):
        label = "Ollama" if engine == RuntimeName.OLLAMA else "LM Studio"
        return ServeRoute(
            "on-demand",
            model,
            f"{label} loads {model.name} on its first request; there is nothing to "
            "start and no GPU is freed.",
        )
    if engine == RuntimeName.VLLM:
        alias = _serving_preset(db, model)
        target = next((t for t in catalog if t.kind == "vllm" and t.key == alias), None)
        if target is not None:
            return ServeRoute(
                "preset", model, f"{model.name} is served by preset {alias}.", target
            )
        return ServeRoute(
            "no-preset",
            model,
            f"No preset serves {model.name} yet; `llmctl swap` needs one.",
        )
    if engine in (RuntimeName.LLAMA_CPP, RuntimeName.PYTHON_SCRIPT):
        return ServeRoute(
            "start",
            model,
            f"{model.name} starts as a {engine.value} session (llmctl start "
            f"{(model.id or '')[:8]}), on a port and GPU the scheduler picks.",
        )
    if engine == RuntimeName.SGLANG:
        return ServeRoute(
            "cannot",
            model,
            "SGLang models run as a router.yaml lane; add a lane for it or serve the "
            "same checkpoint with vLLM.",
        )
    return ServeRoute(
        "cannot", model, f"{engine.value} endpoints are adopted; llmctl does not start them."
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
