"""Starter vLLM preset for a checkpoint, so an added model is swappable.

``llmctl run`` serves presets, so a vLLM model that has only a registry row
cannot be served until someone writes one. :func:`ensure_starter_preset`
writes a conservative one from what the checkpoint's ``config.json`` proves
and ``settings.yaml``'s ``vllm.defaults``:

* ``model_id`` = the HF repo id (vLLM resolves it through ``$HF_HOME``) or the
  local folder.
* ``max_model_len`` = the native context, capped at ``vllm.defaults``.
* ``tensor_parallel_size`` = ``vllm.defaults``, dropped to 1 when the attention
  heads do not divide by it (vLLM refuses to start otherwise).
* The quantization flag is left empty so vLLM reads ``quantization_config``
  itself; a wrong ``--quantization`` is a failed start, an empty one is not.
* Tool and reasoning parsers are left unset: guessing one from the family is
  how a model ends up emitting tool calls nobody parses.

The alias is recorded on the registry row (``metadata.starter_preset``), which
is how ``remove --purge`` knows it may delete that preset with the weights.
Hand-written presets are never touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlmodel import Session

from llmctl.config import VLLMDefaultsConfig, load_settings
from llmctl.db import ModelRecord, RuntimeName
from llmctl.presets import Model as PresetModel
from llmctl.presets import PresetSchemaError, load_all_records, save_preset
from llmctl.schemas import Model

#: Registry metadata key naming the preset ``add`` wrote for a row.
STARTER_KEY = "starter_preset"

_QUANT_LABELS = {
    "fp8": "fp8",
    "awq": "awq",
    "gptq": "gptq",
    "compressed-tensors": "compressed-tensors",
    "compressed_tensors": "compressed-tensors",
    "bitsandbytes": "bnb",
}


@dataclass(frozen=True)
class StarterPreset:
    """Outcome of :func:`ensure_starter_preset`."""

    alias: str
    created: bool
    path: Path | None
    notes: tuple[str, ...] = ()


def alias_for(name: str) -> str:
    """``Qwen/Qwen3-8B`` -> ``qwen3-8b``: the repo's own name, lower-cased."""
    base = name.rstrip("/").rsplit("/", 1)[-1].lower()
    slug = re.sub(r"[^a-z0-9.]+", "-", base).strip("-.")
    return slug or "model"


def _config_for(model: Model) -> dict[str, Any] | None:
    for raw in (model.path, model.source):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_dir() and (path / "config.json").is_file():
            try:
                data = json.loads((path / "config.json").read_text())
            except (OSError, ValueError):
                return None
            return data if isinstance(data, dict) else None
    return None


def _model_id_for(model: Model) -> str:
    """The value vLLM gets as ``--model``: repo id when known, else the path."""
    source = model.source or ""
    if "/" in source and not source.startswith(("/", "~", ".")):
        return source
    return str(Path(model.path or source).expanduser())


def build_starter_preset(
    model: Model, alias: str, defaults: VLLMDefaultsConfig
) -> tuple[PresetModel, tuple[str, ...]]:
    """The preset for ``model`` plus notes on what was defaulted or guessed."""
    cfg = _config_for(model) or {}
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    notes: list[str] = []

    native = text.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    max_len = defaults.max_model_len
    if isinstance(native, int) and native > 0:
        max_len = min(native, defaults.max_model_len)
        if native > max_len:
            notes.append(
                f"max_model_len {max_len} (native {native}); raise it in the preset if "
                "the KV cache fits."
            )
    else:
        notes.append(f"config.json has no context length; using {max_len}.")

    tp = defaults.tensor_parallel
    heads = text.get("num_attention_heads")
    if isinstance(heads, int) and tp > 1 and heads % tp:
        notes.append(f"{heads} attention heads do not split over {tp} GPUs; using TP=1.")
        tp = 1

    qc = cfg.get("quantization_config")
    method = str((qc or {}).get("quant_method") or "").lower() if isinstance(qc, dict) else ""
    label = _QUANT_LABELS.get(method, "none")
    if method and method not in _QUANT_LABELS:
        notes.append(f"quantization {method!r} has no preset label; vLLM still reads it.")

    model_type = str(text.get("model_type") or cfg.get("model_type") or "")
    family_match = re.match(r"[a-z]+", model_type.lower())
    architectures = tuple(str(a) for a in cfg.get("architectures") or () if a)

    preset = PresetModel(
        alias=alias,
        served_name=alias,
        model_id=_model_id_for(model),
        quantization=label,
        vllm_quantization_flag="",
        tensor_parallel_size=tp,
        max_model_len=max_len,
        family=family_match.group(0) if family_match else None,
        architectures=architectures,
        model_ref=model.id,
    )
    notes.append(
        "No tool or reasoning parser is set; add tool_parser/reasoning_parser "
        "before using it for tool calls."
    )
    return preset, tuple(notes)


def ensure_starter_preset(
    db: Session,
    model: Model,
    *,
    defaults: VLLMDefaultsConfig | None = None,
    config_dir: Path | None = None,
) -> StarterPreset:
    """Return the preset that serves ``model``, writing a starter one if none.

    An existing preset naming the same weights wins; nothing is written then.
    A taken alias gets a ``-2``/``-3`` suffix rather than overwriting a
    hand-written preset.
    """
    if model.runtime != RuntimeName.VLLM:
        raise ValueError("starter presets are for vLLM models")
    records = load_all_records(**({"config_dir": config_dir} if config_dir else {}))
    model_id = _model_id_for(model)
    for alias, record in sorted(records.items()):
        if record.model.model_id == model_id or (
            model.id and record.model.model_ref == model.id
        ):
            return StarterPreset(alias=alias, created=False, path=record.source_path)

    base = alias_for(model.source or model.name)
    alias, n = base, 1
    while alias in records:
        n += 1
        alias = f"{base}-{n}"
    if defaults is None:
        defaults = load_settings().vllm.defaults
    try:
        preset, notes = build_starter_preset(model, alias, defaults)
    except PresetSchemaError as exc:
        raise ValueError(f"could not build a preset for {model.name}: {exc}") from exc
    path = save_preset(preset, config_dir=config_dir)

    if model.id:
        record = db.get(ModelRecord, model.id)
        if record is not None:
            record.metadata_json = {**record.metadata_json, STARTER_KEY: alias}
            db.add(record)
            db.commit()
    return StarterPreset(alias=alias, created=True, path=path, notes=notes)


__all__ = [
    "STARTER_KEY",
    "StarterPreset",
    "alias_for",
    "build_starter_preset",
    "ensure_starter_preset",
]
