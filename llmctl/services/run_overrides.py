"""Per-run launch settings for ``llmctl run`` (``--tp``, ``--max-len``,
``--gpu-mem``) and ``--save``, which writes them back.

A model's settings live in one of two places, picked by its engine: a vLLM
preset (a YAML file) or a profile (a database row) for the engines llmctl
starts as sessions. ``run`` takes the same three flags for every engine and
this module translates them, so the operator never has to know which store
applies. ``--save`` edits only the lines it changes in a preset, so the
comments in a hand-written preset survive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llmctl._fsutil import atomic_write_text
from llmctl.db import RuntimeName

#: The settings ``run`` can change, as preset/profile field names.
FIELDS = ("tensor_parallel_size", "max_model_len", "gpu_memory_utilization")
FLAGS = {
    "tensor_parallel_size": "--tp",
    "max_model_len": "--max-len",
    "gpu_memory_utilization": "--gpu-mem",
}
_SINGLE_GPU = {RuntimeName.FREETOKEN, RuntimeName.STRATA}


class OverrideError(ValueError):
    """A flag that does not apply to this model's engine, or a save that cannot happen."""


@dataclass(frozen=True)
class RunOverrides:
    """What the operator asked to change for this run."""

    tp: int | None = None
    max_len: int | None = None
    gpu_mem: float | None = None

    def fields(self) -> dict[str, Any]:
        """The set flags as ``{field: value}``, in :data:`FIELDS` order."""
        values = (self.tp, self.max_len, self.gpu_mem)
        return {k: v for k, v in zip(FIELDS, values, strict=True) if v is not None}

    def __bool__(self) -> bool:
        return bool(self.fields())

    def describe(self) -> str:
        """``--tp 2 --max-len 32768`` for messages."""
        return " ".join(f"{FLAGS[k]} {v}" for k, v in self.fields().items())


def session_parameters(runtime: RuntimeName, overrides: RunOverrides) -> dict[str, Any]:
    """Launch parameters for an engine llmctl starts as a session.

    llama.cpp names the context ``ctx_size`` and has no memory-fraction
    setting; FreeToken and Strata serve from one GPU.
    """
    fields = overrides.fields()
    if runtime in _SINGLE_GPU and fields.get("tensor_parallel_size", 1) > 1:
        raise OverrideError(f"{runtime.value} serves from one GPU, so --tp must be 1.")
    if runtime == RuntimeName.LLAMA_CPP:
        if "gpu_memory_utilization" in fields:
            raise OverrideError(
                "llama.cpp has no memory-fraction setting. --gpu-mem applies to "
                "vLLM, SGLang, FreeToken and Strata."
            )
        if "max_model_len" in fields:
            fields["ctx_size"] = fields.pop("max_model_len")
    return fields


# -- saving to a preset -----------------------------------------------------------------


def save_to_preset(path: Path, fields: dict[str, Any]) -> list[str]:
    """Write ``fields`` into the preset file at ``path``; return what changed.

    Only the matching top-level lines are rewritten (a trailing ``# comment``
    is kept); a key the file lacks is appended. The result is validated as a
    preset before anything is written, so a bad value leaves the file as it was.
    """
    from llmctl.presets.schema import Model

    text = path.read_text()
    old = yaml.safe_load(text) or {}
    lines = text.splitlines(keepends=True)
    changes: list[str] = []
    for key, value in fields.items():
        pattern = re.compile(
            rf"^{re.escape(key)}:[ \t]*[^#\r\n]*?(?P<comment>[ \t]+#.*)?(?P<eol>\r?\n)?$"
        )
        for index, line in enumerate(lines):
            match = pattern.match(line)
            if match:
                lines[index] = f"{key}: {value}{match['comment'] or ''}{match['eol'] or ''}"
                break
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"{key}: {value}\n")
        if old.get(key) != value:
            changes.append(f"{key} {old.get(key, 'unset')} -> {value}")
    new_text = "".join(lines)
    new = yaml.safe_load(new_text)
    for key, value in fields.items():
        if new.get(key) != value:  # the line edit missed (an unusual layout)
            raise OverrideError(
                f"Could not rewrite {key} in {path} safely; edit it by hand "
                f"(`llmctl config {old.get('alias', path.stem)} --edit`)."
            )
    try:
        Model.model_validate(new)
    except ValueError as exc:
        raise OverrideError(f"Not saved: the preset would be invalid: {exc}") from exc
    if new_text != text:
        atomic_write_text(path, new_text)
    return changes


# -- saving to a profile ----------------------------------------------------------------


def save_to_profile(db: Any, model: Any, parameters: dict[str, Any]) -> tuple[str, list[str]]:
    """Store ``parameters`` in ``model``'s profile; return ``(profile name, changes)``.

    A model without a profile gets one named after it and linked as its
    default. A profile other models also use is refused: saving there would
    change those models too.
    """
    from sqlmodel import select

    from llmctl.db import ModelRecord, ProfileRecord
    from llmctl.schemas import ProfileCreate, ProfileUpdate
    from llmctl.services.profiles import ProfileService, effective_parameters

    service = ProfileService(db)
    promoted = {k: v for k, v in parameters.items() if k in FIELDS}
    profile = db.get(ProfileRecord, model.default_profile_id) if model.default_profile_id else None
    if profile is None:
        name = model.name
        taken = {p.name for p in db.exec(select(ProfileRecord)).all()}
        suffix = 2
        while name in taken:
            name, suffix = f"{model.name}-{suffix}", suffix + 1
        created = service.create_profile(
            ProfileCreate(
                name=name,
                runtime=model.runtime,
                description=f"Settings for {model.name}, saved by `llmctl run --save`.",
                parameters=dict(parameters),
                **promoted,
            )
        )
        model.default_profile_id = created.id
        db.add(model)
        db.commit()
        return created.name, [f"{k} unset -> {v}" for k, v in parameters.items()]

    users = db.exec(select(ModelRecord).where(ModelRecord.default_profile_id == profile.id)).all()
    if len(users) > 1:
        raise OverrideError(
            f"Not saved: profile {profile.name} is shared by {len(users)} models, "
            "so saving would change all of them."
        )
    before = effective_parameters(profile)
    merged = dict(profile.parameters or {})
    merged.update(parameters)
    service.update_profile(profile.id, ProfileUpdate(parameters=merged, **promoted))
    changes = [
        f"{k} {before.get(k, 'unset')} -> {v}" for k, v in parameters.items() if before.get(k) != v
    ]
    return profile.name, changes


# -- where a model's settings live ------------------------------------------------------


@dataclass(frozen=True)
class SettingsSource:
    """Where ``run NAME`` reads its settings from, for ``llmctl config``."""

    kind: str  # "preset" | "profile" | "none"
    label: str
    path: Path | None
    values: dict[str, Any]


def preset_source(alias: str, config_dir: Path | None = None) -> SettingsSource | None:
    """The preset named ``alias``, or None."""
    from llmctl.presets import load_all_records

    record = load_all_records(config_dir=config_dir).get(alias)
    if record is None:
        return None
    model = record.model
    return SettingsSource(
        kind="preset",
        label=f"vLLM preset {alias}",
        path=record.source_path,
        values={k: getattr(model, k) for k in FIELDS},
    )


def profile_source(db: Any, model: Any) -> SettingsSource:
    """The profile ``model`` launches with, or a ``none`` source."""
    from llmctl.db import ProfileRecord
    from llmctl.services.profiles import effective_parameters

    profile = db.get(ProfileRecord, model.default_profile_id) if model.default_profile_id else None
    if profile is None:
        return SettingsSource(
            kind="none", label="engine defaults (no saved settings)", path=None, values={}
        )
    params = effective_parameters(profile)
    if "ctx_size" in params and "max_model_len" not in params:
        params["max_model_len"] = params["ctx_size"]
    return SettingsSource(
        kind="profile",
        label=f"profile {profile.name} (in llmctl's database)",
        path=None,
        values={k: params[k] for k in FIELDS if k in params},
    )
