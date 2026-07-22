"""Translate ``llmctl.presets.Model`` presets into llmctl launch specs.

The canonical preset schema is owned by :mod:`llmctl.presets`. This module:

* Loads presets via :func:`llmctl.presets.load_all`
* Maps each :class:`~llmctl.presets.Model` to a
  :class:`~llmctl.integrations.vllm_env.VLLMLaunchSpec` by folding in
  the cross-preset defaults from ``settings.vllm.defaults``
* Surfaces both the spec (for rendering) and a metadata view (for the
  ``llmctl list`` / ``llmctl preset`` CLIs)

Per-preset overrides win over defaults; defaults fill in everything the
preset doesn't pin. ``reasoning_parser`` is folded into ``extra_args``
the same way the legacy gpu-models preset adapter did, since the vLLM
launcher script doesn't model it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llmctl.config import VLLMDefaultsConfig
from llmctl.integrations.vllm_env import VLLMLaunchSpec
from llmctl.presets import Model, load_all, load_all_records
from llmctl.schemas import Model as RegistryModel

#: Resolution state of a preset against the Model registry.
#:
#: - ``"explicit"`` — preset.model_ref was set and matched a Model.id
#: - ``"auto"``     — model_ref unset, fuzzy-matched on model_id/served_name
#: - ``"missing"``  — model_ref set but no Model with that id exists
#: - ``"unlinked"`` — no model_ref and no auto-match found
LinkageState = Literal["explicit", "auto", "missing", "unlinked"]


@dataclass(frozen=True)
class PresetView:
    """Lightweight summary of a loaded preset for CLI/TUI display."""

    alias: str
    served_name: str
    model_id: str
    family: str | None
    param_count_b: float | None
    tensor_parallel: int
    quantization: str
    source_path: Path | None
    #: Populated when the loader is given a registry snapshot;
    #: None when no resolution was attempted (e.g. CLI-only callers).
    linked_model_id: str | None = None
    linked_model_name: str | None = None
    linkage_state: LinkageState | None = None


def resolve_preset_link(
    preset: Model,
    models: list[RegistryModel],
) -> tuple[RegistryModel | None, LinkageState]:
    """Resolve a preset against the Model registry.

    Order matters: an explicit ``model_ref`` always wins (even when it
    points at nothing, surfaced as ``"missing"``), so a user who pinned
    a registry id can see it broke rather than silently falling back to
    fuzzy matching against a different row.
    """
    if preset.model_ref:
        match = next((m for m in models if m.id == preset.model_ref), None)
        return (match, "explicit" if match else "missing")

    match = next((m for m in models if m.source == preset.model_id), None)
    if match is None:
        match = next(
            (m for m in models if m.source == preset.served_name), None
        )
    if match is None:
        match = next((m for m in models if m.name == preset.served_name), None)
    return (match, "auto" if match else "unlinked")


def model_to_launch_spec(
    model: Model,
    defaults: VLLMDefaultsConfig | None = None,
    *,
    port_override: int | None = None,
) -> VLLMLaunchSpec:
    """Project a :class:`Model` preset onto a :class:`VLLMLaunchSpec`.

    Field merge rules:

    * ``model``, ``served_name``, ``quantization`` (via
      ``vllm_quantization_flag``), ``max_model_len``, ``max_num_seqs``,
      ``gpu_memory_utilization``, ``host``, ``tool_parser`` — always
      come from the preset; the launch posture demands they be
      explicit.
    * ``tensor_parallel`` — preset's ``tensor_parallel_size`` wins.
    * ``gpus`` — preset has no ``gpus`` field; ``defaults.gpus`` wins.
    * ``port`` — ``port_override`` (e.g. the managed-unit's
      ``default_port``) wins, then ``defaults.port``; the preset's
      own ``port`` is treated as a stale template default and ignored.
    * ``kv_cache_type`` — preset's ``kv_cache_dtype`` wins when not
      ``"auto"``; otherwise omitted.
    * ``prefix_cache``, ``chunked_prefill``, ``nccl_p2p_disable`` —
      defaults supply these (the canonical ``Model`` doesn't carry them).
    * ``max_batched_tokens`` — preset's ``max_num_batched_tokens`` wins
      when set; otherwise ``defaults.max_batched_tokens``. Hybrid-Mamba
      models (qwen3_5_moe) must raise it above the chunked-prefill
      default.
    * ``reasoning_parser`` — folded into ``extra_args`` as
      ``--reasoning-parser <name>``, matching gpu-models behaviour.
    * ``python_root``, ``spec_config`` — passed straight through; both
      are preset-only concerns the defaults have no opinion on.
    * ``tq`` — ignored here; TurboQuant override is wired in Phase 4
      via a separate CLI flag and resolves to ``kv_cache_type``.
    """
    defaults = defaults or VLLMDefaultsConfig()

    kv_cache_type: str | None = None
    if model.kv_cache_dtype and model.kv_cache_dtype != "auto":
        kv_cache_type = model.kv_cache_dtype

    extra_args: str | None = None
    if model.reasoning_parser:
        extra_args = f"--reasoning-parser {model.reasoning_parser}"

    port = port_override if port_override is not None else defaults.port

    return VLLMLaunchSpec(
        model=model.model_id,
        served_name=model.served_name,
        tensor_parallel=model.tensor_parallel_size,
        gpus=defaults.gpus,
        port=port,
        host=model.host or defaults.host,
        max_model_len=model.max_model_len,
        gpu_memory_utilization=(
            model.gpu_memory_utilization
            if model.gpu_memory_utilization is not None
            else defaults.gpu_memory_utilization
        ),
        quantization=model.vllm_quantization_flag,
        kv_cache_type=kv_cache_type,
        tool_parser=model.tool_parser,
        python_root=model.python_root,
        max_num_seqs=(
            model.max_num_seqs
            if model.max_num_seqs is not None
            else defaults.max_num_seqs
        ),
        max_batched_tokens=(
            model.max_num_batched_tokens
            if model.max_num_batched_tokens is not None
            else defaults.max_batched_tokens
        ),
        prefix_cache=defaults.prefix_cache,
        chunked_prefill=defaults.chunked_prefill,
        spec_config=model.spec_config,
        extra_args=extra_args,
        nccl_p2p_disable=defaults.nccl_p2p_disable,
    )


def preset_count_by_model(
    *,
    config_dir: Path | None = None,
    models: list[RegistryModel],
) -> dict[str, int]:
    """Return ``{Model.id: count}`` of presets linking to each registry row.

    Counts both explicit ``model_ref`` matches and auto-matches —
    anything that resolves to a concrete Model. Models with no
    referring presets are not present in the result.
    """
    records = load_all_records(config_dir=config_dir)
    counts: dict[str, int] = {}
    for record in records.values():
        match, _ = resolve_preset_link(record.model, models)
        if match and match.id:
            counts[match.id] = counts.get(match.id, 0) + 1
    return counts


def load_presets(
    *,
    defaults: VLLMDefaultsConfig | None = None,
    config_dir: Path | None = None,
) -> dict[str, VLLMLaunchSpec]:
    """Load every preset on disk and return ``{alias: VLLMLaunchSpec}``.

    Args:
        defaults: Cross-preset defaults; uses :class:`VLLMDefaultsConfig`
            built-ins when omitted (matches the production posture).
        config_dir: Optional preset directory override for tests and
            one-off callers. ``None`` uses the standard llmctl path
            resolution.
    """
    models = load_all(config_dir=config_dir)
    defaults = defaults or VLLMDefaultsConfig()
    return {alias: model_to_launch_spec(model, defaults) for alias, model in models.items()}


def load_preset_views(
    *,
    config_dir: Path | None = None,
    models: list[RegistryModel] | None = None,
) -> list[PresetView]:
    """Return a metadata view of every loaded preset for CLI/TUI listing.

    When ``models`` is provided, every view is enriched with linkage
    info via :func:`resolve_preset_link`. CLI callers that don't need
    the registry can pass ``None`` and the linkage fields stay ``None``.
    """
    records = load_all_records(config_dir=config_dir)
    views: list[PresetView] = []
    for alias, record in sorted(records.items()):
        linked_id: str | None = None
        linked_name: str | None = None
        state: LinkageState | None = None
        if models is not None:
            match, state = resolve_preset_link(record.model, models)
            linked_id = match.id if match else None
            linked_name = match.name if match else None
        views.append(
            PresetView(
                alias=alias,
                served_name=record.model.served_name,
                model_id=record.model.model_id,
                family=record.model.family,
                param_count_b=record.model.param_count_b,
                tensor_parallel=record.model.tensor_parallel_size,
                quantization=record.model.quantization,
                source_path=record.source_path,
                linked_model_id=linked_id,
                linked_model_name=linked_name,
                linkage_state=state,
            )
        )
    return views
