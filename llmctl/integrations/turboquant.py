"""TurboQuant KV-cache override resolver.

vLLM's ``--kv-cache-dtype turboquant_*`` family compresses KV cache
into INT8/INT4 codes via Hadamard rotation, recovering ~30-50% of the
KV memory budget at a small quality cost. The flag is preset-driven
(via ``Model.kv_cache_dtype``) but CLI users want a one-flag override
to flip TQ on/off without editing the preset YAML.

Resolution rules (matching gpu-models's ``_resolve_kv_dtype``):

* ``override=True`` -> use ``preset_tq_kind`` if set, else
  ``default_tq_kind`` (default ``turboquant_k8v4``).
* ``override=False`` -> return ``None`` (omit VLLM_KV_DTYPE entirely,
  let vLLM pick the model default).
* ``override=None`` -> use the preset's own ``kv_cache_type`` (i.e.
  whatever is currently in the launch spec); the function returns
  ``None`` so the caller knows not to mutate the spec.
"""

from __future__ import annotations

from typing import Literal

DEFAULT_TQ_KIND = "turboquant_k8v4"


def resolve_kv_cache_type(
    *,
    override: bool | None,
    preset_tq_kind: str | None = None,
    default_tq_kind: str = DEFAULT_TQ_KIND,
) -> str | None | Literal["__unset__"]:
    """Compute the KV cache override the launch spec should carry.

    Args:
        override: ``True`` to force TQ on, ``False`` to force TQ off,
            ``None`` to respect the preset.
        preset_tq_kind: Per-preset override (``Model.tq_kv_cache_type``
            equivalent). Lets a preset request a specific TQ variant
            (e.g. ``turboquant_k4v4``) when the user requests TQ on.
        default_tq_kind: Workspace fallback for ``override=True`` with
            no per-preset kind set. Locked to ``turboquant_k8v4``
            unless overridden in ``settings.yaml``.

    Returns:
        * A string starting with ``turboquant_`` when the override
          forces TQ on.
        * ``None`` when the override forces TQ off (caller should clear
          the spec's ``kv_cache_type``).
        * The string ``"__unset__"`` (the literal :class:`str`) when no
          override was requested (``override=None``) — the caller should
          leave the spec's existing value alone.
    """
    if override is True:
        chosen = preset_tq_kind or default_tq_kind
        # Ignore non-TQ values smuggled in via preset_tq_kind — the
        # user asked for TQ on, give them a TQ variant.
        return chosen if chosen.startswith("turboquant_") else default_tq_kind
    if override is False:
        return None
    return "__unset__"


def apply_to_spec_dict(
    spec_dict: dict,
    *,
    override: bool | None,
    preset_tq_kind: str | None = None,
    default_tq_kind: str = DEFAULT_TQ_KIND,
) -> dict:
    """Return a copy of ``spec_dict`` with ``kv_cache_type`` patched.

    Convenience wrapper for callers building a launch spec from a
    preset plus CLI flags. When ``override is None`` the input is
    returned unchanged.
    """
    resolved = resolve_kv_cache_type(
        override=override,
        preset_tq_kind=preset_tq_kind,
        default_tq_kind=default_tq_kind,
    )
    if resolved == "__unset__":
        return spec_dict
    out = dict(spec_dict)
    out["kv_cache_type"] = resolved  # None or "turboquant_*"
    return out
