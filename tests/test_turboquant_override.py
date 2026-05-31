"""Tests for :mod:`llmctl.integrations.turboquant`.

The TQ override is a tri-state CLI flag (on / off / preset-default).
These tests pin the resolution rules so future CLI work can lean on
them.
"""

from __future__ import annotations

import pytest

from llmctl.integrations.turboquant import (
    DEFAULT_TQ_KIND,
    apply_to_spec_dict,
    resolve_kv_cache_type,
)


def test_override_true_returns_default_tq_kind() -> None:
    """No per-preset kind -> default turboquant_k8v4."""
    assert resolve_kv_cache_type(override=True) == DEFAULT_TQ_KIND
    assert DEFAULT_TQ_KIND.startswith("turboquant_")


def test_override_true_respects_preset_tq_kind() -> None:
    """Per-preset kind wins when it's a valid TQ variant."""
    assert (
        resolve_kv_cache_type(override=True, preset_tq_kind="turboquant_k4v4")
        == "turboquant_k4v4"
    )


def test_override_true_ignores_non_tq_preset_kind() -> None:
    """A preset_tq_kind that isn't a TQ variant falls back to the default.

    Protects against a config bug where someone wrote ``tq_kv_cache_type:
    fp8`` intending to mean "use fp8 when --tq" — that's an invalid
    request (the --tq flag means TQ), so we coerce to the safe default."""
    assert resolve_kv_cache_type(override=True, preset_tq_kind="fp8") == DEFAULT_TQ_KIND


def test_override_false_returns_none() -> None:
    """--no-tq -> None (caller should clear the spec's kv_cache_type)."""
    assert resolve_kv_cache_type(override=False) is None
    assert resolve_kv_cache_type(override=False, preset_tq_kind="turboquant_k4v4") is None


def test_no_override_returns_sentinel() -> None:
    """override=None -> ``"__unset__"`` sentinel so callers leave the spec alone."""
    assert resolve_kv_cache_type(override=None) == "__unset__"


def test_custom_default_tq_kind_supported() -> None:
    """Workspace can set a different default via settings.yaml."""
    assert (
        resolve_kv_cache_type(override=True, default_tq_kind="turboquant_k4v4")
        == "turboquant_k4v4"
    )


def test_apply_to_spec_dict_with_override_on_sets_kv_cache_type() -> None:
    spec = {"model": "m", "served_name": "s", "kv_cache_type": "fp8"}
    out = apply_to_spec_dict(spec, override=True)
    assert out["kv_cache_type"] == DEFAULT_TQ_KIND
    # input not mutated
    assert spec["kv_cache_type"] == "fp8"


def test_apply_to_spec_dict_with_override_off_clears_kv_cache_type() -> None:
    spec = {"model": "m", "served_name": "s", "kv_cache_type": "fp8"}
    out = apply_to_spec_dict(spec, override=False)
    assert out["kv_cache_type"] is None  # explicitly cleared


def test_apply_to_spec_dict_with_no_override_is_noop() -> None:
    spec = {"model": "m", "served_name": "s", "kv_cache_type": "fp8"}
    out = apply_to_spec_dict(spec, override=None)
    # When override=None we return the same dict (no copy needed because
    # nothing changed) — but it must compare equal.
    assert out == spec


@pytest.mark.parametrize("override,expected_value", [
    (True, DEFAULT_TQ_KIND),
    (False, None),
])
def test_apply_to_spec_dict_overrides_preset_value(
    override: bool, expected_value: str | None
) -> None:
    """Even a preset that sets kv_cache_type gets overridden by the flag."""
    spec = {"model": "m", "served_name": "s", "kv_cache_type": "fp8"}
    out = apply_to_spec_dict(spec, override=override)
    assert out["kv_cache_type"] == expected_value


def test_apply_to_spec_dict_no_override_keeps_missing_field_missing() -> None:
    """If the spec has no kv_cache_type and no override, output stays clean."""
    spec = {"model": "m", "served_name": "s"}
    out = apply_to_spec_dict(spec, override=None)
    assert "kv_cache_type" not in out
