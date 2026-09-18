"""Drift guard between llmctl's preset schema and llm-models-config's.

Since 2026-09-18 llm-models-config validates every preset with this package's
``Model`` and derives ``LLMCTL_OWNED_KEYS`` from it (FIXPLAN §1.3, long-term
half), so the first test holds by construction. It stays as a tripwire in
case someone reintroduces a hand-kept list.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from llmctl.presets.schema import Model as LlmctlModel

lmc_schema = pytest.importorskip("llm_models_config.schema")


def test_every_llmctl_preset_key_is_known_to_llm_models_config() -> None:
    lmc_keys = {f.name for f in fields(lmc_schema.Model)} | lmc_schema.LLMCTL_OWNED_KEYS
    missing = set(LlmctlModel.model_fields) - lmc_keys
    assert not missing, (
        f"llmctl preset keys unknown to llm-models-config: {sorted(missing)}; "
        "llm_models_config.schema must derive LLMCTL_OWNED_KEYS from llmctl"
    )


def test_owned_keys_are_not_also_lmc_fields() -> None:
    overlap = {f.name for f in fields(lmc_schema.Model)} & lmc_schema.LLMCTL_OWNED_KEYS
    assert not overlap
