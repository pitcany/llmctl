"""Drift guard between llmctl's preset schema and llm-models-config's.

Both packages read and write ``~/.config/llm-models/*.yaml``. A key llmctl
adds that llm-models-config neither models nor lists in
``LLMCTL_OWNED_KEYS`` would be rejected there (and before 2026-09-18 was
silently dropped on the next save). FIXPLAN 2026-09-18 §1.3.
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
        "add them to llm_models_config.schema.LLMCTL_OWNED_KEYS"
    )


def test_owned_keys_are_not_also_lmc_fields() -> None:
    overlap = {f.name for f in fields(lmc_schema.Model)} & lmc_schema.LLMCTL_OWNED_KEYS
    assert not overlap
