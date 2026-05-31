"""Tests for :class:`SlotConfig` / :class:`SlotsConfig` path resolution.

Mirrors the ManagedUnitConfig tests but scoped to slot-specific
env-file naming conventions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmctl.config import FleetUnitsConfig, ManagedUnitsConfig, SlotConfig, SlotsConfig


def test_explicit_env_file_path_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_SLOT_CODER_ENV_FILE", "/should/be/ignored")
    monkeypatch.setenv("AI_HOME", "/also/ignored")
    cfg = SlotConfig(
        gpu="0",
        port=8001,
        unit_name="vllm-coder",
        env_file_path=Path("/explicit/path/vllm-coder.env"),
    )
    assert cfg.resolve_env_file("coder") == Path("/explicit/path/vllm-coder.env")


def test_slot_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-slot env var lets operators relocate one slot without touching others."""
    monkeypatch.setenv("LLMCTL_SLOT_CODER_ENV_FILE", "/from/env/vllm-coder.env")
    monkeypatch.delenv("AI_HOME", raising=False)
    cfg = SlotConfig(gpu="0", port=8001, unit_name="vllm-coder")
    assert cfg.resolve_env_file("coder") == Path("/from/env/vllm-coder.env")


def test_slot_env_var_uses_uppercase_slot_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var name format: LLMCTL_SLOT_<UPPER>_ENV_FILE."""
    monkeypatch.setenv("LLMCTL_SLOT_REASONER_ENV_FILE", "/from/env/r.env")
    monkeypatch.delenv("AI_HOME", raising=False)
    cfg = SlotConfig(gpu="1", port=8002, unit_name="vllm-reasoner")
    # Lowercase slot name in code -> upper in env var lookup
    assert cfg.resolve_env_file("reasoner") == Path("/from/env/r.env")
    # Different slot name -> different env var, falls back
    assert cfg.resolve_env_file("coder") != Path("/from/env/r.env")


def test_ai_home_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMCTL_SLOT_CODER_ENV_FILE", raising=False)
    monkeypatch.setenv("AI_HOME", "/opt/ai")
    cfg = SlotConfig(gpu="0", port=8001, unit_name="vllm-coder")
    assert cfg.resolve_env_file("coder") == Path("/opt/ai/services/vllm-coder.env")


def test_home_ai_last_resort_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMCTL_SLOT_CODER_ENV_FILE", raising=False)
    monkeypatch.delenv("AI_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/test")
    cfg = SlotConfig(gpu="0", port=8001, unit_name="vllm-coder")
    assert cfg.resolve_env_file("coder") == Path("/home/test/AI/services/vllm-coder.env")


def test_slots_config_defaults_match_yannik_desktop() -> None:
    """Defaults preserve the production posture so cutover needs no config."""
    s = SlotsConfig()
    assert s.coder.gpu == "0"
    assert s.coder.port == 8001
    assert s.coder.unit_name == "vllm-coder"
    assert s.reasoner.gpu == "1"
    assert s.reasoner.port == 8002
    assert s.reasoner.unit_name == "vllm-reasoner"
    assert s.coder.enabled is True
    assert s.reasoner.enabled is True


def test_slots_config_get_returns_named_slot() -> None:
    s = SlotsConfig()
    assert s.get("coder") is s.coder
    assert s.get("reasoner") is s.reasoner
    assert s.get("nonexistent") is None


def test_managed_units_config_includes_slots_and_fleet() -> None:
    """ManagedUnitsConfig surfaces SlotsConfig and FleetUnitsConfig too."""
    mu = ManagedUnitsConfig()
    assert isinstance(mu.slots, SlotsConfig)
    assert isinstance(mu.fleet, FleetUnitsConfig)
    # Fleet defaults match the sudoers NOPASSWD scope
    assert mu.fleet.tp == "vllm-tp"
    assert mu.fleet.coder == "vllm-coder"
    assert mu.fleet.reasoner == "vllm-reasoner"
    assert mu.fleet.ollama == "ollama"
    assert mu.fleet.fleet_target == "agents.target"


def test_fleet_units_overridable_per_host() -> None:
    """A different host can re-target everything."""
    f = FleetUnitsConfig(
        tp="alt-tp",
        coder="alt-coder",
        reasoner="alt-reasoner",
        ollama="alt-ollama",
        fleet_target="alt.target",
    )
    assert f.tp == "alt-tp"
    assert f.coder == "alt-coder"
    assert f.fleet_target == "alt.target"
