"""Unit tests for :class:`ManagedUnitConfig` path resolution.

The adapter is intentionally configurable so llmctl can run on hosts
with no ``~/AI/`` layout. These tests pin the resolution precedence so
future refactors don't accidentally change defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmctl.adapters.vllm_systemd import VLLMSystemdAdapter
from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig, Settings


def test_explicit_env_file_path_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_VLLM_ENV_FILE", "/should/be/ignored")
    monkeypatch.setenv("AI_HOME", "/should/also/be/ignored")
    cfg = ManagedUnitConfig(
        unit_name="vllm-tp",
        env_file_path=Path("/explicit/path/vllm.env"),
    )
    assert cfg.resolve_env_file() == Path("/explicit/path/vllm.env")


def test_env_var_override_wins_over_ai_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_VLLM_ENV_FILE", "/from/env/vllm.env")
    monkeypatch.setenv("AI_HOME", "/from/ai/home")
    cfg = ManagedUnitConfig(unit_name="vllm-tp", env_file_path=None)
    assert cfg.resolve_env_file() == Path("/from/env/vllm.env")


def test_ai_home_used_when_no_explicit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMCTL_VLLM_ENV_FILE", raising=False)
    monkeypatch.setenv("AI_HOME", "/opt/ai")
    cfg = ManagedUnitConfig(unit_name="vllm-tp", env_file_path=None)
    assert cfg.resolve_env_file() == Path("/opt/ai/services/vllm-tp.env")


def test_home_ai_fallback_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last-resort default matches the yannik-desktop layout so the
    initial cutover doesn't require any config changes on the existing host."""
    monkeypatch.delenv("LLMCTL_VLLM_ENV_FILE", raising=False)
    monkeypatch.delenv("AI_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/test")
    cfg = ManagedUnitConfig(unit_name="vllm-tp", env_file_path=None)
    assert cfg.resolve_env_file() == Path("/home/test/AI/services/vllm-tp.env")


def test_unit_name_propagates_to_env_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    """Different unit -> different env filename. Lets slot units share the
    same resolution machinery."""
    monkeypatch.delenv("LLMCTL_VLLM_ENV_FILE", raising=False)
    monkeypatch.setenv("AI_HOME", "/opt/ai")
    cfg = ManagedUnitConfig(unit_name="vllm-coder", env_file_path=None)
    assert cfg.resolve_env_file() == Path("/opt/ai/services/vllm-coder.env")


def test_settings_managed_units_defaults_carry_through() -> None:
    """A fresh Settings() should expose the managed unit config
    so callers don't need to special-case missing keys."""
    s = Settings()
    assert isinstance(s.managed_units, ManagedUnitsConfig)
    assert s.managed_units.vllm_tp.unit_name == "vllm-tp"


def test_adapter_consumes_config_unit_name() -> None:
    """The adapter takes its unit name from the injected config."""
    cfg = ManagedUnitConfig(unit_name="custom-vllm")
    adapter = VLLMSystemdAdapter(cfg)
    assert adapter.unit_name == "custom-vllm"


def test_adapter_env_file_path_override_wins_over_config(tmp_path: Path) -> None:
    """Explicit env_file_path arg short-circuits config resolution."""
    cfg = ManagedUnitConfig(unit_name="vllm-tp", env_file_path=Path("/config/path"))
    adapter = VLLMSystemdAdapter(cfg, env_file_path=tmp_path / "ad-hoc")
    assert adapter.env_file_path == tmp_path / "ad-hoc"


def test_launcher_marker_none_disables_legacy_guard(tmp_path: Path) -> None:
    """Setting launcher_marker=None bypasses the ExecStart check."""
    from dataclasses import dataclass

    from llmctl.integrations.systemctl import SystemctlRunner

    @dataclass
    class _FakeCompleted:
        returncode: int = 0
        stdout: str = ""
        stderr: str = ""

    def fake(argv: list[str]) -> _FakeCompleted:
        if argv[-2] == "cat":
            # body that would fail the default marker check
            return _FakeCompleted(stdout="ExecStart=/usr/bin/something-else\n")
        return _FakeCompleted()

    runner = SystemctlRunner(runner=fake)
    runner.available = lambda: True  # type: ignore[method-assign]

    cfg = ManagedUnitConfig(unit_name="vllm-tp", launcher_marker=None)
    adapter = VLLMSystemdAdapter(
        cfg,
        env_file_path=tmp_path / "env",
        systemctl=runner,
    )
    adapter.ensure_launcher_unit()  # no raise


def test_launcher_marker_custom_string_enforced(tmp_path: Path) -> None:
    """Custom launcher_marker is respected for both pass and fail."""
    from dataclasses import dataclass

    from llmctl.adapters.vllm_systemd import LegacyUnitError
    from llmctl.integrations.systemctl import SystemctlRunner

    @dataclass
    class _FakeCompleted:
        returncode: int = 0
        stdout: str = ""
        stderr: str = ""

    def fake_with(body: str):
        def fake(argv: list[str]) -> _FakeCompleted:
            if argv[-2] == "cat":
                return _FakeCompleted(stdout=body)
            return _FakeCompleted()

        runner = SystemctlRunner(runner=fake)
        runner.available = lambda: True  # type: ignore[method-assign]
        return runner

    cfg = ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="my-custom-launcher")

    # pass: body contains custom marker
    adapter_pass = VLLMSystemdAdapter(
        cfg,
        env_file_path=tmp_path / "env",
        systemctl=fake_with("ExecStart=/opt/my-custom-launcher --arg\n"),
    )
    adapter_pass.ensure_launcher_unit()  # no raise

    # fail: body lacks custom marker
    adapter_fail = VLLMSystemdAdapter(
        cfg,
        env_file_path=tmp_path / "env",
        systemctl=fake_with("ExecStart=/usr/bin/python -m vllm.api ...\n"),
    )
    with pytest.raises(LegacyUnitError, match="my-custom-launcher"):
        adapter_fail.ensure_launcher_unit()
