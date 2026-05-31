"""Unit tests for :mod:`llmctl.integrations.hermes`.

The integration is read-only and must never mutate the user's Hermes
config — these tests pin that contract and verify the status returned
in each disposition (not installed, no config, no provider, drift, OK).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmctl.integrations.hermes import (
    HermesStatus,
    is_installed,
    verify_provider,
)


def _hermes_config(providers: list[dict]) -> str:
    """Render a minimal Hermes config YAML body for tests."""
    import yaml

    return yaml.safe_dump({"custom_providers": providers})


def test_not_installed_when_binary_missing() -> None:
    """No ``hermes`` on PATH -> NOT_INSTALLED, no other work attempted."""
    logged: list[str] = []
    result = verify_provider(
        "vllm",
        expected_port=8003,
        config_path=Path("/does/not/matter"),
        which=lambda _name: None,
        logger=logged.append,
    )
    assert result is HermesStatus.NOT_INSTALLED
    assert logged == []  # silent — caller decides whether to print


def test_no_config_file_returns_no_config(tmp_path: Path) -> None:
    """Hermes installed but no config -> NO_CONFIG, one info line logged."""
    logged: list[str] = []
    result = verify_provider(
        "vllm",
        expected_port=8003,
        config_path=tmp_path / "absent.yaml",
        which=lambda _name: "/usr/bin/hermes",
        logger=logged.append,
    )
    assert result is HermesStatus.NO_CONFIG
    assert any("config not found" in line for line in logged)


def test_provider_missing_returns_no_provider(tmp_path: Path) -> None:
    """Provider name absent from custom_providers -> NO_PROVIDER."""
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text(
        _hermes_config(
            [{"name": "ollama-fast", "base_url": "http://127.0.0.1:11434/v1"}]
        )
    )
    logged: list[str] = []
    result = verify_provider(
        "vllm",
        expected_port=8003,
        config_path=cfg,
        which=lambda _name: "/usr/bin/hermes",
        logger=logged.append,
    )
    assert result is HermesStatus.NO_PROVIDER
    assert any("vllm" in line for line in logged)


def test_url_mismatch_returns_drift_warning(tmp_path: Path) -> None:
    """Provider exists but base_url points at wrong port -> URL_MISMATCH."""
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text(
        _hermes_config(
            [{"name": "vllm", "base_url": "http://127.0.0.1:9999/v1"}]
        )
    )
    logged: list[str] = []
    result = verify_provider(
        "vllm",
        expected_port=8003,
        config_path=cfg,
        which=lambda _name: "/usr/bin/hermes",
        logger=logged.append,
    )
    assert result is HermesStatus.URL_MISMATCH
    assert any("WARNING" in line for line in logged)
    assert any("9999" in line and "8003" in line for line in logged)


def test_url_match_returns_ok(tmp_path: Path) -> None:
    """Provider points at the expected port -> OK with confirmation log."""
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text(
        _hermes_config(
            [{"name": "vllm", "base_url": "http://127.0.0.1:8003/v1"}]
        )
    )
    logged: list[str] = []
    result = verify_provider(
        "vllm",
        expected_port=8003,
        config_path=cfg,
        which=lambda _name: "/usr/bin/hermes",
        logger=logged.append,
    )
    assert result is HermesStatus.OK
    assert any("verified" in line for line in logged)


def test_verify_never_writes_to_config(tmp_path: Path) -> None:
    """Read-only contract: verification must not mutate the file."""
    cfg = tmp_path / "hermes.yaml"
    original_body = _hermes_config(
        [{"name": "vllm", "base_url": "http://127.0.0.1:9999/v1"}]
    )
    cfg.write_text(original_body)

    verify_provider(
        "vllm",
        expected_port=8003,
        config_path=cfg,
        which=lambda _name: "/usr/bin/hermes",
        logger=lambda _: None,
    )

    assert cfg.read_text() == original_body


def test_malformed_yaml_returns_no_provider(tmp_path: Path) -> None:
    """Parse errors are treated as "no provider" so a corrupt config
    doesn't crash the start path. The user gets the same "add it"
    message they'd get for a truly missing provider."""
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text("custom_providers: [\n  this is: not valid: YAML\n")
    result = verify_provider(
        "vllm",
        expected_port=8003,
        config_path=cfg,
        which=lambda _name: "/usr/bin/hermes",
        logger=lambda _: None,
    )
    assert result is HermesStatus.NO_PROVIDER


def test_alternate_provider_names_supported(tmp_path: Path) -> None:
    """Slot units use vllm-coder/vllm-reasoner — same machinery."""
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text(
        _hermes_config(
            [
                {"name": "vllm-coder", "base_url": "http://127.0.0.1:8001/v1"},
                {"name": "vllm-reasoner", "base_url": "http://127.0.0.1:8002/v1"},
            ]
        )
    )
    which = lambda _name: "/usr/bin/hermes"  # noqa: E731 - inline shim is clearer than def

    assert (
        verify_provider("vllm-coder", 8001, config_path=cfg, which=which, logger=lambda _: None)
        is HermesStatus.OK
    )
    assert (
        verify_provider(
            "vllm-reasoner", 8002, config_path=cfg, which=which, logger=lambda _: None
        )
        is HermesStatus.OK
    )


def test_is_installed_proxies_to_which() -> None:
    """Cover the tiny helper for completeness."""
    assert is_installed(lambda _name: None) is False
    assert is_installed(lambda _name: "/usr/local/bin/hermes") is True


def test_non_dict_custom_provider_entry_ignored(tmp_path: Path) -> None:
    """A stray string in custom_providers shouldn't crash the loader."""
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text("custom_providers:\n  - bogus_string\n  - {name: vllm, base_url: 'http://127.0.0.1:8003/v1'}\n")
    result = verify_provider(
        "vllm",
        expected_port=8003,
        config_path=cfg,
        which=lambda _name: "/usr/bin/hermes",
        logger=lambda _: None,
    )
    assert result is HermesStatus.OK


def test_non_string_base_url_treated_as_missing(tmp_path: Path) -> None:
    """If a hand-edited config sets base_url to something other than a
    string (e.g. an int port mistakenly), treat as if the provider
    needs reconfiguration rather than crashing."""
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text(
        "custom_providers:\n  - {name: vllm, base_url: 8003}\n"
    )
    result = verify_provider(
        "vllm",
        expected_port=8003,
        config_path=cfg,
        which=lambda _name: "/usr/bin/hermes",
        logger=lambda _: None,
    )
    # Non-string base_url is returned as None from _read_provider_url ->
    # treated as "no provider" (the user needs to fix it).
    assert result is HermesStatus.NO_PROVIDER


@pytest.mark.parametrize("port,expected_url", [
    (8001, "http://127.0.0.1:8001/v1"),
    (8003, "http://127.0.0.1:8003/v1"),
    (11434, "http://127.0.0.1:11434/v1"),
])
def test_expected_url_built_from_port(port: int, expected_url: str, tmp_path: Path) -> None:
    """Verification URL is locked to ``http://127.0.0.1:<port>/v1`` —
    Hermes doesn't support custom hostnames per-provider in our setup."""
    cfg = tmp_path / "hermes.yaml"
    cfg.write_text(_hermes_config([{"name": "p", "base_url": expected_url}]))
    result = verify_provider(
        "p",
        port,
        config_path=cfg,
        which=lambda _: "/usr/bin/hermes",
        logger=lambda _: None,
    )
    assert result is HermesStatus.OK
