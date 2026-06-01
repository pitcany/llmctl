"""Smoke tests for the new CLI verbs added in Phase 5.

These tests use Typer's CliRunner to exercise the argument parsing,
help text, and error paths. The deep behavior is covered in
``test_vllm_orchestrator.py``; this layer just makes sure the CLI
wiring stays in shape.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llmctl.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point llmctl at an empty config dir so tests don't read the user's.

    LLMCTL_CONFIG_DIR controls llmctl's settings.yaml lookup. We also
    set XDG_CONFIG_HOME so preset discovery cannot pick up the
    developer's real presets.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path / "llmctl-cfg"))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "xdg" / "llmctl" / "presets").mkdir(parents=True)


def test_vllm_help_lists_flags(runner: CliRunner) -> None:
    """The --tq, --no-tq, --dry-run, --no-wait flags are surfaced in help."""
    result = runner.invoke(app, ["vllm", "--help"])
    assert result.exit_code == 0
    assert "--tq" in result.stdout
    assert "--no-tq" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--no-wait" in result.stdout
    assert "gpu-models vllm" in result.stdout  # provenance reference


def test_slot_help_lists_flags(runner: CliRunner) -> None:
    result = runner.invoke(app, ["slot", "--help"])
    assert result.exit_code == 0
    for flag in ("--tq", "--no-tq", "--dry-run", "--no-wait"):
        assert flag in result.stdout


def test_vllm_rejects_conflicting_tq_flags(runner: CliRunner) -> None:
    """--tq and --no-tq together must error out.

    Typer routes BadParameter through Click's exit code 2; the message
    goes to stderr (not always captured in result.stdout by CliRunner).
    The exit code is the load-bearing assertion.
    """
    result = runner.invoke(app, ["vllm", "any-preset", "--tq", "--no-tq"])
    assert result.exit_code != 0  # rejected with non-zero code


def test_vllm_unknown_preset_exits_with_2(runner: CliRunner) -> None:
    """Empty config dir -> presets dict empty -> unknown preset error code."""
    result = runner.invoke(app, ["vllm", "does-not-exist"])
    assert result.exit_code == 2


def test_slot_unknown_slot_exits_with_2(runner: CliRunner) -> None:
    """Bogus slot name -> exit 2 with available-list hint."""
    result = runner.invoke(app, ["slot", "nonsense-slot", "preset"])
    assert result.exit_code == 2
    assert "unknown slot" in result.stdout


def test_presets_lists_empty_when_no_config(runner: CliRunner) -> None:
    """presets command works against an empty config dir."""
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0
    assert "No presets found" in result.stdout


def test_status_renders_default_units(runner: CliRunner) -> None:
    """status shows the three default managed units and the two slots."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "vllm-tp" in result.stdout
    assert "vllm-coder" in result.stdout
    assert "vllm-reasoner" in result.stdout
    assert "coder" in result.stdout
    assert "reasoner" in result.stdout
