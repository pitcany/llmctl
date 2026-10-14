"""CLI ergonomics from FIXPLAN 2026-09-18 §2.2 (items 1-3)."""

from __future__ import annotations

from typer.testing import CliRunner

from llmctl.cli import app

runner = CliRunner()


def test_dash_h_is_help_at_root_and_in_subcommands() -> None:
    assert runner.invoke(app, ["-h"]).exit_code == 0
    res = runner.invoke(app, ["model", "show", "-h"])
    assert res.exit_code == 0, res.output
    assert "Usage" in res.output


def test_help_command_explains_a_subcommand() -> None:
    res = runner.invoke(app, ["help", "swap"])
    assert res.exit_code == 0, res.output
    assert "preset" in res.output.lower()  # Typer >= 0.27 lowercases the metavar


def test_help_command_rejects_unknown_command() -> None:
    res = runner.invoke(app, ["help", "nope"])
    assert res.exit_code != 0


def test_unknown_command_suggests_once() -> None:
    res = runner.invoke(app, ["presetz"])
    assert res.exit_code != 0
    assert res.output.count("Did you mean") == 1, res.output


def test_swap_switch_and_vllm_are_the_same_command() -> None:
    for name in ("swap", "switch", "vllm"):
        res = runner.invoke(app, [name, "--help"])
        assert res.exit_code == 0, (name, res.output)
        assert "MUTATES" in res.output, name


# --- noun-surface cleanup (FIXPLAN Phase 3) ----------------------------------


def test_canonical_list_commands_exist() -> None:
    for argv in (["model", "list", "--help"], ["profile", "list", "--help"]):
        res = runner.invoke(app, argv)
        assert res.exit_code == 0, (argv, res.output)


def test_deprecated_twins_are_hidden_but_still_work() -> None:
    listing = runner.invoke(app, ["--help"]).output
    for old in ("add-model", "delete-model", "profiles"):
        assert f" {old} " not in listing, old
        assert runner.invoke(app, [old, "--help"]).exit_code == 0, old
