"""CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from llmctl.cli import app


def test_cli_help_lists_required_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = result.output
    for command in [
        "scan",
        "models",
        "gpus",
        "sessions",
        "add-model",
        "delete-model",
        "start",
        "stop",
        "restart",
        "logs",
        "bench",
        "tui",
        "serve",
        "generate-systemd",
    ]:
        assert command in output
