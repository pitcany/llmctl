"""``not_used`` renders muted in the CLI table, the TUI and the HTTP API."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llmctl.api.app import create_app
from llmctl.cli import app
from llmctl.config import Settings
from llmctl.services.health import HealthService

_RUNTIME_KEYS = {"state", "message", "sessions", "serving_count", "unmanaged_count"}


def _health() -> dict:
    return {
        "state": "ok",
        "safe_mode": True,
        "database_url": "sqlite://",
        "gpu_count": 0,
        "nvml_available": False,
        "session_state_available": True,
        "message": "LLM Mission Control is running.",
        "runtimes": {
            "ollama": {
                "state": "not_used",
                "message": "ollama is not used here",
                "sessions": [],
                "serving_count": 0,
                "unmanaged_count": 0,
            }
        },
    }


def test_cli_health_table_and_json(monkeypatch) -> None:
    monkeypatch.setattr(HealthService, "get_health", lambda self: _health())
    runner = CliRunner()
    result = runner.invoke(app, ["health"])
    assert result.exit_code == 0, result.output
    assert "not_used" in result.output
    assert "not used" in result.output

    result = runner.invoke(app, ["health", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(_health()) <= set(data)
    assert _RUNTIME_KEYS <= set(data["runtimes"]["ollama"])


def test_cli_not_used_row_is_dim(monkeypatch) -> None:
    from rich.console import Console

    import llmctl.cli as cli

    monkeypatch.setattr(HealthService, "get_health", lambda self: _health())
    console = Console(force_terminal=True, color_system="standard", width=120)
    monkeypatch.setattr(cli, "console", console)
    with console.capture() as cap:
        CliRunner().invoke(app, ["health"])
    # SGR 2 = dim; the red "nothing" style (31) must not be used.
    assert "\x1b[2m" in cap.get()
    assert "\x1b[31m" not in cap.get()


def test_api_health_passes_not_used_through(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LLMCTL_API_TOKEN", raising=False)
    monkeypatch.setattr(HealthService, "get_health", lambda self: _health())
    client = TestClient(
        create_app(Settings(), database_url=f"sqlite:///{tmp_path / 'h.sqlite3'}")
    )
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["runtimes"]["ollama"]["state"] == "not_used"


def test_dashboard_renders_not_used_muted(tmp_path, monkeypatch) -> None:
    from textual.widgets import Static

    from llmctl._testing import isolate_tui
    from llmctl.tui import _data
    from llmctl.tui._base import C_MUTED, C_WARN
    from llmctl.tui.app import MissionControlApp
    from llmctl.tui.screens_dashboard import DashboardScreen

    isolate_tui(monkeypatch, tmp_path)
    real = _data.get_overview

    def overview():
        data = real()
        data["runtimes"] = _health()["runtimes"]
        return data

    monkeypatch.setattr(_data, "get_overview", overview)
    seen: list[str] = []

    async def _run() -> None:
        tui = MissionControlApp()
        async with tui.run_test() as pilot:
            await pilot.pause()
            assert isinstance(tui.screen, DashboardScreen)
            tui.screen.render_data(overview())
            await pilot.pause()
            seen.append(str(tui.screen.query_one("#dashboard-runtimes", Static).content))

    asyncio.run(_run())
    line = next(x for x in seen[0].splitlines() if "ollama" in x)
    assert f"[{C_MUTED}]*[/]" in line
    assert C_WARN not in line
