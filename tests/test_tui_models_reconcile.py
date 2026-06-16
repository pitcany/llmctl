"""TUI tests for missing-model rendering and the prune action."""

from __future__ import annotations

import asyncio

from textual.widgets import DataTable

from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import Model
from llmctl.tui import _data
from llmctl.tui._base import C_MUTED, C_WARN
from llmctl.tui.app import MissionControlApp


def _ghost() -> Model:
    return Model(
        id="ghost-id",
        name="ghost",
        runtime=RuntimeName.OLLAMA,
        source="ghost",
        status=ModelStatus.MISSING,
    )


async def _show_models(app: MissionControlApp, pilot) -> None:
    app.action_show_models()
    for _ in range(50):
        await pilot.pause(0.02)
        if app.screen.__class__.__name__.lower().startswith("models"):
            return
    raise AssertionError("ModelsScreen did not become active")


def test_missing_row_is_dimmed_with_warn_status(monkeypatch) -> None:
    monkeypatch.setattr(_data, "get_models", lambda: [_ghost()])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {"ollama": True})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 1)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show_models(app, pilot)
            table = app.screen.query_one("#models-table", DataTable)
            for _ in range(50):
                await pilot.pause(0.02)
                if table.row_count and "ghost" in str(table.get_row_at(0)[1]):
                    break
            row = table.get_row_at(0)
            # Columns: ID, Name, Runtime, Backend, Status, Quant, Path, Presets
            assert "missing" in str(row[4])
            assert C_WARN in str(row[4])
            assert C_MUTED in str(row[1])

    asyncio.run(_run())


def test_prune_action_with_no_missing_notifies(monkeypatch) -> None:
    monkeypatch.setattr(_data, "get_models", lambda: [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    notes: list[tuple] = []

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show_models(app, pilot)
            monkeypatch.setattr(app, "notify", lambda *a, **k: notes.append((a, k)))
            app.screen.action_prune_missing()
            await pilot.pause()
            # No modal pushed; still on the models screen.
            assert app.screen.__class__.__name__.lower().startswith("models")
            assert notes and "No missing" in notes[0][0][0]

    asyncio.run(_run())


def test_missing_count_reflects_inactive_inclusive_data(monkeypatch) -> None:
    # The visible table (get_models) has no rows, but get_missing_count reports
    # inactive MISSING records that prune_missing would still remove.
    monkeypatch.setattr(_data, "get_models", lambda: [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 2)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show_models(app, pilot)
            for _ in range(50):
                await pilot.pause(0.02)
                if app.screen._missing_count == 2:
                    break
            assert app.screen._missing_count == 2

    asyncio.run(_run())
