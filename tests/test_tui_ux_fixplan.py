"""Regression tests for packages/llmctl/docs/TUI-UX-FIXPLAN-2026-09-18.md."""

from __future__ import annotations

import asyncio

from textual.app import App
from textual.widgets import Button, Input, Static

from llmctl.tui._modals_presets import PresetFormModal


class _Host(App[None]):
    def __init__(self, modal, sink: list) -> None:
        super().__init__()
        self._modal = modal
        self._sink = sink

    def on_mount(self) -> None:
        self.push_screen(self._modal, self._sink.append)


def _drive(modal, steps):
    sink: list = []
    state: dict = {}

    async def _main() -> None:
        app = _Host(modal, sink)
        async with app.run_test() as pilot:
            await pilot.pause()
            await steps(app, pilot, state)

    asyncio.run(_main())
    return sink, state


def _fill_preset(app, **overrides: str) -> None:
    values = {
        "preset-alias": "p1",
        "preset-served": "p1",
        "preset-model-id": "org/m",
        "preset-vquant": "awq_marlin",
        "preset-tp": "2",
        "preset-maxlen": "32768",
        **overrides,
    }
    for wid, value in values.items():
        app.screen.query_one(f"#{wid}", Input).value = value


# --- P0 #1 / P3 #23: schema-invalid preset input must not crash --------------


def test_out_of_range_tp_shows_readable_error_and_app_survives() -> None:
    async def steps(app, pilot, state) -> None:
        _fill_preset(app, **{"preset-tp": "99"})
        app.screen.query_one("#preset-submit", Button).press()
        await pilot.pause()
        state["error"] = str(app.screen.query_one("#preset-form-error", Static).render())
        state["alive"] = app.is_running

    sink, state = _drive(PresetFormModal(), steps)
    assert sink == []  # not dismissed
    assert state["alive"]
    assert "TP" in state["error"] and "1..8" in state["error"], state["error"]
    assert "input_value" not in state["error"]


def test_garbage_optional_number_is_refused_not_dropped() -> None:
    async def steps(app, pilot, state) -> None:
        _fill_preset(app, **{"preset-maxseqs": "lots"})
        app.screen.query_one("#preset-submit", Button).press()
        await pilot.pause()
        state["error"] = str(app.screen.query_one("#preset-form-error", Static).render())

    sink, state = _drive(PresetFormModal(), steps)
    assert sink == []
    assert "max_num_seqs" in state["error"]
