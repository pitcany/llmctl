"""Swap screen: the list renders, Enter previews before anything happens."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App
from textual.widgets import DataTable, Static

from llmctl._testing import isolate_tui
from llmctl.services.lanes import Lane, SwapState, build_catalog
from llmctl.tui import screens_swap
from llmctl.tui._modals import ConfirmActionModal

FT = Lane(
    name="flash",
    served_model="ft-model",
    upstream="http://127.0.0.1:1919",
    start_cmd=("ft-start", "--take-gpus"),
    unit="ft.service",
    runtime="freetoken",
)


class _Spec:
    served_name = "qwen3.8-27b"


STATE = SwapState(
    catalog=build_catalog({"qwen3.8-27b": _Spec()}, [FT], vllm_port=8003),
    serving={"flash"},
    active_units={"ft.service"},
    vllm_unit="vllm-tp",
    vllm_active=False,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    isolate_tui(monkeypatch, tmp_path)
    monkeypatch.setattr(screens_swap, "gather_state", lambda settings: STATE)
    ran: list[str] = []
    monkeypatch.setattr(screens_swap, "run_swap", lambda s, t: ran.append(t.key) or (True, "ok"))
    return ran


class _Host(App[None]):
    def on_mount(self) -> None:
        self.push_screen(screens_swap.SwapScreen())


def _drive(body):
    async def _main() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            await body(app, pilot)

    asyncio.run(_main())


def test_lists_presets_and_lanes_with_state(_isolated):
    async def body(app, pilot):
        table = app.screen.query_one("#swap-table", DataTable)
        assert table.row_count == 2
        rendered = " ".join(str(c) for c in table.get_row_at(1))
        assert "FreeToken" in rendered and "serving" in rendered

    _drive(body)


def test_enter_previews_and_confirm_runs_the_swap(_isolated):
    async def body(app, pilot):
        await pilot.press("enter")  # row 0: the vLLM preset
        await pilot.pause()
        assert isinstance(app.screen, ConfirmActionModal)
        text = " ".join(str(w.content) for w in app.screen.query(Static))
        assert "ft.service" in text  # the preview names the lane it stops
        assert _isolated == []  # nothing ran before confirming
        await pilot.click("#confirm-ok")
        await pilot.pause()
        await pilot.pause()
        assert _isolated == ["qwen3.8-27b"]

    _drive(body)


def test_already_serving_lane_does_not_prompt(_isolated):
    async def body(app, pilot):
        await pilot.press("down", "enter")
        await pilot.pause()
        assert not isinstance(app.screen, ConfirmActionModal)
        assert _isolated == []

    _drive(body)
