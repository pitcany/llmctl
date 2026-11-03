"""Model selection: search, engine filter, detail panel, stable anchor (§3B)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from textual.widgets import DataTable, Static

from llmctl._testing import isolate_tui
from llmctl.db import BenchmarkKind, RuntimeName
from llmctl.schemas import BenchmarkResult, Model
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_models import ModelsScreen


@pytest.fixture(autouse=True)
def _isolated_tui(tmp_path, monkeypatch) -> None:
    isolate_tui(monkeypatch, tmp_path)


def _model(mid: str, name: str, runtime: RuntimeName = RuntimeName.OLLAMA, **kw) -> Model:
    return Model(id=mid, name=name, runtime=runtime, source=kw.pop("source", None), **kw)


MODELS = [
    _model("m-alpha", "alpha-8b", source="alpha:8b"),
    _model("m-beta", "beta-70b", RuntimeName.VLLM, source="org/beta-70b"),
    _model("m-gamma", "gamma-mini", RuntimeName.LLAMA_CPP, source="/models/gamma.gguf"),
    _model("m-delta", "delta-tiny", source="delta:latest"),
    _model("m-eps", "epsilon-xl", RuntimeName.VLLM, source="org/epsilon-xl"),
]


def _patched(models, benchmarks=None):
    from contextlib import ExitStack

    stack = ExitStack()
    for patch_obj in _patches(models, benchmarks):
        stack.enter_context(patch_obj)
    return stack

def _patches(models, benchmarks=None):
    return [
        patch("llmctl.tui._data.get_models", return_value=models),
        patch(
            "llmctl.tui._data.get_backend_map",
            return_value={"ollama": True, "vllm": True, "llama_cpp": False},
        ),
        patch("llmctl.tui._data.get_preset_count_by_model", return_value={}),
        patch("llmctl.tui._data.get_missing_count", return_value=0),
        patch("llmctl.tui._data.get_benchmarks", return_value=benchmarks or []),
        patch("llmctl.tui._data.get_missing_model_ids", return_value=[]),
    ]


async def _boot_to_models(pilot, app) -> ModelsScreen:
    await pilot.press("m")
    for _ in range(60):
        await pilot.pause(0.05)
        if isinstance(app.screen, ModelsScreen):
            break
    screen = app.screen
    assert isinstance(screen, ModelsScreen)
    for _ in range(40):
        await pilot.pause(0.05)
        if screen.query_one("#models-table", DataTable).row_count:
            break
    return screen


def test_search_narrows_the_list() -> None:
    async def _run() -> list[str]:
        app = MissionControlApp()
        async with app.run_test(size=(100, 40)) as pilot:
            with _patched(MODELS):
                screen = await _boot_to_models(pilot, app)
                screen._filter_text = "beta"
                screen.render_data(screen._last_data)
                await pilot.pause(0.1)
                return list(screen._ids)

    assert asyncio.run(_run()) == ["m-beta"]


def test_engine_filter_narrows_the_list() -> None:
    async def _run() -> list[str]:
        app = MissionControlApp()
        async with app.run_test(size=(100, 40)) as pilot:
            with _patched(MODELS):
                screen = await _boot_to_models(pilot, app)
                screen._filter_engine = "vllm"
                screen.render_data(screen._last_data)
                await pilot.pause(0.1)
                return list(screen._ids)

    assert asyncio.run(_run()) == ["m-beta", "m-eps"]


def test_no_match_message_and_recovery() -> None:
    async def _run() -> tuple[int, list[str]]:
        app = MissionControlApp()
        async with app.run_test(size=(100, 40)) as pilot:
            with _patched(MODELS):
                screen = await _boot_to_models(pilot, app)
                screen._filter_text = "nothing-matches-this"
                screen.render_data(screen._last_data)
                await pilot.pause(0.1)
                rows = screen.query_one("#models-table", DataTable).row_count
                screen._filter_text = ""
                screen.render_data(screen._last_data)
                await pilot.pause(0.1)
                return rows, list(screen._ids)

    rows, ids = asyncio.run(_run())
    assert rows == 1  # the "no models match" row
    assert ids == [m.id for m in MODELS]


def test_detail_panel_labels_unknown_fit_honestly() -> None:
    async def _run() -> str:
        app = MissionControlApp()
        async with app.run_test(size=(100, 40)) as pilot:
            with _patched(MODELS[:1]):
                screen = await _boot_to_models(pilot, app)
                screen._anchor_id = "m-alpha"
                screen._update_detail()
                await pilot.pause(0.05)
                return str(screen.query_one("#models-detail", Static).content)

    detail = asyncio.run(_run())
    assert "unknown" in detail
    assert "none yet" in detail  # no benchmark evidence is stated, not blank


def test_anchor_survives_refresh_and_filter() -> None:
    """§3B: selection is the model id; a refresh or filter change can never
    retarget Enter to a different model."""

    async def _run() -> tuple[str, str]:
        app = MissionControlApp()
        async with app.run_test(size=(100, 40)) as pilot:
            with _patched(MODELS):
                screen = await _boot_to_models(pilot, app)
                table = screen.query_one("#models-table", DataTable)
                table.move_cursor(row=1)  # beta
                await pilot.pause(0.05)
                assert screen._anchor_id == "m-beta"
                screen.render_data(screen._last_data)
                await pilot.pause(0.05)
                after_refresh = screen._selected_id()
                # Filter to the vllm engine, where beta becomes the first row:
                screen._filter_engine = "vllm"
                screen.render_data(screen._last_data)
                await pilot.pause(0.05)
                after_filter = screen._selected_id()
                return after_refresh or "", after_filter or ""

    after_refresh, after_filter = asyncio.run(_run())
    assert after_refresh == "m-beta"
    assert after_filter == "m-beta"


def test_detail_shows_latest_benchmark_evidence() -> None:
    latest = BenchmarkResult(
        id="b1",
        name="smoke",
        model_id="m-alpha",
        kind=BenchmarkKind.CHAT,
        tokens_per_second=48.5,
        parameters={"mode": "live", "token_count_source": "server_usage"},
        success=True,
        created_at=datetime(2026, 11, 3, 10, 0, tzinfo=UTC),
    )

    async def _run() -> str:
        app = MissionControlApp()
        async with app.run_test(size=(100, 40)) as pilot:
            with _patched(MODELS[:1], [latest]):
                screen = await _boot_to_models(pilot, app)
                screen._anchor_id = "m-alpha"
                screen._update_detail()
                await pilot.pause(0.05)
                return str(screen.query_one("#models-detail", Static).content)

    detail = asyncio.run(_run())
    assert "smoke" in detail
    assert "48.5 tok/s" in detail
    assert "ok" in detail


def test_enter_in_search_returns_to_table_without_serving(monkeypatch) -> None:
    async def run() -> None:
        with _patched(MODELS):
            app = MissionControlApp()
            async with app.run_test(size=(80, 24)) as pilot:
                screen = await _boot_to_models(pilot, app)
                launches = []
                monkeypatch.setattr(
                    screen, "run_action_worker", lambda *args: launches.append(args),
                )
                await pilot.press("/")
                await pilot.press(*"beta")
                await pilot.press("enter")
                await pilot.pause()
                assert app.focused is screen.query_one("#models-table", DataTable)
                assert screen._selected_id() == "m-beta"
                assert launches == []
    asyncio.run(run())
