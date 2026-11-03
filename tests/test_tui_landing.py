"""Landing view + operation progress tests (FIXPLAN §3A/§3C)."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from llmctl._testing import isolate_tui
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_dashboard import DashboardScreen
from llmctl.tui.screens_swap import SwapScreen
from textual.widgets import Static


@pytest.fixture(autouse=True)
def _isolated_tui(tmp_path, monkeypatch) -> None:
    isolate_tui(monkeypatch, tmp_path)


def _landing(**overrides):
    base = {
        "fetched_at": "2026-11-03T12:00:00+00:00",
        "serving": [],
        "operation": None,
        "recent_experiments": [],
        "inventory": {"models": 3, "sessions_running": 1, "profiles": 2, "gpus": 2},
        "safe_mode": True,
        "runtimes": {},
        "scheduler_warnings": [],
        "router": None,
        "nvml_available": False,
    }
    base.update(overrides)
    return base


def _render_landing(landing) -> dict[str, str]:
    """Boot the app, land on the dashboard, return each panel's text."""

    async def _run() -> dict[str, str]:
        with patch("llmctl.tui._data.get_landing", return_value=landing):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, DashboardScreen):
                        break
                assert isinstance(app.screen, DashboardScreen)
                return {
                    "serving": app.screen.query_one("#dashboard-serving", Static).content,
                    "operation": app.screen.query_one(
                        "#dashboard-operation", Static
                    ).content,
                    "experiments": app.screen.query_one(
                        "#dashboard-experiments", Static
                    ).content,
                    "details": app.screen.query_one(
                        "#dashboard-runtimes", Static
                    ).content,
                }

    return asyncio.run(_run())


def test_empty_landing_states_are_explicit() -> None:
    panels = _render_landing(_landing())
    assert "nothing serving" in panels["serving"]
    assert "none" in panels["operation"]
    assert "none yet" in panels["experiments"]


def test_serving_shows_identity_endpoint_and_history_label() -> None:
    panels = _render_landing(
        _landing(
            serving=[
                {
                    "key": "k",
                    "model": "demo-model",
                    "engine": "vLLM",
                    "state": "serving",
                    "endpoint": "http://127.0.0.1:8003/v1",
                    "last_load_s": 95.0,
                }
            ]
        )
    )
    text = panels["serving"]
    assert "demo-model" in text
    assert "http://127.0.0.1:8003/v1" in text
    # Icon + word state signals, not color alone; history is labeled.
    assert "serving" in text
    assert "historical" in text
    assert "observed" in text


def test_loading_state_signals_with_icon_and_text() -> None:
    panels = _render_landing(
        _landing(
            serving=[
                {
                    "key": "k",
                    "model": "m",
                    "engine": "e",
                    "state": "loading",
                    "endpoint": None,
                    "last_load_s": None,
                }
            ]
        )
    )
    assert "loading" in panels["serving"]
    assert "endpoint unknown" in panels["serving"]


def test_running_operation_shows_phase_elapsed_interface() -> None:
    panels = _render_landing(
        _landing(
            operation={
                "id": "op-1",
                "kind": "swap",
                "target": "preset-x",
                "phase": "loading",
                "elapsed_s": 42.0,
                "interface": "api",
                "pid": 4242,
                "terminal": False,
                "recovery": {},
                "output_tail": [],
                "store_url": "sqlite://",
            }
        )
    )
    text = panels["operation"]
    assert "loading" in text
    assert "42s elapsed" in text
    assert "api" in text and "4242" in text


def test_needs_attention_operation_shows_guidance() -> None:
    panels = _render_landing(
        _landing(
            operation={
                "id": "op-2",
                "kind": "swap",
                "target": "preset-x",
                "phase": "needs_attention",
                "elapsed_s": None,
                "interface": "cli",
                "pid": 1,
                "terminal": True,
                "recovery": {
                    "guidance": "verify what is serving (`llmctl status`)",
                    "not_restored": ["lane-a"],
                },
                "output_tail": [],
                "store_url": "sqlite://",
            }
        )
    )
    text = panels["operation"]
    assert "needs_attention" in text
    assert "verify what is serving" in text
    assert "not restored" in text and "lane-a" in text


def test_experiments_label_estimates_and_failures() -> None:
    panels = _render_landing(
        _landing(
            recent_experiments=[
                {
                    "name": "smoke",
                    "kind": "chat",
                    "mode": "live",
                    "tokens_per_second": 61.2,
                    "token_count_source": "server_usage",
                    "success": True,
                    "created_at": "2026-11-03T11:00:00+00:00",
                    "model_id": "m1",
                },
                {
                    "name": "bad",
                    "kind": "chat",
                    "mode": "live",
                    "tokens_per_second": None,
                    "token_count_source": "estimate",
                    "success": False,
                    "created_at": "2026-11-03T10:00:00+00:00",
                    "model_id": "m1",
                },
            ]
        )
    )
    text = panels["experiments"]
    assert "61.2 tok/s" in text
    assert "n" in text  # failed marker
    # Inventory/config flags are demoted to the details panel.
    assert "safe mode" in panels["details"]


def test_config_flags_demoted_from_primary_panels() -> None:
    panels = _render_landing(_landing())
    for key in ("serving", "operation", "experiments"):
        assert "safe mode" not in panels[key]


# -- swap screen: operation panel + anchored selection (§3C) ---------------


def _swap_state(catalog_keys: list[str]):
    from llmctl.services.lanes import Lane, SwapState, SwapTarget

    catalog = [
        SwapTarget(
            key=key,
            kind="lane",
            model=key.upper(),
            engine="vllm",
            upstream=f"http://h:{i}",
            lane=Lane(key, key.upper(), f"http://h:{i}", ("start",)),
        )
        for i, key in enumerate(catalog_keys)
    ]
    return SwapState(
        catalog=catalog, serving=set(), active_units=set(), vllm_unit="v", vllm_active=False
    )


def test_swap_screen_shows_operation_and_keeps_anchor() -> None:
    op = {
        "id": "op-1",
        "kind": "swap",
        "target": "alpha",
        "phase": "stopping",
        "elapsed_s": 5.0,
        "interface": "cli",
        "pid": 99,
        "terminal": False,
        "recovery": {},
        "output_tail": [],
        "store_url": "sqlite://",
    }

    async def _run() -> tuple[str, str]:
        with patch(
            "llmctl.tui.screens_swap.gather_state", return_value=_swap_state(["alpha", "beta"])
        ), patch(
            "llmctl.tui.screens_swap._data.get_current_operation", return_value=op
        ) as _get_op, patch(
            "llmctl.services.vram_fit.free_mib_now", return_value=0
        ):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("w")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, SwapScreen):
                        break
                screen = app.screen
                assert isinstance(screen, SwapScreen)
                panel = screen.query_one("#swap-operation", Static).content
                table = screen.query_one("#swap-table")
                table.move_cursor(row=1)  # select "beta"
                await pilot.pause(0.05)
                anchored = screen._anchor_key
                # A refresh with the same catalog must not retarget.
                with patch(
                    "llmctl.tui.screens_swap.gather_state",
                    return_value=_swap_state(["beta", "alpha"]),
                ):
                    screen.refresh_data()
                    for _ in range(40):
                        await pilot.pause(0.05)
                        row_now = table.cursor_row
                        if table.row_count == 2 and row_now is not None:
                            break
                return str(panel), str(screen._state.catalog[row_now].key)

    panel, selected = asyncio.run(_run())
    assert "stopping" in panel and "5s elapsed" in panel
    # After a refresh that swapped the row order, the same target stays
    # selected: the anchor is the key, never the row number.
    assert selected == "beta"
