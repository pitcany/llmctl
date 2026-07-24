"""One action per screen at a time.

A preset launch rewrites ``vllm-tp.env`` and issues ``systemctl restart``,
and takes 1-3 minutes. Nothing stopped a second Enter from starting a second
``start_vllm_tp`` that interleaves its env write with the first and issues a
competing restart. Two concurrent ``ctrl+s`` scans likewise write the same
SQLite file.

The guard must *refuse to start*, not cancel: Textual thread workers run on the
loop's default executor and cannot be interrupted, so a "cancelled" launch
would keep driving systemctl while the UI believed it had stopped.
"""

from __future__ import annotations

import asyncio
import threading

from llmctl.tui import _data
from llmctl.tui import app as app_mod
from llmctl.tui.app import MissionControlApp


def _stub_models(monkeypatch) -> None:
    monkeypatch.setattr(_data, "get_models", lambda: [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    monkeypatch.setattr(app_mod, "REFRESH_INTERVAL", 3600.0)


async def _show_models(app, pilot) -> None:
    app.action_show_models()
    for _ in range(60):
        await pilot.pause(0.02)
        if app.screen.__class__.__name__.startswith("Models"):
            return
    raise AssertionError("models screen did not activate")


def test_second_action_is_refused_while_one_is_in_flight(monkeypatch) -> None:
    """The second call must not reach the service layer at all."""
    _stub_models(monkeypatch)
    release = threading.Event()
    starts: list[int] = []
    notes: list[str] = []

    def slow_action() -> str:
        starts.append(1)
        release.wait(timeout=5.0)
        return "done"

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show_models(app, pilot)
            screen = app.screen
            monkeypatch.setattr(
                app, "notify", lambda msg, *a, **k: notes.append(str(msg))
            )
            screen.run_action_worker(slow_action, lambda _r: None)
            for _ in range(20):
                await pilot.pause(0.02)
                if starts:
                    break
            # Second attempt while the first is still blocked.
            screen.run_action_worker(slow_action, lambda _r: None)
            for _ in range(20):
                await pilot.pause(0.02)
            assert len(starts) == 1, (
                f"a second action started while one was in flight ({len(starts)})"
            )
            assert any("already running" in n.lower() for n in notes), (
                f"the refusal was not explained to the user: {notes!r}"
            )
            release.set()
            for _ in range(30):
                await pilot.pause(0.02)

    asyncio.run(_run())


def test_actions_work_again_after_the_first_finishes(monkeypatch) -> None:
    """The guard must clear, or the screen is permanently inert."""
    _stub_models(monkeypatch)
    release = threading.Event()
    starts: list[int] = []

    def action() -> str:
        starts.append(1)
        release.wait(timeout=5.0)
        return "done"

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show_models(app, pilot)
            screen = app.screen
            screen.run_action_worker(action, lambda _r: None)
            for _ in range(20):
                await pilot.pause(0.02)
                if starts:
                    break
            release.set()
            for _ in range(40):
                await pilot.pause(0.02)
            release.clear()
            release.set()  # second run returns immediately
            screen.run_action_worker(action, lambda _r: None)
            for _ in range(40):
                await pilot.pause(0.02)
            assert len(starts) == 2, (
                f"the guard did not clear after completion ({len(starts)} runs)"
            )

    asyncio.run(_run())


def test_guard_clears_even_when_the_action_raises(monkeypatch) -> None:
    """A failing action must not wedge the screen forever."""
    _stub_models(monkeypatch)
    starts: list[int] = []

    def boom() -> str:
        starts.append(1)
        raise RuntimeError("nope")

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show_models(app, pilot)
            screen = app.screen
            monkeypatch.setattr(app, "notify", lambda *a, **k: None)
            screen.run_action_worker(boom, lambda _r: None)
            for _ in range(40):
                await pilot.pause(0.02)
            screen.run_action_worker(boom, lambda _r: None)
            for _ in range(40):
                await pilot.pause(0.02)
            assert len(starts) == 2, (
                f"a raising action left the guard set ({len(starts)} runs)"
            )

    asyncio.run(_run())


def test_auto_refresh_is_not_blocked_by_an_in_flight_action(monkeypatch) -> None:
    """The guard covers actions only; data refresh must keep working."""
    fetches: list[int] = []
    monkeypatch.setattr(_data, "get_models", lambda: fetches.append(1) or [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    monkeypatch.setattr(app_mod, "REFRESH_INTERVAL", 3600.0)
    release = threading.Event()

    def slow_action() -> str:
        release.wait(timeout=5.0)
        return "done"

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show_models(app, pilot)
            screen = app.screen
            for _ in range(20):
                await pilot.pause(0.02)
            before = len(fetches)
            screen.run_action_worker(slow_action, lambda _r: None)
            for _ in range(10):
                await pilot.pause(0.02)
            screen.refresh_data()
            for _ in range(30):
                await pilot.pause(0.02)
            assert len(fetches) > before, (
                "refresh was blocked by the in-flight action guard"
            )
            release.set()
            for _ in range(20):
                await pilot.pause(0.02)

    asyncio.run(_run())


def test_launch_flow_chains_two_workers_without_self_blocking(monkeypatch) -> None:
    """probe -> modal -> confirm -> launch must survive the guard.

    ``action_launch_selected`` runs a worker to probe the unit, and confirming
    the modal starts a second worker for the launch itself. If the first
    worker's guard were still set when the second starts, the guard would make
    launching impossible.
    """
    from llmctl.services.preset_loader import PresetView
    from llmctl.tui import screens_presets

    view = PresetView(
        alias="ornith-35b",
        served_name="ornith-35b",
        model_id="org/model",
        family="qwen",
        param_count_b=35.0,
        tensor_parallel=2,
        quantization="fp8",
        source_path=None,
    )
    launched: list[str] = []

    class _Result:
        ok = True
        fleet_failed: list[str] = []
        restart = None
        spec = type("S", (), {"port": 8003})()

    monkeypatch.setattr(_data, "get_preset_views_with_links", lambda: [view])
    monkeypatch.setattr(_data, "get_served_on_tp_unit", lambda: ["previous-model"])
    monkeypatch.setattr(
        screens_presets,
        "start_vllm_tp",
        lambda alias, **kw: launched.append(alias) or _Result(),
    )
    monkeypatch.setattr(app_mod, "REFRESH_INTERVAL", 3600.0)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            app.action_show_presets()
            for _ in range(60):
                await pilot.pause(0.02)
                if app.screen.__class__.__name__.startswith("Presets"):
                    break
            screen = app.screen
            for _ in range(40):
                await pilot.pause(0.02)
                if screen._row_aliases:
                    break
            screen.action_launch_selected()
            for _ in range(60):
                await pilot.pause(0.02)
                if app.screen.__class__.__name__ == "PresetLaunchModal":
                    break
            assert app.screen.__class__.__name__ == "PresetLaunchModal", (
                "the probe worker's guard blocked the launch modal from opening"
            )
            from llmctl.tui._modals_presets import PresetLaunchTarget

            app.screen.dismiss(PresetLaunchTarget.TP)
            for _ in range(80):
                await pilot.pause(0.02)
                if launched:
                    break

    asyncio.run(_run())
    assert launched == ["ornith-35b"], (
        f"the launch never reached the orchestrator: {launched!r}"
    )
