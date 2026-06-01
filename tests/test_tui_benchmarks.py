"""TUI tests for the Benchmarks screen.

Covers the two complaints fixed in this slice:

1. No way to launch a benchmark from the TUI (the screen previously only
   re-ran existing rows, so a fresh install showed an empty table with
   no actionable affordance).
2. Setting a baseline on an empty table silently failed -- the
   notification was easy to miss and offered no next step.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from llmctl.config import Settings
from llmctl.db import BenchmarkKind, RuntimeName
from llmctl.schemas import BenchmarkResult, Model
from llmctl.tui._modals_benchmarks import BenchmarkLaunch, BenchmarkLaunchModal
from llmctl.tui._modals_registry import ConfirmDelete, DeleteModal
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_benchmarks import BenchmarksScreen


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Pin llmctl at a tmp data dir so tests don't touch the real DB."""
    db_path = tmp_path / "db.sqlite3"
    settings = Settings()
    settings.database.url = f"sqlite:///{db_path}"
    monkeypatch.setattr("llmctl.tui._data.load_settings", lambda: settings)
    return settings


def _fake_model(model_id: str = "m1") -> Model:
    return Model(
        id=model_id,
        name="demo",
        runtime=RuntimeName.OLLAMA,
        source="demo:latest",
    )


def _fake_result(name: str = "smoke", success: bool = True) -> BenchmarkResult:
    return BenchmarkResult(
        id="bench-1",
        name=name,
        model_id="m1",
        kind=BenchmarkKind.CHAT,
        backend="ollama",
        tokens_per_second=42.0,
        parameters={"mode": "live"},
        success=success,
    )


def test_pressing_n_opens_launch_modal(temp_db) -> None:
    """`n` on the Benchmarks screen pushes the launch modal."""

    async def _run() -> None:
        with patch("llmctl.tui._data.get_benchmarks", return_value=[]), patch(
            "llmctl.tui._data.get_models", return_value=[_fake_model()]
        ):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("b")  # go to Benchmarks
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, BenchmarksScreen):
                        break
                assert isinstance(app.screen, BenchmarksScreen)
                await pilot.press("n")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, BenchmarkLaunchModal):
                        break
                assert isinstance(app.screen, BenchmarkLaunchModal)

    asyncio.run(_run())


def test_launch_modal_dispatches_run_with_chosen_kind(temp_db) -> None:
    """Submitting the launch modal calls run_benchmark with form values."""

    captured: dict[str, object] = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return _fake_result()

    async def _run() -> None:
        with patch("llmctl.tui._data.get_benchmarks", return_value=[]), patch(
            "llmctl.tui._data.get_models", return_value=[_fake_model()]
        ), patch("llmctl.tui._data.run_benchmark", side_effect=fake_run):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("b")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, BenchmarksScreen):
                        break
                # Drive the screen directly: the modal's button.press path
                # is exercised by the standalone build_launch test below;
                # here we verify the dispatcher wiring on the screen side.
                screen = app.screen
                assert isinstance(screen, BenchmarksScreen)
                launch = BenchmarkLaunch(
                    model_id="m1",
                    kind=BenchmarkKind.HEALTH,
                    name="probe",
                    max_tokens=256,
                    context_length=None,
                    dry_run=False,
                    require_live=True,
                )
                screen._on_launch_chosen(launch)
                # Worker runs off-thread; wait until our fake records the call.
                for _ in range(100):
                    await pilot.pause(0.05)
                    if captured:
                        break

    asyncio.run(_run())
    assert captured["model_id"] == "m1"
    assert captured["kind"] == BenchmarkKind.HEALTH
    assert captured["name"] == "probe"
    assert captured["dry_run"] is False
    assert captured["require_live"] is True


def test_delete_action_invokes_data_layer(temp_db) -> None:
    """Confirming the delete modal calls _data.delete_benchmark for the cursor row."""

    deleted: list[str] = []

    async def _run() -> None:
        with patch(
            "llmctl.tui._data.get_benchmarks", return_value=[_fake_result()]
        ), patch("llmctl.tui._data.get_models", return_value=[_fake_model()]), patch(
            "llmctl.tui._data.delete_benchmark",
            side_effect=lambda bench_id: deleted.append(bench_id) or True,
        ):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("b")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, BenchmarksScreen):
                        break
                screen = app.screen
                assert isinstance(screen, BenchmarksScreen)
                # Skip the modal — exercise the dispatcher path directly so the
                # test stays focused on the wiring under our control.
                screen.action_delete_benchmark()
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, DeleteModal):
                        break
                assert isinstance(app.screen, DeleteModal)
                app.screen.dismiss(ConfirmDelete(delete_files=False))
                for _ in range(100):
                    await pilot.pause(0.05)
                    if deleted:
                        break

    asyncio.run(_run())
    assert deleted == ["bench-1"]


def test_delete_action_on_empty_table_warns(temp_db) -> None:
    """Pressing `d` with no rows surfaces the same 'launch one first' hint."""

    notifications: list[str] = []

    async def _run() -> None:
        with patch("llmctl.tui._data.get_benchmarks", return_value=[]), patch(
            "llmctl.tui._data.get_models", return_value=[]
        ):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("b")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, BenchmarksScreen):
                        break
                assert isinstance(app.screen, BenchmarksScreen)
                with patch.object(
                    app,
                    "notify",
                    side_effect=lambda msg, **kw: notifications.append(msg),
                ):
                    await pilot.press("d")
                    for _ in range(20):
                        await pilot.pause(0.05)
                        if notifications:
                            break

    asyncio.run(_run())
    assert any("Press 'n'" in n for n in notifications), notifications


def test_empty_baseline_press_warns_with_next_step(temp_db) -> None:
    """Pressing `c` on an empty table tells the user to launch one first."""

    notifications: list[str] = []

    async def _run() -> None:
        with patch("llmctl.tui._data.get_benchmarks", return_value=[]), patch(
            "llmctl.tui._data.get_models", return_value=[]
        ):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("b")
                for _ in range(40):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, BenchmarksScreen):
                        break
                assert isinstance(app.screen, BenchmarksScreen)
                with patch.object(
                    app,
                    "notify",
                    side_effect=lambda msg, **kw: notifications.append(msg),
                ):
                    await pilot.press("c")
                    for _ in range(20):
                        await pilot.pause(0.05)
                        if notifications:
                            break

    asyncio.run(_run())
    assert any("Press 'n'" in n for n in notifications), notifications


# -- modal validation (unit-level, no full Textual app) --------------------


def test_launch_modal_rejects_non_positive_max_tokens(temp_db) -> None:
    """`_build_launch` raises on max_tokens <= 0."""

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            modal = BenchmarkLaunchModal([_fake_model()])
            await app.push_screen(modal)
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, BenchmarkLaunchModal):
                    break
            assert isinstance(app.screen, BenchmarkLaunchModal)
            from textual.widgets import Input  # local import for clarity

            app.screen.query_one("#bench-max-tokens", Input).value = "0"
            with pytest.raises(ValueError, match="max_tokens"):
                app.screen._build_launch()

    asyncio.run(_run())


def test_launch_modal_dry_run_when_mode_mock(temp_db) -> None:
    """Setting mode='mock' in the modal produces a dry_run=True payload."""

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            modal = BenchmarkLaunchModal([_fake_model()])
            await app.push_screen(modal)
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, BenchmarkLaunchModal):
                    break
            assert isinstance(app.screen, BenchmarkLaunchModal)
            from textual.widgets import Input  # local import for clarity

            app.screen.query_one("#bench-mode", Input).value = "mock"
            launch = app.screen._build_launch()
            assert launch.dry_run is True
            assert launch.kind == BenchmarkKind.CHAT

    asyncio.run(_run())
