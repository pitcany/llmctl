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
from textual.coordinate import Coordinate
from textual.widgets import DataTable, Static

from llmctl._testing import isolate_tui
from llmctl.config import Settings
from llmctl.db import BenchmarkKind, RuntimeName
from llmctl.schemas import BenchmarkResult, Model
from llmctl.tui._modals_benchmarks import BenchmarkLaunch, BenchmarkLaunchModal
from llmctl.tui._modals_registry import ConfirmDelete, DeleteModal
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_benchmarks import BenchmarksScreen


@pytest.fixture(autouse=True)
def _isolated_tui(tmp_path, monkeypatch) -> None:
    """Keep this file off the developer's registry and off the network.

    Booting ``MissionControlApp`` fetches the dashboard, which opens the real
    database and probes every runtime. See ``llmctl._testing``.
    """
    isolate_tui(monkeypatch, tmp_path)


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


def test_estimated_token_counts_are_labeled(temp_db) -> None:
    """FIXPLAN §1B: throughput from estimated tokens is marked '~' and the
    hint explains the marker; precise (server usage) rows are unmarked."""

    estimated = _fake_result(name="est")
    estimated.parameters = {"mode": "live", "token_count_source": "estimate"}
    precise = _fake_result(name="exact")
    precise.parameters = {"mode": "live", "token_count_source": "server_usage"}

    async def _run() -> tuple[list[str], str]:
        with patch(
            "llmctl.tui._data.get_benchmarks", return_value=[estimated, precise]
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
                table = screen.query_one("#benchmarks-table", DataTable)
                tps_cells = [
                    str(table.get_cell_at(Coordinate(row, 5)))
                    for row in range(table.row_count)
                ]
                hint = screen.query_one("#benchmarks-hint", Static).content
                return tps_cells, hint

    tps_cells, hint = asyncio.run(_run())
    assert any(cell.endswith("~") for cell in tps_cells), tps_cells
    assert any(not cell.endswith("~") for cell in tps_cells), tps_cells
    assert "estimated" in hint.lower()


def test_incomparable_rows_render_nc_not_green(temp_db) -> None:
    """FIXPLAN §1D: a synthetic row never gets a green/red delta against a
    live baseline, and the explanation line says why."""

    baseline = _fake_result(name="base")
    baseline.id = "bench-base"
    baseline.parameters = {"mode": "live"}
    synthetic = _fake_result(name="synth")
    synthetic.id = "bench-synth"
    synthetic.parameters = {"mode": "mock"}
    synthetic.tokens_per_second = 999.0

    async def _run() -> tuple[list[str], str, str]:
        with patch(
            "llmctl.tui._data.get_benchmarks",
            return_value=[baseline, synthetic],
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
                screen._baseline_id = "bench-base"
                screen.refresh_data()
                for _ in range(40):
                    await pilot.pause(0.05)
                    if screen.query_one("#benchmarks-table", DataTable).row_count:
                        break
                table = screen.query_one("#benchmarks-table", DataTable)
                table.move_cursor(row=0)  # newest row first: the synthetic one
                await pilot.pause(0.05)
                delta_cells = [
                    str(table.get_cell_at(Coordinate(row, 6)))
                    for row in range(table.row_count)
                ]
                compare_line = screen.query_one("#benchmarks-compare", Static).content
                hint = screen.query_one("#benchmarks-hint", Static).content
                return delta_cells, str(compare_line), str(hint)

    delta_cells, compare_line, _hint = asyncio.run(_run())
    assert any("n/c" in cell for cell in delta_cells), delta_cells
    assert not any("999" in cell or "+8" in cell for cell in delta_cells), delta_cells
    assert "not comparable" in compare_line.lower(), compare_line


def test_enter_inspects_without_rerunning(temp_db) -> None:
    """§3D: Enter opens the inspection view; it must not launch a workload.
    Re-run is the separate, labeled 'r' action."""
    rerun_calls: list[str] = []

    async def _run() -> bool:
        with patch(
            "llmctl.tui._data.get_benchmarks", return_value=[_fake_result()]
        ), patch(
            "llmctl.tui._data.rerun_benchmark",
            side_effect=lambda bench_id: rerun_calls.append(bench_id),
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
                for _ in range(40):
                    await pilot.pause(0.05)
                    if screen.query_one("#benchmarks-table", DataTable).row_count:
                        break
                await pilot.press("enter")
                for _ in range(20):
                    await pilot.pause(0.05)
                    from llmctl.tui._modals_benchmarks import BenchmarkInspectModal

                    if isinstance(app.screen, BenchmarkInspectModal):
                        break
                opened = isinstance(app.screen, BenchmarkInspectModal)
                if opened:
                    await pilot.press("escape")
                    for _ in range(20):
                        await pilot.pause(0.05)
                        if isinstance(app.screen, BenchmarksScreen):
                            break
                return opened

    assert asyncio.run(_run()) is True
    assert rerun_calls == []  # Enter never re-ran anything


def test_inspect_lines_show_provenance_samples_and_drift() -> None:
    """§3D: the inspection text carries counts, provenance, drift and the
    comparison verdict."""
    result = _fake_result(name="deep")
    result.parameters = {
        "mode": "live",
        "request_count": 6,
        "failed_request_count": 1,
        "concurrency": 4,
        "achieved_concurrency": 4,
        "token_count_source": "estimate",
        "percentiles": {
            "latency_ms": {"n": 5, "p50": 100.0, "p90": 200.0, "p99": 300.0},
            "ttft_ms": {
                "n": 2,
                "p50": 10.0,
                "p90": 20.0,
                "p99": 25.0,
                "insufficient_samples": True,
            },
        },
        "warnings": ["underfilled workload: 4 request(s) cannot reach 8"],
        "rerun_of": "source-run",
        "drift": ["served.endpoint_url: http://a -> http://b"],
    }
    result.manifest = {
        "execution": "live",
        "workload": {"hash": "abcdef1234567890"},
        "served": {"endpoint_url": "http://b"},
        "serving_host": {"gpu_identity": ["RTX 5090"]},
    }
    baseline = _fake_result(name="base")
    baseline.id = "base-id"
    baseline.parameters = {"mode": "mock"}

    lines = BenchmarksScreen._inspect_lines(result, baseline)
    text = "\n".join(lines)
    assert "6 request(s), 1 failed" in text
    assert "estimated (no server usage)" in text
    assert "~" in text  # estimated throughput marker
    assert "n=5" in text and "few samples" in text  # ttft n=2 flagged
    assert "underfilled" in text
    assert "abcdef123456" in text
    assert "RTX 5090" in text
    assert "source-run" in text
    assert "drift" in text
    assert "not comparable" in text


def test_r_key_reruns_selected(temp_db) -> None:
    rerun_calls: list[str] = []

    async def _run() -> list[str]:
        with patch(
            "llmctl.tui._data.get_benchmarks", return_value=[_fake_result()]
        ), patch(
            "llmctl.tui._data.rerun_benchmark",
            side_effect=lambda bench_id: rerun_calls.append(bench_id)
            or _fake_result(name="again"),
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
                for _ in range(40):
                    await pilot.pause(0.05)
                    if screen.query_one("#benchmarks-table", DataTable).row_count:
                        break
                await pilot.press("r")
                for _ in range(100):
                    await pilot.pause(0.05)
                    if rerun_calls:
                        break
        return rerun_calls

    assert asyncio.run(_run()) == ["bench-1"]


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
                    await pilot.press("delete")
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
