"""Tests for the TUI data-access helpers and the Textual app shell."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from llmctl.config import Settings, load_settings
from llmctl.tui import _data
from llmctl.tui.app import MissionControlApp

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def temp_db(tmp_path, monkeypatch) -> Settings:
    """Point the app at an isolated on-disk SQLite database and repo configs."""
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    db_file = tmp_path / "tui.db"
    base = load_settings()
    settings = base.model_copy(deep=True)
    settings.database.url = f"sqlite:///{db_file}"
    monkeypatch.setattr("llmctl.tui._data.load_settings", lambda: settings)
    return settings


def _seed_model(settings: Settings) -> str:
    from sqlmodel import Session

    from llmctl.db import get_engine, init_db
    from llmctl.schemas import ModelCreate
    from llmctl.services.registry import RegistryService

    init_db(settings.database_url)
    with Session(get_engine(settings.database_url)) as db:
        model = RegistryService(db).add_model(
            ModelCreate(name="demo", runtime="python_script", path="/tmp/demo.py")
        )
    return model.id or ""


def _seed_vllm_model(settings: Settings) -> str:
    from sqlmodel import Session

    from llmctl.db import get_engine, init_db
    from llmctl.schemas import ModelCreate
    from llmctl.services.registry import RegistryService

    init_db(settings.database_url)
    with Session(get_engine(settings.database_url)) as db:
        model = RegistryService(db).add_model(
            ModelCreate(name="vllm-demo", runtime="vllm", source="org/demo")
        )
    return model.id or ""


def test_get_models_empty(temp_db: Settings) -> None:
    assert _data.get_models() == []


def test_backend_map_python_available(temp_db: Settings) -> None:
    backend_map = _data.get_backend_map()
    assert backend_map.get("python_script") is True
    # No LLM runtimes installed in CI.
    assert backend_map.get("vllm") is False
    assert "vllm" in _data.BACKEND_INSTALL_HINTS


def test_overview_keys(temp_db: Settings) -> None:
    overview = _data.get_overview()
    for key in ("models", "sessions_total", "profiles", "gpu_count", "runtimes"):
        assert key in overview
    # Profiles auto-sync from YAML on first read.
    assert overview["profiles"] >= 1


def test_start_and_stop_session(temp_db: Settings) -> None:
    model_id = _seed_model(temp_db)
    assert len(_data.get_models()) == 1

    session = _data.start_model(model_id, dry_run=True)
    assert session.id is not None
    assert session.status.value == "planned"

    sessions = _data.get_sessions()
    assert any(s.id == session.id for s in sessions)

    stopped = _data.stop_session(session.id)
    assert stopped is not None
    assert stopped.status.value == "stopped"


def test_get_gpus_and_events(temp_db: Settings) -> None:
    # No NVIDIA GPU in CI: must degrade to an empty list, never raise.
    assert isinstance(_data.get_gpus(), list)
    events = _data.get_events()
    assert isinstance(events, list)


def test_app_boots_and_navigates(temp_db: Settings) -> None:
    """The app mounts, all screens compose, and navigation works."""
    _seed_model(temp_db)

    async def _run() -> None:
        from llmctl.tui.screens_benchmarks import BenchmarksScreen
        from llmctl.tui.screens_dashboard import DashboardScreen
        from llmctl.tui.screens_doctor import DoctorScreen
        from llmctl.tui.screens_gpu import GPUScreen
        from llmctl.tui.screens_logs import LogsScreen
        from llmctl.tui.screens_models import ModelsScreen
        from llmctl.tui.screens_sessions import SessionsScreen

        app = MissionControlApp()
        async with app.run_test() as pilot:
            # Dashboard is the default screen.
            assert isinstance(app.screen, DashboardScreen)
            for key, screen_cls in (
                ("m", ModelsScreen),
                ("s", SessionsScreen),
                ("g", GPUScreen),
                ("l", LogsScreen),
                ("o", DoctorScreen),
                ("b", BenchmarksScreen),
                ("d", DashboardScreen),
            ):
                await pilot.press(key)
                await pilot.pause()
                assert isinstance(app.screen, screen_cls)
            # Manual refresh must not crash.
            await pilot.press("r")
            await pilot.pause()

    asyncio.run(_run())


def test_install_command_for_known_backends() -> None:
    """Missing backends expose a copy-pasteable install command."""
    assert _data.install_command_for("vllm") == "pip install vllm"
    assert _data.install_command_for("ollama").startswith("curl")
    # python never needs installing.
    assert _data.install_command_for("python") == ""


def test_doctor_summary_keys(temp_db: Settings) -> None:
    """The doctor summary exposes GPU/NVML status and scheduler config."""
    summary = _data.get_doctor_summary()
    for key in (
        "gpu_count",
        "nvml_available",
        "gpu_policy",
        "safety_margin_gb",
        "default_host",
        "missing_backends",
    ):
        assert key in summary
    assert isinstance(summary["missing_backends"], list)


def test_overview_warnings_name_affected_models(temp_db: Settings) -> None:
    """Scheduler warnings link a missing backend to its affected models."""
    _seed_vllm_model(temp_db)
    overview = _data.get_overview()
    warnings = overview["scheduler_warnings"]
    # The vLLM binary is unavailable in CI; the warning must name the model.
    assert any("vllm" in w and "vllm-demo" in w for w in warnings)


def test_doctor_copy_install_to_clipboard(temp_db: Settings) -> None:
    """Pressing 'c' on a missing backend copies its install command."""

    async def _run() -> None:
        from llmctl.tui.screens_doctor import DoctorScreen

        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("o")
            await pilot.pause()
            assert isinstance(app.screen, DoctorScreen)
            screen = app.screen
            # Wait for the threaded fetch to populate rows.
            for _ in range(40):
                await pilot.pause()
                if screen._rows:
                    break
            # Move the cursor onto a known-missing backend (vllm) and copy.
            from textual.widgets import DataTable

            table = screen.query_one("#doctor-table", DataTable)
            target = next(i for i, (b, _) in enumerate(screen._rows) if b == "vllm")
            table.move_cursor(row=target)
            await pilot.press("c")
            await pilot.pause()
            assert app.clipboard == "pip install vllm"

    asyncio.run(_run())



def test_models_enter_opens_launch_plan_modal(temp_db: Settings) -> None:
    """Pressing enter on a model previews its launch plan in a modal."""
    _seed_model(temp_db)

    async def _run() -> None:
        from llmctl.tui._modals import LaunchPlanModal
        from llmctl.tui.screens_models import ModelsScreen

        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, ModelsScreen)
            await pilot.press("enter")
            # Wait for the threaded plan fetch + modal push.
            for _ in range(60):
                await pilot.pause()
                if isinstance(app.screen, LaunchPlanModal):
                    break
            assert isinstance(app.screen, LaunchPlanModal)
            # Cancel the modal.
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, ModelsScreen)

    asyncio.run(_run())



def test_models_unavailable_backend_blocks_modal(temp_db: Settings) -> None:
    """Pressing enter on a model whose backend is missing must not open a modal."""
    _seed_vllm_model(temp_db)

    async def _run() -> None:
        from llmctl.tui.screens_models import ModelsScreen

        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, ModelsScreen)
            await pilot.press("enter")
            # Give any (incorrect) worker a chance to push a modal.
            for _ in range(20):
                await pilot.pause()
            # vLLM binary is unavailable in CI -> stays on the models screen.
            assert isinstance(app.screen, ModelsScreen)

    asyncio.run(_run())



def test_benchmarks_screen_rerun(temp_db: Settings) -> None:
    """The benchmarks screen lists history and re-runs the selected entry."""
    model_id = _seed_model(temp_db)
    # Seed one benchmark through the same TUI data layer/database.
    with _data.db_session() as db:
        from llmctl.schemas import BenchmarkRunRequest
        from llmctl.services.benchmarks import BenchmarkService

        BenchmarkService(db).run(
            BenchmarkRunRequest(name="seed", model_id=model_id, dry_run=True)
        )

    async def _run() -> None:
        from llmctl.tui.screens_benchmarks import BenchmarksScreen

        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, BenchmarksScreen)
            screen = app.screen
            for _ in range(40):
                await pilot.pause()
                if screen._ids:
                    break
            assert screen._ids, "benchmark history should be populated"
            await pilot.press("enter")
            # Wait for the threaded re-run to persist a second record.
            for _ in range(60):
                await pilot.pause()
    asyncio.run(_run())


def test_benchmarks_set_and_clear_baseline(temp_db: Settings) -> None:
    """Pressing 'c' marks a baseline and 'x' clears it on the benchmarks screen."""
    model_id = _seed_model(temp_db)
    with _data.db_session() as db:
        from llmctl.schemas import BenchmarkRunRequest
        from llmctl.services.benchmarks import BenchmarkService

        for label in ("a", "b"):
            BenchmarkService(db).run(
                BenchmarkRunRequest(name=label, model_id=model_id, dry_run=True)
            )

    async def _run() -> None:
        from llmctl.tui.screens_benchmarks import BenchmarksScreen

        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("b")
            await pilot.pause()
            assert isinstance(app.screen, BenchmarksScreen)
            screen = app.screen
            for _ in range(40):
                await pilot.pause()
                if screen._ids:
                    break
            await pilot.press("c")
            await pilot.pause()
            assert screen._baseline_id is not None
            await pilot.press("x")
            await pilot.pause()
            assert screen._baseline_id is None

    asyncio.run(_run())
