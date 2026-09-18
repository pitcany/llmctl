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


# --- P0 #2: the command palette lists screens ---------------------------------


def test_palette_offers_go_to_dashboard(tmp_path, monkeypatch) -> None:
    from llmctl._testing import isolate_tui
    from llmctl.tui.app import MissionControlApp

    isolate_tui(monkeypatch, tmp_path)
    titles: list[str] = []

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            titles.extend(c.title for c in app.get_system_commands(app.screen))

    asyncio.run(_run())
    assert "Go to Dashboard" in titles
    assert sum(t.startswith("Go to ") for t in titles) == 10


# --- P0 #4: `d` is Dashboard everywhere --------------------------------------


def test_d_goes_to_dashboard_from_every_crud_screen(tmp_path, monkeypatch) -> None:
    from llmctl._testing import isolate_tui
    from llmctl.tui.app import MissionControlApp
    from llmctl.tui.screens_dashboard import DashboardScreen

    isolate_tui(monkeypatch, tmp_path)
    landed: list[bool] = []

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            for key in ("m", "p", "f", "b"):
                await pilot.press(key)
                await pilot.pause()
                await pilot.press("d")
                await pilot.pause()
                landed.append(isinstance(app.screen, DashboardScreen))

    asyncio.run(_run())
    assert landed == [True, True, True, True]


# --- P0 #3: `x` on an adopted session never offers to terminate it -----------


def _adopted_session():
    from llmctl.db import RuntimeName, SessionKind, SessionStatus
    from llmctl.schemas import Session

    return Session(
        id="adopted-1234567",
        model_id=None,
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
        kind=SessionKind.ADOPTED,
        systemd_unit="vllm-tp.service",
        endpoint_url="http://127.0.0.1:8003",
    )


def test_stop_on_adopted_session_redirects_to_units(tmp_path, monkeypatch) -> None:
    from llmctl._testing import isolate_tui
    from llmctl.tui import _data
    from llmctl.tui._modals import ConfirmActionModal
    from llmctl.tui.app import MissionControlApp

    isolate_tui(monkeypatch, tmp_path)
    monkeypatch.setattr(_data, "get_sessions", lambda: [_adopted_session()])
    monkeypatch.setattr(_data, "tail_log", lambda _id: "")
    seen: dict = {}

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("s")
            for _ in range(20):
                await pilot.pause(0.05)
                if app.screen._ids:  # type: ignore[attr-defined]
                    break
            await pilot.press("x")
            await pilot.pause()
            seen["modal"] = isinstance(app.screen, ConfirmActionModal)
            seen["notes"] = [str(n.message) for n in app._notifications]

    asyncio.run(_run())
    assert not seen["modal"]
    assert any("Units screen" in n and "vllm-tp" in n for n in seen["notes"]), seen["notes"]


# --- P0 #5: unknown systemd state is shown as unknown -------------------------


def test_units_render_unknown_when_systemd_unavailable() -> None:
    from textual.widgets import DataTable

    from llmctl.tui.screens_units import UnitsScreen, _UnitRow

    row = _UnitRow(
        role="vllm-tp", unit_name="vllm-tp", env_path="/x", env_exists=False,
        port=8003, is_active=None, served=None, role_kind="managed",
    )
    cells: dict = {}

    screen = UnitsScreen()
    screen.fetch = lambda: [row]  # type: ignore[method-assign]  # no real systemd

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(screen)

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            table = app.screen.query_one("#units-table", DataTable)
            cells["active"] = str(table.get_row_at(0)[2])

    asyncio.run(_run())
    assert "unknown" in cells["active"]


# --- P0 #6: no brew commands off macOS ----------------------------------------


def test_install_commands_are_platform_aware() -> None:
    from llmctl.tui import _data

    assert _data.install_command_for("llama_cpp", system="Linux") == ""
    assert _data.install_command_for("lmstudio", system="Linux") == ""
    assert _data.install_command_for("llama_cpp", system="Darwin").startswith("brew")
    assert _data.install_command_for("ollama", system="Linux").startswith("curl")


# --- P1 #8: doctor summary in words --------------------------------------------


def test_doctor_summary_has_no_raw_booleans() -> None:
    from llmctl.tui.screens_doctor import DoctorScreen

    rendered: dict = {}
    screen = DoctorScreen()
    screen.fetch = lambda: {  # type: ignore[method-assign]
        "backends": [],
        "summary": {
            "gpu_count": 0, "nvml_available": False, "safe_mode": True,
            "gpu_policy": "balanced", "safety_margin_gb": 2, "allow_public_bind": False,
            "default_host": "127.0.0.1", "missing_backends": [],
        },
    }

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(screen)

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            rendered["text"] = str(app.screen.query_one("#doctor-summary", Static).render())

    asyncio.run(_run())
    text = rendered["text"]
    assert "True" not in text and "False" not in text, text
    assert "CPU-only" in text and "not detected" in text


# --- P1 #9: unit journals are reachable; "Logs" is honestly "Events" -----------


def test_units_j_opens_the_unit_journal(monkeypatch) -> None:
    from llmctl.tui import _data
    from llmctl.tui._modals import TextViewModal
    from llmctl.tui.screens_units import UnitsScreen, _UnitRow

    row = _UnitRow(
        role="vllm-tp", unit_name="vllm-tp", env_path="/x", env_exists=False,
        port=8003, is_active=True, served=["m"], role_kind="managed",
    )
    monkeypatch.setattr(_data, "unit_log_tail", lambda unit, lines=60: f"[rank0]: {unit} up")
    screen = UnitsScreen()
    screen.fetch = lambda: [row]  # type: ignore[method-assign]
    seen: dict = {}

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(screen)

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("j")
            for _ in range(20):
                await pilot.pause(0.05)
                if isinstance(app.screen, TextViewModal):
                    break
            seen["modal"] = isinstance(app.screen, TextViewModal)
            if seen["modal"]:
                seen["body"] = str(app.screen.query_one("#textview-body", Static).render())

    asyncio.run(_run())
    assert seen["modal"]
    assert "[rank0]: vllm-tp up" in seen["body"]  # brackets shown literally


# --- P1 #10: dashboard first-run guidance ------------------------------------


def test_dashboard_guides_a_first_run(tmp_path, monkeypatch) -> None:
    from llmctl._testing import isolate_tui
    from llmctl.tui.app import MissionControlApp

    isolate_tui(monkeypatch, tmp_path)
    seen: dict = {}

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            for _ in range(40):
                await pilot.pause(0.05)
                text = str(app.screen.query_one("#dashboard-runtimes", Static).render())
                if "Runtime health" in text:
                    break
            seen["text"] = text

    asyncio.run(_run())
    assert "No models registered yet" in seen["text"]
    assert "no registered models affected" not in seen["text"]


# --- P1 #11: a reconcile flip is announced ------------------------------------


def test_sessions_announce_a_status_flip(tmp_path, monkeypatch) -> None:
    from llmctl._testing import isolate_tui
    from llmctl.db import SessionStatus
    from llmctl.tui import _data
    from llmctl.tui.app import MissionControlApp

    isolate_tui(monkeypatch, tmp_path)
    state = {"status": SessionStatus.RUNNING}

    def sessions():
        s = _adopted_session()
        return [s.model_copy(update={"status": state["status"]})]

    monkeypatch.setattr(_data, "get_sessions", sessions)
    monkeypatch.setattr(_data, "tail_log", lambda _id: "")
    notes: list[str] = []

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("s")
            for _ in range(20):
                await pilot.pause(0.05)
                if app.screen._last_status:  # type: ignore[attr-defined]
                    break
            state["status"] = SessionStatus.STOPPED
            app.screen.refresh_data()  # type: ignore[attr-defined]
            for _ in range(20):
                await pilot.pause(0.05)
                notes[:] = [str(n.message) for n in app._notifications]
                if any("->" in n for n in notes):
                    break

    asyncio.run(_run())
    assert any("running -> stopped" in n for n in notes), notes


# --- P1 #12: launch effects come from settings, and are honest on failure -----


def test_launch_effects_follow_settings_and_options() -> None:
    from llmctl.config import FleetUnitsConfig, ManagedUnitConfig
    from llmctl.services.vllm_orchestrator import OrchestratorOptions
    from llmctl.tui._modals_presets import describe_launch_effects

    unit = ManagedUnitConfig(unit_name="vllm-alt", env_file_path="/srv/alt.env")
    fleet = FleetUnitsConfig(ollama="ollama-x", tp="vllm-alt")
    steps = describe_launch_effects(
        unit, fleet, OrchestratorOptions(enable_hermes_verify=False, timeout_s=600)
    )
    text = " | ".join(steps)
    assert "ollama-x" in text and "/srv/alt.env" in text and "vllm-alt" in text
    assert "10 min" in text
    assert "Hermes" not in text


def test_missing_link_names_the_dangling_model_ref() -> None:
    from pathlib import Path

    from llmctl.services.preset_loader import PresetView
    from llmctl.tui.screens_presets import _format_link_cell

    view = PresetView(
        alias="a", served_name="a", model_id="m", family=None, param_count_b=None,
        tensor_parallel=2, quantization="awq", source_path=Path("/x"),
        linkage_state="missing", model_ref="gone-model-id",
    )
    assert "gone-model-id" in _format_link_cell(view)


def test_edit_asks_before_suspending_into_the_editor(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from llmctl._testing import isolate_tui
    from llmctl.services.preset_loader import PresetView
    from llmctl.tui import _data
    from llmctl.tui._modals import ConfirmActionModal
    from llmctl.tui.app import MissionControlApp

    isolate_tui(monkeypatch, tmp_path)
    preset = tmp_path / "a.yaml"
    preset.write_text("alias: a\n")
    view = PresetView(
        alias="a", served_name="a", model_id="m", family=None, param_count_b=None,
        tensor_parallel=2, quantization="awq", source_path=Path(preset),
    )
    monkeypatch.setattr(_data, "get_preset_views_with_links", lambda: [view])
    ran: list[Path] = []
    monkeypatch.setattr(_data, "run_editor_on_preset", lambda p: ran.append(p))
    seen: dict = {}

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("p")
            for _ in range(20):
                await pilot.pause(0.05)
                if app.screen._row_aliases:  # type: ignore[attr-defined]
                    break
            await pilot.press("e")
            await pilot.pause()
            seen["asked"] = isinstance(app.screen, ConfirmActionModal)
            seen["ran_before_confirm"] = bool(ran)

    asyncio.run(_run())
    assert seen["asked"] and not seen["ran_before_confirm"]


# --- P1 #13 / P3 #21: GPU rows ---------------------------------------------------


def test_gpu_cells_use_units_and_dash_for_unknowns() -> None:
    from llmctl.schemas import GPUInfo
    from llmctl.tui.screens_gpu import gpu_row_cells

    full = GPUInfo(
        index=0, name="RTX 5090", memory_used_mb=12288, memory_total_mb=32768,
        utilization_gpu_percent=40, temperature_c=55, power_draw_watts=220.4,
    )
    cells = gpu_row_cells(full)
    assert cells[2] == "12.0 / 32.0 GiB" and cells[4] == "55 °C" and cells[5] == "220 W"

    empty = GPUInfo(index=1, name="?")
    assert gpu_row_cells(empty)[2:6] == ("—", "—", "—", "—")


# --- P1 #14: Units can restart a unit, behind a confirmation ------------------


def test_units_restart_is_confirmed_then_runs(monkeypatch) -> None:
    from llmctl.tui import _data
    from llmctl.tui._modals_registry import DeleteModal
    from llmctl.tui.screens_units import UnitsScreen, _UnitRow

    row = _UnitRow(
        role="vllm-tp", unit_name="vllm-tp", env_path="/x", env_exists=True,
        port=8003, is_active=True, served=["m"], role_kind="managed",
    )
    restarted: list[str] = []
    monkeypatch.setattr(_data, "restart_managed_unit", lambda u: restarted.append(u) or "system")
    screen = UnitsScreen()
    screen.fetch = lambda: [row]  # type: ignore[method-assign]
    seen: dict = {}

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(screen)

    async def _run() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            await pilot.press("R")
            await pilot.pause()
            seen["modal"] = isinstance(app.screen, DeleteModal)
            seen["before"] = list(restarted)
            app.screen.query_one("#delete-confirm", Button).press()
            for _ in range(20):
                await pilot.pause(0.05)
                if restarted:
                    break

    asyncio.run(_run())
    assert seen["modal"] and seen["before"] == []
    assert restarted == ["vllm-tp"]


# --- P2 #16 / #18: help modal, ctrl+c hint --------------------------------------


def test_help_modal_lists_global_and_screen_keys(tmp_path, monkeypatch) -> None:
    from llmctl._testing import isolate_tui
    from llmctl.tui._modals import TextViewModal
    from llmctl.tui.app import MissionControlApp

    isolate_tui(monkeypatch, tmp_path)
    seen: dict = {}

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("u")
            await pilot.pause()
            await pilot.press("question_mark")
            await pilot.pause()
            seen["modal"] = isinstance(app.screen, TextViewModal)
            if seen["modal"]:
                seen["text"] = str(app.screen.query_one("#textview-body", Static).render())
            await pilot.press("escape")
            await pilot.pause()
            await pilot.press("ctrl+c")
            await pilot.pause()
            seen["alive"] = app.is_running
            seen["notes"] = [str(n.message) for n in app._notifications]

    asyncio.run(_run())
    assert seen["modal"]
    assert "Dashboard" in seen["text"] and "Journal" in seen["text"]  # global + Units
    assert seen["alive"]
    assert any("q to quit" in n for n in seen["notes"]), seen["notes"]


# --- P2 #17: enter actions are visible in the footer ------------------------


def test_enter_action_is_the_screens_not_the_tables(tmp_path, monkeypatch) -> None:
    from llmctl._testing import isolate_tui
    from llmctl.tui.app import MissionControlApp

    isolate_tui(monkeypatch, tmp_path)
    found: dict = {}

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            for key in ("p", "m", "b"):
                await pilot.press(key)
                await pilot.pause()
                active = app.screen.active_bindings.get("enter")
                found[key] = (active.binding.description, active.binding.show) if active else None

    asyncio.run(_run())
    assert found == {
        "p": ("Launch", True), "m": ("Plan/Launch", True), "b": ("Re-run", True)
    }


# --- P2 #20: Models confirms mutations -----------------------------------------


def test_models_after_callback_toasts_success_and_not_found() -> None:
    from llmctl.tui.screens_models import ModelsScreen

    notes: list[tuple[str, str | None]] = []
    screen = ModelsScreen()

    class _App:
        def notify(self, msg, **kw):  # type: ignore[no-untyped-def]
            notes.append((str(msg), kw.get("severity")))

    screen.refresh_data = lambda: None  # type: ignore[method-assign]
    import unittest.mock as um

    with um.patch.object(type(screen), "app", new_callable=um.PropertyMock, return_value=_App()):
        screen._after("Added model m.")(object())
        screen._after("Deleted model m.")(False)
    assert notes[0] == ("Added model m.", None)
    assert notes[1][1] == "warning"
