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
