"""Drive the Units screen's start/stop keys through a real app + pilot.

The unit-level tests assert that `action_start_unit` calls the right thing.
These assert the *wiring*: that booting the app, navigating to Units, and
pressing the key actually reaches the action with the cursor row attached.
That is the layer where bindings, cursor tracking and modal plumbing break,
and none of it is exercised by calling the method directly.
"""

from __future__ import annotations

import asyncio

import pytest

from llmctl._testing import isolate_tui
from llmctl.config import ManagedUnitConfig, Settings
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.tui import _data
from llmctl.tui.app import MissionControlApp


@pytest.fixture(autouse=True)
def _isolated_tui(tmp_path, monkeypatch) -> None:
    isolate_tui(monkeypatch, tmp_path)


@pytest.fixture
def units_settings(tmp_path, monkeypatch) -> Settings:
    """One user-scope llama.cpp unit, so the cursor has something to act on."""
    settings = Settings()
    settings.database.url = f"sqlite:///{tmp_path / 'db.sqlite3'}"
    settings.managed_units.units = {
        "gpt-oss": ManagedUnitConfig(
            unit_name="gpt-oss-120b", default_port=8005, runtime="llama_cpp"
        )
    }
    monkeypatch.setattr("llmctl.tui.screens_units.load_settings", lambda: settings)
    monkeypatch.setattr("llmctl.tui._data.load_settings", lambda: settings)
    return settings


def _systemctl(active: set[str]) -> SystemctlRunner:
    """gpt-oss-120b is user-scope; vllm-tp is system-scope."""
    user_units = {"gpt-oss-120b"}
    names = {"gpt-oss-120b", "vllm-tp"}

    def fake(argv):
        import subprocess

        in_user = "--user" in argv
        unit = next((a for a in argv if a in names), "")
        if "show" in argv:
            known = in_user if unit in user_units else not in_user
            out = ("loaded" if known else "not-found") + "\n"
            return subprocess.CompletedProcess(argv, 0, out, "")
        if "is-active" in argv:
            hit = unit in active and (in_user == (unit in user_units))
            return subprocess.CompletedProcess(
                argv, 0, ("active" if hit else "inactive") + "\n", ""
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    return SystemctlRunner(runner=fake)


async def _goto_units(app, pilot) -> None:
    app.action_show_units()
    for _ in range(60):
        await pilot.pause(0.02)
        if app.screen.__class__.__name__ == "UnitsScreen":
            break
    else:
        raise AssertionError("Units screen never became active")
    # Let the initial probe land so the table has rows and a cursor.
    for _ in range(40):
        await pilot.pause(0.02)


def _drive(
    monkeypatch, units_settings, *, key: str, active: set[str], row: int = 1
) -> dict:
    """Boot the app, open Units, move to ``row``, press ``key``.

    ``row`` defaults to 1 — the cursor lands on row 0 (`vllm-tp`) and the
    unit under test is the second row. Getting this wrong is the whole
    reason these tests exist: the action acts on the *selected* row, which
    calling the method directly never exercises.
    """
    seen: dict = {"started": [], "stopped": [], "notes": []}

    monkeypatch.setattr(
        _data, "start_managed_unit",
        lambda u: (seen["started"].append(u), ("user", False))[1],
        raising=False,
    )
    monkeypatch.setattr(
        _data, "stop_managed_unit",
        lambda u: (seen["stopped"].append(u), "user")[1],
        raising=False,
    )
    monkeypatch.setattr(
        "llmctl.tui.screens_units.SystemctlRunner", lambda: _systemctl(active)
    )

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _goto_units(app, pilot)
            for _ in range(row):
                await pilot.press("down")
                await pilot.pause(0.02)
            await pilot.press(key)
            for _ in range(40):
                await pilot.pause(0.02)
            assert app.is_running, f"app died on '{key}'"
            seen["notes"] = [str(n.message) for n in app._notifications]
            seen["screen"] = app.screen.__class__.__name__

    asyncio.run(_run())
    return seen


def test_pressing_s_starts_the_selected_unit(monkeypatch, units_settings) -> None:
    """`s` on an inactive unit reaches start_managed_unit with *that* unit."""
    seen = _drive(monkeypatch, units_settings, key="s", active=set())
    assert "gpt-oss-120b" in seen["started"], seen["notes"]
    assert seen["stopped"] == []


def test_pressing_s_on_an_active_unit_refuses(monkeypatch, units_settings) -> None:
    """start is not a synonym for restart — it must not bounce a live unit."""
    seen = _drive(monkeypatch, units_settings, key="s", active={"gpt-oss-120b"})
    assert seen["started"] == []
    assert any("already active" in n for n in seen["notes"]), seen["notes"]


def test_pressing_x_on_an_active_unit_asks_first(monkeypatch, units_settings) -> None:
    """Stopping drops a live model, so it must confirm rather than act."""
    seen = _drive(monkeypatch, units_settings, key="x", active={"gpt-oss-120b"})
    assert seen["stopped"] == [], "stop fired without confirmation"
    assert seen["screen"] == "DeleteModal", seen["screen"]


def test_pressing_x_on_an_inactive_unit_refuses(monkeypatch, units_settings) -> None:
    seen = _drive(monkeypatch, units_settings, key="x", active=set())
    assert seen["stopped"] == []
    assert any("not active" in n for n in seen["notes"]), seen["notes"]


def test_the_action_follows_the_cursor_not_the_first_row(
    monkeypatch, units_settings
) -> None:
    """Row 0 is vllm-tp; row 1 is gpt-oss. The key must act on the cursor."""
    top = _drive(monkeypatch, units_settings, key="s", active=set(), row=0)
    assert top["started"] == ["vllm-tp"], top["notes"]

    second = _drive(monkeypatch, units_settings, key="s", active=set(), row=1)
    assert second["started"] == ["gpt-oss-120b"], second["notes"]


def test_stop_modal_says_stop_not_delete(monkeypatch, units_settings) -> None:
    """Reusing DeleteModal must not ask "Delete unit ...?" before a stop.

    Asserting the modal *appears* is not enough — the first live run showed a
    red "Delete" button and "Delete unit 'gpt-oss-120b'?", which reads as
    though the unit file is about to be removed.
    """
    captured: dict = {}

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _goto_units(app, pilot)
            await pilot.press("down")
            await pilot.pause(0.02)
            await pilot.press("x")
            for _ in range(40):
                await pilot.pause(0.02)
            screen = app.screen
            captured["name"] = screen.__class__.__name__
            captured["title"] = str(screen.query_one("#delete-title").content)
            captured["button"] = str(screen.query_one("#delete-confirm").label)

    monkeypatch.setattr(
        "llmctl.tui.screens_units.SystemctlRunner",
        lambda: _systemctl({"gpt-oss-120b"}),
    )
    asyncio.run(_run())

    assert captured["name"] == "DeleteModal"
    assert "Stop" in captured["title"], captured["title"]
    assert "Delete" not in captured["title"], captured["title"]
    assert captured["button"] == "Stop", captured["button"]
