"""Plugin screens in `llmctl tui`.

A host-specific plugin (a macOS llama-swap one, say) has its own dashboard. It
joins the TUI as one more screen: its own key, a line in the help and the
command palette, and, when the plugin says so, the screen the TUI opens on.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from textual.screen import Screen
from textual.widgets import Static

from llmctl import plugins
from llmctl._testing import isolate_tui
from llmctl.plugins import PluginRegistry, TuiScreen, load_plugins
from llmctl.tui.app import MissionControlApp, help_text


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch) -> Any:
    isolate_tui(monkeypatch, tmp_path)
    monkeypatch.delenv("LLMCTL_PLUGINS", raising=False)
    plugins.reset()
    yield
    plugins.reset()


class _EntryPoint:
    def __init__(self, register: Any) -> None:
        self.name = "host"
        self._register = register

    def load(self) -> Any:
        return self._register


class HostScreen(Screen):
    """A plugin's dashboard."""

    def compose(self):
        yield Static("host lanes")


def _install(*screens: TuiScreen) -> None:
    def register(reg: PluginRegistry) -> None:
        reg.tui_screens.extend(screens)

    load_plugins(eps=[_EntryPoint(register)])


def _host(**overrides: Any) -> TuiScreen:
    spec = {
        "name": "host",
        "title": "Host",
        "help": "This machine's lanes",
        "key": "h",
        "factory": HostScreen,
    }
    return TuiScreen(**{**spec, **overrides})


def _run(keys: tuple[str, ...] = ()) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def go() -> None:
        app = MissionControlApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            seen["first"] = type(app.screen).__name__
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
            seen["last"] = type(app.screen).__name__
            seen["help"] = help_text(app.screen, app.plugin_screen_rows())
            seen["palette"] = [c.title for c in app.get_system_commands(app.screen)]
            seen["notes"] = [n.message for n in app._notifications]

    asyncio.run(go())
    return seen


def test_a_plugin_screen_opens_with_its_key() -> None:
    _install(_host())
    seen = _run(("h",))
    assert seen["first"] == "DashboardScreen"
    assert seen["last"] == "HostScreen"


def test_a_start_screen_is_what_the_tui_opens_on() -> None:
    _install(_host(start=True))
    seen = _run(("d",))
    assert seen["first"] == "HostScreen"
    assert seen["last"] == "DashboardScreen"  # the core screens are a key away


def test_a_plugin_screen_is_in_the_help_and_the_palette() -> None:
    _install(_host())
    seen = _run()
    assert "Host" in seen["help"] and "This machine's lanes" in seen["help"]
    assert "Go to Host" in seen["palette"]


def test_a_plugin_screen_on_a_core_key_is_skipped_and_said() -> None:
    _install(_host(key="d"))
    seen = _run(("d",))
    assert seen["last"] == "DashboardScreen"
    assert any("Host" in note and "'d'" in note for note in seen["notes"])


def test_a_plugin_screen_whose_factory_fails_is_skipped_and_said() -> None:
    def broken() -> Screen:
        raise RuntimeError("no checkout")

    _install(_host(factory=broken, start=True))
    seen = _run()
    assert seen["first"] == "DashboardScreen"
    assert any("no checkout" in note for note in seen["notes"])


def test_without_plugins_the_tui_is_unchanged() -> None:
    seen = _run()
    assert seen["first"] == "DashboardScreen"
    assert "Host" not in seen["help"]
