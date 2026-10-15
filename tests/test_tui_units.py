"""TUI tests for the Managed Units screen (Phase C).

The screen probes systemctl for is-active state and HTTP /v1/models
on each unit's port. Both are injectable so tests don't hit real
systemd or real HTTP.
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llmctl._testing import isolate_tui
from llmctl.config import ManagedUnitConfig, Settings
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_units import UnitsScreen, _UnitRow


@pytest.fixture(autouse=True)
def _isolated_tui(tmp_path, monkeypatch) -> None:
    """Keep this file off the developer's registry and off the network.

    Booting ``MissionControlApp`` fetches the dashboard, which opens the real
    database and probes every runtime. See ``llmctl._testing``.
    """
    isolate_tui(monkeypatch, tmp_path)


@dataclass
class _FakeCompleted:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@pytest.fixture
def temp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Pin llmctl at a tmp data dir."""
    settings = Settings()
    settings.database.url = f"sqlite:///{tmp_path / 'db.sqlite3'}"
    monkeypatch.setattr("llmctl.tui.screens_units.load_settings", lambda: settings)
    monkeypatch.setattr("llmctl.tui._data.load_settings", lambda: settings)
    return settings


def _fake_systemctl(active_units: set[str]) -> SystemctlRunner:
    """A SystemctlRunner that reports unit_name in active_units as active."""

    def fake(argv: list[str]) -> _FakeCompleted:
        if argv[-2] == "is-active":
            return _FakeCompleted(
                stdout="active\n" if argv[-1] in active_units else "inactive\n"
            )
        return _FakeCompleted()

    return SystemctlRunner(runner=fake)


def _http_responder(by_port: dict[int, list[str]]) -> Callable[[str, float], Any]:
    """An http_get that returns /v1/models for the named ports, fails otherwise."""

    def get(url: str, timeout: float) -> Any:
        port = int(url.split(":")[2].split("/")[0])
        if port not in by_port:
            raise urllib.error.URLError(f"port {port} unreachable")
        payload = {"object": "list", "data": [{"id": i} for i in by_port[port]]}
        return io.BytesIO(json.dumps(payload).encode())

    return get


def test_fetch_returns_one_row_per_unit(temp_settings) -> None:
    """1 managed unit = 1 row in the default config."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=_http_responder(by_port={}),
    )
    rows: list[_UnitRow] = screen.fetch()
    assert len(rows) == 1
    roles = [r.role for r in rows]
    assert roles == ["vllm-engine"]


def test_fetch_marks_active_units_correctly(temp_settings) -> None:
    """is-active=active maps to the row's is_active=True."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units={"vllm-engine"}),
        http_get=_http_responder(by_port={}),
    )
    rows = screen.fetch()
    tp_row = next(r for r in rows if r.role == "vllm-engine" and r.role_kind == "managed")
    assert tp_row.is_active is True


def test_fetch_marks_inactive_units_correctly(temp_settings) -> None:
    """is-active=inactive maps to the row's is_active=False."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=_http_responder(by_port={}),
    )
    rows = screen.fetch()
    tp_row = next(r for r in rows if r.role == "vllm-engine" and r.role_kind == "managed")
    assert tp_row.is_active is False


def test_fetch_picks_up_served_names_via_probe(temp_settings) -> None:
    """A unit whose port answers /v1/models populates the served list."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units={"vllm-tp"}),
        http_get=_http_responder(by_port={8003: ["llama-3.3-70b"]}),
    )
    rows = screen.fetch()
    tp_row = next(r for r in rows if r.role == "vllm-engine" and r.role_kind == "managed")
    assert tp_row.served == ["llama-3.3-70b"]


def test_fetch_handles_unreachable_ports_gracefully(temp_settings) -> None:
    """Probe failure -> empty served list, not a crash."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=_http_responder(by_port={}),  # every port fails
    )
    rows = screen.fetch()
    for r in rows:
        # None, not []: nobody answered. `[]` is reserved for "answered with
        # nothing loaded", which is a different fact and renders differently.
        assert r.served is None


def test_fetch_handles_malformed_json_gracefully(temp_settings) -> None:
    """A garbage payload -> unreachable (None), no crash."""

    def fake_get(url: str, timeout: float):
        return io.BytesIO(b"<<not json>>")

    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=fake_get,
    )
    rows = screen.fetch()
    assert all(r.served is None for r in rows)


def test_fetch_handles_systemctl_oserror_gracefully(
    temp_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No systemd binary (container) -> is_active=None (unknown), not a crash.

    It was False, rendered "inactive" -- a claim nobody had checked
    (TUI-UX-FIXPLAN P0 #5).
    """

    class _RaisingRunner:
        # When systemd is unavailable every call fails, not just is_active --
        # scope resolution shells out too.
        def is_active(self, unit: str, *, user: bool = False) -> bool:
            raise OSError("systemctl not found")

        def is_user_unit(self, unit: str) -> bool:
            raise OSError("systemctl not found")

    screen = UnitsScreen(
        systemctl=_RaisingRunner(),  # type: ignore[arg-type]
        http_get=_http_responder(by_port={}),
    )
    rows = screen.fetch()
    assert all(r.is_active is None for r in rows)


def test_fetch_default_ports_match_config(temp_settings) -> None:
    """Default port (8003) comes from settings, not hardcoded."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=_http_responder(by_port={}),
    )
    rows = screen.fetch()
    ports = {(r.role, r.role_kind): r.port for r in rows}
    assert ports[("vllm-engine", "managed")] == 8003


def test_units_screen_keybinding_present() -> None:
    """``u`` is bound on the app."""
    keys = {b.key for b in MissionControlApp.BINDINGS}
    assert "u" in keys


def test_units_screen_installed_in_app(temp_settings) -> None:
    """action_show_units switches to the screen without raising."""

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            app.action_show_units()
            for _ in range(150):
                await pilot.pause(0.05)
                if isinstance(app.screen, UnitsScreen):
                    break
            assert isinstance(app.screen, UnitsScreen)

    asyncio.run(_run())


def _scoped_systemctl(user_units: set[str], active: set[str]) -> SystemctlRunner:
    """Runner where `user_units` are known only to the user manager."""

    names = user_units | active

    def fake(argv: list[str]) -> _FakeCompleted:
        in_user = "--user" in argv
        # `show` puts flags after the unit, so the unit is not argv[-1].
        unit = next((a for a in argv if a in names), "")
        if "show" in argv:
            known = in_user if unit in user_units else not in_user
            return _FakeCompleted(stdout=("loaded" if known else "not-found") + "\n")
        if "is-active" in argv:
            hit = unit in active and (in_user == (unit in user_units))
            return _FakeCompleted(stdout=("active" if hit else "inactive") + "\n")
        return _FakeCompleted()

    return SystemctlRunner(runner=fake)


def test_fetch_sees_a_running_user_scope_unit(temp_settings) -> None:
    """Regression: user units read `inactive` in system scope.

    `gpt-oss-120b` and `deepseek-v4-flash-0731` are user units. Before scope
    resolution this screen asked the *system* manager, which reports them
    inactive even while they serve — so they showed permanently down.
    """
    temp_settings.managed_units.units = {
        "gpt-oss": ManagedUnitConfig(
            unit_name="gpt-oss-120b", default_port=8005, runtime="llama_cpp"
        )
    }
    screen = UnitsScreen(
        systemctl=_scoped_systemctl(
            user_units={"gpt-oss-120b"}, active={"gpt-oss-120b"}
        ),
        http_get=_http_responder(by_port={}),
    )
    rows = {r.unit_name: r for r in screen.fetch()}
    assert rows["gpt-oss-120b"].is_active is True


def test_fetch_still_reads_system_units_in_system_scope(temp_settings) -> None:
    """The vllm-tp path must be unchanged by scope resolution."""
    screen = UnitsScreen(
        systemctl=_scoped_systemctl(user_units=set(), active={"vllm-engine"}),
        http_get=_http_responder(by_port={}),
    )
    rows = {r.unit_name: r for r in screen.fetch()}
    assert rows["vllm-engine"].is_active is True


def test_units_screen_exposes_start_and_stop_bindings() -> None:
    """The screen was read-only; start/stop must be reachable by key."""
    keys = {b[0] if isinstance(b, tuple) else b.key for b in UnitsScreen.BINDINGS}
    assert {"a", "x"} <= keys
    assert "s" not in keys, "`s` is the app-level Sessions binding; do not shadow it"


def test_units_bindings_do_not_shadow_app_navigation() -> None:
    """A screen binding wins over the app's, so it must not reuse those keys.

    Binding `s` here removed Sessions navigation from this screen without any
    error — the key simply did something else.
    """
    from llmctl.tui.app import MissionControlApp

    app_keys = {
        b.key for b in MissionControlApp.BINDINGS if hasattr(b, "key")
    }
    screen_keys = {b[0] if isinstance(b, tuple) else b.key for b in UnitsScreen.BINDINGS}
    assert not (screen_keys & app_keys), screen_keys & app_keys
