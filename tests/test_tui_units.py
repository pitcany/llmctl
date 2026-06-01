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

from llmctl.config import Settings
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_units import UnitsScreen, _UnitRow


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


def test_fetch_returns_one_row_per_unit_and_slot(temp_settings) -> None:
    """3 managed units + 2 slots = 5 rows in the default config."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=_http_responder(by_port={}),
    )
    rows: list[_UnitRow] = screen.fetch()
    assert len(rows) == 5
    roles = [r.role for r in rows]
    assert roles == [
        "vllm-tp", "vllm-coder", "vllm-reasoner",  # managed
        "coder", "reasoner",                       # slots
    ]


def test_fetch_marks_active_units_correctly(temp_settings) -> None:
    """is-active=active maps to the row's is_active=True."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units={"vllm-tp"}),
        http_get=_http_responder(by_port={}),
    )
    rows = screen.fetch()
    tp_row = next(r for r in rows if r.role == "vllm-tp" and r.role_kind == "managed")
    coder_row = next(r for r in rows if r.role == "vllm-coder" and r.role_kind == "managed")
    assert tp_row.is_active is True
    assert coder_row.is_active is False


def test_fetch_picks_up_served_names_via_probe(temp_settings) -> None:
    """A unit whose port answers /v1/models populates the served list."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units={"vllm-tp"}),
        http_get=_http_responder(by_port={8003: ["llama-3.3-70b"]}),
    )
    rows = screen.fetch()
    tp_row = next(r for r in rows if r.role == "vllm-tp" and r.role_kind == "managed")
    assert tp_row.served == ["llama-3.3-70b"]


def test_fetch_handles_unreachable_ports_gracefully(temp_settings) -> None:
    """Probe failure -> empty served list, not a crash."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=_http_responder(by_port={}),  # every port fails
    )
    rows = screen.fetch()
    for r in rows:
        assert r.served == []


def test_fetch_handles_malformed_json_gracefully(temp_settings) -> None:
    """A garbage payload -> [] served, no crash."""

    def fake_get(url: str, timeout: float):
        return io.BytesIO(b"<<not json>>")

    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=fake_get,
    )
    rows = screen.fetch()
    assert all(r.served == [] for r in rows)


def test_fetch_handles_systemctl_oserror_gracefully(
    temp_settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No systemd binary (container) -> is_active=False, not a crash."""

    class _RaisingRunner:
        def is_active(self, unit: str) -> bool:
            raise OSError("systemctl not found")

    screen = UnitsScreen(
        systemctl=_RaisingRunner(),  # type: ignore[arg-type]
        http_get=_http_responder(by_port={}),
    )
    rows = screen.fetch()
    assert all(r.is_active is False for r in rows)


def test_fetch_default_ports_match_config(temp_settings) -> None:
    """Default ports (8003/8001/8002) come from settings, not hardcoded."""
    screen = UnitsScreen(
        systemctl=_fake_systemctl(active_units=set()),
        http_get=_http_responder(by_port={}),
    )
    rows = screen.fetch()
    ports = {(r.role, r.role_kind): r.port for r in rows}
    assert ports[("vllm-tp", "managed")] == 8003
    assert ports[("vllm-coder", "managed")] == 8001
    assert ports[("vllm-reasoner", "managed")] == 8002
    assert ports[("coder", "slot")] == 8001
    assert ports[("reasoner", "slot")] == 8002


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
