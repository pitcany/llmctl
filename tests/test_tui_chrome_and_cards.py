"""Regression tests for the three TUI bugs reported 2026-05-31.

1. Dashboard stat cards rendered as empty boxes (no title visible until
   the first set_value call, sometimes never).
2. No keybinding hints visible — Header/Footer were yielded by the App
   compose but Textual's push_screen replaces the visible area, so the
   chrome disappeared the moment any screen was pushed.
3. ``vllm backend missing`` scheduler warning fired even when
   vllm-tp.service was actively serving on :8003 (services/backends.py
   was binary-on-PATH only, never probed the managed unit).
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
from collections.abc import Callable
from typing import Any

import pytest
from textual.widgets import Footer, Header

from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig, Settings
from llmctl.services.backends import detect_backends, missing_backends
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_dashboard import StatCard

# --- StatCard pre-populated rendering -----------------------------------------


def test_statcard_renders_title_before_set_value() -> None:
    """The card body must include the title at construction time.

    This is the regression for the "empty boxes" complaint. Before the
    fix the card was Static() with no renderable, so even when the
    grid laid out the card with a border, the inside was blank until
    set_value ran. If the worker thread errored, the title NEVER
    appeared. Now the title is in the renderable from __init__.
    """
    card = StatCard("Models", "card-models")
    rendered = card.content
    assert "Models" in rendered


def test_statcard_renders_placeholder_value_before_set_value() -> None:
    """A visible placeholder marks the card as 'waiting on data' rather than broken."""
    card = StatCard("GPUs", "card-gpus")
    rendered = card.content
    assert StatCard._PLACEHOLDER in rendered


def test_statcard_set_value_replaces_placeholder() -> None:
    """After set_value the renderable carries the live value."""
    card = StatCard("Models", "card-models")
    card.set_value("15")
    rendered = card.content
    assert "15" in rendered
    assert "Models" in rendered  # title preserved


# --- Each screen yields Header + Footer ---------------------------------------


@pytest.mark.parametrize(
    "screen_name",
    ["dashboard", "presets", "units", "models", "sessions", "gpus", "logs", "doctor", "benchmarks"],
)
def test_screen_yields_header_and_footer(screen_name: str) -> None:
    """Every installed screen must render its own Header + Footer.

    Textual's push_screen replaces the visible area, so widgets the App
    composes are hidden. Each screen owns its own chrome — verified by
    walking the screen's widget tree after install.
    """

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            # Switch to the named screen so it actually mounts
            getattr(app, f"action_show_{screen_name}")()
            for _ in range(40):
                await pilot.pause(0.02)
                # Wait for the screen to become active
                if app.screen.__class__.__name__.lower().startswith(screen_name):
                    break
            headers = list(app.screen.query(Header))
            footers = list(app.screen.query(Footer))
            assert len(headers) == 1, f"{screen_name}: expected 1 Header, got {len(headers)}"
            assert len(footers) == 1, f"{screen_name}: expected 1 Footer, got {len(footers)}"

    asyncio.run(_run())


def test_app_compose_yields_nothing() -> None:
    """App.compose is intentionally empty — chrome belongs to each screen.

    Yielding Header/Footer at app scope would (a) be hidden the moment
    a screen is pushed, and (b) cause duplicate-id errors when screens
    also yield their own. Pin the empty-body contract.
    """
    app = MissionControlApp()
    composed = list(app.compose())
    assert composed == []


def test_command_palette_rebound_off_ctrl_p() -> None:
    """The command palette is bound to ctrl+\\, not Textual's default ctrl+p.

    ctrl+p is widely intercepted before it reaches the app (VS Code's
    Quick Open, some terminals over SSH), so the "palette" footer entry
    silently did nothing. We rebind it to a chord terminals forward.
    Pin the remap so it can't regress back to the stock binding.
    """

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            palette_keys = [
                key
                for key, active in app.active_bindings.items()
                if active.binding.action == "command_palette"
            ]
            assert palette_keys == ["ctrl+backslash"]
            assert "ctrl+p" not in app.active_bindings

    asyncio.run(_run())


# --- vLLM backend availability via managed-unit HTTP probe --------------------


def _models_payload(ids: list[str]) -> bytes:
    return json.dumps({"object": "list", "data": [{"id": i} for i in ids]}).encode()


def _http_responder(by_port: dict[int, list[str]]) -> Callable[[str, float], Any]:
    """An http_get that returns scripted /v1/models payloads, fails otherwise."""

    def get(url: str, timeout: float) -> Any:
        port = int(url.split(":")[2].split("/")[0])
        if port not in by_port:
            raise urllib.error.URLError(f"port {port} unreachable")
        return io.BytesIO(_models_payload(by_port[port]))

    return get


def _settings_with_managed_units() -> Settings:
    s = Settings()
    s.managed_units = ManagedUnitsConfig(
        vllm_tp=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        vllm_coder=ManagedUnitConfig(unit_name="vllm-coder", default_port=8001),
        vllm_reasoner=ManagedUnitConfig(unit_name="vllm-reasoner", default_port=8002),
    )
    return s


def test_vllm_reported_available_when_managed_unit_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original bug: vllm-tp.service running on :8003, vllm CLI not on PATH,
    detect_backends incorrectly reported vllm as unavailable.

    With the fix the HTTP probe finds llama-3.3-70b on :8003 and vllm is OK.
    """
    monkeypatch.setattr("shutil.which", lambda _: None)
    rows = detect_backends(
        _settings_with_managed_units(),
        http_get=_http_responder(by_port={8003: ["llama-3.3-70b"]}),
    )
    vllm_row = next(r for r in rows if r["backend"] == "vllm")
    assert vllm_row["available"] is True
    assert "vllm-tp" in str(vllm_row["path"])


def test_vllm_reported_missing_when_no_managed_unit_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HTTP probe success + no binary = vllm legitimately unavailable."""
    monkeypatch.setattr("shutil.which", lambda _: None)
    rows = detect_backends(
        _settings_with_managed_units(),
        http_get=_http_responder(by_port={}),  # every port fails
    )
    vllm_row = next(r for r in rows if r["backend"] == "vllm")
    assert vllm_row["available"] is False


def test_vllm_uses_binary_path_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binary on PATH is still authoritative when present (no need to probe)."""
    monkeypatch.setattr("shutil.which", lambda b: "/usr/local/bin/vllm" if b == "vllm" else None)
    rows = detect_backends(
        _settings_with_managed_units(),
        http_get=_http_responder(by_port={}),  # would fail if probed
    )
    vllm_row = next(r for r in rows if r["backend"] == "vllm")
    assert vllm_row["available"] is True
    assert vllm_row["path"] == "/usr/local/bin/vllm"


def test_missing_backends_excludes_vllm_when_managed_unit_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """missing_backends drives the scheduler-warning text on the dashboard."""
    monkeypatch.setattr("shutil.which", lambda b: None)  # nothing on PATH
    missing = missing_backends(
        _settings_with_managed_units(),
        http_get=_http_responder(by_port={8003: ["llama-3.3-70b"]}),
    )
    # vllm should NOT be in missing because the managed unit IS serving
    assert "vllm" not in missing
    # Other binaries-only runtimes are still missing
    assert "llama_cpp" in missing
    assert "lmstudio" in missing


def test_vllm_empty_data_still_falls_back_to_binary_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A managed unit answering /v1/models with an empty data list isn't really
    'serving' — the probe should fall through to the binary check.

    This matches Phase A's behaviour in VLLMAdapter.health_check and keeps
    the two probes structurally identical."""
    monkeypatch.setattr("shutil.which", lambda _: None)

    def fake_get(url: str, timeout: float):
        return io.BytesIO(_models_payload([]))  # answers but empty

    rows = detect_backends(_settings_with_managed_units(), http_get=fake_get)
    vllm_row = next(r for r in rows if r["backend"] == "vllm")
    assert vllm_row["available"] is False
