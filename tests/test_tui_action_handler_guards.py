"""A failing database read from a keypress must not take the whole app down.

``DataScreen._refresh_worker`` wraps the refresh path in a broad ``except`` for
a documented reason: the registry SQLite file is shared with the CLI, the API
and the gateway, so a transient lock is routine, and ``run_worker`` defaults to
``exit_on_error=True``.

The *action* handlers bound to ``e``/``c``/``d``/``n``/``x`` sat outside that
guard and queried SQLite synchronously on the UI thread. One locked read while
the operator pressed a key and the app died with a traceback -- in the same
screens whose refresh path would have shown a toast and carried on.
"""

from __future__ import annotations

import asyncio

import pytest

from llmctl._testing import isolate_tui
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import Model, Profile
from llmctl.tui import _data
from llmctl.tui.app import MissionControlApp


@pytest.fixture(autouse=True)
def _isolated_tui(tmp_path, monkeypatch) -> None:
    isolate_tui(monkeypatch, tmp_path)


def _boom(*args, **kwargs):
    raise RuntimeError("database is locked")


async def _goto(app, pilot, action: str, prefix: str) -> None:
    getattr(app, action)()
    for _ in range(50):
        await pilot.pause(0.02)
        if app.screen.__class__.__name__.lower().startswith(prefix):
            return
    raise AssertionError(f"{prefix} screen did not become active")


def _seed(monkeypatch) -> None:
    """Give each screen one row, so a keypress has something to act on."""
    model = Model(
        id="m1",
        name="demo",
        runtime=RuntimeName.OLLAMA,
        source="demo:latest",
        status=ModelStatus.DISCOVERED,
    )
    profile = Profile(id="p1", name="demo-profile", runtime=RuntimeName.VLLM)
    monkeypatch.setattr(_data, "get_models", lambda: [model], raising=False)
    monkeypatch.setattr(_data, "get_profiles", lambda: [profile], raising=False)
    monkeypatch.setattr(_data, "get_missing_model_ids", lambda: ["m1"], raising=False)
    monkeypatch.setattr(_data, "get_backend_map", lambda: {"ollama": True}, raising=False)
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {}, raising=False)
    monkeypatch.setattr(_data, "get_missing_count", lambda: 1, raising=False)


def _run_keypress(monkeypatch, *, screen_action: str, prefix: str, key: str) -> list:
    """Open a populated screen, break the data layer, press ``key``."""
    notes: list[str] = []
    _seed(monkeypatch)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _goto(app, pilot, screen_action, prefix)
            # Let the initial refresh land so the table has a cursor row.
            for _ in range(30):
                await pilot.pause(0.02)
            # Break the data layer only *after* the screen has loaded, so the
            # failure belongs to the keypress and not to the initial refresh.
            monkeypatch.setattr(_data, "get_models", _boom, raising=False)
            monkeypatch.setattr(_data, "get_profiles", _boom, raising=False)
            monkeypatch.setattr(_data, "get_missing_model_ids", _boom, raising=False)

            await pilot.press(key)
            for _ in range(30):
                await pilot.pause(0.02)

            assert app.is_running, f"app died on '{key}' when the DB read failed"
            notes.extend(str(n.message) for n in app._notifications)

    asyncio.run(_run())
    return notes


@pytest.mark.parametrize(
    ("screen_action", "prefix", "key"),
    [
        ("action_show_models", "models", "e"),  # edit -> _selected_model
        ("action_show_models", "models", "c"),  # clone -> _selected_model
        ("action_show_models", "models", "d"),  # delete -> _selected_model
        ("action_show_models", "models", "x"),  # prune -> get_missing_model_ids
        ("action_show_profiles", "profiles", "e"),  # -> _selected_profile
        ("action_show_benchmarks", "benchmark", "n"),  # -> get_models
    ],
)
def test_keypress_survives_a_failing_query(
    monkeypatch, screen_action: str, prefix: str, key: str
) -> None:
    notes = _run_keypress(
        monkeypatch, screen_action=screen_action, prefix=prefix, key=key
    )
    assert any("database is locked" in n for n in notes), (
        f"'{key}' swallowed the failure instead of surfacing it: {notes}"
    )
