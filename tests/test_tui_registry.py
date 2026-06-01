"""Tests for the TUI registry/profile data helpers + ProfilesScreen.

The CRUD helpers in ``llmctl.tui._data`` are exercised through small
round-trips so screen-side logic can trust them. The :class:`ProfilesScreen`
is mounted via ``app.run_test()`` to verify the new ``f`` keybinding and that
the screen composes without crashing.
"""

from __future__ import annotations

import asyncio
import urllib.error
from pathlib import Path

import pytest

from llmctl.config import Settings, load_settings
from llmctl.schemas import ModelCreate, ModelUpdate, ProfileCreate, ProfileUpdate
from llmctl.tui import _data
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_profiles import ProfilesScreen

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def _no_vllm_http_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(url: str, timeout: float):
        raise urllib.error.URLError("test: vllm probe disabled")

    monkeypatch.setattr("llmctl.services.backends._default_http_get", fail)


@pytest.fixture
def temp_db(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    db_file = tmp_path / "tui.db"
    base = load_settings()
    settings = base.model_copy(deep=True)
    settings.database.url = f"sqlite:///{db_file}"
    monkeypatch.setattr("llmctl.tui._data.load_settings", lambda: settings)
    return settings


def test_data_model_crud_round_trip(temp_db: Settings) -> None:
    added = _data.add_model(
        ModelCreate(name="tui-m", runtime="vllm", source="/srv/tui", path="/srv/tui")
    )
    assert added.id is not None

    updated = _data.update_model(added.id, ModelUpdate(notes="from tui"))
    assert updated is not None
    assert updated.notes == "from tui"

    cloned = _data.clone_model(added.id, "tui-m-clone")
    assert cloned is not None
    assert cloned.name == "tui-m-clone"

    assert _data.delete_model(cloned.id) is True
    assert _data.delete_model(added.id) is True


def test_data_profile_crud_round_trip(temp_db: Settings) -> None:
    created = _data.create_profile(
        ProfileCreate(
            name="tui-prof",
            runtime="vllm",
            tensor_parallel_size=1,
            max_model_len=4096,
        )
    )
    assert created.id is not None

    updated = _data.update_profile(
        created.id, ProfileUpdate(max_model_len=16384, dtype="bfloat16")
    )
    assert updated is not None
    assert updated.max_model_len == 16384
    assert updated.dtype == "bfloat16"

    cloned = _data.clone_profile(created.id, "tui-prof-clone")
    assert cloned is not None
    assert cloned.name == "tui-prof-clone"

    assert _data.delete_profile(cloned.id) is True
    assert _data.delete_profile(created.id) is True


def test_profiles_screen_keybinding_mounts(temp_db: Settings) -> None:
    """Pressing ``f`` reaches ProfilesScreen and composes without crashing."""
    # Initialize the DB before the app boots so concurrent worker threads
    # don't race each other on first CREATE TABLE (matches the pattern used
    # by ``test_app_boots_and_navigates`` in test_tui.py).
    _data.get_models()

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("f")
            await pilot.pause()
            assert isinstance(app.screen, ProfilesScreen)

    asyncio.run(_run())
