"""TUI tests for the Presets screen (Phase B).

Exercises the daily-driver flow: open the TUI, press `p`, see the
preset table, press enter on a row, pick a target from the modal.
The orchestrator is stubbed so no real systemd activity happens.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from llmctl.config import Settings
from llmctl.services.preset_loader import PresetView
from llmctl.tui._modals_presets import PresetLaunchModal, PresetLaunchTarget
from llmctl.tui.app import MissionControlApp
from llmctl.tui.screens_presets import PresetsScreen


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Pin llmctl at a tmp data dir to avoid touching the user's DB."""
    db_path = tmp_path / "db.sqlite3"
    settings = Settings()
    settings.database.url = f"sqlite:///{db_path}"
    monkeypatch.setattr("llmctl.tui._data.load_settings", lambda: settings)
    monkeypatch.setattr("llmctl.tui.screens_presets.load_settings", lambda: settings)
    return settings


@pytest.fixture
def fake_views() -> list[PresetView]:
    """Three views — one TP-capable preset, one slot-capable, one tiny."""
    return [
        PresetView(
            alias="llama-3.3-70b",
            served_name="llama-3.3-70b",
            model_id="casperhansen/llama-3.3-70b-instruct-awq",
            family="llama",
            param_count_b=70.0,
            tensor_parallel=2,
            quantization="awq",
            source_path=Path("/tmp/llama.yaml"),
        ),
        PresetView(
            alias="qwen2.5-coder-32b",
            served_name="qwen2.5-coder-32b",
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            family="qwen",
            param_count_b=32.0,
            tensor_parallel=2,
            quantization="awq",
            source_path=Path("/tmp/qwen.yaml"),
        ),
    ]


def _patch_preset_loader(views: list[PresetView]):
    """Patch screens_presets.load_preset_views to return our fixtures."""
    return patch("llmctl.tui.screens_presets.load_preset_views", return_value=views)


def test_presets_screen_renders_rows(temp_db, fake_views) -> None:
    """Loading the screen shows one row per preset."""

    async def _run() -> None:
        with _patch_preset_loader(fake_views):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("p")
                # Wait for the threaded fetch to complete + render.
                for _ in range(150):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, PresetsScreen):
                        screen = app.screen
                        if len(screen._row_aliases) >= 2:
                            break
                assert isinstance(app.screen, PresetsScreen)
                assert set(app.screen._row_aliases) == {
                    "llama-3.3-70b",
                    "qwen2.5-coder-32b",
                }

    asyncio.run(_run())


def test_presets_screen_handles_empty_preset_dir(temp_db) -> None:
    """Empty preset dir -> single placeholder row, no crash."""

    async def _run() -> None:
        with _patch_preset_loader([]):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("p")
                for _ in range(150):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, PresetsScreen):
                        break
                assert isinstance(app.screen, PresetsScreen)
                assert app.screen._row_aliases == []

    asyncio.run(_run())


def test_enter_on_row_opens_launch_modal(temp_db, fake_views) -> None:
    """Pressing enter on a preset row pops the target picker."""

    async def _attempt() -> bool:
        with _patch_preset_loader(fake_views):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("p")
                for _ in range(150):
                    await pilot.pause(0.05)
                    if (
                        isinstance(app.screen, PresetsScreen)
                        and len(app.screen._row_aliases) >= 2
                    ):
                        break
                if not isinstance(app.screen, PresetsScreen):
                    return False
                await pilot.press("enter")
                for _ in range(150):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, PresetLaunchModal):
                        break
                if not isinstance(app.screen, PresetLaunchModal):
                    return False
                await pilot.press("escape")
                await pilot.pause()
                return isinstance(app.screen, PresetsScreen)

    # Retry pattern matches the existing test_models_enter_opens_launch_plan_modal
    # — Textual worker tail-latency is noisy under suite load.
    for _ in range(3):
        if asyncio.run(_attempt()):
            return
    raise AssertionError("launch modal did not appear within budget")


def test_picker_modal_returns_chosen_target(fake_views) -> None:
    """The picker modal's keyboard shortcuts produce the right enum."""

    captured: list[PresetLaunchTarget | None] = []

    async def _run() -> None:
        from textual.app import App
        from textual.widgets import Static

        class _Host(App[None]):
            def compose(self):
                yield Static("host")

            def on_mount(self):
                self.push_screen(PresetLaunchModal(fake_views[0]), captured.append)

        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(50):
                await pilot.pause(0.02)
                if isinstance(app.screen, PresetLaunchModal):
                    break
            await pilot.press("c")  # coder shortcut
            for _ in range(50):
                await pilot.pause(0.02)
                if captured:
                    break

    asyncio.run(_run())
    assert captured == [PresetLaunchTarget.CODER]


def test_picker_modal_escape_yields_none(fake_views) -> None:
    """Escape closes the modal with None (no launch)."""

    captured: list[PresetLaunchTarget | None] = []

    async def _run() -> None:
        from textual.app import App
        from textual.widgets import Static

        class _Host(App[None]):
            def compose(self):
                yield Static("host")

            def on_mount(self):
                self.push_screen(PresetLaunchModal(fake_views[0]), captured.append)

        app = _Host()
        async with app.run_test() as pilot:
            for _ in range(50):
                await pilot.pause(0.02)
                if isinstance(app.screen, PresetLaunchModal):
                    break
            await pilot.press("escape")
            for _ in range(50):
                await pilot.pause(0.02)
                if captured:
                    break

    asyncio.run(_run())
    assert captured == [None]


def test_presets_screen_keybinding_present() -> None:
    """``p`` is bound on the app for the Presets screen."""
    keys = {b.key for b in MissionControlApp.BINDINGS}
    assert "p" in keys


def test_presets_install_in_app() -> None:
    """The app installs a 'presets' screen so push_screen('presets') works."""

    async def _run() -> None:
        with _patch_preset_loader([]):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                # action_show_presets should switch without raising
                app.action_show_presets()
                for _ in range(50):
                    await pilot.pause(0.02)
                    if isinstance(app.screen, PresetsScreen):
                        break
                assert isinstance(app.screen, PresetsScreen)

    asyncio.run(_run())


def test_presets_screen_a_opens_add_form(temp_db, fake_views) -> None:
    """Pressing `a` on the Presets screen pushes the add form."""
    from llmctl.tui._modals_presets import PresetFormModal

    async def _attempt() -> bool:
        with _patch_preset_loader(fake_views):
            app = MissionControlApp()
            async with app.run_test() as pilot:
                await pilot.press("p")
                for _ in range(150):
                    await pilot.pause(0.05)
                    if (
                        isinstance(app.screen, PresetsScreen)
                        and len(app.screen._row_aliases) >= 2
                    ):
                        break
                if not isinstance(app.screen, PresetsScreen):
                    return False
                await pilot.press("a")
                for _ in range(150):
                    await pilot.pause(0.05)
                    if isinstance(app.screen, PresetFormModal):
                        return True
                return False

    for _ in range(3):
        if asyncio.run(_attempt()):
            return
    raise AssertionError("add-preset form did not appear within budget")


def test_resolve_editor_prefers_visual_then_editor_then_vi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_editor follows the POSIX VISUAL > EDITOR > vi fallback chain."""
    from llmctl.tui._data import resolve_editor

    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.delenv("EDITOR", raising=False)
    assert resolve_editor() == ["vi"]

    monkeypatch.setenv("EDITOR", "nano")
    assert resolve_editor() == ["nano"]

    monkeypatch.setenv("VISUAL", "code --wait")
    assert resolve_editor() == ["code", "--wait"]


def test_validate_preset_file_round_trips_yaml(tmp_path: Path) -> None:
    """validate_preset_file returns the parsed Model when the YAML is good."""
    from llmctl.tui._data import validate_preset_file

    path = tmp_path / "x.yaml"
    path.write_text(
        "alias: x\n"
        "served_name: x\n"
        "model_id: org/x\n"
        "quantization: awq\n"
        "vllm_quantization_flag: awq_marlin\n"
        "tensor_parallel_size: 2\n"
        "max_model_len: 32768\n"
    )
    model = validate_preset_file(path)
    assert model.alias == "x"


def test_validate_preset_file_raises_on_schema_error(tmp_path: Path) -> None:
    """A broken YAML surfaces as PresetSchemaError so the TUI can notify the user."""
    from llmctl.presets import PresetSchemaError
    from llmctl.tui._data import validate_preset_file

    path = tmp_path / "x.yaml"
    path.write_text("alias: BAD-UPPERCASE\nserved_name: x\n")
    with pytest.raises(PresetSchemaError):
        validate_preset_file(path)


def test_run_editor_on_preset_invokes_editor_and_revalidates(
    tmp_path: Path,
) -> None:
    """run_editor_on_preset shells out and re-reads the file on return."""
    from llmctl.tui._data import run_editor_on_preset

    target = tmp_path / "x.yaml"
    target.write_text(
        "alias: x\n"
        "served_name: x\n"
        "model_id: org/x\n"
        "quantization: awq\n"
        "vllm_quantization_flag: awq_marlin\n"
        "tensor_parallel_size: 2\n"
        "max_model_len: 32768\n"
    )

    # Fake editor: rewrites the file to flip a field, proving the
    # function reads back what the editor wrote (not the original).
    fake_editor = tmp_path / "fake-editor.sh"
    fake_editor.write_text(
        "#!/usr/bin/env bash\n"
        "cat > \"$1\" <<'YAML'\n"
        "alias: x\n"
        "served_name: x\n"
        "model_id: org/edited\n"
        "quantization: awq\n"
        "vllm_quantization_flag: awq_marlin\n"
        "tensor_parallel_size: 2\n"
        "max_model_len: 32768\n"
        "YAML\n"
    )
    fake_editor.chmod(0o755)

    model = run_editor_on_preset(target, editor=[str(fake_editor)])
    assert model.model_id == "org/edited"
