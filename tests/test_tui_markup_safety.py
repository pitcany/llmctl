"""Registry and log text must never be parsed as Rich console markup.

Textual runs ``Text.from_markup`` over every ``str`` cell in a DataTable and
over ``Static.update`` content. Model names, checkpoint paths and vLLM log
output routinely contain bracketed substrings — ``[/INST]`` chat templates,
``[rank0]:`` distributed prefixes, ``[/mnt/...]`` paths. Unescaped, a closing
tag raises ``MarkupError`` and takes the whole app down; a non-closing one is
silently deleted from the display.
"""

from __future__ import annotations

import asyncio

from rich.text import Text
from textual.widgets import DataTable, Static

from llmctl.db import ModelStatus, RuntimeName, SessionStatus
from llmctl.schemas import Model, Session
from llmctl.tui import _data
from llmctl.tui.app import MissionControlApp

#: A name with an unbalanced closing tag — the hard-crash case.
HOSTILE_NAME = "qwen [/tmp] test"
#: A chat-template token vLLM echoes at startup.
HOSTILE_TAIL = "RuntimeError: bad dir [/opt/models/x] template [/INST]"
#: Torch distributed prefix — the silent-deletion case.
RANK_TAIL = "ERROR [rank0]: CUDA out of memory"


def _plain(cell: object) -> str:
    """Return what the user actually sees for a markup-formatted cell."""
    return Text.from_markup(str(cell)).plain


def _model(name: str, path: str | None = None) -> Model:
    return Model(
        id="hostile-id",
        name=name,
        runtime=RuntimeName.OLLAMA,
        source=name,
        path=path,
        status=ModelStatus.DISCOVERED,
    )


def _session() -> Session:
    return Session(
        id="sess-id",
        model_id="hostile-id",
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
    )


async def _show(app: MissionControlApp, pilot, action: str, prefix: str) -> None:
    getattr(app, action)()
    for _ in range(50):
        await pilot.pause(0.02)
        if app.screen.__class__.__name__.lower().startswith(prefix):
            return
    raise AssertionError(f"{prefix} screen did not become active")


def test_hostile_model_name_does_not_kill_the_app(monkeypatch) -> None:
    """A model named 'qwen [/tmp] test' must render, not raise MarkupError."""
    monkeypatch.setattr(_data, "get_models", lambda: [_model(HOSTILE_NAME)])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {"ollama": True})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_models", "models")
            table = app.screen.query_one("#models-table", DataTable)
            for _ in range(50):
                await pilot.pause(0.02)
                if table.row_count:
                    break
            assert app.is_running, "app died rendering a bracketed model name"
            # The name must reach the screen intact, not be eaten as a tag.
            assert _plain(table.get_row_at(0)[1]) == HOSTILE_NAME

    asyncio.run(_run())


def test_hostile_path_does_not_kill_the_app(monkeypatch) -> None:
    """Path cells are truncated with [-32:], which can manufacture a closing tag."""
    # Long enough that [-32:] drops the opening tag and strands the closing one.
    hostile_path = "/mnt/storage/llm_models_backup/store/[b]safetensors[/b]/weights"
    monkeypatch.setattr(
        _data, "get_models", lambda: [_model("ok", path=hostile_path)]
    )
    monkeypatch.setattr(_data, "get_backend_map", lambda: {"ollama": True})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_models", "models")
            for _ in range(50):
                await pilot.pause(0.02)
                if app.screen.query_one("#models-table", DataTable).row_count:
                    break
            assert app.is_running, "app died rendering a bracketed path"

    asyncio.run(_run())


def test_log_tail_with_closing_tag_does_not_kill_the_app(monkeypatch) -> None:
    """A '[/INST]' in tailed output must not terminate the TUI."""
    monkeypatch.setattr(_data, "get_sessions", lambda: [_session()])
    monkeypatch.setattr(_data, "tail_log", lambda _sid: HOSTILE_TAIL)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_sessions", "sessions")
            screen = app.screen
            screen._tail_id = "sess-id"
            screen.refresh_data()
            for _ in range(50):
                await pilot.pause(0.02)
            assert app.is_running, "app died rendering a bracketed log tail"

    asyncio.run(_run())


def test_log_tail_preserves_rank_prefix(monkeypatch) -> None:
    """'[rank0]:' must reach the pane; it is the text the pane exists to show."""
    monkeypatch.setattr(_data, "get_sessions", lambda: [_session()])
    monkeypatch.setattr(_data, "tail_log", lambda _sid: RANK_TAIL)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_sessions", "sessions")
            screen = app.screen
            screen._tail_id = "sess-id"
            screen.refresh_data()
            log = screen.query_one("#session-log", Static)
            for _ in range(50):
                await pilot.pause(0.02)
                if "rank0" in _plain(log.content):
                    break
            assert app.is_running
            assert _plain(log.content) == RANK_TAIL, (
                f"rank prefix was eaten as markup: {log.content!r}"
            )

    asyncio.run(_run())
