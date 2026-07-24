"""Destructive dialogs must describe what actually happens.

Every confirmation in the TUI previously shared one hardcoded explainer —
"Soft-delete only ... on-disk files are preserved unless you tick the box
below" — regardless of what the confirmed action did. On the Presets screen it
unlinks YAML files, on Benchmarks it hard-deletes a row, and neither has a box
to tick. These tests pin the dialog copy to the real consequence, and pin the
guards that keep a reflexive Enter from destroying something.
"""

from __future__ import annotations

import asyncio

from textual.app import App
from textual.widgets import Button, Checkbox, Static

from llmctl.tui._modals import ConfirmActionModal
from llmctl.tui._modals_registry import DeleteModal


def _texts(screen) -> str:
    return " ".join(str(w.content) for w in screen.query(Static))


class _Host(App[None]):
    """Bare host so a modal can be pushed in isolation."""

    def __init__(self, modal) -> None:
        super().__init__()
        self._modal = modal

    def on_mount(self) -> None:
        self.push_screen(self._modal)


def _run(modal, body):
    async def _main() -> None:
        app = _Host(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            await body(app, pilot)

    asyncio.run(_main())


def test_delete_modal_states_the_caller_supplied_consequence() -> None:
    """The explainer is supplied per call site, not hardcoded."""
    modal = DeleteModal(
        "preset 'ornith-35b' ",
        "The preset YAML file is deleted from disk.",
        allow_file_delete=False,
    )

    async def body(app, pilot) -> None:
        blob = _texts(app.screen)
        assert "deleted from disk" in blob
        # The old boilerplate must be gone.
        assert "preserved unless you tick" not in blob
        assert "Soft-delete only" not in blob

    _run(modal, body)


def test_delete_modal_without_checkbox_never_mentions_one() -> None:
    """No checkbox is composed, so no copy may refer to 'the box below'."""
    modal = DeleteModal("thing", "The row is removed.", allow_file_delete=False)

    async def body(app, pilot) -> None:
        assert not app.screen.query(Checkbox)
        assert "box below" not in _texts(app.screen)

    _run(modal, body)


def test_delete_modal_focuses_cancel_not_the_destructive_control() -> None:
    """A reflexive Enter must cancel, never arm file deletion or delete."""
    modal = DeleteModal(
        "model 'x'",
        "The registry row is hidden.",
        allow_file_delete=True,
        file_delete_target="/models/x",
    )

    async def body(app, pilot) -> None:
        focused = app.screen.focused
        assert focused is not None, "nothing focused"
        assert focused.id == "delete-cancel", (
            f"default focus is {focused.id!r}, expected the Cancel button"
        )
        await pilot.press("enter")
        await pilot.pause()

    _run(modal, body)
    # Enter on Cancel dismisses with None -> nothing destroyed.


def test_delete_modal_shows_the_path_it_would_remove() -> None:
    """'Also delete files' must name the target; the CLI does."""
    modal = DeleteModal(
        "model 'x'",
        "The registry row is hidden.",
        allow_file_delete=True,
        file_delete_target="/mnt/storage/models/x",
    )

    async def body(app, pilot) -> None:
        blob = _texts(app.screen) + " ".join(
            str(c.label) for c in app.screen.query(Checkbox)
        )
        assert "/mnt/storage/models/x" in blob, (
            "the path that would be deleted is never shown"
        )

    _run(modal, body)


def test_confirm_action_modal_defaults_to_cancel() -> None:
    """The generic confirm gate must not default to the action either."""
    modal = ConfirmActionModal(
        "Stop session abc123?",
        "The process is terminated.",
        confirm_label="Stop",
    )

    async def body(app, pilot) -> None:
        focused = app.screen.focused
        assert focused is not None and focused.id == "confirm-cancel"
        blob = _texts(app.screen)
        assert "process is terminated" in blob

    _run(modal, body)


def test_confirm_action_modal_returns_true_only_on_confirm() -> None:
    """Confirm -> True; cancel/escape -> None."""
    results: list[object] = []

    class Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(
                ConfirmActionModal("t", "c", confirm_label="Go"), results.append
            )

    async def _main() -> None:
        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.query_one("#confirm-ok", Button).press()
            await pilot.pause()

    asyncio.run(_main())
    assert results == [True]

    results.clear()

    async def _main_cancel() -> None:
        app = Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.screen.query_one("#confirm-cancel", Button).press()
            await pilot.pause()

    asyncio.run(_main_cancel())
    assert results == [None]
