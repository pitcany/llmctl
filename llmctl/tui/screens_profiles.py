"""DB-backed launch-profile management screen.

This is the management surface for ``ProfileService`` (create/edit/clone/
delete). It's separate from :class:`PresetsScreen`, which manages the
YAML-based preset *aliases* used by the vLLM orchestrator (TP fleet, coder,
reasoner). Bound to ``f`` from the app — see :class:`MissionControlApp`.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.schemas import ProfileCreate, ProfileUpdate
from llmctl.tui import _data
from llmctl.tui._base import C_MUTED, DataScreen, esc
from llmctl.tui._modals_registry import (
    CloneModal,
    CloneRequest,
    ConfirmDelete,
    DeleteModal,
    ProfileFormModal,
)


class ProfilesScreen(DataScreen):
    """List profiles and expose Create/Edit/Clone/Delete actions."""

    BINDINGS = [
        Binding("a", "add_profile", "Create", show=True),
        Binding("e", "edit_profile", "Edit", show=True),
        Binding("d", "delete_profile", "Delete", show=True),
        Binding("c", "clone_profile", "Clone", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            f"Profiles  -  [{C_MUTED}]a = add, e = edit, c = clone, d = delete[/]",
            classes="panel safe",
            id="profiles-title",
        )
        table: DataTable[str] = DataTable(id="profiles-table", cursor_type="row")
        table.add_columns(
            "ID", "Name", "Backend", "TP", "max_model_len", "Description"
        )
        yield table
        yield Footer()

    def fetch(self) -> Any:
        return _data.get_profiles()

    def render_data(self, data: Any) -> None:
        table = self.query_one("#profiles-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._ids = []
        for profile in data:
            pid = profile.id or ""
            self._ids.append(pid)
            table.add_row(
                pid[:8],
                esc(profile.name),
                profile.runtime.value,
                str(profile.tensor_parallel_size or "-"),
                str(profile.max_model_len or "-"),
                esc(profile.description or "-"),
            )
        if not data:
            table.add_row("-", "No profiles yet (a = create)", "-", "-", "-", "-")
        if 0 <= cursor < len(self._ids):
            table.move_cursor(row=cursor)

    def _selected_profile(self) -> Any:
        """Return the Profile under the cursor, if any."""
        table = self.query_one("#profiles-table", DataTable)
        row = table.cursor_row
        if not (0 <= row < len(self._ids)):
            return None
        target_id = self._ids[row]
        for profile in _data.get_profiles():
            if profile.id == target_id:
                return profile
        return None

    def action_add_profile(self) -> None:
        """Open the create-profile form."""

        def _on_close(payload: ProfileCreate | ProfileUpdate | None) -> None:
            if isinstance(payload, ProfileCreate):
                self.run_action_worker(
                    lambda: self._safe_create(payload), self._after_mutation
                )

        self.app.push_screen(ProfileFormModal(), _on_close)

    def action_edit_profile(self) -> None:
        """Open the edit-profile form for the cursor row."""
        profile = self._selected_profile()
        if profile is None:
            self.app.notify("No profile selected.", severity="warning")
            return

        def _on_close(payload: ProfileCreate | ProfileUpdate | None) -> None:
            if isinstance(payload, ProfileUpdate):
                self.run_action_worker(
                    lambda: self._safe_update(profile.id, payload),
                    self._after_mutation,
                )

        self.app.push_screen(ProfileFormModal(profile), _on_close)

    @staticmethod
    def _safe_create(payload: ProfileCreate) -> Any:
        """Call create_profile, returning the exception on validation errors.

        TUI worker results flow through ``_after_mutation`` on the UI thread.
        Raising in the worker would manifest as ``WorkerFailed`` (no user
        feedback); returning the exception lets the screen render an inline
        notification instead.
        """
        try:
            return _data.create_profile(payload)
        except _data.ProfileValidationError as exc:
            return exc

    @staticmethod
    def _safe_update(profile_id: str, updates: ProfileUpdate) -> Any:
        try:
            return _data.update_profile(profile_id, updates)
        except _data.ProfileValidationError as exc:
            return exc

    def action_clone_profile(self) -> None:
        """Clone the cursor-row profile under a new name."""
        profile = self._selected_profile()
        if profile is None:
            self.app.notify("No profile selected.", severity="warning")
            return

        def _on_close(payload: CloneRequest | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.clone_profile(payload.source_id, payload.new_name),
                self._after_mutation,
            )

        self.app.push_screen(CloneModal(profile.id, profile.name), _on_close)

    def action_delete_profile(self) -> None:
        """Confirm and delete the cursor-row profile."""
        profile = self._selected_profile()
        if profile is None:
            self.app.notify("No profile selected.", severity="warning")
            return

        def _on_close(payload: ConfirmDelete | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.delete_profile(profile.id), self._after_mutation
            )

        self.app.push_screen(
            DeleteModal(f"profile '{profile.name}'", allow_file_delete=False),
            _on_close,
        )

    def _after_mutation(self, result: Any) -> None:
        """Refresh the table after a service call completes, or surface errors."""
        if isinstance(result, _data.ProfileValidationError):
            self.app.notify(
                f"Profile rejected: {result}",
                severity="error",
                title="Validation failed",
            )
            return
        self.refresh_data()
