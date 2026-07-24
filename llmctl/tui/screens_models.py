"""Models registry TUI screen with live data and start/scan actions."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.db import ModelStatus
from llmctl.schemas import ModelCreate, ModelUpdate
from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc
from llmctl.tui._modals import ConfirmActionModal, LaunchPlanModal
from llmctl.tui._modals_registry import (
    CloneModal,
    CloneRequest,
    ConfirmDelete,
    DeleteModal,
    ModelFormModal,
)


class ModelsScreen(DataScreen):
    """Model registry screen: lists models and plans sessions."""

    BINDINGS = [
        Binding("enter", "start_model", "Plan/Launch", show=True),
        Binding("ctrl+s", "scan", "Scan", show=True),
        Binding("a", "add_model", "Add", show=True),
        Binding("e", "edit_model", "Edit", show=True),
        Binding("d", "delete_model", "Delete", show=True),
        Binding("x", "prune_missing", "Prune missing", show=True),
        Binding("c", "clone_model", "Clone", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ids: list[str] = []
        #: model id -> (runtime_value, backend_available)
        self._meta: dict[str, tuple[str, bool]] = {}
        self._missing_count: int = 0

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Open the launch-plan preview when a row is activated (enter/click)."""
        self.action_start_model()

    def compose(self) -> ComposeResult:
        """Compose the models table with screen-scoped Header/Footer."""
        yield Header()
        yield Static(
            f"Models Registry  -  [{C_MUTED}]enter = preview / plan / launch, ctrl+s = scan[/]",
            classes="panel safe",
            id="models-title",
        )
        table: DataTable[str] = DataTable(id="models-table", cursor_type="row")
        table.add_columns(
            "ID",
            "Name",
            "Runtime",
            "Backend",
            "Status",
            "Quant",
            "Path",
            "Presets",
        )
        yield table
        yield Footer()

    def fetch(self) -> Any:
        """Return the current model list, backend availability, preset counts."""
        return {
            "models": _data.get_models(),
            "availability": _data.get_backend_map(),
            "preset_counts": _data.get_preset_count_by_model(),
            "missing_count": _data.get_missing_count(),
        }

    def render_data(self, data: Any) -> None:
        """Render the models table, dimming models with a missing backend."""
        models = data["models"]
        self._missing_count = data.get(
            "missing_count",
            sum(1 for m in models if m.status == ModelStatus.MISSING),
        )
        availability = data["availability"]
        preset_counts: dict[str, int] = data.get("preset_counts", {})
        table = self.query_one("#models-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._ids = []
        self._meta = {}
        for model in models:
            mid = model.id or ""
            runtime = model.runtime.value
            available = availability.get(runtime, True)
            is_missing = model.status == ModelStatus.MISSING
            self._ids.append(mid)
            self._meta[mid] = (runtime, available)
            backend_cell = f"[{C_OK}]ready[/]" if available else f"[{C_ERR}]no binary[/]"
            if available and not is_missing:
                name_cell = esc(model.name)
            else:
                name_cell = f"[{C_MUTED}]{esc(model.name)}[/]"
            status_cell = (
                f"[{C_WARN}]{model.status.value}[/]" if is_missing else model.status.value
            )
            count = preset_counts.get(mid, 0)
            preset_cell = str(count) if count else f"[{C_MUTED}]-[/]"
            table.add_row(
                mid[:8],
                name_cell,
                runtime,
                backend_cell,
                status_cell,
                esc(model.quantization or "-"),
                # Truncate first, then escape: slicing a balanced path can
                # strand a closing tag that only escaping renders harmless.
                esc((model.path or "-")[-32:]),
                preset_cell,
            )
        if not models:
            table.add_row(
                "-", "No models registered (ctrl+s to scan)", "-", "-", "-", "-", "-", "-"
            )
        if 0 <= cursor < len(self._ids):
            table.move_cursor(row=cursor)

    def _selected_id(self) -> str | None:
        """Return the model id under the cursor, if any."""
        table = self.query_one("#models-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._ids):
            return self._ids[row]
        return None

    def action_start_model(self) -> None:
        """Preview a launch plan, or surface an install hint when unavailable."""
        model_id = self._selected_id()
        if not model_id:
            self.app.notify("No model selected.", severity="warning")
            return
        runtime, available = self._meta.get(model_id, ("", True))
        if not available:
            hint = _data.BACKEND_INSTALL_HINTS.get(runtime, "")
            message = f"'{runtime}' backend is not installed." + (f" {hint}" if hint else "")
            self.app.notify(message, severity="warning", title="Backend unavailable")
            return
        self.run_action_worker(
            lambda: _data.get_launch_plan(model_id),
            lambda plan: self._show_plan(model_id, plan),
        )

    def _show_plan(self, model_id: str, plan: Any) -> None:
        """Push the launch-plan modal; plan or launch based on the choice."""

        def _on_close(outcome: str | None) -> None:
            if outcome not in ("plan", "launch"):
                return
            dry_run = outcome == "plan"
            self.run_action_worker(
                lambda: _data.start_model(model_id, dry_run=dry_run, force=dry_run),
                self._after_start,
            )

        self.app.push_screen(LaunchPlanModal(plan), _on_close)

    def _after_start(self, session: Any) -> None:
        """Notify with the real outcome (planned / started / failed) and refresh."""
        short_id = (session.id or "")[:8]
        status = session.status.value
        if status == "planned":
            self.app.notify(
                f"Planned session {short_id} ({session.runtime.value}); no process launched.",
                title="Session planned",
            )
        elif status in ("running", "starting"):
            detail = f" at {esc(session.endpoint_url)}" if session.endpoint_url else ""
            self.app.notify(
                f"Session {short_id} {status}{detail}.",
                title="Session launched",
            )
        else:
            self.app.notify(
                f"Session {short_id} {status}: {esc(session.error or 'unknown error')}",
                title="Launch failed",
                severity="error",
                timeout=10,
            )
        self.refresh_data()

    def action_scan(self) -> None:
        """Confirm, then run adapter discovery and persist the results.

        This is ``scan --import``, not the CLI's non-persisting preview: it
        upserts every discovered model and flags absent ones MISSING, which
        the prune action then removes irreversibly. Worth a gate on a chord
        that everywhere else means "save".
        """

        def _on_close(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self.run_action_worker(_data.scan_models, self._after_scan)

        self.app.push_screen(
            ConfirmActionModal(
                "Scan runtimes and import?",
                "Discovered models are written to the registry. Models a "
                "reachable runtime no longer reports are flagged MISSING, "
                "which prune (x) then removes irreversibly.",
                confirm_label="Scan",
            ),
            _on_close,
        )

    def _after_scan(self, found: Any) -> None:
        """Notify and refresh after a scan completes.

        ``scan()`` returns the whole registry, not just this pass's finds, so
        report it as a total rather than implying a discovery count.
        """
        self.app.notify(f"Scan complete. Registry now holds {len(found)} models.")
        self.refresh_data()

    def action_prune_missing(self) -> None:
        """Confirm and soft-delete the models flagged MISSING right now.

        The ids are captured here, before the dialog opens, and the prune acts
        on exactly that set. Re-deriving it at confirm time let a scan landing
        in between widen the deletion past the number the operator agreed to.
        """
        ids = _data.get_missing_model_ids()
        if not ids:
            self.app.notify("No missing models to prune.", severity="information")
            return

        def _on_close(payload: ConfirmDelete | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.prune_missing_models(ids=ids),
                self._after_prune,
            )

        self.app.push_screen(
            DeleteModal(
                f"{len(ids)} missing model(s)",
                "Each row is hidden from listings. llmctl has no undelete and "
                "a rescan will not revive them. Files on disk are untouched.",
            ),
            _on_close,
        )

    def _after_prune(self, count: Any) -> None:
        """Notify and refresh after a prune completes."""
        self.app.notify(f"Pruned {count} missing model(s).")
        self.refresh_data()

    def _selected_model(self) -> Any:
        """Return the full Model schema for the cursor row, if any."""
        model_id = self._selected_id()
        if not model_id:
            return None
        for model in _data.get_models():
            if model.id == model_id:
                return model
        return None

    def action_add_model(self) -> None:
        """Open the add-model form."""

        def _on_close(payload: ModelCreate | ModelUpdate | None) -> None:
            if isinstance(payload, ModelCreate):
                self.run_action_worker(
                    lambda: _data.add_model(payload), self._after_mutation
                )

        self.app.push_screen(ModelFormModal(), _on_close)

    def action_edit_model(self) -> None:
        """Open the edit-model form for the cursor row."""
        model = self._selected_model()
        if model is None:
            self.app.notify("No model selected.", severity="warning")
            return

        def _on_close(payload: ModelCreate | ModelUpdate | None) -> None:
            if isinstance(payload, ModelUpdate):
                self.run_action_worker(
                    lambda: _data.update_model(model.id, payload),
                    self._after_mutation,
                )

        self.app.push_screen(ModelFormModal(model), _on_close)

    def action_clone_model(self) -> None:
        """Clone the cursor-row model under a new name."""
        model = self._selected_model()
        if model is None:
            self.app.notify("No model selected.", severity="warning")
            return

        def _on_close(payload: CloneRequest | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.clone_model(payload.source_id, payload.new_name),
                self._after_mutation,
            )

        self.app.push_screen(CloneModal(model.id, model.name), _on_close)

    def action_delete_model(self) -> None:
        """Confirm and delete the cursor-row model."""
        model = self._selected_model()
        if model is None:
            self.app.notify("No model selected.", severity="warning")
            return

        def _on_close(payload: ConfirmDelete | None) -> None:
            if payload is None:
                return
            self.run_action_worker(
                lambda: _data.delete_model(
                    model.id, delete_files=payload.delete_files
                ),
                self._after_mutation,
            )

        self.app.push_screen(
            DeleteModal(
                f"model '{model.name}'",
                "The registry row is hidden from listings. llmctl has no undelete "
                "and a rescan will not revive it.",
                allow_file_delete=True,
                file_delete_target=model.path,
            ),
            _on_close,
        )

    def _after_mutation(self, _result: Any) -> None:
        """Refresh the table after add/edit/delete/clone completes."""
        self.refresh_data()
