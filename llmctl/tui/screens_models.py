"""Models registry TUI screen with live data and start/scan actions."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Static

from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, DataScreen
from llmctl.tui._modals import LaunchPlanModal


class ModelsScreen(DataScreen):
    """Model registry screen: lists models and plans sessions."""

    BINDINGS = [
        Binding("enter", "start_model", "Plan", show=True),
        Binding("ctrl+s", "scan", "Scan", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ids: list[str] = []
        #: model id -> (runtime_value, backend_available)
        self._meta: dict[str, tuple[str, bool]] = {}

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Open the launch-plan preview when a row is activated (enter/click)."""
        self.action_start_model()

    def compose(self) -> ComposeResult:
        """Compose the models table."""
        yield Static(
            f"Models Registry  -  [{C_MUTED}]enter = preview & plan, ctrl+s = scan[/]",
            classes="panel safe",
            id="models-title",
        )
        table: DataTable[str] = DataTable(id="models-table", cursor_type="row")
        table.add_columns("ID", "Name", "Runtime", "Backend", "Status", "Quant", "Path")
        yield table

    def fetch(self) -> Any:
        """Return the current model list plus backend availability."""
        return {"models": _data.get_models(), "availability": _data.get_backend_map()}

    def render_data(self, data: Any) -> None:
        """Render the models table, dimming models with a missing backend."""
        models = data["models"]
        availability = data["availability"]
        table = self.query_one("#models-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._ids = []
        self._meta = {}
        for model in models:
            mid = model.id or ""
            runtime = model.runtime.value
            available = availability.get(runtime, True)
            self._ids.append(mid)
            self._meta[mid] = (runtime, available)
            if available:
                backend_cell = f"[{C_OK}]ready[/]"
                name_cell = model.name
            else:
                backend_cell = f"[{C_ERR}]no binary[/]"
                name_cell = f"[{C_MUTED}]{model.name}[/]"
            table.add_row(
                mid[:8],
                name_cell,
                runtime,
                backend_cell,
                model.status.value,
                model.quantization or "-",
                (model.path or "-")[-32:],
            )
        if not models:
            table.add_row("-", "No models registered (ctrl+s to scan)", "-", "-", "-", "-", "-")
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
        """Push the launch-plan modal and start the session on confirmation."""

        def _on_close(confirmed: bool | None) -> None:
            if confirmed:
                self.run_action_worker(
                    lambda: _data.start_model(model_id),
                    self._after_start,
                )

        self.app.push_screen(LaunchPlanModal(plan), _on_close)

    def _after_start(self, session: Any) -> None:
        """Notify and refresh after a session is planned."""
        self.app.notify(
            f"Planned session {(session.id or '')[:8]} ({session.runtime.value}).",
            title="Session planned",
        )
        self.refresh_data()

    def action_scan(self) -> None:
        """Run adapter discovery and refresh the table."""
        self.run_action_worker(_data.scan_models, self._after_scan)

    def _after_scan(self, found: Any) -> None:
        """Notify and refresh after a scan completes."""
        self.app.notify(f"Scan complete: {len(found)} models registered.")
        self.refresh_data()
