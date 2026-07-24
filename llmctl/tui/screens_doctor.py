"""Doctor TUI screen: backend binary diagnostics with copy-install support."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc


class DoctorScreen(DataScreen):
    """Backend health check: lists runtime binaries and missing-install fixes."""

    BINDINGS = [
        Binding("c", "copy_install", "Copy install cmd", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        #: Row index -> (backend_name, install_command)
        self._rows: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        """Compose the doctor diagnostics table with screen-scoped chrome."""
        yield Header()
        yield Static(
            f"Doctor  -  backend binary detection  "
            f"[{C_MUTED}]c = copy install command for selected backend[/]",
            classes="panel safe",
            id="doctor-title",
        )
        table: DataTable[str] = DataTable(id="doctor-table", cursor_type="row")
        table.add_columns("Backend", "Binary", "Status", "Path / Install command")
        yield table
        yield Static("", classes="panel muted", id="doctor-summary")
        yield Footer()

    def fetch(self) -> Any:
        """Return backend availability plus a GPU/scheduler summary."""
        return {
            "backends": _data.get_backends(),
            "summary": _data.get_doctor_summary(),
        }

    def render_data(self, data: Any) -> None:
        """Render backend diagnostics, surfacing install commands for gaps."""
        table = self.query_one("#doctor-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._rows = []
        for entry in data["backends"]:
            backend = str(entry["backend"])
            binary = str(entry["binary"])
            available = bool(entry["available"])
            if available:
                status_cell = f"[{C_OK}]ready[/]"
                detail = esc(str(entry.get("path") or "-"))
                command = ""
            else:
                status_cell = f"[{C_ERR}]missing[/]"
                command = _data.install_command_for(backend)
                detail = (
                    f"[{C_MUTED}]$ {command}[/]"
                    if command
                    else f"[{C_MUTED}]{_data.BACKEND_INSTALL_HINTS.get(backend, '')}[/]"
                )
            self._rows.append((backend, command))
            table.add_row(backend, esc(binary), status_cell, detail)
        if 0 <= cursor < len(self._rows):
            table.move_cursor(row=cursor)
        self._render_summary(data["summary"])

    def _render_summary(self, summary: dict[str, Any]) -> None:
        """Render the GPU/NVML status and scheduler configuration panel."""
        gpu_color = C_OK if summary["gpu_count"] else C_WARN
        nvml_color = C_OK if summary["nvml_available"] else C_WARN
        public_color = C_WARN if summary["allow_public_bind"] else C_OK
        missing = summary["missing_backends"]
        missing_line = (
            f"  [{C_ERR}]Missing backends:[/] {', '.join(missing)}"
            if missing
            else f"  [{C_OK}]All backends available.[/]"
        )
        lines = [
            "[b]System & scheduler[/b]",
            f"  GPUs: [{gpu_color}]{summary['gpu_count']}[/]   "
            f"NVML: [{nvml_color}]{summary['nvml_available']}[/]   "
            f"Safe mode: {summary['safe_mode']}",
            f"  Scheduler policy: {summary['gpu_policy']}   "
            f"safety margin: {summary['safety_margin_gb']} GB   "
            f"default host: {summary['default_host']}",
            f"  Public bind: [{public_color}]{summary['allow_public_bind']}[/]",
            missing_line,
        ]
        self.query_one("#doctor-summary", Static).update("\n".join(lines))

    def action_copy_install(self) -> None:
        """Copy the selected backend's install command to the clipboard."""
        table = self.query_one("#doctor-table", DataTable)
        row = table.cursor_row
        if not (0 <= row < len(self._rows)):
            self.app.notify("No backend selected.", severity="warning")
            return
        backend, command = self._rows[row]
        if not command:
            self.app.notify(
                f"'{backend}' is already available - nothing to install.",
                title="Doctor",
            )
            return
        self.app.copy_to_clipboard(command)
        self.app.notify(
            f"Copied: {command}",
            title=f"Install command for {backend}",
        )
