"""Events screen: llmctl's own audit trail (not process logs).

Retitled from "Logs" (TUI-UX-FIXPLAN P1 #9): it never showed a process log.
Those live on the Sessions screen (log pane) and, for systemd units, behind
``j`` on the Units screen.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc

_LEVEL_COLOR = {
    "debug": C_MUTED,
    "info": C_OK,
    "warning": C_WARN,
    "error": C_ERR,
    "critical": C_ERR,
}


class LogsScreen(DataScreen):
    """Event/log viewer screen showing the live audit trail."""

    def compose(self) -> ComposeResult:
        """Compose the events table with screen-scoped chrome."""
        yield Header()
        yield Static(
            f"Events  -  llmctl's audit trail  [{C_MUTED}]process logs: Sessions "
            f"(log pane) · unit journals: Units, then j[/]",
            classes="panel safe",
            id="logs-title",
        )
        table: DataTable[str] = DataTable(id="logs-table", cursor_type="row")
        table.add_columns("Time", "Level", "Category", "Message")
        yield table
        yield Footer()

    def fetch(self) -> Any:
        """Return the most recent audit events."""
        return _data.get_events(limit=100)

    def render_data(self, data: Any) -> None:
        """Render the events table."""
        table = self.query_one("#logs-table", DataTable)
        table.clear()
        for event in data:
            color = _LEVEL_COLOR.get(event["level"], C_MUTED)
            table.add_row(
                esc(event["time"]),
                f"[{color}]{esc(event['level'])}[/]",
                esc(event["category"]),
                esc(event["message"]),
            )
        if not data:
            table.add_row("-", "-", "-", "No events recorded yet.")
