"""Sessions TUI screen with live data, lifecycle actions, and a log-tail pane."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc
from llmctl.tui._modals import ConfirmActionModal

_STATUS_COLOR = {
    "running": C_OK,
    "planned": C_WARN,
    "starting": C_WARN,
    "degraded": C_ERR,
    "stopping": C_WARN,
    "stopped": C_MUTED,
    "failed": C_ERR,
}


class SessionsScreen(DataScreen):
    """Runtime sessions screen: lists sessions, controls lifecycle, tails logs."""

    BINDINGS = [
        Binding("x", "stop_session", "Stop", show=True),
        Binding("ctrl+r", "restart_session", "Restart", show=True),
        Binding("c", "cleanup", "Cleanup", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ids: list[str] = []
        self._tail_id: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the sessions table + log-tail pane with screen-scoped chrome."""
        yield Header()
        yield Static(
            f"Sessions  -  [{C_MUTED}]x = stop, ctrl+r = restart, c = cleanup[/]",
            classes="panel safe",
            id="sessions-title",
        )
        table: DataTable[str] = DataTable(id="sessions-table", cursor_type="row")
        table.add_columns("ID", "Runtime", "Status", "PID", "Port", "GPUs", "Endpoint")
        yield table
        yield Static(
            f"Session log  [{C_MUTED}](highlight a row to tail)[/]",
            classes="safe",
            id="session-log-title",
        )
        with VerticalScroll(id="session-log-wrap", classes="panel"):
            yield Static("", id="session-log")
        yield Footer()

    def fetch(self) -> Any:
        """Return sessions plus the tail of the highlighted session's log."""
        sessions = _data.get_sessions()
        tail = _data.tail_log(self._tail_id) if self._tail_id else ""
        return {"sessions": sessions, "tail": tail}

    def render_data(self, data: Any) -> None:
        """Render the sessions table and the log pane."""
        sessions = data["sessions"]
        table = self.query_one("#sessions-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._ids = []
        for session in sessions:
            sid = session.id or ""
            self._ids.append(sid)
            color = _STATUS_COLOR.get(session.status.value, C_MUTED)
            gpus = ",".join(str(g) for g in session.gpu_ids) or "cpu"
            table.add_row(
                sid[:8],
                session.runtime.value,
                f"[{color}]{session.status.value}[/]",
                str(session.pid or "-"),
                str(session.port or "-"),
                gpus,
                esc(session.endpoint_url or "-"),
            )
        if not sessions:
            table.add_row("-", "-", "no sessions yet", "-", "-", "-", "-")
        if 0 <= cursor < len(self._ids):
            table.move_cursor(row=cursor)

        log = self.query_one("#session-log", Static)
        if not self._tail_id:
            log.update(f"[{C_MUTED}]Highlight a session to view its log tail.[/]")
        elif not data["tail"]:
            log.update(f"[{C_MUTED}]No log output for this session yet.[/]")
        else:
            # Process output is not markup: '[/INST]' templates and '[rank0]:'
            # torch prefixes appear routinely and would crash or vanish.
            log.update(esc(data["tail"]))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Track the highlighted session and refresh its log tail."""
        row = event.cursor_row
        if 0 <= row < len(self._ids):
            new_id = self._ids[row]
            if new_id != self._tail_id:
                self._tail_id = new_id
                self.refresh_data()

    def _selected_id(self) -> str | None:
        """Return the session id under the cursor, if any."""
        table = self.query_one("#sessions-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._ids):
            return self._ids[row]
        return None

    def action_stop_session(self) -> None:
        """Confirm, then stop the selected session."""
        self._confirm_lifecycle(
            verb="Stop",
            consequence=(
                "The session process is terminated. Any in-flight request "
                "against its endpoint fails."
            ),
            run=_data.stop_session,
            past="Stopped",
        )

    def action_restart_session(self) -> None:
        """Confirm, then restart the selected session.

        Gated even though ``ctrl+r`` is merely "refresh" on the Presets and
        Units screens — that collision is precisely why this needs a gate: the
        harmless meaning is the one an operator's fingers learn first.
        """
        self._confirm_lifecycle(
            verb="Restart",
            consequence=(
                "The session process is terminated and relaunched. It is "
                "unavailable while it reloads."
            ),
            run=_data.restart_session,
            past="Restarted",
        )

    def _confirm_lifecycle(
        self, *, verb: str, consequence: str, run: Any, past: str
    ) -> None:
        """Shared confirm-then-dispatch for the process lifecycle actions."""
        session_id = self._selected_id()
        if not session_id:
            self.app.notify("No session selected.", severity="warning")
            return

        def _on_close(confirmed: bool | None) -> None:
            if not confirmed:
                return
            self.run_action_worker(
                lambda: run(session_id),
                lambda result: self._after_lifecycle(result, session_id, past),
            )

        self.app.push_screen(
            ConfirmActionModal(
                f"{verb} session {session_id[:8]}?",
                consequence,
                confirm_label=verb,
            ),
            _on_close,
        )

    def action_cleanup(self) -> None:
        """Reconcile dead sessions and free their ports."""
        self.run_action_worker(
            lambda: _data.cleanup_sessions(remove_stale=False),
            self._after_cleanup,
        )

    def _after_cleanup(self, report: Any) -> None:
        """Notify and refresh after a cleanup action."""
        freed = ", ".join(str(p) for p in report.get("freed_ports", [])) or "none"
        self.app.notify(
            f"Cleanup: {report.get('dead_marked', 0)} marked dead, ports freed: {freed}.",
            title="Cleanup",
        )
        self.refresh_data()

    def _after_lifecycle(self, session: Any, session_id: str, verb: str) -> None:
        """Notify and refresh after a stop/restart action."""
        if session is None:
            self.app.notify("Session not found.", severity="error")
        else:
            self.app.notify(f"{verb} session {session_id[:8]} ({session.status.value}).")
        self.refresh_data()
