"""Benchmarks TUI screen: history, baseline comparison and a re-run action."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen


class BenchmarksScreen(DataScreen):
    """Benchmark history: lists results, compares to a baseline, and re-runs."""

    BINDINGS = [
        Binding("enter", "rerun", "Re-run", show=True),
        Binding("c", "set_baseline", "Set baseline", show=True),
        Binding("x", "clear_baseline", "Clear baseline", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ids: list[str] = []
        self._baseline_id: str | None = None

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Re-run the selected benchmark when a row is activated."""
        self.action_rerun()

    def compose(self) -> ComposeResult:
        """Compose the benchmarks history table with screen-scoped chrome."""
        yield Header()
        yield Static(
            f"Benchmarks  -  [{C_MUTED}]enter = re-run, c = set baseline, "
            f"x = clear baseline[/]",
            classes="panel safe",
            id="benchmarks-title",
        )
        table: DataTable[str] = DataTable(id="benchmarks-table", cursor_type="row")
        table.add_columns(
            "Name", "Mode", "Tokens", "Tok/s", "vs base", "TTFT", "vs base", "When"
        )
        yield table
        yield Footer()

    def fetch(self) -> Any:
        """Return the recorded benchmark history (latest first)."""
        return list(reversed(_data.get_benchmarks()))

    def render_data(self, data: Any) -> None:
        """Render the benchmark history with optional baseline deltas."""
        table = self.query_one("#benchmarks-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._ids = []
        baseline = self._baseline_for(data)
        for result in data:
            mode = str(result.parameters.get("mode", "?"))
            color = C_OK if mode == "live" else C_WARN
            is_baseline = baseline is not None and result.id == baseline.id
            tps = "-" if result.tokens_per_second is None else f"{result.tokens_per_second:.1f}"
            ttft = (
                "-"
                if result.time_to_first_token_ms is None
                else f"{result.time_to_first_token_ms:.0f} ms"
            )
            when = result.created_at.strftime("%H:%M:%S") if result.created_at else "-"
            name_cell = (
                f"[{C_OK}]* {result.name}[/]" if is_baseline else result.name
            )
            self._ids.append(result.id or "")
            table.add_row(
                name_cell,
                f"[{color}]{mode}[/]",
                str(result.total_tokens or 0),
                tps,
                self._delta_tps(result, baseline, is_baseline),
                ttft,
                self._delta_ttft(result, baseline, is_baseline),
                when,
            )
        if not data:
            table.add_row("-", "No benchmarks yet", "-", "-", "-", "-", "-", "-")
        if 0 <= cursor < len(self._ids):
            table.move_cursor(row=cursor)

    def _baseline_for(self, data: Any) -> Any | None:
        """Return the current baseline result if it is still present."""
        if self._baseline_id is None:
            return None
        match = next((r for r in data if r.id == self._baseline_id), None)
        if match is None:
            self._baseline_id = None
        return match

    @staticmethod
    def _delta_tps(result: Any, baseline: Any, is_baseline: bool) -> str:
        """Render throughput delta vs. baseline (higher is better)."""
        if is_baseline:
            return f"[{C_MUTED}]baseline[/]"
        if (
            baseline is None
            or result.tokens_per_second is None
            or baseline.tokens_per_second is None
        ):
            return "-"
        delta = result.tokens_per_second - baseline.tokens_per_second
        color = C_OK if delta >= 0 else C_ERR
        return f"[{color}]{delta:+.1f}[/]"

    @staticmethod
    def _delta_ttft(result: Any, baseline: Any, is_baseline: bool) -> str:
        """Render TTFT delta vs. baseline (lower is better)."""
        if is_baseline:
            return f"[{C_MUTED}]baseline[/]"
        if (
            baseline is None
            or result.time_to_first_token_ms is None
            or baseline.time_to_first_token_ms is None
        ):
            return "-"
        delta = result.time_to_first_token_ms - baseline.time_to_first_token_ms
        color = C_OK if delta <= 0 else C_ERR
        return f"[{color}]{delta:+.0f} ms[/]"

    def _selected_id(self) -> str | None:
        """Return the benchmark id under the cursor, if any."""
        table = self.query_one("#benchmarks-table", DataTable)
        row = table.cursor_row
        if 0 <= row < len(self._ids):
            return self._ids[row]
        return None

    def action_set_baseline(self) -> None:
        """Mark the selected benchmark as the comparison baseline."""
        benchmark_id = self._selected_id()
        if not benchmark_id:
            self.app.notify("No benchmark selected.", severity="warning")
            return
        self._baseline_id = benchmark_id
        self.app.notify("Baseline set; deltas are relative to it.", title="Compare")
        self.refresh_data()

    def action_clear_baseline(self) -> None:
        """Clear the comparison baseline."""
        if self._baseline_id is None:
            return
        self._baseline_id = None
        self.app.notify("Baseline cleared.", title="Compare")
        self.refresh_data()

    def action_rerun(self) -> None:
        """Re-run the selected benchmark off-thread, then refresh."""
        benchmark_id = self._selected_id()
        if not benchmark_id:
            self.app.notify("No benchmark selected.", severity="warning")
            return
        self.run_action_worker(
            lambda: _data.rerun_benchmark(benchmark_id),
            self._after_rerun,
        )

    def _after_rerun(self, result: Any) -> None:
        """Notify and refresh after a re-run completes."""
        if result is None:
            self.app.notify("Benchmark not found.", severity="error")
        else:
            mode = result.parameters.get("mode", "?")
            self.app.notify(
                f"Re-ran '{result.name}' ({mode}).",
                title="Benchmark complete",
            )
        self.refresh_data()
