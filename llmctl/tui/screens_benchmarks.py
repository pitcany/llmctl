"""Benchmarks TUI screen: history, baseline comparison and a re-run action."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.services.benchmark_compare import comparison_verdict
from llmctl.tui import _data
from llmctl.tui._base import C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc
from llmctl.tui._modals_benchmarks import (
    BenchmarkInspectModal,
    BenchmarkLaunch,
    BenchmarkLaunchModal,
)
from llmctl.tui._modals_registry import ConfirmDelete, DeleteModal


class BenchmarksScreen(DataScreen):
    """Benchmark history: lists results, compares to a baseline, and re-runs.

    §3D: Enter *inspects* the highlighted record (samples, provenance,
    errors, drift, comparison). Re-running is the separate, labeled ``r``
    action — inspecting must never launch another workload by accident.
    """

    BINDINGS = [
        Binding("n", "new_benchmark", "New run", show=True),
        Binding("enter", "inspect", "Inspect", show=True, priority=True),
        Binding("r", "rerun", "Re-run", show=True),
        Binding("c", "set_baseline", "Baseline", show=True),
        Binding("x", "clear_baseline", "Clear baseline", show=True),
        Binding("delete", "delete_benchmark", "Delete", show=True, key_display="del"),
    ]

    def __init__(self, model_filter: str | None = None) -> None:
        """Optionally constrain the screen to a single ``model_id``.

        Used when the screen is reached by drilling in from the Models screen
        so the operator sees only that model's history.
        """
        super().__init__()
        self._ids: list[str] = []
        #: The most recent fetch payload, so inspect can resolve a row id
        #: without a second database read.
        self._last_data: Any = []
        #: benchmark id -> display name, so confirm dialogs can
        #: name a row the operator can actually see (the table
        #: has no ID column).
        self._names: dict[str, str] = {}
        #: benchmark id -> why it is not comparable with the baseline
        #: (empty when comparable); drives the "n/c" cells and the
        #: explanation line under the table.
        self._nc_reasons: dict[str, str] = {}
        self._baseline_id: str | None = None
        self._model_filter = model_filter

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Inspect the selected benchmark when a row is activated (§3D)."""
        self.action_inspect()

    def compose(self) -> ComposeResult:
        """Compose the benchmarks history table with screen-scoped chrome."""
        yield Header()
        chrome = (
            f"Benchmarks  -  [{C_MUTED}]n = new run, enter = inspect, r = re-run, "
            f"c = baseline, x = clear baseline, del = delete[/]"
        )
        if self._model_filter:
            chrome += f"  [{C_MUTED}]filter: model={self._model_filter}[/]"
        yield Static(chrome, classes="panel safe", id="benchmarks-title")
        table: DataTable[str] = DataTable(id="benchmarks-table", cursor_type="row")
        table.add_columns(
            "Name",
            "Kind",
            "Backend",
            "Mode",
            "Tokens",
            "Tok/s",
            "vs base",
            "TTFT",
            "vs base",
            "Peak VRAM",
            "GPU%",
            "OK",
            "When",
        )
        yield table
        yield Static("", classes="panel safe", id="benchmarks-compare")
        yield Static("", classes="panel safe", id="benchmarks-hint")
        yield Footer()

    def on_mount(self) -> None:
        """Load data and focus the table so cursor keys work immediately."""
        super().on_mount()
        try:
            self.query_one("#benchmarks-table", DataTable).focus()
        except Exception:  # noqa: BLE001 — best-effort focus
            pass

    def fetch(self) -> Any:
        """Return the recorded benchmark history (latest first)."""
        return list(reversed(_data.get_benchmarks(model_id=self._model_filter)))

    def render_data(self, data: Any) -> None:
        """Render the benchmark history with optional baseline deltas."""
        self._last_data = data
        table = self.query_one("#benchmarks-table", DataTable)
        cursor = table.cursor_row
        table.clear()
        self._ids = []
        self._names = {}
        self._nc_reasons = {}
        baseline = self._baseline_for(data)
        any_estimated = False
        for result in data:
            mode = str(result.parameters.get("mode", "?"))
            color = C_OK if mode == "live" else C_WARN
            is_baseline = baseline is not None and result.id == baseline.id
            token_source = str(result.parameters.get("token_count_source", ""))
            estimated = token_source in {"estimate", "mixed"}
            any_estimated = any_estimated or estimated
            if result.tokens_per_second is None:
                tps = "-"
            elif estimated:
                # "~" marks throughput computed from estimated token counts
                # (no server usage data), never from precise measurements.
                tps = f"{result.tokens_per_second:.1f}~"
            else:
                tps = f"{result.tokens_per_second:.1f}"
            ttft = (
                "-"
                if result.time_to_first_token_ms is None
                else f"{result.time_to_first_token_ms:.0f} ms"
            )
            peak_vram = (
                "-" if result.peak_vram_mb is None else f"{result.peak_vram_mb}"
            )
            gpu_pct = (
                "-"
                if result.max_gpu_util_pct is None
                else f"{result.max_gpu_util_pct:.0f}"
            )
            ok_color = C_OK if result.success else C_ERR
            ok = f"[{ok_color}]{'y' if result.success else 'n'}[/]"
            when = result.created_at.strftime("%H:%M:%S") if result.created_at else "-"
            name_cell = (
                f"[{C_OK}]* {esc(result.name)}[/]" if is_baseline else esc(result.name)
            )
            self._ids.append(result.id or "")
            self._names[result.id or ""] = result.name
            if baseline is not None and not is_baseline:
                verdict = comparison_verdict(result, baseline)
                if not verdict.eligible:
                    self._nc_reasons[result.id or ""] = verdict.summary
            table.add_row(
                name_cell,
                result.kind.value if result.kind else "-",
                esc(result.backend or "-"),
                f"[{color}]{mode}[/]",
                str(result.total_tokens or 0),
                tps,
                self._delta_tps(result, baseline, is_baseline),
                ttft,
                self._delta_ttft(result, baseline, is_baseline),
                peak_vram,
                gpu_pct,
                ok,
                when,
            )
        hint = self.query_one("#benchmarks-hint", Static)
        if not data:
            hint.update(
                f"[{C_MUTED}]No benchmarks recorded yet. Press [b]n[/b] "
                f"to launch one.[/]"
            )
        elif any_estimated:
            hint.update(
                f"[{C_MUTED}]~ = throughput from estimated token counts "
                f"(server sent no usage data)[/]"
            )
        else:
            hint.update("")
        if 0 <= cursor < len(self._ids):
            table.move_cursor(row=cursor)
        elif self._ids:
            table.move_cursor(row=0)
        self._update_compare_line()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Explain why the highlighted row is (not) comparable to the baseline."""
        self._update_compare_line()

    def _update_compare_line(self) -> None:
        """Render the comparison-eligibility note for the cursor row."""
        line = self.query_one("#benchmarks-compare", Static)
        benchmark_id = self._selected_id()
        if (
            self._baseline_id is None
            or not benchmark_id
            or benchmark_id == self._baseline_id
        ):
            line.update("")
            return
        reason = self._nc_reasons.get(benchmark_id)
        if reason:
            line.update(f"[{C_WARN}]vs baseline: not comparable -- {esc(reason)}[/]")
        else:
            line.update(f"[{C_MUTED}]vs baseline: comparable[/]")

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
        """Render throughput delta vs. baseline (higher is better).

        Deltas are only rendered when the shared eligibility gate says the
        two runs measured the same thing; otherwise the cell reads "n/c"
        (never a green or red number that would read as an improvement).
        """
        if is_baseline:
            return f"[{C_MUTED}]baseline[/]"
        if (
            baseline is None
            or result.tokens_per_second is None
            or baseline.tokens_per_second is None
        ):
            return "-"
        if not comparison_verdict(result, baseline).eligible:
            return f"[{C_MUTED}]n/c[/]"
        delta = result.tokens_per_second - baseline.tokens_per_second
        color = C_OK if delta >= 0 else C_ERR
        return f"[{color}]{delta:+.1f}[/]"

    @staticmethod
    def _delta_ttft(result: Any, baseline: Any, is_baseline: bool) -> str:
        """Render TTFT delta vs. baseline (lower is better), gated like tps."""
        if is_baseline:
            return f"[{C_MUTED}]baseline[/]"
        if (
            baseline is None
            or result.time_to_first_token_ms is None
            or baseline.time_to_first_token_ms is None
        ):
            return "-"
        if not comparison_verdict(result, baseline).eligible:
            return f"[{C_MUTED}]n/c[/]"
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
            if not self._ids:
                self.app.notify(
                    "No benchmarks yet. Press 'n' to launch one.",
                    severity="warning",
                )
            else:
                self.app.notify(
                    "Highlight a row with arrow keys, then press 'c'.",
                    severity="warning",
                )
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

    def action_inspect(self) -> None:
        """Open the read-only inspection view for the cursor-row record."""
        benchmark_id = self._selected_id()
        if not benchmark_id:
            self.app.notify(
                "No benchmark selected.", severity="warning"
            )
            return
        data = self._last_data or []
        result = next((r for r in data if r.id == benchmark_id), None)
        if result is None:
            self.app.notify("Benchmark not found.", severity="warning")
            return
        baseline = next(
            (r for r in data if r.id == self._baseline_id), None
        ) if self._baseline_id else None
        self.app.push_screen(
            BenchmarkInspectModal(
                f"Benchmark: {result.name}", self._inspect_lines(result, baseline)
            )
        )

    @staticmethod
    def _inspect_lines(result: Any, baseline: Any | None) -> list[str]:
        """Build the §3D inspection view: what the number was made of."""
        params = result.parameters or {}
        mode = str(params.get("mode", "?"))
        lines = [
            f"kind: {result.kind.value if result.kind else '-'}   mode: {mode}"
            f"   engine/backend: {esc(result.backend or '-')}",
            f"success: {'yes' if result.success else 'no'}"
            + (f"   error: {esc(str(result.error))}" if result.error else ""),
        ]
        request_count = params.get("request_count")
        if request_count is not None:
            failed = int(params.get("failed_request_count", 0) or 0)
            sample_note = (
                f"{request_count} request(s)"
                + (f", {failed} failed" if failed else "")
            )
        else:
            sample_note = f"{len(result.samples or [])} sample(s)"
        lines.append(f"samples: {sample_note}   concurrency: "
                     f"{params.get('concurrency', 1)}"
                     + (
                         f" (achieved {params.get('achieved_concurrency')})"
                         if params.get("achieved_concurrency") is not None
                         else ""
                     ))
        source = str(params.get("token_count_source", ""))
        if source:
            label = {
                "server_usage": "server usage (precise)",
                "estimate": "estimated (no server usage)",
                "mixed": "mixed (precise + estimated)",
                "synthetic": "synthetic (mock run)",
            }.get(source, source)
            lines.append(f"token counts: {label}")
        est = "~" if source in {"estimate", "mixed"} else ""
        lines.append(
            "throughput: "
            + (
                f"{result.tokens_per_second:.1f}{est} tok/s"
                if result.tokens_per_second is not None
                else "n/a"
            )
            + "   latency: "
            + (
                f"{result.latency_ms:.1f} ms"
                if result.latency_ms is not None
                else "n/a"
            )
        )
        percentiles = params.get("percentiles") or {}
        for metric in ("latency_ms", "ttft_ms"):
            stats = percentiles.get(metric) if isinstance(percentiles, dict) else None
            if not stats:
                continue
            note = " (few samples)" if stats.get("insufficient_samples") else ""
            lines.append(
                f"{metric} p50/p90/p99: {stats.get('p50')} / {stats.get('p90')} / "
                f"{stats.get('p99')} ms  (n={stats.get('n')}){note}"
            )
        warnings = params.get("warnings") or []
        for warning in warnings:
            lines.append(f"[{C_WARN}]warning: {esc(str(warning))}[/]")
        manifest = getattr(result, "manifest", None) or {}
        if manifest:
            workload = manifest.get("workload") or {}
            served = manifest.get("served") or {}
            gpus = (manifest.get("serving_host") or {}).get("gpu_identity")
            lines.append(
                f"manifest: {manifest.get('execution', '?')} · workload "
                f"{str(workload.get('hash', ''))[:12]} · endpoint "
                f"{esc(str(served.get('endpoint_url', '-')))} · "
                f"hardware {esc(', '.join(gpus) if gpus else 'unknown')}"
            )
        else:
            lines.append(
                f"[{C_MUTED}]manifest: unknown (record predates manifests)[/]"
            )
        rerun_of = params.get("rerun_of")
        if rerun_of:
            lines.append(f"re-run of: {rerun_of}")
        for entry in params.get("drift", []) or []:
            lines.append(f"[{C_WARN}]drift: {esc(str(entry))}[/]")
        if baseline is not None and baseline.id != result.id:
            verdict = comparison_verdict(result, baseline)
            if verdict.eligible:
                lines.append(f"[{C_OK}]vs baseline: comparable[/]")
            else:
                lines.append(
                    f"[{C_WARN}]vs baseline: not comparable — "
                    f"{esc(verdict.summary)}[/]"
                )
        return lines

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

    def action_delete_benchmark(self) -> None:
        """Confirm and hard-delete the cursor-row benchmark."""
        benchmark_id = self._selected_id()
        if not benchmark_id:
            if not self._ids:
                self.app.notify(
                    "No benchmarks yet. Press 'n' to launch one.",
                    severity="warning",
                )
            else:
                self.app.notify(
                    "Highlight a row with arrow keys, then press Delete.",
                    severity="warning",
                )
            return

        label = self._names.get(benchmark_id) or benchmark_id

        def _on_close(payload: ConfirmDelete | None) -> None:
            if payload is None:
                return
            if self._baseline_id == benchmark_id:
                self._baseline_id = None
            self.run_action_worker(
                lambda: _data.delete_benchmark(benchmark_id),
                lambda removed: self._after_delete(label, removed),
            )

        self.app.push_screen(
            DeleteModal(
                f"benchmark run '{label}'",
                "The benchmark row is permanently removed from the database. "
                "This is a hard delete, not a soft-delete.",
            ),
            _on_close,
        )

    def _after_delete(self, benchmark_id: str, removed: bool) -> None:
        """Refresh and notify after a delete completes."""
        if removed:
            self.app.notify(
                f"Deleted benchmark {benchmark_id!r}.", title="Benchmarks"
            )
        else:
            self.app.notify(
                f"Benchmark {benchmark_id!r} was already gone.", severity="warning"
            )
        self.refresh_data()

    def action_new_benchmark(self) -> None:
        """Open the launch modal, then dispatch the selected run off-thread."""
        models = self.query_data(_data.get_models, what="Model lookup")
        if models is None:
            return
        modal = BenchmarkLaunchModal(
            models, preselect_model_id=self._model_filter
        )
        self.app.push_screen(modal, self._on_launch_chosen)

    def _on_launch_chosen(self, launch: BenchmarkLaunch | None) -> None:
        """Dispatch the benchmark the operator picked from the modal."""
        if launch is None:
            return
        self.app.notify(
            f"Running '{launch.name}' ({launch.kind.value})...",
            title="Benchmark queued",
        )
        self.run_action_worker(
            lambda: _data.run_benchmark(
                name=launch.name,
                model_id=launch.model_id,
                kind=launch.kind,
                context_length=launch.context_length,
                max_tokens=launch.max_tokens,
                repetitions=launch.repetitions,
                dry_run=launch.dry_run,
                require_live=launch.require_live,
            ),
            self._after_launch,
        )

    def _after_launch(self, result: Any) -> None:
        """Surface the outcome of a launched benchmark, then refresh."""
        if result is None:
            self.app.notify("Benchmark failed to start.", severity="error")
        elif not result.success:
            self.app.notify(
                f"'{result.name}' failed: {result.error or 'unknown error'}",
                severity="error",
                title="Benchmark failed",
            )
        else:
            mode = result.parameters.get("mode", "?")
            tps = (
                "n/a"
                if result.tokens_per_second is None
                else f"{result.tokens_per_second:.1f} tok/s"
            )
            self.app.notify(
                f"'{result.name}' ({mode}): {tps}",
                title="Benchmark complete",
            )
        self.refresh_data()
