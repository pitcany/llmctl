"""GPU telemetry TUI screen with live data and graceful no-GPU fallback."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import DataTable, Static

from llmctl.tui import _data
from llmctl.tui._base import C_MUTED, C_OK, C_WARN, DataScreen


class GPUScreen(DataScreen):
    """GPU telemetry screen backed by NVML (or a clear fallback note)."""

    def compose(self) -> ComposeResult:
        """Compose the GPU table and status note."""
        yield Static("NVIDIA GPUs  -  live telemetry", classes="panel safe", id="gpus-title")
        yield Static("", classes="panel muted", id="gpus-note")
        table: DataTable[str] = DataTable(id="gpus-table", cursor_type="row")
        table.add_columns("Idx", "Name", "Memory (used/total)", "Util", "Temp", "Power", "Procs")
        yield table

    def fetch(self) -> Any:
        """Return GPU telemetry (empty on non-NVIDIA hosts)."""
        return _data.get_gpus()

    def render_data(self, data: Any) -> None:
        """Render the GPU telemetry table."""
        note = self.query_one("#gpus-note", Static)
        table = self.query_one("#gpus-table", DataTable)
        table.clear()
        if not data:
            note.update(
                f"[{C_WARN}]No NVIDIA GPU telemetry available.[/] "
                f"[{C_MUTED}]Running in fallback mode (no NVML / no NVIDIA driver).[/]"
            )
            return
        note.update(f"[{C_OK}]{len(data)} GPU(s) detected via NVML.[/]")
        for gpu in data:
            memory = "unknown"
            if gpu.memory_used_mb is not None and gpu.memory_total_mb is not None:
                memory = f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MiB"
            util = (
                "unknown"
                if gpu.utilization_gpu_percent is None
                else f"{gpu.utilization_gpu_percent}%"
            )
            temp = "n/a" if gpu.temperature_c is None else f"{gpu.temperature_c}C"
            power = "n/a" if gpu.power_draw_watts is None else f"{gpu.power_draw_watts:.0f}W"
            table.add_row(
                str(gpu.index),
                gpu.name,
                memory,
                util,
                temp,
                power,
                str(len(gpu.processes)),
            )
