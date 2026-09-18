"""GPU telemetry TUI screen with live data and graceful no-GPU fallback."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import DataTable, Footer, Header, Static

from llmctl.tui import _data
from llmctl.tui._base import C_MUTED, C_OK, C_WARN, DataScreen, esc

_NONE = "—"


def gpu_row_cells(gpu: Any) -> tuple[str, ...]:
    """Format one GPU as table cells: units spelled out, unknowns as —.

    MiB counts like ``12288/32607 MiB`` and ``55C`` were hard to read at a
    glance, and unknowns showed as three different words (unknown/n/a/-).
    """
    used, total = gpu.memory_used_mb, gpu.memory_total_mb
    memory = (
        f"{used / 1024:.1f} / {total / 1024:.1f} GiB"
        if used is not None and total is not None
        else _NONE
    )
    util = _NONE if gpu.utilization_gpu_percent is None else f"{gpu.utilization_gpu_percent}%"
    temp = _NONE if gpu.temperature_c is None else f"{gpu.temperature_c} °C"
    power = _NONE if gpu.power_draw_watts is None else f"{gpu.power_draw_watts:.0f} W"
    return (
        str(gpu.index),
        esc(gpu.name or _NONE),
        memory,
        util,
        temp,
        power,
        str(len(gpu.processes or [])),
    )


class GPUScreen(DataScreen):
    """GPU telemetry screen backed by NVML (or a clear fallback note)."""

    def compose(self) -> ComposeResult:
        """Compose the GPU table and status note with screen-scoped chrome."""
        yield Header()
        yield Static("NVIDIA GPUs  -  live telemetry", classes="panel safe", id="gpus-title")
        yield Static("", classes="panel muted", id="gpus-note")
        # No row cursor: rows have no action behind them, and a movable
        # highlight suggested Enter would open something.
        table: DataTable[str] = DataTable(id="gpus-table", cursor_type="none")
        table.add_columns("GPU", "Name", "Memory (used / total)", "Util", "Temp", "Power", "Procs")
        yield table
        yield Footer()

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
            table.add_row(*gpu_row_cells(gpu))
