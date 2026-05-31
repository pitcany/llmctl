"""Dashboard TUI screen with live overview metrics."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid
from textual.widgets import Static

from llmctl.tui import _data
from llmctl.tui._base import C_ACCENT, C_MUTED, C_OK, C_WARN, DataScreen


class StatCard(Static):
    """A single labeled metric tile."""

    def __init__(self, title: str, card_id: str) -> None:
        super().__init__(id=card_id, classes="panel stat-card")
        self._title = title

    def set_value(self, value: str, *, color: str = C_ACCENT) -> None:
        """Render the card title and value."""
        self.update(f"[b]{self._title}[/b]\n[{color}][b]{value}[/b][/]")


class DashboardScreen(DataScreen):
    """Mission-control dashboard with live counts and runtime health."""

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield Static(
            "LLM MISSION CONTROL  -  live overview",
            classes="panel safe",
            id="dashboard-title",
        )
        with Grid(id="stat-grid"):
            yield StatCard("Models", "card-models")
            yield StatCard("Sessions running", "card-running")
            yield StatCard("Sessions total", "card-sessions")
            yield StatCard("Profiles", "card-profiles")
            yield StatCard("GPUs", "card-gpus")
            yield StatCard("Safe mode", "card-safe")
        yield Static("", classes="panel muted", id="dashboard-runtimes")

    def fetch(self) -> Any:
        """Return the overview snapshot."""
        return _data.get_overview()

    def render_data(self, data: Any) -> None:
        """Render the overview snapshot into the metric cards."""
        self.query_one("#card-models", StatCard).set_value(str(data["models"]))
        self.query_one("#card-running", StatCard).set_value(
            str(data["sessions_running"]),
            color=C_OK if data["sessions_running"] else C_ACCENT,
        )
        self.query_one("#card-sessions", StatCard).set_value(str(data["sessions_total"]))
        self.query_one("#card-profiles", StatCard).set_value(str(data["profiles"]))
        self.query_one("#card-gpus", StatCard).set_value(
            str(data["gpu_count"]),
            color=C_OK if data["gpu_count"] else C_WARN,
        )
        self.query_one("#card-safe", StatCard).set_value(
            "ON" if data["safe_mode"] else "OFF",
            color=C_OK if data["safe_mode"] else C_WARN,
        )

        runtimes = data.get("runtimes", {})
        lines = ["[b]Runtime health[/b]"]
        if not runtimes:
            lines.append(f"[{C_MUTED}]No runtime adapters reported.[/]")
        for name, info in runtimes.items():
            state = info.get("state", "unknown")
            color = C_OK if state == "ok" else C_WARN
            message = info.get("message", "")
            lines.append(
                f"  [{color}]*[/] {name:<14} {state:<12} [{C_MUTED}]{message}[/]"
            )
        warnings = data.get("scheduler_warnings", [])
        if warnings:
            lines.append("\n[b]Scheduler warnings[/b]")
            for warning in warnings:
                lines.append(f"  [{C_WARN}]![/] {warning}")
        gpu_note = (
            "NVML detected" if data["nvml_available"] else "No NVIDIA GPU / NVML (fallback mode)"
        )
        lines.append(f"\n[{C_MUTED}]Telemetry: {gpu_note}.[/]")
        self.query_one("#dashboard-runtimes", Static).update("\n".join(lines))
