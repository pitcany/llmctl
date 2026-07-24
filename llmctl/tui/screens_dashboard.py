"""Dashboard TUI screen with live overview metrics."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid
from textual.widgets import Footer, Header, Static

from llmctl.tui import _data
from llmctl.tui._base import C_ACCENT, C_MUTED, C_OK, C_WARN, DataScreen, esc


class StatCard(Static):
    """A single labeled metric tile.

    Pre-renders the title + a placeholder in ``__init__`` so the card
    is never visually empty before the first ``set_value`` call. Before
    this, a fresh mount or any silent fetch failure left the user
    staring at six unlabeled boxes — the original ``super().__init__()``
    passed no renderable, and Textual's ``Static`` defaulted that to an
    empty string that the panel-bordered card rendered as a bare frame.
    """

    _PLACEHOLDER = "..."

    def __init__(self, title: str, card_id: str) -> None:
        self._title = title
        # Set the renderable in the super constructor AND via on_mount so
        # both paths produce visible content. Passing it only as a
        # constructor arg leaves the widget rendering an empty body
        # until update() is called — verified empirically by snapping
        # the dashboard and finding the cards visually blank while
        # ``.content`` reported the correct markup at the Python level.
        self._placeholder_markup = self._compose_markup(self._PLACEHOLDER, C_MUTED)
        super().__init__(
            self._placeholder_markup,
            id=card_id,
            classes="panel stat-card",
        )

    def on_mount(self) -> None:
        """Force-apply the placeholder markup once the widget is in the DOM."""
        self.update(self._placeholder_markup)

    def _compose_markup(self, value: str, color: str) -> str:
        """Render the two-line markup body. Title on top, value below."""
        return f"[b]{self._title}[/b]\n[{color}][b]{value}[/b][/]"

    def set_value(self, value: str, *, color: str = C_ACCENT) -> None:
        """Replace the placeholder with the live value."""
        self.update(self._compose_markup(value, color))


class DashboardScreen(DataScreen):
    """Mission-control dashboard with live counts and runtime health."""

    def compose(self) -> ComposeResult:
        """Compose the dashboard layout.

        Yields ``Header()`` and ``Footer()`` at screen scope. In Textual's
        push_screen model, the App's own ``compose`` chrome is hidden when
        a screen is pushed on top — so each screen must yield its own
        chrome if it wants the keybinding hints visible at the bottom.
        Otherwise the user sees the dashboard with no indication of what
        keys do anything.
        """
        yield Header()
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
        yield Footer()

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
                f"  [{color}]*[/] {esc(name):<14} {state:<12} [{C_MUTED}]{esc(message)}[/]"
            )
        warnings = data.get("scheduler_warnings", [])
        if warnings:
            lines.append("\n[b]Scheduler warnings[/b]")
            for warning in warnings:
                lines.append(f"  [{C_WARN}]![/] {esc(warning)}")
        router = data.get("router") or {}
        if router:
            lines.append("\n[b]Router gateway[/b]")
            running = router.get("running")
            color = C_OK if running else C_WARN
            state = "running" if running else "stopped"
            host = router.get("host", "127.0.0.1")
            port = router.get("port", "?")
            auth = "auth" if router.get("auth_required") else "no-auth"
            lines.append(f"  [{color}]*[/] http://{host}:{port}  {state}  {auth}")
            aliases = router.get("aliases") or []
            bound = [a for a in aliases if a.get("target")]
            if not aliases:
                lines.append(f"  [{C_MUTED}]No aliases configured.[/]")
            else:
                for entry in aliases:
                    target = entry.get("target") or "-"
                    healthy = entry.get("healthy")
                    a_color = (
                        C_OK if healthy else C_WARN if entry.get("target") else C_MUTED
                    )
                    session_id = entry.get("session_id") or "-"
                    lines.append(
                        f"  [{a_color}]·[/] {esc(entry['name']):<14} -> {esc(target)}  "
                        f"[{C_MUTED}]session={esc(session_id)}[/]"
                    )
                lines.append(
                    f"  [{C_MUTED}]{len(bound)}/{len(aliases)} aliases bound.[/]"
                )
        gpu_note = (
            "NVML detected" if data["nvml_available"] else "No NVIDIA GPU / NVML (fallback mode)"
        )
        lines.append(f"\n[{C_MUTED}]Telemetry: {gpu_note}.[/]")
        self.query_one("#dashboard-runtimes", Static).update("\n".join(lines))
