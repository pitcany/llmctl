"""Dashboard TUI screen: the research-workflow landing view (FIXPLAN §3A).

Answers, in priority order: what is serving now, what operation is running,
and how the recent experiments turned out. Inventory totals and
configuration flags are secondary detail further down. State is always
signaled with icon + text, never color alone.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Grid
from textual.widgets import Footer, Header, Static

from llmctl.schemas import HealthState
from llmctl.services.operations import OperationPhase
from llmctl.tui import _data
from llmctl.tui._base import C_ACCENT, C_ERR, C_MUTED, C_OK, C_WARN, DataScreen, esc


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


#: Text+icon state markers (FIXPLAN §3A): color is never the only signal.
STATE_ICONS = {
    "serving": ("●", C_OK),
    "loading": ("◐", C_WARN),
    "stopped": ("○", C_MUTED),
    "unknown": ("?", C_MUTED),
}


def _state_cell(state: str) -> str:
    icon, color = STATE_ICONS.get(state, STATE_ICONS["unknown"])
    return f"[{color}]{icon} {esc(state)}[/]"


def _duration_label(seconds: Any) -> str:
    from llmctl.services.lanes import format_duration

    return format_duration(seconds) if seconds else "—"


class DashboardScreen(DataScreen):
    """Research-workflow landing view: serving, operation, experiments."""

    def compose(self) -> ComposeResult:
        """Compose the landing layout (chrome at screen scope).

        Yields ``Header()`` and ``Footer()`` at screen scope. In Textual's
        push_screen model, the App's own ``compose`` chrome is hidden when
        a screen is pushed on top — so each screen must yield its own
        chrome if it wants the keybinding hints visible at the bottom.
        """
        yield Header()
        yield Static(
            "LLM MISSION CONTROL  -  research overview",
            classes="panel safe",
            id="dashboard-title",
        )
        yield Static("…", classes="panel", id="dashboard-serving")
        yield Static("…", classes="panel", id="dashboard-operation")
        yield Static("…", classes="panel", id="dashboard-experiments")
        with Grid(id="stat-grid"):
            yield StatCard("Models", "card-models")
            yield StatCard("Running", "card-running")
            yield StatCard("Profiles", "card-profiles")
            yield StatCard("GPUs", "card-gpus")
        yield Static("", classes="panel muted", id="dashboard-runtimes")
        yield Footer()

    def fetch(self) -> Any:
        """Return the landing snapshot."""
        return _data.get_landing()

    def render_data(self, data: Any) -> None:
        """Render serving / operation / experiments, then the detail rows."""
        self._render_serving(data)
        self._render_operation(data)
        self._render_experiments(data)
        self._render_details(data)

    # -- Serving now --------------------------------------------------------

    def _render_serving(self, data: Any) -> None:
        lines: list[str] = ["[b]Serving now[/b]"]
        serving = data.get("serving") or []
        fetched_at = str(data.get("fetched_at", ""))
        observed = f"  [{C_MUTED}]observed {esc(fetched_at)}[/]" if fetched_at else ""
        if not serving:
            lines.append(f"[{C_MUTED}]nothing serving{observed}[/]")
        else:
            lines[0] += observed
            for row in serving:
                state = str(row.get("state", "unknown"))
                endpoint = row.get("endpoint") or "endpoint unknown"
                # Historical load time: labeled as history, never a promise
                # of how long the next load takes.
                loaded = row.get("last_load_s")
                history = (
                    f"[{C_MUTED}]loaded in {_duration_label(loaded)} (historical)[/]"
                    if loaded
                    else ""
                )
                lines.append(
                    f"  {_state_cell(state)} {esc(row.get('model', '?'))} "
                    f"[{C_MUTED}]({esc(row.get('engine', '?'))}, "
                    f"{esc(row.get('key', '?'))})[/]"
                )
                lines.append(f"      {esc(endpoint)}  {history}")
        self.query_one("#dashboard-serving", Static).update("\n".join(lines))

    # -- Current operation --------------------------------------------------

    def _render_operation(self, data: Any) -> None:
        op = data.get("operation")
        lines = ["[b]Current operation[/b]"]
        if op is None:
            lines.append(f"[{C_MUTED}]none[/]")
        else:
            phase = str(op.get("phase", "?"))
            if op.get("terminal"):
                if phase == OperationPhase.COMPLETED:
                    icon, color = "✓", C_OK
                elif phase in (OperationPhase.NEEDS_ATTENTION, OperationPhase.INTERRUPTED):
                    icon, color = "!", C_ERR
                else:
                    icon, color = "×", C_ERR
                lines.append(
                    f"  [{color}]{icon} last: {esc(phase)} — "
                    f"{esc(op.get('kind', '?'))} {esc(op.get('target', '?'))}[/]"
                )
            else:
                elapsed = op.get("elapsed_s")
                elapsed_label = (
                    f", {elapsed:.0f}s elapsed"
                    if isinstance(elapsed, (int, float))
                    else ""
                )
                lines.append(
                    f"  [{C_WARN}]◐ {esc(phase)}{elapsed_label}[/] — "
                    f"{esc(op.get('kind', '?'))} {esc(op.get('target', '?'))} "
                    f"[{C_MUTED}]via {esc(op.get('interface') or '?')} "
                    f"(pid {op.get('pid')})[/]"
                )
            recovery = op.get("recovery") or {}
            guidance = recovery.get("guidance")
            if guidance:
                lines.append(f"  [{C_ERR}]! {esc(guidance)}[/]")
            not_restored = recovery.get("not_restored") or []
            if not_restored:
                lines.append(
                    f"  [{C_ERR}]! not restored: "
                    f"{esc(', '.join(map(str, not_restored)))}[/]"
                )
        self.query_one("#dashboard-operation", Static).update("\n".join(lines))

    # -- Recent experiments -------------------------------------------------

    def _render_experiments(self, data: Any) -> None:
        lines = ["[b]Recent experiments[/b]"]
        experiments = data.get("recent_experiments") or []
        if not experiments:
            lines.append(f"[{C_MUTED}]none yet — press [b]b[/b] to run a benchmark[/]")
        for exp in experiments:
            ok = bool(exp.get("success"))
            mark = f"[{C_OK}]y[/]" if ok else f"[{C_ERR}]n[/]"
            mode = str(exp.get("mode", "?"))
            tps = exp.get("tokens_per_second")
            source = str(exp.get("token_count_source", ""))
            est = "~" if source in {"estimate", "mixed", "synthetic"} else ""
            tps_label = (
                f"{tps:.1f}{est} tok/s" if isinstance(tps, (int, float)) else "—"
            )
            when = str(exp.get("created_at") or "?")
            lines.append(
                f"  {mark} {esc(exp.get('name', '?'))}  "
                f"[{C_MUTED}]{esc(exp.get('kind', '?'))} · {mode} · "
                f"{tps_label} · {when}[/]"
            )
        self.query_one("#dashboard-experiments", Static).update("\n".join(lines))

    # -- Secondary detail ---------------------------------------------------

    def _render_details(self, data: Any) -> None:
        inventory = data.get("inventory") or {}
        self.query_one("#card-models", StatCard).set_value(
            str(inventory.get("models", 0))
        )
        running = int(inventory.get("sessions_running", 0) or 0)
        self.query_one("#card-running", StatCard).set_value(
            str(running), color=C_OK if running else C_ACCENT
        )
        self.query_one("#card-profiles", StatCard).set_value(
            str(inventory.get("profiles", 0))
        )
        gpus = int(inventory.get("gpus", 0) or 0)
        self.query_one("#card-gpus", StatCard).set_value(
            str(gpus), color=C_OK if gpus else C_WARN
        )

        runtimes = data.get("runtimes", {})
        lines: list[str] = []
        if not inventory.get("models"):
            lines.append(
                f"[{C_WARN}]No models registered yet.[/] Press [b]m[/b] for Models, "
                "then [b]a[/b] to add one or [b]ctrl+s[/b] to scan this machine.\n"
            )
        # safe_mode only tags launch plans; it blocks nothing. Secondary,
        # muted detail so it never reads as active protection.
        safe = "on" if data.get("safe_mode") else "off"
        lines.append(f"[{C_MUTED}]safe mode (label only): {safe}[/]\n")
        lines.append("[b]Runtime health[/b]")
        if not runtimes:
            lines.append(f"[{C_MUTED}]No runtime adapters reported.[/]")
        for name, info in runtimes.items():
            state = info.get("state", "unknown")
            color = (
                C_OK
                if state == HealthState.OK
                else C_MUTED if state == HealthState.NOT_USED else C_WARN
            )
            message = info.get("message", "")
            lines.append(
                f"  [{color}]*[/] {esc(name):<14} {state:<12} "
                f"[{C_MUTED}]{esc(message)}[/]"
            )
        warnings = data.get("scheduler_warnings", [])
        if warnings:
            lines.append("\n[b]Scheduler warnings[/b]")
            for warning in warnings:
                lines.append(f"  [{C_WARN}]![/] {esc(warning)}")
        router = data.get("router") or {}
        if router:
            lines.append("\n[b]Router gateway[/b]")
            running_router = router.get("running")
            color = C_OK if running_router else C_WARN
            state = "running" if running_router else "stopped"
            host = router.get("host", "127.0.0.1")
            port = router.get("port", "?")
            auth = "auth" if router.get("auth_required") else "no-auth"
            lines.append(f"  [{color}]*[/] http://{host}:{port}  {state}  {auth}")
            aliases = router.get("aliases") or []
            bound = [a for a in aliases if a.get("target")]
            if not aliases:
                lines.append(f"[{C_MUTED}]No aliases configured.[/]")
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
                lines.append(f"[{C_MUTED}]{len(bound)}/{len(aliases)} aliases bound.[/]")
        gpu_note = (
            "NVML detected"
            if data.get("nvml_available")
            else "No NVIDIA GPU / NVML (fallback mode)"
        )
        lines.append(f"\n[{C_MUTED}]Telemetry: {gpu_note}.[/]")
        self.query_one("#dashboard-runtimes", Static).update("\n".join(lines))
