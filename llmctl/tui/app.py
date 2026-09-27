"""Textual application: live mission-control TUI."""

from __future__ import annotations

from collections.abc import Iterable

from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.screen import Screen

from llmctl.tui._base import esc
from llmctl.tui._modals import ConfirmActionModal, TextViewModal
from llmctl.tui.screens_benchmarks import BenchmarksScreen
from llmctl.tui.screens_dashboard import DashboardScreen
from llmctl.tui.screens_doctor import DoctorScreen
from llmctl.tui.screens_gpu import GPUScreen
from llmctl.tui.screens_logs import LogsScreen
from llmctl.tui.screens_models import ModelsScreen
from llmctl.tui.screens_presets import PresetsScreen
from llmctl.tui.screens_profiles import ProfilesScreen
from llmctl.tui.screens_sessions import SessionsScreen
from llmctl.tui.screens_swap import SwapScreen
from llmctl.tui.screens_units import UnitsScreen

#: Seconds between automatic data refreshes of the active screen.
REFRESH_INTERVAL = 3.0

#: (screen name, palette title, what it shows, key). One table drives both
#: the command palette and the help modal so they cannot drift apart.
SCREENS: tuple[tuple[str, str, str, str], ...] = (
    ("dashboard", "Dashboard", "Overview: runtimes, sessions, GPUs at a glance", "d"),
    ("swap", "Swap", "Every model that can serve (vLLM + lanes); switch to one", "w"),
    ("presets", "Presets", "vLLM presets; launch one onto the TP fleet", "p"),
    ("units", "Units", "Managed systemd units and what they serve", "u"),
    ("models", "Models", "Add or remove models for any engine (a, del)", "m"),
    ("profiles", "Profiles", "Reusable launch parameters", "f"),
    ("sessions", "Sessions", "Running and adopted sessions", "s"),
    ("gpus", "GPUs", "Per-GPU utilization, memory, temperature", "g"),
    ("logs", "Events", "Recent llmctl events", "l"),
    ("doctor", "Doctor", "Backend install state and fixes", "o"),
    ("benchmarks", "Benchmarks", "Benchmark history", "b"),
)


def _binding_rows(bindings: Iterable[object]) -> list[tuple[str, str]]:
    """(key, description) for bindings given as ``Binding`` or tuples."""
    rows: list[tuple[str, str]] = []
    for b in bindings:
        if isinstance(b, Binding):
            if b.action in {"ctrl_c_hint"}:
                continue
            rows.append((b.key_display or b.key, b.description or b.action))
        elif isinstance(b, tuple) and len(b) >= 3:
            rows.append((str(b[0]), str(b[2])))
    return rows


def help_text(screen: object) -> str:
    """Markup for the help modal: screens, global keys, this screen's keys."""
    lines = ["[b]Screens[/b]"]
    for _name, title, what, key in SCREENS:
        lines.append(f"  [b]{key}[/b]  {title:<11} {esc(what)}")
    lines.append("\n[b]Everywhere[/b]")
    lines.append("  [b]r[/b]  refresh   [b]?[/b] / [b]F1[/b]  this help   "
                 "[b]ctrl+\\[/b]  command palette   [b]q[/b]  quit")
    own = _binding_rows(getattr(type(screen), "BINDINGS", []))
    if own:
        lines.append(f"\n[b]This screen ({esc(type(screen).__name__)})[/b]")
        for key, desc in own:
            lines.append(f"  [b]{esc(key)}[/b]  {esc(desc)}")
    return "\n".join(lines)


class MissionControlApp(App[None]):
    """Terminal mission-control UI with live backend data binding."""

    #: Action workers currently running, across every screen. Screen-level
    #: ``_action_busy`` only knows about its own screen, but a launch started
    #: on Presets must still be announced if the operator quits from Models.
    _actions_in_flight: int = 0

    CSS = """
    Screen { background: #071113; color: #d8f3f0; }
    .panel { border: round #2dd4bf; padding: 1; margin: 1; }
    .safe { color: #5eead4; text-style: bold; }
    .muted { color: #8aa6a3; }
    .ok { color: #86efac; text-style: bold; }
    .warn { color: #fbbf24; text-style: bold; }
    .err { color: #fb7185; text-style: bold; }
    .stat-value { color: #5eead4; text-style: bold; }

    #stat-grid {
        grid-size: 3 2;
        grid-gutter: 1;
        /* Each row needs enough vertical space for: border (2 rows) +
           padding (2 rows) + 2 content rows = 6 minimum. Pad to 7 so a
           narrow terminal doesn't clip the value line. Without explicit
           grid-rows, Textual's grid was sizing rows to fit the container's
           remaining height, collapsing card content to zero rows. */
        grid-rows: 7 7;
        height: auto;
        margin: 0 1;
    }
    /* Override .panel's padding for stat cards specifically: 0 vertical
       padding (border alone is enough chrome) so both title and value
       fit inside the 7-row card body. */
    .stat-card {
        content-align: left top;
        padding: 0 1;
    }

    DataTable { margin: 1; height: 1fr; }
    DataTable > .datatable--header { background: #0c1a1c; text-style: bold; }
    DataTable > .datatable--cursor { background: #134e4a; }

    #session-log-title { margin: 0 2; color: #5eead4; text-style: bold; }
    #session-log-wrap { height: 10; margin: 0 1 1 1; }

    LaunchPlanModal { align: center middle; }
    #plan-dialog { width: 80; height: auto; max-height: 90%; background: #0c1a1c; }
    #plan-buttons { height: auto; margin-top: 1; }
    #plan-buttons Button { width: 100%; margin-bottom: 1; }

    TextViewModal { align: center middle; }
    #textview-dialog { width: 90%; height: 80%; background: #0c1a1c; }
    #textview-scroll { height: 1fr; }

    AddModelModal { align: center middle; }
    #add-dialog { width: 90; height: auto; max-height: 95%; background: #0c1a1c; }
    #add-preview { margin-top: 1; height: auto; }
    #add-examples { margin-top: 1; }
    #add-buttons { height: auto; margin-top: 1; }
    #add-buttons Button { margin-right: 2; }

    ConfirmActionModal { align: center middle; }
    #confirm-dialog { width: 70; height: auto; max-height: 90%; background: #0c1a1c; }
    #confirm-consequence { margin-top: 1; color: #d8f3f0; }
    #confirm-buttons { height: auto; margin-top: 1; }
    #confirm-buttons Button { margin-right: 2; }
    """
    TITLE = "LLM Mission Control"
    SUB_TITLE = "Live local runtime control"
    # Textual's command palette defaults to ctrl+p, but that chord is widely
    # intercepted upstream of the app (VS Code's "Quick Open", some terminals)
    # so over SSH the keystroke never reaches Textual and the advertised
    # "palette" footer entry does nothing. Rebind to ctrl+\ — a chord
    # terminals reliably forward — so the palette is actually reachable.
    COMMAND_PALETTE_BINDING = "ctrl+backslash"
    BINDINGS = [
        Binding("d", "show_dashboard", "Dashboard", show=True),
        Binding("w", "show_swap", "Swap", show=True),
        Binding("p", "show_presets", "Presets", show=True),
        Binding("u", "show_units", "Units", show=True),
        Binding("m", "show_models", "Models", show=True),
        Binding("f", "show_profiles", "Profiles", show=True),
        Binding("s", "show_sessions", "Sessions", show=True),
        Binding("g", "show_gpus", "GPUs", show=True),
        Binding("l", "show_logs", "Events", show=True),
        Binding("o", "show_doctor", "Doctor", show=True),
        Binding("b", "show_benchmarks", "Benchmarks", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("question_mark", "show_help", "Help", show=True, key_display="?"),
        Binding("f1", "show_help", "Help", show=False),
        Binding("q", "quit", "Quit", show=True),
        # ctrl+c does not quit a Textual app; say what does instead of
        # silently doing nothing. Focused inputs keep ctrl+c for copy.
        Binding("ctrl+c", "ctrl_c_hint", "Quit hint", show=False),
    ]

    def compose(self) -> ComposeResult:
        """The app body is empty — each screen owns its own Header/Footer.

        In Textual's push_screen model an installed screen replaces the
        visible area, so widgets yielded at app scope (Header/Footer
        included) would disappear the moment any screen was pushed.
        Each :class:`DataScreen` subclass yields its own chrome for that
        reason; this method intentionally produces nothing.
        """
        return
        yield  # unreachable — keeps the generator/ComposeResult typing honest

    def on_mount(self) -> None:
        """Install screens, show the dashboard, and start the refresh timer."""
        self.install_screen(DashboardScreen(), name="dashboard")
        self.install_screen(SwapScreen(), name="swap")
        self.install_screen(PresetsScreen(), name="presets")
        self.install_screen(UnitsScreen(), name="units")
        self.install_screen(ModelsScreen(), name="models")
        self.install_screen(ProfilesScreen(), name="profiles")
        self.install_screen(SessionsScreen(), name="sessions")
        self.install_screen(GPUScreen(), name="gpus")
        self.install_screen(LogsScreen(), name="logs")
        self.install_screen(DoctorScreen(), name="doctor")
        self.install_screen(BenchmarksScreen(), name="benchmarks")
        self.push_screen("dashboard")
        self.set_interval(REFRESH_INTERVAL, self._tick_refresh)

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        """Command palette: every screen is reachable by name ("Go to ...")."""
        yield from super().get_system_commands(screen)
        for name, title, help_text, key in SCREENS:
            yield SystemCommand(
                f"Go to {title}",
                f"{help_text}  (key: {key})",
                lambda name=name: self._switch(name),
            )
        yield SystemCommand(
            "Refresh", "Reload the current screen now  (key: r)", self.action_refresh
        )

    def action_show_help(self) -> None:
        """Show every key: global navigation plus the current screen's own."""
        self.push_screen(TextViewModal("Keys", help_text(self.screen), markup=True))

    def action_ctrl_c_hint(self) -> None:
        """ctrl+c is not quit here; point at the key that is."""
        self.notify("Press q to quit (ctrl+\\ opens the command palette).", title="Quit")

    def _tick_refresh(self) -> None:
        """Refresh the active screen's data if it supports it.

        Not named ``_auto_refresh``: Textual's ``DOMNode.__init__`` assigns an
        instance attribute of that name, which shadows the method and silently
        registers the interval timer with a ``None`` callback.
        """
        self.action_refresh()

    def action_refresh(self) -> None:
        """Refresh the currently active screen's data."""
        screen = self.screen
        refresh = getattr(screen, "refresh_data", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as exc:  # noqa: BLE001 - keep the UI alive on data errors
                self.notify(f"Refresh error: {exc}", severity="error")

    def _switch(self, name: str) -> None:
        """Switch to a named screen and refresh it immediately."""
        self.switch_screen(name)

    async def action_quit(self) -> None:
        """Quit, but confirm first if an action is still running.

        Action workers are daemon threads, so quitting is immediate and does
        *not* wait for them — which is the point, but it also means a launch
        can be abandoned part-way. One that has already stopped ollama but not
        yet brought vLLM up would leave the machine serving nothing, so say so
        before letting the operator walk away from it.
        """
        if self._actions_in_flight:

            def _on_close(confirmed: bool | None) -> None:
                if confirmed:
                    self.exit()

            self.push_screen(
                ConfirmActionModal(
                    "Quit while an action is running?",
                    "It will be abandoned part-way. A preset launch that has "
                    "already stopped ollama but not finished starting vLLM "
                    "would leave the machine with nothing serving.",
                    confirm_label="Quit anyway",
                ),
                _on_close,
            )
            return
        await super().action_quit()

    def action_show_dashboard(self) -> None:
        """Show dashboard screen."""
        self._switch("dashboard")

    def action_show_swap(self) -> None:
        """Show the swap screen."""
        self._switch("swap")

    def action_show_presets(self) -> None:
        """Show presets screen."""
        self._switch("presets")

    def action_show_units(self) -> None:
        """Show managed units screen."""
        self._switch("units")

    def action_show_models(self) -> None:
        """Show models screen."""
        self._switch("models")

    def action_show_profiles(self) -> None:
        """Show DB-backed profiles screen."""
        self._switch("profiles")

    def action_show_sessions(self) -> None:
        """Show sessions screen."""
        self._switch("sessions")

    def action_show_gpus(self) -> None:
        """Show GPUs screen."""
        self._switch("gpus")

    def action_show_logs(self) -> None:
        """Show logs screen."""
        self._switch("logs")

    def action_show_doctor(self) -> None:
        """Show doctor (backend diagnostics) screen."""
        self._switch("doctor")

    def action_show_benchmarks(self) -> None:
        """Show benchmarks history screen."""
        self._switch("benchmarks")
