"""Textual application: live mission-control TUI."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen

from llmctl.tui.screens_benchmarks import BenchmarksScreen
from llmctl.tui.screens_dashboard import DashboardScreen
from llmctl.tui.screens_doctor import DoctorScreen
from llmctl.tui.screens_gpu import GPUScreen
from llmctl.tui.screens_logs import LogsScreen
from llmctl.tui.screens_models import ModelsScreen
from llmctl.tui.screens_presets import PresetsScreen
from llmctl.tui.screens_profiles import ProfilesScreen
from llmctl.tui.screens_sessions import SessionsScreen
from llmctl.tui.screens_units import UnitsScreen

#: Seconds between automatic data refreshes of the active screen.
REFRESH_INTERVAL = 3.0


class MissionControlApp(App[None]):
    """Terminal mission-control UI with live backend data binding."""

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
    """
    TITLE = "LLM Mission Control"
    SUB_TITLE = "Live local runtime control"
    BINDINGS = [
        Binding("d", "show_dashboard", "Dashboard", show=True),
        Binding("p", "show_presets", "Presets", show=True),
        Binding("u", "show_units", "Units", show=True),
        Binding("m", "show_models", "Models", show=True),
        Binding("f", "show_profiles", "Profiles", show=True),
        Binding("s", "show_sessions", "Sessions", show=True),
        Binding("g", "show_gpus", "GPUs", show=True),
        Binding("l", "show_logs", "Logs", show=True),
        Binding("o", "show_doctor", "Doctor", show=True),
        Binding("b", "show_benchmarks", "Benchmarks", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("q", "quit", "Quit", show=True),
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
        self.set_interval(REFRESH_INTERVAL, self._auto_refresh)

    def _auto_refresh(self) -> None:
        """Refresh the active screen's data if it supports it."""
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

    def on_screen_resume(self, event: Screen.ScreenResume) -> None:  # type: ignore[name-defined]
        """Refresh a screen whenever it becomes active."""
        self.action_refresh()

    def action_show_dashboard(self) -> None:
        """Show dashboard screen."""
        self._switch("dashboard")

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
