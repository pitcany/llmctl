"""Shared base screen and color constants for the TUI.

Data fetching for every screen runs in a worker thread. This keeps the UI
responsive and, critically, gives blocking service calls their own context so
``asyncio.run`` (used by adapters/health) works without colliding with
Textual's running event loop. Widget updates are marshalled back to the UI
thread via ``call_from_thread``.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual import events
from textual.screen import Screen

#: Inline-markup colors (Textual content markup accepts hex color tags).
C_ACCENT = "#5eead4"
C_OK = "#86efac"
C_WARN = "#fbbf24"
C_ERR = "#fb7185"
C_MUTED = "#8aa6a3"


def esc(value: object) -> str:
    """Render ``value`` as literal text inside a Rich-markup string.

    Registry names, checkpoint paths and log output are attacker-agnostic but
    bracket-rich: ``[/INST]`` chat tokens, ``[rank0]:`` torch prefixes and
    ``[/mnt/...]`` paths all parse as markup. An unbalanced closing tag raises
    ``MarkupError``; a stray ``[word]`` is deleted from the display.
    """
    return escape(str(value))


class DataScreen(Screen[None]):
    """Screen that fetches data off-thread and renders on the UI thread."""

    #: True while an action worker is in flight. A class-level default so
    #: subclasses that never call ``DataScreen.__init__`` still read False.
    _action_busy: bool = False

    def on_mount(self) -> None:
        """Mount hook, kept so subclasses can extend it via ``super()``.

        The initial load deliberately does *not* happen here. Textual posts
        ``ScreenResume`` on first activation as well as on every switch-back,
        so :meth:`on_screen_resume` covers both with exactly one fetch;
        loading from both hooks would start two workers per screen entry,
        and a thread worker already handed to the executor still renders even
        when a newer one supersedes it.
        """

    def on_screen_resume(self, event: events.ScreenResume) -> None:
        """Load data whenever this screen becomes the active one.

        Lives on the screen, not the App: ``ScreenResume`` is declared
        ``bubble=False``, so an App-level handler never receives it.
        """
        self.refresh_data()

    def fetch(self) -> Any:
        """Return the data payload for this screen (runs in a worker thread)."""
        raise NotImplementedError

    def render_data(self, data: Any) -> None:
        """Render ``data`` into widgets (runs on the UI thread)."""
        raise NotImplementedError

    def refresh_data(self) -> None:
        """Schedule a threaded refresh of this screen's data."""
        self.run_worker(
            self._refresh_worker,
            thread=True,
            exclusive=True,
            group="refresh",
        )

    def _refresh_worker(self) -> None:
        """Worker body: fetch data, then marshal rendering to the UI thread.

        Both halves are guarded. ``run_worker`` defaults to
        ``exit_on_error=True``, and ``call_from_thread`` re-raises the
        callback's exception in this thread, so an unguarded fetch or render
        terminates the whole app — a transient SQLite lock on the registry
        file (shared with the CLI, the API and the gateway) is enough. Failing
        loudly but staying alive beats a traceback on the operator's terminal.
        """
        try:
            data = self.fetch()
        except Exception as exc:
            self.app.call_from_thread(self._notify_error, exc, "Refresh failed")
            return
        try:
            self.app.call_from_thread(self.render_data, data)
        except Exception as exc:
            self.app.call_from_thread(self._notify_error, exc, "Render failed")

    def run_action_worker(self, func: Any, after: Any) -> None:
        """Run a blocking action ``func`` off-thread, then ``after(result)``.

        Action failures (e.g. stopping an adopted session, which the service
        layer refuses with ``AdoptError``) surface as an error notification
        instead of crashing the worker. The ``after`` callback is guarded too:
        it runs on the UI thread but its exception propagates back here.

        Only one action runs per screen at a time. A second attempt is
        *refused*, not queued and not cancelled: a preset launch rewrites
        ``vllm-tp.env`` and issues ``systemctl restart`` over 1-3 minutes, and
        two of those interleave into a corrupt env file and competing
        restarts. Refusing is also the only honest option — Textual thread
        workers run on the event loop's default executor and cannot be
        interrupted, so "cancelling" a launch would leave it driving systemctl
        while the UI reported it stopped.

        Data refreshes are deliberately not covered: they use their own
        worker group, so the screen keeps updating while an action runs.
        """
        if self._action_busy:
            self.notify(
                "An action is already running on this screen. Wait for it to "
                "finish.",
                title="Busy",
                severity="warning",
            )
            return
        self._action_busy = True

        def _worker() -> None:
            try:
                try:
                    result = func()
                except Exception as exc:
                    self.app.call_from_thread(self._notify_error, exc, "Action failed")
                    return
                try:
                    self.app.call_from_thread(after, result)
                except Exception as exc:
                    self.app.call_from_thread(self._notify_error, exc, "Action failed")
            finally:
                # Must clear on every path, or the screen is inert until restart.
                self._action_busy = False

        self.run_worker(_worker, thread=True)

    def _notify_error(self, exc: Exception, title: str = "Action failed") -> None:
        """Show a persistent, visible error without letting markup parse it."""
        self.notify(escape(str(exc)), title=title, severity="error", timeout=10)
