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

    def on_mount(self) -> None:
        """Load data when the screen is first mounted."""
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
        """

        def _worker() -> None:
            try:
                result = func()
            except Exception as exc:
                self.app.call_from_thread(self._notify_error, exc, "Action failed")
                return
            try:
                self.app.call_from_thread(after, result)
            except Exception as exc:
                self.app.call_from_thread(self._notify_error, exc, "Action failed")

        self.run_worker(_worker, thread=True)

    def _notify_error(self, exc: Exception, title: str = "Action failed") -> None:
        """Show a persistent, visible error without letting markup parse it."""
        self.notify(escape(str(exc)), title=title, severity="error", timeout=10)
