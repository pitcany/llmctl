"""Shared base screen and color constants for the TUI.

Data fetching for every screen runs in a worker thread. This keeps the UI
responsive and, critically, gives blocking service calls their own context so
``asyncio.run`` (used by adapters/health) works without colliding with
Textual's running event loop. Widget updates are marshalled back to the UI
thread via ``call_from_thread``.
"""

from __future__ import annotations

from typing import Any

from textual.screen import Screen

#: Inline-markup colors (Textual content markup accepts hex color tags).
C_ACCENT = "#5eead4"
C_OK = "#86efac"
C_WARN = "#fbbf24"
C_ERR = "#fb7185"
C_MUTED = "#8aa6a3"


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
        """Worker body: fetch data, then marshal rendering to the UI thread."""
        data = self.fetch()
        self.app.call_from_thread(self.render_data, data)

    def run_action_worker(self, func: Any, after: Any) -> None:
        """Run a blocking action ``func`` off-thread, then ``after(result)``.

        Action failures (e.g. stopping an adopted session, which the service
        layer refuses with ``AdoptError``) surface as an error notification
        instead of crashing the worker.
        """

        def _worker() -> None:
            try:
                result = func()
            except Exception as exc:
                self.app.call_from_thread(self._notify_action_error, exc)
                return
            self.app.call_from_thread(after, result)

        self.run_worker(_worker, thread=True)

    def _notify_action_error(self, exc: Exception) -> None:
        """Show a persistent, visible error for a failed action."""
        self.notify(str(exc), title="Action failed", severity="error", timeout=10)
