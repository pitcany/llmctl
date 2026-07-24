"""Quitting must return the shell promptly, even mid-action.

Textual's ``run_worker(thread=True)`` runs on the event loop's *default*
executor, and ``asyncio.run`` joins that executor on the way out with a 300s
grace period. A preset launch blocks up to 300s waiting for readiness, so
pressing ``q`` during one made the TUI vanish while the shell stayed dead for
minutes with no output — indistinguishable from a hang.

Action workers therefore run on daemon threads that shutdown does not join.
That makes quitting instant, which in turn makes it possible to abandon a
half-finished launch, so quitting while an action is in flight asks first.
"""

from __future__ import annotations

import asyncio
import threading
import time

from llmctl.tui import _data
from llmctl.tui import app as app_mod
from llmctl.tui.app import MissionControlApp


def _stub_models(monkeypatch) -> None:
    monkeypatch.setattr(_data, "get_models", lambda: [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    monkeypatch.setattr(app_mod, "REFRESH_INTERVAL", 3600.0)


def test_quitting_mid_action_returns_promptly(monkeypatch) -> None:
    """The regression: App.run() must not join a long-running action."""
    _stub_models(monkeypatch)
    release = threading.Event()
    started = threading.Event()

    def slow_action() -> str:
        started.set()
        release.wait(timeout=30.0)  # stands in for a 300s readiness wait
        return "done"

    class _App(MissionControlApp):
        def on_mount(self) -> None:
            # No super() call: Textual dispatches on_mount to every class in
            # the MRO, so MissionControlApp.on_mount already runs.
            self.set_timer(0.05, self._kick)

        def _kick(self) -> None:
            self.screen.run_action_worker(slow_action, lambda _r: None)
            self.set_timer(0.35, self.exit)

    t0 = time.monotonic()
    try:
        _App().run(headless=True)
        elapsed = time.monotonic() - t0
    finally:
        release.set()

    assert started.is_set(), "the action never started; test proves nothing"
    assert elapsed < 5.0, (
        f"quit blocked on the in-flight action for {elapsed:.1f}s — shutdown is "
        "joining the worker"
    )


def test_worker_finishing_after_quit_is_silent(monkeypatch) -> None:
    """An action outliving the app must not raise on the way out.

    ``call_from_thread`` raises RuntimeError('App is not running') once the app
    has stopped; unguarded in a daemon thread that prints a traceback over the
    user's prompt.
    """
    _stub_models(monkeypatch)
    started = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    def quick_action() -> str:
        started.set()
        time.sleep(0.4)  # finishes after the app is gone
        return "done"

    def excepthook(args) -> None:
        errors.append(args.exc_value)

    original_hook = threading.excepthook
    threading.excepthook = excepthook

    class _App(MissionControlApp):
        def on_mount(self) -> None:
            # No super() call: Textual dispatches on_mount to every class in
            # the MRO, so MissionControlApp.on_mount already runs.
            self.set_timer(0.05, self._kick)

        def _kick(self) -> None:
            self.screen.run_action_worker(
                quick_action, lambda _r: finished.set()
            )
            self.set_timer(0.15, self.exit)

    try:
        _App().run(headless=True)
        time.sleep(1.0)  # let the orphaned worker land
    finally:
        threading.excepthook = original_hook

    assert started.is_set()
    assert not errors, f"orphaned worker raised on the way out: {errors!r}"


def test_quit_while_busy_asks_first(monkeypatch) -> None:
    """Instant quit means a launch can be abandoned; warn before doing so."""
    _stub_models(monkeypatch)
    release = threading.Event()

    def slow_action() -> str:
        release.wait(timeout=10.0)
        return "done"

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            app.action_show_models()
            for _ in range(40):
                await pilot.pause(0.02)
                if app.screen.__class__.__name__.startswith("Models"):
                    break
            app.screen.run_action_worker(slow_action, lambda _r: None)
            for _ in range(15):
                await pilot.pause(0.02)
            await app.action_quit()
            for _ in range(20):
                await pilot.pause(0.02)
            assert app.is_running, "quit did not stop to ask"
            assert app.screen.__class__.__name__ == "ConfirmActionModal", (
                f"no confirmation; screen is {app.screen.__class__.__name__}"
            )
            app.screen.dismiss(None)
            for _ in range(15):
                await pilot.pause(0.02)
            assert app.is_running, "cancelling the quit prompt still quit"
            release.set()

    asyncio.run(_run())


def test_quit_when_idle_does_not_ask(monkeypatch) -> None:
    """The prompt must only appear when something is actually in flight."""
    _stub_models(monkeypatch)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            app.action_show_models()
            for _ in range(40):
                await pilot.pause(0.02)
                if app.screen.__class__.__name__.startswith("Models"):
                    break
            await app.action_quit()
            for _ in range(20):
                await pilot.pause(0.02)
            assert not app.is_running, "idle quit was blocked by a prompt"

    asyncio.run(_run())
