"""The periodic refresh must actually fire, and must survive a failing fetch.

Two defects motivate these tests:

* ``App.set_interval(REFRESH_INTERVAL, self._auto_refresh)`` silently bound a
  ``None`` callback, because Textual's ``DOMNode.__init__`` sets an instance
  attribute of that name which shadows the method. The TUI never refreshed.
* ``DataScreen._refresh_worker`` wrapped neither ``fetch`` nor ``render_data``,
  and ``run_worker`` defaults to ``exit_on_error=True`` — so a transient
  SQLite lock on the shared registry file terminated the whole app.
"""

from __future__ import annotations

import asyncio

from llmctl.tui import _data
from llmctl.tui import app as app_mod
from llmctl.tui.app import MissionControlApp


async def _settle(pilot, times: int = 40, delay: float = 0.02) -> None:
    for _ in range(times):
        await pilot.pause(delay)


def test_periodic_refresh_callback_is_callable() -> None:
    """The interval timer must be given a real callable, not None."""
    captured: list[object] = []

    async def _run() -> None:
        app = MissionControlApp()
        original = app.set_interval

        def spy(interval, callback=None, *args, **kwargs):
            captured.append(callback)
            return original(interval, callback, *args, **kwargs)

        app.set_interval = spy  # type: ignore[method-assign]
        async with app.run_test() as pilot:
            await pilot.pause()

    asyncio.run(_run())
    assert captured, "no interval timer was registered"
    assert all(callable(cb) for cb in captured), (
        f"interval registered with a non-callable callback: {captured!r}"
    )


def test_periodic_refresh_actually_refetches(monkeypatch) -> None:
    """With a short interval the active screen must refetch more than once."""
    calls: list[int] = []
    real_overview = _data.get_overview

    def counting_overview():
        calls.append(1)
        return real_overview()

    monkeypatch.setattr(_data, "get_overview", counting_overview)
    monkeypatch.setattr(app_mod, "REFRESH_INTERVAL", 0.1)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _settle(pilot, times=60, delay=0.02)

    asyncio.run(_run())
    assert len(calls) >= 2, (
        f"auto-refresh never re-fetched (got {len(calls)} fetch(es)); "
        "the interval timer is not firing"
    )


def test_failing_fetch_notifies_instead_of_killing_the_app(monkeypatch) -> None:
    """A raising fetch must leave the app alive with a visible error."""
    def boom():
        raise RuntimeError("database is locked")

    monkeypatch.setattr(_data, "get_models", boom)
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    notes: list[str] = []

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            monkeypatch.setattr(
                app, "notify", lambda msg, *a, **k: notes.append(str(msg))
            )
            app.action_show_models()
            await _settle(pilot)
            assert app.is_running, "a failing fetch terminated the app"

    asyncio.run(_run())
    assert any("database is locked" in n for n in notes), (
        f"fetch failure was not surfaced to the user: {notes!r}"
    )


def test_failing_render_notifies_instead_of_killing_the_app(monkeypatch) -> None:
    """render_data runs on the UI thread via call_from_thread; guard it too."""
    monkeypatch.setattr(_data, "get_models", lambda: [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    notes: list[str] = []

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            monkeypatch.setattr(
                app, "notify", lambda msg, *a, **k: notes.append(str(msg))
            )
            app.action_show_models()
            await _settle(pilot, times=20)
            screen = app.screen

            def bad_render(_data_payload):
                raise KeyError("models")

            monkeypatch.setattr(screen, "render_data", bad_render)
            screen.refresh_data()
            await _settle(pilot)
            assert app.is_running, "a failing render terminated the app"

    asyncio.run(_run())
    assert any("models" in n for n in notes), (
        f"render failure was not surfaced to the user: {notes!r}"
    )


def test_switching_back_to_a_screen_refetches(monkeypatch) -> None:
    """A revisited screen must reload, not show data from first mount.

    ``MissionControlApp.on_screen_resume`` was dead code: Textual declares
    ``ScreenResume`` with ``bubble=False``, so the event never reached the
    App. The handler belongs on the screen itself.
    """
    calls: list[int] = []
    monkeypatch.setattr(_data, "get_models", lambda: calls.append(1) or [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    # Keep the interval timer from muddying the count.
    monkeypatch.setattr(app_mod, "REFRESH_INTERVAL", 3600.0)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            app.action_show_models()
            await _settle(pilot, times=30)
            first = len(calls)
            assert first >= 1, "screen never loaded on first activation"
            app.action_show_dashboard()
            await _settle(pilot, times=30)
            app.action_show_models()
            await _settle(pilot, times=30)
            assert len(calls) > first, (
                "returning to the screen did not refetch "
                f"({first} fetch(es) before, {len(calls)} after)"
            )

    asyncio.run(_run())


def test_first_activation_fetches_once(monkeypatch) -> None:
    """on_mount and on_screen_resume must not both trigger a load.

    ``ScreenResume`` fires on first activation too, so refreshing from both
    hooks would double every screen entry — two concurrent workers whose
    render order is not their start order.
    """
    calls: list[int] = []
    monkeypatch.setattr(_data, "get_models", lambda: calls.append(1) or [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    monkeypatch.setattr(app_mod, "REFRESH_INTERVAL", 3600.0)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            app.action_show_models()
            await _settle(pilot, times=40)

    asyncio.run(_run())
    assert len(calls) == 1, f"first activation fetched {len(calls)} times, want 1"
