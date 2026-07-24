"""Unconfirmed destructive keystrokes must be gated, and counts must be true.

Three specific gaps this pins:

* Sessions ``x`` killed a process and ``ctrl+r`` terminated-and-relaunched one
  with no confirmation at all — and ``ctrl+r`` means *refresh* on every other
  screen, so the harmless meaning is the one an operator learns first.
* Models ``ctrl+s`` was an unconfirmed ``scan --import``: it persists rows and
  flags absent ones MISSING, feeding the irreversible prune on ``x``.
* The prune dialog's count came from the last render while the pruned set was
  re-derived at confirm time, so a scan landing in between made the number a
  lie in the dangerous direction.
"""

from __future__ import annotations

import asyncio

from llmctl.db import ModelStatus, RuntimeName, SessionStatus
from llmctl.schemas import Model, Session
from llmctl.tui import _data
from llmctl.tui.app import MissionControlApp


def _session() -> Session:
    return Session(
        id="sess-1234abcd",
        model_id="m1",
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
    )


def _missing(idx: int) -> Model:
    return Model(
        id=f"missing-{idx}",
        name=f"ghost{idx}",
        runtime=RuntimeName.OLLAMA,
        source=f"ghost{idx}",
        status=ModelStatus.MISSING,
    )


async def _show(app: MissionControlApp, pilot, action: str, prefix: str) -> None:
    getattr(app, action)()
    for _ in range(60):
        await pilot.pause(0.02)
        if app.screen.__class__.__name__.lower().startswith(prefix):
            return
    raise AssertionError(f"{prefix} screen did not activate")


def _stub_sessions(monkeypatch) -> list[str]:
    stopped: list[str] = []
    monkeypatch.setattr(_data, "get_sessions", lambda: [_session()])
    monkeypatch.setattr(_data, "tail_log", lambda _s: "")
    monkeypatch.setattr(
        _data, "stop_session", lambda sid: stopped.append(sid) or _session()
    )
    monkeypatch.setattr(
        _data, "restart_session", lambda sid: stopped.append(sid) or _session()
    )
    return stopped


def test_stop_session_requires_confirmation(monkeypatch) -> None:
    """Pressing x must open a gate, not kill the process outright."""
    stopped = _stub_sessions(monkeypatch)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_sessions", "sessions")
            for _ in range(30):
                await pilot.pause(0.02)
                if app.screen._ids:
                    break
            app.screen.action_stop_session()
            for _ in range(20):
                await pilot.pause(0.02)
            assert not stopped, "session was stopped with no confirmation"
            assert app.screen.__class__.__name__ == "ConfirmActionModal", (
                f"no confirm gate; screen is {app.screen.__class__.__name__}"
            )
            # The dialog must name the session it would kill.
            assert "sess1234" in str(app.screen._title) or "sess-123" in str(
                app.screen._title
            )

    asyncio.run(_run())


def test_restart_session_requires_confirmation(monkeypatch) -> None:
    """ctrl+r means refresh elsewhere; on Sessions it relaunches a process."""
    stopped = _stub_sessions(monkeypatch)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_sessions", "sessions")
            for _ in range(30):
                await pilot.pause(0.02)
                if app.screen._ids:
                    break
            app.screen.action_restart_session()
            for _ in range(20):
                await pilot.pause(0.02)
            assert not stopped, "session was restarted with no confirmation"
            assert app.screen.__class__.__name__ == "ConfirmActionModal"

    asyncio.run(_run())


def test_confirming_stop_actually_stops_the_right_session(monkeypatch) -> None:
    """The gate must not make the action unreachable, or target the wrong row."""
    stopped = _stub_sessions(monkeypatch)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_sessions", "sessions")
            for _ in range(30):
                await pilot.pause(0.02)
                if app.screen._ids:
                    break
            app.screen.action_stop_session()
            for _ in range(20):
                await pilot.pause(0.02)
            app.screen.dismiss(True)
            for _ in range(30):
                await pilot.pause(0.02)

    asyncio.run(_run())
    assert stopped == ["sess-1234abcd"], f"stop did not reach the service: {stopped!r}"


def test_cancelling_stop_leaves_the_session_alone(monkeypatch) -> None:
    """Escape/cancel must be a real no-op."""
    stopped = _stub_sessions(monkeypatch)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_sessions", "sessions")
            for _ in range(30):
                await pilot.pause(0.02)
                if app.screen._ids:
                    break
            app.screen.action_stop_session()
            for _ in range(20):
                await pilot.pause(0.02)
            app.screen.dismiss(None)
            for _ in range(30):
                await pilot.pause(0.02)

    asyncio.run(_run())
    assert stopped == [], "cancelling the gate still stopped the session"


def test_confirming_scan_runs_the_import(monkeypatch) -> None:
    """Confirming the scan gate must actually scan."""
    scanned: list[int] = []
    monkeypatch.setattr(_data, "get_models", lambda: [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    monkeypatch.setattr(_data, "scan_models", lambda: scanned.append(1) or [])

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_models", "models")
            app.screen.action_scan()
            for _ in range(20):
                await pilot.pause(0.02)
            app.screen.dismiss(True)
            for _ in range(30):
                await pilot.pause(0.02)

    asyncio.run(_run())
    assert scanned == [1], "confirming the gate did not run the scan"


def test_scan_requires_confirmation(monkeypatch) -> None:
    """ctrl+s persists rows and flags MISSING; it must say so first."""
    scanned: list[int] = []
    monkeypatch.setattr(_data, "get_models", lambda: [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 0)
    monkeypatch.setattr(_data, "scan_models", lambda: scanned.append(1) or [])

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_models", "models")
            app.screen.action_scan()
            for _ in range(20):
                await pilot.pause(0.02)
            assert not scanned, "scan --import ran with no confirmation"
            assert app.screen.__class__.__name__ == "ConfirmActionModal"

    asyncio.run(_run())


def test_prune_count_is_bound_at_keypress_not_rederived(monkeypatch) -> None:
    """The dialog's number must be exactly what gets deleted.

    Simulates a scan landing between keypress and confirm: the visible count
    was 2, so confirming must prune those 2 rows and not the 40 the later scan
    flagged.
    """
    monkeypatch.setattr(_data, "get_models", lambda: [])
    monkeypatch.setattr(_data, "get_backend_map", lambda: {})
    monkeypatch.setattr(_data, "get_preset_count_by_model", lambda: {})
    monkeypatch.setattr(_data, "get_missing_count", lambda: 2)
    monkeypatch.setattr(
        _data, "get_missing_model_ids", lambda: ["missing-0", "missing-1"]
    )
    pruned_with: list[object] = []
    monkeypatch.setattr(
        _data,
        "prune_missing_models",
        lambda ids=None: pruned_with.append(ids) or len(ids or []),
    )

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await _show(app, pilot, "action_show_models", "models")
            for _ in range(30):
                await pilot.pause(0.02)
                if app.screen._missing_count == 2:
                    break
            screen = app.screen
            screen.action_prune_missing()
            for _ in range(20):
                await pilot.pause(0.02)
            assert app.screen.__class__.__name__ == "DeleteModal"
            # A scan lands now and flags 40 more rows.
            monkeypatch.setattr(_data, "get_missing_count", lambda: 42)
            monkeypatch.setattr(
                _data,
                "get_missing_model_ids",
                lambda: [f"missing-{i}" for i in range(42)],
            )
            from llmctl.tui._modals_registry import ConfirmDelete

            app.screen.dismiss(ConfirmDelete(delete_files=False))
            for _ in range(30):
                await pilot.pause(0.02)

    asyncio.run(_run())
    assert pruned_with, "prune never ran"
    assert pruned_with[0] == ["missing-0", "missing-1"], (
        f"prune widened past the confirmed set: {pruned_with[0]!r}"
    )


def test_prune_service_honours_an_explicit_id_allowlist(tmp_path) -> None:
    """RegistryService.prune_missing(ids=...) must touch only those rows."""
    from sqlmodel import Session as DBSession

    from llmctl.db import ModelRecord, get_engine, init_db
    from llmctl.services.registry import RegistryService

    url = f"sqlite:///{tmp_path}/prune.db"
    init_db(url)
    with DBSession(get_engine(url)) as db:
        for i in range(3):
            db.add(
                ModelRecord(
                    id=f"m{i}",
                    name=f"g{i}",
                    runtime=RuntimeName.OLLAMA,
                    source=f"g{i}",
                    status=ModelStatus.MISSING,
                    active=True,
                )
            )
        db.commit()
        count = RegistryService(db).prune_missing(ids=["m0", "m2"])
        assert count == 2
        remaining = {
            m.name for m in RegistryService(db).list_models() if m.name.startswith("g")
        }
        assert remaining == {"g1"}
