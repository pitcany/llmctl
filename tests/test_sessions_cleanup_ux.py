"""Sessions-screen cleanup honesty, and per-session health reporting.

Two related confusions, both reported live on 2026-08-05:

* Pressing ``c`` on the Sessions screen ran ``cleanup(remove_stale=False)``,
  which reconciles but deletes nothing — so stopped/failed rows stayed on
  screen and the action read as broken. The notification never said so, and
  the TUI had no way to purge at all.
* ``llmctl health`` reported ``llama.cpp binary found`` whether or not any
  server was running, which reads as healthy when nothing is up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlmodel import Session

from llmctl.db import RuntimeName, SessionRecord, SessionStatus, get_engine, init_db
from llmctl.services.health import HealthService
from llmctl.tui.screens_sessions import SessionsScreen


def test_cleanup_notification_says_nothing_was_removed() -> None:
    """The `c` action must not imply it deleted rows — it does not."""
    captured: dict[str, Any] = {}

    class _App:
        def notify(self, message: str, **kw: Any) -> None:
            captured["message"] = message

    class _Screen(SessionsScreen):
        """Test double: Textual's `app` is a read-only property, so shadow it."""

        def __init__(self) -> None:
            self._stub_app = _App()

        @property
        def app(self) -> Any:  # type: ignore[override]
            return self._stub_app

        def refresh_data(self) -> None:
            captured["refreshed"] = True

    _Screen()._after_cleanup(
        {"dead_marked": 2, "freed_ports": [8000], "stale_removed": 0}
    )
    msg = captured["message"]
    assert "No records removed" in msg
    assert "shift+C" in msg
    assert captured["refreshed"] is True


def test_purge_binding_is_registered() -> None:
    """shift+C must exist, or the cleanup message points at nothing."""
    keys = {b.key for b in SessionsScreen.BINDINGS}
    assert "C" in keys
    assert "c" in keys
    actions = {b.key: b.action for b in SessionsScreen.BINDINGS}
    assert actions["C"] == "purge_stale"
    assert actions["c"] == "cleanup"
    assert hasattr(SessionsScreen, "action_purge_stale")


def _db_with(tmp_path: Path, *records: SessionRecord) -> Session:
    url = f"sqlite:///{tmp_path / 'health.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    for r in records:
        db.add(r)
    db.commit()
    return db


def test_health_reports_live_sessions_per_runtime(tmp_path: Path) -> None:
    db = _db_with(
        tmp_path,
        SessionRecord(
            runtime=RuntimeName.LLAMA_CPP,
            status=SessionStatus.RUNNING,
            served_name="deepseek-v4-flash",
            endpoint_url="http://127.0.0.1:8000",
        ),
        SessionRecord(
            runtime=RuntimeName.LLAMA_CPP,
            status=SessionStatus.STOPPED,
            served_name="old-model",
        ),
    )
    try:
        data = HealthService(db=db).get_health()
        llama = data["runtimes"]["llama_cpp"]
        assert data["session_state_available"] is True
        assert llama["serving_count"] == 1
        assert llama["sessions"][0]["served_name"] == "deepseek-v4-flash"
        # A stopped session must not count as serving.
        assert all(s["status"] == "running" for s in llama["sessions"])
    finally:
        db.close()


def test_health_without_db_declares_session_state_unavailable() -> None:
    """serving_count == 0 must not be readable as 'nothing is running'."""
    data = HealthService().get_health()
    assert data["session_state_available"] is False
    for info in data["runtimes"].values():
        assert info["serving_count"] == 0
