"""The TUI must not render session state it never checked.

``_data.get_sessions`` documented itself as returning sessions "after
reconciling any dead processes" but only called ``list_sessions()`` -- which
``SessionService`` explicitly documents as a *pure DB read*. ``reconcile``
appeared nowhere in ``llmctl/tui/``.

So a session whose process had died rendered as ``running`` in the TUI forever,
and the dashboard counted it, while `llmctl sessions` in a second terminal
showed the same row as ``stopped``. The two front-ends disagreed about the one
question the tool exists to answer.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session as DBSession

from llmctl.db import (
    RuntimeName,
    SessionKind,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.tui import _data

from ._tui_isolation import isolate_tui

#: A pid that is not running. Chosen far above the default pid_max so the test
#: cannot collide with a real process on the host.
DEAD_PID = 4_000_123


@pytest.fixture(autouse=True)
def _isolated_tui(tmp_path, monkeypatch) -> None:
    isolate_tui(monkeypatch, tmp_path)


def _seed_dead_session(monkeypatch) -> str:
    """Insert an OWNED row that claims to be running on a dead pid."""
    url = _data.load_settings().database_url
    init_db(url)
    with DBSession(get_engine(url)) as db:
        row = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.RUNNING,
            kind=SessionKind.OWNED,
            endpoint_url="http://127.0.0.1:8003",
            port=8003,
            pid=DEAD_PID,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id


def test_get_sessions_reconciles_a_dead_process(monkeypatch) -> None:
    session_id = _seed_dead_session(monkeypatch)

    sessions = _data.get_sessions()

    row = next(s for s in sessions if s.id == session_id)
    assert row.status == SessionStatus.STOPPED, (
        "the TUI rendered a dead process as running"
    )


def test_overview_counts_do_not_include_dead_sessions(monkeypatch) -> None:
    """The dashboard's running-count is the same claim, aggregated."""
    _seed_dead_session(monkeypatch)

    overview = _data.get_overview()

    assert overview["sessions_running"] == 0, overview


def test_get_sessions_leaves_a_live_session_alone(monkeypatch) -> None:
    """Reconciling must not knock down a session whose process is alive.

    No ``endpoint_url``, so this exercises the pid-liveness branch on its own:
    a row *with* an endpoint and a live pid is legitimately DEGRADED when the
    endpoint does not answer, which is a different (and correct) verdict.
    """
    import os

    url = _data.load_settings().database_url
    init_db(url)
    with DBSession(get_engine(url)) as db:
        row = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.RUNNING,
            kind=SessionKind.OWNED,
            pid=os.getpid(),  # this test process is certainly running
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        session_id = row.id

    sessions = _data.get_sessions()

    live = next(s for s in sessions if s.id == session_id)
    assert live.status == SessionStatus.RUNNING
