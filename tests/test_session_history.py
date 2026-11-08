"""The session lifecycle trail reaches the API and survives older databases."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlmodel import Session

from llmctl.api.app import create_app
from llmctl.db import (
    RuntimeName,
    SessionKind,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.services.sessions import record_to_session, set_status


def _db(tmp_path: Path) -> tuple[str, Session]:
    url = f"sqlite:///{tmp_path / 'history.sqlite3'}"
    init_db(url)
    return url, Session(get_engine(url))


def test_history_route_returns_the_trail_oldest_first(tmp_path: Path) -> None:
    url, db = _db(tmp_path)
    try:
        row = SessionRecord(
            runtime=RuntimeName.VLLM, status=SessionStatus.STARTING, kind=SessionKind.OWNED
        )
        set_status(row, SessionStatus.RUNNING, why="owned endpoint probe")
        set_status(row, SessionStatus.DEGRADED, why="owned endpoint probe")
        db.add(row)
        db.commit()
        session_id = row.id
    finally:
        db.close()

    client = TestClient(create_app(database_url=url))
    response = client.get(f"/sessions/{session_id}/history")
    assert response.status_code == 200
    trail = response.json()
    assert [(c["old"], c["new"]) for c in trail] == [
        ("starting", "running"),
        ("running", "degraded"),
    ]
    assert trail[0]["why"] == "owned endpoint probe" and "at" in trail[0]

    # A prefix works like everywhere else; an unknown id is a 404.
    assert client.get(f"/sessions/{session_id[:8]}/history").status_code == 200
    assert client.get("/sessions/nope/history").status_code == 404


def test_cli_history_prints_the_trail_and_json(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from llmctl import cli

    url, db = _db(tmp_path)
    try:
        row = SessionRecord(runtime=RuntimeName.VLLM, status=SessionStatus.RUNNING)
        set_status(row, SessionStatus.STOPPING, why="stop requested")
        set_status(row, SessionStatus.STOPPED, why="stop")
        db.add(row)
        db.commit()
        session_id = row.id
    finally:
        db.close()
    monkeypatch.setattr(cli, "_session", lambda: Session(get_engine(url)))

    result = CliRunner().invoke(cli.app, ["history", session_id[:8]])
    assert result.exit_code == 0, result.output
    assert "stop requested" in result.output and "stopping" in result.output

    result = CliRunner().invoke(cli.app, ["history", session_id, "--json"])
    assert result.exit_code == 0, result.output
    assert [c["why"] for c in json.loads(result.output)] == ["stop requested", "stop"]

    result = CliRunner().invoke(cli.app, ["history", "nope"])
    assert result.exit_code == 2 and "Session not found" in result.output


def test_rows_from_before_the_column_read_as_an_empty_trail(tmp_path: Path) -> None:
    url, db = _db(tmp_path)
    try:
        row = SessionRecord(runtime=RuntimeName.VLLM, status=SessionStatus.RUNNING)
        db.add(row)
        db.commit()
        # Simulate a row written by an older llmctl: the column is NULL.
        db.execute(text("UPDATE sessions SET status_history = NULL WHERE id = :id"), {"id": row.id})
        db.commit()
        db.expire_all()
        reloaded = db.get(SessionRecord, row.id)
        assert reloaded is not None
        assert record_to_session(reloaded).status_history == []
        # The next move starts the trail from nothing instead of failing on None.
        set_status(reloaded, SessionStatus.STOPPED, why="stop")
        assert [e["new"] for e in reloaded.status_history] == ["stopped"]
    finally:
        db.close()
