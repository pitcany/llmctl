"""`llmctl schema` / GET /schema describe the running vocabularies, and `logs --json` is a list."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session
from typer.testing import CliRunner

from llmctl import cli
from llmctl.api.app import create_app
from llmctl.db import SESSION_TRANSITIONS, EventLevel, SessionStatus, get_engine, init_db
from llmctl.schema_export import ENUMS, build_schema
from llmctl.services.events import log_event
from llmctl.services.operations import OperationPhase


def test_build_schema_lists_every_enum_member_and_the_tables() -> None:
    doc = build_schema()
    for enum in ENUMS:
        assert doc["enums"][enum.__name__] == [m.value for m in enum]
    sessions = doc["sessions"]
    assert set(sessions["transitions"]) == {s.value for s in SessionStatus}
    for status, targets in SESSION_TRANSITIONS.items():
        assert set(sessions["transitions"][status.value]) == {t.value for t in targets}
    assert sessions["active"] == ["starting", "running", "degraded"]
    assert sessions["terminal"] == ["stopped", "failed"]
    assert set(doc["operations"]["phase_views"]) == {p.value for p in OperationPhase}
    assert doc["operations"]["phase_views"]["needs_attention"]["tone"] == "error"
    assert "status_history" in doc["models"]["Session"]["properties"]
    assert doc["models"]["SwapJobSummary"]["properties"]["status"]
    json.dumps(doc)  # must be plain JSON, no enum or datetime objects left


def test_schema_route_and_cli_serve_the_same_document(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'schema.sqlite3'}"
    client = TestClient(create_app(database_url=url))
    response = client.get("/schema")
    assert response.status_code == 200
    from_api = response.json()

    result = CliRunner().invoke(cli.app, ["schema"])
    assert result.exit_code == 0, result.output
    from_cli = json.loads(result.output)

    assert from_cli == from_api
    assert from_cli["enums"]["SessionStatus"] == [s.value for s in SessionStatus]


def test_logs_json_lists_events(tmp_path: Path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path / 'events.sqlite3'}"
    init_db(url)
    with Session(get_engine(url)) as db:
        log_event(db, EventLevel.INFO, "session", "Session abc stopped.")
    monkeypatch.setattr(cli, "_session", lambda: Session(get_engine(url)))

    result = CliRunner().invoke(cli.app, ["logs", "--json"])
    assert result.exit_code == 0, result.output
    events = json.loads(result.output)
    assert [e["message"] for e in events] == ["Session abc stopped."]
    assert events[0]["level"] == "info" and events[0]["category"] == "session"
    assert "created_at" in events[0]
