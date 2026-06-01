"""API behavior smoke tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from llmctl.api.app import create_app


def test_api_model_session_benchmark_flow(tmp_path: Path) -> None:
    client = TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'flow.sqlite3'}"))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["safe_mode"] is True

    created_model = client.post(
        "/models",
        json={"name": "example", "runtime": "ollama", "source": "example:latest"},
    )
    assert created_model.status_code == 201
    model_id = created_model.json()["id"]

    models = client.get("/models")
    assert models.status_code == 200
    assert len(models.json()) == 1

    session = client.post(
        "/sessions/start",
        json={"model_id": model_id, "runtime": "ollama", "dry_run": True},
    )
    assert session.status_code == 201
    assert session.json()["status"] == "planned"

    benchmark = client.post(
        "/benchmarks/run",
        json={"model_id": model_id, "name": "smoke", "dry_run": True},
    )
    assert benchmark.status_code == 201
    assert benchmark.json()["name"] == "smoke"

    gpus = client.get("/gpus")
    assert gpus.status_code == 200
    assert isinstance(gpus.json(), list)

    deleted = client.delete(f"/models/{model_id}")
    assert deleted.status_code == 204


def test_api_stop_restart_adopted_session_returns_409(tmp_path: Path) -> None:
    """ADOPTED sessions can't be stopped/restarted via the API — expect 409."""
    from sqlmodel import Session

    from llmctl.db import (
        RuntimeName,
        SessionKind,
        SessionRecord,
        SessionStatus,
        get_engine,
        init_db,
        utcnow,
    )

    db_url = f"sqlite:///{tmp_path / 'adopt-api.sqlite3'}"
    init_db(db_url)
    with Session(get_engine(db_url)) as db:
        record = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.RUNNING,
            kind=SessionKind.ADOPTED,
            endpoint_url="http://127.0.0.1:8003",
            port=8003,
            served_name="llama-3.3-70b",
            systemd_unit="vllm-tp.service",
            adopted_at=utcnow(),
            launch_plan={"command": []},
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        session_id = record.id

    client = TestClient(create_app(database_url=db_url))

    stop_resp = client.post(f"/sessions/{session_id}/stop")
    assert stop_resp.status_code == 409
    assert "adopted" in stop_resp.json()["detail"].lower()

    restart_resp = client.post(f"/sessions/{session_id}/restart")
    assert restart_resp.status_code == 409
    assert "adopted" in restart_resp.json()["detail"].lower()
