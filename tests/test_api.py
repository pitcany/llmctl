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
