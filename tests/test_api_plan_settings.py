"""A launch preview must use the same application settings as its start."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmctl.api.app import create_app
from llmctl.config import RuntimeConfig, Settings


def test_plan_and_dry_run_start_share_injected_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llmctl.services.scheduler.get_gpu_info", lambda: [])
    settings = Settings()
    settings.runtimes["ollama"] = RuntimeConfig(endpoint="http://127.0.0.1:11499")
    settings.paths.logs_dir = tmp_path / "logs"
    client = TestClient(create_app(settings, database_url=f"sqlite:///{tmp_path / 'api.db'}"))
    model = client.post(
        "/models", json={"name": "test-model", "runtime": "ollama", "source": "test-model"},
    )
    assert model.status_code == 201
    payload = {"model_id": model.json()["id"], "runtime": "ollama", "dry_run": True}
    preview = client.post("/sessions/plan", json=payload)
    started = client.post("/sessions/start", json=payload)
    assert preview.status_code == 200
    assert started.status_code == 201
    assert started.json()["status"] == "planned"
    assert preview.json()["endpoint_url"] == started.json()["endpoint_url"]
    assert preview.json()["endpoint_url"] == "http://127.0.0.1:11499"
