"""`DELETE /models/{id}` removes a model the same way `llmctl model rm` does.

The route used the old RegistryService.delete_model: `?delete_files=true`
removed only `record.path` (so an Ollama purge never ran `ollama rm`, and a
starter preset was left behind), and refusals were swallowed behind a 204.
It now runs ModelOps.remove, the path the CLI and TUI use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmctl.api.app import create_app
from llmctl.services import model_ops
from llmctl.services.model_ops import ModelOpsError, RemoveResult


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}"))


def _add(client: TestClient, runtime: str, path: str) -> str:
    created = client.post(
        "/models", json={"name": "m", "runtime": runtime, "source": path, "path": path}
    )
    assert created.status_code == 201
    return created.json()["id"]


@pytest.mark.parametrize(
    ("query", "purge"),
    [("", False), ("?purge=true", True), ("?delete_files=true", True)],
)
def test_delete_runs_the_same_remove_as_the_cli(client, monkeypatch, query, purge) -> None:
    calls: list[tuple[str, bool]] = []

    def remove(self, model_id: str, *, purge: bool = False) -> RemoveResult:
        calls.append((model_id, purge))
        return RemoveResult(True, purge, "Removed m.")

    monkeypatch.setattr(model_ops.ModelOps, "remove", remove)
    model_id = _add(client, "ollama", "qwen3:0.6b")
    assert client.delete(f"/models/{model_id}{query}").status_code == 204
    assert calls == [(model_id, purge)]


def test_a_refused_purge_says_why_and_keeps_the_model(client, tmp_path) -> None:
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "weights.gguf").write_text("keep me")
    model_id = _add(client, "llama_cpp", str(outside / "weights.gguf"))

    response = client.delete(f"/models/{model_id}?purge=true")
    assert response.status_code == 409
    assert response.json()["detail"].startswith("Not removed")
    assert (outside / "weights.gguf").read_text() == "keep me"
    assert client.get(f"/models/{model_id}").status_code == 200


def test_a_failed_engine_delete_is_a_409(client, monkeypatch) -> None:
    def remove(self, model_id: str, *, purge: bool = False) -> RemoveResult:
        raise ModelOpsError("ollama rm failed: connection refused")

    monkeypatch.setattr(model_ops.ModelOps, "remove", remove)
    model_id = _add(client, "ollama", "qwen3:0.6b")
    response = client.delete(f"/models/{model_id}?purge=true")
    assert response.status_code == 409
    assert "connection refused" in response.json()["detail"]


def test_unknown_model_is_a_404(client) -> None:
    assert client.delete("/models/nope?purge=true").status_code == 404
