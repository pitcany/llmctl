"""API tests for the registry/profile management endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from llmctl.api.app import create_app


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}"))


def test_models_full_crud(tmp_path: Path) -> None:
    client = _client(tmp_path)
    created = client.post(
        "/models",
        json={
            "name": "m-api",
            "runtime": "vllm",
            "source": "/srv/m",
            "path": "/srv/m",
            "max_context": 16384,
            "tags": ["api"],
        },
    )
    assert created.status_code == 201
    model_id = created.json()["id"]

    listed = client.get("/models")
    assert listed.status_code == 200
    assert any(m["id"] == model_id for m in listed.json())

    show = client.get(f"/models/{model_id}")
    assert show.status_code == 200
    assert show.json()["max_context"] == 16384
    assert show.json()["active"] is True

    updated = client.put(
        f"/models/{model_id}",
        json={"notes": "via api", "max_context": 32768},
    )
    assert updated.status_code == 200
    assert updated.json()["notes"] == "via api"
    assert updated.json()["max_context"] == 32768

    disabled = client.put(f"/models/{model_id}", json={"active": False})
    assert disabled.status_code == 200
    assert disabled.json()["active"] is False
    # Default listing hides inactive rows.
    assert all(m["id"] != model_id for m in client.get("/models").json())
    # ?include_inactive=true brings them back.
    assert any(
        m["id"] == model_id
        for m in client.get("/models?include_inactive=true").json()
    )

    deleted = client.delete(f"/models/{model_id}")
    assert deleted.status_code == 204
    assert client.get(f"/models/{model_id}").status_code == 404


def test_profiles_full_crud(tmp_path: Path) -> None:
    client = _client(tmp_path)
    created = client.post(
        "/profiles",
        json={
            "name": "api-fast",
            "runtime": "vllm",
            "description": "api smoke",
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.8,
        },
    )
    assert created.status_code == 201, created.text
    profile_id = created.json()["id"]

    listed = client.get("/profiles")
    assert listed.status_code == 200
    assert any(p["id"] == profile_id for p in listed.json())

    # Lookup by name should work too.
    show = client.get("/profiles/api-fast")
    assert show.status_code == 200
    assert show.json()["max_model_len"] == 8192

    updated = client.put(
        f"/profiles/{profile_id}", json={"max_model_len": 16384}
    )
    assert updated.status_code == 200
    assert updated.json()["max_model_len"] == 16384

    deleted = client.delete(f"/profiles/{profile_id}")
    assert deleted.status_code == 204
    assert client.get(f"/profiles/{profile_id}").status_code == 404


def test_profile_create_rejects_validation_errors(tmp_path: Path) -> None:
    client = _client(tmp_path)
    bad = client.post(
        "/profiles",
        json={
            "name": "bad",
            "runtime": "vllm",
            "gpu_memory_utilization": 2.0,  # out of range
        },
    )
    assert bad.status_code == 422
    body = bad.json()
    issues = body["detail"]["issues"]
    assert any(issue["field"] == "gpu_memory_utilization" for issue in issues)


def test_profile_create_rejects_duplicate_name(tmp_path: Path) -> None:
    client = _client(tmp_path)
    client.post("/profiles", json={"name": "dup", "runtime": "vllm"})
    again = client.post("/profiles", json={"name": "dup", "runtime": "vllm"})
    assert again.status_code == 409


def test_profile_validate_endpoint(tmp_path: Path) -> None:
    client = _client(tmp_path)
    created = client.post(
        "/profiles", json={"name": "pre", "runtime": "vllm"}
    )
    profile_id = created.json()["id"]
    issues = client.post(
        f"/profiles/{profile_id}/validate",
        json={"gpu_memory_utilization": 0.97},
    )
    assert issues.status_code == 200
    payload = issues.json()
    assert any(
        issue["field"] == "gpu_memory_utilization"
        and issue["severity"] == "warning"
        for issue in payload
    )
