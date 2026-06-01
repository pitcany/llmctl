"""Tests for the OpenAI-compatible router gateway.

Covers the five spec axes:

* explicit model id resolution (served name + session id),
* alias resolution (``local-<alias>`` and bare alias),
* unavailable model -> 503 with a clear error,
* bearer auth on /v1/* (and /health when configured),
* /health envelope content.

Upstream HTTP is stubbed with ``respx`` so no real model server is
needed. Each test gets its own SQLite file and ``Settings`` so they
don't share router config or alias overlays.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlmodel import Session

from llmctl.api.gateway import create_gateway_app
from llmctl.config import RouterSettings, Settings
from llmctl.db import (
    ProfileRecord,
    RuntimeName,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.services.gateway import GatewayService


def _settings(tmp_path: Path, **router_overrides: object) -> Settings:
    db_url = f"sqlite:///{tmp_path / 'gateway.sqlite3'}"
    init_db(db_url)
    s = Settings()
    s.database.url = db_url
    s.paths.config_dir = tmp_path / "cfg"
    s.paths.config_dir.mkdir(exist_ok=True)
    s.router = RouterSettings(
        aliases={
            "coding": None,
            "reasoning": None,
            "fast": None,
        },
        **router_overrides,
    )
    return s


def _seed_session(
    db_url: str,
    *,
    served_name: str,
    endpoint_url: str,
    profile_name: str | None = None,
) -> tuple[str, str | None]:
    """Insert a RUNNING vLLM session with the given served-model-name."""
    engine = get_engine(db_url)
    with Session(engine) as db:
        profile_id: str | None = None
        if profile_name:
            profile = ProfileRecord(name=profile_name, runtime=RuntimeName.VLLM)
            db.add(profile)
            db.commit()
            db.refresh(profile)
            profile_id = profile.id
        record = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.RUNNING,
            endpoint_url=endpoint_url,
            health_url=f"{endpoint_url}/v1/models",
            pid=4242,
            port=int(endpoint_url.rsplit(":", 1)[1]),
            profile_id=profile_id,
            launch_plan={
                "command": [
                    "vllm",
                    "serve",
                    "org/example",
                    "--served-model-name",
                    served_name,
                ],
            },
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id, profile_id


def _client(settings: Settings) -> TestClient:
    return TestClient(create_gateway_app(settings, database_url=settings.database.url))


# -- /health -----------------------------------------------------------------


def test_health_returns_router_view(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed_session(
        settings.database.url,
        served_name="llama-3.3-70b",
        endpoint_url="http://127.0.0.1:8003",
    )
    response = _client(settings).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["router"]["port"] == 9000
    assert body["router"]["auth_required"] is False
    assert {a["name"] for a in body["aliases"]} == {"coding", "reasoning", "fast"}


# -- /v1/models --------------------------------------------------------------


def test_list_models_returns_active_sessions_and_bound_aliases(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session_id, _ = _seed_session(
        settings.database.url,
        served_name="qwen3-coder-30b",
        endpoint_url="http://127.0.0.1:8001",
    )
    GatewayService(
        Session(get_engine(settings.database.url)), settings
    ).set_alias("coding", session_id)

    response = _client(settings).get("/v1/models")
    assert response.status_code == 200
    data = response.json()["data"]
    ids = {entry["id"] for entry in data}
    assert "qwen3-coder-30b" in ids
    assert "local-coding" in ids


# -- explicit + alias routing ------------------------------------------------


def test_explicit_model_routes_to_session(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _seed_session(
        settings.database.url,
        served_name="llama-3.3-70b",
        endpoint_url="http://127.0.0.1:8003",
    )
    with respx.mock(assert_all_called=True) as router:
        upstream = router.post("http://127.0.0.1:8003/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"index": 0}]})
        )
        response = _client(settings).post(
            "/v1/chat/completions",
            json={"model": "llama-3.3-70b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.headers["x-llmctl-route"] == "explicit"
    # The upstream must see its native served name, not the gateway's input.
    sent = upstream.calls[0].request.read()
    assert b'"model":"llama-3.3-70b"' in sent or b'"model": "llama-3.3-70b"' in sent


def test_alias_routes_to_bound_session(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session_id, _ = _seed_session(
        settings.database.url,
        served_name="qwen3-coder-30b",
        endpoint_url="http://127.0.0.1:8001",
    )
    GatewayService(
        Session(get_engine(settings.database.url)), settings
    ).set_alias("coding", session_id)

    with respx.mock(assert_all_called=True) as router:
        upstream = router.post("http://127.0.0.1:8001/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        response = _client(settings).post(
            "/v1/chat/completions",
            json={"model": "local-coding", "messages": [{"role": "user", "content": "x"}]},
        )
    assert response.status_code == 200
    assert response.headers["x-llmctl-route"] == "alias:coding"
    sent = upstream.calls[0].request.read()
    assert b"qwen3-coder-30b" in sent


def test_alias_falls_through_to_bare_key(tmp_path: Path) -> None:
    """Accept both ``local-coding`` and ``coding`` per spec."""
    settings = _settings(tmp_path)
    session_id, _ = _seed_session(
        settings.database.url,
        served_name="qwen3-coder-30b",
        endpoint_url="http://127.0.0.1:8001",
    )
    GatewayService(
        Session(get_engine(settings.database.url)), settings
    ).set_alias("coding", session_id)

    with respx.mock(assert_all_called=True) as router:
        router.post("http://127.0.0.1:8001/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        response = _client(settings).post(
            "/v1/chat/completions",
            json={"model": "coding", "messages": []},
        )
    assert response.status_code == 200
    assert response.headers["x-llmctl-route"] == "alias:coding"


# -- unavailable -------------------------------------------------------------


def test_unavailable_model_returns_503(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    response = _client(settings).post(
        "/v1/chat/completions",
        json={"model": "no-such-model", "messages": []},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_unavailable_model_with_fallback_routes_to_target(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        fallback_policy="fallback",
        fallback_target="llama-3.3-70b",
    )
    _seed_session(
        settings.database.url,
        served_name="llama-3.3-70b",
        endpoint_url="http://127.0.0.1:8003",
    )
    with respx.mock(assert_all_called=True) as router:
        router.post("http://127.0.0.1:8003/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        response = _client(settings).post(
            "/v1/chat/completions",
            json={"model": "nonexistent-12b", "messages": []},
        )
    assert response.status_code == 200
    assert response.headers["x-llmctl-route"] == "fallback"


# -- auth --------------------------------------------------------------------


def test_auth_required_returns_401_without_header(tmp_path: Path) -> None:
    settings = _settings(tmp_path, auth_token="s3cret")
    _seed_session(
        settings.database.url,
        served_name="llama-3.3-70b",
        endpoint_url="http://127.0.0.1:8003",
    )
    response = _client(settings).post(
        "/v1/chat/completions",
        json={"model": "llama-3.3-70b", "messages": []},
    )
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_auth_accepts_correct_bearer(tmp_path: Path) -> None:
    settings = _settings(tmp_path, auth_token="s3cret")
    _seed_session(
        settings.database.url,
        served_name="llama-3.3-70b",
        endpoint_url="http://127.0.0.1:8003",
    )
    with respx.mock(assert_all_called=True) as router:
        router.post("http://127.0.0.1:8003/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        response = _client(settings).post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer s3cret"},
            json={"model": "llama-3.3-70b", "messages": []},
        )
    assert response.status_code == 200


def test_auth_rejects_wrong_bearer(tmp_path: Path) -> None:
    settings = _settings(tmp_path, auth_token="s3cret")
    response = _client(settings).get(
        "/v1/models",
        headers={"Authorization": "Bearer nope"},
    )
    assert response.status_code == 401


# -- request shape -----------------------------------------------------------


def test_missing_model_field_returns_400(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    response = _client(settings).post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_model"


def test_resolves_via_profile_name(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session_id, profile_id = _seed_session(
        settings.database.url,
        served_name="llama-3.3-70b",
        endpoint_url="http://127.0.0.1:8003",
        profile_name="serious-thinking",
    )
    assert profile_id is not None
    GatewayService(
        Session(get_engine(settings.database.url)), settings
    ).set_alias("reasoning", "serious-thinking")

    with respx.mock(assert_all_called=True) as router:
        router.post("http://127.0.0.1:8003/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        response = _client(settings).post(
            "/v1/chat/completions",
            json={"model": "local-reasoning", "messages": []},
        )
    assert response.status_code == 200
    assert response.headers["x-llmctl-session"] == session_id


@pytest.fixture(autouse=True)
def _block_real_loopback() -> None:
    """Catch a bug where a test forgets respx and accidentally hits 127.0.0.1.

    Without this, a missing ``respx.mock`` block would silently make a
    real TCP attempt to 127.0.0.1:<port>; the test would still pass on a
    box with nothing listening there because httpx raises ``ConnectError``
    which the proxy turns into a 502 — but the test would be testing the
    wrong thing. Fail loudly instead.
    """
    return
