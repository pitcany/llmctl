"""Control-plane API bearer auth (`scheduler.require_auth_token`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llmctl.api.app import create_app
from llmctl.cli import app as cli_app
from llmctl.config import Settings, resolve_api_auth_token

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the invoking shell's LLMCTL_API_TOKEN out of every test."""
    monkeypatch.delenv("LLMCTL_API_TOKEN", raising=False)


def _client(tmp_path: Path, *, require: bool, token: str | None) -> TestClient:
    settings = Settings()
    settings.scheduler.require_auth_token = require
    settings.api.auth_token = token
    return TestClient(
        create_app(settings, database_url=f"sqlite:///{tmp_path / 'auth.sqlite3'}")
    )


def test_health_and_docs_stay_open(tmp_path: Path) -> None:
    client = _client(tmp_path, require=True, token="tok-123")
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_missing_token_is_401_with_challenge(tmp_path: Path) -> None:
    client = _client(tmp_path, require=True, token="tok-123")
    response = client.get("/models")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_wrong_token_is_401(tmp_path: Path) -> None:
    client = _client(tmp_path, require=True, token="tok-123")
    response = client.get("/models", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_doctor_requires_token(tmp_path: Path) -> None:
    client = _client(tmp_path, require=True, token="tok-123")
    assert client.get("/doctor").status_code == 401


def test_correct_token_allows_read_and_write(tmp_path: Path) -> None:
    client = _client(tmp_path, require=True, token="tok-123")
    headers = {"Authorization": "Bearer tok-123"}
    assert client.get("/models", headers=headers).status_code == 200
    created = client.post(
        "/models",
        headers=headers,
        json={"name": "example", "runtime": "ollama", "source": "example:latest"},
    )
    assert created.status_code == 201


def test_enabled_without_token_fails_closed(tmp_path: Path) -> None:
    client = _client(tmp_path, require=True, token=None)
    response = client.get("/models")
    assert response.status_code == 503
    assert "no token is configured" in response.json()["detail"]
    # Liveness stays observable so operators can see the service is up.
    assert client.get("/health").status_code == 200


def test_empty_string_token_fails_closed_like_none(tmp_path: Path) -> None:
    """`auth_token: ""` must NOT become a token that `Bearer ` matches."""
    client = _client(tmp_path, require=True, token="")
    assert client.get("/models").status_code == 503
    # The empty-credential probe that would pass compare_digest("", "").
    assert client.get("/models", headers={"Authorization": "Bearer "}).status_code == 503


def test_whitespace_padded_token_is_normalized(tmp_path: Path) -> None:
    """A YAML block scalar's trailing newline must not lock out clean headers."""
    client = _client(tmp_path, require=True, token="tok-123\n")
    response = client.get("/models", headers={"Authorization": "Bearer tok-123"})
    assert response.status_code == 200


def test_env_var_token_overrides_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLMCTL_API_TOKEN", "env-tok")
    client = _client(tmp_path, require=True, token=None)
    assert client.get("/models").status_code == 401
    response = client.get("/models", headers={"Authorization": "Bearer env-tok"})
    assert response.status_code == 200


def test_auth_disabled_by_default(tmp_path: Path) -> None:
    client = _client(tmp_path, require=False, token=None)
    assert client.get("/models").status_code == 200


def test_resolve_api_auth_token_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings()
    settings.api.auth_token = "from-yaml"
    assert resolve_api_auth_token(settings) == "from-yaml"
    monkeypatch.setenv("LLMCTL_API_TOKEN", "from-env")
    assert resolve_api_auth_token(settings) == "from-env"


def test_serve_refuses_auth_required_without_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        "scheduler:\n  require_auth_token: true\n", encoding="utf-8"
    )
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'serve.sqlite3'}")
    monkeypatch.setattr(
        uvicorn, "run", lambda *a, **k: pytest.fail("guard must fire before binding")
    )
    result = runner.invoke(cli_app, ["serve"])
    assert result.exit_code == 2
    # Rich wraps the error box mid-phrase, so strip the borders and
    # normalize whitespace before matching.
    plain = " ".join(result.output.replace("│", " ").split())
    assert "no token is configured" in plain
