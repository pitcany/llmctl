"""GET /swap and GET /swap/jobs/{id}."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from llmctl.api.app import create_app
from llmctl.config import Settings
from llmctl.services import swap_view
from llmctl.services.gateway import GatewayService
from llmctl.services.lanes import SwapState, SwapTarget


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMCTL_API_TOKEN", raising=False)


def _state() -> SwapState:
    catalog = [
        SwapTarget(key="lane-a", kind="lane", model="a", engine="vllm", upstream="http://h:1"),
        SwapTarget(key="sess-b", kind="session", model="b", engine="llama.cpp", upstream=""),
    ]
    return SwapState(
        catalog=catalog,
        serving={"lane-a"},
        active_units=set(),
        vllm_unit="v",
        vllm_active=False,
    )


def _make(tmp_path: Path, **auth) -> tuple[TestClient, Settings]:
    settings = Settings()
    url = f"sqlite:///{tmp_path / 'swap.sqlite3'}"
    settings.database.url = url
    if auth:
        settings.scheduler.require_auth_token = True
        settings.api.auth_token = auth["token"]
    return TestClient(create_app(settings, database_url=url)), settings


def _get(client: TestClient, **kw):
    with patch("llmctl.services.lanes.gather_state", return_value=_state()):
        return client.get("/swap", **kw)


def test_swap_rows_match_builder(tmp_path: Path) -> None:
    client, settings = _make(tmp_path)
    body = _get(client).json()
    assert body["rows"] == swap_view.rows_json(_state(), [])
    assert [list(r) for r in body["rows"]] == [list(r) for r in swap_view.rows_json(_state())]
    r = settings.router
    assert body["gateway_url"] == f"http://{r.host}:{r.port}/v1"
    assert body["job"] is None


def test_swap_aliases(tmp_path: Path) -> None:
    client, _ = _make(tmp_path)
    alias = SimpleNamespace(name="fast", healthy=True, resolved_served_name="a")
    with patch.object(GatewayService, "alias_view", return_value=[alias]):
        rows = _get(client).json()["rows"]
    assert rows[0]["aliases"] == ["local-fast"]
    assert rows[1]["aliases"] == []


def test_swap_alias_failure_still_200(tmp_path: Path) -> None:
    client, _ = _make(tmp_path)
    with patch.object(GatewayService, "alias_view", side_effect=RuntimeError("boom")):
        resp = _get(client)
    assert resp.status_code == 200
    assert resp.json()["rows"][0]["aliases"] == []


def test_unknown_job_404(tmp_path: Path) -> None:
    client, _ = _make(tmp_path)
    assert client.get("/swap/jobs/unknown").status_code == 404


def test_swap_requires_token(tmp_path: Path) -> None:
    client, _ = _make(tmp_path, token="tok")
    assert _get(client).status_code == 401
    assert _get(client, headers={"Authorization": "Bearer tok"}).status_code == 200
