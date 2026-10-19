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
from llmctl.services.lanes import Lane, SwapState, SwapTarget


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMCTL_API_TOKEN", raising=False)


def _state() -> SwapState:
    catalog = [
        SwapTarget(
            key="lane-a",
            kind="lane",
            model="a",
            engine="vllm",
            upstream="http://h:1",
            lane=Lane("lane-a", "a", "http://h:1", ("start",)),
        ),
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


# --- POST /swap/run -------------------------------------------------------

import sys  # noqa: E402
import threading  # noqa: E402


def _post(client: TestClient, name: str, dry_run: bool = False):
    with patch("llmctl.services.lanes.gather_state", return_value=_state()):
        return client.post("/swap/run", json={"name": name, "dry_run": dry_run})


def test_run_unknown_404_with_suggestions(tmp_path: Path) -> None:
    client, _ = _make(tmp_path)
    resp = _post(client, "lane-x")
    assert resp.status_code == 404
    assert "lane-a" in resp.json()["detail"]


def test_run_row_numbers(tmp_path: Path) -> None:
    client, _ = _make(tmp_path)
    assert _post(client, "2", dry_run=True).json()["name"] == "sess-b"
    assert _post(client, "0").status_code == 404
    assert _post(client, "3").status_code == 404


def test_run_dry_run_and_already_serving(tmp_path: Path) -> None:
    client, _ = _make(tmp_path)
    with patch("llmctl.services.swap_jobs.run_argv") as run:
        dry = _post(client, "sess-b", dry_run=True)
        serving = _post(client, "lane-a")
        assert run.call_count == 0
    assert dry.status_code == 200
    assert dry.json()["already_serving"] is False
    assert serving.status_code == 200
    assert serving.json()["already_serving"] is True
    assert serving.json()["endpoint"] == "http://h:1/v1"
    assert _get(client).json()["job"] is None


def test_run_job_lifecycle_and_busy(tmp_path: Path) -> None:
    client, _ = _make(tmp_path)
    gate = threading.Event()
    seen: list[list[str]] = []

    def fake(argv, on_line):
        seen.append(argv)
        on_line("hello")
        gate.wait(5)
        return 0

    with patch("llmctl.services.swap_jobs.run_argv", fake):
        resp = _post(client, "sess-b")
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        assert resp.json()["status"] == "running"
        busy = _post(client, "sess-b")
        assert busy.status_code == 409
        assert job_id in busy.json()["detail"]
        assert _post(client, "sess-b", dry_run=True).status_code == 200
        assert _get(client).json()["job"]["job_id"] == job_id
        gate.set()
        from llmctl.api.deps import get_swap_jobs

        client.app.dependency_overrides[get_swap_jobs]().wait(job_id, 5)
        done = client.get(f"/swap/jobs/{job_id}").json()
    assert done["status"] == "succeeded"
    assert done["exit_code"] == 0
    assert done["output"] == ["hello"]
    assert seen == [[sys.executable, "-m", "llmctl.main", "run", "sess-b", "--yes"]]


def test_a_huge_row_number_is_404_not_500(tmp_path: Path) -> None:
    """int() refuses strings over 4300 digits; that must not become a 500."""
    client, _ = _make(tmp_path)
    assert _post(client, "9" * 5000).status_code == 404
