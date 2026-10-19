"""POST /swap/stop."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from llmctl.api.app import create_app
from llmctl.config import Settings
from llmctl.services.lanes import Lane, SwapState, SwapTarget
from llmctl.services.stop_serving import StopDeps


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLMCTL_API_TOKEN", raising=False)


def _state(lanes: int = 1, vllm: bool = False) -> SwapState:
    catalog = [
        SwapTarget(
            key="lane-a",
            kind="lane",
            model="a",
            engine="vllm",
            upstream="http://h:1",
            lane=Lane("lane-a", "a", "http://h:1", ("start",), unit="ua", gpus=frozenset({1})),
        ),
        SwapTarget(key="sess-b", kind="session", model="b", engine="llama.cpp", upstream=""),
    ]
    return SwapState(
        catalog=catalog,
        serving={"lane-a"} if lanes else set(),
        active_units={"ua"} if lanes else set(),
        vllm_unit="v",
        vllm_active=vllm,
    )


def _client(tmp_path: Path) -> TestClient:
    settings = Settings()
    url = f"sqlite:///{tmp_path / 'swap.sqlite3'}"
    settings.database.url = url
    return TestClient(create_app(settings, database_url=url))


class _Deps(StopDeps):
    def __init__(self, ok: bool = True, boom: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.ok, self.boom = ok, boom

    def _do(self, kind: str, ref: str) -> bool:
        self.calls.append((kind, ref))
        if self.boom:
            raise RuntimeError("x")
        return self.ok

    def stop_vllm(self, ref):  # type: ignore[override]
        return self._do("vllm", ref)

    def stop_lane(self, ref):  # type: ignore[override]
        return self._do("lane", ref)

    def stop_session(self, ref):  # type: ignore[override]
        return self._do("session", ref)


def _stop(client, state, deps, **body):
    with (
        patch("llmctl.services.lanes.gather_state", return_value=state),
        patch("llmctl.services.stop_serving.real_deps", return_value=deps),
    ):
        return client.post("/swap/stop", json=body)


def test_nothing_serving(tmp_path: Path) -> None:
    r = _stop(_client(tmp_path), _state(0), _Deps())
    assert r.status_code == 200
    assert r.json() == {"items": [], "results": [], "ok": True}


def test_unknown_name_404(tmp_path: Path) -> None:
    assert _stop(_client(tmp_path), _state(), _Deps(), name="zzz").status_code == 404


def test_two_items_need_all(tmp_path: Path) -> None:
    client, deps = _client(tmp_path), _Deps()
    state = _state(vllm=True)
    r = _stop(client, state, deps)
    assert r.status_code == 409
    assert "lane lane-a" in r.json()["detail"] and "vLLM unit v" in r.json()["detail"]
    assert deps.calls == []
    r = _stop(client, state, deps, all=True)
    assert r.status_code == 200
    assert sorted(deps.calls) == [("lane", "ua"), ("vllm", "v")]
    assert r.json()["ok"] is True


def test_items_shape_and_dry_run(tmp_path: Path) -> None:
    deps = _Deps()
    r = _stop(_client(tmp_path), _state(vllm=True), deps, all=True, dry_run=True)
    body = r.json()
    assert r.status_code == 200 and body["results"] == [] and body["ok"] is True
    assert all(set(i) == {"kind", "label", "ref", "cards"} for i in body["items"])
    assert {i["kind"]: i["cards"] for i in body["items"]} == {"vllm": "all", "lane": "1"}
    assert deps.calls == []


@pytest.mark.parametrize("deps", [_Deps(ok=False), _Deps(boom=True)])
def test_failed_stop_not_ok(tmp_path: Path, deps: _Deps) -> None:
    body = _stop(_client(tmp_path), _state(), deps, name="lane-a").json()
    assert body["ok"] is False
    assert set(body["results"][0]) == {"label", "ref", "stopped"}
    assert body["results"][0]["stopped"] is False


def test_real_stop_blocked_by_run_job(tmp_path: Path) -> None:
    client, deps, gate = _client(tmp_path), _Deps(), threading.Event()

    def fake(argv, on_line):
        gate.wait(5)
        return 0

    with patch("llmctl.services.swap_jobs.run_argv", fake):
        with patch("llmctl.services.lanes.gather_state", return_value=_state()):
            job = client.post("/swap/run", json={"name": "sess-b"}).json()["job_id"]
        real = _stop(client, _state(), deps, name="lane-a")
        dry = _stop(client, _state(), deps, name="lane-a", dry_run=True)
        gate.set()
    assert real.status_code == 409 and job in real.json()["detail"]
    assert dry.status_code == 200
    assert deps.calls == []
