import asyncio
from typing import Any

from typer.testing import CliRunner

from llmctl import cli
from llmctl.schemas import Model, RuntimeName
from llmctl.services import health as health_mod
from llmctl.services.health import HealthService, _default_lane_probe
from llmctl.services.lanes import Lane
from tests.test_health_reality import _Adapter, _Router


def _lane(name: str, runtime: str = "ollama") -> Lane:
    return Lane(
        name=name,
        served_model=f"model-{name}",
        upstream="http://127.0.0.1:11434",
        start_cmd=("true",),
        runtime=runtime,
    )


def test_ollama_lane_answers_only_when_loaded(monkeypatch) -> None:
    monkeypatch.setattr(health_mod, "probe_ollama_loaded", lambda url, t: ["model-gemma1"])
    lanes = [_lane("gemma1"), _lane("gemma2"), _lane("gemma3")]
    got = [asyncio.run(_default_lane_probe(ln)) for ln in lanes]
    assert got == [True, False, False]


def test_non_ollama_lane_skips_api_ps(monkeypatch) -> None:
    def boom(*a: Any, **k: Any):
        raise AssertionError("api/ps must not be used")

    urls: list[str] = []

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def get(self, url: str):
            urls.append(url)
            return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(health_mod, "probe_ollama_loaded", boom)
    monkeypatch.setattr("httpx.AsyncClient", _Client)
    assert asyncio.run(_default_lane_probe(_lane("v", runtime="vllm"))) is True
    assert urls == ["http://127.0.0.1:11434/v1/models"]


def test_lane_owned_model_reported_once() -> None:
    lanes = [_lane("gemma1"), _lane("gemma2")]
    ad = _Adapter(
        "Ollama ok",
        loaded=[Model(id="x", name="model-gemma1", runtime=RuntimeName.OLLAMA)],
        runtime=RuntimeName.OLLAMA,
    )

    async def probe(lane: Lane) -> bool:
        return lane.name == "gemma1"

    router = _Router({RuntimeName.OLLAMA: ad})
    svc = HealthService(None, router, None, lanes=lanes, lane_probe=probe)
    rt = svc.get_health()["runtimes"]["ollama"]
    assert len(rt["sessions"]) == 1
    s = rt["sessions"][0]
    assert s["served_name"] == "model-gemma1" and s["lane"] == "gemma1" and s["managed"] is False
    assert rt["serving_count"] == 1 and rt["unmanaged_count"] == 1


def test_cli_labels(monkeypatch) -> None:
    data = {
        "state": "ok", "safe_mode": False, "gpu_count": 0, "nvml_available": False,
        "runtimes": {"ollama": {"state": "ok", "message": "m", "sessions": [
            {"id": None, "served_name": "a", "endpoint_url": None,
             "managed": False, "lane": "gemma1"},
            {"id": None, "served_name": "b", "endpoint_url": None, "managed": False},
        ]}},
    }  # fmt: skip
    monkeypatch.setattr(HealthService, "get_health", lambda self: data)
    out = CliRunner().invoke(cli.app, ["health"]).output
    assert "(lane gemma1)" in out and "(unmanaged)" in out
