"""Tests for the FreeToken runtime adapter.

The behaviour not shared with the other HTTP adapters, and the reason this
file exists: FreeToken answers ``/health`` with **200 while still loading**
(``{"status": "loading"}``), so reachable is not the same as ready. The version
also lives in that body; there is no separate version endpoint.
"""

from __future__ import annotations

import httpx
import pytest

from llmctl.adapters.freetoken import FreeTokenAdapter
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import HealthState

# Shape captured from FreeToken 0.1.3 on 2026-09-18.
MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "qwen3.8-flash-next-freetoken",
            "object": "model",
            "created": 1789755595,
            "owned_by": "FreeToken",
            "root": "/home/yannik/models/Qwen3.8-Flash-Next-NVFP4",
            "max_model_len": 131072,
            "context_length": 262144,
            "supported_reasoning_efforts": ["xhigh", "medium", "low"],
            "default_reasoning_effort": "xhigh",
        }
    ],
}

HEALTH_OK = {
    "status": "ok",
    "model": "qwen3.8-flash-next-freetoken",
    "uptime_s": 12418,
    "maintenance": "serving",
    "version": "0.1.3",
}


def _client_factory(handler):
    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://freetoken.test"
        )

    return factory


def _adapter(handler) -> FreeTokenAdapter:
    return FreeTokenAdapter("http://freetoken.test", client_factory=_client_factory(handler))


@pytest.mark.asyncio
async def test_discover_models_carries_served_context_window():
    adapter = _adapter(lambda request: httpx.Response(200, json=MODELS_PAYLOAD))
    models = await adapter.discover_models()
    assert len(models) == 1
    model = models[0]
    assert model.name == "qwen3.8-flash-next-freetoken"
    assert model.runtime is RuntimeName.FREETOKEN
    assert model.status is ModelStatus.DISCOVERED
    assert model.source == "/home/yannik/models/Qwen3.8-Flash-Next-NVFP4"
    # max_model_len wins over context_length when both are present.
    assert model.max_context == 131072


@pytest.mark.asyncio
async def test_discover_falls_back_to_context_length():
    row = {k: v for k, v in MODELS_PAYLOAD["data"][0].items() if k != "max_model_len"}
    adapter = _adapter(lambda request: httpx.Response(200, json={"data": [row]}))
    models = await adapter.discover_models()
    assert models[0].max_context == 262144


@pytest.mark.asyncio
async def test_health_ok_when_status_ok():
    status = await _adapter(lambda request: httpx.Response(200, json=HEALTH_OK)).health_check()
    assert status.state is HealthState.OK
    assert status.details["version"] == "0.1.3"


@pytest.mark.asyncio
async def test_health_degraded_while_loading():
    """A 200 with status=loading is reachable but must not read as serving."""
    body = {"status": "loading", "model": "qwen3.8-flash-next-freetoken"}
    status = await _adapter(lambda request: httpx.Response(200, json=body)).health_check()
    assert status.state is HealthState.DEGRADED
    assert status.details["status"] == "loading"


@pytest.mark.asyncio
async def test_health_unavailable_when_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    status = await _adapter(handler).health_check()
    assert status.state is HealthState.UNAVAILABLE


@pytest.mark.asyncio
async def test_version_from_health_body():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json=HEALTH_OK)

    assert await _adapter(handler).version() == "0.1.3"


@pytest.mark.asyncio
async def test_version_none_when_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await _adapter(handler).version() is None


@pytest.mark.asyncio
async def test_never_claims_process_control():
    """The systemd unit and its VRAM guard own the lifecycle, not llmctl."""
    caps = _adapter(lambda r: httpx.Response(200, json=MODELS_PAYLOAD)).capabilities()
    assert caps["launch_process"] is False
    assert caps["stop_process"] is False
    assert caps["discover_models"] is True
    assert caps["health_check"] is True
    assert caps["version"] is True


@pytest.mark.asyncio
async def test_malformed_payload_yields_no_models():
    adapter = _adapter(lambda request: httpx.Response(200, json={"data": "not-a-list"}))
    assert await adapter.discover_models() == []
