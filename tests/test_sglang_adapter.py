"""Tests for the SGLang runtime adapter.

Two behaviours here are not shared with the other HTTP adapters and are the
reason this file exists:

* SGLang answers ``/health`` with a bare 200 and a **zero-byte body**, so the
  inherited JSON-parsing probe would report a healthy server as unavailable.
* Its ``/v1/models`` rows carry ``max_model_len`` — the window actually being
  served, which is what decides whether a harness fits.
"""

from __future__ import annotations

import httpx
import pytest

from llmctl.adapters.sglang import SGLangAdapter
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import HealthState

MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {
            "id": "qwen3.8-27b-nvfp4",
            "object": "model",
            "owned_by": "sglang",
            "root": "qwen3.8-27b-nvfp4",
            "max_model_len": 32768,
        }
    ],
}


def _client_factory(handler):
    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://sglang.test"
        )

    return factory


def _adapter(handler) -> SGLangAdapter:
    return SGLangAdapter("http://sglang.test", client_factory=_client_factory(handler))


@pytest.mark.asyncio
async def test_discover_models_carries_served_context_window():
    adapter = _adapter(lambda request: httpx.Response(200, json=MODELS_PAYLOAD))
    models = await adapter.discover_models()
    assert len(models) == 1
    model = models[0]
    assert model.name == "qwen3.8-27b-nvfp4"
    assert model.runtime is RuntimeName.SGLANG
    assert model.status is ModelStatus.DISCOVERED
    # The served window, not the checkpoint's native 262144.
    assert model.max_context == 32768


@pytest.mark.asyncio
async def test_health_ok_despite_empty_body():
    """A zero-byte 200 on /health is healthy — the real server's behaviour."""
    adapter = _adapter(lambda request: httpx.Response(200, content=b""))
    status = await adapter.health_check()
    assert status.state is HealthState.OK


@pytest.mark.asyncio
async def test_health_unavailable_when_refused_and_not_installed(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    adapter = _adapter(handler)
    monkeypatch.setattr(adapter, "installed_binary", lambda: None)
    status = await adapter.health_check()
    assert status.state is HealthState.UNAVAILABLE
    # Installed but idle is not a fault: llmctl can start it.
    monkeypatch.setattr(adapter, "installed_binary", lambda: "/opt/sglang/bin/python")
    assert (await adapter.health_check()).state is HealthState.OK


@pytest.mark.asyncio
async def test_version_from_server_info():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/get_server_info"
        return httpx.Response(200, json={"version": "0.5.19", "model_path": "x"})

    assert await _adapter(handler).version() == "0.5.19"


@pytest.mark.asyncio
async def test_version_none_when_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await _adapter(handler).version() is None


@pytest.mark.asyncio
async def test_claims_process_control_and_discovery():
    """llmctl launches `python -m sglang.launch_server` itself and watches it."""
    caps = _adapter(lambda r: httpx.Response(200, json=MODELS_PAYLOAD)).capabilities()
    assert caps["launch_process"] is True
    assert caps["stop_process"] is True
    assert caps["discover_models"] is True
    assert caps["health_check"] is True
    assert caps["version"] is True


@pytest.mark.asyncio
async def test_malformed_payload_yields_no_models():
    adapter = _adapter(lambda request: httpx.Response(200, json={"data": "not-a-list"}))
    assert await adapter.discover_models() == []
