"""Tests for runtime adapters (HTTP and process-launch)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

from llmctl.adapters.llama_cpp import LlamaCppAdapter
from llmctl.adapters.lmstudio import LMStudioAdapter
from llmctl.adapters.ollama import OllamaAdapter
from llmctl.adapters.python_script import PythonScriptAdapter
from llmctl.config import RuntimeConfig
from llmctl.db import RuntimeName, SessionStatus
from llmctl.schemas import HealthState, LaunchPlan
from llmctl.telemetry.process import ProcessSupervisor


def _factory(handler):
    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))

    return factory


def test_ollama_discovery_and_health() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "llama3:8b",
                            "size": 4096,
                            "digest": "abc",
                            "details": {
                                "format": "gguf",
                                "quantization_level": "Q4_0",
                                "parameter_size": "8B",
                            },
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"version": "0.1.0"})

    adapter = OllamaAdapter(client_factory=_factory(handler))
    models = asyncio.run(adapter.discover_models())
    assert len(models) == 1
    assert models[0].name == "llama3:8b"
    assert models[0].quantization == "Q4_0"

    health = asyncio.run(adapter.health_check())
    assert health.state == HealthState.OK


def test_ollama_unreachable_is_graceful() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    adapter = OllamaAdapter(client_factory=_factory(handler))
    assert asyncio.run(adapter.discover_models()) == []
    health = asyncio.run(adapter.health_check())
    assert health.state == HealthState.UNAVAILABLE


def test_lmstudio_discovery() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "qwen2.5", "owned_by": "local"}]})

    adapter = LMStudioAdapter(client_factory=_factory(handler))
    models = asyncio.run(adapter.discover_models())
    assert len(models) == 1
    assert models[0].name == "qwen2.5"
    assert models[0].runtime == RuntimeName.LMSTUDIO


def test_process_adapter_health_missing_binary() -> None:
    config = RuntimeConfig(binary="definitely-not-a-real-binary-xyz")
    adapter = LlamaCppAdapter(config, ProcessSupervisor())
    health = asyncio.run(adapter.health_check())
    assert health.state == HealthState.UNAVAILABLE


def test_process_adapter_dry_run_does_not_launch() -> None:
    adapter = PythonScriptAdapter(RuntimeConfig(), ProcessSupervisor())
    plan = LaunchPlan(runtime=RuntimeName.PYTHON_SCRIPT, command=["echo", "hi"], dry_run=True)
    session = asyncio.run(adapter.start(plan))
    assert session.status == SessionStatus.PLANNED
    assert session.pid is None


def test_process_adapter_real_launch_and_stop(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(log_dir=tmp_path)
    adapter = PythonScriptAdapter(RuntimeConfig(), supervisor)
    plan = LaunchPlan(
        runtime=RuntimeName.PYTHON_SCRIPT,
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
        dry_run=False,
    )
    session = asyncio.run(adapter.start(plan))
    try:
        assert session.status == SessionStatus.RUNNING
        assert session.pid is not None
        assert supervisor.is_running(session.pid)
    finally:
        status = asyncio.run(adapter.stop(session))
    assert status.state == HealthState.OK
    assert not supervisor.is_running(session.pid)


def test_process_adapter_missing_command_fails() -> None:
    adapter = PythonScriptAdapter(RuntimeConfig(), ProcessSupervisor())
    plan = LaunchPlan(runtime=RuntimeName.PYTHON_SCRIPT, command=[], dry_run=False)
    session = asyncio.run(adapter.start(plan))
    assert session.status == SessionStatus.FAILED
    assert session.error is not None
