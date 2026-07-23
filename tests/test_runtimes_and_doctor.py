"""Tests for adapter capability honesty, runtime inventory, doctor v2, and JSON output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmctl.adapters.llama_cpp import LlamaCppAdapter
from llmctl.adapters.lmstudio import LMStudioAdapter
from llmctl.adapters.ollama import OllamaAdapter
from llmctl.adapters.python_script import PythonScriptAdapter
from llmctl.cli import app
from llmctl.config import RuntimeConfig

runner = CliRunner()

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Isolated config dir + DB so CLI invocations never touch user state."""
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'cli.sqlite3'}")
    return tmp_path


def test_capability_honesty_per_adapter() -> None:
    ollama = OllamaAdapter().capabilities()
    assert ollama["delete_model"] and ollama["version"] and ollama["list_loaded_models"]
    assert not ollama["launch_process"] and not ollama["logs"]

    lmstudio = LMStudioAdapter().capabilities()
    assert lmstudio["list_loaded_models"]
    assert not lmstudio["delete_model"] and not lmstudio["launch_process"]

    llama = LlamaCppAdapter(RuntimeConfig()).capabilities()
    assert llama["launch_process"] and llama["stop_process"] and llama["logs"]
    assert llama["version"] and llama["discover_models"]
    assert not llama["delete_model"]

    script = PythonScriptAdapter(RuntimeConfig()).capabilities()
    assert not script["discover_models"]  # no filesystem discovery for scripts
    assert script["launch_process"]


def test_capability_keys_are_stable() -> None:
    from llmctl.adapters.base import CAPABILITY_KEYS

    for adapter in (OllamaAdapter(), LlamaCppAdapter(RuntimeConfig())):
        assert tuple(adapter.capabilities().keys()) == CAPABILITY_KEYS


def test_ollama_version_and_loaded(monkeypatch) -> None:
    import asyncio

    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.9.9"})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(
        client_factory=lambda: httpx.AsyncClient(
            base_url="http://test", transport=transport
        )
    )
    assert asyncio.run(adapter.version()) == "0.9.9"
    loaded = asyncio.run(adapter.list_loaded_models())
    assert loaded is not None and loaded[0].name == "qwen3:8b"


def test_runtime_inventory_degrades_on_probe_errors(monkeypatch) -> None:
    from llmctl.services.runtimes import runtime_inventory

    class BoomAdapter:
        display_name = "Boom"
        endpoint = None

        async def health_check(self):
            raise RuntimeError("probe exploded")

        async def version(self):
            raise RuntimeError("no version")

        async def list_loaded_models(self):
            return None

        def capabilities(self):
            return {"health_check": True}

    class FakeRouter:
        def list_runtimes(self):
            from llmctl.db import RuntimeName

            return [RuntimeName.OLLAMA]

        def get_adapter(self, runtime):
            return BoomAdapter()

    rows = runtime_inventory(router=FakeRouter())
    assert rows[0]["state"] == "unknown"
    assert "probe exploded" in rows[0]["message"]
    assert rows[0]["version"] is None
    assert rows[0]["loaded"] is None


def test_runtimes_json_is_parseable(isolated_env) -> None:
    result = runner.invoke(app, ["runtimes", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    names = {row["runtime"] for row in rows}
    assert {"vllm", "llama_cpp", "lmstudio", "ollama", "python_script"} <= names
    for row in rows:
        assert set(row) >= {"runtime", "state", "version", "capabilities", "loaded"}


def test_runtimes_inspect_unknown_is_usage_error(isolated_env) -> None:
    result = runner.invoke(app, ["runtimes", "inspect", "nope"])
    assert result.exit_code == 2


def test_models_and_sessions_json(isolated_env) -> None:
    result = runner.invoke(app, ["models", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []

    result = runner.invoke(app, ["sessions", "--no-fresh", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_doctor_reports_sections(isolated_env) -> None:
    from llmctl.config import load_settings
    from llmctl.services.doctor import run_doctor

    report = run_doctor(load_settings())
    assert set(report) == {"passed", "warnings", "failures", "ok"}
    assert report["ok"] == (not report["failures"])
    names = {c["name"] for c in report["passed"] + report["warnings"] + report["failures"]}
    assert "database" in names
    assert "stale-sessions" in names


def test_doctor_flags_port_collisions(isolated_env, tmp_path) -> None:
    from sqlmodel import Session as DBSession

    from llmctl.config import load_settings
    from llmctl.db import (
        RuntimeName,
        SessionRecord,
        SessionStatus,
        get_engine,
        init_db,
    )
    from llmctl.services.doctor import run_doctor

    settings = load_settings()
    init_db(settings.database_url)
    with DBSession(get_engine(settings.database_url)) as db:
        for _ in range(2):
            db.add(
                SessionRecord(
                    runtime=RuntimeName.LLAMA_CPP,
                    status=SessionStatus.RUNNING,
                    port=8123,
                )
            )
        db.commit()
    report = run_doctor(settings)
    assert not report["ok"]
    assert any(c["name"] == "port-collisions" for c in report["failures"])


def test_doctor_json_exit_code(isolated_env) -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    report = json.loads(result.output)
    assert result.exit_code == (0 if report["ok"] else 1)
