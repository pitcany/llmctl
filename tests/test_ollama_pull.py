"""Ollama model pull: streaming POST /api/pull adapter method + `llmctl pull`."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from typer.testing import CliRunner

from llmctl.adapters.ollama import OllamaAdapter
from llmctl.cli import app as cli_app
from llmctl.schemas import AdapterStatus, HealthState

runner = CliRunner()


def _factory(handler):
    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))

    return factory


def _ndjson(*events: dict) -> bytes:
    return "".join(json.dumps(e) + "\n" for e in events).encode()


def test_pull_streams_progress_and_succeeds() -> None:
    seen_request: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pull"
        seen_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            content=_ndjson(
                {"status": "pulling manifest"},
                {"status": "downloading sha256:abc", "completed": 5, "total": 10},
                {"status": "downloading sha256:abc", "completed": 10, "total": 10},
                {"status": "verifying sha256 digest"},
                {"status": "success"},
            ),
        )

    adapter = OllamaAdapter(client_factory=_factory(handler))
    events: list[tuple[str, int | None, int | None]] = []
    status = asyncio.run(
        adapter.pull_model("qwen3:32b", on_progress=lambda *a: events.append(a))
    )
    assert status.state == HealthState.OK
    assert "qwen3:32b" in status.message
    assert seen_request == {"name": "qwen3:32b", "stream": True}
    assert events[0] == ("pulling manifest", None, None)
    assert ("downloading sha256:abc", 10, 10) in events
    assert events[-1] == ("success", None, None)


def test_pull_error_event_is_degraded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(
                {"status": "pulling manifest"},
                {"error": "pull model manifest: file does not exist"},
            ),
        )

    adapter = OllamaAdapter(client_factory=_factory(handler))
    status = asyncio.run(adapter.pull_model("nope:latest"))
    assert status.state == HealthState.DEGRADED
    assert "file does not exist" in status.message


def test_pull_http_error_is_degraded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    adapter = OllamaAdapter(client_factory=_factory(handler))
    status = asyncio.run(adapter.pull_model("qwen3:32b"))
    assert status.state == HealthState.DEGRADED


def test_pull_daemon_down_is_degraded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    adapter = OllamaAdapter(client_factory=_factory(handler))
    status = asyncio.run(adapter.pull_model("qwen3:32b"))
    assert status.state == HealthState.DEGRADED
    assert "connection refused" in status.message


def test_pull_skips_malformed_ndjson_lines() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'not-json\n[1,2]\n{"status": "success"}\n',
        )

    adapter = OllamaAdapter(client_factory=_factory(handler))
    events: list[tuple[str, int | None, int | None]] = []
    status = asyncio.run(
        adapter.pull_model("qwen3:32b", on_progress=lambda *a: events.append(a))
    )
    assert status.state == HealthState.OK
    assert events == [("success", None, None)]


class _FakePullAdapter(OllamaAdapter):
    """OllamaAdapter whose pull_model replays canned progress events."""

    def __init__(self, result_state: HealthState) -> None:
        super().__init__()
        self._result_state = result_state

    async def pull_model(self, name, *, on_progress=None):  # type: ignore[override]
        if on_progress is not None:
            on_progress("pulling manifest", None, None)
            on_progress("downloading sha256:abc", 1024**3, 2 * 1024**3)
            on_progress("success", None, None)
        return AdapterStatus(
            runtime=self.runtime,
            state=self._result_state,
            message=f"pull of '{name}' finished ({self._result_state.value}).",
        )


def _patch_router(monkeypatch: pytest.MonkeyPatch, adapter: OllamaAdapter) -> None:
    from llmctl.services import router as router_mod

    monkeypatch.setattr(
        router_mod.RuntimeRouter, "get_adapter", lambda self, runtime: adapter
    )


def test_cli_pull_prints_progress(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'pull.sqlite3'}")
    _patch_router(monkeypatch, _FakePullAdapter(HealthState.OK))
    result = runner.invoke(cli_app, ["pull", "qwen3:32b"])
    assert result.exit_code == 0
    assert "pulling manifest" in result.output
    assert "50%" in result.output
    assert "llmctl scan" in result.output


def test_cli_pull_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'pull2.sqlite3'}")
    _patch_router(monkeypatch, _FakePullAdapter(HealthState.DEGRADED))
    result = runner.invoke(cli_app, ["pull", "nope:latest"])
    assert result.exit_code == 1
