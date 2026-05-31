"""Tests for the benchmark runner and systemd unit generation."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from llmctl.api.app import create_app
from llmctl.config import load_settings
from llmctl.db import get_engine, init_db
from llmctl.schemas import BenchmarkRunRequest, ModelCreate, SessionStartRequest
from llmctl.services.benchmarks import BenchmarkService
from llmctl.services.registry import RegistryService
from llmctl.services.sessions import SessionService
from llmctl.services.systemd import render_api_unit, render_session_unit

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def db(tmp_path, monkeypatch) -> Session:
    """Yield an isolated DB session bound to the repo configs."""
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    url = f"sqlite:///{tmp_path / 'bench.db'}"
    init_db(url)
    with Session(get_engine(url)) as session:
        yield session


def _ollama_model(db: Session) -> str:
    return RegistryService(db).add_model(
        ModelCreate(name="demo", runtime="ollama", source="demo:latest")
    ).id or ""


# -- benchmark runner -------------------------------------------------------


def test_benchmark_dry_run_uses_mock(db: Session) -> None:
    model_id = _ollama_model(db)
    result = BenchmarkService(db).run(
        BenchmarkRunRequest(name="smoke", model_id=model_id, dry_run=True)
    )
    assert result.success is True
    assert result.parameters["mode"] == "mock"
    assert result.parameters["reason"] == "dry-run requested"
    assert result.tokens_per_second == 50.0
    assert result.total_tokens and result.total_tokens > 0


def test_benchmark_live_with_injected_client(db: Session) -> None:
    model_id = _ollama_model(db)

    sse = (
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":" b c"}}]}\n\n'
        'data: {"choices":[{"delta":{}}],'
        '"usage":{"prompt_tokens":7,"completion_tokens":12}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse.encode("utf-8"))

    factory = lambda: httpx.Client(transport=httpx.MockTransport(handler))  # noqa: E731
    service = BenchmarkService(db, client_factory=factory)
    result = service.run(
        BenchmarkRunRequest(
            name="live", model_id=model_id, prompts=["hello"], dry_run=False
        )
    )
    assert result.parameters["mode"] == "live"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 12
    assert result.tokens_per_second is not None
    assert result.time_to_first_token_ms is not None
    assert result.samples and result.samples[0]["response"] == "a b c"


def test_benchmark_unreachable_falls_back_to_mock(db: Session) -> None:
    model_id = _ollama_model(db)

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    factory = lambda: httpx.Client(transport=httpx.MockTransport(handler))  # noqa: E731
    result = BenchmarkService(db, client_factory=factory).run(
        BenchmarkRunRequest(name="x", model_id=model_id, dry_run=False)
    )
    assert result.parameters["mode"] == "mock"
    assert "unreachable" in str(result.parameters["reason"])


def test_benchmark_no_endpoint_falls_back_to_mock(db: Session) -> None:
    model_id = RegistryService(db).add_model(
        ModelCreate(name="v", runtime="vllm", source="org/model")
    ).id
    result = BenchmarkService(db).run(
        BenchmarkRunRequest(name="x", model_id=model_id, dry_run=False)
    )
    assert result.parameters["mode"] == "mock"
    assert result.parameters["reason"] == "no reachable runtime endpoint"


def test_benchmark_concurrency_live(db: Session) -> None:
    model_id = _ollama_model(db)
    sse = (
        'data: {"choices":[{"delta":{"content":"x y"}}],'
        '"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse.encode("utf-8"))

    factory = lambda: httpx.Client(transport=httpx.MockTransport(handler))  # noqa: E731
    result = BenchmarkService(db, client_factory=factory).run(
        BenchmarkRunRequest(
            name="load",
            model_id=model_id,
            prompts=["a", "b", "c", "d"],
            concurrency=4,
            dry_run=False,
        )
    )
    assert result.parameters["mode"] == "live"
    assert result.parameters["concurrency"] == 4
    assert len(result.samples) == 4
    assert result.completion_tokens == 16
    assert result.tokens_per_second is not None


def test_benchmark_run_sweep(db: Session) -> None:
    model_id = _ollama_model(db)
    results = BenchmarkService(db).run_sweep(
        BenchmarkRunRequest(
            name="sweep", model_id=model_id, sweep=[4, 1, 2], dry_run=True
        )
    )
    assert len(results) == 3
    levels = sorted(int(r.parameters["concurrency"]) for r in results)
    assert levels == [1, 2, 4]
    assert all("(c=" in r.name for r in results)
    assert len(BenchmarkService(db).list_results()) == 3


def test_benchmark_delta_helpers() -> None:
    from types import SimpleNamespace

    from llmctl.tui.screens_benchmarks import BenchmarksScreen

    base = SimpleNamespace(
        id="b", tokens_per_second=100.0, time_to_first_token_ms=50.0
    )
    faster = SimpleNamespace(
        id="f", tokens_per_second=120.0, time_to_first_token_ms=40.0
    )
    assert "+20.0" in BenchmarksScreen._delta_tps(faster, base, False)
    assert "-10 ms" in BenchmarksScreen._delta_ttft(faster, base, False)
    assert "baseline" in BenchmarksScreen._delta_tps(base, base, True)
    assert BenchmarksScreen._delta_tps(faster, None, False) == "-"


def test_benchmark_mock_has_samples_and_ttft(db: Session) -> None:
    model_id = _ollama_model(db)
    result = BenchmarkService(db).run(
        BenchmarkRunRequest(
            name="s", model_id=model_id, prompts=["one", "two"], dry_run=True
        )
    )
    assert result.time_to_first_token_ms is not None
    assert len(result.samples) == 2
    assert result.samples[0]["prompt"] == "one"


def test_benchmark_rerun_creates_new_record(db: Session) -> None:
    model_id = _ollama_model(db)
    first = BenchmarkService(db).run(
        BenchmarkRunRequest(
            name="rr", model_id=model_id, prompts=["hi"], dry_run=True
        )
    )
    second = BenchmarkService(db).rerun(first.id or "")
    assert second is not None
    assert second.id != first.id
    assert second.name == "rr"
    assert len(BenchmarkService(db).list_results()) == 2


def test_benchmark_list_results(db: Session) -> None:
    model_id = _ollama_model(db)
    BenchmarkService(db).run(BenchmarkRunRequest(name="a", model_id=model_id, dry_run=True))
    results = BenchmarkService(db).list_results()
    assert len(results) == 1
    assert results[0].name == "a"


# -- systemd unit generation ------------------------------------------------


def test_render_api_unit_contains_execstart() -> None:
    unit = render_api_unit(load_settings(), user=True)
    assert unit.name == "llm-mission-control.service"
    assert "ExecStart=llmctl serve" in unit.content
    assert "WantedBy=default.target" in unit.content
    assert any("systemctl --user" in cmd for cmd in unit.install_commands())


def test_render_session_unit_process_runtime_embeds_command(
    db: Session, tmp_path: Path
) -> None:
    script = tmp_path / "serve.py"
    script.write_text("print('hi')")
    model_id = RegistryService(db).add_model(
        ModelCreate(name="s", runtime="python_script", path=str(script))
    ).id
    request = SessionStartRequest(
        model_id=model_id or "",
        runtime="python_script",
        gpu_mode="cpu",
        allow_cpu=True,
        dry_run=True,
        force=True,
    )
    session = SessionService(db).start(request)
    unit = render_session_unit(session, user=False)
    assert unit.name.startswith("llmctl-session-")
    assert "ExecStart=" in unit.content
    assert "serve.py" in unit.content
    assert "WantedBy=multi-user.target" in unit.content


def test_render_session_unit_server_runtime_warns(db: Session) -> None:
    model_id = _ollama_model(db)
    request = SessionStartRequest(
        model_id=model_id, runtime="ollama", dry_run=True, force=True
    )
    session = SessionService(db).start(request)
    unit = render_session_unit(session, user=True)
    assert any("daemon" in warning for warning in unit.warnings)
    assert "llmctl start" in unit.content


def test_session_systemd_unit_endpoint(tmp_path: Path) -> None:
    client = TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'api.db'}"))
    model = client.post(
        "/models", json={"name": "o", "runtime": "ollama", "source": "o:latest"}
    )
    model_id = model.json()["id"]
    started = client.post(
        "/sessions/start",
        json={"model_id": model_id, "runtime": "ollama", "dry_run": True},
    )
    session_id = started.json()["id"]
    response = client.get(f"/sessions/{session_id}/systemd-unit")
    assert response.status_code == 200
    body = response.json()
    assert body["name"].startswith("llmctl-session-")
    assert "ExecStart" in body["content"]
    assert isinstance(body["install_commands"], list)

    missing = client.get("/sessions/does-not-exist/systemd-unit")
    assert missing.status_code == 404


def test_install_unit_dry_run_makes_no_changes() -> None:
    from llmctl.services.systemd import install_unit, unit_install_path

    unit = render_api_unit(load_settings(), user=True)
    report = install_unit(unit, dry_run=True)
    assert report.dry_run is True
    assert report.written is False
    assert report.enabled is False
    assert any("daemon-reload" in action for action in report.actions)
    # Nothing should have been written to disk.
    assert not unit_install_path(unit).exists() or report.written is False


def test_install_unit_writes_file(tmp_path: Path, monkeypatch) -> None:
    from llmctl.services import systemd

    unit = render_api_unit(load_settings(), user=True)
    target = tmp_path / "units" / unit.name
    monkeypatch.setattr(systemd, "unit_install_path", lambda _unit: target)
    report = systemd.install_unit(unit, dry_run=False, enable=False)
    assert report.written is True
    assert target.exists()
    assert "ExecStart=llmctl serve" in target.read_text()


def test_benchmark_rerun_endpoint(tmp_path: Path) -> None:
    client = TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'b.db'}"))
    model = client.post(
        "/models", json={"name": "o", "runtime": "ollama", "source": "o:latest"}
    )
    model_id = model.json()["id"]
    first = client.post(
        "/benchmarks/run",
        json={"model_id": model_id, "name": "smoke", "dry_run": True},
    )
    assert first.status_code == 201
    benchmark_id = first.json()["id"]

    rerun = client.post(f"/benchmarks/{benchmark_id}/rerun")
    assert rerun.status_code == 200
    assert rerun.json()["id"] != benchmark_id
    assert len(client.get("/benchmarks").json()) == 2

    missing = client.post("/benchmarks/does-not-exist/rerun")
    assert missing.status_code == 404


def test_benchmark_sweep_endpoint(tmp_path: Path) -> None:
    client = TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'sweep.db'}"))
    model = client.post(
        "/models", json={"name": "o", "runtime": "ollama", "source": "o:latest"}
    )
    model_id = model.json()["id"]
    response = client.post(
        "/benchmarks/sweep",
        json={"model_id": model_id, "name": "load", "sweep": [1, 2], "dry_run": True},
    )
    assert response.status_code == 201
    body = response.json()
    assert len(body) == 2
    assert {item["parameters"]["concurrency"] for item in body} == {1, 2}


def test_install_systemd_all(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from llmctl.cli import app as cli_app

    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    db_url = f"sqlite:///{tmp_path / 'all.db'}"
    monkeypatch.setenv("LLMCTL_DB_URL", db_url)
    init_db(db_url)
    with Session(get_engine(db_url)) as session:
        model = RegistryService(session).add_model(
            ModelCreate(name="o", runtime="ollama", source="o:latest")
        )
        SessionService(session).start(
            SessionStartRequest(
                model_id=model.id or "", runtime="ollama", dry_run=True, force=True
            )
        )

    result = CliRunner().invoke(cli_app, ["install-systemd", "--all", "--dry-run"])
    assert result.exit_code == 0
    assert "Persisting 1 active session" in result.stdout
    assert "DRY RUN" in result.stdout
    assert "llmctl-session-" in result.stdout


