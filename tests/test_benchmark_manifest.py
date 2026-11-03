"""Tests for benchmark experiment manifests, export, and rerun drift (§1C)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from llmctl.api.app import create_app
from llmctl.db import BenchmarkKind, BenchmarkRecord, get_engine, init_db
from llmctl.schemas import BenchmarkRunRequest, ModelCreate
from llmctl.services.benchmark_manifest import (
    MANIFEST_VERSION,
    build_manifest,
    manifest_drift,
    workload_hash,
)
from llmctl.services.benchmarks import BenchmarkService
from llmctl.services.registry import RegistryService

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

SSE_CHUNK = (
    'data: {"choices":[{"delta":{"content":"hi"}}],'
    '"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture
def db(tmp_path, monkeypatch) -> Session:
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    url = f"sqlite:///{tmp_path / 'manifest.db'}"
    init_db(url)
    with Session(get_engine(url)) as session:
        yield session


def _ollama_model(db: Session) -> str:
    return RegistryService(db).add_model(
        ModelCreate(name="demo", runtime="ollama", source="demo:latest")
    ).id or ""


def _live_run(db: Session, **kwargs) -> object:
    handler = lambda _r: httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))  # noqa: E731
    factory = lambda: httpx.Client(transport=httpx.MockTransport(handler))  # noqa: E731
    return BenchmarkService(db, client_factory=factory).run(
        BenchmarkRunRequest(
            name="m", model_id=_ollama_model(db), prompts=["q"], **kwargs
        )
    )


# -- manifest capture -------------------------------------------------------


def test_live_run_captures_manifest(db: Session) -> None:
    result = _live_run(db)
    assert result.success is True
    manifest = result.manifest
    assert manifest["version"] == MANIFEST_VERSION
    assert manifest["execution"] == "live"
    assert manifest["model"]["name"] == "demo"
    assert manifest["model"]["source"] == "demo:latest"
    assert manifest["model"]["runtime"] == "ollama"
    assert manifest["workload"]["hash"]
    assert manifest["workload"]["prompt_count"] == 1
    assert manifest["workload"]["total_requests"] == 1
    assert manifest["runner_host"]["telemetry_scope"] == "runner_host"
    # Loopback endpoint: serving host is the runner, with GPU identity.
    assert manifest["serving_host"]["location"] == "runner_host"


def test_synthetic_run_manifest_is_labeled(db: Session) -> None:
    result = BenchmarkService(db).run(
        BenchmarkRunRequest(name="s", model_id=_ollama_model(db), dry_run=True)
    )
    assert result.manifest["execution"] == "synthetic"
    assert result.manifest["served"]["endpoint_url"] == "unknown"
    assert result.manifest["model"]["name"] == "demo"


def test_unknown_fields_are_recorded_as_unknown() -> None:
    request = BenchmarkRunRequest(name="x", kind=BenchmarkKind.CHAT)
    manifest = build_manifest(
        request,
        model=None,
        session=None,
        backend=None,
        endpoint_url=None,
        base_prompts=["q"],
        gpu_snapshot={"gpus": []},
        execution="synthetic",
    )
    assert manifest["model"]["revision"] == "unknown"
    assert manifest["engine"]["version"] == "unknown"
    assert manifest["launch"]["gpu_ids"] == "unknown"
    assert manifest["served"]["backend"] == "unknown"


def test_remote_endpoint_not_attributed_runner_gpus() -> None:
    request = BenchmarkRunRequest(name="x")
    manifest = build_manifest(
        request,
        model=None,
        session=None,
        backend="vllm",
        endpoint_url="http://10.0.0.80:8003",
        base_prompts=["q"],
        gpu_snapshot={"gpus": [{"index": 0, "name": "RTX 5090"}]},
        execution="live",
    )
    serving = manifest["serving_host"]
    assert serving["location"] == "remote"
    assert serving["endpoint_host"] == "10.0.0.80"
    assert "not attributed" in serving["telemetry"]
    # Runner telemetry still describes the runner only.
    assert manifest["runner_host"]["gpu_count"] == 1


def test_workload_hash_stable_and_concurrency_free() -> None:
    a = BenchmarkRunRequest(name="x", prompts=["p"], concurrency=1)
    b = BenchmarkRunRequest(name="x", prompts=["p"], concurrency=8)
    assert workload_hash(a, ["p"]) == workload_hash(b, ["p"])
    c = BenchmarkRunRequest(name="x", prompts=["p"], parameters={"max_tokens": 128})
    assert workload_hash(a, ["p"]) != workload_hash(c, ["p"])


def test_manifest_excludes_secrets_and_env(db: Session) -> None:
    result = _live_run(db)
    blob = json.dumps(result.manifest)
    # No environment maps and no credential-ish keys anywhere in the tree.
    assert '"env"' not in blob
    for marker in ("api_key", "authorization", "password", "secret", "credential"):
        assert marker not in blob.lower(), marker


# -- legacy rows + drift ----------------------------------------------------


def test_legacy_record_reads_with_unknown_provenance(db: Session) -> None:
    legacy = BenchmarkRecord(
        name="old",
        kind="chat",
        parameters={"mode": "live"},
        manifest=None,
        success=True,
    )
    db.add(legacy)
    db.commit()
    result = BenchmarkService(db).get_result(legacy.id or "")
    assert result is not None
    assert result.manifest == {}
    payload = BenchmarkService(db).export(benchmark_id=legacy.id or "")
    assert payload and payload[0]["manifest"]["provenance"] == "unknown"


def test_rerun_links_source_and_detects_drift(db: Session) -> None:
    service = BenchmarkService(db)
    first = service.run(
        BenchmarkRunRequest(
            name="d", model_id=_ollama_model(db), prompts=["q"], dry_run=True
        )
    )
    # Simulate the environment changing after the original run: edit the
    # stored parameters so the rerun resolves different effective settings.
    record = db.get(BenchmarkRecord, first.id or "")
    assert record is not None
    params = dict(record.parameters)
    params["max_tokens"] = 99
    record.parameters = params
    db.add(record)
    db.commit()

    second = service.rerun(first.id or "")
    assert second is not None
    assert second.parameters["rerun_of"] == first.id
    drift = second.parameters.get("drift") or []
    assert any("max_tokens" in entry for entry in drift), drift
    assert second.id != first.id  # a new experiment with its own identity


def test_manifest_drift_ignores_telemetry_and_timestamps() -> None:
    base = build_manifest(
        BenchmarkRunRequest(name="x", prompts=["q"]),
        model=None,
        session=None,
        backend="vllm",
        endpoint_url="http://127.0.0.1:8003",
        base_prompts=["q"],
        gpu_snapshot={"gpus": [{"name": "A", "memory_total_mb": 1}]},
        execution="live",
    )
    noisy = json.loads(json.dumps(base))
    noisy["runner_host"]["gpus"][0]["memory_total_mb"] = 2
    noisy["runner_host"]["gpu_count"] = 9
    assert manifest_drift(base, noisy) == []

    changed = json.loads(json.dumps(base))
    changed["served"]["endpoint_url"] = "http://127.0.0.1:9999"
    drift = manifest_drift(base, changed)
    assert drift == ["served.endpoint_url: http://127.0.0.1:8003 -> http://127.0.0.1:9999"]


def test_migration_adds_manifest_column(tmp_path: Path) -> None:
    """A database created before manifests exist gains the column in place,
    and its old rows read back with unknown (not fabricated) provenance."""
    from sqlalchemy import text

    from llmctl.db import apply_migrations, get_engine

    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = get_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE benchmarks ("
                'id VARCHAR PRIMARY KEY, "name" VARCHAR NOT NULL, '
                "success BOOLEAN NOT NULL)"
            )
        )
        conn.execute(
            text("INSERT INTO benchmarks (id, name, success) VALUES ('b1', 'old', 1)")
        )
    apply_migrations(engine)
    with engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(benchmarks)"))}
        manifest_value = conn.execute(
            text("SELECT manifest FROM benchmarks WHERE id = 'b1'")
        ).scalar()
    assert "manifest" in columns
    assert manifest_value is None


# -- export surfaces --------------------------------------------------------


def test_export_round_trips_effective_settings(db: Session, tmp_path: Path) -> None:
    from llmctl.services.benchmark_manifest import write_export

    service = BenchmarkService(db)
    _live_run(db, parameters={"max_tokens": 64, "temperature": 0.3})
    payloads = service.export()
    assert len(payloads) == 1
    document = payloads[0]
    assert document["manifest"]["effective_parameters"]["max_tokens"] == 64
    assert document["manifest"]["effective_parameters"]["temperature"] == 0.3
    assert document["result"]["parameters"]["max_tokens"] == 64

    path = write_export(tmp_path / "out" / "export.json", payloads)
    reloaded = json.loads(path.read_text())
    assert reloaded["format"] == "llmctl-benchmark-export"
    assert reloaded["results"][0]["manifest"]["workload"]["hash"]


def test_api_export_endpoints(tmp_path: Path) -> None:
    client = TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'api.db'}"))
    model = client.post(
        "/models", json={"name": "o", "runtime": "ollama", "source": "o:latest"}
    )
    model_id = model.json()["id"]
    created = client.post(
        "/benchmarks/run",
        json={"model_id": model_id, "name": "smoke", "dry_run": True},
    )
    benchmark_id = created.json()["id"]

    one = client.get(f"/benchmarks/{benchmark_id}/export")
    assert one.status_code == 200
    assert one.json()["manifest"]["execution"] == "synthetic"

    many = client.get("/benchmarks/export")
    assert many.status_code == 200
    assert many.json()["format"] == "llmctl-benchmark-export"
    assert len(many.json()["results"]) == 1

    missing = client.get("/benchmarks/does-not-exist/export")
    assert missing.status_code == 404


def test_cli_export_writes_file(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from llmctl.cli import app as cli_app

    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    db_url = f"sqlite:///{tmp_path / 'cli-export.db'}"
    monkeypatch.setenv("LLMCTL_DB_URL", db_url)
    init_db(db_url)
    with Session(get_engine(db_url)) as session:
        model_id = RegistryService(session).add_model(
            ModelCreate(name="o", runtime="ollama", source="o:latest")
        ).id
        BenchmarkService(session).run(
            BenchmarkRunRequest(name="e", model_id=model_id, dry_run=True)
        )

    out = tmp_path / "export.json"
    result = CliRunner().invoke(cli_app, ["benchmarks", "--export", str(out)])
    assert result.exit_code == 0, result.output
    document = json.loads(out.read_text())
    assert document["format"] == "llmctl-benchmark-export"
    assert len(document["results"]) == 1
    assert document["results"][0]["manifest"]["execution"] == "synthetic"
