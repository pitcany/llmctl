"""Tests for benchmark kinds (chat/completion/health/long_context).

These tests focus on the behaviour added when the benchmark service grew a
``kind`` dispatch, GPU sampler, error-persistence path, and filter-aware
``list_results``. They use ``httpx.MockTransport`` to stand in for the
runtime endpoint so the suite runs offline (no vLLM, no NVIDIA driver).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from sqlmodel import Session

from llmctl.db import BenchmarkKind, get_engine, init_db
from llmctl.schemas import BenchmarkRunRequest, ModelCreate
from llmctl.services.benchmarks import (
    DEFAULT_PROMPT,
    BenchmarkService,
)
from llmctl.services.registry import RegistryService

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

SSE_CHUNK = (
    'data: {"choices":[{"delta":{"content":"hello world"}}],'
    '"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture
def db(tmp_path, monkeypatch) -> Session:
    """Isolated DB session pointed at the repo's bundled configs."""
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    url = f"sqlite:///{tmp_path / 'bench.db'}"
    init_db(url)
    with Session(get_engine(url)) as session:
        yield session


def _ollama_model(db: Session, name: str = "demo") -> str:
    """Create a registered ollama model and return its id."""
    model = RegistryService(db).add_model(
        ModelCreate(name=name, runtime="ollama", source=f"{name}:latest")
    )
    return model.id or ""


def _client_factory(handler):
    """Build a transport-backed httpx.Client factory for injection."""
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


# -- default prompt ---------------------------------------------------------


def test_default_prompt_is_spec_mle_bayes() -> None:
    """The default prompt should match the spec's MLE/Bayesian text."""
    assert "maximum likelihood" in DEFAULT_PROMPT
    assert "Bayesian" in DEFAULT_PROMPT


# -- execution mode + workload validation (FIXPLAN 2026-11-03 §1A) ----------


def test_request_defaults_are_strict_live_single_pass() -> None:
    """Ordinary requests default to strict live, one pass, no sweep."""
    request = BenchmarkRunRequest(name="x")
    assert request.require_live is True
    assert request.dry_run is False
    assert request.repetitions == 1
    assert request.concurrency == 1
    assert request.sweep == []


def test_request_rejects_invalid_workload_numbers() -> None:
    """repetitions/concurrency/sweep are validated at the schema boundary."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BenchmarkRunRequest(repetitions=0)
    with pytest.raises(ValidationError):
        BenchmarkRunRequest(concurrency=0)
    with pytest.raises(ValidationError):
        BenchmarkRunRequest(sweep=[0, 2])


# -- kind dispatch ----------------------------------------------------------


def test_kind_chat_targets_chat_endpoint(db: Session) -> None:
    """Default kind=chat posts to /v1/chat/completions."""
    seen: list[str] = []
    model_id = _ollama_model(db)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))

    result = BenchmarkService(db, client_factory=_client_factory(handler)).run(
        BenchmarkRunRequest(
            name="chat",
            model_id=model_id,
            kind=BenchmarkKind.CHAT,
            prompts=["hi"],
        )
    )
    assert result.success is True
    assert result.kind == BenchmarkKind.CHAT
    assert seen and seen[0].endswith("/v1/chat/completions")
    assert result.backend == "ollama"
    assert result.parameters["endpoint_kind"] == "chat"


def test_kind_completion_targets_completion_endpoint(db: Session) -> None:
    """kind=completion posts to /v1/completions instead of /chat/completions."""
    seen: list[str] = []
    model_id = _ollama_model(db)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))

    result = BenchmarkService(db, client_factory=_client_factory(handler)).run(
        BenchmarkRunRequest(
            name="completion",
            model_id=model_id,
            kind=BenchmarkKind.COMPLETION,
            prompts=["hi"],
        )
    )
    assert result.success is True
    assert result.kind == BenchmarkKind.COMPLETION
    assert seen and seen[0].endswith("/v1/completions")


def test_kind_health_targets_models_endpoint(db: Session) -> None:
    """kind=health does a GET against /v1/models and records latency only."""
    seen: list[tuple[str, str]] = []
    model_id = _ollama_model(db)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url)))
        return httpx.Response(
            200, json={"object": "list", "data": [{"id": "demo:latest"}]}
        )

    result = BenchmarkService(db, client_factory=_client_factory(handler)).run(
        BenchmarkRunRequest(
            name="health",
            model_id=model_id,
            kind=BenchmarkKind.HEALTH,
        )
    )
    assert result.success is True
    assert result.kind == BenchmarkKind.HEALTH
    assert seen == [("GET", seen[0][1])] and seen[0][1].endswith("/v1/models")
    # health benchmarks have no completion tokens but still time the request.
    assert result.completion_tokens == 0
    assert result.latency_ms is not None and result.latency_ms >= 0
    # FIXPLAN §1B: a health check is not generation -- latency must not be
    # presented as TTFT.
    assert result.time_to_first_token_ms is None
    assert result.parameters["measurement"] == "health_check"
    assert result.samples[0]["measurement"] == "health_check"


def test_kind_long_context_builds_padded_prompt(db: Session) -> None:
    """kind=long_context pads the seed prompt to approximate context_length."""
    seen_bodies: list[bytes] = []
    model_id = _ollama_model(db)

    def handler(request: httpx.Request) -> httpx.Response:
        seen_bodies.append(request.read())
        return httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))

    result = BenchmarkService(db, client_factory=_client_factory(handler)).run(
        BenchmarkRunRequest(
            name="long",
            model_id=model_id,
            kind=BenchmarkKind.LONG_CONTEXT,
            context_length=2048,
        )
    )
    assert result.success is True
    assert result.kind == BenchmarkKind.LONG_CONTEXT
    assert result.context_length == 2048
    # ~4 chars per token rule of thumb, plus trailing question.
    assert seen_bodies and len(seen_bodies[0]) >= 2048 * 4
    # JSON-encodes \n -> \\n in the body, so check for the question text alone.
    assert b"what was the topic of the preamble" in seen_bodies[0]


# -- measurement provenance (FIXPLAN 2026-11-03 §1B) ------------------------


def test_token_counts_record_server_usage_source(db: Session) -> None:
    """Samples with server usage carry token_count_source=server_usage."""
    model_id = _ollama_model(db)
    result = BenchmarkService(db, client_factory=_client_factory(
        lambda _r: httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))
    )).run(BenchmarkRunRequest(name="src", model_id=model_id, prompts=["hi"]))
    assert result.success is True
    assert result.samples[0]["token_count_source"] == "server_usage"
    assert result.parameters["token_count_source"] == "server_usage"


def test_token_counts_estimate_source_and_no_invented_token(db: Session) -> None:
    """Without usage data the counts are labeled estimates, and an empty
    response never becomes one fabricated token."""
    model_id = _ollama_model(db)
    sse_no_usage = (
        'data: {"choices":[{"delta":{"content":""}}]}\n\n' "data: [DONE]\n\n"
    )
    result = BenchmarkService(db, client_factory=_client_factory(
        lambda _r: httpx.Response(200, content=sse_no_usage.encode("utf-8"))
    )).run(BenchmarkRunRequest(name="est", model_id=model_id, prompts=["hi"]))
    assert result.success is True
    sample = result.samples[0]
    assert sample["completion_tokens"] == 0
    assert sample["token_count_source"] == "estimate"
    assert result.parameters["token_count_source"] == "estimate"


def test_reasoning_chunks_set_first_token_but_not_ttft(db: Session) -> None:
    """TTFT is the first answer *content*; reasoning/tool activity counts
    only toward first_token_ms."""
    model_id = _ollama_model(db)
    sse = (
        'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    result = BenchmarkService(db, client_factory=_client_factory(
        lambda _r: httpx.Response(200, content=sse.encode("utf-8"))
    )).run(BenchmarkRunRequest(name="reason", model_id=model_id, prompts=["hi"]))
    assert result.success is True
    sample = result.samples[0]
    assert sample["response"] == "answer"
    assert sample["ttft_ms"] is not None
    assert sample["first_token_ms"] is not None
    assert sample["first_token_ms"] <= sample["ttft_ms"]
    assert sample["reasoning_chars"] == len("thinking...")


def test_partial_failure_keeps_successful_samples(db: Session) -> None:
    """§1B acceptance: one failed request in a batch fails the record but
    preserves (and reports) the successful samples."""
    model_id = _ollama_model(db)

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.read())
        content = body["messages"][0]["content"]
        if "bad" in content:
            return httpx.Response(500, content=b"boom")
        return httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))

    result = BenchmarkService(db, client_factory=_client_factory(handler)).run(
        BenchmarkRunRequest(
            name="mixed",
            model_id=model_id,
            prompts=["good1", "bad", "good2"],
            dry_run=False,
        )
    )
    assert result.success is False
    assert "1 of 3" in (result.error or "")
    assert result.parameters["request_count"] == 3
    assert result.parameters["failed_request_count"] == 1
    ok_samples = [s for s in result.samples if s.get("ok") is not False]
    assert len(ok_samples) == 2
    assert all(s["completion_tokens"] == 2 for s in ok_samples)
    failed = [s for s in result.samples if s.get("ok") is False]
    assert failed and "HTTPStatusError" in str(failed[0]["error"])
    # Successful tokens still aggregate (2 per good sample).
    assert result.completion_tokens == 4


def test_percentiles_include_sample_count_and_small_n_flag(db: Session) -> None:
    """§1B: latency/TTFT percentiles carry n and flag insufficient samples."""
    from llmctl.services.benchmarks import MIN_PERCENTILE_SAMPLES

    model_id = _ollama_model(db)
    result = BenchmarkService(db, client_factory=_client_factory(
        lambda _r: httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))
    )).run(
        BenchmarkRunRequest(
            name="pct",
            model_id=model_id,
            prompts=[f"p{i}" for i in range(MIN_PERCENTILE_SAMPLES)],
            dry_run=False,
        )
    )
    stats = result.parameters["percentiles"]["latency_ms"]
    assert stats["n"] == MIN_PERCENTILE_SAMPLES
    assert stats["insufficient_samples"] is False
    assert stats["p50"] is not None and stats["p90"] is not None

    small = BenchmarkService(db, client_factory=_client_factory(
        lambda _r: httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))
    )).run(BenchmarkRunRequest(name="pct2", model_id=model_id, prompts=["one"]))
    small_stats = small.parameters["percentiles"]["latency_ms"]
    assert small_stats["n"] == 1
    assert small_stats["insufficient_samples"] is True


def test_warmup_requests_excluded_from_measurements(db: Session) -> None:
    """§1B: warm-up requests hit the server but never the samples."""
    model_id = _ollama_model(db)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.read())
        seen.append(body["messages"][0]["content"])
        return httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))

    result = BenchmarkService(db, client_factory=_client_factory(handler)).run(
        BenchmarkRunRequest(
            name="warm", model_id=model_id, prompts=["q"], warmup=2, dry_run=False
        )
    )
    assert result.success is True
    assert len(seen) == 3  # 2 warm-ups + 1 measured
    assert len(result.samples) == 1
    assert result.parameters["warmup_requests"] == 2
    assert result.parameters["request_count"] == 1


# -- failure persistence ----------------------------------------------------


def test_live_failure_records_success_false(db: Session) -> None:
    """A 5xx response on a live run is stored, not silently mocked."""
    model_id = _ollama_model(db)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    result = BenchmarkService(db, client_factory=_client_factory(handler)).run(
        BenchmarkRunRequest(name="oops", model_id=model_id, prompts=["hi"])
    )
    assert result.success is False
    assert result.parameters["mode"] == "live"
    assert "HTTPStatusError" in (result.error or "")
    persisted = BenchmarkService(db).list_results()
    assert any(r.success is False for r in persisted)


# -- list_results filters ---------------------------------------------------


def test_list_results_filters_by_model_kind_and_limit(db: Session) -> None:
    """Filters compose with AND; limit returns newest first."""
    model_a = _ollama_model(db, name="alpha")
    model_b = _ollama_model(db, name="beta")
    service = BenchmarkService(db)
    # Persist three records via dry-run so we don't need any HTTP shape.
    service.run(BenchmarkRunRequest(name="a1", model_id=model_a, dry_run=True))
    service.run(BenchmarkRunRequest(name="b1", model_id=model_b, dry_run=True))
    service.run(
        BenchmarkRunRequest(
            name="b-health",
            model_id=model_b,
            kind=BenchmarkKind.HEALTH,
            dry_run=True,
        )
    )

    only_a = service.list_results(model_id=model_a)
    assert {r.model_id for r in only_a} == {model_a}

    health_only = service.list_results(kind=BenchmarkKind.HEALTH)
    assert all(r.kind == BenchmarkKind.HEALTH for r in health_only)

    latest_two = service.list_results(limit=2)
    assert len(latest_two) == 2
    # newest first
    assert latest_two[0].name == "b-health"


# -- persistence: new fields ------------------------------------------------


def test_run_persists_new_metadata_columns(db: Session) -> None:
    """kind/backend/context_length/peak_vram round-trip through SQLite."""
    model_id = _ollama_model(db)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=SSE_CHUNK.encode("utf-8"))

    service = BenchmarkService(db, client_factory=_client_factory(handler))
    result = service.run(
        BenchmarkRunRequest(
            name="chat-pers",
            model_id=model_id,
            kind=BenchmarkKind.CHAT,
            context_length=1024,
            prompts=["hi"],
        )
    )
    fresh = BenchmarkService(db).get_result(result.id or "")
    assert fresh is not None
    assert fresh.kind == BenchmarkKind.CHAT
    assert fresh.backend == "ollama"
    assert fresh.context_length == 1024
    # GPU columns are nullable on hosts without pynvml; the sampler should
    # still report a sample_count >= 0 in the snapshot dict.
    assert "sample_count" in fresh.gpu_snapshot
