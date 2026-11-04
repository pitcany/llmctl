"""Benchmark service: real (streaming) execution with explicit synthetic mode.

Streams the request workload against a live OpenAI-compatible endpoint (vLLM,
llama.cpp, LM Studio and Ollama all expose ``/v1/chat/completions``) and
measures end-to-end latency, time-to-first-token (TTFT) and token throughput.

Execution-mode contract (FIXPLAN 2026-11-03 §1A): ordinary runs are
strict-live. When no endpoint resolves, the run is persisted as a *failure*
record -- never a fabricated synthetic measurement -- so benchmark history is
trustworthy evidence. Synthetic execution is the explicit ``dry_run=True``
path (``--synthetic`` on the CLI, 'mock' mode in the TUI): deterministic,
labeled ``mode=mock`` in every surface, and intended for hosts without a
runtime (e.g. CI). The historical silent fallback remains reachable only via
the deprecated ``require_live=False`` escape hatch.

Each prompt/response sample is persisted on the record so the TUI can show
history and offer a one-key re-run.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
from sqlmodel import Session as DBSession
from sqlmodel import select

from llmctl.config import Settings, load_settings
from llmctl.db import (
    BenchmarkKind,
    BenchmarkRecord,
    ModelRecord,
    SessionRecord,
    SessionStatus,
)
from llmctl.schemas import BenchmarkResult, BenchmarkRunRequest
from llmctl.services.benchmark_manifest import (
    build_manifest,
    export_payload,
    manifest_drift,
)
from llmctl.services.benchmark_parameters import DEFAULT_MAX_TOKENS, effective_parameters
from llmctl.telemetry.gpu import get_gpu_info, nvml_available
from llmctl.telemetry.gpu_sampler import GPUSampler, GPUSamplerSummary

#: Default chat prompt used when a request supplies none. Picked to exercise
#: reasoning + token throughput on a paragraph-length response.
DEFAULT_PROMPT = (
    "Explain the difference between maximum likelihood and Bayesian inference "
    "in one paragraph."
)
#: Long-context smoke test prompt seed; we repeat this to fill the requested
#: context_length so the model has to ingest a real load.
LONG_CONTEXT_SEED = (
    "The following is a stress test of long-context handling. Please answer "
    "the question at the end after reading the preamble.\n\n"
)
#: Marker question appended to the long-context prompt.
LONG_CONTEXT_QUESTION = (
    "\n\nQUESTION: In one sentence, what was the topic of the preamble?"
)
#: Assumed throughput (tokens/sec) for the synthetic mock benchmark.
MOCK_TOKENS_PER_SECOND = 50.0
#: HTTP timeout (seconds) for live benchmark calls.
REQUEST_TIMEOUT = 60.0
#: HTTP timeout (seconds) for health checks (short — they should be snappy).
HEALTH_TIMEOUT = 5.0
#: Maximum characters of generated text stored per sample.
SAMPLE_CHAR_LIMIT = 500

#: Factory used to build the (sync) HTTP client; injectable for tests.
ClientFactory = Callable[[], httpx.Client]

#: Sentinel returned by :func:`_parse_sse_line` for the ``[DONE]`` event.
_DONE = object()


@dataclass
class _Target:
    """A resolved live benchmark endpoint."""

    base_url: str
    model_name: str
    backend: str | None = None
    #: The session row the endpoint was resolved through, if any (used to
    #: capture launch configuration for the run manifest).
    session: SessionRecord | None = None

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"

    @property
    def completion_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/completions"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/v1/models"


def record_to_benchmark(record: BenchmarkRecord) -> BenchmarkResult:
    """Convert benchmark DB record to schema."""
    kind_value: BenchmarkKind | None
    if record.kind is None:
        kind_value = None
    else:
        try:
            kind_value = BenchmarkKind(record.kind)
        except ValueError:
            kind_value = None
    return BenchmarkResult(
        id=record.id,
        model_id=record.model_id,
        session_id=record.session_id,
        profile_id=record.profile_id,
        name=record.name,
        kind=kind_value,
        backend=record.backend,
        context_length=record.context_length,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        latency_ms=record.latency_ms,
        tokens_per_second=record.tokens_per_second,
        time_to_first_token_ms=record.ttft_ms,
        peak_vram_mb=record.peak_vram_mb,
        avg_gpu_util_pct=record.avg_gpu_util_pct,
        max_gpu_util_pct=record.max_gpu_util_pct,
        gpu_snapshot=record.gpu_snapshot,
        parameters=record.parameters,
        samples=record.samples,
        manifest=record.manifest or {},
        success=record.success,
        error=record.error,
        created_at=record.created_at,
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~whitespace words) when a server omits usage data.

    An empty response estimates to zero -- never a fabricated single token.
    Every count produced here is labeled ``token_count_source="estimate"``
    on the sample so downstream surfaces can distinguish it from server
    usage counts.
    """
    return len(text.split())


def _parse_sse_line(line: str) -> object | None:
    """Parse a single SSE ``data:`` line; return a dict, ``_DONE`` or ``None``."""
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return _DONE
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _extract_delta(payload: dict[str, object]) -> str:
    """Pull incremental *content* text out of an OpenAI-style streaming chunk."""
    try:
        choice = payload["choices"][0]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(choice, dict):
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            return str(delta["content"])
        if choice.get("text"):
            return str(choice["text"])
    return ""


def _extract_reasoning(payload: dict[str, object]) -> str:
    """Pull incremental reasoning text (e.g. ``reasoning_content``) from a chunk.

    vLLM/OpenRouter-style servers stream model reasoning separately from the
    final answer. Reasoning is generated activity (it counts for
    ``first_token_ms``) but is *not* answer content, so it does not start the
    TTFT clock -- see :meth:`BenchmarkService._stream_one`.
    """
    try:
        choice = payload["choices"][0]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(choice, dict):
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("reasoning_content"):
            return str(delta["reasoning_content"])
    return ""


def _has_tool_delta(payload: dict[str, object]) -> bool:
    """Return True when a streaming chunk carries tool-call deltas.

    Like reasoning chunks, tool activity is generated output (counted for
    ``first_token_ms``) but not answer content, so it does not define TTFT.
    """
    try:
        choice = payload["choices"][0]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return False
    if not isinstance(choice, dict):
        return False
    delta = choice.get("delta")
    return isinstance(delta, dict) and bool(delta.get("tool_calls"))


#: Sample count below which a percentile is flagged as insufficient: with
#: very few samples the p90/p99 spread says more about scheduling noise than
#: about the endpoint.
MIN_PERCENTILE_SAMPLES = 5


def _percentiles(values: list[float]) -> dict[str, object] | None:
    """Nearest-rank p50/p90/p99 with the sample count and a small-n flag.

    ``insufficient_samples`` marks estimates computed from fewer than
    :data:`MIN_PERCENTILE_SAMPLES` observations so surfaces can say
    "p90 ≈ X ms (n=2, too few samples)" instead of implying precision.
    """
    n = len(values)
    if n == 0:
        return None
    ordered = sorted(values)

    def _rank(pct: float) -> float:
        return ordered[max(0, min(n - 1, math.ceil(pct * n) - 1))]

    return {
        "n": n,
        "p50": round(float(_rank(0.50)), 2),
        "p90": round(float(_rank(0.90)), 2),
        "p99": round(float(_rank(0.99)), 2),
        "insufficient_samples": n < MIN_PERCENTILE_SAMPLES,
    }


def _aggregate_token_source(samples: list[dict[str, object]]) -> str | None:
    """Summarize per-sample token provenance for the record.

    ``server_usage`` when every counted token came from the server's usage
    block; ``estimate`` when every count was estimated; ``mixed`` when both
    are present. Returns ``None`` when there is nothing to summarize.
    """
    sources = {
        str(sample.get("token_count_source"))
        for sample in samples
        if sample.get("token_count_source")
    }
    sources.discard("none")
    if not sources:
        return None
    if len(sources) == 1:
        return sources.pop()
    return "mixed"


def _gpu_snapshot() -> dict[str, object]:
    """Capture a compact GPU snapshot for the benchmark record."""
    gpus = get_gpu_info()
    return {
        "gpu_count": len(gpus),
        "nvml_available": nvml_available(),
        "gpus": [
            {
                "index": gpu.index,
                "name": gpu.name,
                "memory_used_mb": gpu.memory_used_mb,
                "memory_total_mb": gpu.memory_total_mb,
            }
            for gpu in gpus[:4]
        ],
    }


class BenchmarkService:
    """Service interface for benchmark orchestration and history."""

    def __init__(
        self,
        db: DBSession,
        settings: Settings | None = None,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.db = db
        self.settings = settings or load_settings()
        self._client_factory = client_factory

    def list_results(
        self,
        *,
        model_id: str | None = None,
        session_id: str | None = None,
        kind: BenchmarkKind | str | None = None,
        limit: int | None = None,
    ) -> list[BenchmarkResult]:
        """List recorded benchmark results, newest first when ``limit`` is given.

        Filters compose with AND semantics. ``limit`` is applied after sorting
        by ``created_at DESC`` so callers can ask for the "latest N for model X".
        """
        statement = select(BenchmarkRecord)
        if model_id is not None:
            statement = statement.where(BenchmarkRecord.model_id == model_id)
        if session_id is not None:
            statement = statement.where(BenchmarkRecord.session_id == session_id)
        if kind is not None:
            kind_str = kind.value if isinstance(kind, BenchmarkKind) else str(kind)
            statement = statement.where(BenchmarkRecord.kind == kind_str)
        if limit is not None:
            statement = statement.order_by(BenchmarkRecord.created_at.desc()).limit(limit)
        records = self.db.exec(statement).all()
        return [record_to_benchmark(record) for record in records]

    def get_result(self, benchmark_id: str) -> BenchmarkResult | None:
        """Return a single recorded benchmark by id, or ``None`` if missing."""
        record = self.db.get(BenchmarkRecord, benchmark_id)
        return record_to_benchmark(record) if record is not None else None

    def delete(self, benchmark_id: str) -> bool:
        """Hard-delete a recorded benchmark row; returns ``False`` if missing.

        Benchmarks are immutable historical run records — there's no soft-delete
        state to preserve, and the operator's intent when pruning history is to
        actually drop the row (e.g. a botched dry-run cluttering the screen).
        """
        record = self.db.get(BenchmarkRecord, benchmark_id)
        if record is None:
            return False
        self.db.delete(record)
        self.db.commit()
        return True

    def export(
        self,
        *,
        benchmark_id: str | None = None,
        model_id: str | None = None,
        limit: int | None = None,
        session_id: str | None = None,
        kind: BenchmarkKind | str | None = None,
    ) -> list[dict[str, object]]:
        """Return portable export payloads (result + manifest) for records.

        A single ``benchmark_id`` exports that record; otherwise the filtered
        history (same filters as :meth:`list_results`). Legacy rows without a
        manifest export with ``manifest.provenance == "unknown"``.
        """
        if benchmark_id is not None:
            result = self.get_result(benchmark_id)
            return [export_payload(result)] if result is not None else []
        results = self.list_results(
            model_id=model_id, session_id=session_id, kind=kind, limit=limit,
        )
        return [export_payload(result) for result in results]

    def run(self, request: BenchmarkRunRequest) -> BenchmarkResult:
        """Execute a benchmark and persist its result.

        Streams a real run against a live runtime endpoint. On an explicit
        synthetic request (``dry_run=True``) it records a deterministic mock
        result with ``parameters.mode == "mock"``. Live runs are strict: a
        missing endpoint records a failure (``require_live=False`` opts back
        into the deprecated mock fallback), and live runs that fail mid-flight
        are persisted with ``success=False`` and the error message so failures
        are inspectable in the history.
        """
        base_prompts = self._resolve_prompts(request)
        workload = self._expand_workload(base_prompts, request.repetitions)
        params = effective_parameters(request.parameters)
        params.setdefault("concurrency", max(1, request.concurrency))
        params.setdefault("warmup", max(0, request.warmup))
        params.update(self._workload_meta(base_prompts, request.repetitions))
        snapshot = _gpu_snapshot()
        target = None if request.dry_run else self._resolve_target(request)
        with GPUSampler() as sampler:
            metrics = self._measure(request, workload, params, target)
        gpu_summary = sampler.summary()
        mode = str((metrics.get("parameters") or {}).get("mode", "live"))
        manifest = build_manifest(
            request,
            model=self.db.get(ModelRecord, request.model_id)
            if request.model_id
            else None,
            session=target.session if target is not None else None,
            backend=(target.backend if target else None) or metrics.get("backend"),
            endpoint_url=target.base_url if target is not None else None,
            base_prompts=base_prompts,
            gpu_snapshot=snapshot,
            execution="synthetic" if mode == "mock" else "live",
        )
        record = BenchmarkRecord(
            model_id=request.model_id,
            session_id=request.session_id,
            profile_id=request.profile_id,
            name=request.name,
            kind=request.kind.value,
            backend=metrics.get("backend"),
            context_length=request.context_length,
            prompt_tokens=metrics["prompt_tokens"],
            completion_tokens=metrics["completion_tokens"],
            total_tokens=metrics["total_tokens"],
            latency_ms=metrics["latency_ms"],
            tokens_per_second=metrics["tokens_per_second"],
            ttft_ms=metrics["ttft_ms"],
            peak_vram_mb=gpu_summary.peak_vram_mb,
            avg_gpu_util_pct=gpu_summary.avg_gpu_util_pct,
            max_gpu_util_pct=gpu_summary.max_gpu_util_pct,
            gpu_snapshot=self._merge_snapshot(snapshot, gpu_summary),
            parameters=metrics["parameters"],
            samples=metrics["samples"],
            manifest=manifest,
            success=metrics["success"],
            error=metrics["error"],
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record_to_benchmark(record)

    def rerun(self, benchmark_id: str) -> BenchmarkResult | None:
        """Re-execute a stored benchmark, reusing its prompts and parameters.

        Reuses the stored *workload*: the base prompt list and repetition
        count recorded on the source run, in their original order. A record
        originally produced in synthetic mode re-runs synthetically (it was
        an explicitly-mock experiment); anything else re-runs strict-live so
        a missing endpoint surfaces as a failure instead of a silent mock.
        """
        record = self.db.get(BenchmarkRecord, benchmark_id)
        if record is None:
            return None
        stored: dict[str, object] = dict(record.parameters or {})
        workload_prompts = [
            str(sample["prompt"])
            for sample in (record.samples or [])
            if sample.get("prompt")
        ]
        repetitions = max(1, int(stored.get("repetitions", 1) or 1))
        prompt_count = int(stored.get("prompt_count", 0) or 0)
        # Samples are stored in workload order (base prompts repeated
        # ``repetitions`` times), so the first ``prompt_count`` entries are
        # the base list. Legacy rows without workload metadata replay their
        # stored sample prompts once.
        if "base_prompts" in stored:
            base_prompts = list(stored["base_prompts"])
        elif repetitions > 1 and prompt_count >= 1 and len(workload_prompts) >= prompt_count:
            base_prompts = workload_prompts[:prompt_count]
        else:
            base_prompts = workload_prompts
            repetitions = 1
        params = {
            key: stored[key]
            for key in ("max_tokens", "temperature", "concurrency", "warmup")
            if key in stored
        }
        kind = (
            BenchmarkKind(record.kind)
            if record.kind in {k.value for k in BenchmarkKind}
            else BenchmarkKind.CHAT
        )
        request = BenchmarkRunRequest(
            name=record.name,
            model_id=record.model_id,
            session_id=record.session_id,
            profile_id=record.profile_id,
            kind=kind,
            context_length=record.context_length,
            prompts=base_prompts,
            parameters=params,
            concurrency=int(stored.get("concurrency", 1) or 1),
            repetitions=repetitions,
            warmup=max(0, int(stored.get("warmup", 0) or 0)),
            dry_run=str(stored.get("mode", "live")) == "mock",
            require_live=True,
        )
        result = self.run(request)
        if result is None or result.id is None:
            return result
        # The re-run is a new experiment with its own identity, linked to
        # its source. Configuration drift between the two manifests is
        # recorded on the new row so "same benchmark, different number" is
        # explainable instead of mysterious.
        new_record = self.db.get(BenchmarkRecord, result.id)
        if new_record is None:
            return result
        drift = manifest_drift(record.manifest, result.manifest)
        updated = dict(new_record.parameters or {})
        updated["rerun_of"] = benchmark_id
        if drift:
            updated["drift"] = drift
        new_record.parameters = updated
        self.db.add(new_record)
        self.db.commit()
        self.db.refresh(new_record)
        return record_to_benchmark(new_record)

    def run_sweep(self, request: BenchmarkRunRequest) -> list[BenchmarkResult]:
        """Run the benchmark at several concurrency levels (a load sweep).

        Each level is persisted as its own record named ``"{name} (c=N)"`` with
        ``parameters.concurrency == N`` so throughput-under-load can be compared
        directly in the Benchmarks screen. The total workload (prompts x
        repetitions) stays fixed across levels -- only the concurrency ceiling
        changes, so a level whose workload cannot fill its concurrency records
        the shortfall instead of silently shrinking the experiment.
        """
        levels = request.sweep or [max(1, request.concurrency)]
        ordered = sorted({max(1, int(level)) for level in levels})
        results: list[BenchmarkResult] = []
        for level in ordered:
            params = dict(request.parameters)
            params["concurrency"] = level
            sub = request.model_copy(
                update={
                    "name": f"{request.name} (c={level})",
                    "parameters": params,
                    "concurrency": level,
                    "sweep": [],
                }
            )
            results.append(self.run(sub))
        return results

    # -- measurement --------------------------------------------------------

    def _measure(
        self,
        request: BenchmarkRunRequest,
        prompts: list[str],
        params: dict[str, object],
        target: _Target | None,
    ) -> dict[str, object]:
        """Return benchmark metrics for the requested ``kind``.

        ``target`` is the pre-resolved endpoint (``None`` means dry-run, or
        strict-live with no endpoint). Behaviour:
          * Explicit ``dry_run`` -> deterministic mock result for the kind.
          * No reachable endpoint -> failure record (strict live is the
            default). Only an explicit ``require_live=False`` restores the
            deprecated mock fallback with reason
            ``"no reachable runtime endpoint"``.
          * Endpoint reachable but the call fails -> *failure record*:
            ``success=False``, the exception in ``error``. Mock fallback is
            NOT used for live failures (we want failures to be inspectable).
        """
        if request.dry_run:
            return self._mock(request, prompts, params, "dry-run requested")
        if target is None:
            if request.require_live:
                return self._no_endpoint_failure(request, prompts, params)
            return self._mock(
                request, prompts, params, "no reachable runtime endpoint"
            )
        try:
            return self._run_kind(request, target, prompts, params)
        except Exception as exc:  # noqa: BLE001 - persist as failure record
            return self._failure(request, target, prompts, params, exc)

    def _run_kind(
        self,
        request: BenchmarkRunRequest,
        target: _Target,
        prompts: list[str],
        params: dict[str, object],
    ) -> dict[str, object]:
        """Dispatch the live runner appropriate to ``request.kind``."""
        if request.kind == BenchmarkKind.HEALTH:
            return self._run_health(target, params)
        if request.kind == BenchmarkKind.COMPLETION:
            return self._run_live(target, prompts, params, endpoint="completion")
        if request.kind == BenchmarkKind.LONG_CONTEXT:
            ctx = [self._build_long_context_prompt(request)]
            return self._run_live(
                target, self._expand_workload(ctx, request.repetitions), params, endpoint="chat"
            )
        return self._run_live(target, prompts, params, endpoint="chat")

    def _resolve_target(self, request: BenchmarkRunRequest) -> _Target | None:
        """Resolve the live OpenAI-compatible endpoint for the request, if any."""
        endpoint, model_name, backend, session = self._resolve_endpoint(request)
        if not endpoint:
            return None
        return _Target(
            base_url=endpoint,
            model_name=model_name or "local-model",
            backend=backend,
            session=session,
        )

    def _resolve_endpoint(
        self, request: BenchmarkRunRequest
    ) -> tuple[str | None, str | None, str | None, SessionRecord | None]:
        """Find a serving endpoint via an explicit session, a running session,
        or the runtime's default server endpoint.

        Returns ``(url, model, backend, session)``; the session (when the
        endpoint was resolved through one) carries launch configuration for
        the run manifest.
        """
        if request.session_id:
            record = self.db.get(SessionRecord, request.session_id)
            if record and record.endpoint_url:
                backend = record.runtime.value if record.runtime else None
                return record.endpoint_url, self._model_name(record.model_id), backend, record
        if request.model_id:
            model = self.db.get(ModelRecord, request.model_id)
            if model:
                name = model.source or model.name
                backend = model.runtime.value
                # Match owned sessions first (started by llmctl, model_id set).
                running = self.db.exec(
                    select(SessionRecord).where(
                        SessionRecord.model_id == model.id,
                        SessionRecord.status == SessionStatus.RUNNING,
                    )
                ).first()
                # Adopted sessions (e.g. vllm-tp.service tracked via
                # `llmctl adopt`) carry served_name but not model_id, so fall
                # back to matching the served name against model.source / .name.
                if running is None and name:
                    running = self.db.exec(
                        select(SessionRecord).where(
                            SessionRecord.served_name == name,
                            SessionRecord.status == SessionStatus.RUNNING,
                        )
                    ).first()
                if running is None and model.name and model.name != name:
                    running = self.db.exec(
                        select(SessionRecord).where(
                            SessionRecord.served_name == model.name,
                            SessionRecord.status == SessionStatus.RUNNING,
                        )
                    ).first()
                if running and running.endpoint_url:
                    return running.endpoint_url, name, backend, running
                config = self.settings.runtime_config(model.runtime.value)
                if config.endpoint:
                    return config.endpoint, name, backend, None
        return None, None, None, None

    def _model_name(self, model_id: str | None) -> str | None:
        """Return a model's served name (source preferred) by id."""
        if not model_id:
            return None
        model = self.db.get(ModelRecord, model_id)
        return (model.source or model.name) if model else None

    def _resolve_prompts(self, request: BenchmarkRunRequest) -> list[str]:
        """Pick prompts for a run based on kind and request input."""
        if request.kind == BenchmarkKind.HEALTH:
            return []
        if request.prompts:
            return list(request.prompts)
        return [DEFAULT_PROMPT]

    @staticmethod
    def _expand_workload(prompts: list[str], repetitions: int) -> list[str]:
        """Expand base prompts into the full request workload.

        Repetition is cyclic (``a, b, a, b``), not blocked (``a, a, b, b``),
        so concurrent workers spread across distinct prompts instead of
        hammering the same one. The policy is recorded on the result via
        :meth:`_workload_meta`; prompts are never duplicated implicitly.
        """
        if repetitions <= 1:
            return list(prompts)
        return [prompt for _ in range(repetitions) for prompt in prompts]

    @staticmethod
    def _workload_meta(base_prompts: list[str], repetitions: int) -> dict[str, object]:
        """Return the workload-size metadata recorded on every benchmark."""
        return {
            "base_prompts": list(base_prompts),
            "prompt_count": len(base_prompts),
            "repetitions": repetitions,
            "total_requests": len(base_prompts) * repetitions,
            "repetition_policy": "cyclic" if repetitions > 1 else "none",
        }

    @staticmethod
    def _build_long_context_prompt(request: BenchmarkRunRequest) -> str:
        """Pad ``LONG_CONTEXT_SEED`` to roughly ``context_length`` tokens.

        We approximate 1 token == 4 characters (the standard OpenAI rule of
        thumb) when the user gives a token budget. Defaults to ~8K tokens
        when no ``context_length`` is provided.
        """
        target_tokens = request.context_length or 8192
        target_chars = max(target_tokens * 4, len(LONG_CONTEXT_SEED) + 1)
        body = LONG_CONTEXT_SEED
        while len(body) < target_chars:
            body += LONG_CONTEXT_SEED
        return body[:target_chars] + LONG_CONTEXT_QUESTION

    @staticmethod
    def _merge_snapshot(
        snapshot: dict[str, object], summary: GPUSamplerSummary
    ) -> dict[str, object]:
        """Add sampler aggregates onto the point-in-time GPU snapshot."""
        merged = dict(snapshot)
        merged["sample_count"] = summary.sample_count
        merged["peak_vram_mb"] = summary.peak_vram_mb
        merged["avg_gpu_util_pct"] = summary.avg_gpu_util_pct
        merged["max_gpu_util_pct"] = summary.max_gpu_util_pct
        return merged

    def _run_health(
        self, target: _Target, params: dict[str, object]
    ) -> dict[str, object]:
        """Time a GET against ``/v1/models``.

        This is an endpoint health check, not generation: the measured
        number is request latency only. ``ttft_ms`` stays ``None`` (no
        tokens were generated) and the sample is marked
        ``measurement="health_check"`` so surfaces never present health
        latency as generation TTFT.
        """
        client = (
            self._client_factory()
            if self._client_factory is not None
            else httpx.Client(timeout=HEALTH_TIMEOUT)
        )
        start = time.perf_counter()
        body_text = ""
        status_code: int | None = None
        with client:
            response = client.get(target.models_url)
            status_code = response.status_code
            response.raise_for_status()
            body_text = response.text[:SAMPLE_CHAR_LIMIT]
        elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": elapsed_ms,
            "tokens_per_second": None,
            "ttft_ms": None,
            "samples": [
                {
                    "prompt": "GET /v1/models",
                    "response": body_text,
                    "status_code": status_code,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "token_count_source": "none",
                    "measurement": "health_check",
                    "latency_ms": elapsed_ms,
                    "ttft_ms": None,
                }
            ],
            "parameters": {
                **params,
                "mode": "live",
                "endpoint": target.models_url,
                "kind": BenchmarkKind.HEALTH.value,
                "measurement": "health_check",
            },
            "backend": target.backend,
            "success": True,
            "error": None,
        }

    def _run_live(
        self,
        target: _Target,
        prompts: list[str],
        params: dict[str, object],
        *,
        endpoint: str = "chat",
    ) -> dict[str, object]:
        """Stream the request workload against a live endpoint; aggregate.

        ``concurrency`` is the maximum number of simultaneous in-flight
        requests. The pool never spins up more workers than there are
        requests, and the record stores both the requested ceiling
        (``requested_concurrency``) and the peak actually observed
        (``achieved_concurrency``); an underfilled workload is flagged in
        ``parameters.warnings`` instead of being reported as if it reached
        the ceiling. Throughput is computed from wall-clock time; TTFT is
        averaged across requests. The ``endpoint`` selector picks between
        ``"chat"`` (``/v1/chat/completions``) and ``"completion"``
        (``/v1/completions``) -- the same streaming runner handles both,
        since both speak SSE with the same delta shape.
        """
        params = effective_parameters(params)
        max_tokens = int(params["max_tokens"])
        temperature = float(params["temperature"])
        requested = max(1, int(params.get("concurrency", 1)))
        workers = max(1, min(requested, len(prompts))) if prompts else 1
        client = (
            self._client_factory()
            if self._client_factory is not None
            else httpx.Client(timeout=REQUEST_TIMEOUT)
        )
        endpoint_url = (
            target.completion_url if endpoint == "completion" else target.chat_url
        )
        inflight_lock = threading.Lock()
        inflight = {"now": 0, "peak": 0}

        def _tracked(prompt: str) -> dict[str, object]:
            with inflight_lock:
                inflight["now"] += 1
                inflight["peak"] = max(inflight["peak"], inflight["now"])
            try:
                return self._stream_one(
                    client, target, prompt, max_tokens, temperature, endpoint
                )
            except Exception as exc:  # noqa: BLE001 - per-request outcome
                return {
                    "prompt": prompt,
                    "response": "",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "token_count_source": "none",
                    "latency_ms": None,
                    "ttft_ms": None,
                    "first_token_ms": None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                with inflight_lock:
                    inflight["now"] -= 1

        warmup = max(0, int(params.get("warmup", 0) or 0))
        warmup_errors: list[str] = []
        with client:
            if warmup and prompts:
                for i in range(warmup):
                    warm_prompt = prompts[i % len(prompts)]
                    try:
                        self._stream_one(
                            client, target, warm_prompt, max_tokens, temperature, endpoint
                        )
                    except Exception as exc:  # noqa: BLE001 - recorded as warning
                        warmup_errors.append(f"{type(exc).__name__}: {exc}")
            wall_start = time.perf_counter()
            samples: list[dict[str, object]] = []
            if workers == 1:
                for prompt in prompts:
                    samples.append(_tracked(prompt))
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_tracked, prompt) for prompt in prompts]
                    samples = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - wall_start
        peak = inflight["peak"]
        warnings: list[str] = []
        if prompts and len(prompts) < requested:
            warnings.append(
                f"underfilled workload: {len(prompts)} request(s) cannot reach "
                f"requested concurrency {requested}; achieved {peak}"
            )
        elif prompts and peak < requested:
            warnings.append(
                f"requested concurrency {requested} was not observed; achieved {peak}"
            )
        for warmup_error in warmup_errors[:3]:
            warnings.append(f"warm-up request failed: {warmup_error}")
        failed_samples = [s for s in samples if s.get("ok") is False]
        ok_samples = [s for s in samples if s.get("ok") is not False]
        total_prompt = sum(int(sample["prompt_tokens"]) for sample in ok_samples)
        total_completion = sum(int(sample["completion_tokens"]) for sample in ok_samples)
        ttft_values = [
            float(sample["ttft_ms"])
            for sample in ok_samples
            if sample["ttft_ms"] is not None
        ]
        tps = round(total_completion / wall_seconds, 2) if wall_seconds > 0 else None
        avg_ttft = round(sum(ttft_values) / len(ttft_values), 2) if ttft_values else None
        success = not failed_samples
        error: str | None = None
        if failed_samples:
            error = (
                f"{len(failed_samples)} of {len(samples)} request(s) failed; "
                f"first: {failed_samples[0].get('error')}"
            )
        token_source = _aggregate_token_source(ok_samples)
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "latency_ms": round(wall_seconds * 1000.0, 2),
            "tokens_per_second": tps,
            "ttft_ms": avg_ttft,
            "samples": samples,
            "parameters": {
                **params,
                "mode": "live",
                "endpoint": endpoint_url,
                "endpoint_kind": endpoint,
                "requested_concurrency": requested,
                "concurrency": requested,
                "achieved_concurrency": peak,
                "request_count": len(samples),
                "failed_request_count": len(failed_samples),
                "warmup_requests": warmup,
                "warnings": warnings,
                "percentiles": {
                    "latency_ms": _percentiles(
                        [s["latency_ms"] for s in ok_samples if s.get("latency_ms") is not None]
                    ),
                    "ttft_ms": _percentiles(ttft_values),
                },
                **({"token_count_source": token_source} if token_source else {}),
            },
            "backend": target.backend,
            "success": success,
            "error": error,
        }

    def _stream_one(
        self,
        client: httpx.Client,
        target: _Target,
        prompt: str,
        max_tokens: int,
        temperature: float,
        endpoint: str = "chat",
    ) -> dict[str, object]:
        """Stream a single prompt, capturing TTFT and the response text.

        ``endpoint`` is ``"chat"`` or ``"completion"``; the body shape differs
        (``messages=[...]`` vs ``prompt=...``) but the SSE chunks come back
        in the same delta-with-content format both ways.
        """
        if endpoint == "completion":
            url = target.completion_url
            body: dict[str, object] = {
                "model": target.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        else:
            url = target.chat_url
            body = {
                "model": target.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        start = time.perf_counter()
        ttft: float | None = None
        first_token: float | None = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, object] = {}
        with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                payload = _parse_sse_line(line)
                if payload is None:
                    continue
                if payload is _DONE:
                    break
                chunk_usage = payload.get("usage")  # type: ignore[union-attr]
                if chunk_usage:
                    usage = chunk_usage  # type: ignore[assignment]
                delta = _extract_delta(payload)  # type: ignore[arg-type]
                reasoning = _extract_reasoning(payload)  # type: ignore[arg-type]
                generated = bool(delta or reasoning or _has_tool_delta(payload))  # type: ignore[arg-type]
                if generated and first_token is None:
                    # First *any* generated output: content, reasoning or a
                    # tool call. TTFT below stays content-only.
                    first_token = time.perf_counter() - start
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    content_parts.append(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
        elapsed = time.perf_counter() - start
        text = "".join(content_parts)
        completion = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        have_completion = completion is not None
        have_prompt = prompt_tokens is not None
        completion = int(completion) if have_completion else _estimate_tokens(text)
        prompt_tokens = (
            int(prompt_tokens) if have_prompt else _estimate_tokens(prompt)
        )
        if have_completion and have_prompt:
            token_source = "server_usage"
        elif have_completion or have_prompt:
            token_source = "mixed"
        else:
            token_source = "estimate"
        return {
            "prompt": prompt,
            "response": text[:SAMPLE_CHAR_LIMIT],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "token_count_source": token_source,
            "latency_ms": round(elapsed * 1000.0, 2),
            "ttft_ms": round(ttft * 1000.0, 2) if ttft is not None else None,
            "first_token_ms": (
                round(first_token * 1000.0, 2) if first_token is not None else None
            ),
            "reasoning_chars": len("".join(reasoning_parts)),
        }

    def _failure(
        self,
        request: BenchmarkRunRequest,
        target: _Target,
        prompts: list[str],
        params: dict[str, object],
        exc: BaseException,
    ) -> dict[str, object]:
        """Persist a live-run failure as ``success=False`` with the error msg.

        Unlike the mock fallback (used only on dry-run or no-endpoint cases),
        live failures are recorded as-is so the operator can see what broke
        when reviewing the benchmark history.
        """
        endpoint_url = (
            target.chat_url
            if request.kind != BenchmarkKind.HEALTH
            else target.models_url
        )
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": None,
            "tokens_per_second": None,
            "ttft_ms": None,
            "samples": [
                {
                    "prompt": prompt,
                    "response": "",
                    "prompt_tokens": _estimate_tokens(prompt),
                    "completion_tokens": 0,
                    "token_count_source": "none",
                    "latency_ms": None,
                    "ttft_ms": None,
                    "ok": False,
                }
                for prompt in prompts
            ],
            "parameters": {
                **params,
                "mode": "live",
                "endpoint": endpoint_url,
                "kind": request.kind.value,
                "concurrency": max(1, int(params.get("concurrency", 1))),
            },
            "backend": target.backend,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    def _no_endpoint_failure(
        self,
        request: BenchmarkRunRequest,
        prompts: list[str],
        params: dict[str, object],
    ) -> dict[str, object]:
        """Persist a strict-live miss as ``success=False`` with a hint.

        Used when the caller passes ``require_live=True`` (e.g. the TUI when
        the operator explicitly picked "live" mode) so endpoint
        misconfiguration surfaces in the history instead of being papered
        over by a synthetic mock result.
        """
        hint = (
            "No live endpoint resolved for this model. "
            "Adopt the running runtime (e.g. `llmctl adopt vllm <url>`) "
            "or pick the 'mock' mode if a synthetic run is what you want."
        )
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": None,
            "tokens_per_second": None,
            "ttft_ms": None,
            "samples": [],
            "parameters": {
                **params,
                "mode": "live",
                "reason": "no reachable runtime endpoint",
                "kind": request.kind.value,
                "concurrency": max(1, int(params.get("concurrency", 1))),
            },
            "backend": None,
            "success": False,
            "error": hint,
        }

    def _mock(
        self,
        request: BenchmarkRunRequest,
        prompts: list[str],
        params: dict[str, object],
        reason: str,
    ) -> dict[str, object]:
        """Return deterministic synthetic metrics for environments without a runtime.

        For ``HEALTH`` kind there are no prompts; we emit a single fake sample
        so the record still carries a ``mode=mock`` row in the history.
        """
        max_tokens = int(params.get("max_tokens", DEFAULT_MAX_TOKENS))
        per_prompt_latency_ms = round(max_tokens / MOCK_TOKENS_PER_SECOND * 1000.0, 2)
        ttft_ms = round(1000.0 / MOCK_TOKENS_PER_SECOND, 2)
        if request.kind == BenchmarkKind.HEALTH:
            samples = [
                {
                    "prompt": "GET /v1/models",
                    "response": f"[mock health: {reason}]",
                    "status_code": None,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "token_count_source": "synthetic",
                    "measurement": "health_check",
                    "latency_ms": ttft_ms,
                    "ttft_ms": None,
                }
            ]
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms": ttft_ms,
                "tokens_per_second": None,
                "ttft_ms": None,
                "samples": samples,
                "parameters": {
                    **params,
                    "mode": "mock",
                    "reason": reason,
                    "kind": request.kind.value,
                    "measurement": "health_check",
                    "token_count_source": "synthetic",
                    "concurrency": max(1, int(params.get("concurrency", 1))),
                },
                "backend": None,
                "success": True,
                "error": None,
            }
        samples = [
            {
                "prompt": prompt,
                "response": f"[mock benchmark: {reason}]",
                "prompt_tokens": _estimate_tokens(prompt),
                "completion_tokens": max_tokens,
                "token_count_source": "synthetic",
                "latency_ms": per_prompt_latency_ms,
                "ttft_ms": ttft_ms,
            }
            for prompt in prompts
        ]
        prompt_tokens = sum(int(sample["prompt_tokens"]) for sample in samples)
        completion_tokens = max_tokens * len(prompts)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": round(per_prompt_latency_ms * len(prompts), 2),
            "tokens_per_second": MOCK_TOKENS_PER_SECOND,
            "ttft_ms": ttft_ms,
            "samples": samples,
            "parameters": {
                **params,
                "mode": "mock",
                "reason": reason,
                "kind": request.kind.value,
                "token_count_source": "synthetic",
                "concurrency": max(1, int(params.get("concurrency", 1))),
            },
            "backend": None,
            "success": True,
            "error": None,
        }
