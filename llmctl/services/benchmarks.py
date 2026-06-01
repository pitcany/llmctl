"""Benchmark service: real (streaming) execution with a graceful-mock fallback.

Streams prompts against a live OpenAI-compatible endpoint (vLLM, llama.cpp, LM
Studio and Ollama all expose ``/v1/chat/completions``) and measures end-to-end
latency, time-to-first-token (TTFT) and token throughput. When no endpoint is
reachable -- e.g. on a host without the runtime installed, or for an explicit
dry run -- it degrades gracefully to a deterministic synthetic benchmark so the
control plane stays usable everywhere, including the no-GPU CI container.

Each prompt/response sample is persisted on the record so the TUI can show
history and offer a one-key re-run.
"""

from __future__ import annotations

import json
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
from llmctl.telemetry.gpu import get_gpu_info, nvml_available
from llmctl.telemetry.gpu_sampler import GPUSampler, GPUSamplerSummary

#: Default chat prompt used when a request supplies none. Picked to exercise
#: reasoning + token throughput on a paragraph-length response.
DEFAULT_PROMPT = (
    "Explain the difference between maximum likelihood and Bayesian inference "
    "in one paragraph."
)
#: Default completion length requested per prompt.
DEFAULT_MAX_TOKENS = 256
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
        success=record.success,
        error=record.error,
        created_at=record.created_at,
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~whitespace words) when a server omits usage data."""
    return max(1, len(text.split()))


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
    """Pull incremental text out of an OpenAI-style streaming chunk."""
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

    def run(self, request: BenchmarkRunRequest) -> BenchmarkResult:
        """Execute a benchmark and persist its result.

        Streams a real run against a live runtime endpoint. On ``--dry-run`` (or
        when no endpoint can be resolved at all) it records a deterministic
        synthetic result with ``parameters.mode == "mock"``. Live runs that
        fail mid-flight are persisted with ``success=False`` and the error
        message so failures are inspectable in the history.
        """
        prompts = self._resolve_prompts(request)
        params = dict(request.parameters)
        params.setdefault("concurrency", max(1, request.concurrency))
        snapshot = _gpu_snapshot()
        with GPUSampler() as sampler:
            metrics = self._measure(request, prompts, params)
        gpu_summary = sampler.summary()
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
            success=metrics["success"],
            error=metrics["error"],
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record_to_benchmark(record)

    def rerun(self, benchmark_id: str) -> BenchmarkResult | None:
        """Re-execute a stored benchmark, reusing its prompts and parameters."""
        record = self.db.get(BenchmarkRecord, benchmark_id)
        if record is None:
            return None
        prompts = [
            str(sample["prompt"])
            for sample in (record.samples or [])
            if sample.get("prompt")
        ]
        params = {
            key: record.parameters[key]
            for key in ("max_tokens", "temperature", "concurrency")
            if key in record.parameters
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
            prompts=prompts,
            parameters=params,
            concurrency=int(record.parameters.get("concurrency", 1) or 1),
            dry_run=False,
        )
        return self.run(request)

    def run_sweep(self, request: BenchmarkRunRequest) -> list[BenchmarkResult]:
        """Run the benchmark at several concurrency levels (a load sweep).

        Each level is persisted as its own record named ``"{name} (c=N)"`` with
        ``parameters.concurrency == N`` so throughput-under-load can be compared
        directly in the Benchmarks screen.
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
    ) -> dict[str, object]:
        """Return benchmark metrics for the requested ``kind``.

        Behaviour:
          * Explicit ``dry_run`` -> deterministic mock result for the kind.
          * No reachable endpoint -> mock result with reason
            ``"no reachable runtime endpoint"``.
          * Endpoint reachable but the call fails -> *failure record*:
            ``success=False``, the exception in ``error``. Mock fallback is
            NOT used for live failures (we want failures to be inspectable).
        """
        if request.dry_run:
            return self._mock(request, prompts, params, "dry-run requested")
        target = self._resolve_target(request)
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
            ctx_prompts = [self._build_long_context_prompt(request)]
            return self._run_live(target, ctx_prompts, params, endpoint="chat")
        return self._run_live(target, prompts, params, endpoint="chat")

    def _resolve_target(self, request: BenchmarkRunRequest) -> _Target | None:
        """Resolve the live OpenAI-compatible endpoint for the request, if any."""
        endpoint, model_name, backend = self._resolve_endpoint(request)
        if not endpoint:
            return None
        return _Target(
            base_url=endpoint,
            model_name=model_name or "local-model",
            backend=backend,
        )

    def _resolve_endpoint(
        self, request: BenchmarkRunRequest
    ) -> tuple[str | None, str | None, str | None]:
        """Find a serving endpoint via an explicit session, a running session,
        or the runtime's default server endpoint. Returns (url, model, backend).
        """
        if request.session_id:
            record = self.db.get(SessionRecord, request.session_id)
            if record and record.endpoint_url:
                backend = record.runtime.value if record.runtime else None
                return record.endpoint_url, self._model_name(record.model_id), backend
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
                    return running.endpoint_url, name, backend
                config = self.settings.runtime_config(model.runtime.value)
                if config.endpoint:
                    return config.endpoint, name, backend
        return None, None, None

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
        """Time a GET against ``/v1/models``; record latency as ttft+total."""
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
            "ttft_ms": elapsed_ms,
            "samples": [
                {
                    "prompt": "GET /v1/models",
                    "response": body_text,
                    "status_code": status_code,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_ms": elapsed_ms,
                    "ttft_ms": elapsed_ms,
                }
            ],
            "parameters": {
                **params,
                "mode": "live",
                "endpoint": target.models_url,
                "kind": BenchmarkKind.HEALTH.value,
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
        """Stream prompts against a live endpoint; aggregate latency/TTFT/tps.

        With ``concurrency > 1`` requests are dispatched in parallel and
        throughput is computed from wall-clock time, measuring sustained
        tokens/sec under load. TTFT is averaged across requests. The
        ``endpoint`` selector picks between ``"chat"`` (``/v1/chat/completions``)
        and ``"completion"`` (``/v1/completions``) -- the same streaming
        runner handles both, since both speak SSE with the same delta shape.
        """
        max_tokens = int(params.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(params.get("temperature", 0.0))
        concurrency = max(1, int(params.get("concurrency", 1)))
        client = (
            self._client_factory()
            if self._client_factory is not None
            else httpx.Client(timeout=REQUEST_TIMEOUT)
        )
        endpoint_url = (
            target.completion_url if endpoint == "completion" else target.chat_url
        )
        wall_start = time.perf_counter()
        samples: list[dict[str, object]] = []
        with client:
            if concurrency == 1:
                for prompt in prompts:
                    samples.append(
                        self._stream_one(
                            client, target, prompt, max_tokens, temperature, endpoint
                        )
                    )
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(
                            self._stream_one,
                            client,
                            target,
                            prompt,
                            max_tokens,
                            temperature,
                            endpoint,
                        )
                        for prompt in prompts
                    ]
                    samples = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - wall_start
        total_prompt = sum(int(sample["prompt_tokens"]) for sample in samples)
        total_completion = sum(int(sample["completion_tokens"]) for sample in samples)
        ttft_values = [
            float(sample["ttft_ms"])
            for sample in samples
            if sample["ttft_ms"] is not None
        ]
        tps = round(total_completion / wall_seconds, 2) if wall_seconds > 0 else None
        avg_ttft = round(sum(ttft_values) / len(ttft_values), 2) if ttft_values else None
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
                "prompt_count": len(prompts),
                "concurrency": concurrency,
            },
            "backend": target.backend,
            "success": True,
            "error": None,
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
        content_parts: list[str] = []
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
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    content_parts.append(delta)
        elapsed = time.perf_counter() - start
        text = "".join(content_parts)
        completion = usage.get("completion_tokens")
        completion = int(completion) if completion is not None else _estimate_tokens(text)
        prompt_tokens = usage.get("prompt_tokens")
        prompt_tokens = (
            int(prompt_tokens) if prompt_tokens is not None else _estimate_tokens(prompt)
        )
        return {
            "prompt": prompt,
            "response": text[:SAMPLE_CHAR_LIMIT],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "latency_ms": round(elapsed * 1000.0, 2),
            "ttft_ms": round(ttft * 1000.0, 2) if ttft is not None else None,
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
        endpoint_url = target.chat_url if request.kind != BenchmarkKind.HEALTH else target.models_url
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
                    "latency_ms": None,
                    "ttft_ms": None,
                }
                for prompt in prompts
            ],
            "parameters": {
                **params,
                "mode": "live",
                "endpoint": endpoint_url,
                "kind": request.kind.value,
                "prompt_count": len(prompts),
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
                "prompt_count": len(prompts),
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
                    "latency_ms": ttft_ms,
                    "ttft_ms": ttft_ms,
                }
            ]
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_ms": ttft_ms,
                "tokens_per_second": None,
                "ttft_ms": ttft_ms,
                "samples": samples,
                "parameters": {
                    **params,
                    "mode": "mock",
                    "reason": reason,
                    "kind": request.kind.value,
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
                "prompt_count": len(prompts),
                "concurrency": max(1, int(params.get("concurrency", 1))),
            },
            "backend": None,
            "success": True,
            "error": None,
        }
