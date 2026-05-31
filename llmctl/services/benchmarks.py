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
from llmctl.db import BenchmarkRecord, ModelRecord, SessionRecord, SessionStatus
from llmctl.schemas import BenchmarkResult, BenchmarkRunRequest
from llmctl.telemetry.gpu import get_gpu_info, nvml_available

#: Default prompt used when a request supplies none.
DEFAULT_PROMPT = "In one sentence, explain what a GPU is."
#: Default completion length requested per prompt.
DEFAULT_MAX_TOKENS = 64
#: Assumed throughput (tokens/sec) for the synthetic mock benchmark.
MOCK_TOKENS_PER_SECOND = 50.0
#: HTTP timeout (seconds) for live benchmark calls.
REQUEST_TIMEOUT = 60.0
#: Maximum characters of generated text stored per sample.
SAMPLE_CHAR_LIMIT = 500

#: Factory used to build the (sync) HTTP client; injectable for tests.
ClientFactory = Callable[[], httpx.Client]

#: Sentinel returned by :func:`_parse_sse_line` for the ``[DONE]`` event.
_DONE = object()


@dataclass
class _Target:
    """A resolved live benchmark endpoint."""

    chat_url: str
    model_name: str


def record_to_benchmark(record: BenchmarkRecord) -> BenchmarkResult:
    """Convert benchmark DB record to schema."""
    return BenchmarkResult(
        id=record.id,
        model_id=record.model_id,
        session_id=record.session_id,
        profile_id=record.profile_id,
        name=record.name,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        latency_ms=record.latency_ms,
        tokens_per_second=record.tokens_per_second,
        time_to_first_token_ms=record.ttft_ms,
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

    def list_results(self) -> list[BenchmarkResult]:
        """List recorded benchmark results."""
        records = self.db.exec(select(BenchmarkRecord)).all()
        return [record_to_benchmark(record) for record in records]

    def run(self, request: BenchmarkRunRequest) -> BenchmarkResult:
        """Execute a benchmark and persist its result.

        Streams a real run against a live runtime endpoint and falls back to a
        deterministic synthetic benchmark when the runtime is unavailable or a
        dry run is requested. The fallback is recorded with ``parameters.mode ==
        "mock"`` and a human-readable ``reason``.
        """
        prompts = request.prompts or [DEFAULT_PROMPT]
        params = dict(request.parameters)
        params.setdefault("concurrency", max(1, request.concurrency))
        snapshot = _gpu_snapshot()
        metrics = self._measure(request, prompts, params)
        record = BenchmarkRecord(
            model_id=request.model_id,
            session_id=request.session_id,
            profile_id=request.profile_id,
            name=request.name,
            prompt_tokens=metrics["prompt_tokens"],
            completion_tokens=metrics["completion_tokens"],
            total_tokens=metrics["total_tokens"],
            latency_ms=metrics["latency_ms"],
            tokens_per_second=metrics["tokens_per_second"],
            ttft_ms=metrics["ttft_ms"],
            gpu_snapshot=snapshot,
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
        request = BenchmarkRunRequest(
            name=record.name,
            model_id=record.model_id,
            session_id=record.session_id,
            profile_id=record.profile_id,
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
        """Return benchmark metrics, choosing live execution or a mock fallback."""
        if request.dry_run:
            return self._mock(prompts, params, "dry-run requested")
        target = self._resolve_target(request)
        if target is None:
            return self._mock(prompts, params, "no reachable runtime endpoint")
        try:
            return self._run_live(target, prompts, params)
        except Exception as exc:  # noqa: BLE001 - graceful fallback to mock
            return self._mock(prompts, params, f"runtime unreachable: {exc}")

    def _resolve_target(self, request: BenchmarkRunRequest) -> _Target | None:
        """Resolve the live OpenAI-compatible endpoint for the request, if any."""
        endpoint, model_name = self._resolve_endpoint(request)
        if not endpoint:
            return None
        chat_url = endpoint.rstrip("/") + "/v1/chat/completions"
        return _Target(chat_url=chat_url, model_name=model_name or "local-model")

    def _resolve_endpoint(
        self, request: BenchmarkRunRequest
    ) -> tuple[str | None, str | None]:
        """Find a serving endpoint via an explicit session, a running session,
        or the runtime's default server endpoint."""
        if request.session_id:
            record = self.db.get(SessionRecord, request.session_id)
            if record and record.endpoint_url:
                return record.endpoint_url, self._model_name(record.model_id)
        if request.model_id:
            model = self.db.get(ModelRecord, request.model_id)
            if model:
                name = model.source or model.name
                running = self.db.exec(
                    select(SessionRecord).where(
                        SessionRecord.model_id == model.id,
                        SessionRecord.status == SessionStatus.RUNNING,
                    )
                ).first()
                if running and running.endpoint_url:
                    return running.endpoint_url, name
                config = self.settings.runtime_config(model.runtime.value)
                if config.endpoint:
                    return config.endpoint, name
        return None, None

    def _model_name(self, model_id: str | None) -> str | None:
        """Return a model's served name (source preferred) by id."""
        if not model_id:
            return None
        model = self.db.get(ModelRecord, model_id)
        return (model.source or model.name) if model else None

    def _run_live(
        self,
        target: _Target,
        prompts: list[str],
        params: dict[str, object],
    ) -> dict[str, object]:
        """Stream prompts against a live endpoint; aggregate latency/TTFT/tps.

        With ``concurrency > 1`` requests are dispatched in parallel and
        throughput is computed from wall-clock time, measuring sustained
        tokens/sec under load. TTFT is averaged across requests.
        """
        max_tokens = int(params.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(params.get("temperature", 0.0))
        concurrency = max(1, int(params.get("concurrency", 1)))
        client = (
            self._client_factory()
            if self._client_factory is not None
            else httpx.Client(timeout=REQUEST_TIMEOUT)
        )
        wall_start = time.perf_counter()
        samples: list[dict[str, object]] = []
        with client:
            if concurrency == 1:
                for prompt in prompts:
                    samples.append(
                        self._stream_one(client, target, prompt, max_tokens, temperature)
                    )
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(
                            self._stream_one, client, target, prompt, max_tokens, temperature
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
                "endpoint": target.chat_url,
                "prompt_count": len(prompts),
                "concurrency": concurrency,
            },
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
    ) -> dict[str, object]:
        """Stream a single prompt, capturing TTFT and the response text."""
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
        with client.stream("POST", target.chat_url, json=body) as response:
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

    def _mock(
        self,
        prompts: list[str],
        params: dict[str, object],
        reason: str,
    ) -> dict[str, object]:
        """Return deterministic synthetic metrics for environments without a runtime."""
        max_tokens = int(params.get("max_tokens", DEFAULT_MAX_TOKENS))
        per_prompt_latency_ms = round(max_tokens / MOCK_TOKENS_PER_SECOND * 1000.0, 2)
        ttft_ms = round(1000.0 / MOCK_TOKENS_PER_SECOND, 2)
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
                "prompt_count": len(prompts),
                "concurrency": max(1, int(params.get("concurrency", 1))),
            },
            "success": True,
            "error": None,
        }
