"""Live engine load: what a serving engine is doing right now.

GPU telemetry says a card is 99% busy. It cannot say *why*: three requests at
200k tokens each have filled the KV cache, the scheduler is preempting and
recomputing, and generation has fallen to a few tokens per second. The engine
knows all of that and publishes it -- vLLM, SGLang and llama.cpp on
``/metrics`` (Prometheus text), Strata on ``/metrics`` (JSON), Ollama on
``/api/ps`` -- but nothing in llmctl read it. ``llmctl metrics``, ``health``
and ``doctor`` do now.

Every probe is read-only, loopback-only and short (``PROBE_TIMEOUT_S``), and
any failure yields ``None`` rather than an exception: an engine that does not
answer is simply not reporting, and a dashboard refresh must never stall on
it. Counters are lifetime values since the engine started; the ratios derived
from them (prefix-cache hit rate, speculative acceptance) are lifetime
averages, which is the right reading for "is this configuration working".
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

#: Seconds an engine may take to answer its metrics probe.
PROBE_TIMEOUT_S = 2.0

#: KV-cache usage at or above which the engine is "high": a long request or
#: one more concurrent client will start evicting prefix-cache blocks.
KV_HIGH_PCT = 85.0
#: KV-cache usage at or above which the engine is "saturated": the scheduler
#: preempts running requests and recomputes them, and generation stalls.
KV_SATURATED_PCT = 95.0

Pressure = str  # "ok" | "high" | "saturated" | "unknown"

#: ``key="value"`` pairs inside a Prometheus label set; the value may hold
#: commas and escaped quotes.
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')

HttpGet = Callable[[str, float], Any]


def _default_http_get(url: str, timeout: float) -> Any:
    """Production HTTP GET. Patched in tests."""
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - localhost only


def _read(url: str, timeout: float, http_get: HttpGet) -> str | None:
    """Body of ``GET url`` as text, or ``None`` on any failure."""
    try:
        resp = http_get(url, timeout)
        return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


@dataclass(frozen=True)
class EngineLoad:
    """One engine's live load. ``None`` means the engine does not report it."""

    upstream: str
    runtime: str
    model: str | None = None
    version: str | None = None
    running: int | None = None
    waiting: int | None = None
    #: KV-cache usage, 0-100.
    kv_usage_pct: float | None = None
    #: Prefix-cache hit rate since start, 0-100.
    prefix_hit_pct: float | None = None
    #: Requests preempted (and recomputed) since start.
    preemptions: int | None = None
    #: Speculative decoding: accepted / drafted tokens since start, 0-100.
    spec_accept_pct: float | None = None
    #: Speculative decoding: tokens produced per decode step, 1 + accepted/drafts.
    spec_tokens_per_step: float | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    requests: int | None = None
    #: Decode speed the engine reports for the request in flight (Strata).
    tokens_per_s: float | None = None
    source: str = ""

    @property
    def pressure(self) -> Pressure:
        """Rule of thumb for "should I send one more request here".

        ``saturated``: waiting requests, or KV at :data:`KV_SATURATED_PCT`.
        ``high``: KV at :data:`KV_HIGH_PCT`. ``unknown``: the engine reports
        neither queue nor KV usage. Otherwise ``ok``.
        """
        if self.waiting:
            return "saturated"
        if self.kv_usage_pct is None:
            return "ok" if self.waiting is not None else "unknown"
        if self.kv_usage_pct >= KV_SATURATED_PCT:
            return "saturated"
        if self.kv_usage_pct >= KV_HIGH_PCT:
            return "high"
        return "ok"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["pressure"] = self.pressure
        return data

    def summary(self) -> str:
        """One line for a table cell or a log: only the facts the engine has."""
        parts: list[str] = []
        if self.running is not None or self.waiting is not None:
            parts.append(f"{self.running or 0} running, {self.waiting or 0} waiting")
        if self.kv_usage_pct is not None:
            parts.append(f"KV {self.kv_usage_pct:.0f}%")
        if self.prefix_hit_pct is not None:
            parts.append(f"prefix hit {self.prefix_hit_pct:.0f}%")
        if self.spec_accept_pct is not None:
            parts.append(f"spec accept {self.spec_accept_pct:.0f}%")
        if self.preemptions:
            parts.append(f"preempted {self.preemptions}x")
        if self.tokens_per_s is not None:
            parts.append(f"{self.tokens_per_s:.0f} tok/s")
        if parts:
            return ", ".join(parts)
        return "loaded, no queue or cache metrics" if self.model else "no load data"


# --------------------------------------------------------------------------- #
# Prometheus text
# --------------------------------------------------------------------------- #


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Minimal Prometheus text-format parser: name -> [(labels, value)].

    Enough for the exposition engines produce (no escapes inside label values
    beyond ``\\"``). Lines that do not parse are skipped, never fatal.
    """
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        labels: dict[str, str] = {}
        if "{" in line:
            name, rest = line.split("{", 1)
            body, _, value_text = rest.rpartition("}")
            for key, val in _LABEL_RE.findall(body):
                labels[key] = val.replace('\\"', '"')
        else:
            name, _, value_text = line.partition(" ")
        value_text = value_text.strip().split()[0] if value_text.strip() else ""
        try:
            value = float(value_text)
        except ValueError:
            continue
        out.setdefault(name.strip(), []).append((labels, value))
    return out


def _total(metrics: dict[str, list[tuple[dict[str, str], float]]], *names: str) -> float | None:
    """Sum of every sample of the first metric name present, else ``None``."""
    for name in names:
        samples = metrics.get(name)
        if samples:
            return sum(v for _, v in samples)
    return None


def _max(metrics: dict[str, list[tuple[dict[str, str], float]]], *names: str) -> float | None:
    for name in names:
        samples = metrics.get(name)
        if samples:
            return max(v for _, v in samples)
    return None


def _label(
    metrics: dict[str, list[tuple[dict[str, str], float]]], key: str, *names: str
) -> str | None:
    for name in names:
        for labels, _ in metrics.get(name) or []:
            if labels.get(key):
                return labels[key]
    return None


def _pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return 100.0 * numerator / denominator


def _int(value: float | None) -> int | None:
    return None if value is None else int(value)


def parse_vllm(text: str, *, upstream: str = "", version: str | None = None) -> EngineLoad:
    """vLLM's ``/metrics`` (V1 engine names; the pre-V1 ``gpu_cache_usage_perc`` too)."""
    m = parse_prometheus(text)
    drafted = _total(m, "vllm:spec_decode_num_draft_tokens_total")
    accepted = _total(m, "vllm:spec_decode_num_accepted_tokens_total")
    drafts = _total(m, "vllm:spec_decode_num_drafts_total")
    kv = _max(m, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    return EngineLoad(
        upstream=upstream,
        runtime="vllm",
        model=_label(m, "model_name", "vllm:num_requests_running", "vllm:kv_cache_usage_perc"),
        version=version,
        running=_int(_total(m, "vllm:num_requests_running")),
        waiting=_int(_total(m, "vllm:num_requests_waiting")),
        kv_usage_pct=None if kv is None else kv * 100.0,
        prefix_hit_pct=_pct(
            _total(m, "vllm:prefix_cache_hits_total"), _total(m, "vllm:prefix_cache_queries_total")
        ),
        preemptions=_int(_total(m, "vllm:num_preemptions_total")),
        spec_accept_pct=_pct(accepted, drafted),
        spec_tokens_per_step=(None if accepted is None or not drafts else 1.0 + accepted / drafts),
        prompt_tokens=_int(_total(m, "vllm:prompt_tokens_total")),
        generation_tokens=_int(_total(m, "vllm:generation_tokens_total")),
        requests=_int(_total(m, "vllm:request_success_total")),
        source="/metrics",
    )


def parse_llama_cpp(text: str, *, upstream: str = "") -> EngineLoad:
    """llama.cpp server ``/metrics`` (needs ``--metrics`` on the server)."""
    m = parse_prometheus(text)
    kv = _max(m, "llamacpp:kv_cache_usage_ratio")
    return EngineLoad(
        upstream=upstream,
        runtime="llama_cpp",
        running=_int(_total(m, "llamacpp:requests_processing")),
        waiting=_int(_total(m, "llamacpp:requests_deferred")),
        kv_usage_pct=None if kv is None else kv * 100.0,
        prompt_tokens=_int(_total(m, "llamacpp:prompt_tokens_total")),
        generation_tokens=_int(_total(m, "llamacpp:tokens_predicted_total")),
        source="/metrics",
    )


def parse_sglang(text: str, *, upstream: str = "") -> EngineLoad:
    """SGLang ``/metrics`` (``--enable-metrics``)."""
    m = parse_prometheus(text)
    usage = _max(m, "sglang:token_usage")
    hit = _max(m, "sglang:cache_hit_rate")
    accept_len = _max(m, "sglang:spec_accept_length")
    return EngineLoad(
        upstream=upstream,
        runtime="sglang",
        model=_label(m, "model_name", "sglang:num_running_reqs", "sglang:token_usage"),
        running=_int(_total(m, "sglang:num_running_reqs")),
        waiting=_int(_total(m, "sglang:num_queue_reqs")),
        kv_usage_pct=None if usage is None else usage * 100.0,
        prefix_hit_pct=None if hit is None else hit * (100.0 if hit <= 1.0 else 1.0),
        spec_tokens_per_step=accept_len,
        prompt_tokens=_int(_total(m, "sglang:prompt_tokens_total")),
        generation_tokens=_int(_total(m, "sglang:generation_tokens_total")),
        source="/metrics",
    )


def parse_strata(payload: Any, *, upstream: str = "") -> EngineLoad:
    """Strata's JSON ``/metrics``: ``live`` (state, running, waiting, tok/s) and ``engine``."""
    live = payload.get("live") if isinstance(payload, dict) else None
    engine = payload.get("engine") if isinstance(payload, dict) else None
    live = live if isinstance(live, dict) else {}
    engine = engine if isinstance(engine, dict) else {}
    running = live.get("running")
    if not isinstance(running, int):
        running = 1 if live.get("state") in {"reading", "generating"} else 0
    waiting = live.get("waiting")
    if not isinstance(waiting, int):
        waiting = live.get("queued") if isinstance(live.get("queued"), int) else 0
    tok_s = live.get("tok_s")
    model = engine.get("model")
    return EngineLoad(
        upstream=upstream,
        runtime="strata",
        model=model if isinstance(model, str) else None,
        running=running,
        waiting=waiting,
        tokens_per_s=float(tok_s) if isinstance(tok_s, int | float) else None,
        source="/metrics",
    )


def parse_ollama_ps(payload: Any, *, upstream: str = "") -> EngineLoad:
    """Ollama ``/api/ps``: only which models are resident; Ollama has no queue metrics."""
    models = payload.get("models") if isinstance(payload, dict) else None
    names = [
        m.get("name") or m.get("model")
        for m in (models or [])
        if isinstance(m, dict) and isinstance(m.get("name") or m.get("model"), str)
    ]
    return EngineLoad(
        upstream=upstream,
        runtime="ollama",
        model=", ".join(names) if names else None,
        source="/api/ps",
    )


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


def fetch_engine_load(
    upstream: str,
    runtime: str | None,
    *,
    timeout: float = PROBE_TIMEOUT_S,
    http_get: HttpGet | None = None,
) -> EngineLoad | None:
    """Probe ``upstream`` for its live load, or ``None`` when it does not answer.

    ``runtime`` picks the parser (``vllm``, ``llama_cpp``, ``sglang``,
    ``strata``, ``ollama``). Unknown or ``None`` tries vLLM's Prometheus
    shape, which llama.cpp and SGLang share, and tells them apart by the
    metric prefix.
    """
    http_get = http_get or _default_http_get
    base = upstream.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    runtime = (runtime or "").lower()

    if runtime == "ollama":
        text = _read(f"{base}/api/ps", timeout, http_get)
        if text is None:
            return None
        try:
            return parse_ollama_ps(json.loads(text), upstream=upstream)
        except ValueError:
            return None

    text = _read(f"{base}/metrics", timeout, http_get)
    if text is None:
        return None
    stripped = text.lstrip()
    if runtime == "strata" or stripped.startswith("{"):
        try:
            return parse_strata(json.loads(text), upstream=upstream)
        except ValueError:
            return None
    if runtime == "llama_cpp" or "llamacpp:" in text:
        return parse_llama_cpp(text, upstream=upstream)
    if runtime == "sglang" or "sglang:" in text:
        return parse_sglang(text, upstream=upstream)
    version = None
    version_text = _read(f"{base}/version", timeout, http_get)
    if version_text:
        try:
            raw = json.loads(version_text).get("version")
            version = raw if isinstance(raw, str) else None
        except (ValueError, AttributeError):
            version = None
    return parse_vllm(text, upstream=upstream, version=version)
