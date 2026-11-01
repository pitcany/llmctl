"""Collect live engine load for every endpoint llmctl knows about.

Endpoints come from two places: the managed units in settings (``vllm-engine``
on :8003, ``ollama-gpu1`` on :11435, ...) and the plugin lanes (a Strata or
llama.cpp lane with its own upstream). Both already describe a port and a
runtime; this module probes each one for what it is doing right now
(:mod:`llmctl.telemetry.engine_metrics`) and returns one row per endpoint,
so ``llmctl metrics``, ``health`` and ``doctor`` share one picture.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from llmctl.adapters.ollama import DEFAULT_ENDPOINT as OLLAMA_DEFAULT_ENDPOINT
from llmctl.config import Settings
from llmctl.services.lanes import Lane, load_lanes
from llmctl.telemetry.engine_metrics import (
    PROBE_TIMEOUT_S,
    EngineLoad,
    HttpGet,
    fetch_engine_load,
)


@dataclass(frozen=True)
class EngineLoadRow:
    """One endpoint and what it reported (``load`` is ``None`` when it did not answer)."""

    name: str
    runtime: str | None
    upstream: str
    #: ``unit`` (a managed_units role), ``runtime`` (a server runtime from
    #: settings, e.g. Ollama) or ``lane`` (a plugin lane).
    source: str
    load: EngineLoad | None

    @property
    def answered(self) -> bool:
        return self.load is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runtime": self.runtime,
            "upstream": self.upstream,
            "source": self.source,
            "answered": self.answered,
            "load": self.load.as_dict() if self.load else None,
        }


def _endpoint_key(upstream: str) -> str:
    parts = urlsplit(upstream)
    host = (parts.hostname or "").lower()
    if host in {"localhost", "0.0.0.0", "::1", "::"}:  # noqa: S104 - normalising, not binding
        host = "127.0.0.1"
    return f"{host}:{parts.port or ''}"


def endpoints(settings: Settings, lanes: list[Lane] | None = None) -> list[EngineLoadRow]:
    """Every endpoint to probe, unprobed (``load=None``), de-duplicated by host:port.

    Managed-unit roles come first (their names are what ``llmctl status``
    shows), then server runtimes from settings, then plugin lanes; a lane on
    a port already listed is folded into the earlier row.
    """
    rows: list[EngineLoadRow] = []
    seen: set[str] = set()

    def add(name: str, runtime: str | None, upstream: str, source: str) -> None:
        key = _endpoint_key(upstream)
        if key in seen:
            return
        seen.add(key)
        rows.append(EngineLoadRow(name, runtime, upstream, source, None))

    roles = dict(settings.managed_units.roles())
    two = settings.managed_units.vllm_engine2
    if two is not None:
        roles.setdefault("vllm-engine2", two)
    for role, unit in roles.items():
        runtime = getattr(unit, "runtime", None) or "vllm"
        add(role, runtime, f"http://127.0.0.1:{unit.default_port}", "unit")
    # Server runtimes llmctl talks to but does not manage (Ollama on :11434).
    for runtime, default in (("ollama", OLLAMA_DEFAULT_ENDPOINT),):
        cfg = settings.runtimes.get(runtime)
        add(runtime, runtime, (cfg.endpoint if cfg and cfg.endpoint else default), "runtime")
    if lanes is None:
        try:
            lanes = load_lanes(settings.managed_units)
        except Exception:  # noqa: BLE001 - lanes are advisory
            lanes = []
    for lane in lanes:
        add(lane.name, lane.runtime, lane.upstream, "lane")
    return rows


def collect_engine_loads(
    settings: Settings,
    *,
    lanes: list[Lane] | None = None,
    http_get: HttpGet | None = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> list[EngineLoadRow]:
    """Probe every endpoint concurrently; rows keep their :func:`endpoints` order."""
    rows = endpoints(settings, lanes)
    if not rows:
        return []

    def probe(row: EngineLoadRow) -> EngineLoadRow:
        try:
            load = fetch_engine_load(row.upstream, row.runtime, timeout=timeout, http_get=http_get)
        except Exception:  # noqa: BLE001 - an endpoint that cannot be read is not answering
            load = None
        return EngineLoadRow(row.name, row.runtime, row.upstream, row.source, load)

    with ThreadPoolExecutor(max_workers=min(8, len(rows))) as pool:
        return list(pool.map(probe, rows))
