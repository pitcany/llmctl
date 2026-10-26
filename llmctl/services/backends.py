"""Backend availability detection for the ``doctor`` command + dashboard.

Reports which runtimes are usable on the current host. For most runtimes
that means "binary on PATH"; for vLLM specifically, it also means
"managed systemd unit answering /v1/models" (Phase A introduced HTTP
probes for the adapter; this helper was reading a stale binary-only
view, producing a bogus "vllm backend missing" scheduler warning even
when vllm-tp.service was actively serving). Detection is read-only and
never launches anything.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from llmctl.config import ManagedUnitConfig, Settings, load_settings

#: Logical backend -> default executable name probed on PATH.
_BACKEND_BINARIES: dict[str, str] = {
    "vllm": "vllm",
    "llama_cpp": "llama-server",
    "sglang": "python3",
    "freetoken": "ft",
    "strata": "python3",
    "lmstudio": "lms",
    "ollama": "ollama",
    "python": "python",
}

#: Per-port probe timeout when checking vLLM managed units. Short by
#: design so an unreachable unit doesn't stall doctor/dashboard refresh.
_PROBE_TIMEOUT_S = 1.5


def _default_http_get(url: str, timeout: float) -> Any:
    """Production HTTP GET used for vLLM probes. Patched in tests."""
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - localhost only


def probe_openai_v1_models(
    base_url: str,
    timeout: float = _PROBE_TIMEOUT_S,
    http_get: Callable[[str, float], Any] | None = None,
) -> list[str] | None:
    """Probe ``<base_url>/v1/models`` and return served model ids.

    Returns the list of served model IDs on success and ``None`` on any
    failure (host unreachable, malformed payload, timeout, etc.) — all
    treated the same so the caller has a single "is this endpoint
    useful?" signal. Used by the doctor/dashboard probe path *and* by
    the adopt flow that tracks externally-managed endpoints.
    """
    http_get = http_get or _default_http_get
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        resp = http_get(url, timeout)
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    ids: list[str] = []
    for m in payload.get("data", []):
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if isinstance(mid, str):
            ids.append(mid)
    return ids


def probe_ollama_loaded(
    base_url: str,
    timeout: float = _PROBE_TIMEOUT_S,
    http_get: Callable[[str, float], Any] | None = None,
) -> list[str] | None:
    """Models an Ollama server holds in memory now (``<base_url>/api/ps``).

    Ollama's ``/v1/models`` lists every installed model, loaded or not, so it
    cannot tell which of two lanes on one Ollama server is serving. Returns
    ``None`` on any failure, like :func:`probe_openai_v1_models`.
    """
    http_get = http_get or _default_http_get
    url = f"{base_url.rstrip('/')}/api/ps"
    try:
        resp = http_get(url, timeout)
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    names: list[str] = []
    for m in payload.get("models", []) if isinstance(payload, dict) else []:
        if isinstance(m, dict):
            names += [v for v in (m.get("name"), m.get("model")) if isinstance(v, str)]
    return names


def probe_llama_swap_running(
    base_url: str,
    timeout: float = _PROBE_TIMEOUT_S,
    http_get: Callable[[str, float], Any] | None = None,
) -> list[str] | None:
    """Models llama-swap has loaded and ready now (``<base_url>/running``).

    Like Ollama, llama-swap's ``/v1/models`` lists every configured model,
    loaded or not. A model still starting is not listed. Returns ``None`` on
    any failure, like :func:`probe_openai_v1_models`.
    """
    http_get = http_get or _default_http_get
    url = f"{base_url.rstrip('/')}/running"
    try:
        resp = http_get(url, timeout)
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    rows = payload.get("running") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []  # llama-swap (Go) sends an idle router's empty list as null
    return [
        m["model"]
        for m in rows
        if isinstance(m, dict) and isinstance(m.get("model"), str) and m.get("state") == "ready"
    ]


#: Runtime id -> probe for what its server holds in memory, for servers whose
#: ``/v1/models`` lists everything they could serve.
LOADED_PROBES: dict[str, Callable[[str], list[str] | None]] = {
    "ollama": probe_ollama_loaded,
    "llama_swap": probe_llama_swap_running,
}


def _probe_managed_unit(
    unit: ManagedUnitConfig,
    http_get: Callable[[str, float], Any] | None = None,
) -> list[str] | None:
    """Probe a managed-unit's local port for ``/v1/models``.

    Thin wrapper around :func:`probe_openai_v1_models` for code paths
    that already have a :class:`ManagedUnitConfig` in hand.
    """
    return probe_openai_v1_models(
        f"http://localhost:{unit.default_port}",
        _PROBE_TIMEOUT_S,
        http_get,
    )


def _vllm_managed_unit_available(
    cfg: Settings,
    http_get: Callable[[str, float], Any] | None = None,
) -> tuple[bool, str | None]:
    """Return ``(available, served_unit_name)`` based on HTTP probes.

    If any managed vLLM unit answers /v1/models with at least one served
    model, vLLM is available. Otherwise the caller falls back to the
    binary-on-PATH check.
    """
    candidates = [
        cfg.managed_units.vllm_engine,
    ]
    for unit in candidates:
        ids = _probe_managed_unit(unit, http_get)
        if ids:
            return True, unit.unit_name
    return False, None


def detect_backends(
    settings: Settings | None = None,
    *,
    http_get: Callable[[str, float], Any] | None = None,
) -> list[dict[str, object]]:
    """Return availability info for every known backend.

    For most runtimes "available" means "binary resolves on PATH." For
    vLLM specifically we also accept "a managed systemd unit answers
    /v1/models" — the real production posture on hosts where vLLM runs
    under systemd and the ``vllm`` CLI isn't necessarily on PATH.

    ``http_get`` is injectable for tests; defaults to
    :func:`urllib.request.urlopen`.
    """
    from llmctl.services.engines import _really_installed

    cfg = settings or load_settings()
    results: list[dict[str, object]] = []
    for backend, default_binary in _BACKEND_BINARIES.items():
        if backend == "python":
            results.append(
                {"backend": backend, "binary": "python", "path": sys.executable, "available": True}
            )
            continue
        override = (
            cfg.runtime_config(backend).binary
            if backend in {"vllm", "llama_cpp", "sglang", "freetoken", "strata"}
            else None
        )
        binary = override or default_binary
        path = shutil.which(binary)
        if path is not None and not _really_installed(backend, path):
            path = None  # SGLang and Strata launch through a python; any python3 is not them
        if backend == "lmstudio" and path is None:
            from llmctl.adapters.lmstudio import lms_binary

            path = lms_binary()
        available = path is not None

        # vLLM-specific fallback: also accept "managed unit serving"
        # so the doctor + dashboard don't lie about a vllm-tp.service
        # that's actively running on the box.
        if backend == "vllm" and not available:
            available, serving_unit = _vllm_managed_unit_available(cfg, http_get)
            if available:
                results.append(
                    {
                        "backend": backend,
                        "binary": binary,
                        "path": f"managed unit: {serving_unit}",
                        "available": True,
                    }
                )
                continue

        results.append(
            {
                "backend": backend,
                "binary": binary,
                "path": path,
                "available": available,
            }
        )
    return results


def missing_backends(
    settings: Settings | None = None,
    *,
    http_get: Callable[[str, float], Any] | None = None,
) -> list[str]:
    """Return the names of backends whose binary/endpoint could not be resolved."""
    return [
        b["backend"]
        for b in detect_backends(settings, http_get=http_get)
        if not b["available"]
    ]
