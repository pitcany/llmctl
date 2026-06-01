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


def _probe_managed_unit(
    unit: ManagedUnitConfig,
    http_get: Callable[[str, float], Any] | None = None,
) -> list[str] | None:
    """Probe ``http://localhost:<port>/v1/models``.

    Returns the served model IDs on success, ``None`` on failure (unit
    not running, port not bound, malformed payload — all treated the
    same). Mirrors :meth:`llmctl.adapters.vllm.VLLMAdapter._probe_unit`
    intentionally — keeping the two probes structurally identical means
    the "vllm available?" answer cannot disagree between health-check
    and scheduler-warning code paths.
    """
    http_get = http_get or _default_http_get
    url = f"http://localhost:{unit.default_port}/v1/models"
    try:
        resp = http_get(url, _PROBE_TIMEOUT_S)
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
        cfg.managed_units.vllm_tp,
        cfg.managed_units.vllm_coder,
        cfg.managed_units.vllm_reasoner,
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
    cfg = settings or load_settings()
    results: list[dict[str, object]] = []
    for backend, default_binary in _BACKEND_BINARIES.items():
        if backend == "python":
            results.append(
                {"backend": backend, "binary": "python", "path": sys.executable, "available": True}
            )
            continue
        override = cfg.runtime_config(backend).binary if backend in {"vllm", "llama_cpp"} else None
        binary = override or default_binary
        path = shutil.which(binary)
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
