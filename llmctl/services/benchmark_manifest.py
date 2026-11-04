"""Immutable experiment manifests for benchmark records (FIXPLAN §1C).

A manifest is a versioned, allowlisted snapshot of *what was measured*: the
resolved model/served identity, engine identity, launch configuration,
hardware context, the workload (by hash and by parameters), and the effective
generation parameters. It is captured once at run time and never rewritten,
so a record can always answer "what exactly produced this number?".

Rules encoded here:

* Only allowlisted keys are captured -- never whole records, environments,
  or credentials (launch-plan ``env`` is deliberately excluded).
* Unknown values are recorded as the string ``"unknown"`` rather than
  omitted, so a missing fact is distinguishable from an absent field.
* Runner-host GPU telemetry is labeled as such. When the endpoint is remote
  the manifest says so instead of attributing local GPU readings to a
  serving host we cannot see.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

from llmctl.db import ModelRecord, SessionRecord
from llmctl.schemas import BenchmarkRunRequest
from llmctl.services.benchmark_parameters import effective_parameters

#: Bump when the allowlist changes shape; readers can branch on this.
MANIFEST_VERSION = 1

UNKNOWN = "unknown"

#: Sections compared by :func:`manifest_drift`. Deliberately excludes
#: ``created_at`` and all GPU telemetry (volatile between runs even on an
#: otherwise identical host).
_DRIFT_SECTIONS = (
    "model",
    "served",
    "engine",
    "launch",
    "workload",
    "effective_parameters",
)

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", ""}


def _unknown_dict(keys: list[str]) -> dict[str, Any]:
    return {key: UNKNOWN for key in keys}


def workload_hash(request: BenchmarkRunRequest, base_prompts: list[str]) -> str:
    """Stable hash of the workload identity (kind, prompts, size, sampling).

    Two runs with the same hash sent the same bytes with the same sampling
    settings; concurrency is excluded on purpose -- a sweep changes only the
    concurrency ceiling, not the workload.
    """
    parameters = effective_parameters(request.parameters)
    payload = {
        "kind": request.kind.value,
        "prompts": list(base_prompts),
        "prompt_count": len(base_prompts),
        "repetitions": request.repetitions,
        "max_tokens": parameters["max_tokens"],
        "temperature": parameters["temperature"],
        "context_length": request.context_length,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _endpoint_host(endpoint_url: str | None) -> str:
    if not endpoint_url:
        return ""
    try:
        from urllib.parse import urlparse

        return urlparse(endpoint_url).hostname or ""
    except ValueError:
        return ""


def _serving_host_section(
    endpoint_url: str | None, runner_gpus: list[dict[str, Any]]
) -> dict[str, Any]:
    """Describe where the model was served, honestly.

    Loopback endpoints are served on the runner host, so the runner's GPU
    identity applies. Anything else is remote: llmctl can see its endpoint
    but not its hardware, and the manifest says so instead of guessing.
    """
    if not endpoint_url:
        return {"location": UNKNOWN, "endpoint_host": UNKNOWN}
    host = _endpoint_host(endpoint_url)
    if endpoint_url and host not in _LOOPBACK_HOSTS:
        return {
            "location": "remote",
            "endpoint_host": host,
            "telemetry": (
                "unavailable (remote endpoint; runner GPU readings not attributed)"
            ),
        }
    return {
        "location": "runner_host",
        "endpoint_host": host or "loopback",
        "gpu_identity": [str(g.get("name", UNKNOWN)) for g in runner_gpus],
    }


def _model_section(model: ModelRecord | None, model_id: str | None) -> dict[str, Any]:
    if model is None:
        return _unknown_dict(["id", "name", "source", "runtime", "revision", "host"])
    return {
        "id": model.id or model_id or UNKNOWN,
        "name": model.name or UNKNOWN,
        "source": model.source or UNKNOWN,
        "runtime": model.runtime.value if model.runtime else UNKNOWN,
        # Local-file revision pinning is not implemented; record honestly.
        "revision": UNKNOWN,
        "host": model.host or UNKNOWN,
    }


def _launch_section(session: SessionRecord | None) -> dict[str, Any]:
    """Allowlisted launch configuration from the serving session's plan.

    ``env`` and raw ``command`` are excluded on purpose: environments can
    carry credentials and the plan's structural knobs are what an experiment
    needs to be comparable.
    """
    keys = [
        "gpu_ids",
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "dtype",
        "quantization",
        "port",
    ]
    if session is None:
        return {**_unknown_dict(keys), "systemd_unit": UNKNOWN}
    plan = session.launch_plan or {}
    return {
        "gpu_ids": plan.get("gpu_ids", UNKNOWN),
        "tensor_parallel_size": plan.get("tensor_parallel_size", UNKNOWN),
        "max_model_len": plan.get("max_model_len", UNKNOWN),
        "gpu_memory_utilization": plan.get("gpu_memory_utilization", UNKNOWN),
        "dtype": plan.get("dtype", UNKNOWN),
        "quantization": plan.get("quantization", UNKNOWN),
        "port": session.port if session.port is not None else UNKNOWN,
        "systemd_unit": session.systemd_unit or UNKNOWN,
    }


def _engine_section(backend: str | None, endpoint_url: str | None) -> dict[str, Any]:
    return {
        "name": backend or UNKNOWN,
        "version": UNKNOWN,  # engine version probing is not wired yet
        "endpoint": endpoint_url or UNKNOWN,
    }


def build_manifest(
    request: BenchmarkRunRequest,
    *,
    model: ModelRecord | None,
    session: SessionRecord | None,
    backend: str | None,
    endpoint_url: str | None,
    base_prompts: list[str],
    gpu_snapshot: dict[str, Any] | None,
    execution: str,
) -> dict[str, Any]:
    """Assemble the versioned, allowlisted manifest for one benchmark run.

    ``execution`` is ``"live"`` or ``"synthetic"``. ``gpu_snapshot`` is the
    runner-host point-in-time snapshot captured by the benchmark service.
    """
    runner_gpus = list((gpu_snapshot or {}).get("gpus", []))
    parameters = effective_parameters(request.parameters)
    return {
        "version": MANIFEST_VERSION,
        "execution": execution,
        "model": _model_section(model, request.model_id),
        "served": {
            "endpoint_url": endpoint_url or UNKNOWN,
            "served_name": (
                (session.served_name if session and session.served_name else None)
                or UNKNOWN
            ),
            "backend": backend or UNKNOWN,
        },
        "engine": _engine_section(backend, endpoint_url),
        "launch": _launch_section(session),
        "workload": {
            "kind": request.kind.value,
            "hash": workload_hash(request, base_prompts),
            "prompt_count": len(base_prompts),
            "repetitions": request.repetitions,
            "total_requests": len(base_prompts) * request.repetitions,
            "context_length": request.context_length,
        },
        "effective_parameters": {
            "max_tokens": parameters["max_tokens"],
            "temperature": parameters["temperature"],
            "stream": True,
            "endpoint_kind": request.kind.value,
        },
        "runner_host": {
            "hostname": socket.gethostname(),
            "gpu_count": len(runner_gpus),
            "gpus": [
                {
                    "index": g.get("index", UNKNOWN),
                    "name": g.get("name", UNKNOWN),
                    "memory_total_mb": g.get("memory_total_mb", UNKNOWN),
                }
                for g in runner_gpus
            ],
            # Runner telemetry describes the benchmark runner, not
            # necessarily the serving host; see serving_host below.
            "telemetry_scope": "runner_host",
        },
        "serving_host": _serving_host_section(endpoint_url, runner_gpus),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts to dotted paths for comparison."""
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    else:
        flat[prefix] = value
    return flat


def manifest_drift(
    source: dict[str, Any] | None, current: dict[str, Any] | None
) -> list[str]:
    """Return human-readable drift entries between two manifests.

    Only identity/config sections are compared (see :data:`_DRIFT_SECTIONS`);
    timestamps and GPU telemetry move between any two runs and are not
    drift. Each entry reads ``"section.key: <old> -> <new>"``.
    """
    if not source or not current:
        return []
    old_flat: dict[str, Any] = {}
    new_flat: dict[str, Any] = {}
    for section in _DRIFT_SECTIONS:
        old_flat.update(_flatten(source.get(section), section))
        new_flat.update(_flatten(current.get(section), section))
    drift: list[str] = []
    for key in sorted(set(old_flat) | set(new_flat)):
        old_value = old_flat.get(key, "<missing>")
        new_value = new_flat.get(key, "<missing>")
        if old_value != new_value:
            drift.append(f"{key}: {old_value} -> {new_value}")
    return drift


def export_payload(result: Any) -> dict[str, Any]:
    """Build the portable export document for one benchmark result.

    ``result`` is a :class:`~llmctl.schemas.BenchmarkResult`. Legacy rows
    without a manifest export with ``manifest.provenance = "unknown"`` so
    the document is honest about what was never captured.
    """
    manifest = dict(getattr(result, "manifest", None) or {})
    if not manifest:
        manifest = {"version": MANIFEST_VERSION, "provenance": "unknown"}
    return {
        "result": result.model_dump(mode="json"),
        "manifest": manifest,
    }


def write_export(path: Path, payloads: list[dict[str, Any]]) -> Path:
    """Write export payloads as one JSON document (results + manifests)."""
    document = {"format": "llmctl-benchmark-export", "version": 1, "results": payloads}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True))
    return path
