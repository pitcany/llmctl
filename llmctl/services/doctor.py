"""Doctor service: environment inspection with pass/warn/fail verdicts.

Runs read-only checks over configuration, storage, runtimes, GPU telemetry,
tracked sessions, and registered state, and returns a structured report the
CLI and API render identically. Nothing here modifies the system.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from sqlmodel import Session as DBSession
from sqlmodel import select

from llmctl.config import Settings, load_model_dirs, load_settings
from llmctl.db import SessionRecord, SessionStatus, get_engine, init_db
from llmctl.telemetry.process import is_pid_running

_ACTIVE = {SessionStatus.RUNNING, SessionStatus.STARTING, SessionStatus.DEGRADED}


@dataclass(frozen=True)
class DoctorCheck:
    """One doctor finding."""

    name: str
    verdict: str  # "pass" | "warn" | "fail"
    detail: str
    remediation: str | None = None


def run_doctor(settings: Settings | None = None) -> dict[str, Any]:
    """Run all doctor checks and return a structured report.

    Returns ``{"passed": [...], "warnings": [...], "failures": [...],
    "ok": bool}`` where each entry is a :class:`DoctorCheck` dict. ``ok``
    is False only when there is at least one failure.
    """
    cfg = settings or load_settings()
    checks: list[DoctorCheck] = []
    named_checks = (
        ("storage", lambda: _check_storage(cfg)),
        ("backends", lambda: _check_backends(cfg)),
        ("gpu", _check_gpu),
        ("sessions", lambda: _check_sessions(cfg)),
        ("drift", lambda: _check_drift(cfg)),
        ("gateway", lambda: _check_gateway(cfg)),
    )
    for name, check in named_checks:
        # A diagnostic must degrade one broken check (e.g. malformed YAML in
        # model_dirs.yaml crashing the drift scan) into a failure entry, not
        # take down the whole report.
        try:
            checks.extend(check())
        except Exception as exc:
            checks.append(
                DoctorCheck(
                    f"doctor:{name}",
                    "fail",
                    f"check crashed: {exc}",
                    "This usually means a malformed config/state file; the error "
                    "message above names the culprit.",
                )
            )

    report = {
        "passed": [asdict(c) for c in checks if c.verdict == "pass"],
        "warnings": [asdict(c) for c in checks if c.verdict == "warn"],
        "failures": [asdict(c) for c in checks if c.verdict == "fail"],
    }
    report["ok"] = not report["failures"]
    return report


def _check_storage(cfg: Settings) -> list[DoctorCheck]:
    """Database reachable/writable, logs directory writable."""
    checks: list[DoctorCheck] = []
    try:
        init_db(cfg.database_url)
        with DBSession(get_engine(cfg.database_url)) as db:
            db.exec(select(SessionRecord).limit(1)).first()
        checks.append(DoctorCheck("database", "pass", f"reachable at {cfg.database_url}"))
    except Exception as exc:
        checks.append(
            DoctorCheck(
                "database",
                "fail",
                f"cannot open {cfg.database_url}: {exc}",
                "Check LLMCTL_DB_URL / database.url and directory permissions.",
            )
        )
    logs_dir = cfg.scheduler.logs_dir
    if logs_dir:
        writable = os.access(logs_dir, os.W_OK) if os.path.isdir(logs_dir) else False
        if writable:
            checks.append(DoctorCheck("logs-dir", "pass", f"writable: {logs_dir}"))
        else:
            checks.append(
                DoctorCheck(
                    "logs-dir",
                    "warn",
                    f"scheduler.logs_dir is not a writable directory: {logs_dir}",
                    "Create the directory or fix scheduler.logs_dir in settings.yaml.",
                )
            )
    return checks


def _check_backends(cfg: Settings) -> list[DoctorCheck]:
    """Runtime binaries / endpoints detected."""
    from llmctl.services.backends import detect_backends

    checks: list[DoctorCheck] = []
    for entry in detect_backends(cfg):
        name = f"backend:{entry['backend']}"
        if entry["available"]:
            checks.append(DoctorCheck(name, "pass", f"found: {entry['path']}"))
        else:
            checks.append(
                DoctorCheck(
                    name,
                    "warn",
                    f"binary '{entry['binary']}' not found on PATH",
                    "Install the runtime or set its binary path in settings.yaml "
                    "(runtimes.<name>.binary). Unused runtimes can be ignored.",
                )
            )
    return checks


def _check_gpu() -> list[DoctorCheck]:
    """NVML/GPU telemetry availability (absence is a warning, not an error)."""
    from llmctl.telemetry.gpu import get_gpu_info, nvml_available

    if not nvml_available():
        return [
            DoctorCheck(
                "gpu-telemetry",
                "warn",
                "NVML unavailable; VRAM-aware scheduling degrades to CPU-only heuristics",
                "Install NVIDIA drivers + nvidia-ml-py on GPU hosts; ignore on CPU-only hosts.",
            )
        ]
    gpus = get_gpu_info()
    return [DoctorCheck("gpu-telemetry", "pass", f"NVML OK; {len(gpus)} GPU(s) visible")]


def _check_sessions(cfg: Settings) -> list[DoctorCheck]:
    """Dead PIDs still marked active; duplicate port assignments."""
    checks: list[DoctorCheck] = []
    try:
        init_db(cfg.database_url)
        with DBSession(get_engine(cfg.database_url)) as db:
            records = db.exec(
                select(SessionRecord).where(SessionRecord.status.in_(_ACTIVE))  # type: ignore[attr-defined]
            ).all()
    except Exception:
        return checks  # storage check already reported the failure
    dead = [r.id for r in records if r.pid and not is_pid_running(r.pid)]
    if dead:
        checks.append(
            DoctorCheck(
                "stale-sessions",
                "warn",
                f"{len(dead)} active session(s) whose process is gone: {', '.join(dead)}",
                "Run `llmctl cleanup` (or `llmctl reconcile`) to mark them stopped.",
            )
        )
    else:
        checks.append(DoctorCheck("stale-sessions", "pass", "no dead PIDs tracked as active"))
    ports: dict[int, list[str]] = {}
    for record in records:
        if record.port:
            ports.setdefault(record.port, []).append(record.id)
    collisions = {port: ids for port, ids in ports.items() if len(ids) > 1}
    if collisions:
        detail = "; ".join(f"port {port}: {', '.join(ids)}" for port, ids in collisions.items())
        checks.append(
            DoctorCheck(
                "port-collisions",
                "fail",
                f"multiple active sessions share a port — {detail}",
                "Stop or detach the duplicates; routing to a shared port is ambiguous.",
            )
        )
    else:
        checks.append(DoctorCheck("port-collisions", "pass", "no duplicate active ports"))
    return checks


def _check_drift(cfg: Settings) -> list[DoctorCheck]:
    """Fold in the read-only validate checks (paths, symlinks, unit ports)."""
    from llmctl.presets.store import load_all as load_all_presets
    from llmctl.services import validate as validate_svc
    from llmctl.services.registry import RegistryService

    try:
        with DBSession(get_engine(cfg.database_url)) as db:
            models = RegistryService(db).list_models(include_inactive=True)
    except Exception:
        models = []
    findings = [
        *validate_svc.check_preset_model_ids(load_all_presets()),
        *validate_svc.check_registry_paths(models),
        *validate_svc.check_model_root_symlinks(load_model_dirs()),
        *validate_svc.check_managed_unit_ports([cfg.managed_units.vllm_tp]),
    ]
    if not findings:
        return [DoctorCheck("state-drift", "pass", "presets, registry paths, and units line up")]
    return [
        DoctorCheck(
            f"drift:{finding.check}",
            "fail",
            f"{finding.target}: {finding.detail}",
            "See `llmctl validate` for the full drift report.",
        )
        for finding in findings
    ]


def _check_gateway(cfg: Settings) -> list[DoctorCheck]:
    """Gateway reachability (informational — the gateway is optional)."""
    import httpx

    host = cfg.router.host if cfg.router.host not in ("0.0.0.0", "") else "127.0.0.1"
    url = f"http://{host}:{cfg.router.port}/health"
    try:
        response = httpx.get(url, timeout=1.0)
        if response.status_code in (200, 401):
            return [DoctorCheck("gateway", "pass", f"answering at {url}")]
        return [
            DoctorCheck(
                "gateway",
                "warn",
                f"unexpected HTTP {response.status_code} from {url}",
                "Check `llmctl gateway` logs.",
            )
        ]
    except Exception:
        return [
            DoctorCheck(
                "gateway",
                "warn",
                f"not reachable at {url} (fine if you don't use the router)",
                "Start it with `llmctl gateway` if you want OpenAI-compatible routing.",
            )
        ]
