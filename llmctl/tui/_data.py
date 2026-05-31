"""Data-access helpers for the Textual TUI.

These thin functions open a short-lived database session, invoke the existing
service layer, and return plain schema objects (or simple dicts). Keeping all
data access here makes the screen widgets small, declarative, and easy to
unit-test without instantiating Textual widgets.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlmodel import Session

from llmctl.config import load_settings
from llmctl.db import get_engine, init_db
from llmctl.schemas import (
    BenchmarkResult,
    BenchmarkRunRequest,
    GPUInfo,
    LaunchPlan,
    Model,
    SessionStartRequest,
)
from llmctl.schemas import Session as RuntimeSession
from llmctl.services.backends import detect_backends, missing_backends
from llmctl.services.benchmarks import BenchmarkService
from llmctl.services.events import list_events
from llmctl.services.health import HealthService
from llmctl.services.profiles import ProfileService
from llmctl.services.registry import RegistryService
from llmctl.services.sessions import SessionService
from llmctl.telemetry.gpu import get_gpu_info, nvml_available


@contextmanager
def db_session() -> Iterator[Session]:
    """Yield a short-lived database session with tables initialized."""
    settings = load_settings()
    init_db(settings.database_url)
    with Session(get_engine(settings.database_url)) as db:
        yield db


#: Maps a model runtime value to the backend key reported by ``detect_backends``.
_RUNTIME_TO_BACKEND = {"python_script": "python"}

#: Inline install hints surfaced when a backend binary is missing.
BACKEND_INSTALL_HINTS: dict[str, str] = {
    "vllm": "Install with: pip install vllm",
    "llama_cpp": "Build llama.cpp and put 'llama-server' on PATH.",
    "lmstudio": "Install LM Studio + CLI ('lms') and start its server.",
    "ollama": "Install Ollama (https://ollama.com) and run 'ollama serve'.",
    "python_script": "",
}

#: Copy-pasteable install commands, keyed by the backend name reported by
#: ``detect_backends`` (note: ``python`` resolves to the running interpreter and
#: never needs installing).
BACKEND_INSTALL_COMMANDS: dict[str, str] = {
    "vllm": "pip install vllm",
    "llama_cpp": "brew install llama.cpp",
    "lmstudio": "brew install --cask lm-studio",
    "ollama": "curl -fsSL https://ollama.com/install.sh | sh",
}

#: Maps the backend name from ``detect_backends`` to its model runtime value.
_BACKEND_TO_RUNTIME = {"python": "python_script"}


def install_command_for(backend: str) -> str:
    """Return a copy-pasteable install command for ``backend`` (or empty)."""
    return BACKEND_INSTALL_COMMANDS.get(backend, "")


def get_backend_map() -> dict[str, bool]:
    """Return a map of model runtime value -> backend-binary availability."""
    reverse = {v: k for k, v in _RUNTIME_TO_BACKEND.items()}
    result: dict[str, bool] = {}
    for entry in detect_backends():
        backend = str(entry["backend"])
        runtime = reverse.get(backend, backend)
        result[runtime] = bool(entry["available"])
    return result


def get_models() -> list[Model]:
    """Return all non-deleted registered/discovered models."""
    with db_session() as db:
        return RegistryService(db).list_models()


def scan_models() -> list[Model]:
    """Run adapter discovery and return the refreshed model list."""
    with db_session() as db:
        return RegistryService(db).scan()


def get_sessions() -> list[RuntimeSession]:
    """Return all sessions after reconciling any dead processes."""
    with db_session() as db:
        return SessionService(db).list_sessions()


def _build_request(
    db: Session,
    model_id: str,
    profile_name: str | None,
    gpu_mode: str,
    *,
    dry_run: bool,
    force: bool,
) -> SessionStartRequest:
    """Resolve model/profile and build a session start request for the TUI."""
    model = next(
        (m for m in RegistryService(db).list_models() if m.id == model_id),
        None,
    )
    if model is None:
        raise ValueError(f"Model not found: {model_id}")
    profile_id: str | None = None
    if profile_name:
        resolved = ProfileService(db).get_by_name(profile_name)
        if resolved is not None:
            profile_id = resolved.id
    mode = (gpu_mode or "auto").lower()
    return SessionStartRequest(
        model_id=model_id,
        profile_id=profile_id,
        runtime=model.runtime,
        gpu_mode=mode,
        gpus_auto=mode in {"auto", "balanced", "most-free", "least-used"},
        allow_cpu=mode == "cpu",
        force=force,
        dry_run=dry_run,
    )


def get_launch_plan(
    model_id: str, profile_name: str | None = None, gpu_mode: str = "auto"
) -> LaunchPlan:
    """Build an inspectable launch plan (with warnings/refusals) for preview."""
    with db_session() as db:
        request = _build_request(
            db, model_id, profile_name, gpu_mode, dry_run=True, force=False
        )
        return SessionService(db).plan(request)


def start_model(
    model_id: str,
    profile_name: str | None = None,
    gpu_mode: str = "auto",
    *,
    dry_run: bool = True,
    force: bool = True,
) -> RuntimeSession:
    """Plan (or launch) a session for ``model_id``.

    Defaults to a forced dry-run plan so the TUI records a planned session
    safely after the user has reviewed the launch plan; the scheduler still
    allocates ports/GPUs and records any warnings.
    """
    with db_session() as db:
        request = _build_request(
            db, model_id, profile_name, gpu_mode, dry_run=dry_run, force=force
        )
        return SessionService(db).start(request)


def stop_session(session_id: str) -> RuntimeSession | None:
    """Stop a session by id."""
    with db_session() as db:
        return SessionService(db).stop(session_id)


def restart_session(session_id: str) -> RuntimeSession | None:
    """Restart a session by id, reusing its stored launch plan."""
    with db_session() as db:
        return SessionService(db).restart(session_id)


def cleanup_sessions(*, remove_stale: bool = False) -> dict[str, Any]:
    """Reconcile dead sessions and optionally purge stale records."""
    with db_session() as db:
        return SessionService(db).cleanup(remove_stale=remove_stale)


def tail_log(session_id: str, lines: int = 200) -> str:
    """Return the tail of a session's log file (empty string when unavailable)."""
    with db_session() as db:
        content = SessionService(db).tail_log(session_id, lines=lines)
    return content or ""


def get_gpus() -> list[GPUInfo]:
    """Return GPU telemetry (empty list on non-NVIDIA hosts)."""
    return get_gpu_info()


def get_events(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent audit events as detached, render-ready dicts."""
    with db_session() as db:
        return [
            {
                "time": event.created_at.isoformat(timespec="seconds"),
                "level": event.level.value,
                "category": event.category,
                "message": event.message,
            }
            for event in list_events(db, limit=limit)
        ]


def get_backends() -> list[dict[str, Any]]:
    """Return backend binary availability info."""
    return detect_backends()


def get_doctor_summary() -> dict[str, Any]:
    """Return GPU/NVML status + scheduler config for the doctor screen."""
    settings = load_settings()
    gpus = get_gpu_info()
    sched = settings.scheduler
    return {
        "gpu_count": len(gpus),
        "nvml_available": nvml_available(),
        "safe_mode": settings.app.safe_mode,
        "gpu_policy": sched.gpu_policy,
        "safety_margin_gb": sched.safety_margin_gb,
        "allow_public_bind": sched.allow_public_bind,
        "default_host": sched.default_host,
        "missing_backends": missing_backends(settings),
    }


def get_benchmarks() -> list[BenchmarkResult]:
    """Return all recorded benchmark results (oldest first)."""
    with db_session() as db:
        return BenchmarkService(db).list_results()


def rerun_benchmark(benchmark_id: str) -> BenchmarkResult | None:
    """Re-run a stored benchmark, reusing its prompts/parameters."""
    with db_session() as db:
        return BenchmarkService(db).rerun(benchmark_id)


def run_benchmark(
    name: str,
    model_id: str | None = None,
    *,
    dry_run: bool = False,
) -> BenchmarkResult:
    """Run a new benchmark for a model (real streaming with mock fallback)."""
    with db_session() as db:
        return BenchmarkService(db).run(
            BenchmarkRunRequest(name=name, model_id=model_id, dry_run=dry_run)
        )


def get_overview() -> dict[str, Any]:
    """Return aggregate counts and health for the dashboard."""
    with db_session() as db:
        models = RegistryService(db).list_models()
        sessions = SessionService(db).list_sessions()
        profiles = ProfileService(db).list_profiles()
    gpus = get_gpu_info()
    health = HealthService().get_health()
    running = sum(1 for s in sessions if s.status.value == "running")
    planned = sum(1 for s in sessions if s.status.value == "planned")
    missing = missing_backends()
    warnings: list[str] = []
    for backend in missing:
        runtime_val = _BACKEND_TO_RUNTIME.get(backend, backend)
        affected = [m.name for m in models if m.runtime.value == runtime_val]
        if affected:
            shown = ", ".join(affected[:3])
            more = f" (+{len(affected) - 3} more)" if len(affected) > 3 else ""
            warnings.append(
                f"{backend} backend missing - affects {len(affected)} model(s): {shown}{more}"
            )
        else:
            warnings.append(f"{backend} backend missing - no registered models affected")
    if not gpus:
        warnings.append("No NVIDIA GPU detected; vLLM launches require --cpu or --force.")
    return {
        "models": len(models),
        "sessions_total": len(sessions),
        "sessions_running": running,
        "sessions_planned": planned,
        "profiles": len(profiles),
        "gpu_count": len(gpus),
        "nvml_available": nvml_available(),
        "safe_mode": bool(health.get("safe_mode")),
        "state": str(health.get("state")),
        "runtimes": health.get("runtimes", {}),
        "scheduler_warnings": warnings,
    }
