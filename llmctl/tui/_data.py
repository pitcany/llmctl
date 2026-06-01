"""Data-access helpers for the Textual TUI.

These thin functions open a short-lived database session, invoke the existing
service layer, and return plain schema objects (or simple dicts). Keeping all
data access here makes the screen widgets small, declarative, and easy to
unit-test without instantiating Textual widgets.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from sqlmodel import Session

from llmctl.config import load_settings
from llmctl.db import BenchmarkKind, get_engine, init_db
from llmctl.presets import (
    Model as PresetModel,
)
from llmctl.presets import (
    PresetSchemaError,
)
from llmctl.presets import (
    delete_preset as _delete_preset_files,
)
from llmctl.presets import (
    load_all_records as _load_preset_records,
)
from llmctl.presets import (
    save_preset as _save_preset,
)
from llmctl.schemas import (
    BenchmarkResult,
    BenchmarkRunRequest,
    GPUInfo,
    LaunchPlan,
    Model,
    ModelCreate,
    ModelUpdate,
    Profile,
    ProfileCreate,
    ProfileUpdate,
    SessionStartRequest,
)
from llmctl.schemas import Session as RuntimeSession
from llmctl.services.backends import detect_backends, missing_backends
from llmctl.services.benchmarks import BenchmarkService
from llmctl.services.events import list_events
from llmctl.services.gateway import GatewayService
from llmctl.services.health import HealthService
from llmctl.services.profiles import ProfileService
from llmctl.services.registry import RegistryService
from llmctl.services.sessions import SessionService
from llmctl.telemetry.gpu import get_gpu_info, nvml_available

# Each worker thread can open its own short-lived DB session, but only one
# thread should run init_db() at a time — concurrent CREATE TABLE on SQLite
# is not atomic (checkfirst inspects then creates, and a second thread can
# beat the create). The cache below remembers which URL has already been
# initialised so subsequent sessions skip the lock entirely.
_init_lock = threading.Lock()
_initialised_urls: set[str] = set()


@contextmanager
def db_session() -> Iterator[Session]:
    """Yield a short-lived database session with tables initialized."""
    settings = load_settings()
    url = settings.database_url
    if url not in _initialised_urls:
        with _init_lock:
            if url not in _initialised_urls:
                init_db(url)
                _initialised_urls.add(url)
    with Session(get_engine(url)) as db:
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


def add_model(payload: ModelCreate) -> Model:
    """Register a model from a TUI form."""
    with db_session() as db:
        return RegistryService(db).add_model(payload)


def update_model(model_id: str, updates: ModelUpdate) -> Model | None:
    """Apply a TUI edit to a model."""
    with db_session() as db:
        return RegistryService(db).update_model(model_id, updates)


def clone_model(model_id: str, new_name: str) -> Model | None:
    """Duplicate a model under a new name."""
    with db_session() as db:
        return RegistryService(db).clone_model(model_id, new_name)


def delete_model(model_id: str, *, delete_files: bool = False) -> bool:
    """Soft-delete a model; optionally remove the artifact."""
    with db_session() as db:
        return RegistryService(db).delete_model(model_id, delete_files=delete_files)


class PresetValidationError(ValueError):
    """Raised when the TUI submits a preset payload that fails the schema."""


def get_preset_views_with_links() -> list[Any]:
    """Return preset views enriched with registry linkage info."""
    from llmctl.services.preset_loader import load_preset_views

    with db_session() as db:
        models = RegistryService(db).list_models()
    return load_preset_views(models=models)


def get_preset_count_by_model() -> dict[str, int]:
    """Return ``{Model.id: preset_count}`` for the Models screen."""
    from llmctl.services.preset_loader import preset_count_by_model

    with db_session() as db:
        models = RegistryService(db).list_models()
    return preset_count_by_model(models=models)


def get_preset(alias: str) -> PresetModel | None:
    """Return the canonical preset for ``alias`` or None."""
    record = _load_preset_records().get(alias)
    return record.model if record else None


def save_preset(model: PresetModel) -> Path:
    """Persist a preset to the canonical llmctl directory.

    The schema validation already ran when the form built the ``Model``;
    this helper is a thin wrapper so the TUI worker has a single call
    site analogous to ``add_model`` / ``update_model``.
    """
    return _save_preset(model)


def add_preset(model: PresetModel) -> Path:
    """Create a new preset; alias must not already exist."""
    existing = _load_preset_records()
    if model.alias in existing:
        raise PresetValidationError(
            f"preset {model.alias!r} already exists at {existing[model.alias].source_path}"
        )
    return _save_preset(model)


def clone_preset(source_alias: str, new_alias: str) -> Path:
    """Duplicate a preset under a new alias."""
    records = _load_preset_records()
    src = records.get(source_alias)
    if src is None:
        raise PresetValidationError(f"preset {source_alias!r} not found")
    if new_alias in records:
        raise PresetValidationError(f"preset {new_alias!r} already exists")
    try:
        clone = src.model.model_copy(update={"alias": new_alias, "served_name": new_alias})
    except PresetSchemaError as exc:
        raise PresetValidationError(str(exc)) from exc
    return _save_preset(clone)


def delete_preset(alias: str) -> list[Path]:
    """Remove the preset's YAML file(s); returns the deleted paths."""
    return _delete_preset_files(alias)


def preset_source_path(alias: str) -> Path | None:
    """Return the on-disk path the loader currently resolves ``alias`` to."""
    record = _load_preset_records().get(alias)
    return record.source_path if record else None


def resolve_editor() -> list[str]:
    """Return the argv to invoke for ``$EDITOR``.

    Honours :envvar:`VISUAL` first, then :envvar:`EDITOR`, then falls
    back to ``vi``. Shell-style argument splitting matches what the
    shell would do with ``$EDITOR file`` so wrappers like
    ``code --wait`` work as expected.
    """
    raw = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    parts = shlex.split(raw)
    return parts or ["vi"]


def validate_preset_file(path: Path) -> PresetModel:
    """Re-read and validate a preset YAML, raising on schema errors.

    Used after the TUI shells out to ``$EDITOR``: the loader would
    silently drop a malformed file with a log warning, but the user
    needs an in-TUI notification instead.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise PresetSchemaError(f"{path}: top-level YAML must be a mapping")
    return PresetModel.model_validate(raw)


def run_editor_on_preset(
    path: Path,
    *,
    editor: list[str] | None = None,
) -> PresetModel:
    """Spawn ``$EDITOR`` synchronously on ``path`` and revalidate it.

    The caller is responsible for releasing the terminal first (Textual
    screens use :meth:`App.suspend`). On a non-zero exit code we still
    revalidate — editors like ``nvim`` return non-zero on ``:cq`` and
    users may legitimately want to discard, but the file on disk is
    the source of truth either way.
    """
    argv = editor or resolve_editor()
    subprocess.run([*argv, str(path)], check=False)
    return validate_preset_file(path)


def get_profiles() -> list[Profile]:
    """Return all profiles, syncing from YAML when the table is empty."""
    with db_session() as db:
        return ProfileService(db).list_profiles()


class ProfileValidationError(ValueError):
    """Raised when the TUI submits a profile payload that fails validation.

    Carries the structured ``issues`` list so the calling screen can format a
    user-facing notification with the offending fields. Matches the CLI/API
    posture: ``severity="error"`` blocks the write; warnings are surfaced
    separately but allowed through.
    """

    def __init__(self, message: str, issues: list[Any]) -> None:
        super().__init__(message)
        self.issues = issues


def _check_profile_validation(service: ProfileService, payload: Any) -> None:
    """Raise ProfileValidationError if the payload has any error-severity issues."""
    issues = service.validate(payload)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        summary = "; ".join(
            f"{issue.field or '?'}: {issue.message}" for issue in errors
        )
        raise ProfileValidationError(summary, issues)


def create_profile(payload: ProfileCreate) -> Profile:
    """Create a profile from a TUI form, validating first.

    Raises :class:`ProfileValidationError` on any error-severity issue so
    the TUI surfaces the same guard the CLI and API enforce. Warnings are
    not blocking and are not raised here.
    """
    with db_session() as db:
        service = ProfileService(db)
        _check_profile_validation(service, payload)
        return service.create_profile(payload)


def update_profile(profile_id: str, updates: ProfileUpdate) -> Profile | None:
    """Update a profile from a TUI form, validating first."""
    with db_session() as db:
        service = ProfileService(db)
        _check_profile_validation(service, updates)
        return service.update_profile(profile_id, updates)


def clone_profile(profile_id: str, new_name: str) -> Profile | None:
    """Duplicate a profile."""
    with db_session() as db:
        return ProfileService(db).clone_profile(profile_id, new_name)


def delete_profile(profile_id: str) -> bool:
    """Delete a profile."""
    with db_session() as db:
        return ProfileService(db).delete_profile(profile_id)


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


def get_benchmarks(*, model_id: str | None = None) -> list[BenchmarkResult]:
    """Return recorded benchmark results, optionally filtered by model.

    Results are returned in insertion order (oldest first) so the caller can
    reverse to render newest-on-top.
    """
    with db_session() as db:
        return BenchmarkService(db).list_results(model_id=model_id)


def rerun_benchmark(benchmark_id: str) -> BenchmarkResult | None:
    """Re-run a stored benchmark, reusing its prompts/parameters."""
    with db_session() as db:
        return BenchmarkService(db).rerun(benchmark_id)


def run_benchmark(
    name: str,
    model_id: str | None = None,
    *,
    kind: BenchmarkKind = BenchmarkKind.CHAT,
    context_length: int | None = None,
    profile_id: str | None = None,
    max_tokens: int | None = None,
    dry_run: bool = False,
) -> BenchmarkResult:
    """Run a new benchmark for a model.

    Live failures persist as ``success=False`` records; the caller can
    surface ``result.error`` in the UI to point at what broke.
    """
    parameters: dict[str, object] = {}
    if max_tokens is not None:
        parameters["max_tokens"] = max_tokens
    with db_session() as db:
        return BenchmarkService(db).run(
            BenchmarkRunRequest(
                name=name,
                model_id=model_id,
                profile_id=profile_id,
                kind=kind,
                context_length=context_length,
                parameters=parameters,
                dry_run=dry_run,
            )
        )


def get_overview() -> dict[str, Any]:
    """Return aggregate counts and health for the dashboard."""
    settings = load_settings()
    with db_session() as db:
        models = RegistryService(db).list_models()
        sessions = SessionService(db).list_sessions()
        profiles = ProfileService(db).list_profiles()
        aliases = GatewayService(db, settings).alias_view()
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
    router_running = _probe_gateway(settings)
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
        "router": {
            "running": router_running,
            "host": settings.router.host,
            "port": settings.router.port,
            "auth_required": bool(settings.router.auth_token),
            "aliases": [
                {
                    "name": a.name,
                    "target": a.target,
                    "session_id": a.resolved_session_id,
                    "healthy": a.healthy,
                }
                for a in aliases
            ],
        },
    }


def _probe_gateway(settings: Any) -> bool:
    """Best-effort liveness probe of the gateway /health endpoint."""
    import httpx

    host = settings.router.host if settings.router.host not in ("0.0.0.0", "") else "127.0.0.1"
    url = f"http://{host}:{settings.router.port}/health"
    headers = {}
    if settings.router.auth_token:
        headers["Authorization"] = f"Bearer {settings.router.auth_token}"
    try:
        return httpx.get(url, headers=headers, timeout=0.5).status_code == 200
    except httpx.HTTPError:
        return False
