"""Pydantic schemas shared by API, CLI, TUI, and services."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from llmctl.db import BenchmarkKind, ModelStatus, RuntimeName, SessionKind, SessionStatus


class HealthState(StrEnum):
    """Common health state labels."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    NOT_USED = "not_used"


class Model(BaseModel):
    """Model registry schema."""

    id: str | None = None
    name: str
    runtime: RuntimeName
    source: str | None = None
    path: str | None = None
    host: str | None = None
    format: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    estimated_vram_gb: float | None = None
    max_context: int | None = None
    parameter_count: int | None = None
    notes: str | None = None
    default_profile_id: str | None = None
    active: bool = True
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: ModelStatus = ModelStatus.REGISTERED
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ModelCreate(BaseModel):
    """Payload to register a model."""

    name: str
    runtime: RuntimeName
    source: str | None = None
    path: str | None = None
    host: str | None = None
    format: str | None = None
    quantization: str | None = None
    estimated_vram_gb: float | None = None
    max_context: int | None = None
    parameter_count: int | None = None
    notes: str | None = None
    default_profile_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelUpdate(BaseModel):
    """Partial-update payload for a model. All fields optional."""

    name: str | None = None
    runtime: RuntimeName | None = None
    source: str | None = None
    path: str | None = None
    host: str | None = None
    format: str | None = None
    quantization: str | None = None
    estimated_vram_gb: float | None = None
    max_context: int | None = None
    parameter_count: int | None = None
    notes: str | None = None
    default_profile_id: str | None = None
    active: bool | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class Profile(BaseModel):
    """Reusable launch profile schema."""

    id: str | None = None
    name: str
    runtime: RuntimeName
    description: str | None = None
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    quantization: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    environment_variables: dict[str, str] = Field(default_factory=dict)
    scheduler_preferences: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    gpu_policy: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)


class ProfileCreate(BaseModel):
    """Payload to create a profile."""

    name: str
    runtime: RuntimeName
    description: str | None = None
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    quantization: str | None = None
    extra_args: list[str] = Field(default_factory=list)
    environment_variables: dict[str, str] = Field(default_factory=dict)
    scheduler_preferences: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    gpu_policy: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)


class ProfileUpdate(BaseModel):
    """Partial-update payload for a profile. All fields optional."""

    name: str | None = None
    runtime: RuntimeName | None = None
    description: str | None = None
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    quantization: str | None = None
    extra_args: list[str] | None = None
    environment_variables: dict[str, str] | None = None
    scheduler_preferences: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    gpu_policy: dict[str, Any] | None = None
    safety: dict[str, Any] | None = None


class ValidationIssue(BaseModel):
    """Single validation warning or error raised against a profile/model."""

    severity: str = "warning"
    field: str | None = None
    message: str


class RegistryExport(BaseModel):
    """Portable export bundle for the model registry.

    Used by ``llmctl export-registry`` / ``import-registry``. ``version`` lets
    importers recognise legacy bundles and ``settings`` is reserved for future
    feature toggles (currently empty).
    """

    version: int = 1
    models: list[Model] = Field(default_factory=list)
    profiles: list[Profile] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class GPUInfo(BaseModel):
    """NVIDIA GPU telemetry snapshot."""

    index: int
    uuid: str | None = None
    name: str
    driver_version: str | None = None
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    memory_free_mb: int | None = None
    utilization_gpu_percent: int | None = None
    utilization_memory_percent: int | None = None
    temperature_c: int | None = None
    power_draw_watts: float | None = None
    processes: list[dict[str, Any]] = Field(default_factory=list)


class LaunchPlan(BaseModel):
    """Dry-run or executable launch plan.

    The plan is the single inspectable, explainable record of *how* a runtime
    would be launched: selected GPUs/port, estimated vs. free VRAM, the command
    preview, plus any non-fatal ``warnings`` and hard ``refusal_reasons``.
    """

    runtime: RuntimeName
    model_id: str | None = None
    model_name: str | None = None
    profile_id: str | None = None
    profile_name: str | None = None
    command: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    gpu_ids: list[int] = Field(default_factory=list)
    gpu_selection_mode: str = "auto"
    tensor_parallel_size: int = 1
    port: int | None = None
    endpoint_url: str | None = None
    health_url: str | None = None
    estimated_vram_gb: float | None = None
    free_vram_gb: float | None = None
    log_name: str | None = None
    dry_run: bool = True
    safety_checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    refusal_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def command_preview(self) -> str:
        """Return a shell-style preview of the launch command."""
        return " ".join(self.command) if self.command else "(server-managed; no command)"


class StatusChange(BaseModel):
    """One move in a session's lifecycle trail: when, from what, to what, and why."""

    at: datetime
    old: SessionStatus
    new: SessionStatus
    why: str


class Session(BaseModel):
    """Runtime session schema."""

    id: str | None = None
    model_id: str | None = None
    profile_id: str | None = None
    runtime: RuntimeName
    status: SessionStatus = SessionStatus.PLANNED
    kind: SessionKind = SessionKind.OWNED
    pid: int | None = None
    port: int | None = None
    endpoint_url: str | None = None
    log_path: str | None = None
    gpu_ids: list[int] = Field(default_factory=list)
    launch_plan: LaunchPlan | None = None
    error: str | None = None
    systemd_unit: str | None = None
    served_name: str | None = None
    adopted_at: datetime | None = None
    #: Set when a stop/restart attempt left the process alive. Separate from
    #: ``status`` on purpose — see ``db.SessionRecord.terminate_failed_at``.
    terminate_failed_at: datetime | None = None
    #: The lifecycle trail, oldest first; ``llmctl history`` and
    #: ``GET /sessions/{id}/history`` show it.
    status_history: list[StatusChange] = Field(default_factory=list)
    created_at: datetime | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None


class SessionStartRequest(BaseModel):
    """Payload to request session start (planning or real launch)."""

    model_id: str
    profile_id: str | None = None
    runtime: RuntimeName
    gpu_ids: list[int] = Field(default_factory=list)
    gpu_mode: str = "auto"
    gpus_auto: bool = False
    allow_cpu: bool = False
    force: bool = False
    dry_run: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Benchmark result schema."""

    id: str | None = None
    model_id: str | None = None
    session_id: str | None = None
    profile_id: str | None = None
    name: str
    kind: BenchmarkKind | None = None
    backend: str | None = None
    context_length: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    tokens_per_second: float | None = None
    time_to_first_token_ms: float | None = None
    peak_vram_mb: int | None = None
    avg_gpu_util_pct: float | None = None
    max_gpu_util_pct: float | None = None
    gpu_snapshot: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    samples: list[dict[str, Any]] = Field(default_factory=list)
    #: Immutable experiment manifest (empty dict on legacy rows -- unknown
    #: provenance, not fabricated). See ``services/benchmark_manifest.py``.
    manifest: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: str | None = None
    created_at: datetime | None = None


class BenchmarkRunRequest(BaseModel):
    """Payload to request a benchmark run.

    Execution-mode contract (FIXPLAN 2026-11-03 §1A): ordinary runs are
    strict-live. A request that resolves no endpoint records a *failure*,
    never a fabricated synthetic measurement. Synthetic execution is the
    explicit ``dry_run=True`` path and is labeled ``mode=mock`` in the
    record. ``require_live=False`` is a legacy escape hatch that restores
    the old mock fallback for scripted callers; new callers should not
    use it.
    """

    model_id: str | None = None
    session_id: str | None = None
    profile_id: str | None = None
    name: str = "smoke"
    kind: BenchmarkKind = BenchmarkKind.CHAT
    context_length: int | None = None
    prompts: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    #: Maximum number of simultaneous in-flight requests (a ceiling, not a
    #: guarantee: a workload smaller than ``concurrency`` cannot fill it and
    #: the record flags the shortfall via ``achieved_concurrency``).
    concurrency: int = Field(default=1, ge=1)
    #: Workload repetition: each prompt is sent ``repetitions`` times, so
    #: the total workload is ``len(prompts) * repetitions`` requests. This
    #: is independent of ``concurrency`` on purpose -- a concurrency sweep
    #: must not silently change the total workload.
    repetitions: int = Field(default=1, ge=1)
    #: Warm-up requests executed before measurement; excluded from samples,
    #: timings and throughput. Warms caches/compilation so the recorded
    #: measurements describe steady state.
    warmup: int = Field(default=0, ge=0)
    sweep: list[int] = Field(default_factory=list)
    dry_run: bool = False
    #: Default True: refuse to mock-fallback when no endpoint resolves.
    #: Explicit ``False`` restores the deprecated lenient behavior (mock
    #: on endpoint miss) for compatibility with old scripted callers.
    require_live: bool = True

    @field_validator("sweep")
    @classmethod
    def _sweep_levels_positive(cls, value: list[int]) -> list[int]:
        """Concurrency sweep levels must be >= 1."""
        if any(level < 1 for level in value):
            raise ValueError("sweep levels must be >= 1")
        return value


class AdapterStatus(BaseModel):
    """Runtime adapter status response."""

    runtime: RuntimeName
    state: HealthState = HealthState.UNKNOWN
    message: str = "Adapter not implemented yet."
    details: dict[str, Any] = Field(default_factory=dict)
