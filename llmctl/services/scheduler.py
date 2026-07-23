"""Launch scheduler service.

Builds inspectable, explainable :class:`LaunchPlan` objects: GPU placement
(explicit / auto / balanced / most-free / least-used / cpu), VRAM-aware
admission control, free-port allocation, command vectors, and a list of
non-fatal ``warnings`` and hard ``refusal_reasons``. The scheduler is the single
place that decides *how* a runtime is invoked; adapters merely execute the plan.

Refusals are recorded on the plan and enforced by :meth:`SchedulerService.validate`
(raising :class:`SchedulerError`) so the same plan can be previewed by ``llmctl
plan`` or the TUI without launching anything.
"""

from __future__ import annotations

import shutil
import socket
import sys
from pathlib import Path

from sqlmodel import Session as DBSession
from sqlmodel import select

from llmctl.config import RuntimeConfig, Settings, load_settings
from llmctl.db import (
    ModelRecord,
    ProfileRecord,
    RuntimeName,
    SessionRecord,
    SessionStatus,
)
from llmctl.schemas import GPUInfo, LaunchPlan, SessionStartRequest
from llmctl.telemetry.gpu import get_gpu_info

HTTP_SERVER_RUNTIMES = {RuntimeName.OLLAMA, RuntimeName.LMSTUDIO}
GPU_REQUIRED_RUNTIMES = {RuntimeName.VLLM}
LOCAL_FILE_RUNTIMES = {RuntimeName.LLAMA_CPP, RuntimeName.PYTHON_SCRIPT}
VALID_GPU_MODES = {"auto", "balanced", "most-free", "least-used"}
_ACTIVE_STATES = {SessionStatus.RUNNING, SessionStatus.STARTING, SessionStatus.DEGRADED}


class SchedulerError(ValueError):
    """Raised when a launch is refused and not overridden with ``--force``."""


class SchedulerService:
    """Builds safe, explainable launch plans and GPU placement decisions."""

    def __init__(self, db: DBSession | None = None, settings: Settings | None = None) -> None:
        self.db = db
        self.settings = settings or load_settings()

    # -- public API ---------------------------------------------------------

    def create_launch_plan(self, request: SessionStartRequest) -> LaunchPlan:
        """Build a fully populated, inspectable launch plan. Never launches."""
        runtime = request.runtime
        runtime_config = self.settings.runtime_config(runtime.value)
        model = self._get_model(request.model_id)
        profile = self._get_profile(request.profile_id)

        # Promoted knobs (tp, quantization, ...) may live in typed columns
        # (profile create) or the parameters dict (YAML import); honour both.
        from llmctl.services.profiles import effective_parameters

        parameters = effective_parameters(profile) if profile else {}
        parameters.update(request.parameters)
        tp = max(1, int(parameters.get("tensor_parallel_size", 1) or 1))

        warnings: list[str] = []
        refusals: list[str] = []
        notes: list[str] = []

        if runtime == RuntimeName.OPENAI:
            # Adopt-only: there is no local process to plan. Recorded as a
            # refusal (not a raise) so `llmctl plan`/`preview` and
            # POST /sessions/plan preview it gracefully like every other
            # refusal; only validate() raises.
            refusals.append(
                "Runtime 'openai' is adopt-only (an external OpenAI-compatible "
                "endpoint). Use `llmctl adopt --endpoint <url> --runtime openai` "
                "instead of start."
            )

        if profile is not None and profile.runtime != runtime:
            refusals.append(
                f"Profile '{profile.name}' targets {profile.runtime.value}, "
                f"incompatible with backend {runtime.value}."
            )

        mode = self._effective_mode(request)
        gpus_info = get_gpu_info()
        gpu_ids = self._select_gpus(runtime, request, mode, tp, gpus_info, refusals)

        env = dict(runtime_config.env)
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)
        elif request.allow_cpu and runtime in GPU_REQUIRED_RUNTIMES:
            env["CUDA_VISIBLE_DEVICES"] = ""

        if gpu_ids and tp > len(gpu_ids):
            refusals.append(
                f"Requested tensor_parallel_size={tp} exceeds selected GPU count {len(gpu_ids)}."
            )

        estimated = self._estimated_vram(model, parameters)
        free_vram_gb = self._free_vram_gb(gpu_ids, gpus_info)
        self._check_vram(estimated, gpu_ids, gpus_info, warnings, refusals)

        host = runtime_config.host or self.settings.scheduler.default_host
        if host not in ("127.0.0.1", "localhost") and not self.settings.scheduler.allow_public_bind:
            warnings.append(
                f"Binding to public host {host}; set scheduler.allow_public_bind=true to allow."
            )

        port: int | None = None
        health_url: str | None = None
        command: list[str] = []
        if runtime in HTTP_SERVER_RUNTIMES:
            endpoint = runtime_config.endpoint
            notes.append(f"{runtime.value} is server-managed; attaches to {endpoint}.")
        else:
            port = self._allocate_port(runtime, runtime_config)
            if port is None:
                refusals.append("No free port available in the configured range.")
                fallback = self._port_range(runtime, runtime_config)
                port = fallback[0]
            endpoint = f"http://{host}:{port}"
            health_url = f"{endpoint}/v1/models"
            command = self._build_command(runtime, runtime_config, model, parameters, host, port)
            if not command:
                warnings.append("No command could be built; verify the model path/source.")
            self._check_binary(runtime_config, refusals)
            self._check_model_path(runtime, model, parameters, refusals)

        safety_checks: list[str] = []
        if request.dry_run:
            safety_checks.append("dry_run_no_process_launch")
        if self.settings.app.safe_mode:
            safety_checks.append("safe_mode_enabled")
        if request.allow_cpu:
            notes.append("CPU mode requested; GPUs hidden from the runtime.")

        return LaunchPlan(
            runtime=runtime,
            model_id=request.model_id,
            model_name=model.name if model else None,
            profile_id=request.profile_id,
            profile_name=profile.name if profile else None,
            command=command,
            env=env,
            gpu_ids=gpu_ids,
            gpu_selection_mode=mode,
            tensor_parallel_size=tp,
            port=port,
            endpoint_url=endpoint,
            health_url=health_url,
            estimated_vram_gb=estimated,
            free_vram_gb=free_vram_gb,
            dry_run=request.dry_run,
            safety_checks=safety_checks,
            warnings=warnings,
            refusal_reasons=refusals,
            notes=notes,
        )

    def validate(self, plan: LaunchPlan, *, force: bool, dry_run: bool) -> None:
        """Raise :class:`SchedulerError` if ``plan`` is refused and not overridden.

        A dry-run (planning) request and ``--force`` both bypass enforcement;
        refusal reasons remain on the plan for inspection either way.
        """
        if plan.refusal_reasons and not force and not dry_run:
            joined = "; ".join(plan.refusal_reasons)
            raise SchedulerError(f"Refusing to launch: {joined} (use --force to override).")

    # -- model / profile lookups -------------------------------------------

    def _get_model(self, model_id: str | None) -> ModelRecord | None:
        """Return the model record for ``model_id`` when a DB is available."""
        if not model_id or self.db is None:
            return None
        return self.db.get(ModelRecord, model_id)

    def _get_profile(self, profile_id: str | None) -> ProfileRecord | None:
        """Return the profile record for ``profile_id`` when a DB is available."""
        if not profile_id or self.db is None:
            return None
        return self.db.get(ProfileRecord, profile_id)

    # -- GPU placement ------------------------------------------------------

    def _effective_mode(self, request: SessionStartRequest) -> str:
        """Resolve the effective GPU selection mode for a request."""
        if request.allow_cpu:
            return "cpu"
        if request.gpu_ids:
            return "explicit"
        mode = (request.gpu_mode or "").strip().lower()
        if mode in VALID_GPU_MODES:
            return mode
        if request.gpus_auto:
            return "auto"
        return self.settings.scheduler.gpu_policy or "most-free"

    def _select_gpus(
        self,
        runtime: RuntimeName,
        request: SessionStartRequest,
        mode: str,
        tp: int,
        gpus_info: list[GPUInfo],
        refusals: list[str],
    ) -> list[int]:
        """Select GPU indices honoring explicit ids, cpu, and ranking modes."""
        if mode == "cpu" or request.allow_cpu:
            return []
        if request.gpu_ids:
            return list(request.gpu_ids)

        gpu_required = runtime in GPU_REQUIRED_RUNTIMES
        if not gpus_info:
            if gpu_required:
                refusals.append(
                    "No NVIDIA GPUs detected. Use --cpu to run on CPU or --force to override."
                )
            return []

        ranked = self._rank_gpus(mode, gpus_info)
        return [gpu.index for gpu in ranked[:tp]]

    def _rank_gpus(self, mode: str, gpus_info: list[GPUInfo]) -> list[GPUInfo]:
        """Rank GPUs according to the selection ``mode``."""
        if mode == "least-used":
            return sorted(
                gpus_info,
                key=lambda g: (g.utilization_gpu_percent or 0, -(g.memory_free_mb or 0)),
            )
        if mode == "balanced":
            counts = self._gpu_session_counts()
            return sorted(
                gpus_info,
                key=lambda g: (counts.get(g.index, 0), -(g.memory_free_mb or 0)),
            )
        # "auto" and "most-free" both prefer the most free VRAM.
        return sorted(gpus_info, key=lambda g: g.memory_free_mb or 0, reverse=True)

    def _gpu_session_counts(self) -> dict[int, int]:
        """Return the number of active sessions placed on each GPU index."""
        counts: dict[int, int] = {}
        if self.db is None:
            return counts
        records = self.db.exec(
            select(SessionRecord).where(SessionRecord.status.in_(_ACTIVE_STATES))  # type: ignore[attr-defined]
        ).all()
        for record in records:
            for gpu in record.gpu_ids or []:
                counts[gpu] = counts.get(gpu, 0) + 1
        return counts

    # -- VRAM admission control --------------------------------------------

    @staticmethod
    def _estimated_vram(model: ModelRecord | None, parameters: dict[str, object]) -> float | None:
        """Return estimated VRAM (GB) from profile hint or model metadata."""
        hint = parameters.get("estimated_vram_gb")
        if hint is not None:
            try:
                return float(hint)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
        if model is not None and model.estimated_vram_gb is not None:
            return float(model.estimated_vram_gb)
        return None

    @staticmethod
    def _free_vram_gb(gpu_ids: list[int], gpus_info: list[GPUInfo]) -> float | None:
        """Return total free VRAM (GB) across selected GPUs (or all when none)."""
        if not gpus_info:
            return None
        index = {g.index: (g.memory_free_mb or 0) for g in gpus_info}
        targets = gpu_ids or list(index.keys())
        total_mb = sum(index.get(idx, 0) for idx in targets)
        return round(total_mb / 1024, 2)

    def _check_vram(
        self,
        estimated: float | None,
        gpu_ids: list[int],
        gpus_info: list[GPUInfo],
        warnings: list[str],
        refusals: list[str],
    ) -> None:
        """Compare estimated VRAM against per-GPU free VRAM with a safety margin."""
        if estimated is None:
            warnings.append(
                "Estimated VRAM unknown; cannot verify the model fits. "
                "Set estimated_vram_gb on the model/profile, or pass --force."
            )
            return
        if not gpu_ids or not gpus_info:
            return
        margin = self.settings.scheduler.safety_margin_gb
        free_map = {g.index: (g.memory_free_mb or 0) / 1024 for g in gpus_info}
        per_gpu_need = estimated / max(1, len(gpu_ids))
        for idx in gpu_ids:
            free = free_map.get(idx, 0.0)
            if per_gpu_need + margin > free:
                refusals.append(
                    f"Estimated {per_gpu_need:.1f} GB on GPU {idx} exceeds free "
                    f"{free:.1f} GB (safety margin {margin:.1f} GB)."
                )

    # -- safety checks ------------------------------------------------------

    @staticmethod
    def _check_binary(runtime_config: RuntimeConfig, refusals: list[str]) -> None:
        """Refuse when a configured backend binary is missing from PATH."""
        binary = runtime_config.binary
        if binary and shutil.which(binary) is None:
            refusals.append(f"Backend binary '{binary}' not found on PATH.")

    @staticmethod
    def _check_model_path(
        runtime: RuntimeName,
        model: ModelRecord | None,
        parameters: dict[str, object],
        refusals: list[str],
    ) -> None:
        """Refuse when a local-file runtime points at a missing path."""
        if runtime not in LOCAL_FILE_RUNTIMES:
            return
        if runtime == RuntimeName.PYTHON_SCRIPT:
            target = parameters.get("script")
            if not target and model is not None:
                target = model.path or model.source
        else:
            target = (model.path or model.source) if model else None
        if not target:
            refusals.append(f"No model path/source configured for {runtime.value}.")
            return
        if not Path(str(target)).exists():
            refusals.append(f"Model path does not exist: {target}")

    # -- command building ---------------------------------------------------

    def _build_command(
        self,
        runtime: RuntimeName,
        runtime_config: RuntimeConfig,
        model: ModelRecord | None,
        parameters: dict[str, object],
        host: str,
        port: int,
    ) -> list[str]:
        """Build the argv for a process-launch runtime, or ``[]`` if not possible."""
        if runtime == RuntimeName.VLLM:
            return self._build_vllm_command(runtime_config, model, parameters, host, port)
        if runtime == RuntimeName.LLAMA_CPP:
            return self._build_llama_cpp_command(runtime_config, model, parameters, host, port)
        if runtime == RuntimeName.PYTHON_SCRIPT:
            return self._build_python_command(model, parameters)
        return []

    def _build_vllm_command(
        self,
        runtime_config: RuntimeConfig,
        model: ModelRecord | None,
        parameters: dict[str, object],
        host: str,
        port: int,
    ) -> list[str]:
        """Build a ``vllm serve`` command."""
        model_ref = self._model_reference(model)
        if not model_ref:
            return []
        binary = runtime_config.binary or "vllm"
        command = [binary, "serve", model_ref, "--host", host, "--port", str(port)]
        if "tensor_parallel_size" in parameters:
            command += ["--tensor-parallel-size", str(parameters["tensor_parallel_size"])]
        if "gpu_memory_utilization" in parameters:
            command += ["--gpu-memory-utilization", str(parameters["gpu_memory_utilization"])]
        if "max_model_len" in parameters:
            command += ["--max-model-len", str(parameters["max_model_len"])]
        if parameters.get("dtype"):
            command += ["--dtype", str(parameters["dtype"])]
        if parameters.get("quantization"):
            command += ["--quantization", str(parameters["quantization"])]
        if parameters.get("served_model_name"):
            command += ["--served-model-name", str(parameters["served_model_name"])]
        command += self._extra_args(parameters, runtime_config)
        return command

    def _build_llama_cpp_command(
        self,
        runtime_config: RuntimeConfig,
        model: ModelRecord | None,
        parameters: dict[str, object],
        host: str,
        port: int,
    ) -> list[str]:
        """Build a ``llama-server`` command."""
        model_path = (model.path or model.source) if model else None
        if not model_path:
            return []
        binary = runtime_config.binary or "llama-server"
        command = [binary, "-m", model_path, "--host", host, "--port", str(port)]
        if "n_gpu_layers" in parameters:
            command += ["-ngl", str(parameters["n_gpu_layers"])]
        if "ctx_size" in parameters:
            command += ["-c", str(parameters["ctx_size"])]
        command += self._extra_args(parameters, runtime_config)
        return command

    def _build_python_command(
        self,
        model: ModelRecord | None,
        parameters: dict[str, object],
    ) -> list[str]:
        """Build a ``python <script> [args...]`` command."""
        script = parameters.get("script")
        if not script and model:
            script = model.path or model.source
        if not script:
            return []
        command = [sys.executable, str(script)]
        args = parameters.get("args")
        if isinstance(args, list):
            command += [str(arg) for arg in args]
        return command

    @staticmethod
    def _extra_args(parameters: dict[str, object], runtime_config: RuntimeConfig) -> list[str]:
        """Return combined extra args from the profile and runtime config."""
        extra: list[str] = []
        profile_extra = parameters.get("extra_args")
        if isinstance(profile_extra, list):
            extra += [str(arg) for arg in profile_extra]
        extra += list(runtime_config.extra_args)
        return extra

    @staticmethod
    def _model_reference(model: ModelRecord | None) -> str | None:
        """Return the model identifier vLLM should serve."""
        if model is None:
            return None
        return model.source or model.path or model.name

    # -- port allocation ----------------------------------------------------

    def _port_range(self, runtime: RuntimeName, runtime_config: RuntimeConfig) -> list[int]:
        """Return the effective [start, end] port range for a runtime."""
        configured = self.settings.scheduler.port_ranges.get(runtime.value)
        if configured:
            return configured
        return runtime_config.port_range or [8000, 8099]

    def _allocate_port(self, runtime: RuntimeName, runtime_config: RuntimeConfig) -> int | None:
        """Return a free TCP port within the runtime's range, or ``None``."""
        port_range = self._port_range(runtime, runtime_config)
        start = port_range[0]
        end = port_range[1] if len(port_range) > 1 else port_range[0]
        used = self._used_ports()
        host = runtime_config.host or self.settings.scheduler.default_host
        for port in range(start, end + 1):
            if port in used:
                continue
            if self._is_port_free(host, port):
                return port
        return None

    def _used_ports(self) -> set[int]:
        """Return ports already reserved by active sessions in the database."""
        if self.db is None:
            return set()
        ports: set[int] = set()
        records = self.db.exec(
            select(SessionRecord).where(SessionRecord.status.in_(_ACTIVE_STATES))  # type: ignore[attr-defined]
        ).all()
        for record in records:
            if record.port:
                ports.add(record.port)
            elif record.endpoint_url and ":" in record.endpoint_url:
                try:
                    ports.add(int(record.endpoint_url.rsplit(":", 1)[1]))
                except ValueError:
                    continue
        return ports

    @staticmethod
    def _is_port_free(host: str, port: int) -> bool:
        """Return True when ``port`` can be bound on ``host``."""
        bind_host = "0.0.0.0" if host in ("0.0.0.0", "") else host
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
                return True
            except OSError:
                return False
