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

import re
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
GPU_REQUIRED_RUNTIMES = {
    RuntimeName.VLLM,
    RuntimeName.SGLANG,
    RuntimeName.FREETOKEN,
    RuntimeName.STRATA,
}
#: Engines that serve from one GPU only.
SINGLE_GPU_RUNTIMES = {RuntimeName.FREETOKEN, RuntimeName.STRATA}
LOCAL_FILE_RUNTIMES = {RuntimeName.LLAMA_CPP, RuntimeName.PYTHON_SCRIPT, RuntimeName.STRATA}
VALID_GPU_MODES = {"auto", "balanced", "most-free", "least-used"}
_ACTIVE_STATES = {SessionStatus.RUNNING, SessionStatus.STARTING, SessionStatus.DEGRADED}

#: llama.cpp split-GGUF naming: ``<name>-00001-of-00004.gguf``. Only the first
#: shard is passed to the server; the rest are opened implicitly deep into
#: loading, so a missing tail shard costs ~2 minutes before it fails.
_SHARDED_GGUF_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$")


def _forced_busy_port_warning(port: int) -> str:
    return (
        f"--force: launching on port {port}, which is IN USE. The server will "
        "fail to bind, or collide with whatever already listens there."
    )


class SchedulerError(ValueError):
    """Raised when a launch is refused and not overridden with ``--force``."""


def _strata_server_for(python: str) -> Path | None:
    """``<repo>/serve/server.py`` for ``<repo>/.venv/bin/python``, if it is there."""
    candidate = Path(python).resolve().parent.parent.parent / "serve" / "server.py"
    return candidate if candidate.is_file() else None


def _python_has_module(python: str, module: str) -> bool:
    """Whether ``python`` can import ``module`` (checked without importing it).

    ``find_spec`` keeps this to a few milliseconds; importing sglang itself
    would pull in torch and take seconds at plan time.
    """
    import subprocess

    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, interpreter from settings
            [
                python,
                "-c",
                f"import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)",
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def python_script_refusal(script: str) -> str | None:
    """Return why ``script`` may not be run as a python script, or ``None``.

    ``runtime=python_script`` builds ``[sys.executable, script, *args]``. A
    script path beginning with ``-`` reaches the interpreter as a *flag*:
    ``script="-c"`` turns ``args[0]`` into executable code, which is remote code
    execution once the control-plane API is reachable.

    This is a hard stop rather than an ordinary refusal reason, because
    ``force=true`` on the start request makes :meth:`SchedulerService.validate`
    skip every refusal. It is enforced by *refusing to build a command at all*:
    an empty ``plan.command`` fails closed at launch, and ``force`` cannot
    conjure one.

    Existence is deliberately not checked. Without a leading ``-`` the
    interpreter reads the argument as a path, so a missing file is a failed
    launch rather than code execution — and requiring existence here would break
    ``llmctl plan``, which previews a command before the script is necessarily
    in place. Non-existence is already surfaced as a plan refusal.
    """
    if script.startswith("-"):
        return (
            f"python_script refuses {script!r}: a script path cannot start with "
            "'-'; the interpreter would read it as a flag, making the arguments "
            "executable code."
        )
    return None


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

        # Runtime env first, then the profile's on top: `runtimes.<name>.env` is
        # shared by every model on that runtime, so per-model settings are the
        # more specific of the two and win. This field has existed on Profile
        # since the schema was written but was never read here, which made
        # anything set on a profile silently inert.
        env = dict(runtime_config.env)
        if profile is not None:
            # `or {}`: this is a ProfileRecord, whose JSON column can be NULL.
            profile_env = dict(profile.environment_variables or {})
            # GPU visibility is the scheduler's decision, never a profile's.
            # Dropped before the merge rather than overwritten after, because
            # the overwrite only happens when a GPU was selected -- a forced
            # launch with no GPUs would otherwise inherit the profile's pin and
            # see a card the plan says it is not using.
            profile_env.pop("CUDA_VISIBLE_DEVICES", None)
            env.update(profile_env)
        if gpu_ids and runtime == RuntimeName.STRATA:
            # Strata picks its card by nvidia-smi index (`--gpu`), which
            # CUDA_VISIBLE_DEVICES would renumber underneath it.
            parameters.setdefault("strata_gpu", gpu_ids[0])
        elif gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpu_ids)
        elif request.allow_cpu and runtime in GPU_REQUIRED_RUNTIMES:
            env["CUDA_VISIBLE_DEVICES"] = ""

        if gpu_ids and tp > len(gpu_ids):
            refusals.append(
                f"Requested tensor_parallel_size={tp} exceeds selected GPU count {len(gpu_ids)}."
            )
        if runtime in SINGLE_GPU_RUNTIMES and tp > 1:
            refusals.append(
                f"{runtime.value} serves from one GPU; tensor_parallel_size={tp} is not possible."
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
            pinned = self._pinned_port(parameters)
            if pinned is not None:
                # An explicit pin is honoured verbatim, including outside the
                # configured range: explicit beats implicit. If it is taken we
                # refuse rather than relocate -- silently moving the endpoint is
                # exactly what a pin exists to prevent, and it would leave the
                # recorded port pointing somewhere the server is not listening.
                port = pinned
                if not self._port_available(runtime_config, port):
                    refusals.append(
                        f"Pinned port {port} is already in use; refusing to "
                        "relocate it. Stop whatever holds it, or unset "
                        "`parameters.port` to let the scheduler choose."
                    )
                    if request.force:
                        warnings.append(_forced_busy_port_warning(port))
            else:
                port = self._allocate_port(runtime, runtime_config)
                if port is None:
                    fallback = self._port_range(runtime, runtime_config)
                    refusals.append(
                        "No free port available in the configured range "
                        f"{fallback[0]}-{fallback[-1]}."
                    )
                    # Only --force gets past the refusal, and it must not do so
                    # quietly: this port is known to be taken.
                    port = fallback[0]
                    if request.force:
                        warnings.append(_forced_busy_port_warning(port))
            endpoint = f"http://{host}:{port}"
            health_url = f"{endpoint}/v1/models"
            command = self._build_command(runtime, runtime_config, model, parameters, host, port)
            if not command:
                warnings.append("No command could be built; verify the model path/source.")
            if runtime == RuntimeName.PYTHON_SCRIPT:
                script = parameters.get("script") or (model.path or model.source if model else None)
                reason = python_script_refusal(str(script)) if script else None
                if reason:
                    # A refusal, not just the generic "no command" warning, so
                    # the operator is told *why* the preview is empty.
                    refusals.append(reason)
            self._check_engine_installed(runtime, runtime_config, refusals)
            self._check_model_path(runtime, model, parameters, refusals)
            self._check_shard_completeness(runtime, model, refusals)

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
    def _check_engine_installed(
        runtime: RuntimeName, runtime_config: RuntimeConfig, refusals: list[str]
    ) -> None:
        """Refuse when the engine's binary is missing (or, for SGLang, the module)."""
        binary = runtime_config.binary
        if not binary:
            return
        resolved = shutil.which(binary)
        if resolved is None:
            refusals.append(
                f"{runtime.value} is not installed: binary '{binary}' not found on PATH. "
                f"Run `llmctl setup {runtime.value}`, or set runtimes.{runtime.value}.binary "
                "in settings.yaml."
            )
            return
        if runtime == RuntimeName.SGLANG and not _python_has_module(resolved, "sglang"):
            refusals.append(
                f"sglang is not installed in {resolved}. Run `llmctl setup sglang`, or set "
                "runtimes.sglang.binary to the python of an env that has it."
            )
        if runtime == RuntimeName.STRATA and _strata_server_for(resolved) is None:
            refusals.append(
                f"strata is not installed: {resolved} is not the python of a Strata checkout "
                "(no serve/server.py beside its venv). Run `llmctl setup strata`."
            )
        if runtime == RuntimeName.STRATA and _strata_server_for(resolved) is None:
            refusals.append(
                f"strata is not installed: {resolved} is not the python of a Strata checkout "
                "(no serve/server.py beside its venv). Run `llmctl setup strata`."
            )

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

    @staticmethod
    def _check_shard_completeness(
        runtime: RuntimeName,
        model: ModelRecord | None,
        refusals: list[str],
    ) -> None:
        """Refuse when a sharded GGUF is incomplete or still downloading.

        Ported from the ``llama-server-guarded`` wrapper so it applies to
        every llama.cpp launch. Two unambiguous signals refuse:

        * the model path matches ``*-NNNNN-of-MMMMM.gguf`` and one of the
          MMMMM sibling shards is absent (llama.cpp opens them implicitly,
          failing only ~2 minutes into loading);
        * HuggingFace staging files (``*.incomplete`` under
          ``<root>/.cache/huggingface/download/<subdir>/``) exist next to
          the model, meaning the download is still in flight.

        Everything else fails OPEN — a guard that blocks a valid launch
        because it misread a filename is worse than no guard.
        """
        if runtime != RuntimeName.LLAMA_CPP:
            return
        target = (model.path or model.source) if model else None
        if not target:
            return
        path = Path(str(target))
        if not path.exists():
            return  # _check_model_path already refused with a clearer message
        match = _SHARDED_GGUF_RE.match(path.name)
        if match:
            total = int(match.group("total"))
            missing: list[str] = []
            for i in range(1, total + 1):
                shard = path.with_name(
                    f"{match.group('stem')}-{i:05d}-of-{match.group('total')}.gguf"
                )
                if not shard.is_file():
                    missing.append(shard.name)
            if missing:
                refusals.append(
                    f"Sharded GGUF is incomplete: {total} shard(s) expected, "
                    f"missing {', '.join(missing)}."
                )
        try:
            staging = (
                path.parent.parent / ".cache" / "huggingface" / "download" / path.parent.name
            )
            if staging.is_dir() and any(staging.glob("*.incomplete")):
                refusals.append(
                    f"Model is still downloading: '*.incomplete' staging files "
                    f"present in {staging}."
                )
        except OSError:
            return  # unreadable staging dir must never block a valid launch

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
        if runtime == RuntimeName.SGLANG:
            return self._build_sglang_command(runtime_config, model, parameters, host, port)
        if runtime == RuntimeName.FREETOKEN:
            return self._build_freetoken_command(runtime_config, model, parameters, host, port)
        if runtime == RuntimeName.STRATA:
            return self._build_strata_command(runtime_config, model, parameters, host, port)
        return []

    def _build_strata_command(
        self,
        runtime_config: RuntimeConfig,
        model: ModelRecord | None,
        parameters: dict[str, object],
        host: str,
        port: int,
    ) -> list[str]:
        """Build ``<venv python> serve/server.py --engine strata --config X.json``.

        A Strata model *is* its config JSON (written by Strata's setup): the
        engine binary, the weight pack and the flags are all in there. The
        server script is found next to the interpreter's venv only
        (``<repo>/.venv/bin/python`` -> ``<repo>/serve/server.py``): a
        request parameter must not be able to name a script to run, since
        ``POST /sessions/start`` would turn that into code execution. The GPU
        is passed as ``--gpu`` by nvidia-smi index because Strata numbers
        cards that way. An API key is not put on the command line (visible
        in ``ps`` and in every plan); keep the server on 127.0.0.1 instead.
        """
        config = (model.path or model.source) if model else None
        if not config:
            return []
        python = runtime_config.binary or "python3"
        server = _strata_server_for(python)
        if not server:
            return []
        command = [
            python, str(server), "--engine", "strata", "--config", str(config),
            "--host", host, "--port", str(port),
        ]
        gpu = parameters.get("strata_gpu")
        if gpu is not None:
            command += ["--gpu", str(gpu)]
        command += self._extra_args(parameters, runtime_config)
        return command

    def _build_sglang_command(
        self,
        runtime_config: RuntimeConfig,
        model: ModelRecord | None,
        parameters: dict[str, object],
        host: str,
        port: int,
    ) -> list[str]:
        """Build ``python -m sglang.launch_server ...`` (``binary`` is the interpreter)."""
        model_ref = self._model_reference(model)
        if not model_ref:
            return []
        python = runtime_config.binary or "python3"
        command = [
            python, "-m", "sglang.launch_server",
            "--model-path", model_ref, "--host", host, "--port", str(port),
        ]
        if "tensor_parallel_size" in parameters:
            command += ["--tp", str(parameters["tensor_parallel_size"])]
        if "gpu_memory_utilization" in parameters:
            command += ["--mem-fraction-static", str(parameters["gpu_memory_utilization"])]
        if "max_model_len" in parameters:
            command += ["--context-length", str(parameters["max_model_len"])]
        if parameters.get("dtype"):
            command += ["--dtype", str(parameters["dtype"])]
        if parameters.get("quantization"):
            command += ["--quantization", str(parameters["quantization"])]
        if parameters.get("served_model_name"):
            command += ["--served-model-name", str(parameters["served_model_name"])]
        command += self._extra_args(parameters, runtime_config)
        return command

    def _build_freetoken_command(
        self,
        runtime_config: RuntimeConfig,
        model: ModelRecord | None,
        parameters: dict[str, object],
        host: str,
        port: int,
    ) -> list[str]:
        """Build ``ft serve --model ...``; the GPU comes from CUDA_VISIBLE_DEVICES."""
        model_ref = self._model_reference(model)
        if not model_ref:
            return []
        binary = runtime_config.binary or "ft"
        command = [binary, "serve", "--model", model_ref, "--host", host, "--port", str(port)]
        if "gpu_memory_utilization" in parameters:
            command += ["--memory-ratio", str(parameters["gpu_memory_utilization"])]
        if "max_model_len" in parameters:
            command += ["--max-seq-len-override", str(parameters["max_model_len"])]
        if parameters.get("served_model_name"):
            command += ["--served-model-name", str(parameters["served_model_name"])]
        command += self._extra_args(parameters, runtime_config)
        return command

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
        """Build a ``python <script> [args...]`` command.

        Returns an empty command when the script is unusable — see
        :func:`python_script_refusal`. That is deliberately not an exception:
        planning must always yield a ``LaunchPlan`` (with refusal reasons)
        rather than blowing up ``llmctl plan`` / ``POST /sessions/plan``, and an
        empty command already fails closed at launch with "Launch plan has no
        command to execute", which ``force`` cannot override.
        """
        script = parameters.get("script")
        if not script and model:
            script = model.path or model.source
        if not script:
            return []
        if python_script_refusal(str(script)):
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

    @staticmethod
    def _pinned_port(parameters: dict[str, object]) -> int | None:
        """Return an explicitly pinned port from the profile, if any.

        Lets a profile launch on the same port its systemd unit hardcodes, so a
        model reachable both ways answers at one documented endpoint regardless
        of who started it. Garbage is ignored rather than raising: an unusable
        value falls back to range allocation, which is the previous behaviour.
        """
        raw = parameters.get("port")
        # Only a whole number is a port. `int(raw)` would have truncated 8005.9
        # to 8005 -- silently turning nonsense into a plausible, *different*
        # port, which is the opposite of falling back. A clean digit string is
        # accepted because a hand-edited profile can round-trip `port: "8005"`.
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            port = raw
        elif isinstance(raw, str) and raw.strip().isdigit():
            port = int(raw.strip())
        else:
            return None
        return port if 1 <= port <= 65535 else None

    def _port_available(self, runtime_config: RuntimeConfig, port: int) -> bool:
        """True when ``port`` is neither reserved by a session nor bound."""
        if port in self._used_ports():
            return False
        host = runtime_config.host or self.settings.scheduler.default_host
        return self._is_port_free(host, port)

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
        """Return ports already reserved by active sessions in the database.

        A PLANNED row from a *real* start also reserves its port: it is the
        window between planning and binding, and without it two concurrent
        ``llmctl start`` calls both saw the same port as free. PLANNED rows
        left by ``--dry-run`` (plan says ``dry_run: true``) reserve nothing.
        """
        if self.db is None:
            return set()
        ports: set[int] = set()
        records = self.db.exec(
            select(SessionRecord).where(
                SessionRecord.status.in_(_ACTIVE_STATES | {SessionStatus.PLANNED})  # type: ignore[attr-defined]
            )
        ).all()
        for record in records:
            if record.status == SessionStatus.PLANNED and (record.launch_plan or {}).get(
                "dry_run"
            ):
                continue
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
