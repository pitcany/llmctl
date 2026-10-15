"""Size a Hugging Face model for the detected GPUs and render a `vllm serve` command."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hardware import HardwareInfo
from .inspect import ModelMetadata
from .parsing import detect_family, detect_param_count, detect_quantization

# Architectures vLLM is known NOT to support out of the box. Conservative list.
VLLM_KNOWN_UNSUPPORTED: set[str] = set()


@dataclass
class LaunchPlan:
    model_id: str
    family: str | None
    param_count_b: float | None
    quantization: str | None
    architectures: list[str]
    vllm_supported: bool
    vllm_quantization_flag: str | None
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    trust_remote_code: bool
    dtype: str | None
    host: str = "127.0.0.1"
    port: int = 8000
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# Bytes per parameter at common precisions.
_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "awq": 0.5,  # 4-bit weights + overhead
    "gptq": 0.5,
    "bnb": 0.5,
    "gguf": 0.5,
}


def _estimate_weight_gib(param_count_b: float | None, quant: str | None) -> float | None:
    if param_count_b is None:
        return None
    bpp = _BYTES_PER_PARAM.get(quant or "fp16", 2.0)
    return param_count_b * 1e9 * bpp / (1024**3)


def _params_from_config(cfg: dict[str, Any]) -> float | None:
    """Try to read a parameter count from config.json. Often missing."""
    # Some configs include num_parameters or n_params; usually they don't.
    for key in ("num_parameters", "n_params", "total_params"):
        v = cfg.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v) / 1e9
    return None


def _quant_from_config(cfg: dict[str, Any]) -> str | None:
    """Read quantization_config.quant_method from config.json if present."""
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        method = qc.get("quant_method") or qc.get("quantization_method")
        if isinstance(method, str):
            return method.lower()
    return None


def _max_position_embeddings(cfg: dict[str, Any]) -> int | None:
    for key in ("max_position_embeddings", "max_seq_len", "model_max_length"):
        v = cfg.get(key)
        if isinstance(v, int) and 0 < v < 10_000_000:
            return v
    return None


def _pick_tp(
    gpu_count: int,
    param_count_b: float | None,
    weight_gib: float | None,
    per_gpu_vram_gib: float | None,
) -> int:
    """Choose tensor_parallel_size that fits weights + slack."""
    if gpu_count <= 1:
        return 1
    if weight_gib is None or per_gpu_vram_gib is None:
        return 1 if (param_count_b is None or param_count_b < 30) else min(gpu_count, 2)
    # Leave ~25% headroom for activations + KV cache.
    usable_per_gpu = per_gpu_vram_gib * 0.75
    for tp in (1, 2, 4, 8):
        if tp > gpu_count:
            break
        if weight_gib / tp <= usable_per_gpu:
            return tp
    return min(gpu_count, 8)


def _pick_max_len(
    model_cap: int | None, weight_gib: float | None, per_gpu_vram_gib: float | None, tp: int
) -> int:
    """Derate context length by weight-pressure on the GPU.

    KV cache shares the same GPU pool as weights + CUDA graphs + activations,
    so a 32B AWQ model that takes ~half the GPU can't also serve 32k context
    on a single 32 GiB card. We bucket by `(weight_gib / tp) / per_gpu_vram`.
    """
    model_cap = model_cap or 8192
    hard_cap = 32768
    if weight_gib is None or per_gpu_vram_gib is None or per_gpu_vram_gib <= 0:
        return min(model_cap, hard_cap)
    pressure = (weight_gib / tp) / per_gpu_vram_gib
    if pressure < 0.35:
        derated = 32768
    elif pressure < 0.55:
        derated = 16384
    elif pressure < 0.75:
        derated = 8192
    else:
        derated = 4096
    return min(model_cap, derated)


def _pick_gpu_util(
    quant: str | None, weight_gib: float | None, per_gpu_vram_gib: float | None, tp: int
) -> float:
    """Be conservative for quantized small models, more aggressive for larger fits."""
    if weight_gib and per_gpu_vram_gib:
        frac = (weight_gib / tp) / per_gpu_vram_gib
        if frac < 0.4:
            return 0.85
        if frac < 0.6:
            return 0.90
        return 0.92
    return 0.90


def build_plan(
    meta: ModelMetadata, hw: HardwareInfo, host: str = "127.0.0.1", port: int = 8000
) -> LaunchPlan:
    cfg = meta.config
    archs = cfg.get("architectures") or []
    if isinstance(archs, str):
        archs = [archs]

    family = detect_family(meta.model_id) or detect_family(" ".join(archs))
    param_count = _params_from_config(cfg) or detect_param_count(meta.model_id)
    quant = (
        _quant_from_config(cfg)
        or detect_quantization(meta.model_id)
        or detect_quantization(" ".join(meta.tags))
    )

    weight_gib = _estimate_weight_gib(param_count, quant)
    per_gpu = min((g.vram_gib for g in hw.gpus), default=None)
    tp = _pick_tp(hw.count, param_count, weight_gib, per_gpu)
    max_len = _pick_max_len(_max_position_embeddings(cfg), weight_gib, per_gpu, tp)
    util = _pick_gpu_util(quant, weight_gib, per_gpu, tp)

    # dtype hint: don't force when quantized.
    dtype: str | None = None
    if quant in (None, "fp16", "bf16"):
        dtype = "auto"

    # trust_remote_code: required for many community models with custom modeling code.
    trust = bool(cfg.get("auto_map")) or any(
        kw in meta.model_id.lower() for kw in ("deepseek", "yi-", "internlm", "command-r")
    )

    # vllm quantization flag mapping.
    vq_map = {"awq": "awq", "gptq": "gptq", "fp8": "fp8", "bnb": "bitsandbytes"}
    vllm_quant_flag = vq_map.get(quant) if quant else None

    supported = True
    warnings: list[str] = []
    notes: list[str] = []

    if quant == "gguf":
        supported = False
        warnings.append("GGUF is not natively supported by vLLM. Use llama.cpp instead.")
    if archs and archs[0] in VLLM_KNOWN_UNSUPPORTED:
        supported = False
        warnings.append(f"Architecture {archs[0]} may not be supported by vLLM.")
    if not hw.gpus:
        warnings.append("No NVIDIA GPUs detected. vLLM requires CUDA.")
    if weight_gib and per_gpu and (weight_gib / tp) > per_gpu * 0.85:
        warnings.append(
            f"OOM risk: estimated {weight_gib:.1f} GiB weights across tp={tp} "
            f"is ≥85% of per-GPU VRAM ({per_gpu:.1f} GiB). Consider higher --tp "
            "or a quantized variant."
        )
    if meta.gated:
        warnings.append("Gated model — ensure HF_TOKEN is set and license accepted.")
    if param_count:
        notes.append(f"Estimated weights ~{weight_gib:.1f} GiB at {quant or 'fp16'}.")

    return LaunchPlan(
        model_id=meta.model_id,
        family=family,
        param_count_b=param_count,
        quantization=quant,
        architectures=list(archs),
        vllm_supported=supported,
        vllm_quantization_flag=vllm_quant_flag,
        tensor_parallel_size=tp,
        max_model_len=max_len,
        gpu_memory_utilization=util,
        trust_remote_code=trust,
        dtype=dtype,
        host=host,
        port=port,
        warnings=warnings,
        notes=notes,
    )


def render_command(plan: LaunchPlan) -> list[str]:
    """Turn the plan into a `vllm serve ...` argv list."""
    cmd = [
        "vllm",
        "serve",
        plan.model_id,
        "--host",
        plan.host,
        "--port",
        str(plan.port),
        "--tensor-parallel-size",
        str(plan.tensor_parallel_size),
        "--max-model-len",
        str(plan.max_model_len),
        "--gpu-memory-utilization",
        f"{plan.gpu_memory_utilization:.2f}",
    ]
    if plan.vllm_quantization_flag:
        cmd += ["--quantization", plan.vllm_quantization_flag]
    if plan.dtype:
        cmd += ["--dtype", plan.dtype]
    if plan.trust_remote_code:
        cmd += ["--trust-remote-code"]
    return cmd
