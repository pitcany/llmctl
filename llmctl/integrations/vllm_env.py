"""Render the EnvironmentFile body consumed by ``scripts/vllm-launcher.sh``.

The launcher script (lives in ``~/AI/scripts/vllm-launcher.sh``) reads
its configuration from ``services/vllm-tp.env`` via systemd's
``EnvironmentFile=`` directive. This module is the canonical renderer
for that body — line ordering, key spelling, and value formatting are
all locked to match the existing gpu-models output byte-for-byte so the
cutover from gpu-models to llmctl produces no diff on disk.

One render function covers the supported unit shape:

* :func:`render_vllm_env` — the TP-fleet unit (vllm-tp.service, TP=2
  across both GPUs, served_name from the preset).

Pure functions only. No file I/O. No subprocess calls. The caller is
responsible for writing the returned string to the appropriate
``services/*.env`` file.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llmctl.integrations.launcher_env import launcher_env_lines


class VLLMLaunchSpec(BaseModel):
    """Fully-resolved input to :func:`render_vllm_env`.

    A ``Profile`` + ``Model`` + scheduler decision collapses to one of
    these objects. Defaults mirror ``scripts/vllm-launcher.sh`` so a
    minimally-populated spec produces a runnable env file.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description="HuggingFace repo id or absolute local path.")
    served_name: str = Field(description="Name advertised in /v1/models.")
    # Default TP=2 matches gpu-models's _write_env default. The TP-fleet
    # unit on yannik-desktop is dual-GPU; per-GPU slot units (TP=1)
    # construct their own spec with tensor_parallel=1 explicitly.
    tensor_parallel: int = Field(default=2, ge=1)
    gpus: str = Field(default="0,1", description="CUDA_VISIBLE_DEVICES value.")
    port: int = Field(default=8003, ge=1, le=65535)
    host: str = Field(default="0.0.0.0")
    max_model_len: int = Field(default=32768, ge=1)
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, le=1.0)
    quantization: str | None = None
    kv_cache_type: str | None = None
    tool_parser: str | None = None
    max_num_seqs: int | None = None
    max_batched_tokens: int | None = None
    prefix_cache: bool = True
    chunked_prefill: bool = True
    spec_config: dict[str, Any] | str | None = None
    extra_args: str | None = None
    nccl_p2p_disable: bool = False


def render_vllm_env(spec: VLLMLaunchSpec) -> str:
    """Return the ``services/vllm-tp.env`` body for ``spec``.

    Line ordering matches ``gpu_models.backends.vllm._write_env``:

    1. CUDA_VISIBLE_DEVICES
    2. CUDA_DEVICE_ORDER (constant)
    3. PYTORCH_CUDA_ALLOC_CONF (constant)
    4. LD_LIBRARY_PATH, PATH, HF_HOME (from ``launcher_env_lines``)
    5. NCCL_P2P_DISABLE (optional, only when ``nccl_p2p_disable=True``)
    6. VLLM_MODEL, VLLM_SERVED_NAME, VLLM_TP, VLLM_PORT, VLLM_HOST,
       VLLM_MAX_LEN, VLLM_GPU_MEM (always)
    7. VLLM_QUANT, VLLM_KV_DTYPE, VLLM_TOOL_PARSER, VLLM_MAX_SEQS,
       VLLM_MAX_BATCHED_TOKENS (optional, omitted when ``None``)
    8. VLLM_PREFIX_CACHE=0 / VLLM_CHUNKED_PREFILL=0 (only when explicitly
       disabled — the launcher defaults them ON, so we keep the env file
       terse by omitting the truthy case)
    9. VLLM_SPEC_CONFIG (compact JSON, no spaces — matches gpu-models'
       ``json.dumps(..., separators=(",", ":"))``)
    10. VLLM_EXTRA (free-form passthrough)

    Ends with a trailing newline.
    """
    lines: list[str] = [
        f"CUDA_VISIBLE_DEVICES={spec.gpus}",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        *launcher_env_lines(),
    ]

    if spec.nccl_p2p_disable:
        lines.append("NCCL_P2P_DISABLE=1")

    lines.extend(
        [
            f"VLLM_MODEL={spec.model}",
            f"VLLM_SERVED_NAME={spec.served_name}",
            f"VLLM_TP={spec.tensor_parallel}",
            f"VLLM_PORT={spec.port}",
            f"VLLM_HOST={spec.host}",
            f"VLLM_MAX_LEN={spec.max_model_len}",
            f"VLLM_GPU_MEM={spec.gpu_memory_utilization}",
        ]
    )

    if spec.quantization:
        lines.append(f"VLLM_QUANT={spec.quantization}")
    if spec.kv_cache_type:
        lines.append(f"VLLM_KV_DTYPE={spec.kv_cache_type}")
    if spec.tool_parser:
        lines.append(f"VLLM_TOOL_PARSER={spec.tool_parser}")
    if spec.max_num_seqs is not None:
        lines.append(f"VLLM_MAX_SEQS={spec.max_num_seqs}")
    if spec.max_batched_tokens is not None:
        lines.append(f"VLLM_MAX_BATCHED_TOKENS={spec.max_batched_tokens}")

    if spec.prefix_cache is False:
        lines.append("VLLM_PREFIX_CACHE=0")
    if spec.chunked_prefill is False:
        lines.append("VLLM_CHUNKED_PREFILL=0")

    if spec.spec_config:
        if isinstance(spec.spec_config, dict):
            spec_str = json.dumps(spec.spec_config, separators=(",", ":"))
        else:
            spec_str = str(spec.spec_config)
        lines.append(f"VLLM_SPEC_CONFIG={spec_str}")

    if spec.extra_args:
        lines.append(f"VLLM_EXTRA={spec.extra_args}")

    return "\n".join(lines) + "\n"
