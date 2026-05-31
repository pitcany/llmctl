"""Byte-diff parity: llmctl renderer == gpu-models renderer.

This is the load-bearing test for the gpu-models -> llmctl migration.
If it passes, llmctl can take over writing ``services/vllm-tp.env``
without producing a single byte of difference on disk — which is the
strongest possible safety guarantee for the cutover.

Strategy
--------
gpu-models's ``_write_env`` writes to ``VLLM_ENV_FILE``. We monkeypatch
that path to a temp file, call it, and read the bytes back. Then we
build the equivalent :class:`VLLMLaunchSpec` and assert
``render_vllm_env(spec) == open(env_file).read()``.

Coverage
--------
Cases match the variety in the real preset library:

* minimal (just model + served_name)
* full TP=2 production preset (mirrors llama-3.3-70b)
* speculative decoding (spec_config JSON dict)
* TurboQuant override (tq_kv_cache_type)
* NCCL p2p disable (heterogeneous GPU setups)
* extra_args passthrough
* prefix_cache / chunked_prefill disabled
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import pytest

from llmctl.integrations.vllm_env import VLLMLaunchSpec, render_vllm_env


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin host-specific env so both renderers emit identical lines.

    gpu-models's ``launcher_env_lines`` and llmctl's both read these vars
    at call time, so setting them once via monkeypatch covers both.
    """
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("GPU_MODELS_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("GPU_MODELS_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/home/yannik/AI/cache/huggingface")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


def _gpu_models_render(
    model: dict[str, Any],
    defaults: dict[str, Any],
    kv_dtype: str,
    env_file: Path,
) -> str:
    """Call gpu-models's ``_write_env`` with the env path patched.

    Reloads the module so monkeypatched env vars take effect for
    constants computed at import time.
    """
    gm_config = importlib.import_module("gpu_models.config")
    gm_vllm = importlib.import_module("gpu_models.backends.vllm")
    # Patch the module-level constant gpu-models uses to write the env file
    original = gm_config.VLLM_ENV_FILE
    try:
        gm_config.VLLM_ENV_FILE = env_file
        gm_vllm.VLLM_ENV_FILE = env_file
        gm_vllm._write_env(model, defaults, kv_dtype)
    finally:
        gm_config.VLLM_ENV_FILE = original
        gm_vllm.VLLM_ENV_FILE = original
    return env_file.read_text()


# Preset specs: (model_dict, defaults_dict, kv_dtype_for_gpu_models, llmctl_spec_kwargs).
# llmctl_spec_kwargs is the equivalent VLLMLaunchSpec input — the bridge
# between the two schemas. This is what Phase 2's profile loader will do
# programmatically; here we spell it out for the parity test.
PARITY_CASES: list[tuple[str, dict[str, Any], dict[str, Any], str, dict[str, Any]]] = [
    (
        "minimal",
        {"model": "test/model", "served_name": "test"},
        {},
        "",
        {"model": "test/model", "served_name": "test"},
    ),
    (
        "llama-3.3-70b-awq",  # mirrors the production preset
        {
            "model": "casperhansen/llama-3.3-70b-instruct-awq",
            "served_name": "llama-3.3-70b",
            "tensor_parallel": 2,
            "gpus": "0,1",
            "port": 8003,
            "host": "0.0.0.0",
            "max_model_len": 65536,
            "gpu_memory_utilization": 0.85,
            "quantization": "awq_marlin",
            "tool_parser": "llama3_json",
            "max_num_seqs": 64,
        },
        {},
        "fp8",  # kv_dtype is a separate arg in gpu-models _write_env
        {
            "model": "casperhansen/llama-3.3-70b-instruct-awq",
            "served_name": "llama-3.3-70b",
            "tensor_parallel": 2,
            "gpus": "0,1",
            "port": 8003,
            "host": "0.0.0.0",
            "max_model_len": 65536,
            "gpu_memory_utilization": 0.85,
            "quantization": "awq_marlin",
            "kv_cache_type": "fp8",
            "tool_parser": "llama3_json",
            "max_num_seqs": 64,
        },
    ),
    (
        "spec-decoding",
        {
            "model": "casperhansen/llama-3.3-70b-instruct-awq",
            "served_name": "llama-3.3-70b-awq-spec",
            "tensor_parallel": 2,
            "max_model_len": 49152,
            "gpu_memory_utilization": 0.85,
            "quantization": "awq_marlin",
            "tool_parser": "llama3_json",
            "max_num_seqs": 32,
            "max_batched_tokens": 4096,
            "spec_config": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "num_speculative_tokens": 4,
            },
        },
        {},
        "fp8",
        {
            "model": "casperhansen/llama-3.3-70b-instruct-awq",
            "served_name": "llama-3.3-70b-awq-spec",
            "tensor_parallel": 2,
            "max_model_len": 49152,
            "gpu_memory_utilization": 0.85,
            "quantization": "awq_marlin",
            "kv_cache_type": "fp8",
            "tool_parser": "llama3_json",
            "max_num_seqs": 32,
            "max_batched_tokens": 4096,
            "spec_config": {
                "model": "meta-llama/Llama-3.2-1B-Instruct",
                "num_speculative_tokens": 4,
            },
        },
    ),
    (
        "nccl-p2p-disable",
        {"model": "test", "served_name": "test"},
        {"nccl_p2p_disable": True},
        "",
        {"model": "test", "served_name": "test", "nccl_p2p_disable": True},
    ),
    (
        "extra-args",
        {
            "model": "test",
            "served_name": "test",
            "extra_args": "--enforce-eager --reasoning-parser deepseek_r1",
        },
        {},
        "",
        {
            "model": "test",
            "served_name": "test",
            "extra_args": "--enforce-eager --reasoning-parser deepseek_r1",
        },
    ),
    (
        "prefix-cache-disabled",
        {
            "model": "test",
            "served_name": "test",
            "prefix_cache": False,
            "chunked_prefill": False,
        },
        {},
        "",
        {
            "model": "test",
            "served_name": "test",
            "prefix_cache": False,
            "chunked_prefill": False,
        },
    ),
    (
        "kv-dtype-via-arg",  # gpu-models accepts kv_dtype as a separate arg
        {"model": "test", "served_name": "test"},
        {},
        "fp8",
        {"model": "test", "served_name": "test", "kv_cache_type": "fp8"},
    ),
    (
        "turboquant-k8v4",
        {"model": "test", "served_name": "test", "tq_kv_cache_type": "turboquant_k8v4"},
        {},
        "turboquant_k8v4",  # what _resolve_kv_dtype would produce under --tq
        {"model": "test", "served_name": "test", "kv_cache_type": "turboquant_k8v4"},
    ),
]


@pytest.mark.parametrize(
    "name,model,defaults,kv_dtype,spec_kwargs",
    PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_byte_diff_parity_with_gpu_models(
    name: str,
    model: dict[str, Any],
    defaults: dict[str, Any],
    kv_dtype: str,
    spec_kwargs: dict[str, Any],
    tmp_path: Path,
) -> None:
    """llmctl render must equal gpu-models render to the byte."""
    pytest.importorskip("gpu_models")

    env_file = tmp_path / "vllm-tp.env"
    gpu_models_output = _gpu_models_render(model, defaults, kv_dtype, env_file)
    llmctl_output = render_vllm_env(VLLMLaunchSpec(**spec_kwargs))

    assert llmctl_output == gpu_models_output, (
        f"\nllmctl ({name}):\n{llmctl_output}\n"
        f"gpu-models ({name}):\n{gpu_models_output}\n"
        f"diff: lengths {len(llmctl_output)} vs {len(gpu_models_output)}"
    )


def test_default_minimal_matches_actual_production_env_shape(tmp_path: Path) -> None:
    """The production ``services/vllm-tp.env`` on disk has known shape.

    Sanity-check that a minimal llmctl render produces all the fixed
    constants in the same order they appear in the live file
    (``CUDA_VISIBLE_DEVICES`` first, ``HF_HOME`` immediately after the
    PATH line, etc).
    """
    body = render_vllm_env(VLLMLaunchSpec(model="x", served_name="x"))
    lines = body.splitlines()
    # The first six lines are always the same in gpu-models output; pin them.
    assert lines[0].startswith("CUDA_VISIBLE_DEVICES=")
    assert lines[1] == "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    assert lines[2] == "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    assert lines[3].startswith("LD_LIBRARY_PATH=")
    assert lines[4].startswith("PATH=")
    assert lines[5].startswith("HF_HOME=")
    # The HF_HOME value must match what was injected via the fixture.
    assert lines[5] == f"HF_HOME={os.environ['HF_HOME']}"
