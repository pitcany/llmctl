"""One-shot script to capture gpu-models' env-file output as fixtures.

Run once before deleting the gpu-models package; the captured files
become the byte-parity ground truth that survives the cutover.

Usage:
    LLMCTL_PYTHON_ROOT=/home/yannik/miniconda3/envs/vllm-serve \
    LLMCTL_CUDA_ROOT=/usr/local/cuda \
    HF_HOME=/home/yannik/AI/cache/huggingface \
    uv run python tests/fixtures/_capture_parity_fixtures.py

This file lives in tests/fixtures/ rather than tools/ because:
* It documents how the fixtures were generated.
* It can be re-run by any developer who wants to regenerate them
  before the gpu-models package is finally removed.
* It should NOT be re-run after gpu-models is gone (the import will
  fail clearly, signalling "the fixtures are the source of truth now").
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

# Inputs match the parity test cases verbatim. Editing this list is
# fine; just remember to commit the regenerated fixtures.
RENDER_CASES: list[tuple[str, dict, dict, str]] = [
    (
        "minimal",
        {"model": "test/model", "served_name": "test"},
        {},
        "",
    ),
    (
        "llama-3.3-70b-awq",
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
        "fp8",
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
    ),
    (
        "nccl-p2p-disable",
        {"model": "test", "served_name": "test"},
        {"nccl_p2p_disable": True},
        "",
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
    ),
    (
        "kv-dtype-via-arg",
        {"model": "test", "served_name": "test"},
        {},
        "fp8",
    ),
    (
        "turboquant-k8v4",
        {"model": "test", "served_name": "test", "tq_kv_cache_type": "turboquant_k8v4"},
        {},
        "turboquant_k8v4",
    ),
]

PRESET_FIXTURES: list[tuple[str, str]] = [
    (
        "llama-3.3-70b",
        """
        alias: llama-3.3-70b
        served_name: llama-3.3-70b
        model_id: casperhansen/llama-3.3-70b-instruct-awq
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 65536
        family: llama
        param_count_b: 70.0
        architectures:
        - LlamaForCausalLM
        max_num_seqs: 64
        gpu_memory_utilization: 0.85
        kv_cache_dtype: fp8
        dtype: null
        trust_remote_code: false
        host: 0.0.0.0
        port: 8000
        tool_parser: llama3_json
        reasoning_parser: null
        tq: false
        shortcuts: []
        schema_version: 1
        """,
    ),
    (
        "deepseek-r1-70b",
        """
        alias: deepseek-r1-70b
        served_name: deepseek-r1-70b
        model_id: casperhansen/deepseek-r1-distill-llama-70b-awq
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 65536
        family: llama
        param_count_b: 70.0
        architectures:
        - LlamaForCausalLM
        max_num_seqs: 64
        gpu_memory_utilization: 0.85
        kv_cache_dtype: fp8
        dtype: null
        trust_remote_code: false
        host: 0.0.0.0
        port: 8000
        tool_parser: llama3_json
        reasoning_parser: deepseek_r1
        tq: false
        shortcuts: []
        schema_version: 1
        """,
    ),
    (
        "qwq-32b-awq",
        """
        alias: qwq-32b-awq
        served_name: qwq-32b-awq
        model_id: Qwen/QwQ-32B-AWQ
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 40960
        family: qwen
        param_count_b: 32.0
        architectures:
        - Qwen2ForCausalLM
        max_num_seqs: 16
        gpu_memory_utilization: 0.85
        kv_cache_dtype: fp8
        dtype: null
        trust_remote_code: false
        host: 0.0.0.0
        port: 8000
        tool_parser: hermes
        reasoning_parser: qwen3
        tq: false
        shortcuts: []
        schema_version: 1
        """,
    ),
    (
        "qwen3-coder-next-80b",
        """
        alias: qwen3-coder-next-80b
        served_name: qwen3-coder-next-80b
        model_id: cyankiwi/Qwen3-Coder-Next-AWQ-4bit
        quantization: compressed-tensors
        vllm_quantization_flag: compressed-tensors
        tensor_parallel_size: 2
        max_model_len: 32768
        family: qwen
        param_count_b: 80.0
        architectures:
        - Qwen3NextForCausalLM
        max_num_seqs: 64
        gpu_memory_utilization: 0.85
        kv_cache_dtype: fp8
        dtype: null
        trust_remote_code: false
        host: 0.0.0.0
        port: 8000
        tool_parser: hermes
        reasoning_parser: null
        tq: false
        shortcuts: []
        schema_version: 1
        """,
    ),
]

SLOT_CASES: list[tuple[str, dict, dict, str, int]] = [
    (
        "coder-qwen2.5-coder-32b",
        {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            "served_name": "qwen2.5-coder-32b",
            "quantization": "awq_marlin",
            "max_model_len": 32768,
            "max_num_seqs": 32,
            "kv_cache_type": "fp8",
            "tool_parser": "hermes",
        },
        {
            "gpus": "0,1",
            "tensor_parallel": 2,
            "dtype": "float16",
            "gpu_memory_utilization": 0.85,
            "port": 8003,
            "host": "0.0.0.0",
        },
        "0",
        8001,
    ),
    (
        "reasoner-r1-with-reasoning-parser",
        {
            "model": "casperhansen/deepseek-r1-distill-llama-70b-awq",
            "served_name": "deepseek-r1-70b",
            "quantization": "awq_marlin",
            "max_model_len": 32768,
            "max_num_seqs": 64,
            "kv_cache_type": "fp8",
            "tool_parser": "llama3_json",
            "extra_args": "--reasoning-parser deepseek_r1",
        },
        {
            "gpus": "0,1",
            "tensor_parallel": 2,
            "dtype": "float16",
            "gpu_memory_utilization": 0.85,
            "port": 8003,
            "host": "0.0.0.0",
        },
        "1",
        8002,
    ),
]


def capture_env_renders(out_dir: Path) -> None:
    import gpu_models.backends.vllm as gm_vllm
    import gpu_models.config as gm_config

    importlib.reload(gm_config)
    importlib.reload(gm_vllm)

    for name, model, defaults, kv_dtype in RENDER_CASES:
        target = out_dir / f"vllm_env__{name}.txt"
        original = gm_config.VLLM_ENV_FILE
        try:
            gm_config.VLLM_ENV_FILE = target
            gm_vllm.VLLM_ENV_FILE = target
            gm_vllm._write_env(model, defaults, kv_dtype=kv_dtype)
        finally:
            gm_config.VLLM_ENV_FILE = original
            gm_vllm.VLLM_ENV_FILE = original
        print(f"wrote {target.name} ({target.stat().st_size} bytes)")


def capture_preset_renders(out_dir: Path, xdg: Path) -> None:
    import os

    import gpu_models.backends.vllm as gm_vllm
    import gpu_models.config as gm_config
    import llm_models_config
    import llm_models_config.paths
    import llm_models_config.store

    preset_dir = xdg / "llm-models"
    preset_dir.mkdir(parents=True, exist_ok=True)

    os.environ["XDG_CONFIG_HOME"] = str(xdg)
    importlib.reload(llm_models_config.paths)
    importlib.reload(llm_models_config.store)
    importlib.reload(llm_models_config)
    importlib.reload(gm_config)
    importlib.reload(gm_vllm)

    for alias, body in PRESET_FIXTURES:
        (preset_dir / f"{alias}.yaml").write_text(textwrap.dedent(body).strip() + "\n")

    for alias, _ in PRESET_FIXTURES:
        cfg = gm_config.load_config()
        model = cfg["vllm"]["models"][alias]
        defaults = cfg["vllm"]["defaults"]
        tq_cfg = cfg.get("turboquant", {})
        kv_dtype = gm_vllm._resolve_kv_dtype(model, defaults, tq_cfg, tq_override=None)

        target = out_dir / f"preset_env__{alias}.txt"
        original = gm_config.VLLM_ENV_FILE
        try:
            gm_config.VLLM_ENV_FILE = target
            gm_vllm.VLLM_ENV_FILE = target
            gm_vllm._write_env(model, defaults, kv_dtype=kv_dtype)
        finally:
            gm_config.VLLM_ENV_FILE = original
            gm_vllm.VLLM_ENV_FILE = original
        print(f"wrote {target.name} ({target.stat().st_size} bytes)")


def capture_slot_renders(out_dir: Path) -> None:
    import gpu_models.slot as gm_slot

    importlib.reload(gm_slot)

    for name, preset, defaults, gpu, port in SLOT_CASES:
        slot_name = "coder" if name.startswith("coder") else "reasoner"
        target = out_dir / f"slot_env__{name}.txt"
        slot_dict = {
            "gpu": gpu,
            "port": port,
            "service": f"vllm-{slot_name}",
            "env_file": target,
        }
        body = gm_slot._render_env(slot_name, slot_dict, preset, defaults)
        target.write_text(body)
        print(f"wrote {target.name} ({target.stat().st_size} bytes)")


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "env_renders"
    out_dir.mkdir(exist_ok=True)
    xdg = Path("/tmp/llmctl-fixture-xdg")
    xdg.mkdir(exist_ok=True)
    print(f"writing fixtures to {out_dir}")
    capture_env_renders(out_dir)
    capture_preset_renders(out_dir, xdg)
    capture_slot_renders(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
