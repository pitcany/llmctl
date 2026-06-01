"""Byte-diff parity: llmctl renderer == frozen gpu-models output.

This file used to import ``gpu_models`` directly and assert byte
equality at test time. After Phase 7 removed the gpu-models package
from the workspace, we keep the same guarantee by comparing against
**captured fixtures** at ``tests/fixtures/env_renders/`` — those
files were produced by gpu-models's ``_write_env`` before deletion,
so a passing test here proves llmctl writes the exact same bytes
gpu-models would have.

Regenerate fixtures (only meaningful while gpu-models still exists
in some checkout) with ``tests/fixtures/_capture_parity_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmctl.integrations.vllm_env import VLLMLaunchSpec, render_vllm_env

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "env_renders"


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin host-specific env so the rendered launcher_env_lines match the captured fixtures."""
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/home/yannik/AI/cache/huggingface")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("GPU_MODELS_PYTHON_ROOT", raising=False)
    monkeypatch.delenv("GPU_MODELS_CUDA_ROOT", raising=False)


# Each entry: (fixture stem, llmctl VLLMLaunchSpec kwargs)
# The kwargs map to the renderer's inputs; the fixture is the
# byte-by-byte expected output (captured from gpu-models pre-Phase-7).
PARITY_CASES: list[tuple[str, dict[str, Any]]] = [
    (
        "minimal",
        {"model": "test/model", "served_name": "test"},
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
        {"model": "test", "served_name": "test", "nccl_p2p_disable": True},
    ),
    (
        "extra-args",
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
    ),
    (
        "kv-dtype-via-arg",
        {"model": "test", "served_name": "test", "kv_cache_type": "fp8"},
    ),
    (
        "turboquant-k8v4",
        {"model": "test", "served_name": "test", "kv_cache_type": "turboquant_k8v4"},
    ),
]


@pytest.mark.parametrize(
    "fixture_stem,spec_kwargs",
    PARITY_CASES,
    ids=[case[0] for case in PARITY_CASES],
)
def test_render_matches_frozen_gpu_models_fixture(
    fixture_stem: str,
    spec_kwargs: dict[str, Any],
) -> None:
    """llmctl render must equal the captured gpu-models output byte-for-byte."""
    fixture = FIXTURE_DIR / f"vllm_env__{fixture_stem}.txt"
    expected = fixture.read_text()
    actual = render_vllm_env(VLLMLaunchSpec(**spec_kwargs))
    assert actual == expected, (
        f"\n--- llmctl ({fixture_stem}) ---\n{actual}"
        f"\n--- fixture ({fixture.name}) ---\n{expected}"
    )


def test_default_minimal_matches_production_env_shape() -> None:
    """Sanity: the minimal render's fixed-constant prefix is invariant."""
    body = render_vllm_env(VLLMLaunchSpec(model="x", served_name="x"))
    lines = body.splitlines()
    assert lines[0].startswith("CUDA_VISIBLE_DEVICES=")
    assert lines[1] == "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    assert lines[2] == "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    assert lines[3].startswith("LD_LIBRARY_PATH=")
    assert lines[4].startswith("PATH=")
    assert lines[5].startswith("HF_HOME=")
