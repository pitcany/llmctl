"""Unit + parity tests for slot-flavored env rendering.

The slot env file is similar to the TP env file but with three
slot-only fixtures (header comment, always-on NCCL flags, slot-overridden
served name / TP / port). These tests pin both the structural
differences and full semantic parity against captured gpu-models
slot output at ``tests/fixtures/env_renders/slot_env__*.txt``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmctl.integrations.vllm_env import (
    VLLMLaunchSpec,
    VLLMSlotInfo,
    render_slot_env,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "env_renders"


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/home/yannik/AI/cache/huggingface")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


def test_slot_render_emits_header_comment() -> None:
    body = render_slot_env(
        VLLMLaunchSpec(model="m", served_name="ignored-in-slot-mode"),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    lines = body.splitlines()
    assert lines[0].startswith("# Auto-written by llmctl for slot 'coder'")
    assert lines[3] == ""
    assert lines[4] == "CUDA_VISIBLE_DEVICES=0"


def test_slot_render_forces_tp1_and_slot_identity() -> None:
    """Slot mode pins TP=1 and uses the slot's name as served name."""
    body = render_slot_env(
        VLLMLaunchSpec(
            model="m",
            served_name="llama-3.3-70b",
            tensor_parallel=2,
            gpus="0,1",
            port=8003,
        ),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    assert "VLLM_TP=1" in body
    assert "VLLM_SERVED_NAME=coder" in body
    assert "VLLM_PORT=8001" in body
    assert "CUDA_VISIBLE_DEVICES=0\n" in body
    assert "VLLM_SERVED_NAME=llama-3.3-70b" not in body


def test_slot_render_always_emits_nccl_flags() -> None:
    body = render_slot_env(
        VLLMLaunchSpec(model="m", served_name="s", nccl_p2p_disable=False),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    assert "NCCL_P2P_DISABLE=1" in body
    assert "NCCL_IB_DISABLE=1" in body
    assert "NCCL_SHM_DISABLE=0" in body


def test_slot_render_preserves_preset_optionals() -> None:
    body = render_slot_env(
        VLLMLaunchSpec(
            model="m",
            served_name="s",
            quantization="awq_marlin",
            kv_cache_type="fp8",
            tool_parser="llama3_json",
            max_num_seqs=32,
            max_batched_tokens=2048,
            spec_config={"model": "tiny-1b", "num_speculative_tokens": 4},
            extra_args="--reasoning-parser deepseek_r1",
        ),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    for token in (
        "VLLM_QUANT=awq_marlin",
        "VLLM_KV_DTYPE=fp8",
        "VLLM_TOOL_PARSER=llama3_json",
        "VLLM_MAX_SEQS=32",
        "VLLM_MAX_BATCHED_TOKENS=2048",
        "VLLM_SPEC_CONFIG=",
        "VLLM_EXTRA=--reasoning-parser deepseek_r1",
    ):
        assert token in body, f"missing {token}"


def test_slot_render_prefix_chunked_disabled_emitted() -> None:
    body = render_slot_env(
        VLLMLaunchSpec(
            model="m",
            served_name="s",
            prefix_cache=False,
            chunked_prefill=False,
        ),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    assert "VLLM_PREFIX_CACHE=0" in body
    assert "VLLM_CHUNKED_PREFILL=0" in body


def test_slot_render_ends_with_trailing_newline() -> None:
    body = render_slot_env(
        VLLMLaunchSpec(model="m", served_name="s"),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    assert body.endswith("\n")


def test_slot_info_rejects_invalid_port() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        VLLMSlotInfo(name="coder", gpu="0", port=70000)


def test_slot_header_identifies_llmctl_and_slot_name() -> None:
    body = render_slot_env(
        VLLMLaunchSpec(model="m", served_name="s"),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    header = "\n".join(line for line in body.splitlines() if line.startswith("#"))
    assert "llmctl" in header
    assert "'coder'" in header
    assert "systemctl restart vllm-coder" in header


# ----- Semantic parity against frozen gpu-models fixtures ---------------------


def _strip_comments(body: str) -> str:
    """Drop comment lines for semantic parity comparison.

    The header comment block is intentionally different (llmctl owns
    the file now and identifies itself in the provenance line). What
    matters for the cutover is that the launcher-relevant env vars are
    byte-identical — comments are inert to vllm-launcher.sh.
    """
    return "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    ) + ("\n" if body.endswith("\n") else "")


PARITY_CASES: list[tuple[str, dict[str, Any], str, int]] = [
    (
        "coder-qwen2.5-coder-32b",
        {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            "served_name": "qwen2.5-coder-32b",
            "tensor_parallel": 2,
            "gpus": "0,1",
            "port": 8003,
            "host": "0.0.0.0",
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.85,
            "quantization": "awq_marlin",
            "kv_cache_type": "fp8",
            "tool_parser": "hermes",
            "max_num_seqs": 32,
        },
        "0",
        8001,
    ),
    (
        "reasoner-r1-with-reasoning-parser",
        {
            "model": "casperhansen/deepseek-r1-distill-llama-70b-awq",
            "served_name": "deepseek-r1-70b",
            "tensor_parallel": 2,
            "gpus": "0,1",
            "port": 8003,
            "host": "0.0.0.0",
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.85,
            "quantization": "awq_marlin",
            "kv_cache_type": "fp8",
            "tool_parser": "llama3_json",
            "max_num_seqs": 64,
            "extra_args": "--reasoning-parser deepseek_r1",
        },
        "1",
        8002,
    ),
]


@pytest.mark.parametrize(
    "fixture_stem,spec_kwargs,slot_gpu,slot_port",
    PARITY_CASES,
    ids=[c[0] for c in PARITY_CASES],
)
def test_slot_semantic_parity_with_frozen_fixture(
    fixture_stem: str,
    spec_kwargs: dict[str, Any],
    slot_gpu: str,
    slot_port: int,
) -> None:
    """llmctl slot env vars must equal captured gpu-models slot output.

    Compares with header comments stripped — the comment block is
    provenance and inert to the launcher script. Every KEY=value line
    and blank-line separator must match exactly.
    """
    slot_name = "coder" if fixture_stem.startswith("coder") else "reasoner"
    actual = render_slot_env(
        VLLMLaunchSpec(**spec_kwargs),
        VLLMSlotInfo(name=slot_name, gpu=slot_gpu, port=slot_port),
    )
    expected = (FIXTURE_DIR / f"slot_env__{fixture_stem}.txt").read_text()

    actual_stripped = _strip_comments(actual)
    expected_stripped = _strip_comments(expected)

    assert actual_stripped == expected_stripped, (
        f"\n--- llmctl ({fixture_stem}, no comments) ---\n{actual_stripped}"
        f"\n--- fixture ({fixture_stem}, no comments) ---\n{expected_stripped}"
    )
