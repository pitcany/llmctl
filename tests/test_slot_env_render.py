"""Unit + parity tests for slot-flavored env rendering.

The slot env file is similar to the TP env file but with three
slot-only fixtures (header comment, always-on NCCL flags, slot-overridden
served name / TP / port). These tests pin both the structural
differences and full byte parity against ``gpu_models.slot._render_env``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from llmctl.integrations.vllm_env import (
    VLLMLaunchSpec,
    VLLMSlotInfo,
    render_slot_env,
)


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("GPU_MODELS_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("GPU_MODELS_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/home/yannik/AI/cache/huggingface")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


def _gpu_models_slot_render(
    slot_name: str,
    slot_dict: dict[str, Any],
    preset: dict[str, Any],
    defaults: dict[str, Any],
) -> str:
    """Invoke gpu-models's slot _render_env exactly as the slot subcommand does."""
    import gpu_models.slot as gm_slot

    importlib.reload(gm_slot)
    return gm_slot._render_env(slot_name, slot_dict, preset, defaults)


def test_slot_render_emits_header_comment() -> None:
    body = render_slot_env(
        VLLMLaunchSpec(model="m", served_name="ignored-in-slot-mode"),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    lines = body.splitlines()
    assert lines[0].startswith("# Auto-written by llmctl for slot 'coder'")
    assert lines[3] == ""  # blank line separating header from env
    assert lines[4] == "CUDA_VISIBLE_DEVICES=0"


def test_slot_render_forces_tp1_and_slot_identity() -> None:
    """Slot mode pins TP=1 and uses the slot's name as served name."""
    body = render_slot_env(
        VLLMLaunchSpec(
            model="m",
            served_name="llama-3.3-70b",  # preset's served name — must be ignored
            tensor_parallel=2,  # spec says 2 — must be overridden
            gpus="0,1",  # spec says both — must be overridden
            port=8003,  # spec says TP port — must be overridden
        ),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    assert "VLLM_TP=1" in body
    assert "VLLM_SERVED_NAME=coder" in body
    assert "VLLM_PORT=8001" in body
    assert "CUDA_VISIBLE_DEVICES=0\n" in body
    # the preset's served name MUST NOT leak through
    assert "VLLM_SERVED_NAME=llama-3.3-70b" not in body


def test_slot_render_always_emits_nccl_flags() -> None:
    """NCCL_P2P/IB/SHM are always on in slot mode regardless of spec."""
    body = render_slot_env(
        VLLMLaunchSpec(model="m", served_name="s", nccl_p2p_disable=False),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    assert "NCCL_P2P_DISABLE=1" in body
    assert "NCCL_IB_DISABLE=1" in body
    assert "NCCL_SHM_DISABLE=0" in body


def test_slot_render_preserves_preset_optionals() -> None:
    """quant/kv/tool/max_seqs/spec_config/extra_args all flow through."""
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
        "VLLM_SPEC_CONFIG=",  # compact JSON
        "VLLM_EXTRA=--reasoning-parser deepseek_r1",
    ):
        assert token in body, f"missing {token}"


def test_slot_render_prefix_chunked_disabled_emitted() -> None:
    """Same omit-when-default-on contract as TP rendering."""
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
    """Port must be in valid range."""
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        VLLMSlotInfo(name="coder", gpu="0", port=70000)


# ----- byte-diff parity vs gpu_models.slot._render_env ------------------------


def _gpu_slot_dict(gpu: str, port: int, service: str, env_file: Path) -> dict[str, Any]:
    """Build the dict shape gpu_models.slot expects."""
    return {"gpu": gpu, "port": port, "service": service, "env_file": env_file}


PARITY_CASES: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = [
    (
        "coder-llama-32b",
        # preset dict (matches as_gpu_models_preset output for a 32B model)
        {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            "served_name": "qwen2.5-coder-32b",  # ignored by slot render
            "quantization": "awq_marlin",
            "max_model_len": 32768,
            "max_num_seqs": 32,
            "kv_cache_type": "fp8",
            "tool_parser": "hermes",
        },
        # defaults (matches _base.yaml vllm.defaults)
        {
            "gpus": "0,1",
            "tensor_parallel": 2,
            "dtype": "float16",
            "gpu_memory_utilization": 0.85,
            "port": 8003,
            "host": "0.0.0.0",
        },
        # spec_kwargs for the llmctl path — what the loader would produce
        {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            "served_name": "qwen2.5-coder-32b",
            "tensor_parallel": 2,  # overridden to 1 by slot render
            "gpus": "0,1",  # overridden by slot.gpu
            "port": 8003,  # overridden by slot.port
            "host": "0.0.0.0",
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.85,
            "quantization": "awq_marlin",
            "kv_cache_type": "fp8",
            "tool_parser": "hermes",
            "max_num_seqs": 32,
        },
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
    ),
]


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


@pytest.mark.parametrize(
    "slot_name,preset,defaults,spec_kwargs",
    [
        (case[0].rsplit("-", 1)[0] if "coder" in case[0] else "reasoner", *case[1:])
        for case in PARITY_CASES
    ],
    ids=[c[0] for c in PARITY_CASES],
)
def test_slot_semantic_parity_with_gpu_models(
    slot_name: str,
    preset: dict[str, Any],
    defaults: dict[str, Any],
    spec_kwargs: dict[str, Any],
    tmp_path: Path,
) -> None:
    """llmctl slot env vars must equal gpu-models slot env vars.

    Compares output with header comments stripped — the comment block
    is provenance and inert to the launcher script. Every KEY=value
    line and the blank-line separators must match exactly.
    """
    pytest.importorskip("gpu_models")

    slot_gpu = "0" if slot_name == "coder" else "1"
    slot_port = 8001 if slot_name == "coder" else 8002
    slot_service = f"vllm-{slot_name}"
    slot_dict = _gpu_slot_dict(
        gpu=slot_gpu, port=slot_port, service=slot_service,
        env_file=tmp_path / f"{slot_service}.env",
    )

    gpu_models_out = _gpu_models_slot_render(slot_name, slot_dict, preset, defaults)
    llmctl_out = render_slot_env(
        VLLMLaunchSpec(**spec_kwargs),
        VLLMSlotInfo(name=slot_name, gpu=slot_gpu, port=slot_port),
    )

    llmctl_stripped = _strip_comments(llmctl_out)
    gpu_models_stripped = _strip_comments(gpu_models_out)

    assert llmctl_stripped == gpu_models_stripped, (
        f"\n--- llmctl {slot_name} (no comments) ---\n{llmctl_stripped}"
        f"\n--- gpu-models {slot_name} (no comments) ---\n{gpu_models_stripped}"
    )


def test_slot_header_identifies_llmctl_and_slot_name() -> None:
    """Provenance contract: the header names llmctl + the slot + the restart hint."""
    body = render_slot_env(
        VLLMLaunchSpec(model="m", served_name="s"),
        VLLMSlotInfo(name="coder", gpu="0", port=8001),
    )
    header = "\n".join(line for line in body.splitlines() if line.startswith("#"))
    assert "llmctl" in header
    assert "'coder'" in header
    assert "systemctl restart vllm-coder" in header
