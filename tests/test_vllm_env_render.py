"""Unit tests for :mod:`llmctl.integrations.vllm_env`.

These tests pin the EnvironmentFile body format that
``scripts/vllm-launcher.sh`` consumes. The byte-diff parity tests
against gpu-models live in ``test_vllm_env_parity.py``.
"""

from __future__ import annotations

import json
import os

import pytest

from llmctl.integrations.vllm_env import VLLMLaunchSpec, render_vllm_env


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the host-specific env values so test output is deterministic.

    ``launcher_env_lines`` resolves ``$CONDA_PREFIX`` etc. at call time;
    without pinning, the rendered body differs between developer boxes
    and CI runners. The values chosen match yannik-desktop so the
    parity test against gpu-models produces zero diff there too.
    """
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/home/yannik/AI/cache/huggingface")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("GPU_MODELS_PYTHON_ROOT", raising=False)
    monkeypatch.delenv("GPU_MODELS_CUDA_ROOT", raising=False)


def test_minimal_spec_emits_required_keys() -> None:
    body = render_vllm_env(
        VLLMLaunchSpec(model="casperhansen/llama-3.3-70b-instruct-awq", served_name="llama-70b")
    )
    keys = {line.partition("=")[0] for line in body.splitlines() if line}
    required = {
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "PYTORCH_CUDA_ALLOC_CONF",
        "LD_LIBRARY_PATH",
        "PATH",
        "HF_HOME",
        "VLLM_MODEL",
        "VLLM_SERVED_NAME",
        "VLLM_TP",
        "VLLM_PORT",
        "VLLM_HOST",
        "VLLM_MAX_LEN",
        "VLLM_GPU_MEM",
    }
    assert required <= keys


def test_optional_keys_omitted_by_default() -> None:
    body = render_vllm_env(
        VLLMLaunchSpec(model="m", served_name="s")
    )
    assert "VLLM_QUANT=" not in body
    assert "VLLM_KV_DTYPE=" not in body
    assert "VLLM_TOOL_PARSER=" not in body
    assert "VLLM_MAX_SEQS=" not in body
    assert "VLLM_MAX_BATCHED_TOKENS=" not in body
    assert "VLLM_SPEC_CONFIG=" not in body
    assert "VLLM_EXTRA=" not in body
    # prefix_cache / chunked_prefill default ON; the launcher defaults
    # them ON too, so the env file omits them in that case.
    assert "VLLM_PREFIX_CACHE=" not in body
    assert "VLLM_CHUNKED_PREFILL=" not in body
    assert "NCCL_P2P_DISABLE" not in body


def test_disabled_prefix_cache_and_chunked_prefill_emit_zero() -> None:
    body = render_vllm_env(
        VLLMLaunchSpec(
            model="m",
            served_name="s",
            prefix_cache=False,
            chunked_prefill=False,
        )
    )
    assert "VLLM_PREFIX_CACHE=0" in body
    assert "VLLM_CHUNKED_PREFILL=0" in body


def test_nccl_p2p_disable_emitted_when_true() -> None:
    body = render_vllm_env(
        VLLMLaunchSpec(model="m", served_name="s", nccl_p2p_disable=True)
    )
    assert "NCCL_P2P_DISABLE=1" in body


def test_spec_config_dict_serialized_compactly() -> None:
    spec = VLLMLaunchSpec(
        model="m",
        served_name="s",
        spec_config={"model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 4},
    )
    body = render_vllm_env(spec)
    spec_line = next(line for line in body.splitlines() if line.startswith("VLLM_SPEC_CONFIG="))
    # gpu-models uses separators=(",", ":") — no spaces. Pin that.
    assert ", " not in spec_line
    assert ": " not in spec_line
    decoded = json.loads(spec_line.split("=", 1)[1])
    assert decoded == {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 4,
    }


def test_spec_config_string_passed_through() -> None:
    body = render_vllm_env(
        VLLMLaunchSpec(model="m", served_name="s", spec_config="raw-json-blob")
    )
    assert "VLLM_SPEC_CONFIG=raw-json-blob" in body


def test_extra_args_passthrough() -> None:
    body = render_vllm_env(
        VLLMLaunchSpec(
            model="m",
            served_name="s",
            extra_args="--enforce-eager --reasoning-parser deepseek_r1",
        )
    )
    assert "VLLM_EXTRA=--enforce-eager --reasoning-parser deepseek_r1" in body


def test_line_ordering_matches_gpu_models_layout() -> None:
    """The render order is the same as gpu_models.backends.vllm._write_env.

    Locking this order means the env file diff is purely value-driven
    when comparing the two implementations.
    """
    body = render_vllm_env(
        VLLMLaunchSpec(
            model="m",
            served_name="s",
            quantization="awq_marlin",
            kv_cache_type="fp8",
            tool_parser="llama3_json",
            max_num_seqs=64,
            max_batched_tokens=4096,
            spec_config={"x": 1},
            extra_args="--enforce-eager",
            nccl_p2p_disable=True,
        )
    )
    lines = body.splitlines()
    order = [line.partition("=")[0] for line in lines if line]
    expected_prefix = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "PYTORCH_CUDA_ALLOC_CONF",
        "LD_LIBRARY_PATH",
        "PATH",
        "HF_HOME",
        "NCCL_P2P_DISABLE",
        "VLLM_MODEL",
        "VLLM_SERVED_NAME",
        "VLLM_TP",
        "VLLM_PORT",
        "VLLM_HOST",
        "VLLM_MAX_LEN",
        "VLLM_GPU_MEM",
        "VLLM_QUANT",
        "VLLM_KV_DTYPE",
        "VLLM_TOOL_PARSER",
        "VLLM_MAX_SEQS",
        "VLLM_MAX_BATCHED_TOKENS",
        "VLLM_SPEC_CONFIG",
        "VLLM_EXTRA",
    ]
    assert order == expected_prefix


def test_body_ends_with_trailing_newline() -> None:
    body = render_vllm_env(VLLMLaunchSpec(model="m", served_name="s"))
    assert body.endswith("\n")


def test_extra_keys_rejected_by_pydantic() -> None:
    """``extra='forbid'`` catches preset YAML typos at load time."""
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        VLLMLaunchSpec(model="m", served_name="s", typo_key=True)  # type: ignore[call-arg]


def test_launcher_env_resolves_from_alt_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back-compat: gpu-models's $GPU_MODELS_PYTHON_ROOT still works."""
    monkeypatch.delenv("LLMCTL_PYTHON_ROOT", raising=False)
    monkeypatch.setenv("GPU_MODELS_PYTHON_ROOT", "/opt/other-python")
    body = render_vllm_env(VLLMLaunchSpec(model="m", served_name="s"))
    assert "PATH=/opt/other-python/bin:" in body


def test_launcher_env_raises_when_no_python_root_resolvable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "LLMCTL_PYTHON_ROOT",
        "GPU_MODELS_PYTHON_ROOT",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
    ):
        monkeypatch.delenv(var, raising=False)
    from llmctl.integrations.launcher_env import LauncherEnvError

    with pytest.raises(LauncherEnvError):
        render_vllm_env(VLLMLaunchSpec(model="m", served_name="s"))


def test_default_hf_home_uses_user_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """When $HF_HOME is unset, fall back to ~/.cache/huggingface."""
    monkeypatch.delenv("HF_HOME", raising=False)
    body = render_vllm_env(VLLMLaunchSpec(model="m", served_name="s"))
    expected = f"HF_HOME={os.path.expanduser('~/.cache/huggingface')}"
    assert expected in body
