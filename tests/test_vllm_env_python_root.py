"""Unit tests for per-preset interpreter selection (``python_root``).

Some models cannot be served by the default ``vllm-serve`` interpreter —
e.g. Laguna needs vLLM >= 0.21 while the deployed environment is pinned
lower. A preset therefore may name its own Python root; the renderer must
then point ``VLLM_PYTHON``, ``LD_LIBRARY_PATH`` and ``PATH`` at that root
instead of the caller's environment.

The ``PATH`` case is load-bearing rather than cosmetic: the launcher
``exec``s the interpreter directly, so a wrong ``PATH`` leaves flashinfer's
JIT unable to find ``ninja`` and the engine dies during profiling.
"""

from __future__ import annotations

import pytest

from llmctl.integrations.launcher_env import launcher_env_lines
from llmctl.integrations.vllm_env import VLLMLaunchSpec, render_vllm_env
from llmctl.presets.schema import Model
from llmctl.services.preset_loader import model_to_launch_spec

_ALT_ROOT = "/home/yannik/miniconda3/envs/vllm-laguna"


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/home/yannik/AI/cache/huggingface")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("GPU_MODELS_PYTHON_ROOT", raising=False)
    monkeypatch.delenv("GPU_MODELS_CUDA_ROOT", raising=False)


def _env(body: str) -> dict[str, str]:
    return dict(
        line.partition("=")[::2] for line in body.splitlines() if line and "=" in line
    )


def test_launcher_env_lines_honours_explicit_root() -> None:
    lines = launcher_env_lines(python_root=_ALT_ROOT)
    assert f"LD_LIBRARY_PATH={_ALT_ROOT}/lib:/usr/local/cuda/lib64" in lines
    assert any(line.startswith(f"PATH={_ALT_ROOT}/bin:") for line in lines)


def test_launcher_env_lines_without_root_resolves_from_environment() -> None:
    """Omitting the argument must preserve the pre-existing behaviour."""
    lines = launcher_env_lines()
    assert any("/envs/vllm-serve/lib:" in line for line in lines)


def test_python_root_sets_vllm_python_and_paths() -> None:
    body = render_vllm_env(
        VLLMLaunchSpec(
            model="/home/yannik/models/laguna/Laguna-XS-2.1-FP8",
            served_name="laguna-xs",
            python_root=_ALT_ROOT,
        )
    )
    env = _env(body)
    assert env["VLLM_PYTHON"] == f"{_ALT_ROOT}/bin/python"
    assert env["LD_LIBRARY_PATH"].startswith(f"{_ALT_ROOT}/lib:")
    # The ninja fix: the alt env's bin must come first on PATH.
    assert env["PATH"].startswith(f"{_ALT_ROOT}/bin:")


def test_python_root_omitted_emits_no_vllm_python() -> None:
    """Presets that use the default interpreter must render unchanged."""
    body = render_vllm_env(VLLMLaunchSpec(model="m", served_name="s"))
    assert "VLLM_PYTHON=" not in body
    assert "/envs/vllm-serve/bin:" in _env(body)["PATH"]


def test_preset_threads_python_root_and_spec_config_to_env() -> None:
    """End to end: a preset naming an alt interpreter renders a usable body."""
    draft = "/home/yannik/models/laguna/Laguna-XS-2.1-DFlash-FP8"
    model = Model(
        alias="laguna-xs",
        served_name="laguna-xs",
        model_id="/home/yannik/models/laguna/Laguna-XS-2.1-FP8",
        quantization="compressed-tensors",
        vllm_quantization_flag="compressed-tensors",
        tensor_parallel_size=2,
        max_model_len=131072,
        python_root=_ALT_ROOT,
        spec_config={
            "model": draft,
            "num_speculative_tokens": 7,
            "method": "dflash",
        },
    )
    env = _env(render_vllm_env(model_to_launch_spec(model)))
    assert env["VLLM_PYTHON"] == f"{_ALT_ROOT}/bin/python"
    assert env["PATH"].startswith(f"{_ALT_ROOT}/bin:")
    assert draft in env["VLLM_SPEC_CONFIG"]


def test_spec_config_renders_compact_json() -> None:
    body = render_vllm_env(
        VLLMLaunchSpec(
            model="m",
            served_name="s",
            spec_config={
                "model": "/home/yannik/models/laguna/Laguna-XS-2.1-DFlash-FP8",
                "num_speculative_tokens": 7,
                "method": "dflash",
            },
        )
    )
    line = _env(body)["VLLM_SPEC_CONFIG"]
    assert " " not in line
    assert '"method":"dflash"' in line
