"""End-to-end parity: preset YAML on disk -> rendered env file.

Loads a real preset YAML (via :func:`llm_models_config.load_all`),
projects through :func:`model_to_launch_spec`, renders via
:func:`render_vllm_env`, and asserts byte-equality against captured
gpu-models fixtures at ``tests/fixtures/env_renders/``. After Phase 7
removed the gpu-models package, the fixtures are the source of truth
(see ``tests/fixtures/_capture_parity_fixtures.py``).
"""

from __future__ import annotations

import importlib
import textwrap
from pathlib import Path

import pytest

from llmctl.config import VLLMDefaultsConfig
from llmctl.integrations.vllm_env import render_vllm_env
from llmctl.services.preset_loader import load_presets

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "env_renders"


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin host-specific env so the renderer matches the captured fixtures."""
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/home/yannik/AI/cache/huggingface")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


@pytest.fixture
def isolated_preset_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``llm_models_config.user_config_dir`` at a temp directory.

    Reloads the module on setup AND teardown so downstream tests
    (test_tui in particular) don't inherit our temp config view.
    """
    xdg = tmp_path / "xdg"
    preset_dir = xdg / "llm-models"
    preset_dir.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(tmp_path))
    import llm_models_config
    import llm_models_config.paths
    import llm_models_config.store
    importlib.reload(llm_models_config.paths)
    importlib.reload(llm_models_config.store)
    importlib.reload(llm_models_config)
    yield preset_dir
    importlib.reload(llm_models_config.paths)
    importlib.reload(llm_models_config.store)
    importlib.reload(llm_models_config)


def _write_preset(preset_dir: Path, alias: str, body: str) -> Path:
    """Write a preset YAML into the isolated config dir."""
    path = preset_dir / f"{alias}.yaml"
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path


# (alias, YAML body to write to disk) — fixtures captured pre-Phase-7
# from gpu-models's _write_env give the expected output.
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


@pytest.mark.parametrize("alias,body", PRESET_FIXTURES, ids=[a for a, _ in PRESET_FIXTURES])
def test_preset_loader_matches_frozen_fixture(
    alias: str,
    body: str,
    isolated_preset_dir: Path,
) -> None:
    """End-to-end parity for each shipped preset shape."""
    _write_preset(isolated_preset_dir, alias, body)

    specs = load_presets(defaults=VLLMDefaultsConfig())
    assert alias in specs, f"loader missed {alias!r}; saw {sorted(specs)}"
    actual = render_vllm_env(specs[alias])

    expected = (FIXTURE_DIR / f"preset_env__{alias}.txt").read_text()
    assert actual == expected, (
        f"\n--- llmctl ({alias}) ---\n{actual}"
        f"\n--- fixture ({alias}) ---\n{expected}"
    )


def test_loader_returns_empty_when_no_presets(isolated_preset_dir: Path) -> None:
    """Empty config dir -> empty mapping. No crash."""
    assert load_presets() == {}


def test_loader_skips_underscore_prefixed_files(
    isolated_preset_dir: Path,
) -> None:
    """Files starting with ``_`` are config (e.g. ``_shortcuts.yaml``),
    not presets — the loader must skip them."""
    (isolated_preset_dir / "_shortcuts.yaml").write_text("data: irrelevant\n")
    _write_preset(
        isolated_preset_dir,
        "real-preset",
        """
        alias: real-preset
        served_name: real
        model_id: org/real
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """,
    )
    specs = load_presets()
    assert list(specs.keys()) == ["real-preset"]


def test_reasoning_parser_routed_through_extra_args(
    isolated_preset_dir: Path,
) -> None:
    """A preset with reasoning_parser produces VLLM_EXTRA=--reasoning-parser X."""
    _write_preset(
        isolated_preset_dir,
        "deepseek-r1-70b",
        """
        alias: deepseek-r1-70b
        served_name: r1
        model_id: org/r1
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        kv_cache_dtype: fp8
        reasoning_parser: deepseek_r1
        """,
    )
    specs = load_presets()
    body = render_vllm_env(specs["deepseek-r1-70b"])
    assert "VLLM_EXTRA=--reasoning-parser deepseek_r1" in body


def test_kv_cache_auto_omitted_from_env(isolated_preset_dir: Path) -> None:
    """``kv_cache_dtype: auto`` should produce no VLLM_KV_DTYPE line."""
    _write_preset(
        isolated_preset_dir,
        "auto-kv",
        """
        alias: auto-kv
        served_name: auto-kv
        model_id: org/m
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        kv_cache_dtype: auto
        """,
    )
    specs = load_presets()
    body = render_vllm_env(specs["auto-kv"])
    assert "VLLM_KV_DTYPE" not in body


def test_user_defaults_override_built_in(isolated_preset_dir: Path) -> None:
    """Custom ``VLLMDefaultsConfig`` should change the rendered output."""
    _write_preset(
        isolated_preset_dir,
        "p",
        """
        alias: p
        served_name: p
        model_id: org/m
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """,
    )
    custom = VLLMDefaultsConfig(
        gpus="0",
        tensor_parallel=1,  # per-preset wins; this default won't apply
        port=9000,
        nccl_p2p_disable=True,
        max_batched_tokens=2048,
    )
    specs = load_presets(defaults=custom)
    body = render_vllm_env(specs["p"])
    assert "CUDA_VISIBLE_DEVICES=0\n" in body
    assert "VLLM_PORT=9000" in body
    assert "NCCL_P2P_DISABLE=1" in body
    assert "VLLM_MAX_BATCHED_TOKENS=2048" in body
    assert "VLLM_TP=2" in body


def test_port_override_wins_over_defaults(isolated_preset_dir: Path) -> None:
    """``port_override`` on :func:`model_to_launch_spec` lets a managed
    unit pin the port even when defaults say something different."""
    from llm_models_config import load_all

    from llmctl.services.preset_loader import model_to_launch_spec

    _write_preset(
        isolated_preset_dir,
        "p",
        """
        alias: p
        served_name: p
        model_id: org/m
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """,
    )
    models = load_all()
    spec = model_to_launch_spec(
        models["p"],
        VLLMDefaultsConfig(port=8003),
        port_override=9999,
    )
    assert spec.port == 9999
