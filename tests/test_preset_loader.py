"""End-to-end parity: preset YAML on disk -> rendered env file.

Strategy: write a real preset YAML in the canonical schema to a temp
``$XDG_CONFIG_HOME/llm-models/`` directory, then assert both paths
produce byte-identical output:

* **llmctl path**: ``load_presets()`` -> :func:`model_to_launch_spec` ->
  :func:`render_vllm_env`
* **gpu-models path**: ``llm_models_config.load_all()`` ->
  :func:`~llm_models_config.adapters.as_gpu_models_preset` ->
  ``gpu_models.backends.vllm._write_env``

Both share the same input YAML and the same canonical Model object;
the only thing being compared is the env-file rendering layer. If
this test passes for every preset shape, the on-disk cutover is safe.
"""

from __future__ import annotations

import importlib
import textwrap
from pathlib import Path

import pytest

from llmctl.config import VLLMDefaultsConfig
from llmctl.integrations.vllm_env import render_vllm_env
from llmctl.services.preset_loader import load_presets


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin host-specific env so both renderers emit identical lines."""
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("GPU_MODELS_PYTHON_ROOT", "/home/yannik/miniconda3/envs/vllm-serve")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("GPU_MODELS_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/home/yannik/AI/cache/huggingface")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


@pytest.fixture
def isolated_preset_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``llm_models_config.user_config_dir`` at a temp directory.

    ``user_config_dir`` consults ``$XDG_CONFIG_HOME``, falling back to
    ``~/.config``; we set both so the user's real ``~/.config/llm-models``
    is invisible regardless of which path the implementation uses.
    """
    xdg = tmp_path / "xdg"
    preset_dir = xdg / "llm-models"
    preset_dir.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(tmp_path))
    # Reload llm_models_config to clear any cached paths
    import llm_models_config.paths
    importlib.reload(llm_models_config.paths)
    import llm_models_config.store
    importlib.reload(llm_models_config.store)
    import llm_models_config
    importlib.reload(llm_models_config)
    return preset_dir


def _write_preset(preset_dir: Path, alias: str, body: str) -> Path:
    """Write a preset YAML into the isolated config dir."""
    path = preset_dir / f"{alias}.yaml"
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path


def _gpu_models_render(alias: str, env_file: Path) -> str:
    """Run the gpu-models path: load_all -> as_gpu_models_preset -> _write_env.

    Resolves kv_dtype the same way gpu-models's CLI does (via
    ``_resolve_kv_dtype`` with no ``--tq`` override), so the test
    captures the production code path end-to-end.
    """
    import gpu_models.backends.vllm as gm_vllm
    import gpu_models.config as gm_config

    importlib.reload(gm_config)
    importlib.reload(gm_vllm)

    cfg = gm_config.load_config()
    model = cfg["vllm"]["models"][alias]
    defaults = cfg["vllm"]["defaults"]
    tq_cfg = cfg.get("turboquant", {})
    kv_dtype = gm_vllm._resolve_kv_dtype(model, defaults, tq_cfg, tq_override=None)

    original = gm_config.VLLM_ENV_FILE
    try:
        gm_config.VLLM_ENV_FILE = env_file
        gm_vllm.VLLM_ENV_FILE = env_file
        gm_vllm._write_env(model, defaults, kv_dtype=kv_dtype)
    finally:
        gm_config.VLLM_ENV_FILE = original
        gm_vllm.VLLM_ENV_FILE = original
    return env_file.read_text()


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
def test_preset_loader_parity_with_gpu_models(
    alias: str,
    body: str,
    isolated_preset_dir: Path,
    tmp_path: Path,
) -> None:
    """End-to-end parity for each shipped preset shape."""
    pytest.importorskip("gpu_models")

    _write_preset(isolated_preset_dir, alias, body)

    # Both paths render to a fresh path each — the renderer doesn't care
    # which file it writes to.
    env_file = tmp_path / "vllm-tp.env"

    gpu_models_out = _gpu_models_render(alias, env_file)

    # llmctl path: load through our loader, render via render_vllm_env
    specs = load_presets(defaults=VLLMDefaultsConfig())
    assert alias in specs, f"loader missed {alias!r}; saw {sorted(specs)}"
    llmctl_out = render_vllm_env(specs[alias])

    assert llmctl_out == gpu_models_out, (
        f"\nllmctl ({alias}):\n{llmctl_out}\n"
        f"gpu-models ({alias}):\n{gpu_models_out}\n"
    )


def test_loader_returns_empty_when_no_presets(isolated_preset_dir: Path) -> None:
    """Empty config dir -> empty mapping. No crash."""
    assert load_presets() == {}


def test_loader_skips_underscore_prefixed_files(
    isolated_preset_dir: Path,
    tmp_path: Path,
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
    """``kv_cache_dtype: auto`` (the default) should produce no VLLM_KV_DTYPE line.

    Matches gpu-models behaviour: ``as_gpu_models_preset`` only emits
    ``kv_cache_type`` when non-auto, so the launcher inherits vLLM's
    own model default.
    """
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
        tensor_parallel=1,  # Note: per-preset wins; this default won't apply
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
    # tensor_parallel comes from the preset itself (2), not defaults
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
