"""Tests for filesystem model discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmctl.config import ModelDirsConfig, ModelRoot, load_model_dirs
from llmctl.db import RuntimeName
from llmctl.discovery import discover_filesystem_models

BUNDLED_MODEL_DIRS = (
    Path(__file__).resolve().parents[1] / "configs" / "model_dirs.yaml"
)


def _config_for(root: Path, runtime: str) -> ModelDirsConfig:
    return ModelDirsConfig(
        model_roots=[
            ModelRoot(name="test-root", enabled=True, relative_path=str(root), runtimes=[runtime])
        ],
        scan={"max_depth": 4, "follow_symlinks": False},
    )


def test_discover_gguf_models(tmp_path: Path) -> None:
    (tmp_path / "model-a.gguf").write_bytes(b"x")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "model-b.gguf").write_bytes(b"y")
    (tmp_path / "notes.txt").write_text("ignore")

    models = discover_filesystem_models(
        RuntimeName.LLAMA_CPP, _config_for(tmp_path, "llama_cpp")
    )
    names = sorted(model.name for model in models)
    assert names == ["model-a", "model-b"]
    assert all(model.format == "gguf" for model in models)
    assert all(model.runtime == RuntimeName.LLAMA_CPP for model in models)


def test_discover_hf_models(tmp_path: Path) -> None:
    model_dir = tmp_path / "my-llm"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

    models = discover_filesystem_models(RuntimeName.VLLM, _config_for(tmp_path, "vllm"))
    assert len(models) == 1
    assert models[0].name == "my-llm"
    assert models[0].format == "hf"


def test_discover_hf_cache_uses_org_repo_name(tmp_path: Path) -> None:
    """HF hub layout: models--<org>--<repo>/snapshots/<sha>/config.json."""
    snap = tmp_path / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")

    models = discover_filesystem_models(RuntimeName.VLLM, _config_for(tmp_path, "vllm"))
    assert len(models) == 1
    assert models[0].name == "Qwen/Qwen2.5-7B-Instruct"


def test_discover_hf_cache_handles_dashed_org_name(tmp_path: Path) -> None:
    """First '--' is the org/repo separator; single '-' inside names is kept."""
    snap = tmp_path / "models--meta-llama--Llama-3.3-70B-Instruct" / "snapshots" / "sha"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")

    models = discover_filesystem_models(RuntimeName.VLLM, _config_for(tmp_path, "vllm"))
    assert models[0].name == "meta-llama/Llama-3.3-70B-Instruct"


def test_discover_hf_cache_dedups_snapshots(tmp_path: Path) -> None:
    """Two snapshot SHAs of the same repo → one Model entry."""
    repo = tmp_path / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots"
    for sha in ("aaa", "bbb"):
        (repo / sha).mkdir(parents=True)
        (repo / sha / "config.json").write_text("{}")

    models = discover_filesystem_models(RuntimeName.VLLM, _config_for(tmp_path, "vllm"))
    assert len(models) == 1
    assert models[0].name == "Qwen/Qwen2.5-7B-Instruct"


def test_discover_hf_non_cache_still_uses_dir_name(tmp_path: Path) -> None:
    """A model dir outside the cache layout falls back to the leaf folder name."""
    model_dir = tmp_path / "my-finetune"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    models = discover_filesystem_models(RuntimeName.VLLM, _config_for(tmp_path, "vllm"))
    assert len(models) == 1
    assert models[0].name == "my-finetune"


def test_discover_lmstudio_models(tmp_path: Path) -> None:
    (tmp_path / "publisher").mkdir()
    (tmp_path / "publisher" / "qwen-7b.gguf").write_bytes(b"x")
    (tmp_path / "publisher" / "README.md").write_text("ignore")

    models = discover_filesystem_models(
        RuntimeName.LMSTUDIO, _config_for(tmp_path, "lmstudio")
    )
    assert len(models) == 1
    assert models[0].name == "qwen-7b"
    assert models[0].format == "gguf"
    assert models[0].runtime == RuntimeName.LMSTUDIO


def test_discover_ollama_manifest_models(tmp_path: Path) -> None:
    library = tmp_path / "manifests" / "registry.ollama.ai" / "library"
    (library / "qwen3-coder").mkdir(parents=True)
    (library / "qwen3-coder" / "30b-a3b-q8_0").write_text("{}")
    (library / "gemma3").mkdir()
    (library / "gemma3" / "latest").write_text("{}")
    # A bare blobs dir without a manifest should be ignored entirely.
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "sha256-abc").write_bytes(b"z")

    models = discover_filesystem_models(
        RuntimeName.OLLAMA, _config_for(tmp_path, "ollama")
    )
    names = sorted(m.name for m in models)
    assert names == ["gemma3:latest", "qwen3-coder:30b-a3b-q8_0"]
    assert all(m.runtime == RuntimeName.OLLAMA for m in models)
    assert all(m.format == "ollama-manifest" for m in models)


def test_discover_ollama_no_manifests_dir_is_safe(tmp_path: Path) -> None:
    # A configured ollama root that doesn't have the manifests/ subtree
    # (e.g. /usr/share/ollama/.ollama/models on a host without the
    # system daemon installed) must not raise.
    assert (
        discover_filesystem_models(RuntimeName.OLLAMA, _config_for(tmp_path, "ollama")) == []
    )


def test_discover_missing_root_is_safe(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    config = _config_for(missing, "llama_cpp")
    assert discover_filesystem_models(RuntimeName.LLAMA_CPP, config) == []


def test_resolve_path_prefers_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_target = tmp_path / "from-env"
    env_target.mkdir()
    monkeypatch.setenv("LLMCTL_TEST_HF", str(tmp_path))
    root = ModelRoot(
        name="hf",
        env_var="LLMCTL_TEST_HF",
        relative_path="from-env",
        default_path="/nonexistent/should-not-be-used",
        runtimes=["vllm"],
    )
    assert root.resolve_path() == env_target


def test_resolve_path_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fallback = tmp_path / "default-loc"
    fallback.mkdir()
    monkeypatch.delenv("LLMCTL_TEST_HF", raising=False)
    root = ModelRoot(
        name="hf",
        env_var="LLMCTL_TEST_HF",
        relative_path="hub",
        default_path=str(fallback),
        runtimes=["vllm"],
    )
    assert root.resolve_path() == fallback


def test_resolve_path_returns_none_when_nothing_set() -> None:
    root = ModelRoot(name="empty", env_var="NO_SUCH_ENV_VAR_XYZ_123", runtimes=["vllm"])
    assert root.resolve_path() is None


def test_bundled_lmstudio_default_path_matches_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped lm-studio default_path must match the real LM Studio location.

    LM Studio 0.3.x and newer store models under ``~/.lmstudio/models`` on
    Linux/macOS/Windows; earlier drafts of this config pointed at
    ``~/.cache/lm-studio/models``, which is a directory that has never
    existed on a stock LM Studio install. Guard against regressing.
    """
    monkeypatch.delenv("LMSTUDIO_MODELS_DIR", raising=False)
    config = load_model_dirs(BUNDLED_MODEL_DIRS)
    lm_studio = next(root for root in config.model_roots if root.name == "lm-studio")
    assert lm_studio.resolve_path() == Path.home() / ".lmstudio" / "models"


def test_duplicate_resolved_roots_scan_once(tmp_path: Path) -> None:
    """Env-var root and default_path root pointing at the same dir → one scan."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    config = ModelDirsConfig(
        model_roots=[
            ModelRoot(
                name="via-relative", relative_path=str(tmp_path), runtimes=["llama_cpp"]
            ),
            ModelRoot(
                name="via-default", default_path=str(tmp_path), runtimes=["llama_cpp"]
            ),
        ],
        scan={"max_depth": 4, "follow_symlinks": False},
    )
    models = discover_filesystem_models(RuntimeName.LLAMA_CPP, config)
    assert len(models) == 1
