"""Tests for filesystem model discovery."""

from __future__ import annotations

import json
from pathlib import Path

from llmctl.config import ModelDirsConfig, ModelRoot
from llmctl.db import RuntimeName
from llmctl.discovery import discover_filesystem_models


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


def test_discover_skips_unknown_runtime(tmp_path: Path) -> None:
    assert discover_filesystem_models(RuntimeName.OLLAMA, _config_for(tmp_path, "ollama")) == []


def test_discover_missing_root_is_safe(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    config = _config_for(missing, "llama_cpp")
    assert discover_filesystem_models(RuntimeName.LLAMA_CPP, config) == []
