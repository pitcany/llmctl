"""Filesystem model discovery.

Scans the configured model roots (see ``configs/model_dirs.yaml``) for local
model artifacts. Discovery is read-only and safe: missing roots, unreadable
directories, and unset environment variables are skipped gracefully.

Heuristics by runtime:

* ``llama_cpp``: each ``*.gguf`` file is treated as a model.
* ``vllm``: each directory containing ``config.json`` is treated as a
  Hugging Face style model.
"""

from __future__ import annotations

import os
from pathlib import Path

from llmctl.config import ModelDirsConfig, load_model_dirs
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import Model

_DEFAULT_MAX_DEPTH = 4


def _iter_files(root: Path, max_depth: int, follow_symlinks: bool) -> list[Path]:
    """Return files under ``root`` limited to ``max_depth`` directory levels."""
    results: list[Path] = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if depth >= max_depth:
            dirnames[:] = []
        for filename in filenames:
            results.append(current / filename)
    return results


def _size_bytes(path: Path) -> int | None:
    """Return file size in bytes, or None when unavailable."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def _discover_gguf(root: Path, max_depth: int, follow_symlinks: bool) -> list[Model]:
    """Discover llama.cpp GGUF models under ``root``."""
    models: list[Model] = []
    for file in _iter_files(root, max_depth, follow_symlinks):
        if file.suffix.lower() != ".gguf":
            continue
        models.append(
            Model(
                name=file.stem,
                runtime=RuntimeName.LLAMA_CPP,
                source=str(file),
                path=str(file),
                format="gguf",
                size_bytes=_size_bytes(file),
                status=ModelStatus.DISCOVERED,
            )
        )
    return models


def _discover_hf(root: Path, max_depth: int, follow_symlinks: bool) -> list[Model]:
    """Discover Hugging Face style models (directories with config.json)."""
    models: list[Model] = []
    seen: set[str] = set()
    for file in _iter_files(root, max_depth, follow_symlinks):
        if file.name != "config.json":
            continue
        model_dir = file.parent
        key = str(model_dir)
        if key in seen:
            continue
        seen.add(key)
        models.append(
            Model(
                name=model_dir.name,
                runtime=RuntimeName.VLLM,
                source=str(model_dir),
                path=str(model_dir),
                format="hf",
                status=ModelStatus.DISCOVERED,
            )
        )
    return models


def discover_filesystem_models(
    runtime: RuntimeName,
    config: ModelDirsConfig | None = None,
) -> list[Model]:
    """Discover on-disk models for a filesystem-backed runtime.

    Args:
        runtime: Target runtime (``llama_cpp`` or ``vllm``). Other runtimes
            return an empty list because they are server/API managed.
        config: Optional pre-loaded model-dirs config; loaded from disk if omitted.

    Returns:
        A list of discovered (non-persisted) :class:`Model` schemas.
    """
    if runtime not in (RuntimeName.LLAMA_CPP, RuntimeName.VLLM):
        return []

    cfg = config or load_model_dirs()
    scan = cfg.scan or {}
    max_depth = int(scan.get("max_depth", _DEFAULT_MAX_DEPTH))
    follow_symlinks = bool(scan.get("follow_symlinks", False))

    models: list[Model] = []
    for root in cfg.model_roots:
        if not root.enabled or runtime.value not in root.runtimes:
            continue
        resolved = root.resolve_path()
        if resolved is None or not resolved.exists() or not resolved.is_dir():
            continue
        if runtime == RuntimeName.LLAMA_CPP:
            models.extend(_discover_gguf(resolved, max_depth, follow_symlinks))
        else:
            models.extend(_discover_hf(resolved, max_depth, follow_symlinks))
    return models
