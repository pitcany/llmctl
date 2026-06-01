"""Filesystem model discovery.

Scans the configured model roots (see ``configs/model_dirs.yaml``) for local
model artifacts. Discovery is read-only and safe: missing roots, unreadable
directories, and unset environment variables are skipped gracefully.

Heuristics by runtime:

* ``llama_cpp``: each ``*.gguf`` file is treated as a model.
* ``vllm``: each directory containing ``config.json`` is treated as a
  Hugging Face style model.
* ``lmstudio``: each ``*.gguf`` file under ``~/.cache/lm-studio/models``
  (or ``LMSTUDIO_MODELS_DIR``) is treated as a model.
* ``ollama``: each entry in
  ``<root>/manifests/registry.ollama.ai/library/<name>/<tag>`` is treated
  as a ``name:tag`` model. The blob payload is intentionally not parsed.
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


def _hf_cache_repo_name(model_dir: Path) -> str | None:
    """Map an HF cache snapshot directory to its ``org/repo`` name.

    The HF hub cache stores each downloaded revision at::

        <root>/models--<org>--<repo>/snapshots/<sha>/

    Returns ``org/repo`` for that layout, or ``None`` when ``model_dir``
    doesn't match (e.g. a hand-rolled model directory outside the cache).
    """
    parts = model_dir.parts
    if len(parts) < 3 or parts[-2] != "snapshots":
        return None
    repo_dir = parts[-3]
    if not repo_dir.startswith("models--"):
        return None
    suffix = repo_dir[len("models--") :]
    # HF uses '--' as the org/repo separator; single '-' inside names is kept.
    return suffix.replace("--", "/", 1)


def _discover_hf(root: Path, max_depth: int, follow_symlinks: bool) -> list[Model]:
    """Discover Hugging Face style models (directories with config.json).

    Inside the HF hub cache, multiple snapshot SHAs of the same repo collapse
    to a single entry named ``org/repo`` (first-seen wins). Outside the cache,
    each ``config.json`` directory is named after its leaf folder.
    """
    models: list[Model] = []
    seen_paths: set[str] = set()
    seen_names: set[str] = set()
    for file in _iter_files(root, max_depth, follow_symlinks):
        if file.name != "config.json":
            continue
        model_dir = file.parent
        path_key = str(model_dir)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        name = _hf_cache_repo_name(model_dir) or model_dir.name
        if name in seen_names:
            continue
        seen_names.add(name)
        models.append(
            Model(
                name=name,
                runtime=RuntimeName.VLLM,
                source=str(model_dir),
                path=str(model_dir),
                format="hf",
                status=ModelStatus.DISCOVERED,
            )
        )
    return models


def _discover_lmstudio(root: Path, max_depth: int, follow_symlinks: bool) -> list[Model]:
    """Discover LM Studio GGUF models (same on-disk shape as llama.cpp)."""
    models: list[Model] = []
    for file in _iter_files(root, max_depth, follow_symlinks):
        if file.suffix.lower() != ".gguf":
            continue
        models.append(
            Model(
                name=file.stem,
                runtime=RuntimeName.LMSTUDIO,
                source=str(file),
                path=str(file),
                format="gguf",
                size_bytes=_size_bytes(file),
                status=ModelStatus.DISCOVERED,
            )
        )
    return models


def _discover_ollama(root: Path) -> list[Model]:
    """Discover Ollama models by enumerating manifest entries.

    Layout: ``<root>/manifests/registry.ollama.ai/library/<name>/<tag>``
    where each ``<tag>`` is a JSON manifest file pointing into ``blobs/``.
    Ignores blob content; the name is what callers need for ``ollama pull``
    / ``ollama run``.
    """
    library = root / "manifests" / "registry.ollama.ai" / "library"
    if not library.is_dir():
        return []
    models: list[Model] = []
    for name_dir in sorted(library.iterdir()):
        if not name_dir.is_dir():
            continue
        for tag_file in sorted(name_dir.iterdir()):
            if not tag_file.is_file():
                continue
            full_name = f"{name_dir.name}:{tag_file.name}"
            models.append(
                Model(
                    name=full_name,
                    runtime=RuntimeName.OLLAMA,
                    source=str(tag_file),
                    path=str(tag_file),
                    format="ollama-manifest",
                    size_bytes=_size_bytes(tag_file),
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
    supported = (
        RuntimeName.LLAMA_CPP,
        RuntimeName.VLLM,
        RuntimeName.LMSTUDIO,
        RuntimeName.OLLAMA,
    )
    if runtime not in supported:
        return []

    cfg = config or load_model_dirs()
    scan = cfg.scan or {}
    max_depth = int(scan.get("max_depth", _DEFAULT_MAX_DEPTH))
    follow_symlinks = bool(scan.get("follow_symlinks", False))

    models: list[Model] = []
    seen_roots: set[Path] = set()
    for root in cfg.model_roots:
        if not root.enabled or runtime.value not in root.runtimes:
            continue
        resolved = root.resolve_path()
        if resolved is None or not resolved.exists() or not resolved.is_dir():
            continue
        # Different roots (env var + default fallback) can resolve to the
        # same on-disk location; scan each unique path only once.
        canonical = resolved.resolve()
        if canonical in seen_roots:
            continue
        seen_roots.add(canonical)
        if runtime == RuntimeName.LLAMA_CPP:
            models.extend(_discover_gguf(resolved, max_depth, follow_symlinks))
        elif runtime == RuntimeName.VLLM:
            models.extend(_discover_hf(resolved, max_depth, follow_symlinks))
        elif runtime == RuntimeName.LMSTUDIO:
            models.extend(_discover_lmstudio(resolved, max_depth, follow_symlinks))
        elif runtime == RuntimeName.OLLAMA:
            models.extend(_discover_ollama(resolved))
    return models
