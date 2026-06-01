"""On-disk preset loading for llmctl."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from llmctl.presets.paths import default_preset_dir, user_config_dir
from llmctl.presets.schema import Model, PresetSchemaError

log = logging.getLogger(__name__)


def _iter_yaml_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return ()
    return sorted(p for p in directory.glob("*.yaml") if not p.name.startswith("_"))


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise PresetSchemaError(f"{path}: top-level YAML must be a mapping")
    return raw


def _load_directory(directory: Path) -> dict[str, Model]:
    result: dict[str, Model] = {}
    for path in _iter_yaml_files(directory):
        try:
            model = Model.model_validate(_load_yaml(path))
        except (PresetSchemaError, yaml.YAMLError, TypeError, OSError) as exc:
            log.warning("skipping malformed preset %s: %s", path, exc)
            continue
        result[model.alias] = model
    return result


def migrate_legacy_presets() -> int:
    """Symlink legacy presets into the new llmctl preset directory once."""
    target_dir = default_preset_dir()
    legacy_dir = user_config_dir()
    if target_dir.exists():
        return 0

    legacy_files = list(_iter_yaml_files(legacy_dir))
    if not legacy_files:
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    migrated = 0
    for source in legacy_files:
        target = target_dir / source.name
        if target.exists() or target.is_symlink():
            continue
        relative_source = os.path.relpath(source, start=target_dir)
        target.symlink_to(relative_source)
        migrated += 1

    if migrated:
        log.info(
            "llmctl: migrated %s presets from %s to %s via symlinks. "
            "The old directory is unchanged - delete it manually if you no longer need it.",
            migrated,
            legacy_dir,
            target_dir,
        )
    return migrated


def load_all(config_dir: Path | None = None) -> dict[str, Model]:
    """Load all presets, keyed by alias."""
    if config_dir is not None:
        return _load_directory(Path(config_dir))

    migrate_legacy_presets()
    result = _load_directory(default_preset_dir())
    result.update(_load_directory(user_config_dir()))
    return result


def load_one(alias: str, config_dir: Path | None = None) -> Model | None:
    """Load one preset by alias, returning ``None`` when absent."""
    return load_all(config_dir=config_dir).get(alias)
