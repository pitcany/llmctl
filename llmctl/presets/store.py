"""On-disk preset loading for llmctl."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llmctl.presets.paths import default_preset_dir, user_config_dir
from llmctl.presets.schema import Model, PresetSchemaError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresetRecord:
    """A preset loaded from disk together with the file it came from."""

    model: Model
    source_path: Path


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
    return {alias: rec.model for alias, rec in _load_directory_records(directory).items()}


def _load_directory_records(directory: Path) -> dict[str, PresetRecord]:
    result: dict[str, PresetRecord] = {}
    for path in _iter_yaml_files(directory):
        try:
            model = Model.model_validate(_load_yaml(path))
        except (PresetSchemaError, yaml.YAMLError, TypeError, OSError) as exc:
            log.warning("skipping malformed preset %s: %s", path, exc)
            continue
        result[model.alias] = PresetRecord(model=model, source_path=path)
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


def load_all_records(config_dir: Path | None = None) -> dict[str, PresetRecord]:
    """Load all presets with their source paths, keyed by alias.

    Same precedence as :func:`load_all` (user dir wins over default dir).
    Used by callers that need to know which file a preset came from —
    e.g. the TUI's edit/delete flow.
    """
    if config_dir is not None:
        return _load_directory_records(Path(config_dir))

    migrate_legacy_presets()
    result = _load_directory_records(default_preset_dir())
    result.update(_load_directory_records(user_config_dir()))
    return result


def save_preset(
    model: Model,
    *,
    config_dir: Path | None = None,
) -> Path:
    """Persist ``model`` as YAML and return the written path.

    With ``config_dir`` explicit, writes ``<config_dir>/<alias>.yaml``
    unconditionally. With no override the resolver respects whatever
    directory the alias currently lives in:

    * If the canonical ``default_preset_dir()`` already holds a real
      file for the alias, overwrite it.
    * If only the legacy ``user_config_dir()`` holds the alias, write
      back to that file (matching :func:`load_all` precedence — user
      dir wins).
    * If the canonical dir holds a symlink (typically a legacy-migrated
      preset), replace the symlink with a real file in the canonical
      dir AND drop the legacy copy so loaders converge on the new YAML.
    * Otherwise (new alias) write a fresh file under the canonical dir.

    The function trusts the model — schema validation already ran when
    the form built the ``Model``.
    """
    if config_dir is not None:
        target_dir = Path(config_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{model.alias}.yaml"
        if target.is_symlink():
            target.unlink()
        _write_yaml(target, model)
        return target

    canonical_dir = default_preset_dir()
    legacy_dir = user_config_dir()
    canonical = canonical_dir / f"{model.alias}.yaml"
    legacy = legacy_dir / f"{model.alias}.yaml"

    if canonical.is_symlink():
        canonical.unlink()
        canonical_dir.mkdir(parents=True, exist_ok=True)
        _write_yaml(canonical, model)
        if legacy.is_symlink() or legacy.exists():
            legacy.unlink()
        return canonical

    if canonical.exists():
        _write_yaml(canonical, model)
        return canonical

    if legacy.exists():
        _write_yaml(legacy, model)
        return legacy

    canonical_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(canonical, model)
    return canonical


def _write_yaml(path: Path, model: Model) -> None:
    payload = model.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))


def delete_preset(
    alias: str,
    *,
    config_dir: Path | None = None,
) -> list[Path]:
    """Remove the YAML for ``alias`` and return the paths that were deleted.

    When ``config_dir`` is provided only that directory is touched.
    Otherwise both the canonical llmctl preset dir and the legacy
    ``~/.config/llm-models/`` dir are checked so the alias is truly
    gone — symlinks in the canonical dir that point at legacy files
    leave a dangling reference otherwise.
    """
    if config_dir is not None:
        candidates: list[Path] = [Path(config_dir) / f"{alias}.yaml"]
    else:
        candidates = [
            default_preset_dir() / f"{alias}.yaml",
            user_config_dir() / f"{alias}.yaml",
        ]
    removed: list[Path] = []
    for path in candidates:
        if path.is_symlink() or path.exists():
            path.unlink()
            removed.append(path)
    return removed
