"""Read-only scan of the preset directories.

Reads both directories directly, never through the loaders that migrate, so
inspecting presets cannot change the filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llmctl.presets.paths import default_preset_dir, user_config_dir
from llmctl.presets.schema import Model, PresetSchemaError
from llmctl.presets.store import _iter_yaml_files


@dataclass(frozen=True)
class SourceFile:
    path: Path
    dir: str  # "canonical" | "legacy"
    kind: str  # "file" | "symlink"
    real: Path
    target: Path | None
    keys: frozenset[str]
    model: Model

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "dir": self.dir,
            "kind": self.kind,
            "target": str(self.target) if self.target is not None else None,
        }


@dataclass
class SourceScan:
    canonical_dir: Path
    legacy_dir: Path
    #: alias -> files in load order (canonical sorted, then legacy sorted)
    files: dict[str, list[SourceFile]] = field(default_factory=dict)
    problems: list[tuple[Path, str]] = field(default_factory=list)

    def effective(self, alias: str) -> SourceFile:
        return self.files[alias][-1]

    def collision(self, alias: str) -> bool:
        return len({f.real for f in self.files[alias]}) > 1

    def has_collision(self) -> bool:
        return any(self.collision(a) for a in self.files)

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_dir": str(self.canonical_dir),
            "legacy_dir": str(self.legacy_dir),
            "presets": [
                {
                    "alias": alias,
                    "effective": str(self.effective(alias).real),
                    "collision": self.collision(alias),
                    "files": [f.as_dict() for f in self.files[alias]],
                }
                for alias in sorted(self.files)
            ],
            "problems": [{"path": str(p), "problem": text} for p, text in self.problems],
        }


def _short(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text if len(text) <= 160 else text[:157] + "..."


def _scan_file(path: Path, dir_name: str) -> SourceFile:
    is_link = path.is_symlink()
    try:
        real = Path.cwd() / path if not path.is_absolute() else path
        real = real.resolve(strict=True)
    except FileNotFoundError:
        raise PresetSchemaError("broken symlink" if is_link else "file not found") from None
    except (RuntimeError, OSError) as exc:
        if isinstance(exc, RuntimeError) or getattr(exc, "errno", None) == 40:
            raise PresetSchemaError("symlink cycle") from None
        raise PresetSchemaError(f"unreadable: {_short(exc)}") from None
    try:
        raw = yaml.safe_load(real.read_text())
    except OSError as exc:
        raise PresetSchemaError(f"unreadable: {_short(exc)}") from None
    except yaml.YAMLError as exc:
        raise PresetSchemaError(f"malformed YAML: {_short(exc)}") from None
    except ValueError as exc:
        raise PresetSchemaError(f"unreadable: {_short(exc)}") from None
    if not isinstance(raw, dict):
        raise PresetSchemaError("top-level YAML must be a mapping")
    try:
        model = Model.model_validate(raw)
    except Exception as exc:  # noqa: BLE001 - any validation failure is a problem entry
        raise PresetSchemaError(f"invalid preset: {_short(exc)}") from None
    return SourceFile(
        path=path if path.is_absolute() else Path.cwd() / path,
        dir=dir_name,
        kind="symlink" if is_link else "file",
        real=real,
        target=real if is_link else None,
        keys=frozenset(str(k) for k in raw),
        model=model,
    )


def scan_sources() -> SourceScan:
    """Scan both preset directories without touching the filesystem."""
    canonical = default_preset_dir()
    legacy = user_config_dir()
    scan = SourceScan(canonical_dir=canonical.absolute(), legacy_dir=legacy.absolute())
    for directory, name in ((canonical, "canonical"), (legacy, "legacy")):
        try:
            paths = list(_iter_yaml_files(directory))
        except OSError as exc:
            scan.problems.append((directory.absolute(), f"unreadable directory: {_short(exc)}"))
            continue
        for path in paths:
            try:
                item = _scan_file(path, name)
            except PresetSchemaError as exc:
                scan.problems.append((path.absolute(), str(exc)))
                continue
            scan.files.setdefault(item.model.alias, []).append(item)
    return scan


def field_provenance(alias: str, scan: SourceScan | None = None) -> dict[str, Any] | None:
    """Per-field value and origin for ``alias``, or ``None`` if not a preset."""
    scan = scan or scan_sources()
    if alias not in scan.files:
        return None
    win = scan.effective(alias)
    dumped = win.model.model_dump(mode="json")
    fields = {
        name: {"value": dumped[name], "source": "preset" if name in win.keys else "default"}
        for name in Model.model_fields
    }
    shadowed = [str(f.path) for f in scan.files[alias] if f is not win]
    return {
        "name": alias,
        "kind": "preset",
        "path": str(win.real),
        "fields": fields,
        "shadowed": shadowed,
    }
