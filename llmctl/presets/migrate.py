"""Plan, apply and roll back the move of legacy presets into the canonical dir.

Legacy ``$XDG_CONFIG_HOME/llm-models/<x>.yaml`` real files become symlinks
(relative) to real files in ``$XDG_CONFIG_HOME/llmctl/presets``.  Every write
is a temp file in the destination directory followed by ``os.replace`` on the
destination path itself, so a symlink at the destination is replaced and never
followed.  Planning reads the directories directly and creates nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from llmctl.presets.paths import default_preset_dir, user_config_dir
from llmctl.presets.schema import Model
from llmctl.presets.store import _iter_yaml_files

MANIFEST_DIRNAME = ".migrations"
MANIFEST_VERSION = 1


class ManifestError(Exception):
    """The manifest is missing or malformed."""


@dataclass(frozen=True)
class Action:
    alias: str
    action: str  # move|identical|already-migrated|collision|invalid
    legacy: Path
    canonical: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "alias": self.alias,
            "action": self.action,
            "from": str(self.legacy),
            "to": str(self.canonical),
        }


@dataclass
class ApplyResult:
    actions: list[Action]
    manifest: Path | None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_real_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _validate(data: bytes) -> str | None:
    """Return the alias when ``data`` is a valid preset, else ``None``."""
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
        if not isinstance(raw, dict):
            return None
        return Model.model_validate(raw).alias
    except Exception:
        return None


def _classify(legacy: Path, canonical: Path) -> Action:
    stem = legacy.stem

    def act(alias: str, name: str) -> Action:
        return Action(alias, name, legacy, canonical)

    if legacy.is_symlink():
        try:
            same = canonical.resolve(strict=True) == legacy.resolve(strict=True)
        except (OSError, RuntimeError):
            same = False
        if same and _is_real_file(canonical):
            return act(stem, "already-migrated")
        return act(stem, "invalid")
    if not legacy.is_file():
        return act(stem, "invalid")
    try:
        data = legacy.read_bytes()
    except OSError:
        return act(stem, "invalid")
    alias = _validate(data)
    if alias is None:
        return act(stem, "invalid")

    if canonical.is_symlink():
        try:
            target = canonical.resolve(strict=True)
        except (OSError, RuntimeError):
            return act(alias, "move")  # broken link or cycle: no canonical real file
        if target == legacy.resolve(strict=True):
            return act(alias, "move")
        return act(alias, "collision")
    if not canonical.exists():
        return act(alias, "move")
    if canonical.is_file():
        try:
            same_bytes = canonical.read_bytes() == data
        except OSError:
            same_bytes = False
        return act(alias, "identical" if same_bytes else "collision")
    return act(alias, "collision")


def plan(legacy_dir: Path | None = None, canonical_dir: Path | None = None) -> list[Action]:
    legacy_dir = legacy_dir or user_config_dir()
    canonical_dir = canonical_dir or default_preset_dir()
    same_dir = os.path.realpath(legacy_dir) == os.path.realpath(canonical_dir)
    actions = []
    for path in _iter_yaml_files(legacy_dir):
        canonical = canonical_dir / path.name
        if same_dir:
            actions.append(Action(path.stem, "invalid", path, canonical))
        else:
            actions.append(_classify(path, canonical))
    return actions


def _replace_with_file(dest: Path, data: bytes, mode: int) -> None:
    """Atomically make ``dest`` a regular file (replacing, never following, a link)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _replace_with_symlink(dest: Path, target: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.{os.getpid()}.lnk"
    with contextlib.suppress(FileNotFoundError):
        os.unlink(tmp)
    os.symlink(target, tmp)
    try:
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _canonical_before(canonical: Path) -> dict[str, Any]:
    if canonical.is_symlink():
        return {"type": "symlink", "target": os.readlink(canonical)}
    if canonical.exists():
        data = canonical.read_bytes()
        return {
            "type": "file",
            "sha256": _sha(data),
            "mode": stat.S_IMODE(canonical.stat().st_mode),
        }
    return {"type": "absent"}


def _write_manifest(canonical_dir: Path, payload: dict[str, Any]) -> Path:
    mdir = canonical_dir / MANIFEST_DIRNAME
    mdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=mdir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        n = 0
        while True:
            final = mdir / (f"{stamp}.json" if n == 0 else f"{stamp}-{n}.json")
            try:
                os.link(tmp, final)  # fails if the name is taken
                return final
            except FileExistsError:
                n += 1
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def apply(legacy_dir: Path | None = None, canonical_dir: Path | None = None) -> ApplyResult:
    legacy_dir = legacy_dir or user_config_dir()
    canonical_dir = canonical_dir or default_preset_dir()
    actions = plan(legacy_dir, canonical_dir)
    changes: list[dict[str, Any]] = []
    error: BaseException | None = None
    try:
        for a in actions:
            if a.action not in ("move", "identical"):
                continue
            data = a.legacy.read_bytes()
            mode = stat.S_IMODE(a.legacy.stat().st_mode)
            before = _canonical_before(a.canonical)
            _replace_with_file(a.canonical, data, mode)
            rel = os.path.relpath(os.path.realpath(a.canonical), os.path.realpath(a.legacy.parent))
            _replace_with_symlink(a.legacy, rel)
            changes.append(
                {
                    "alias": a.alias,
                    "legacy": str(a.legacy),
                    "canonical": str(a.canonical),
                    "action": a.action,
                    "mode": mode,
                    "canonical_before": before,
                    "sha256": _sha(data),
                    "link_target": rel,
                }
            )
    except BaseException as exc:  # record what was done, then re-raise
        error = exc
    manifest = None
    if changes:
        manifest = _write_manifest(
            canonical_dir,
            {
                "version": MANIFEST_VERSION,
                "created": datetime.now(UTC).isoformat(),
                "canonical_dir": str(canonical_dir),
                "legacy_dir": str(legacy_dir),
                "changes": changes,
            },
        )
    if error is not None:
        raise error
    return ApplyResult(actions, manifest)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    changes = payload.get("changes") if isinstance(payload, dict) else None
    if not isinstance(changes, list):
        raise ManifestError(f"malformed manifest {path}: no 'changes' list")
    for c in changes:
        if (
            not isinstance(c, dict)
            or not all(isinstance(c.get(k), str) for k in ("legacy", "canonical", "action"))
            or c["action"] not in ("move", "identical")
            or not isinstance(c.get("mode"), int)
            or isinstance(c.get("mode"), bool)
            or not isinstance(c.get("canonical_before"), dict)
            or c["canonical_before"].get("type") not in ("absent", "symlink", "file")
        ):
            raise ManifestError(f"malformed manifest {path}: bad change entry")
    return changes


def rollback(manifest: Path) -> tuple[list[str], list[str]]:
    """Undo a migration. Returns (restored aliases, skipped descriptions)."""
    changes = _load_manifest(manifest)
    restored: list[str] = []
    skipped: list[str] = []
    for c in changes:
        legacy, canonical = Path(c["legacy"]), Path(c["canonical"])
        name = str(c.get("alias") or legacy.stem)
        if not legacy.is_symlink():
            skipped.append(f"{name}: legacy path is not a symlink")
            continue
        try:
            if legacy.resolve(strict=True) != canonical.resolve(strict=True):
                skipped.append(f"{name}: legacy link no longer points at canonical file")
                continue
            data = canonical.read_bytes()
        except (OSError, RuntimeError):
            skipped.append(f"{name}: canonical file unreadable")
            continue
        _replace_with_file(legacy, data, c["mode"])
        if c["action"] == "move":
            before = c["canonical_before"]
            if before["type"] == "symlink" and isinstance(before.get("target"), str):
                _replace_with_symlink(canonical, before["target"])
            else:
                with contextlib.suppress(FileNotFoundError):
                    canonical.unlink()
        restored.append(name)
    return restored, skipped
