"""Host validation — find drift between what llmctl records and reality.

Every check answers the same question in a different place: *does the
thing llmctl thinks exists still exist, where llmctl thinks it is?*

* :func:`check_preset_model_ids` — preset ``model_id`` points at a
  checkpoint that is gone (including a symlink that no longer resolves).
* :func:`check_registry_paths` — a registry row's ``path`` is gone.
* :func:`check_model_root_symlinks` — dangling symlinks in the
  configured model roots, whether or not anything references them.
* :func:`check_managed_unit_ports` — a managed unit is active but its
  registered port serves nothing, i.e. the service moved.

Checks are read-only and take their inputs as arguments, so the CLI
owns all the loading and the tests can drive each check in isolation.
Nothing here knows any host-specific path: every location comes from
llmctl's own config (presets, ``model_dirs.yaml``, ``managed_units``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmctl.config import ManagedUnitConfig, ModelDirsConfig
from llmctl.discovery import iter_broken_symlinks
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.presets.schema import Model as PresetModel
from llmctl.schemas import Model

_DEFAULT_PROBE_TIMEOUT_S = 1.5


@dataclass(frozen=True)
class Finding:
    """One piece of detected drift.

    Args:
        check: Stable machine-readable check id (``preset-model-missing``,
            ``registry-path-missing``, ``broken-symlink``, ``port-drift``).
        target: What the finding is about — a preset alias, model name,
            symlink path, or unit name.
        detail: Human-readable explanation, including the path or port.
    """

    check: str
    target: str
    detail: str


def as_local_path(value: str) -> Path | None:
    """Return an expanded :class:`Path` when ``value`` names a local location.

    A bare Hugging Face repo id (``org/model``) is a valid ``model_id``
    and a valid ``/v1/models`` ``root``, but it is not a claim about the
    filesystem — checking it for existence would report drift that isn't
    there. Only rooted or home-relative values are treated as paths.
    """
    if value.startswith(("/", "~", "./")):
        return Path(value).expanduser()
    return None


def check_preset_model_ids(presets: Mapping[str, PresetModel]) -> list[Finding]:
    """Flag presets whose ``model_id`` path is absent from disk.

    ``Path.exists()`` follows symlinks, so a preset pointing through a
    store symlink whose target was deleted is reported here too.
    """
    findings: list[Finding] = []
    for alias, preset in sorted(presets.items()):
        path = as_local_path(preset.model_id)
        if path is None or path.exists():
            continue
        findings.append(
            Finding(
                check="preset-model-missing",
                target=alias,
                detail=f"model_id does not exist: {preset.model_id}",
            )
        )
    return findings


def check_registry_paths(models: Iterable[Model]) -> list[Finding]:
    """Flag registry rows whose recorded ``path`` is absent from disk."""
    findings: list[Finding] = []
    for model in models:
        if not model.path:
            continue
        path = as_local_path(model.path)
        if path is None or path.exists():
            continue
        findings.append(
            Finding(
                check="registry-path-missing",
                target=model.name,
                detail=f"registered path does not exist: {model.path}",
            )
        )
    return findings


def check_model_root_symlinks(config: ModelDirsConfig) -> list[Finding]:
    """Flag dangling symlinks under every enabled model root.

    This is the only check that finds *orphans* — links nothing points
    at any more. Roots that are unset, missing, or resolve to the same
    directory are swept once or skipped, matching
    :func:`llmctl.discovery.discover_filesystem_models`.
    """
    max_depth = int((config.scan or {}).get("max_depth", 4))
    findings: list[Finding] = []
    seen: set[Path] = set()
    for root in config.model_roots:
        if not root.enabled:
            continue
        resolved = root.resolve_path()
        if resolved is None or not resolved.is_dir():
            continue
        canonical = resolved.resolve()
        if canonical in seen:
            continue
        seen.add(canonical)
        for link in iter_broken_symlinks(resolved, max_depth):
            findings.append(
                Finding(
                    check="broken-symlink",
                    target=root.name,
                    detail=f"symlink resolves to nothing: {link}",
                )
            )
    return findings


def check_managed_unit_ports(
    units: Iterable[ManagedUnitConfig],
    *,
    systemctl: SystemctlRunner | None = None,
    http_get: Callable[[str, float], Any] | None = None,
    probe_timeout_s: float = _DEFAULT_PROBE_TIMEOUT_S,
) -> list[Finding]:
    """Flag active managed units that serve nothing on their registered port.

    The gate is systemd, not the config's ``enabled`` flag: ``enabled``
    says whether llmctl may *manage* the unit, while this check only asks
    whether a unit systemd reports as running is reachable where llmctl
    records it. An inactive unit is not drift, so it is skipped — as is
    every unit on a host with no ``systemctl``.
    """
    runner = systemctl or SystemctlRunner()
    if not runner.available():
        return []
    get = http_get or _default_http_get
    findings: list[Finding] = []
    for unit in units:
        if not runner.is_active(unit.unit_name):
            continue
        if _serves_models(get, unit.default_port, probe_timeout_s):
            continue
        findings.append(
            Finding(
                check="port-drift",
                target=unit.unit_name,
                detail=(
                    f"unit is active but nothing answers /v1/models on the "
                    f"registered port {unit.default_port}"
                ),
            )
        )
    return findings


def _serves_models(
    http_get: Callable[[str, float], Any], port: int, timeout: float
) -> bool:
    """Return True when ``port`` answers ``/v1/models`` with a model list."""
    url = f"http://localhost:{port}/v1/models"
    try:
        resp = http_get(url, timeout)
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("data"))


def _default_http_get(url: str, timeout: float) -> Any:
    """Production HTTP GET — patched in tests."""
    return urllib.request.urlopen(url, timeout=timeout)  # noqa: S310 - localhost only
