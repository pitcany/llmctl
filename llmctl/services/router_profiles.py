"""Keep llm-router's vLLM profile files in step with llmctl's presets.

llm-router swaps vLLM models by copying a ``profile_env`` file
(``services/vllm-tp.env.<profile>``) onto ``vllm-tp.env``; ``llmctl swap``
renders that same file from a preset in ``~/.config/llm-models``. Two copies of
one config drift: a preset edit never reached the router's profile, so a
router-triggered swap could load a different config than ``llmctl swap``.

The preset is the source of truth. Each ``profile_env`` backend is matched to
the preset whose served name equals the backend's ``served_model`` (the router
already keys "is it loaded?" on that name), the preset is rendered exactly as
``llmctl swap --dry-run`` would, and the profile's ``KEY=VALUE`` lines are
compared with it. :func:`merge_env` rewrites only those lines, so the history
comments in the profile files survive a sync.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llmctl._fsutil import atomic_write_text
from llmctl.config import VLLMDefaultsConfig
from llmctl.presets import load_all
from llmctl.services.lanes import router_config_path
from llmctl.services.vllm_orchestrator import UnknownPresetError, render_preset_env

_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


@dataclass(frozen=True)
class ProfileBackend:
    """A router.yaml backend launched by copying a vLLM profile env."""

    name: str
    served_model: str
    path: Path


@dataclass
class ProfileDrift:
    """How one profile file compares with its preset's render."""

    backend: ProfileBackend
    preset: str | None = None
    #: key -> (value in the profile file, value the preset renders); None = absent.
    changes: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    rendered: str | None = None
    error: str | None = None

    @property
    def in_sync(self) -> bool:
        return self.error is None and not self.changes


def load_profile_backends(path: Path | None = None) -> list[ProfileBackend]:
    """Return router.yaml's ``profile_env`` backends, in file order."""
    path = path or router_config_path()
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(raw, dict) or not isinstance(raw.get("backends"), list):
        return []
    switch = raw.get("switch") if isinstance(raw.get("switch"), dict) else {}
    services_dir = Path(str(switch.get("services_dir") or path.parent)).expanduser()
    found: list[ProfileBackend] = []
    for entry in raw["backends"]:
        if not isinstance(entry, dict):
            continue
        name, served = entry.get("name"), entry.get("served_model")
        profile = entry.get("profile_env")
        if name and served and profile:
            found.append(ProfileBackend(str(name), str(served), services_dir / str(profile)))
    return found


def parse_env(text: str) -> dict[str, str]:
    """``KEY=VALUE`` lines of an env file, in order (the first of a repeated key wins)."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _ASSIGN.match(line)
        if match and match.group(1) not in values:
            values[match.group(1)] = match.group(2).strip()
    return values


def diff_env(current: str, rendered: str) -> dict[str, tuple[str | None, str | None]]:
    """Keys whose value differs between ``current`` and ``rendered``."""
    have, want = parse_env(current), parse_env(rendered)
    return {
        key: (have.get(key), want.get(key))
        for key in [*want, *(k for k in have if k not in want)]
        if have.get(key) != want.get(key)
    }


def merge_env(current: str, rendered: str) -> str:
    """``current`` with its assignments replaced by ``rendered``'s.

    Comments and blank lines stay where they are; an unchanged line is kept
    byte for byte. Keys the preset no longer renders are dropped, and new keys
    are appended at the end.
    """
    want = parse_env(rendered)
    out: list[str] = []
    seen: set[str] = set()
    for line in current.splitlines():
        match = _ASSIGN.match(line)
        if match is None:
            out.append(line)
            continue
        key = match.group(1)
        if key not in want or key in seen:
            continue
        seen.add(key)
        out.append(line if match.group(2).strip() == want[key] else f"{key}={want[key]}")
    out.extend(f"{key}={value}" for key, value in want.items() if key not in seen)
    return "\n".join(out) + "\n"


def check_profiles(
    *,
    port: int,
    defaults: VLLMDefaultsConfig | None = None,
    config_dir: Path | None = None,
    router_config: Path | None = None,
) -> list[ProfileDrift]:
    """Compare every ``profile_env`` backend with its preset's render."""
    rendered_by_served: dict[str, list[tuple[str, str]]] = {}
    for alias in sorted(load_all(config_dir=config_dir)):
        try:
            spec, body = render_preset_env(
                alias, port=port, defaults=defaults, config_dir=config_dir
            )
        except (UnknownPresetError, ValueError):
            continue  # a preset that does not render cannot be anyone's source
        rendered_by_served.setdefault(spec.served_name, []).append((alias, body))

    report: list[ProfileDrift] = []
    for backend in load_profile_backends(router_config):
        drift = ProfileDrift(backend)
        report.append(drift)
        matches = rendered_by_served.get(backend.served_model, [])
        if not matches:
            drift.error = f"no llmctl preset serves {backend.served_model!r}"
            continue
        if len(matches) > 1:
            names = ", ".join(alias for alias, _ in matches)
            drift.error = f"several presets serve {backend.served_model!r}: {names}"
            continue
        drift.preset, drift.rendered = matches[0]
        try:
            current = backend.path.read_text()
        except OSError:
            current = ""
            drift.changes = {"(file)": ("missing", str(backend.path))}
        drift.changes.update(diff_env(current, drift.rendered))
    return report


def write_profile(drift: ProfileDrift) -> None:
    """Bring ``drift.backend``'s profile file in line with its preset."""
    assert drift.rendered is not None
    path = drift.backend.path
    current = path.read_text() if path.exists() else ""
    atomic_write_text(path, merge_env(current, drift.rendered))
