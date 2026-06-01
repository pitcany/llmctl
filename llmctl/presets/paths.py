"""Filesystem paths for llmctl presets."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _xdg_config_home() -> Path:
    raw = os.environ.get("XDG_CONFIG_HOME")
    if raw:
        return Path(raw)
    return Path.home() / ".config"


def user_config_dir() -> Path:
    """Return the legacy ``~/.config/llm-models`` preset directory."""
    return _xdg_config_home() / "llm-models"


def default_preset_dir() -> Path:
    """Return the canonical llmctl preset directory."""
    return _xdg_config_home() / "llmctl" / "presets"


def validate_alias(alias: str) -> None:
    if not _ALIAS_RE.match(alias):
        raise ValueError(f"alias must match {_ALIAS_RE.pattern!r}; got {alias!r}")
