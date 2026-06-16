"""Hermes Agent provider verification.

Hermes routes to local LLM endpoints via ``custom_providers`` in
``~/.hermes/config.yaml``. URLs are static (``http://127.0.0.1:<port>/v1``)
and the provider auto-discovers served models via ``/v1/models``, so a
backend swap normally needs no config mutation. This module:

* Checks that Hermes is installed and the config file exists.
* Verifies that the configured ``base_url`` for a named provider matches
  the port llmctl is about to serve on.
* Prints a one-line status (OK / drift / missing) — **never** mutates
  the user's Hermes config. Repair is left to the user (``hermes config
  edit``) to avoid silently overwriting hand-tuned providers.

Hermes integration is optional. When Hermes isn't installed, every
function returns the appropriate "not installed" status and does
nothing — llmctl works fine without it.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"


class HermesStatus(StrEnum):
    """Result of a single Hermes verification call."""

    NOT_INSTALLED = "not_installed"  # `hermes` binary not on PATH
    NO_CONFIG = "no_config"  # config file doesn't exist
    NO_PROVIDER = "no_provider"  # provider name not in custom_providers
    URL_MISMATCH = "url_mismatch"  # base_url differs from expected
    OK = "ok"


def is_installed(which: Callable[[str], str | None] = shutil.which) -> bool:
    """Return ``True`` when the ``hermes`` binary is on PATH."""
    return which("hermes") is not None


def verify_provider(
    provider: str,
    expected_port: int,
    *,
    config_path: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    logger: Callable[[str], None] = print,
) -> HermesStatus:
    """Verify a named Hermes provider points at ``expected_port``.

    Args:
        provider: Provider name in ``custom_providers``
            (e.g. ``"vllm"``, ``"vllm-coder"``).
        expected_port: The port llmctl is about to serve on; verified
            against the provider's ``base_url``.
        config_path: Override the Hermes config path (default:
            ``~/.hermes/config.yaml``).
        which: Injected for tests; defaults to :func:`shutil.which`.
        logger: Injected for tests; defaults to :func:`print`.

    Returns:
        :class:`HermesStatus` reporting what was found.
    """
    if not is_installed(which):
        return HermesStatus.NOT_INSTALLED

    path = config_path or DEFAULT_CONFIG_PATH
    if not path.is_file():
        logger(
            f"  hermes: config not found at {path} — install with "
            "`hermes init`"
        )
        return HermesStatus.NO_CONFIG

    expected_url = f"http://127.0.0.1:{expected_port}/v1"
    current = _read_provider_url(path, provider)
    if current is None:
        logger(
            f"  hermes: no {provider!r} provider in {path.name} — "
            "add it with `hermes config edit`"
        )
        return HermesStatus.NO_PROVIDER

    if current != expected_url:
        logger(
            f"  hermes: WARNING — {provider}.base_url is {current}, "
            f"expected {expected_url}. Fix with `hermes config edit`."
        )
        return HermesStatus.URL_MISMATCH

    logger(f"  hermes: {provider} -> {expected_url} (verified)")
    return HermesStatus.OK


def _read_provider_url(config_path: Path, provider: str) -> str | None:
    """Return the endpoint URL for ``provider`` in ``config_path``.

    Hermes accepts two on-disk shapes (both consumed by its own
    ``get_compatible_custom_providers``):

    * **Legacy** ``custom_providers`` — a list of ``{name, base_url}`` dicts.
    * **v12+** ``providers`` — a map keyed by provider name whose entries
      carry the URL under ``base_url``, ``url``, or ``api`` (Hermes resolves
      them in that order; see ``_normalize_custom_provider_entry``).

    Both are checked so llmctl's verification matches whatever Hermes itself
    would route to. Returns ``None`` for any failure (missing file, parse
    error, missing provider, non-string value). Callers translate that into
    :class:`HermesStatus.NO_PROVIDER`.
    """
    try:
        with config_path.open() as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    # Legacy list form: custom_providers: [{name, base_url}, ...]
    for entry in data.get("custom_providers") or []:
        if isinstance(entry, dict) and entry.get("name") == provider:
            url = entry.get("base_url")
            return url if isinstance(url, str) else None
    # v12+ keyed-map form: providers: {<name>: {api|url|base_url}}
    providers = data.get("providers")
    if isinstance(providers, dict):
        entry = providers.get(provider)
        if isinstance(entry, dict):
            for key in ("base_url", "url", "api"):
                url = entry.get(key)
                if isinstance(url, str) and url.strip():
                    return url.strip()
    return None
