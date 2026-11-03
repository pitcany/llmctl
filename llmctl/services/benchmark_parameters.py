"""Resolve generation defaults once for execution and experiment provenance."""

from __future__ import annotations

from typing import Any

DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0


def effective_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return a new map with omitted generation settings filled in.

    Numeric conversion remains in the live runner so invalid values still
    produce a persisted benchmark failure through its existing error handling.
    """
    return {
        **parameters,
        "max_tokens": parameters.get("max_tokens", DEFAULT_MAX_TOKENS),
        "temperature": parameters.get("temperature", DEFAULT_TEMPERATURE),
    }
