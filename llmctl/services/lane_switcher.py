"""Ask llm-router (the one lane switch engine) to make a model live.

Since 2026-10-20 (gateway consolidation, P1) ``llmctl run`` no longer swaps
the GPUs itself when llm-router is up: it POSTs ``/admin/switch`` and lets the
same engine that serves request-driven swaps do it, with the same guards (heal's
broken list, lane leases). ``force`` skips only the two request-thrash guards
(failure cooldown, ``min_hold_s``), which is what an operator means by "swap".

The old in-process path stays behind ``llmctl run --direct`` and is also the
fallback when the router is down or does not know the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from llmctl.integrations.launcher_env import parse_environment_file

KEY_ENV = "ROUTER_API_KEY"


@dataclass(frozen=True)
class RouterSwitch:
    """What ``/admin/switch`` answered."""

    backend: str
    served_model: str
    upstream: str
    switched: bool
    already_active: bool
    previous: str | None


class RouterRefused(RuntimeError):
    """The router knows the model but will not (or could not) start it now."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def router_key(env_file: Path | None) -> str | None:
    """The router's shared secret: ``$ROUTER_API_KEY``, else the env file llm-router reads."""
    key = os.environ.get(KEY_ENV)
    if key:
        return key
    if env_file is not None and env_file.exists():
        return parse_environment_file(env_file).get(KEY_ENV) or None
    return None


def router_up(url: str, *, timeout: float = 3.0) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def switch_via_router(
    model: str,
    *,
    url: str,
    key: str,
    force: bool = True,
    timeout_s: float = 1900.0,
) -> RouterSwitch | None:
    """Make ``model`` live through llm-router.

    Returns ``None`` when the router does not know the model (404): the caller
    should fall back to its own path. Raises :class:`RouterRefused` on 503 (a
    guard refused, or the start failed) and ``httpx.HTTPError`` when the router
    cannot be reached mid-call.
    """
    response = httpx.post(
        f"{url}/admin/switch",
        json={"backend": model, "force": force},
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(timeout_s, connect=5.0),
    )
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = response.text[:300]
        retry = response.headers.get("Retry-After")
        raise RouterRefused(message, retry_after=float(retry) if retry else None)
    body = response.json()
    return RouterSwitch(
        backend=body["backend"],
        served_model=body["served_model"],
        upstream=body["upstream"],
        switched=bool(body["switched"]),
        already_active=bool(body["already_active"]),
        previous=body.get("previous"),
    )
