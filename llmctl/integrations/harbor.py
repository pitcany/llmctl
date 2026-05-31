"""Harbor (Open WebUI + companions) Docker integration.

Harbor runs Open WebUI, the OpenAI-compatible reverse-proxy, and
several companion services (ollama, speaches, ...) as Docker containers.
Two interactions matter for llmctl:

1. **Preflight**: stop ``harbor.ollama`` before starting a GPU-claiming
   vLLM unit. The ollama container holds GPU memory; failing to stop
   it causes vLLM init to OOM on a partially-occupied GPU.
2. **Pin validation**: WebUI Workspace > Models lets users define
   custom models that wrap a base served name. If the base name is
   absent from ``/v1/models``, the UI fails with "model not found".
   :func:`webui_custom_models` reads the WebUI sqlite DB so the caller
   can warn before a swap removes a pinned base.

Both interactions are optional. When Docker isn't installed or the
named container isn't running, every function returns the appropriate
"unavailable" value and does nothing — llmctl works fine without
Harbor.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

DEFAULT_OLLAMA_CONTAINER = "harbor.ollama"
DEFAULT_WEBUI_CONTAINER = "harbor.webui"
DEFAULT_WEBUI_DB_PATH = "/app/backend/data/webui.db"


class StopOutcome(StrEnum):
    """Result of :func:`stop_ollama_container`."""

    DOCKER_MISSING = "docker_missing"
    NOT_RUNNING = "not_running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class WebUIPin:
    """One row from the WebUI ``model`` table."""

    id: str
    name: str
    base_model_id: str


def is_docker_available(which: Callable[[str], str | None] = shutil.which) -> bool:
    """Return ``True`` when the ``docker`` binary is on PATH."""
    return which("docker") is not None


def stop_ollama_container(
    *,
    container: str = DEFAULT_OLLAMA_CONTAINER,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    logger: Callable[[str], None] = print,
) -> StopOutcome:
    """Stop the Harbor Ollama container if it's running.

    Stopping the container frees GPU memory before a competing vLLM unit
    starts. Idempotent: returns :attr:`StopOutcome.NOT_RUNNING` cleanly
    when the container is already stopped.
    """
    if not is_docker_available(which):
        return StopOutcome.DOCKER_MISSING

    if not _is_container_running(container, runner):
        return StopOutcome.NOT_RUNNING

    try:
        result = _run(["docker", "stop", container], runner, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger(f"  harbor: timed out stopping {container}")
        return StopOutcome.FAILED

    if result.returncode != 0:
        logger(f"  harbor: failed to stop {container}: {result.stderr.strip()}")
        return StopOutcome.FAILED

    logger(f"  harbor: stopped {container} (freed GPU memory)")
    return StopOutcome.STOPPED


def webui_custom_models(
    *,
    container: str = DEFAULT_WEBUI_CONTAINER,
    db_path: str = DEFAULT_WEBUI_DB_PATH,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[WebUIPin] | None:
    """List OpenWebUI custom models pinned by ``base_model_id``.

    Returns ``None`` when Docker is missing, the container isn't
    running, or sqlite query fails. The caller treats ``None`` as
    "skip the pin-validation check, no harm done."
    """
    if not is_docker_available(which):
        return None
    if not _is_container_running(container, runner):
        return None

    try:
        result = _run(
            [
                "docker", "exec", container, "sqlite3", db_path,
                "SELECT id || '|' || name || '|' || "
                "COALESCE(base_model_id, '') FROM model;",
            ],
            runner,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    pins: list[WebUIPin] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|", 2)
        if len(parts) == 3:
            pins.append(WebUIPin(id=parts[0], name=parts[1], base_model_id=parts[2]))
    return pins


def served_model_ids(
    port: int,
    *,
    timeout_s: float = 2.0,
    http_get: Callable[[str, float], Any] | None = None,
) -> set[str] | None:
    """Query ``/v1/models`` on ``port`` and return the set of served IDs.

    Returns ``None`` when the endpoint is unreachable (treated as
    "service down"), an empty set when reachable with no models loaded.
    Used together with :func:`webui_custom_models` to warn about pins
    that would break under a swap.
    """
    url = f"http://localhost:{port}/v1/models"
    try:
        if http_get is not None:
            resp = http_get(url, timeout_s)
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        else:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310 - localhost only
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    ids: set[str] = set()
    for m in payload.get("data", []):
        mid = m.get("id")
        if isinstance(mid, str):
            ids.add(mid)
    return ids


def find_orphaned_pins(
    next_served_name: str,
    *,
    container: str = DEFAULT_WEBUI_CONTAINER,
    db_path: str = DEFAULT_WEBUI_DB_PATH,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[WebUIPin] | None:
    """List WebUI pins whose ``base_model_id`` won't match after the swap.

    A pin "matches" if its ``base_model_id`` equals ``next_served_name``.
    Pins with empty ``base_model_id`` are not considered orphaned (they
    don't pin anything). Returns ``None`` when WebUI is unavailable.
    """
    pins = webui_custom_models(
        container=container,
        db_path=db_path,
        runner=runner,
        which=which,
    )
    if pins is None:
        return None
    return [p for p in pins if p.base_model_id and p.base_model_id != next_served_name]


def _is_container_running(
    container: str,
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None,
) -> bool:
    """Return ``True`` when ``docker inspect`` reports the container Running."""
    try:
        result = _run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            runner,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.stdout.strip() == "true"


def _run(
    argv: list[str],
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None,
    *,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Invoke ``argv`` via the injected runner or :func:`subprocess.run`."""
    if runner is not None:
        return runner(argv)
    return subprocess.run(  # noqa: S603 - argv is constants + caller-validated container name
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
