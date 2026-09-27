"""LM Studio runtime adapter.

LM Studio exposes an OpenAI-compatible HTTP server. Discovery and health both
use ``GET /v1/models``.

The OpenAI ``/v1/models`` payload carries no device attribution, but a single
LM Studio server federates a whole LM Link fleet, so its listing mixes models
from several machines. To recover which device each model lives on, discovery
enriches the HTTP listing with the ``lms`` CLI (``lms link status --json`` maps
device identifiers to names; ``lms ls --json`` maps each model key to its
device identifier). Enrichment is best-effort: when the ``lms`` binary is
absent or any step fails, models simply carry no host and the registry treats
them as local.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from llmctl.adapters._common import ClientFactory, HttpRuntimeAdapter
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import Model

DEFAULT_ENDPOINT = "http://127.0.0.1:1234"

#: LM Studio's native listing. Unlike ``/v1/models`` it reports a per-model
#: ``state``, which is what separates resident models from merely downloaded.
LOADED_MODELS_PATH = "/api/v0/models"

#: Returns a ``{modelKey: device_name}`` map for the LM Studio fleet, or ``{}``
#: when device attribution can't be determined.
DeviceMapFn = Callable[[], dict[str, str]]

_LMS_TIMEOUT = 10.0


def lms_binary() -> str | None:
    """Path of the ``lms`` CLI, or ``None`` when it is not installed.

    LM Studio installs ``lms`` in ``~/.lmstudio/bin`` and puts that on PATH
    only from ``~/.bashrc``, so a TUI started from tmux, ssh or a launcher
    does not see it. Fall back to the install location.
    """
    found = shutil.which("lms")
    if found:
        return found
    installed = Path.home() / ".lmstudio" / "bin" / "lms"
    return str(installed) if installed.is_file() else None


def _run_lms_json(args: list[str], timeout: float) -> object | None:
    """Run ``lms <args> --json`` and return parsed JSON, or ``None`` on failure.

    Swallows every failure mode (binary missing, non-zero exit, timeout,
    malformed JSON) because device attribution is an optional enrichment: a
    broken ``lms`` must never fail the underlying HTTP discovery.
    """
    binary = lms_binary()
    if not binary:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed binary, no shell
            [binary, *args, "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return json.loads(completed.stdout)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def lms_device_map(timeout: float = _LMS_TIMEOUT) -> dict[str, str]:
    """Map each LM Studio ``modelKey`` to the LM Link device it lives on.

    Combines ``lms link status --json`` (device identifier -> device name, plus
    the local device) with ``lms ls --json`` (model key -> device identifier).
    Models on the local device report a null identifier and resolve to the
    local device name. Returns ``{}`` when the ``lms`` CLI is unavailable or
    either call fails, so callers degrade to no host attribution.
    """
    status = _run_lms_json(["link", "status"], timeout)
    if not isinstance(status, dict):
        return {}
    local_name = status.get("deviceName")
    id_to_name: dict[str, str] = {}
    for peer in status.get("peers", []):
        if not isinstance(peer, dict):
            continue
        identifier, name = peer.get("deviceIdentifier"), peer.get("deviceName")
        if identifier and name:
            id_to_name[identifier] = name

    listing = _run_lms_json(["ls"], timeout)
    if not isinstance(listing, list):
        return {}
    mapping: dict[str, str] = {}
    for entry in listing:
        if not isinstance(entry, dict):
            continue
        key = entry.get("modelKey")
        if not key:
            continue
        identifier = entry.get("deviceIdentifier")
        # A null identifier means the model is on this (local) machine.
        host = id_to_name.get(identifier) if identifier else local_name
        if host:
            mapping[str(key)] = host
    return mapping


class LMStudioAdapter(HttpRuntimeAdapter):
    """Adapter for the LM Studio local server."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout: float = 5.0,
        client_factory: ClientFactory | None = None,
        device_map_fn: DeviceMapFn | None = None,
    ) -> None:
        super().__init__(
            RuntimeName.LMSTUDIO,
            "LM Studio",
            endpoint or DEFAULT_ENDPOINT,
            health_path="/v1/models",
            timeout=timeout,
            client_factory=client_factory,
        )
        # Injectable for tests; defaults to the real ``lms`` CLI probe.
        self._device_map_fn = device_map_fn or lms_device_map

    @property
    def models_path(self) -> str:
        """OpenAI-compatible model listing endpoint."""
        return "/v1/models"

    async def discover_models(self) -> list[Model]:
        """Discover served models and attribute each to its LM Link device.

        The HTTP listing is the source of truth for *which* models exist (and
        for ``last_discovery_ok``); the ``lms`` CLI only annotates them with a
        host. A failed enrichment leaves ``host`` unset rather than dropping or
        duplicating any model.
        """
        models = await super().discover_models()
        if not models:
            return models
        try:
            device_map = await asyncio.to_thread(self._device_map_fn)
        except Exception:
            device_map = {}
        if not device_map:
            return models
        return [
            model.model_copy(update={"host": device_map.get(model.name, model.host)})
            for model in models
        ]

    async def list_loaded_models(self) -> list[Model] | None:
        """Return only models resident in memory, via ``GET /api/v0/models``.

        ``/v1/models`` cannot answer this: the OpenAI-compatible listing
        includes every downloaded model, loaded or not, so using it reports the
        whole catalog as serving. LM Studio's native endpoint carries a
        ``state`` field (``loaded`` / ``not-loaded``) and is the only way to
        tell them apart.

        When that endpoint is unavailable -- an older LM Studio, say -- this
        returns ``None`` (cannot answer) rather than falling back to the
        catalog, which would overstate what is running.
        """
        ok, data, _ = await self._get_json(LOADED_MODELS_PATH)
        if not ok or not isinstance(data, dict):
            return None
        resident = [
            item
            for item in data.get("data", [])
            if isinstance(item, dict) and item.get("state") == "loaded"
        ]
        return self._parse_models({"data": resident})

    def _parse_models(self, data: object) -> list[Model]:
        """Parse an OpenAI-style ``/v1/models`` payload into models."""
        if not isinstance(data, dict):
            return []
        models: list[Model] = []
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not model_id:
                continue
            models.append(
                Model(
                    name=str(model_id),
                    runtime=RuntimeName.LMSTUDIO,
                    source=str(model_id),
                    status=ModelStatus.DISCOVERED,
                    metadata={"owned_by": item.get("owned_by")},
                )
            )
        return models
