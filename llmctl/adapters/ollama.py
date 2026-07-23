"""Ollama runtime adapter.

Talks to a local Ollama daemon over its native HTTP API. Discovery uses
``GET /api/tags`` and health uses ``GET /api/version``. Model deletion maps to
``DELETE /api/delete`` and pulling to a streaming ``POST /api/pull``.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from llmctl.adapters._common import ClientFactory, HttpRuntimeAdapter
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import AdapterStatus, HealthState, Model

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

#: Pull progress callback: ``(status, completed_bytes, total_bytes)``.
#: Sizes are ``None`` for phases that don't report them (manifest, verify).
PullProgress = Callable[[str, int | None, int | None], None]


class OllamaAdapter(HttpRuntimeAdapter):
    """Adapter for the Ollama runtime."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout: float = 5.0,
        client_factory: ClientFactory | None = None,
    ) -> None:
        super().__init__(
            RuntimeName.OLLAMA,
            "Ollama",
            endpoint or DEFAULT_ENDPOINT,
            health_path="/api/version",
            timeout=timeout,
            client_factory=client_factory,
        )

    @property
    def models_path(self) -> str:
        """Ollama model listing endpoint."""
        return "/api/tags"

    def capabilities(self) -> dict[str, bool]:
        """Ollama additionally supports remote deletion and version reporting."""
        caps = super().capabilities()
        caps.update({"delete_model": True, "version": True})
        return caps

    async def version(self) -> str | None:
        """Return the daemon version from ``GET /api/version``."""
        ok, data, _ = await self._get_json("/api/version")
        if ok and isinstance(data, dict) and isinstance(data.get("version"), str):
            return data["version"]
        return None

    async def list_loaded_models(self) -> list[Model] | None:
        """Return models currently loaded into memory (``GET /api/ps``)."""
        ok, data, _ = await self._get_json("/api/ps")
        if not ok or not isinstance(data, dict):
            return None
        return self._parse_models(data)

    def _parse_models(self, data: object) -> list[Model]:
        """Parse the ``/api/tags`` payload into models."""
        if not isinstance(data, dict):
            return []
        models: list[Model] = []
        for item in data.get("models", []):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if not name:
                continue
            details = item.get("details") or {}
            models.append(
                Model(
                    name=str(name),
                    runtime=RuntimeName.OLLAMA,
                    source=str(name),
                    format=details.get("format"),
                    quantization=details.get("quantization_level"),
                    size_bytes=item.get("size"),
                    status=ModelStatus.DISCOVERED,
                    metadata={
                        "digest": item.get("digest"),
                        "parameter_size": details.get("parameter_size"),
                        "family": details.get("family"),
                    },
                )
            )
        return models

    async def pull_model(
        self, name: str, *, on_progress: PullProgress | None = None
    ) -> AdapterStatus:
        """Pull ``name`` into the Ollama library via streaming ``POST /api/pull``.

        Emits one ``on_progress`` call per NDJSON event. Pulls run
        minutes-to-hours, so the read timeout is unbounded while connect
        keeps the adapter's normal budget. Never raises — errors (daemon
        down, unknown tag, mid-pull failure event) come back as a
        ``DEGRADED`` :class:`AdapterStatus`.
        """
        timeout = httpx.Timeout(self.timeout, read=None)
        try:
            async with self._client() as client:
                async with client.stream(
                    "POST",
                    "/api/pull",
                    json={"name": name, "stream": True},
                    timeout=timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("error"):
                            return AdapterStatus(
                                runtime=self.runtime,
                                state=HealthState.DEGRADED,
                                message=f"Ollama pull of '{name}' failed: {event['error']}",
                            )
                        if on_progress is not None:
                            completed = event.get("completed")
                            total = event.get("total")
                            on_progress(
                                str(event.get("status") or ""),
                                completed if isinstance(completed, int) else None,
                                total if isinstance(total, int) else None,
                            )
        except Exception as exc:
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.DEGRADED,
                message=f"Ollama pull of '{name}' failed: {exc}",
            )
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.OK,
            message=f"Pulled Ollama model '{name}'.",
        )

    async def delete_model(self, model: Model) -> AdapterStatus:
        """Delete a model from the Ollama library via ``DELETE /api/delete``."""
        name = model.source or model.name
        try:
            async with self._client() as client:
                response = await client.request(
                    "DELETE", "/api/delete", json={"name": name}
                )
                response.raise_for_status()
        except Exception as exc:
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.DEGRADED,
                message=f"Failed to delete Ollama model '{name}': {exc}",
            )
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.OK,
            message=f"Deleted Ollama model '{name}'.",
        )
