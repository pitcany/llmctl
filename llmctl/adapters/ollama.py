"""Ollama runtime adapter.

Talks to a local Ollama daemon over its native HTTP API. Discovery uses
``GET /api/tags`` and health uses ``GET /api/version``. Model deletion maps to
``DELETE /api/delete``.
"""

from __future__ import annotations

from llmctl.adapters._common import ClientFactory, HttpRuntimeAdapter
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import AdapterStatus, HealthState, Model

DEFAULT_ENDPOINT = "http://127.0.0.1:11434"


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
