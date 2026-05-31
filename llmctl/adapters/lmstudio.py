"""LM Studio runtime adapter.

LM Studio exposes an OpenAI-compatible HTTP server. Discovery and health both
use ``GET /v1/models``.
"""

from __future__ import annotations

from llmctl.adapters._common import ClientFactory, HttpRuntimeAdapter
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import Model

DEFAULT_ENDPOINT = "http://127.0.0.1:1234"


class LMStudioAdapter(HttpRuntimeAdapter):
    """Adapter for the LM Studio local server."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout: float = 5.0,
        client_factory: ClientFactory | None = None,
    ) -> None:
        super().__init__(
            RuntimeName.LMSTUDIO,
            "LM Studio",
            endpoint or DEFAULT_ENDPOINT,
            health_path="/v1/models",
            timeout=timeout,
            client_factory=client_factory,
        )

    @property
    def models_path(self) -> str:
        """OpenAI-compatible model listing endpoint."""
        return "/v1/models"

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
