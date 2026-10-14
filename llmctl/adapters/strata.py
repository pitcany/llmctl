"""Strata runtime adapter (github.com/Niko1221/Strata).

Strata serves Qwen3.8-Flash-Next variants from one GPU plus system RAM with
an OpenAI-compatible API. Discovery and health use ``/v1/models``; llmctl
starts it as ``<venv python> serve/server.py --engine strata --config X.json``
(see :mod:`llmctl.services.scheduler`), where the config JSON is the model.
"""

from __future__ import annotations

from typing import Any

from llmctl.adapters._common import ClientFactory, LaunchableHttpAdapter
from llmctl.config import RuntimeConfig, default_runtime_configs
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import Model

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"


class StrataAdapter(LaunchableHttpAdapter):
    """Adapter for a Strata server."""

    def __init__(
        self,
        config: RuntimeConfig | str | None = None,
        supervisor: Any = None,
        *,
        endpoint: str | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        if isinstance(config, str):
            endpoint, config = config, None
        super().__init__(
            RuntimeName.STRATA,
            "Strata",
            config or default_runtime_configs()["strata"],
            supervisor,
            endpoint=endpoint,
            health_path="/v1/models",
            client_factory=client_factory,
        )

    @property
    def models_path(self) -> str:
        return "/v1/models"

    def capabilities(self) -> dict[str, bool]:
        caps = super().capabilities()
        caps["list_loaded_models"] = True
        return caps

    def _parse_models(self, data: object) -> list[Model]:
        rows = data.get("data") if isinstance(data, dict) else None
        models: list[Model] = []
        for row in rows or []:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            models.append(
                Model(
                    name=row["id"],
                    runtime=RuntimeName.STRATA,
                    source=row["id"],
                    status=ModelStatus.DISCOVERED,
                    metadata={"endpoint": self.endpoint},
                )
            )
        return models

    async def list_loaded_models(self) -> list[Model] | None:
        return await self.discover_models()
