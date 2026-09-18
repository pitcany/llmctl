"""FreeToken runtime adapter.

FreeToken (``ft serve``) serves an OpenAI-compatible HTTP API. Discovery uses
``GET /v1/models``, whose rows carry ``max_model_len`` — the served context
window — plus FreeToken's reasoning-effort fields.

Health differs from the other HTTP runtimes: ``/health`` answers **200 while the
model is still loading**, with ``{"status": "loading"}`` in the body. Readiness
is ``status == "ok"``, so the inherited reachable-means-healthy probe would
report a server that cannot answer a request yet as OK. The same body carries
the runtime version; FreeToken 0.1.3 has no separate version endpoint.

**This adapter deliberately cannot launch or stop anything.** On this box the
FreeToken lane is a systemd user unit (``qwen3.8-flash-next-ft``) whose
``ExecStartPre`` guards free VRAM on the x16 GPU and whose wrapper
(``qwen38-ft-start --take-gpus``) evicts the other big-MoE units first. Same
posture as the SGLang and llama.cpp lanes: llmctl *tracks* the lane, systemd
*runs* it (``llmctl start-unit`` / ``stop -s`` drive the unit). The unit adopts
itself on start and detaches on stop.

Before this adapter existed the lane was adopted as ``RuntimeName.OPENAI``, the
generic catch-all, so it could not be told apart from LiteLLM or a remote box
and reported no version.
"""

from __future__ import annotations

from llmctl.adapters._common import ClientFactory, HttpRuntimeAdapter
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import AdapterStatus, HealthState, Model

DEFAULT_ENDPOINT = "http://127.0.0.1:1919"


class FreeTokenAdapter(HttpRuntimeAdapter):
    """Adapter for a FreeToken server. Discovery, health and version only."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        super().__init__(
            RuntimeName.FREETOKEN,
            "FreeToken",
            endpoint,
            health_path="/health",
            client_factory=client_factory,
        )

    @property
    def models_path(self) -> str:
        """FreeToken serves the OpenAI listing at ``/v1/models``."""
        return "/v1/models"

    def capabilities(self) -> dict[str, bool]:
        """Discovery, health and version — never process control.

        ``launch_process`` and ``stop_process`` stay False on purpose: the
        systemd unit and its VRAM guard own the lifecycle. See the module
        docstring.
        """
        caps = super().capabilities()
        caps["list_loaded_models"] = True
        caps["version"] = True
        return caps

    async def health_check(self) -> AdapterStatus:
        """OK only when ``/health`` reports ``status == "ok"``.

        A 200 with ``"status": "loading"`` is reachable but not ready, and is
        reported DEGRADED so callers do not route requests to it yet.
        """
        ok, data, error = await self._get_json(self.health_path)
        details: dict[str, object] = {"endpoint": self.endpoint}
        if not ok:
            details["error"] = error
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.UNAVAILABLE,
                message=f"{self.display_name} is not reachable at {self.endpoint}.",
                details=details,
            )
        status = data.get("status") if isinstance(data, dict) else None
        if isinstance(data, dict):
            for key in ("model", "version", "maintenance"):
                if data.get(key) is not None:
                    details[key] = data[key]
        if status == "ok":
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.OK,
                message=f"{self.display_name} is serving at {self.endpoint}.",
                details=details,
            )
        details["status"] = status
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.DEGRADED,
            message=(
                f"{self.display_name} is reachable at {self.endpoint} but not ready "
                f"(status={status!r})."
            ),
            details=details,
        )

    async def version(self) -> str | None:
        """Return the FreeToken version from the ``/health`` body, or None."""
        ok, data, _ = await self._get_json(self.health_path)
        if not ok or not isinstance(data, dict):
            return None
        version = data.get("version")
        return str(version) if version else None

    def _parse_models(self, data: object) -> list[Model]:
        """Parse FreeToken's ``/v1/models`` payload into models.

        ``max_model_len`` maps onto the registry's ``max_context``. One server
        serves one model, so anything listed here is loaded.
        """
        if not isinstance(data, dict):
            return []
        rows = data.get("data")
        if not isinstance(rows, list):
            return []
        models: list[Model] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("id")
            if not name:
                continue
            max_context = row.get("max_model_len", row.get("context_length"))
            models.append(
                Model(
                    name=str(name),
                    runtime=RuntimeName.FREETOKEN,
                    source=str(row.get("root") or name),
                    status=ModelStatus.DISCOVERED,
                    max_context=int(max_context) if isinstance(max_context, int) else None,
                    metadata={
                        "owned_by": row.get("owned_by"),
                        "default_reasoning_effort": row.get("default_reasoning_effort"),
                    },
                )
            )
        return models

    async def list_loaded_models(self) -> list[Model] | None:
        """Return the served model. One server, one model — so same as discovery."""
        return await self.discover_models()
