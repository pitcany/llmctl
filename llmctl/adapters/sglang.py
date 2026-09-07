"""SGLang runtime adapter.

SGLang serves an OpenAI-compatible HTTP API, so discovery and health both use
``GET /v1/models``. Its listing is richer than the OpenAI baseline: each entry
carries ``max_model_len``, which is the served context window rather than the
checkpoint's native one — the number that actually decides whether a harness
fits.

**This adapter deliberately cannot launch or stop anything.** On this box the
SGLang lane is a systemd user unit whose wrapper (``sglang-qwen38-start``)
performs a free-VRAM guard and evicts ``vllm-tp`` before exec'ing the server;
putting that lifecycle behind llmctl as well would give two owners to one
transaction. Same posture as the llama.cpp units: llmctl *tracks* the lane,
systemd *runs* it. The unit adopts itself on start and detaches on stop.

Before this adapter existed the lane was adopted as ``RuntimeName.OPENAI``, the
generic catch-all. That worked but reported the runtime as a bare OpenAI
endpoint, with no version and no way to tell it apart from LiteLLM or a remote
box in ``llmctl health``.
"""

from __future__ import annotations

from llmctl.adapters._common import ClientFactory, HttpRuntimeAdapter
from llmctl.db import ModelStatus, RuntimeName
from llmctl.schemas import AdapterStatus, HealthState, Model

DEFAULT_ENDPOINT = "http://127.0.0.1:30000"

#: SGLang's own server-info endpoint. Carries the runtime version plus the
#: resolved launch arguments, which is where the served context length and the
#: speculative-decoding algorithm live.
SERVER_INFO_PATH = "/get_server_info"


class SGLangAdapter(HttpRuntimeAdapter):
    """Adapter for an SGLang server. Discovery, health and version only."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        super().__init__(
            RuntimeName.SGLANG,
            "SGLang",
            endpoint,
            health_path="/health",
            client_factory=client_factory,
        )

    @property
    def models_path(self) -> str:
        """SGLang serves the OpenAI listing; discovery and health differ here.

        Health uses ``/health`` (cheap, no model work) while discovery uses
        ``/v1/models``, so the two are deliberately not the same path.
        """
        return "/v1/models"

    def capabilities(self) -> dict[str, bool]:
        """Discovery, health and version — never process control.

        ``launch_process`` and ``stop_process`` stay False on purpose: the
        systemd unit and its guard wrapper own the lifecycle. See the module
        docstring.
        """
        caps = super().capabilities()
        caps["list_loaded_models"] = True
        caps["version"] = True
        return caps

    async def health_check(self) -> AdapterStatus:
        """Probe ``/health`` by status code, not by parsing a body.

        SGLang answers ``/health`` with a bare 200 and a **zero-byte body**, so
        the inherited JSON-parsing probe reports a healthy server as
        unavailable. ``/get_server_info`` would parse, but it returns ~32 KB of
        resolved launch arguments — far too heavy to poll as a health check.
        """
        try:
            async with self._client() as client:
                response = await client.get(self.health_path)
            reachable = response.status_code < 400
            error: str | None = None if reachable else f"HTTP {response.status_code}"
        except Exception as exc:  # connection refused / timeout
            reachable, error = False, str(exc)
        if reachable:
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.OK,
                message=f"{self.display_name} is reachable at {self.endpoint}.",
                details={"endpoint": self.endpoint},
            )
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.UNAVAILABLE,
            message=f"{self.display_name} is not reachable at {self.endpoint}.",
            details={"endpoint": self.endpoint, "error": error},
        )

    async def version(self) -> str | None:
        """Return the SGLang version from ``/get_server_info``, or None."""
        ok, data, _ = await self._get_json(SERVER_INFO_PATH)
        if not ok or not isinstance(data, dict):
            return None
        version = data.get("version")
        return str(version) if version else None

    def _parse_models(self, data: object) -> list[Model]:
        """Parse an OpenAI ``/v1/models`` payload into models.

        SGLang adds ``max_model_len`` per entry; it maps onto the registry's
        ``max_context`` so ``llmctl models`` shows the window the server is
        *actually* serving, not the checkpoint's native one. A server serves
        exactly one model, so anything answering here is loaded, not merely
        installed.
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
            max_context = row.get("max_model_len")
            models.append(
                Model(
                    name=str(name),
                    runtime=RuntimeName.SGLANG,
                    source=str(row.get("root") or name),
                    status=ModelStatus.DISCOVERED,
                    max_context=int(max_context) if isinstance(max_context, int) else None,
                    metadata={"owned_by": row.get("owned_by")},
                )
            )
        return models

    async def list_loaded_models(self) -> list[Model] | None:
        """Return the served model. One server, one model — so same as discovery."""
        return await self.discover_models()
