"""Health service.

Aggregates a conservative health snapshot across configuration, database, GPU
telemetry, and per-runtime adapter availability.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llmctl.config import Settings, load_settings
from llmctl.schemas import HealthState
from llmctl.services.router import RuntimeRouter
from llmctl.telemetry.gpu import get_gpu_info, nvml_available


class HealthService:
    """Aggregates health for config, database, runtimes, and GPUs."""

    def __init__(
        self,
        settings: Settings | None = None,
        router: RuntimeRouter | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._router = router

    @property
    def router(self) -> RuntimeRouter:
        """Return the runtime router, constructing a default on first use."""
        if self._router is None:
            self._router = RuntimeRouter(self.settings)
        return self._router

    def _runtime_health(self) -> dict[str, dict[str, Any]]:
        """Return per-runtime adapter health states (probed concurrently)."""
        order = list(self.router.list_runtimes())

        async def probe_all() -> list[Any]:
            return await asyncio.gather(
                *(self.router.get_adapter(runtime).health_check() for runtime in order),
                return_exceptions=True,
            )

        runtimes: dict[str, dict[str, Any]] = {}
        try:
            results = asyncio.run(probe_all())
        except Exception as exc:  # event-loop level failure; report on every runtime
            results = [exc] * len(order)
        for runtime, result in zip(order, results, strict=True):
            if isinstance(result, BaseException):
                runtimes[runtime.value] = {
                    "state": HealthState.UNKNOWN.value,
                    "message": f"Health check error: {result}",
                }
            else:
                runtimes[runtime.value] = {
                    "state": result.state.value,
                    "message": result.message,
                }
        return runtimes

    def get_health(self) -> dict[str, Any]:
        """Return a conservative health snapshot."""
        gpus = get_gpu_info()
        return {
            "state": HealthState.OK,
            "safe_mode": self.settings.app.safe_mode,
            "database_url": self.settings.database_url,
            "gpu_count": len(gpus),
            "nvml_available": nvml_available(),
            "runtimes": self._runtime_health(),
            "message": "LLM Mission Control is running.",
        }
