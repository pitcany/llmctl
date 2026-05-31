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
        """Return per-runtime adapter health states."""
        runtimes: dict[str, dict[str, Any]] = {}
        for runtime in self.router.list_runtimes():
            adapter = self.router.get_adapter(runtime)
            try:
                status = asyncio.run(adapter.health_check())
                runtimes[runtime.value] = {
                    "state": status.state.value,
                    "message": status.message,
                }
            except Exception as exc:
                runtimes[runtime.value] = {
                    "state": HealthState.UNKNOWN.value,
                    "message": f"Health check error: {exc}",
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
