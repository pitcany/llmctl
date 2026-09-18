"""Health service.

Aggregates a conservative health snapshot across configuration, database, GPU
telemetry, and per-runtime adapter availability.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llmctl._asyncutil import run_sync
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
        db: Any | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._router = router
        # Optional: with a DB session, health also reports what is actually
        # SERVING per runtime. Without one it degrades to binary presence,
        # which is what every caller got before.
        self._db = db

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
            results = run_sync(probe_all())
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

    def _serving_by_runtime(self) -> dict[str, list[dict[str, Any]]]:
        """Return the live (non-terminal) sessions grouped by runtime value.

        Answers "what is actually serving right now", which per-runtime binary
        presence cannot: ``llama.cpp binary found`` is true whether or not any
        server is running, and reads as healthy when nothing is up.

        Best-effort: any DB problem yields an empty mapping rather than
        failing the whole health call — a degraded extra is better than no
        health output.
        """
        if self._db is None:
            return {}
        try:
            from llmctl.db import SessionStatus
            from llmctl.services.sessions import SessionService

            live = {SessionStatus.RUNNING, SessionStatus.STARTING, SessionStatus.DEGRADED}
            grouped: dict[str, list[dict[str, Any]]] = {}
            for session in SessionService(self._db, self.settings).list_sessions():
                if session.status not in live:
                    continue
                grouped.setdefault(session.runtime.value, []).append(
                    {
                        "id": session.id,
                        "status": session.status.value,
                        "served_name": session.served_name,
                        "endpoint_url": session.endpoint_url,
                        "kind": (session.kind.value if session.kind else None),
                        "managed": True,
                    }
                )
            return grouped
        except Exception:
            return {}

    def _loaded_by_runtime(self) -> dict[str, list[dict[str, Any]]]:
        """Return models each runtime reports as loaded, grouped by runtime.

        Complements :meth:`_serving_by_runtime`, which can only see what llmctl
        launched. A runtime started by hand -- a systemd vLLM unit, or a model
        Ollama pulled into memory on first request -- has no SessionRecord, so
        the DB lookup reports nothing while the GPU is fully committed.

        ``list_loaded_models`` returning ``None`` means the runtime cannot
        answer; that is reported as no entries rather than as "idle".

        Best-effort: a probe that fails yields no entries for that runtime
        rather than failing the whole health call.
        """
        order = list(self.router.list_runtimes())

        async def probe_all() -> list[Any]:
            return await asyncio.gather(
                *(self.router.get_adapter(runtime).list_loaded_models() for runtime in order),
                return_exceptions=True,
            )

        try:
            results = run_sync(probe_all())
        except Exception:
            return {}

        grouped: dict[str, list[dict[str, Any]]] = {}
        for runtime, result in zip(order, results, strict=True):
            if isinstance(result, BaseException) or not result:
                continue
            for model in result:
                grouped.setdefault(runtime.value, []).append(
                    {
                        "id": None,
                        "status": "running",
                        "served_name": model.name,
                        "endpoint_url": None,
                        "kind": None,
                        "managed": False,
                    }
                )
        return grouped

    @staticmethod
    def _merge_serving(
        managed: list[dict[str, Any]], loaded: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return managed sessions plus loaded models llmctl did not launch.

        A model llmctl started is reported by both sources; the managed record
        wins because it carries the session id and endpoint.
        """
        known = {s.get("served_name") for s in managed if s.get("served_name")}
        extra = [m for m in loaded if m.get("served_name") not in known]
        return [*managed, *extra]

    def get_health(self) -> dict[str, Any]:
        """Return a conservative health snapshot."""
        gpus = get_gpu_info()
        runtimes = self._runtime_health()
        serving = self._serving_by_runtime()
        loaded = self._loaded_by_runtime()
        for name, info in runtimes.items():
            info["sessions"] = self._merge_serving(serving.get(name, []), loaded.get(name, []))
            info["serving_count"] = len(info["sessions"])
            info["unmanaged_count"] = sum(
                1 for s in info["sessions"] if not s.get("managed", True)
            )
        return {
            "state": HealthState.OK,
            "safe_mode": self.settings.app.safe_mode,
            "database_url": self.settings.database_url,
            "gpu_count": len(gpus),
            "nvml_available": nvml_available(),
            # False when no DB was supplied: callers must not read
            # serving_count == 0 as "nothing is running" in that case.
            "session_state_available": self._db is not None,
            "runtimes": runtimes,
            "message": "LLM Mission Control is running.",
        }
