"""Runtime inventory service.

Builds one normalized view of every configured runtime adapter — health,
version, endpoint, capability flags, and currently loaded models — so the
CLI, TUI, and API can answer "what runtimes exist and what can each do?"
from the same code path.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llmctl.config import Settings
from llmctl.schemas import HealthState
from llmctl.services.router import RuntimeRouter


def runtime_inventory(
    settings: Settings | None = None,
    router: RuntimeRouter | None = None,
) -> list[dict[str, Any]]:
    """Return a normalized inventory row per configured runtime.

    Each row contains::

        runtime, display_name, state, message, endpoint, version,
        capabilities (stable-key bool map), loaded (served model names or
        None when the runtime cannot answer)

    All probes run concurrently with the adapters' own timeouts; a probe
    failure degrades that row instead of failing the inventory.
    """
    router = router or RuntimeRouter(settings)
    order = list(router.list_runtimes())

    async def inspect_one(runtime) -> dict[str, Any]:
        adapter = router.get_adapter(runtime)
        health, version, loaded = await asyncio.gather(
            adapter.health_check(),
            adapter.version(),
            adapter.list_loaded_models(),
            return_exceptions=True,
        )
        if isinstance(health, BaseException):
            state, message = HealthState.UNKNOWN.value, f"probe error: {health}"
        else:
            state, message = health.state.value, health.message
        return {
            "runtime": runtime.value,
            "display_name": getattr(adapter, "display_name", runtime.value),
            "state": state,
            "message": message,
            "endpoint": getattr(adapter, "endpoint", None),
            "version": None if isinstance(version, BaseException) else version,
            "capabilities": adapter.capabilities(),
            "loaded": (
                None
                if isinstance(loaded, BaseException) or loaded is None
                else [model.name for model in loaded]
            ),
        }

    async def inspect_all() -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(inspect_one(rt) for rt in order)))

    return asyncio.run(inspect_all())
