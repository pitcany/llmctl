"""Health service.

Aggregates a conservative health snapshot across configuration, database, GPU
telemetry, and per-runtime adapter availability.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from llmctl._asyncutil import run_sync
from llmctl.config import Settings, load_settings
from llmctl.schemas import HealthState
from llmctl.services.backends import probe_ollama_loaded
from llmctl.services.engines import ENGINE_NAMES
from llmctl.services.lanes import Lane, load_lanes
from llmctl.services.router import RuntimeRouter
from llmctl.telemetry.gpu import get_gpu_info, nvml_available

#: Seconds a lane may take to answer its ``/v1/models`` probe.
LANE_PROBE_TIMEOUT_S = 2.0

LaneProbe = Callable[[Lane], Awaitable[bool]]


async def _default_lane_probe(lane: Lane) -> bool:
    """True when the lane answers its probe.

    Ollama lanes answer only while their model is loaded (``/api/ps``): its
    ``/v1/models`` lists every installed model, so it cannot tell lanes apart.
    Other lanes answer when ``GET /v1/models`` returns HTTP 200.
    """
    if lane.runtime == "ollama":
        try:
            loaded = await asyncio.to_thread(
                probe_ollama_loaded, lane.upstream, LANE_PROBE_TIMEOUT_S
            )
        except Exception:  # noqa: BLE001 - an unreachable lane is simply not answering
            return False
        return bool(lane.served_model) and lane.served_model in (loaded or [])
    try:
        import httpx

        async with httpx.AsyncClient(timeout=LANE_PROBE_TIMEOUT_S) as client:
            response = await client.get(f"{lane.upstream}/v1/models")
        return response.status_code == 200
    except Exception:  # noqa: BLE001 - an unreachable lane is simply not answering
        return False


#: The adapters' "start it" hint; wrong next to a live session.
_START_HINT = re.compile(r";?\s*`llmctl run NAME` starts it\.?")


def _without_start_hint(message: str) -> str:
    out = _START_HINT.sub("", message).rstrip().rstrip(";").rstrip()
    return out if out.endswith(".") else f"{out}."


class HealthService:
    """Aggregates health for config, database, runtimes, and GPUs."""

    def __init__(
        self,
        settings: Settings | None = None,
        router: RuntimeRouter | None = None,
        db: Any | None = None,
        *,
        lanes: Sequence[Lane] | None = None,
        engines: Any | None = None,
        model_users: Any | None = None,
        lane_probe: LaneProbe | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self._lanes = list(lanes) if lanes is not None else None
        self._engines = engines
        self._model_users = model_users
        self._lane_probe: LaneProbe = lane_probe or _default_lane_probe
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

    def _load_lanes(self) -> list[Lane]:
        """Return the injected lanes, or what the lane sources load ([] on failure)."""
        if self._lanes is not None:
            return self._lanes
        try:
            return list(load_lanes(self.settings.managed_units))
        except Exception:  # noqa: BLE001 - lanes are advisory
            return []

    def _runtime_health(
        self, lanes: Sequence[Lane] = ()
    ) -> tuple[dict[str, dict[str, Any]], list[bool]]:
        """Return per-runtime adapter health and per-lane answers (probed concurrently)."""
        order = list(self.router.list_runtimes())

        async def probe_lane(lane: Lane) -> bool:
            try:
                return bool(await self._lane_probe(lane))
            except Exception:  # noqa: BLE001 - a probe that raises is not answering
                return False

        async def probe_all() -> list[Any]:
            return await asyncio.gather(
                *(self.router.get_adapter(runtime).health_check() for runtime in order),
                *(probe_lane(lane) for lane in lanes),
                return_exceptions=True,
            )

        runtimes: dict[str, dict[str, Any]] = {}
        try:
            gathered = run_sync(probe_all())
        except Exception as exc:  # event-loop level failure; report on every runtime
            gathered = [exc] * len(order) + [False] * len(lanes)
        results = gathered[: len(order)]
        answers = [r is True for r in gathered[len(order) :]]
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
        return runtimes, answers

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

    def _apply_engine_rules(
        self, runtimes: dict[str, dict[str, Any]], lanes: Sequence[Lane], answers: list[bool]
    ) -> None:
        """Overlay lane reality on engine runtimes (rules A, B, F); others are untouched."""
        for name, info in runtimes.items():
            if name not in ENGINE_NAMES:
                continue
            mine = [(ln, ok) for ln, ok in zip(lanes, answers, strict=True) if ln.runtime == name]
            up = [ln for ln, ok in mine if ok]
            # Lanes of this runtime that did not answer: listed so a stopped lane
            # (e.g. glm53 beside a running strata lane) stays visible in `health`.
            info["stopped_lanes"] = [
                {"name": ln.name, "upstream": ln.upstream} for ln, ok in mine if not ok
            ]
            sessions: list[dict[str, Any]] = info["sessions"]
            if up:  # A: a lane answering its probe is serving
                lane = up[0]
                info["state"] = HealthState.OK.value
                info["message"] = f"serving via lane {lane.name} at {lane.upstream}"
                info["lane"] = lane.name
                known = {s.get("served_name") for s in sessions}
                for answering in up:
                    if answering.served_model and answering.served_model not in known:
                        known.add(answering.served_model)
                        sessions.append(
                            {
                                "id": None,
                                "status": "running",
                                "served_name": answering.served_model,
                                "endpoint_url": answering.upstream,
                                "kind": None,
                                "managed": False,
                            }
                        )
            elif sessions:  # F: a runtime with a session is never "not running"
                message = str(info.get("message", ""))
                if "not running" in message:
                    message = message.replace(" and not running", "").replace(
                        "not running", "serving"
                    )
                    info["message"] = _without_start_hint(message)
            elif mine:  # B: lanes exist, none answers, nothing else serves
                lane = mine[0][0]
                info["state"] = HealthState.OK.value
                info["message"] = (
                    f"lane {lane.name} is not running ({lane.upstream}); "
                    f"`llmctl run {lane.name}` starts it."
                )
            owners = {ln.served_model: ln.name for ln, _ in mine if ln.served_model}
            for s in sessions:
                owner = owners.get(s.get("served_name"))
                if owner:
                    s["lane"] = owner
            info["serving_count"] = len(sessions)
            info["unmanaged_count"] = sum(1 for s in sessions if not s.get("managed", True))

    def _engine_rows(self, lanes: Sequence[Lane]) -> dict[str, Any]:
        """Return engine name -> EngineStatus (injected, or what ``setup`` would show)."""
        rows = self._engines
        if rows is None:
            try:
                from llmctl.services.engines import engine_statuses

                # No version probes: health must stay fast.
                rows = engine_statuses(
                    self.settings,
                    lanes=lanes,
                    run=lambda argv: subprocess.CompletedProcess(list(argv), 1, "", ""),
                )
            except Exception:  # noqa: BLE001 - engine rows are advisory
                rows = []
        return {row.engine: row for row in rows}

    def _users_by_runtime(self) -> dict[str, list[str]]:
        if self._model_users is not None:
            return dict(self._model_users)
        try:
            from llmctl.services import doctor

            return doctor._model_users(self.settings) or {}
        except Exception:  # noqa: BLE001 - model users are advisory
            return {}

    def _apply_status_rules(
        self, runtimes: dict[str, dict[str, Any]], lanes: Sequence[Lane], answers: list[bool]
    ) -> None:
        """Overlay engine rows and model users (rules C, D, E) on missing engines."""
        rows = self._engine_rows(lanes)
        missing = {n for n, r in rows.items() if r.source == "missing" and n in runtimes}
        if not missing:
            return
        from llmctl.services.doctor import _name_models

        users = self._users_by_runtime()
        for name in missing:
            info = runtimes[name]
            answering = any(
                ln.runtime == name and ok for ln, ok in zip(lanes, answers, strict=True)
            )
            if answering:  # A wins
                continue
            if info.get("sessions"):  # never NOT_USED; E: it is not an installed engine
                info["message"] = _without_start_hint(
                    re.sub(r" is installed \([^)]*\)", " has a live session", str(info["message"]))
                )
                continue
            if info.get("state") == HealthState.OK.value and " is installed (" not in str(
                info.get("message", "")
            ):
                # The adapter's HTTP probe answered (e.g. Ollama in Docker, no
                # binary on PATH): reachable is not "not used".
                continue
            names = list(users.get(name) or [])
            if not names:  # C
                info["state"] = HealthState.NOT_USED.value
                info["message"] = "not used"
            else:  # D
                info["state"] = HealthState.UNAVAILABLE.value
                info["message"] = (
                    f"{_name_models(names)} use {name} but it is not installed; "
                    f"`llmctl setup {name}` installs it."
                )

    def get_health(self) -> dict[str, Any]:
        """Return a conservative health snapshot."""
        gpus = get_gpu_info()
        lanes = self._load_lanes()
        runtimes, answers = self._runtime_health(lanes)
        serving = self._serving_by_runtime()
        loaded = self._loaded_by_runtime()
        for name, info in runtimes.items():
            info["sessions"] = self._merge_serving(serving.get(name, []), loaded.get(name, []))
            info["serving_count"] = len(info["sessions"])
            info["unmanaged_count"] = sum(1 for s in info["sessions"] if not s.get("managed", True))
        self._apply_engine_rules(runtimes, lanes, answers)
        self._apply_status_rules(runtimes, lanes, answers)
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
