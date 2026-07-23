"""Shared base classes for runtime adapters.

Two reusable bases cover the two integration styles:

* :class:`HttpRuntimeAdapter` — runtimes exposed as long-lived HTTP servers
  (Ollama, LM Studio). Discovery and health use ``httpx``.
* :class:`ProcessRuntimeAdapter` — runtimes launched as child processes
  (vLLM, llama.cpp, arbitrary python scripts) supervised by
  :class:`~llmctl.telemetry.process.ProcessSupervisor`.

Every method degrades gracefully when the backing runtime is unavailable so the
control plane stays responsive on hosts without GPUs or installed runtimes.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from llmctl.adapters.base import RuntimeAdapter, SpawnCallback
from llmctl.config import RuntimeConfig
from llmctl.db import RuntimeName, SessionStatus, utcnow
from llmctl.discovery import discover_filesystem_models
from llmctl.schemas import AdapterStatus, HealthState, LaunchPlan, Model, Session
from llmctl.telemetry.process import ProcessSupervisor

ClientFactory = Callable[[], httpx.AsyncClient]


def _tail_log(log_path: str | None, lines: int = 5, max_bytes: int = 8192) -> str | None:
    """Return the last ``lines`` lines of a launch log, or ``None`` if unreadable."""
    if not log_path:
        return None
    try:
        path = Path(log_path)
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(size - max_bytes, 0))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    tail = [line for line in text.splitlines() if line.strip()][-lines:]
    return "\n".join(tail) if tail else None


class HttpRuntimeAdapter(RuntimeAdapter):
    """Base adapter for HTTP-server backed runtimes."""

    def __init__(
        self,
        runtime: RuntimeName,
        display_name: str,
        endpoint: str,
        *,
        health_path: str = "/",
        timeout: float = 5.0,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.runtime = runtime
        self.runtime_name = runtime.value
        self.display_name = display_name
        self.endpoint = endpoint.rstrip("/")
        self.health_path = health_path
        self.timeout = timeout
        self._client_factory = client_factory
        self._last_discovery_ok = True

    def _client(self) -> httpx.AsyncClient:
        """Return an HTTP client, honoring an injected factory for tests."""
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(base_url=self.endpoint, timeout=self.timeout)

    async def _get_json(self, path: str) -> tuple[bool, object | None, str | None]:
        """GET ``path`` and parse JSON, returning ``(ok, data, error)``."""
        try:
            async with self._client() as client:
                response = await client.get(path)
                response.raise_for_status()
                return True, response.json(), None
        except httpx.HTTPStatusError as exc:  # reachable but error status
            return False, None, f"HTTP {exc.response.status_code}"
        except Exception as exc:  # connection refused / timeout / parse error
            return False, None, str(exc)

    def _parse_models(self, data: object) -> list[Model]:  # pragma: no cover - overridden
        """Parse a runtime-specific discovery payload into models."""
        raise NotImplementedError

    async def discover_models(self) -> list[Model]:
        """Discover models exposed by the HTTP runtime; empty when unavailable."""
        ok, data, _ = await self._get_json(self.models_path)
        self._last_discovery_ok = ok
        if not ok or data is None:
            return []
        return self._parse_models(data)

    @property
    def last_discovery_ok(self) -> bool:
        """Whether the most recent ``discover_models`` HTTP call succeeded.

        ``discover_models`` returns ``[]`` both when the listing endpoint is
        unreachable/errors and when the catalog is genuinely empty. Callers
        that must not treat a failed listing as an empty catalog (the scan
        reconcile pass) consult this flag.
        """
        return self._last_discovery_ok

    @property
    def models_path(self) -> str:  # pragma: no cover - overridden
        """Path to the runtime model-listing endpoint."""
        raise NotImplementedError

    def capabilities(self) -> dict[str, bool]:
        """HTTP runtimes: discovery + health; no process control from llmctl."""
        caps = super().capabilities()
        caps["list_loaded_models"] = True
        return caps

    async def health_check(self) -> AdapterStatus:
        """Return OK when the runtime endpoint is reachable."""
        ok, _, error = await self._get_json(self.health_path)
        if ok:
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

    async def start(self, plan: LaunchPlan, on_spawn: SpawnCallback | None = None) -> Session:
        """Attach a session to the shared server endpoint.

        HTTP runtimes manage their own processes, so no child process is
        spawned and ``on_spawn`` is never invoked.
        """
        if plan.dry_run:
            return Session(
                model_id=plan.model_id,
                profile_id=plan.profile_id,
                runtime=self.runtime,
                status=SessionStatus.PLANNED,
                endpoint_url=plan.endpoint_url or self.endpoint,
                gpu_ids=plan.gpu_ids,
                launch_plan=plan,
            )
        health = await self.health_check()
        running = health.state == HealthState.OK
        return Session(
            model_id=plan.model_id,
            profile_id=plan.profile_id,
            runtime=self.runtime,
            status=SessionStatus.RUNNING if running else SessionStatus.FAILED,
            endpoint_url=plan.endpoint_url or self.endpoint,
            gpu_ids=plan.gpu_ids,
            launch_plan=plan,
            error=None if running else health.message,
            started_at=utcnow() if running else None,
        )

    async def stop(self, session: Session) -> AdapterStatus:
        """Detach from the shared server without stopping the daemon."""
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.OK,
            message=(
                f"{self.display_name} is a shared server; session detached "
                "without stopping the daemon."
            ),
        )

    async def status(self, session: Session | None = None) -> AdapterStatus:
        """Return endpoint reachability as session status."""
        return await self.health_check()

    async def delete_model(self, model: Model) -> AdapterStatus:
        """Default: deletion not supported by this HTTP runtime."""
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.UNKNOWN,
            message=f"{self.display_name} does not support remote model deletion.",
        )


class ProcessRuntimeAdapter(RuntimeAdapter):
    """Base adapter for process-launch runtimes."""

    #: Whether a non-dry-run start blocks on endpoint readiness. True for
    #: runtimes that are HTTP servers by construction (vLLM, llama.cpp);
    #: False where serving is optional (arbitrary python scripts may be
    #: batch jobs that never open the port the scheduler reserved).
    readiness_gated = True

    def __init__(
        self,
        runtime: RuntimeName,
        display_name: str,
        config: RuntimeConfig,
        supervisor: ProcessSupervisor | None = None,
        *,
        filesystem_discovery: bool = False,
    ) -> None:
        self.runtime = runtime
        self.runtime_name = runtime.value
        self.display_name = display_name
        self.config = config
        self.supervisor = supervisor or ProcessSupervisor()
        self.filesystem_discovery = filesystem_discovery

    def capabilities(self) -> dict[str, bool]:
        """Process runtimes: full local process control + supervised logs."""
        caps = super().capabilities()
        caps.update(
            {
                "discover_models": self.filesystem_discovery,
                "launch_process": True,
                "stop_process": True,
                "logs": True,
            }
        )
        return caps

    async def discover_models(self) -> list[Model]:
        """Discover on-disk models when this runtime is filesystem-backed."""
        if not self.filesystem_discovery:
            return []
        return discover_filesystem_models(self.runtime)

    async def health_check(self) -> AdapterStatus:
        """Return OK when the runtime binary is resolvable on PATH."""
        binary = self.config.binary
        if not binary:
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.OK,
                message=f"{self.display_name} requires no fixed binary.",
            )
        resolved = shutil.which(binary)
        if resolved:
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.OK,
                message=f"{self.display_name} binary found: {resolved}.",
                details={"binary": resolved},
            )
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.UNAVAILABLE,
            message=f"{self.display_name} binary '{binary}' not found on PATH.",
            details={"binary": binary},
        )

    async def start(self, plan: LaunchPlan, on_spawn: SpawnCallback | None = None) -> Session:
        """Launch the planned command as a supervised child process."""
        if plan.dry_run:
            return Session(
                model_id=plan.model_id,
                profile_id=plan.profile_id,
                runtime=self.runtime,
                status=SessionStatus.PLANNED,
                endpoint_url=plan.endpoint_url,
                gpu_ids=plan.gpu_ids,
                launch_plan=plan,
            )
        if not plan.command:
            return Session(
                model_id=plan.model_id,
                profile_id=plan.profile_id,
                runtime=self.runtime,
                status=SessionStatus.FAILED,
                gpu_ids=plan.gpu_ids,
                launch_plan=plan,
                error="Launch plan has no command to execute.",
            )
        try:
            log_name = plan.log_name or f"{self.runtime.value}-{plan.model_id or 'session'}"
            result = self.supervisor.launch(
                plan.command,
                env=plan.env,
                log_name=log_name,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            return Session(
                model_id=plan.model_id,
                profile_id=plan.profile_id,
                runtime=self.runtime,
                status=SessionStatus.FAILED,
                gpu_ids=plan.gpu_ids,
                launch_plan=plan,
                error=f"Failed to launch {self.display_name}: {exc}",
            )
        if on_spawn is not None:
            on_spawn(result.pid, result.log_path)
        session = Session(
            model_id=plan.model_id,
            profile_id=plan.profile_id,
            runtime=self.runtime,
            status=SessionStatus.RUNNING,
            pid=result.pid,
            endpoint_url=plan.endpoint_url,
            log_path=result.log_path,
            gpu_ids=plan.gpu_ids,
            launch_plan=plan,
            started_at=result.started_at,
        )
        if plan.endpoint_url and self.readiness_gated:
            session = await self._await_readiness(session, plan)
        return session

    async def _await_readiness(self, session: Session, plan: LaunchPlan) -> Session:
        """Wait for the launched endpoint to answer before declaring RUNNING.

        Success is a real readiness signal, not a spawned PID. Three outcomes:

        * endpoint answers → ``RUNNING``;
        * process dies while waiting → ``FAILED`` with a log-tail excerpt;
        * timeout with the process still alive → ``STARTING`` (large models
          legitimately load for minutes; reconcile promotes the session once
          the endpoint responds).
        """
        deadline = time.monotonic() + max(self.config.readiness_timeout_s, 0.0)
        interval = max(self.config.readiness_poll_interval_s, 0.05)
        while True:
            if session.pid and not self.supervisor.is_running(session.pid):
                session.status = SessionStatus.FAILED
                session.error = self._startup_failure_message(session)
                session.started_at = None
                session.pid = None  # dead; a stale pid would invite killing a reused one
                return session
            if await self._endpoint_alive(plan.endpoint_url):
                session.status = SessionStatus.RUNNING
                session.error = None
                return session
            if time.monotonic() >= deadline:
                session.status = SessionStatus.STARTING
                session.error = None
                return session
            await asyncio.sleep(interval)

    def _startup_failure_message(self, session: Session) -> str:
        """Describe a startup death, including a short log tail when available."""
        message = f"{self.display_name} process exited before becoming ready."
        tail = _tail_log(session.log_path)
        if tail:
            message += f" Last log lines:\n{tail}"
        return message

    async def stop(self, session: Session) -> AdapterStatus:
        """Terminate the supervised process for ``session``."""
        if not session.pid or not self.supervisor.is_running(session.pid):
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.OK,
                message=f"{self.display_name} process is not running.",
            )
        stopped = self.supervisor.terminate(session.pid)
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.OK if stopped else HealthState.DEGRADED,
            message=(
                f"{self.display_name} process {session.pid} "
                f"{'terminated' if stopped else 'did not terminate cleanly'}."
            ),
            details={"pid": session.pid, "stopped": stopped},
        )

    async def status(self, session: Session | None = None) -> AdapterStatus:
        """Return session status from HTTP health (if any) then process liveness."""
        if session is None or not session.pid:
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.UNKNOWN,
                message=f"{self.display_name} has no associated process.",
            )
        running = self.supervisor.is_running(session.pid)
        endpoint = session.endpoint_url
        http_ok = await self._endpoint_alive(endpoint) if endpoint else None
        if http_ok:
            return AdapterStatus(
                runtime=self.runtime,
                state=HealthState.OK,
                message=f"{self.display_name} is serving at {endpoint}.",
                details={"pid": session.pid, "running": running, "http": True},
            )
        if running:
            state = HealthState.DEGRADED if endpoint else HealthState.OK
            message = (
                f"{self.display_name} process {session.pid} is running"
                + (" but the HTTP endpoint is not ready yet." if endpoint else ".")
            )
            return AdapterStatus(
                runtime=self.runtime,
                state=state,
                message=message,
                details={"pid": session.pid, "running": True, "http": False},
            )
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.UNAVAILABLE,
            message=f"{self.display_name} process {session.pid} is not running.",
            details={"pid": session.pid, "running": False},
        )

    async def _endpoint_alive(self, endpoint: str | None) -> bool:
        """Return True when ``GET {endpoint}/v1/models`` responds successfully."""
        if not endpoint:
            return False
        url = f"{endpoint.rstrip('/')}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False

    async def delete_model(self, model: Model) -> AdapterStatus:
        """Filesystem/script runtimes do not delete model files from here."""
        return AdapterStatus(
            runtime=self.runtime,
            state=HealthState.UNKNOWN,
            message=(
                f"{self.display_name} does not delete model files; "
                "remove them from disk manually."
            ),
        )
