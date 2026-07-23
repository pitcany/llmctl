"""Abstract runtime adapter interface.

Adapters provide the *mechanism* for talking to a specific local LLM runtime
(discovery, health, start/stop, deletion). Orchestration and persistence live in
the service layer. Adapters never persist database state themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from llmctl.schemas import AdapterStatus, LaunchPlan, Model, Session

#: Callback invoked the moment a child process exists: ``(pid, log_path)``.
#: Lets the service layer persist the pid *before* any readiness wait, so an
#: interrupted start never orphans a process the control plane can't find.
SpawnCallback = Callable[[int, str | None], None]

#: Stable capability keys every adapter reports. Values are booleans; a key
#: that is False means the operation is *unsupported by the runtime*, not
#: merely unavailable right now.
CAPABILITY_KEYS = (
    "discover_models",
    "list_loaded_models",
    "launch_process",
    "stop_process",
    "delete_model",
    "version",
    "logs",
    "health_check",
)


class RuntimeAdapter(ABC):
    """Interface implemented by every local LLM runtime adapter."""

    runtime_name: str

    def capabilities(self) -> dict[str, bool]:
        """Report which operations this adapter genuinely supports.

        Keys are :data:`CAPABILITY_KEYS`. Callers use this for honest UI
        (e.g. not offering "delete" for a runtime whose ``delete_model`` is a
        polite no-op) instead of pretending all runtimes behave identically.
        """
        return {key: key in {"discover_models", "health_check"} for key in CAPABILITY_KEYS}

    async def version(self) -> str | None:
        """Return the runtime's version string, or ``None`` when unknowable."""
        return None

    async def list_loaded_models(self) -> list[Model] | None:
        """Return models currently loaded/served, or ``None`` when unsupported.

        Distinct from :meth:`discover_models`, which lists what is *installed*
        (on disk or in the runtime's library). ``None`` means the runtime has
        no way to answer; ``[]`` means it answered and nothing is loaded.
        """
        return None

    @abstractmethod
    async def discover_models(self) -> list[Model]:
        """Discover models known to this runtime."""

    @abstractmethod
    async def start(self, plan: LaunchPlan, on_spawn: SpawnCallback | None = None) -> Session:
        """Start a runtime session from a launch plan.

        When ``plan.dry_run`` is True the adapter must not launch any process and
        should return a ``PLANNED`` session. Otherwise it performs real control.
        Adapters that spawn a child process MUST call ``on_spawn(pid, log_path)``
        as soon as the process exists, before any readiness wait. The returned
        :class:`Session` is not persisted by the adapter.
        """

    @abstractmethod
    async def stop(self, session: Session) -> AdapterStatus:
        """Stop a running session (terminating processes when applicable)."""

    @abstractmethod
    async def status(self, session: Session | None = None) -> AdapterStatus:
        """Return runtime or session status."""

    @abstractmethod
    async def health_check(self) -> AdapterStatus:
        """Return runtime health (binary/endpoint availability)."""

    @abstractmethod
    async def delete_model(self, model: Model) -> AdapterStatus:
        """Delete or unregister a model from this runtime."""
