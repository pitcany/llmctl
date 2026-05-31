"""Abstract runtime adapter interface.

Adapters provide the *mechanism* for talking to a specific local LLM runtime
(discovery, health, start/stop, deletion). Orchestration and persistence live in
the service layer. Adapters never persist database state themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llmctl.schemas import AdapterStatus, LaunchPlan, Model, Session


class RuntimeAdapter(ABC):
    """Interface implemented by every local LLM runtime adapter."""

    runtime_name: str

    @abstractmethod
    async def discover_models(self) -> list[Model]:
        """Discover models known to this runtime."""

    @abstractmethod
    async def start(self, plan: LaunchPlan) -> Session:
        """Start a runtime session from a launch plan.

        When ``plan.dry_run`` is True the adapter must not launch any process and
        should return a ``PLANNED`` session. Otherwise it performs real control.
        The returned :class:`Session` is not persisted by the adapter.
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
