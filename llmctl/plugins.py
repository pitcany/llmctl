"""Plugin seam: how a host-specific package extends llmctl.

llmctl's core knows the engines (vLLM, llama.cpp, Ollama, LM Studio,
SGLang), GPUs, presets, sessions and systemd units. Anything tied to one
particular machine's stack — a Docker container that must be stopped before
vLLM claims the GPUs, an agent framework whose config is checked after a
launch, a router whose config file lists extra servers, a model lab with
its own CLI verb — lives in a *plugin* package and registers through this
module.

A plugin is a distribution with an entry point in the ``llmctl.plugins``
group whose target is a ``register(registry)`` callable::

    [project.entry-points."llmctl.plugins"]
    mystack = "llmctl_mystack:register"

``register`` receives the :class:`PluginRegistry` and appends to its lists.
Every extension point is optional and every list may stay empty: with no
plugins installed llmctl behaves as a plain control plane.

Extension points
----------------

* :class:`Companion` — a service that holds GPU memory and is stopped
  before a vLLM launch, then started again once vLLM is ready (or on
  rollback). systemd units are already covered by
  ``managed_units.fleet``; companions cover everything else (containers,
  user services, remote hooks).
* :class:`PostStartCheck` — runs once the launched model answers; returns
  a status the CLI prints. Must never mutate anything.
* :data:`LaneSource` — returns extra :class:`~llmctl.services.lanes.Lane`
  rows for ``llmctl run`` (servers with their own start script and unit).
* :data:`CommandHook` — adds Typer commands to the ``llmctl`` app.
* :data:`DoctorHook` — returns extra ``llmctl doctor`` findings
  (:class:`~llmctl.services.doctor.DoctorCheck`). Read-only, like doctor.
* :data:`StatusHook` — returns a Rich renderable (a string is read as Rich
  markup; escape untrusted text) that ``llmctl status``
  prints after the units table (not part of ``--json``).

``LLMCTL_PLUGINS=none`` disables plugin discovery (tests, debugging); a
plugin that fails to import is reported on stderr and skipped so one broken
package cannot take the CLI down.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import typer

    from llmctl.config import ManagedUnitsConfig, Settings
    from llmctl.services.doctor import DoctorCheck
    from llmctl.services.lanes import Lane

ENTRY_POINT_GROUP = "llmctl.plugins"
DISABLE_ENV = "LLMCTL_PLUGINS"

Logger = Callable[[str], None]


@runtime_checkable
class Companion(Protocol):
    """A GPU-holding service paused around a vLLM launch.

    ``stop`` returns ``True`` when it stopped the service, ``False`` when it
    tried and failed, and ``None`` when there was nothing to stop (not
    running, tooling missing). Only a ``True`` is started again later.
    """

    name: str
    #: Shell command the operator runs if llmctl could not start it again.
    start_hint: str

    def stop(self, *, logger: Logger) -> bool | None: ...

    def start(self, *, logger: Logger) -> bool: ...

    def describe(self) -> str:
        """One line for the launch confirmation dialog."""
        ...


@runtime_checkable
class PostStartCheck(Protocol):
    """A read-only check run after the launched model answers."""

    name: str

    def run(self, spec: Any, *, logger: Logger) -> object: ...

    def describe(self) -> str: ...


LaneSource = Callable[["ManagedUnitsConfig | None"], "list[Lane]"]
CommandHook = Callable[["typer.Typer"], None]
DoctorHook = Callable[["Settings"], "list[DoctorCheck]"]
StatusHook = Callable[[], Any]


@dataclass
class PluginRegistry:
    """What plugins have registered. One instance per process."""

    companions: list[Companion] = field(default_factory=list)
    post_start_checks: list[PostStartCheck] = field(default_factory=list)
    lane_sources: list[LaneSource] = field(default_factory=list)
    commands: list[CommandHook] = field(default_factory=list)
    doctor_checks: list[DoctorHook] = field(default_factory=list)
    status_sections: list[StatusHook] = field(default_factory=list)
    #: Distribution names that registered, for ``llmctl doctor``.
    loaded: list[str] = field(default_factory=list)
    #: ``(name, error)`` for plugins that failed to load.
    failed: list[tuple[str, str]] = field(default_factory=list)


_registry: PluginRegistry | None = None


def load_plugins(
    *,
    eps: Sequence[Any] | None = None,
    stderr: Logger | None = None,
) -> PluginRegistry:
    """Discover and register every plugin; cached for the process.

    ``eps`` injects entry points for tests. Call :func:`reset` to drop the
    cache.
    """
    global _registry
    if _registry is not None:
        return _registry
    registry = PluginRegistry()
    if os.environ.get(DISABLE_ENV, "").strip().lower() in {"none", "0", "off"}:
        _registry = registry
        return registry
    report = stderr or (lambda msg: print(msg, file=sys.stderr))
    if eps is None:
        eps = list(entry_points(group=ENTRY_POINT_GROUP))
    for ep in eps:
        try:
            register = ep.load()
            register(registry)
        except Exception as exc:  # noqa: BLE001 - one bad plugin must not kill the CLI
            registry.failed.append((ep.name, f"{type(exc).__name__}: {exc}"))
            report(f"llmctl: plugin {ep.name!r} failed to load and was skipped: {exc}")
            continue
        registry.loaded.append(ep.name)
    _registry = registry
    return registry


def registry() -> PluginRegistry:
    """The process-wide registry (loads plugins on first use)."""
    return load_plugins()


def reset() -> None:
    """Forget loaded plugins (tests)."""
    global _registry
    _registry = None


@dataclass
class SystemdCompanion:
    """A systemd unit paused around a launch; the built-in companion kind.

    ``user`` selects ``systemctl --user``. ``sudo`` prefixes the system-scope
    calls, matching the NOPASSWD contract llmctl's launcher units assume.
    """

    unit: str
    user: bool = False
    systemctl: Any = None

    @property
    def name(self) -> str:
        return self.unit

    @property
    def start_hint(self) -> str:
        scope = "--user " if self.user else ""
        prefix = "" if self.user else "sudo "
        return f"{prefix}systemctl {scope}start {self.unit}"

    def _runner(self) -> Any:
        if self.systemctl is None:
            from llmctl.integrations.systemctl import SystemctlRunner

            self.systemctl = SystemctlRunner()
        return self.systemctl

    def stop(self, *, logger: Logger) -> bool | None:
        runner = self._runner()
        if not runner.is_active(self.unit, user=self.user):
            return None
        result = runner.stop(self.unit, user=self.user)
        if result.ok:
            logger(f"  stopped {self.unit}")
            return True
        logger(f"  FAILED to stop {self.unit}: {result.stderr.strip() or result.stdout.strip()}")
        return False

    def start(self, *, logger: Logger) -> bool:
        ok = bool(self._runner().start(self.unit, user=self.user).ok)
        logger(f"  started {self.unit} again" if ok else f"  could not start {self.unit}")
        return ok

    def describe(self) -> str:
        return f"stops {self.unit} (to free VRAM)"
