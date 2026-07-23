"""Thin wrapper around ``systemctl`` for the managed-unit adapter.

The vLLM adapter does not launch processes directly. Instead it writes
``services/vllm-tp.env`` and asks systemd to (re)start the pre-installed
``vllm-tp.service`` unit. This module is the single chokepoint for
``systemctl`` calls so the rest of the codebase has one place to mock
during tests and one place to audit for privilege escalation.

NOPASSWD contract
-----------------
The host workstation has ``NOPASSWD`` configured for ``systemctl
start|stop|restart|status`` on a specific set of unit names. We honor
that contract by only passing bare unit names (no ``.service`` suffix
when the sudoers entry doesn't include it) and one unit per call.
The unit names we use are the same names gpu-models uses today,
verbatim, so the sudoers entries keep matching.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum


class SystemctlVerb(StrEnum):
    """Subset of ``systemctl`` verbs we issue."""

    START = "start"
    STOP = "stop"
    RESTART = "restart"
    STATUS = "status"
    CAT = "cat"
    IS_ACTIVE = "is-active"


@dataclass(frozen=True)
class SystemctlResult:
    """Result of a single ``systemctl`` invocation."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """``True`` when the command exited with status 0."""
        return self.returncode == 0


class SystemctlRunner:
    """Run ``systemctl`` commands, optionally via ``sudo``.

    Read-only verbs (``status``, ``cat``, ``is-active``) skip ``sudo``.
    Write verbs (``start``, ``stop``, ``restart``) prepend ``sudo`` so
    the NOPASSWD entries on the workstation take effect.

    Inject ``runner`` for tests — any callable taking ``list[str]`` and
    returning a :class:`subprocess.CompletedProcess` works.
    """

    _READ_ONLY = {SystemctlVerb.STATUS, SystemctlVerb.CAT, SystemctlVerb.IS_ACTIVE}

    def __init__(
        self,
        runner: object | None = None,
        *,
        systemctl_bin: str = "systemctl",
        sudo_bin: str = "sudo",
    ) -> None:
        self._runner = runner
        self.systemctl_bin = systemctl_bin
        self.sudo_bin = sudo_bin

    def available(self) -> bool:
        """``True`` when ``systemctl`` is on PATH (false in most containers)."""
        return shutil.which(self.systemctl_bin) is not None

    def run(self, verb: SystemctlVerb, unit: str, *extra: str) -> SystemctlResult:
        """Invoke ``systemctl <verb> <unit> [extra...]``.

        Adds ``sudo`` for write verbs. Always captures stdout/stderr as
        text. Never raises on non-zero exit — caller inspects ``.ok``.
        """
        argv: list[str] = []
        if verb not in self._READ_ONLY:
            argv.append(self.sudo_bin)
        argv.extend([self.systemctl_bin, verb.value, unit, *extra])
        if self._runner is not None:
            completed = self._runner(argv)  # type: ignore[misc]
        else:
            try:
                completed = subprocess.run(  # noqa: S603 - argv from constants + unit name
                    argv,
                    capture_output=True,
                    text=True,
                    check=False,
                    # Bounded so a wedged systemd/D-Bus can't hang callers
                    # (doctor, validate, the API) forever. Generous enough for
                    # slow unit stops (TimeoutStopSec defaults to 90s).
                    timeout=120.0,
                )
            except subprocess.TimeoutExpired:
                return SystemctlResult(
                    returncode=124,
                    stdout="",
                    stderr=f"systemctl {verb.value} {unit} timed out after 120s",
                )
        return SystemctlResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )

    def start(self, unit: str) -> SystemctlResult:
        """``sudo systemctl start <unit>``."""
        return self.run(SystemctlVerb.START, unit)

    def stop(self, unit: str) -> SystemctlResult:
        """``sudo systemctl stop <unit>``."""
        return self.run(SystemctlVerb.STOP, unit)

    def restart(self, unit: str) -> SystemctlResult:
        """``sudo systemctl restart <unit>``."""
        return self.run(SystemctlVerb.RESTART, unit)

    def is_active(self, unit: str) -> bool:
        """``True`` when ``systemctl is-active <unit>`` reports active."""
        return self.run(SystemctlVerb.IS_ACTIVE, unit).stdout.strip() == "active"

    def cat(self, unit: str) -> str:
        """Return the resolved unit file body, or empty string on error."""
        result = self.run(SystemctlVerb.CAT, unit)
        return result.stdout if result.ok else ""

    def try_stop(self, unit: str) -> bool:
        """Stop ``unit`` if it's active. Return ``True`` when a stop was issued.

        Mirrors gpu-models's ``ProcessManager.try_stop`` so the preflight
        ("stop competing services first") reads the same in both packages.
        """
        if not self.is_active(unit):
            return False
        return self.stop(unit).ok
