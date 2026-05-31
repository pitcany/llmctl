"""Fleet-aware preflight: stop competing units before starting a target.

Multiple GPU-claiming services can't coexist on the same hardware. The
TP-fleet unit (``vllm-tp``) and the per-GPU slot units
(``vllm-coder``, ``vllm-reasoner``) both want the same GPUs, so
starting one requires stopping the others first. The fleet target
(``agents.target``) gates slot startup as a unit; stopping it tears
down both slots in one call.

This module formalises the preflight rules so the adapter and the CLI
both see the same orchestration logic.

Rules (matching gpu-models behaviour):

* Starting **TP** stops: agents.target, coder, reasoner, ollama, then
  TP itself (idempotent restart). Optionally also stops
  ``harbor.ollama`` via the Harbor integration.
* Starting **coder** or **reasoner** stops: TP, ollama. The sibling
  slot keeps running (slots are designed to coexist).
* The Harbor integration runs as a separate hook (see
  :mod:`llmctl.integrations.harbor`); this module covers systemd units
  only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from llmctl.config import FleetUnitsConfig
from llmctl.integrations.systemctl import SystemctlRunner


class FleetRole(StrEnum):
    """The unit you're about to start. Determines the stop list."""

    TP = "tp"
    CODER = "coder"
    REASONER = "reasoner"


@dataclass(frozen=True)
class StopReport:
    """Result of one preflight invocation."""

    stopped: list[str]
    """Units that were active and got stopped."""

    skipped: list[str]
    """Units that were already inactive (no stop attempted)."""

    failed: list[str]
    """Units we tried to stop but got a non-zero exit from."""

    @property
    def all_clean(self) -> bool:
        """``True`` when every targeted unit is now inactive."""
        return not self.failed


def units_to_stop(target: FleetRole, fleet: FleetUnitsConfig) -> list[str]:
    """Return the ordered stop list for ``target``.

    Order matters: the fleet target is stopped before the slot
    services it gates so systemd doesn't immediately restart them
    via the target's ``Wants=`` chain. TP is stopped last when
    starting TP (the idempotent restart) so the caller can read its
    new env file after preflight completes.

    Args:
        target: The unit you're about to start.
        fleet: Configured unit names — defaults match the NOPASSWD
            sudoers scope on yannik-desktop.
    """
    if target is FleetRole.TP:
        return [fleet.fleet_target, fleet.coder, fleet.reasoner, fleet.ollama, fleet.tp]
    if target is FleetRole.CODER:
        return [fleet.tp, fleet.ollama]
    if target is FleetRole.REASONER:
        return [fleet.tp, fleet.ollama]
    raise ValueError(f"unknown target {target!r}")


def preflight_stop(
    target: FleetRole,
    fleet: FleetUnitsConfig,
    systemctl: SystemctlRunner,
    *,
    logger: Callable[[str], None] = print,
) -> StopReport:
    """Issue ``systemctl stop`` for every unit that competes with ``target``.

    Uses :meth:`SystemctlRunner.try_stop` so already-inactive units are
    skipped (avoids spurious sudo prompts for nothing).
    Returns a typed :class:`StopReport` the caller can use to decide
    whether to abort the start (``not report.all_clean``).
    """
    stopped: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for unit in units_to_stop(target, fleet):
        if not systemctl.is_active(unit):
            skipped.append(unit)
            continue
        result = systemctl.stop(unit)
        if result.ok:
            stopped.append(unit)
            logger(f"  fleet: stopped {unit}")
        else:
            failed.append(unit)
            logger(
                f"  fleet: FAILED to stop {unit}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

    return StopReport(stopped=stopped, skipped=skipped, failed=failed)
