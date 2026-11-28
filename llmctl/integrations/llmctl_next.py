"""Delegate lifecycle writes to llmctl-next (Rust M5b and M6b).

Decision 2 of the M5 plan: during the pilot the legacy writers are not
removed — they delegate. For a workload llmctl-next owns, a
start/stop/restart goes to the llmctl-next CLI instead of systemd; when
the agent is down or refuses, that is an honest refusal with a clear
message, never a silent fallback to the old write path. Units llmctl-next
does not own keep the old path unchanged.

Cutover mode refuses lifecycle writes to units outside the map and refuses
``LLMCTL_NEXT_DISABLED=1``. Remove ``mode: cutover`` to restore delegate mode.
In delegate mode, set ``enabled: false`` or use ``LLMCTL_NEXT_DISABLED=1``
to turn delegation off.

settings.yaml section (all optional; absent = disabled)::

    llmctl_next:
      enabled: true
      mode: cutover  # optional; takes precedence over enabled
      cmd: [/path/to/llmctl-next, --data-dir, /path/to/data]
      units:
        vllm-engine.service: vllm-engine
        gpt-oss-120b.service: gpt-oss-120b

``cmd`` is the agent CLI prefix (argv, not a shell line). ``LLMCTL_NEXT_CMD``
overrides it for tests (shlex-split, so ``"/tmp/standin --data-dir /tmp/x"``
works); the bare binary name ``llmctl-next`` is used when ``cmd`` is unset.

Exit codes (the same map llm-router uses): 3 agent unreachable,
4 refused at admission, 5 workload not enrolled, 6 bad request/protocol.
Those mean nothing changed; any other non-zero code is an operation that
was admitted and did not succeed. Both surface as a failed
:class:`~llmctl.integrations.systemctl.SystemctlResult` with the refusal
in ``stderr`` — callers already report that honestly.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from llmctl.config import _read_yaml, settings_file_path  # noqa: SLF001 - same package
from llmctl.services.op_lock import handed_off

#: Exit codes that mean "refused before anything changed" (see module doc).
REFUSED_BEFORE_CHANGE = frozenset({3, 4, 5, 6})

_SECTION = "llmctl_next"
# Two 1800 s readiness windows (start and rollback), plus stop time.
_TIMEOUT_S = 4200.0

#: systemctl verb -> llmctl-next verb.
_VERBS = {"start": "run", "stop": "stop", "restart": "restart"}


@dataclass(frozen=True)
class NextDelegation:
    """What the guard delegates, from settings.yaml + environment."""

    enabled: bool = False
    cmd: tuple[str, ...] = ("llmctl-next",)
    units: dict[str, str] = field(default_factory=dict)
    cutover: bool = False

    def workload_for(self, unit: str) -> str | None:
        """The llmctl-next workload behind ``unit``, if this host delegates it.

        Unit names arrive in both forms (`vllm-engine` from
        ``vllm-on-gpus`` and router lanes, `ollama-gpu1.service` from
        elsewhere); the map may be keyed either way. Both directions are
        normalized so a near-miss key can never silently defeat the gate
        and fall through to systemctl (security review 2026-11-18).
        """
        if not (self.enabled or self.cutover):
            return None
        bare = unit.removesuffix(".service")
        candidates = (unit, bare, f"{bare}.service") if bare != unit else (unit, f"{unit}.service")
        for key in candidates:
            if key in self.units:
                return self.units[key]
        return None

    def write_refusal(self, verb: str, unit: str) -> tuple[int, str] | None:
        """Return a refusal code and message for a write blocked by cutover."""
        if not self.cutover:
            return None
        if os.environ.get("LLMCTL_NEXT_DISABLED") == "1":
            return 4, "llmctl-next cutover refuses LLMCTL_NEXT_DISABLED=1. Unset it and retry."
        next_verb = _VERBS.get(verb)
        if next_verb is None:
            return 6, f"llmctl-next cutover refuses the systemctl verb {verb!r} on {unit}."
        if self.workload_for(unit) is None:
            name = unit.removesuffix(".service")
            return (
                5,
                f"llmctl-next cutover: {unit} has no workload in llmctl_next.units. "
                f"Use llmctl-next {next_verb} {name} --wait with its enrolled workload name.",
            )
        return None


def load_delegation(settings_path: Path | None = None) -> NextDelegation:
    """Read the ``llmctl_next:`` section (absent/disabled = no delegation)."""
    section = _read_yaml(settings_file_path(settings_path)).get(_SECTION)
    if not isinstance(section, dict):
        return NextDelegation()
    cutover = section.get("mode") == "cutover"
    if not cutover and os.environ.get("LLMCTL_NEXT_DISABLED") == "1":
        return NextDelegation()
    enabled = cutover or bool(section.get("enabled", False))
    units_raw = section.get("units") or {}
    units = {str(k): str(v) for k, v in units_raw.items()} if isinstance(units_raw, dict) else {}
    cmd_raw = section.get("cmd") or ["llmctl-next"]
    cmd = tuple(str(x) for x in cmd_raw) if isinstance(cmd_raw, (list, tuple)) else (str(cmd_raw),)
    env = os.environ.get("LLMCTL_NEXT_CMD")
    if env:
        cmd = tuple(shlex.split(env))
    return NextDelegation(enabled=enabled, cmd=cmd, units=units, cutover=cutover)


def refusal_message(
    verb: str, unit: str, workload: str, rc: int, detail: str, *, cutover: bool = False
) -> str:
    """The one-line honest refusal the guard reports through stderr."""
    kind = "refused before anything changed" if rc in REFUSED_BEFORE_CHANGE else "did not succeed"
    reversal = (
        "remove llmctl_next.mode: cutover in settings.yaml to restore delegate mode."
        if cutover
        else "set llmctl_next.enabled: false in settings.yaml to reverse the handover."
    )
    return (
        f"llmctl-next {verb} {workload} (unit {unit}, rc={rc}) {kind}: {detail}. "
        "Not falling back to systemctl: the workload is owned by llmctl-next. "
        f"Fix the agent (systemctl --user status llmctl-next) or {reversal}"
    )


def delegate(
    verb: str, unit: str, delegation: NextDelegation | None = None
) -> tuple[int, str, str] | None:
    """Run one lifecycle verb through llmctl-next.

    ``verb`` is the systemctl word (``start``/``stop``/``restart``). Returns
    ``(rc, stdout, stderr)`` with the refusal message folded into stderr on
    failure. In delegate mode, ``None`` leaves the caller on its own path.
    Lock reacquisition failures propagate.
    """
    delegation = delegation or load_delegation()
    refusal = delegation.write_refusal(verb, unit)
    if refusal is not None:
        rc, message = refusal
        return rc, "", message
    workload = delegation.workload_for(unit)
    if workload is None:
        return None
    next_verb = _VERBS.get(verb)
    if next_verb is None:
        return None
    argv = [*delegation.cmd, next_verb, workload, "--wait"]
    rc, out, err = _invoke(argv)
    if rc != 0:
        detail = (err.strip() or out.strip()).splitlines()[-1:] or [""]
        return rc, out, refusal_message(
            verb, unit, workload, rc, detail[0], cutover=delegation.cutover
        )
    return rc, out, err


def run_preset(
    preset: str,
    workload: str,
    delegation: NextDelegation,
    runner=None,
) -> tuple[int, str, str]:
    """``llmctl-next run <preset> --preset --workload <w> --wait`` (M5b).

    The agent switches the engine to the preset's profile env, starts it,
    verifies the served model and rolls back on failure; llmctl writes no
    env of its own (the agent would refuse that file as drift). Returns
    ``(rc, stdout, stderr)``; on failure stderr carries the honest refusal.
    ``runner`` (tests) takes the argv and returns that triple.
    """
    argv = [*delegation.cmd, "run", preset, "--preset", "--workload", workload, "--wait"]
    rc, out, err = _invoke(argv, runner)
    if rc != 0:
        detail = (err.strip() or out.strip()).splitlines()[-1:] or [""]
        verb = f"run (preset {preset}) via"
        return rc, out, refusal_message(
            verb, workload, workload, rc, detail[0], cutover=delegation.cutover
        )
    return rc, out, err


def _invoke(argv: list[str], runner=None) -> tuple[int, str, str]:
    """Delegate under the agent's lock, then restore the caller's ownership.

    Lock reacquisition failures propagate: callers must abort remaining
    mutations when they no longer hold the host lock.
    """
    with handed_off(timeout=_TIMEOUT_S):
        try:
            if runner is not None:
                rc, out, err = runner(argv)
            else:
                proc = subprocess.run(  # noqa: S603 - argv from settings, no shell
                    argv, capture_output=True, text=True, check=False, timeout=_TIMEOUT_S
                )
                rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
        except subprocess.TimeoutExpired as exc:
            rc, out, err = 124, "", str(exc) or f"timed out after {_TIMEOUT_S:.0f}s"
        except OSError as exc:
            rc, out, err = 3, "", f"could not run {argv[0]}: {exc}"
        return rc, out, err
