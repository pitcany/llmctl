"""Delegate lifecycle writes to llmctl-next for units it owns (Rust M5b).

Decision 2 of the M5 plan: during the pilot the legacy writers are not
removed — they delegate. For a workload llmctl-next owns, a
start/stop/restart goes to the llmctl-next CLI instead of systemd; when
the agent is down or refuses, that is an honest refusal with a clear
message, never a silent fallback to the old write path. Units llmctl-next
does not own keep the old path unchanged.

Reversal is one setting: ``llmctl_next.enabled: false`` in settings.yaml
(or deleting the section, or ``LLMCTL_NEXT_DISABLED=1``).

settings.yaml section (all optional; absent = disabled)::

    llmctl_next:
      enabled: true
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

#: Exit codes that mean "refused before anything changed" (see module doc).
REFUSED_BEFORE_CHANGE = frozenset({3, 4, 5, 6})

_SECTION = "llmctl_next"
_TIMEOUT_S = 1200.0  # a vLLM cold load is minutes; the CLI waits under --wait

#: systemctl verb -> llmctl-next verb.
_VERBS = {"start": "run", "stop": "stop", "restart": "restart"}


@dataclass(frozen=True)
class NextDelegation:
    """What the guard delegates, from settings.yaml + environment."""

    enabled: bool = False
    cmd: tuple[str, ...] = ("llmctl-next",)
    units: dict[str, str] = field(default_factory=dict)

    def workload_for(self, unit: str) -> str | None:
        """The llmctl-next workload behind ``unit``, if this host delegates it.

        Unit names arrive in both forms (`vllm-engine` from
        ``vllm-on-gpus`` and router lanes, `ollama-gpu1.service` from
        elsewhere); the map may be keyed either way. Both directions are
        normalized so a near-miss key can never silently defeat the gate
        and fall through to systemctl (security review 2026-11-18).
        """
        if not self.enabled:
            return None
        bare = unit.removesuffix(".service")
        candidates = (unit, bare, f"{bare}.service") if bare != unit else (unit, f"{unit}.service")
        for key in candidates:
            if key in self.units:
                return self.units[key]
        return None


def load_delegation(settings_path: Path | None = None) -> NextDelegation:
    """Read the ``llmctl_next:`` section (absent/disabled = no delegation)."""
    if os.environ.get("LLMCTL_NEXT_DISABLED") == "1":
        return NextDelegation(enabled=False)
    section = _read_yaml(settings_file_path(settings_path)).get(_SECTION)
    if not isinstance(section, dict):
        return NextDelegation(enabled=False)
    enabled = bool(section.get("enabled", False))
    units_raw = section.get("units") or {}
    units = {str(k): str(v) for k, v in units_raw.items()} if isinstance(units_raw, dict) else {}
    cmd_raw = section.get("cmd") or ["llmctl-next"]
    cmd = tuple(str(x) for x in cmd_raw) if isinstance(cmd_raw, (list, tuple)) else (str(cmd_raw),)
    env = os.environ.get("LLMCTL_NEXT_CMD")
    if env:
        cmd = tuple(shlex.split(env))
    return NextDelegation(enabled=enabled, cmd=cmd, units=units)


def refusal_message(verb: str, unit: str, workload: str, rc: int, detail: str) -> str:
    """The one-line honest refusal the guard reports through stderr."""
    kind = "refused before anything changed" if rc in REFUSED_BEFORE_CHANGE else "did not succeed"
    return (
        f"llmctl-next {verb} {workload} (unit {unit}, rc={rc}) {kind}: {detail}. "
        "Not falling back to systemctl: the workload is owned by llmctl-next. "
        "Fix the agent (systemctl --user status llmctl-next) or set "
        "llmctl_next.enabled: false in settings.yaml to reverse the handover."
    )


def delegate(
    verb: str, unit: str, delegation: NextDelegation | None = None
) -> tuple[int, str, str] | None:
    """Run one lifecycle verb through llmctl-next.

    ``verb`` is the systemctl word (``start``/``stop``/``restart``). Returns
    ``(rc, stdout, stderr)`` with the refusal message folded into stderr on
    failure, or ``None`` when this unit is not delegated — the caller then
    keeps its own path. Never raises.
    """
    delegation = delegation or load_delegation()
    workload = delegation.workload_for(unit)
    if workload is None:
        return None
    next_verb = _VERBS.get(verb)
    if next_verb is None:
        return None
    argv = [*delegation.cmd, next_verb, workload, "--wait"]
    try:
        proc = subprocess.run(  # noqa: S603 - argv from settings, no shell
            argv, capture_output=True, text=True, check=False, timeout=_TIMEOUT_S
        )
        rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        detail = str(exc) or f"timed out after {_TIMEOUT_S:.0f}s"
        return 124, "", refusal_message(verb, unit, workload, 124, detail)
    except OSError as exc:
        # Binary gone ≈ agent down: the same honest refusal, rc 3.
        return 3, "", refusal_message(verb, unit, workload, 3, f"could not run {argv[0]}: {exc}")
    if rc != 0:
        detail = (err.strip() or out.strip()).splitlines()[-1:] or [""]
        return rc, out, refusal_message(verb, unit, workload, rc, detail[0])
    return rc, out, err
