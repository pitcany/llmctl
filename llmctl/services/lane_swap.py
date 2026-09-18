"""Swap onto a lane (FreeToken, llama.cpp, ...) with readiness wait and rollback.

The lane's start script does the GPU hand-over; this module only sequences it:

1. Stop other active lane units (user scope). systemd ``Conflicts=`` would do
   most of this too, but not for every pair, and the preview promised it.
2. Run the start script (its output streams to the terminal).
3. Wait until the upstream lists the served model.
4. On failure, stop the half-started unit and bring back what was serving:
   ``vllm-tp`` via ``sudo -n systemctl start`` (its env file is untouched, so it
   comes back on the previous preset) and lanes via their own start scripts.

Every side effect goes through :class:`LaneSwapDeps` so tests run without
systemd, sudo or a GPU.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llmctl.services.lanes import Lane, record_load

#: Extra seconds the start script may run past the lane's health budget; the
#: scripts wait for readiness themselves, then print a status block.
_SCRIPT_GRACE_S = 120.0


def _run_streaming(argv: Sequence[str], timeout: float) -> int:
    """Run ``argv`` with inherited stdout/stderr; return its exit code."""
    try:
        return subprocess.run(list(argv), check=False, timeout=timeout).returncode  # noqa: S603
    except subprocess.TimeoutExpired:
        return 124
    except OSError:
        return 127


def _run_quiet(argv: Sequence[str]) -> int:
    try:
        return subprocess.run(  # noqa: S603 - fixed argv
            list(argv), check=False, capture_output=True, timeout=120.0
        ).returncode
    except (subprocess.TimeoutExpired, OSError):
        return 1


def _default_probe(upstream: str) -> list[str] | None:
    from llmctl.services.backends import probe_openai_v1_models

    return probe_openai_v1_models(upstream, timeout=3.0)


@dataclass
class LaneSwapDeps:
    """Side-effect seams. Defaults are the real machine."""

    run_script: Callable[[Sequence[str], float], int] = _run_streaming
    run_quiet: Callable[[Sequence[str]], int] = _run_quiet
    probe: Callable[[str], list[str] | None] = _default_probe
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = print
    record: Callable[[str, float], None] = record_load
    poll_interval_s: float = 3.0


@dataclass
class LaneSwapResult:
    """What happened. ``ok`` means the target answers with its model."""

    ok: bool
    seconds: float = 0.0
    error: str | None = None
    stopped_units: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    restore_failed: list[str] = field(default_factory=list)


def stop_user_unit(unit: str, deps: LaneSwapDeps) -> bool:
    """``systemctl --user stop unit``; True on success."""
    return deps.run_quiet(["systemctl", "--user", "stop", unit]) == 0


def wait_ready(upstream: str, model: str, timeout_s: float, deps: LaneSwapDeps) -> bool:
    """Poll ``upstream/v1/models`` until ``model`` is listed or time runs out."""
    deadline = deps.clock() + timeout_s
    last_note = deps.clock()
    while True:
        ids = deps.probe(upstream)
        if ids and model in ids:
            return True
        now = deps.clock()
        if now >= deadline:
            return False
        if now - last_note >= 30:
            deps.log(f"  still waiting for {model} ({int(timeout_s - (deadline - now))}s)...")
            last_note = now
        deps.sleep(deps.poll_interval_s)


def restore_previous(
    previous_lanes: Sequence[Lane],
    *,
    restart_vllm: bool,
    vllm_unit: str,
    deps: LaneSwapDeps,
    result: LaneSwapResult,
) -> None:
    """Bring back what was serving before a failed swap."""
    if restart_vllm:
        argv = ["sudo", "-n", "systemctl", "start", vllm_unit]
        if deps.run_quiet(argv) == 0:
            result.restored.append(vllm_unit)
            deps.log(f"  rollback: started {vllm_unit} again (previous preset)")
        else:
            result.restore_failed.append(vllm_unit)
            deps.log(
                f"  rollback: could not start {vllm_unit}; run: sudo systemctl start {vllm_unit}"
            )
    for lane in previous_lanes:
        deps.log(f"  rollback: restarting {lane.name} ({' '.join(lane.start_cmd)})")
        rc = deps.run_script(lane.start_cmd, lane.health_timeout_s + _SCRIPT_GRACE_S)
        label = lane.unit or lane.name
        if rc == 0 and wait_ready(lane.upstream, lane.served_model, lane.health_timeout_s, deps):
            result.restored.append(label)
        else:
            result.restore_failed.append(label)
            deps.log(f"  rollback: {lane.name} did not come back; run: {' '.join(lane.start_cmd)}")


def swap_to_lane(
    target: Lane,
    *,
    other_active: Sequence[Lane],
    vllm_active: bool,
    vllm_unit: str,
    deps: LaneSwapDeps | None = None,
) -> LaneSwapResult:
    """Start ``target`` and roll back on failure. See the module docstring.

    ``other_active`` are the lanes whose units are active now (stopped first,
    restarted on failure). ``vllm_active`` says whether vllm-tp was running,
    which the start script stops and a failure restarts.
    """
    deps = deps or LaneSwapDeps()
    result = LaneSwapResult(ok=False)
    started = deps.clock()

    for lane in other_active:
        if lane.unit is None:
            continue
        deps.log(f"stopping {lane.unit} ...")
        if stop_user_unit(lane.unit, deps):
            result.stopped_units.append(lane.unit)
        else:
            result.error = f"could not stop {lane.unit}"
            break

    if result.error is None:
        deps.log(f"starting {target.name}: {' '.join(target.start_cmd)}")
        rc = deps.run_script(target.start_cmd, target.health_timeout_s + _SCRIPT_GRACE_S)
        if rc != 0:
            result.error = f"start script exited {rc}"
        else:
            deps.log(f"waiting for {target.served_model} on {target.upstream} ...")
            if wait_ready(target.upstream, target.served_model, target.health_timeout_s, deps):
                result.ok = True
            else:
                result.error = (
                    f"{target.served_model} not listed on {target.upstream} "
                    f"after {int(target.health_timeout_s)}s"
                )

    result.seconds = deps.clock() - started
    if result.ok:
        deps.record(target.name, result.seconds)
        return result

    deps.log(f"swap failed: {result.error}; restoring the previous state")
    if target.unit is not None:
        stop_user_unit(target.unit, deps)
        deps.log(f"  rollback: stopped {target.unit}")
    stopped = [lane for lane in other_active if lane.unit in result.stopped_units]
    restore_previous(
        stopped,
        restart_vllm=vllm_active,
        vllm_unit=vllm_unit,
        deps=deps,
        result=result,
    )
    return result


def _run_captured(argv: Sequence[str], timeout: float) -> int:
    """Like :func:`_run_streaming` but swallows output (for the TUI)."""
    try:
        return subprocess.run(  # noqa: S603 - argv from router.yaml
            list(argv), check=False, capture_output=True, timeout=timeout
        ).returncode
    except subprocess.TimeoutExpired:
        return 124
    except OSError:
        return 127


def quiet_deps(log: Callable[[str], None] | None = None) -> LaneSwapDeps:
    """Deps that never write to the terminal — a TUI owns the screen."""
    return LaneSwapDeps(run_script=_run_captured, log=log or (lambda _msg: None))


def swap_to_vllm(
    active_lanes: Sequence[Lane],
    start: Callable[[], object],
    *,
    deps: LaneSwapDeps | None = None,
) -> tuple[object | None, LaneSwapResult]:
    """Stop ``active_lanes``, call ``start`` (the vLLM launch), undo on failure.

    ``start`` returns an object with ``.ok`` (an ``OrchestratorResult``). If a
    lane refuses to stop, vLLM is not started (it would OOM next to it) and
    the lanes already stopped are restarted. Returns ``(start_result, lanes)``;
    ``start_result`` is None when vLLM was never started.
    """
    deps = deps or LaneSwapDeps()
    lanes = LaneSwapResult(ok=True)
    stopped: list[Lane] = []
    for lane in active_lanes:
        if lane.unit is None:
            continue
        if stop_user_unit(lane.unit, deps):
            stopped.append(lane)
            lanes.stopped_units.append(lane.unit)
        else:
            lanes.ok = False
            lanes.error = f"could not stop {lane.unit}"
            restore_previous(stopped, restart_vllm=False, vllm_unit="", deps=deps, result=lanes)
            return None, lanes
    result = start()
    if not getattr(result, "ok", False) and stopped:
        lanes.ok = False
        restore_previous(stopped, restart_vllm=False, vllm_unit="", deps=deps, result=lanes)
    return result, lanes
