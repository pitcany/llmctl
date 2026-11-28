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

Cutover mode (``llmctl_next.mode: cutover``, llmctl M6b): llmctl-next owns the
lifecycle. Every unit write here goes through the guard
(:func:`llmctl.integrations.llmctl_next.delegate`): an enrolled unit is sent to
the agent, any other unit is refused, and no systemctl, sudo or lane script
runs. The agent plans the stops and the rollback itself, so a swap is one
``llmctl-next run <workload> --wait``. In any other mode nothing changes.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llmctl.services.lanes import Lane, record_load, share_start_cmd
from llmctl.services.op_lock import gpu_operation_lock

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


def _remember_vram(key: str, lane: Lane) -> object:
    from llmctl.services.vram_fit import remember_lane_vram

    return remember_lane_vram(key, lane)


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
    #: Measure and remember a lane's per-card VRAM once it serves.
    remember_vram: Callable[[str, Lane], object] = _remember_vram
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


def cutover_delegation():  # noqa: ANN201 - NextDelegation, imported lazily
    """The delegation when the mode is cutover, else None."""
    from llmctl.integrations.llmctl_next import load_delegation

    delegation = load_delegation()
    return delegation if delegation.cutover else None


def unit_write(verb: str, unit: str, argv: Sequence[str], deps: LaneSwapDeps) -> bool:
    """One lifecycle write (``start``/``stop``) on ``unit``; True on success.

    Cutover mode: through the guard. An enrolled unit goes to the agent; any
    other unit is refused with the llmctl-next verb to use. ``argv`` does not
    run. Any other mode: ``argv``, exactly as before.
    """
    delegation = cutover_delegation()
    if delegation is None:
        return deps.run_quiet(argv) == 0
    from llmctl.integrations.llmctl_next import delegate

    answer = delegate(verb, unit, delegation)
    rc, err = (answer[0], answer[2]) if answer is not None else (6, f"no answer for {verb} {unit}")
    if rc != 0:
        deps.log(f"  {err.strip()}")
    return rc == 0


def stop_user_unit(unit: str, deps: LaneSwapDeps) -> bool:
    """``systemctl --user stop unit``; True on success. Cutover: see :func:`unit_write`."""

    with gpu_operation_lock(purpose=f"stop unit {unit}", interface="llmctl"):
        return unit_write("stop", unit, ["systemctl", "--user", "stop", unit], deps)


def _cutover_lane_start(target: Lane, delegation, deps: LaneSwapDeps) -> LaneSwapResult:  # noqa: ANN001
    """Cutover: start ``target`` with one agent operation. No script, no systemctl."""
    from llmctl.integrations.llmctl_next import delegate

    started = deps.clock()
    if target.unit is None:
        return LaneSwapResult(
            ok=False,
            error=(
                f"llmctl-next cutover: lane {target.name} has no unit, so it has no workload. "
                "Use llmctl-next run with its enrolled workload name."
            ),
        )
    deps.log(f"starting {target.name} through llmctl-next ...")
    answer = delegate("start", target.unit, delegation)
    rc, err = (answer[0], answer[2]) if answer is not None else (6, "no answer from the guard")
    result = LaneSwapResult(ok=rc == 0, seconds=deps.clock() - started)
    if result.ok:
        deps.record(target.name, result.seconds)
    else:
        result.error = err.strip() or f"llmctl-next exited {rc}"
    return result


def wait_ready(
    upstream: str, model: str, timeout_s: float, deps: LaneSwapDeps, *, restoring: bool = False,
) -> bool:
    """Poll ``upstream/v1/models`` until ``model`` is listed or time runs out."""
    from llmctl.services.operations import OperationPhase, mark_phase

    if not restoring:
        mark_phase(OperationPhase.CHECKING_READINESS)
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

    from llmctl.services.operations import OperationPhase, mark_phase

    mark_phase(OperationPhase.RESTORING)
    with gpu_operation_lock(purpose="rollback restart", interface="llmctl"):
        cutover = cutover_delegation() is not None
        if restart_vllm:
            argv = ["sudo", "-n", "systemctl", "start", vllm_unit]
            if unit_write("start", vllm_unit, argv, deps):
                result.restored.append(vllm_unit)
                deps.log(f"  rollback: started {vllm_unit} again (previous preset)")
            else:
                result.restore_failed.append(vllm_unit)
                deps.log(
                    f"  rollback: could not start {vllm_unit}; run: "
                    f"sudo systemctl start {vllm_unit}"
                )
        for lane in previous_lanes:
            label = lane.unit or lane.name
            if cutover:
                # No lane script in cutover mode: the agent starts the unit.
                ok = lane.unit is not None and unit_write("start", lane.unit, (), deps)
                (result.restored if ok else result.restore_failed).append(label)
                continue
            deps.log(f"  rollback: restarting {lane.name} ({' '.join(lane.start_cmd)})")
            rc = deps.run_script(lane.start_cmd, lane.health_timeout_s + _SCRIPT_GRACE_S)
            # A failed start script cannot become ready: do not sit out the
            # whole health timeout before reporting it.
            ready = rc == 0 and wait_ready(
                lane.upstream, lane.served_model, lane.health_timeout_s, deps, restoring=True,
            )
            if ready:
                result.restored.append(label)
            else:
                result.restore_failed.append(label)
                deps.log(
                    f"  rollback: {lane.name} did not come back; "
                    f"run: {' '.join(lane.start_cmd)}"
                )


def swap_to_lane(
    target: Lane,
    *,
    other_active: Sequence[Lane],
    vllm_active: bool,
    vllm_unit: str,
    deps: LaneSwapDeps | None = None,
    share_gpus: bool = False,
) -> LaneSwapResult:
    """Start ``target`` and roll back on failure. See the module docstring.

    ``other_active`` are the lanes whose units are active now (stopped first,
    restarted on failure). ``vllm_active`` says whether vllm-tp was running,
    which the start script stops and a failure restarts.

    ``share_gpus``: the target fits beside what runs (the caller checked), so
    its script runs with ``--share-gpus`` and stops nothing; vllm-tp is left
    alone and a failure only stops the target.
    """

    with gpu_operation_lock(purpose=f"swap to lane {target.name}", interface="llmctl"):
        deps = deps or LaneSwapDeps()
        delegation = cutover_delegation()
        if delegation is not None:
            return _cutover_lane_start(target, delegation, deps)
        result = LaneSwapResult(ok=False)
        started = deps.clock()
        start_cmd = share_start_cmd(target.start_cmd) if share_gpus else target.start_cmd
        if share_gpus:
            vllm_active = False

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
            deps.log(f"starting {target.name}: {' '.join(start_cmd)}")
            rc = deps.run_script(start_cmd, target.health_timeout_s + _SCRIPT_GRACE_S)
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
            deps.remember_vram(target.name, target)
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
    """Like :func:`_run_streaming` but swallows output (for the TUI).

    No stdin either: a prompt whose output is swallowed would wait unseen.
    """
    try:
        return subprocess.run(  # noqa: S603 - argv from router.yaml
            list(argv),
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
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

    with gpu_operation_lock(purpose="swap to vllm", interface="llmctl"):
        deps = deps or LaneSwapDeps()
        lanes = LaneSwapResult(ok=True)
        if cutover_delegation() is not None:
            # The agent stops what is in the way and rolls back (as `llmctl run` does).
            return start(), lanes
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
