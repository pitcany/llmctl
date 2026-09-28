"""High-level vLLM start orchestration.

Glues the pieces from Phases 1–4 into the operation the daily
driver actually wants:

* :func:`start_vllm_tp` — start the TP-fleet unit on a preset.

It composes: preset loading -> TQ override -> spec build ->
fleet preflight -> Harbor preflight -> adapter restart -> readiness
poll -> Hermes verify. Everything is injectable so the CLI tests can
substitute fakes without touching real systemd / docker / hermes.

The CLI commands in ``llmctl/cli.py`` are thin wrappers around this
function plus :func:`preset_choices` for tab-completion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmctl._fsutil import atomic_write_text
from llmctl.adapters.vllm_systemd import ManagedRestartResult, VLLMSystemdAdapter
from llmctl.config import (
    ManagedUnitConfig,
    VLLMDefaultsConfig,
)
from llmctl.integrations.fleet import FleetRole, FleetUnitsConfig, preflight_stop
from llmctl.integrations.harbor import (
    DEFAULT_OLLAMA_CONTAINER,
    StopOutcome,
    start_ollama_container,
    stop_ollama_container,
)
from llmctl.integrations.hermes import (
    DEFAULT_CONFIG_PATH as HERMES_DEFAULT_CONFIG_PATH,
)
from llmctl.integrations.hermes import HermesStatus, verify_provider
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.integrations.turboquant import apply_to_spec_dict
from llmctl.integrations.vllm_env import (
    VLLMLaunchSpec,
    render_vllm_env,
)
from llmctl.presets import load_all, load_one
from llmctl.services.preset_loader import load_presets, model_to_launch_spec


class UnknownPresetError(KeyError):
    """Raised when the requested preset alias isn't on disk."""


class MissingModelPathError(FileNotFoundError):
    """Raised when a preset's *local* model path is not present on disk."""


#: Prefixes that mark a preset ``model_id`` as a filesystem path rather than
#: a HuggingFace repo id. Repo ids ("org/name") never start with these.
_LOCAL_PATH_PREFIXES = ("/", "~", "./", "../")


@dataclass
class OrchestratorOptions:
    """Everything the daily-driver orchestrator can be told to do or skip.

    All defaults match production: enable fleet preflight + Harbor
    preflight + Hermes verify, no TQ override, do the real systemctl
    restart and wait for readiness.
    """

    tq_override: bool | None = None
    dry_run: bool = False
    #: Bypass the local-model-path refusal. The check only fires on a path
    #: that is provably absent, but an unreadable parent directory can make
    #: a present model look absent, and locking the operator out of their
    #: own unit is worse than one doomed restart.
    force: bool = False
    wait_for_ready: bool = True
    #: Readiness wait for a healthy but slow load. Was 300 s, and it had been
    #: kept short because a start that died still ran out the whole wait. The
    #: adapter now ends the wait as soon as the unit crashes, so this only
    #: bounds loads that are progressing: muse-glimmer-30b with DFlash needs
    #: ~330 s (compile + drafter capture), qwen3.8-27b-uncensored ~260 s.
    timeout_s: float = 900.0
    enable_fleet_preflight: bool = True
    enable_harbor_preflight: bool = True
    enable_hermes_verify: bool = True
    hermes_provider: str = "vllm"
    #: On a failed launch, put the env file and the stopped fleet back the
    #: way they were (see :func:`_rollback`). Off only for tests/debugging.
    rollback_on_failure: bool = True
    #: After a launch that was *verified* ready, start again what the
    #: preflight stopped (ollama, the Harbor container). They were only
    #: stopped so vLLM could size its KV cache from free VRAM at startup;
    #: once it has, they coexist (the WebUI embedder runs on CPU). Before
    #: this, a successful swap left ollama down and WebUI retrieval 400ing.
    restore_companions: bool = True


@dataclass
class RollbackReport:
    """What :func:`_rollback` managed to undo after a failed launch."""

    #: ``True`` restored, ``False`` tried and failed, ``None`` untouched
    #: (the failure happened before the env file was written).
    env_restored: bool | None = None
    env_path: Path | None = None
    #: The TP unit was stopped because the failed launch left it half-up.
    tp_stopped: bool = False
    restarted: list[str] = field(default_factory=list)
    restart_failed: list[str] = field(default_factory=list)
    #: ``None`` when the Harbor container was not stopped by this launch.
    harbor_restarted: bool | None = None

    @property
    def clean(self) -> bool:
        """``True`` when everything this launch changed was put back."""
        return (
            self.env_restored is not False
            and not self.restart_failed
            and self.harbor_restarted is not False
        )


@dataclass
class OrchestratorResult:
    """Aggregated outcome of one start operation."""

    spec: VLLMLaunchSpec
    restart: ManagedRestartResult | None = None
    fleet_stopped: list[str] = field(default_factory=list)
    fleet_failed: list[str] = field(default_factory=list)
    harbor_outcome: StopOutcome | None = None
    hermes_status: HermesStatus | None = None
    dry_run: bool = False
    rollback: RollbackReport | None = None
    #: Preflight-stopped services started again after a verified-ready launch.
    companions_restarted: list[str] = field(default_factory=list)
    companions_failed: list[str] = field(default_factory=list)
    #: Stopped and deliberately NOT restarted because readiness was not
    #: checked (``--no-wait``): vLLM may still be sizing its KV cache.
    companions_pending: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """``True`` when the launch succeeded end-to-end.

        ``restart.ready is None`` means ``--no-wait``: the restart was issued
        cleanly and nobody probed the endpoint. That is a success with an
        unknown endpoint, not a failure — exiting non-zero would punish the
        operator for a flag they chose. The *reporting* layer is what must not
        call it "ready" (see ``cli.vllm_cmd``).
        """
        if self.dry_run:
            return True
        if self.fleet_failed or self.restart is None:
            return False
        if self.restart.error:
            return False
        return self.restart.ready is not False


# Injection seam for tests — production code uses the real defaults.
@dataclass
class Dependencies:
    """Injected dependencies. Production callers pass nothing; tests
    construct one with fakes for each external interface."""

    config_dir: Path | None = None
    adapter_factory: Callable[..., VLLMSystemdAdapter] = VLLMSystemdAdapter
    systemctl: SystemctlRunner | None = None
    harbor_stop: Callable[..., StopOutcome] = stop_ollama_container
    harbor_start: Callable[..., bool] = start_ollama_container
    hermes_verify: Callable[..., HermesStatus] = verify_provider
    fleet_preflight: Callable[..., Any] = preflight_stop
    logger: Callable[[str], None] = print


def start_vllm_tp(
    preset_name: str,
    *,
    managed_unit: ManagedUnitConfig,
    defaults: VLLMDefaultsConfig | None = None,
    fleet: FleetUnitsConfig | None = None,
    options: OrchestratorOptions | None = None,
    deps: Dependencies | None = None,
) -> OrchestratorResult:
    """Start the TP-fleet unit on ``preset_name``.

    Lifecycle (when ``options`` defaults are kept):

    1. Load presets -> build :class:`VLLMLaunchSpec` (port from
       ``managed_unit.default_port``).
    2. Apply TQ override if requested.
    3. Stop competing systemd units (ollama).
    4. Stop the Harbor Ollama container if running.
    5. Write the env file + restart the unit.
    6. Wait for ``/v1/models`` to respond.
    7. Verify the Hermes ``vllm`` provider URL.

    Returns an :class:`OrchestratorResult` summarising every step.
    """
    deps = deps or Dependencies()
    options = options or OrchestratorOptions()
    fleet = fleet or FleetUnitsConfig()

    spec = _build_spec(
        preset_name,
        deps=deps,
        defaults=defaults,
        port_override=managed_unit.default_port,
        tq_override=options.tq_override,
    )

    if not options.force:
        _check_model_path(spec)

    return _run_lifecycle(
        spec=spec,
        managed_unit=managed_unit,
        fleet=fleet,
        fleet_role=FleetRole.TP,
        options=options,
        deps=deps,
        renderer=render_vllm_env,
    )


def preset_choices(
    *,
    defaults: VLLMDefaultsConfig | None = None,
    config_dir: Path | None = None,
) -> list[str]:
    """Return sorted preset aliases (for CLI tab-completion)."""
    return sorted(load_presets(defaults=defaults, config_dir=config_dir).keys())


def render_preset_env(
    preset_name: str,
    *,
    port: int,
    defaults: VLLMDefaultsConfig | None = None,
    config_dir: Path | None = None,
) -> tuple[VLLMLaunchSpec, str]:
    """Return ``(spec, env body)`` that ``llmctl swap PRESET`` would write.

    No TQ override and no side effects: the same render a ``--dry-run`` prints.
    Raises :class:`UnknownPresetError` like :func:`start_vllm_tp`.
    """
    spec = _build_spec(
        preset_name,
        deps=Dependencies(config_dir=config_dir),
        defaults=defaults,
        port_override=port,
        tq_override=None,
    )
    return spec, render_vllm_env(spec)


def _build_spec(
    preset_name: str,
    *,
    deps: Dependencies,
    defaults: VLLMDefaultsConfig | None,
    port_override: int,
    tq_override: bool | None,
) -> VLLMLaunchSpec:
    """Resolve preset -> spec, applying TQ override if requested.

    Preset access flows through ``deps.config_dir`` so tests can seed a
    real directory without patching process-wide XDG state.
    """
    model = load_one(preset_name, config_dir=deps.config_dir)
    if model is None:
        models = load_all(config_dir=deps.config_dir)
        available = ", ".join(sorted(models)) or (
            "(none - write one to ~/.config/llmctl/presets/)"
        )
        raise UnknownPresetError(
            f"unknown preset {preset_name!r}. Available: {available}"
        )

    base = model_to_launch_spec(
        model, defaults, port_override=port_override
    ).model_dump()
    base = apply_to_spec_dict(base, override=tq_override)
    return VLLMLaunchSpec.model_validate(base)


def _check_model_path(spec: VLLMLaunchSpec) -> None:
    """Refuse to swap the managed unit onto a local model path that is gone.

    ``llmctl vllm <preset>`` rewrites the env file and restarts the
    **single** TP unit, so a preset naming deleted weights first evicts
    whatever is currently serving and only then fails to load — the
    generator stays down until a human notices and re-points it by hand.
    That is precisely what a stale ``ornith-35b-refusal`` preset would
    have done after its checkpoint was deleted on 2026-07-26.

    Only **local paths** are checked. A HuggingFace repo id (``org/name``)
    resolves from the HF cache or is fetched on demand, so it fails OPEN —
    the same rule the scheduler's shard-completeness check follows. The
    refusal fires before fleet preflight, so a refused start never stops
    ollama or touches the env file.
    """
    model = (spec.model or "").strip()
    if not model.startswith(_LOCAL_PATH_PREFIXES):
        return
    path = Path(model).expanduser()
    if path.exists():
        return
    raise MissingModelPathError(
        f"preset model path does not exist: {path}. Refusing to restart the "
        f"unit onto missing weights — it would evict the currently served "
        f"model and then fail to load. Fix the preset's model_id, or pass "
        f"--force to override."
    )


def _run_lifecycle(
    *,
    spec: VLLMLaunchSpec,
    managed_unit: ManagedUnitConfig,
    fleet: FleetUnitsConfig,
    fleet_role: FleetRole,
    options: OrchestratorOptions,
    deps: Dependencies,
    renderer: Callable[[VLLMLaunchSpec], str],
    hermes_provider_override: str | None = None,
) -> OrchestratorResult:
    """Start lifecycle for the TP unit."""
    result = OrchestratorResult(spec=spec, dry_run=options.dry_run)
    systemctl = deps.systemctl or SystemctlRunner()

    if options.dry_run:
        # Build the body without writing — the CLI prints it for the operator.
        body = renderer(spec)
        deps.logger(f"--- dry-run: would write {managed_unit.resolve_env_file()}: ---")
        for line in body.splitlines():
            deps.logger(line)
        deps.logger("--- dry-run: would systemctl restart " + managed_unit.unit_name)
        return result

    if options.enable_fleet_preflight:
        report = deps.fleet_preflight(
            fleet_role,
            fleet,
            systemctl,
            logger=deps.logger,
        )
        result.fleet_stopped = list(report.stopped)
        result.fleet_failed = list(report.failed)
        if not report.all_clean:
            # A partial preflight still stopped some units; put them back.
            if options.rollback_on_failure and result.fleet_stopped:
                result.rollback = _rollback(
                    result, managed_unit, systemctl, deps, env_written=False
                )
            return result

    if options.enable_harbor_preflight:
        result.harbor_outcome = deps.harbor_stop(
            container=DEFAULT_OLLAMA_CONTAINER,
            logger=deps.logger,
        )

    adapter = deps.adapter_factory(
        managed_unit,
        systemctl=systemctl,
        renderer=renderer,
    )
    # Captured before the adapter overwrites it so a failed launch can put
    # the previous preset back instead of leaving the new (broken) one on
    # disk for the next `systemctl restart` to pick up.
    env_path: Path | None = getattr(adapter, "env_file_path", None)
    prior_env = _read_prior_env(env_path)

    try:
        result.restart = adapter.restart_with_spec(
            spec,
            wait_for_ready=options.wait_for_ready,
            timeout_s=options.timeout_s,
        )
    except Exception:
        if options.rollback_on_failure:
            result.rollback = _rollback(
                result,
                managed_unit,
                systemctl,
                deps,
                env_written=True,
                env_path=env_path,
                prior_env=prior_env,
            )
        raise

    if options.rollback_on_failure and not result.ok:
        result.rollback = _rollback(
            result,
            managed_unit,
            systemctl,
            deps,
            env_written=True,
            env_path=env_path,
            prior_env=prior_env,
        )
        return result

    if options.restore_companions:
        _restore_companions(result, {managed_unit.unit_name, fleet.tp}, systemctl, deps)

    if options.enable_hermes_verify and (
        result.restart is not None and result.restart.ready
    ):
        provider_name = hermes_provider_override or options.hermes_provider
        result.hermes_status = deps.hermes_verify(
            provider_name,
            expected_port=spec.port,
            config_path=HERMES_DEFAULT_CONFIG_PATH,
            logger=deps.logger,
        )

    return result


def _restore_companions(
    result: OrchestratorResult,
    tp_units: set[str],
    systemctl: SystemctlRunner,
    deps: Dependencies,
) -> None:
    """Start again what the preflight stopped, once vLLM holds its memory.

    Only services this launch stopped, never the TP unit itself (it is in
    ``fleet_stopped`` when it was running, and the restart already brought
    it back). Without verified readiness nothing is started: vLLM may still
    be profiling free VRAM, which is the reason they were stopped.
    """
    names = [u for u in result.fleet_stopped if u not in tp_units]
    harbor = result.harbor_outcome is StopOutcome.STOPPED
    if not names and not harbor:
        return
    ready = result.restart is not None and result.restart.ready is True
    if not ready:
        result.companions_pending = names + ([DEFAULT_OLLAMA_CONTAINER] if harbor else [])
        return
    for unit in names:
        try:
            ok = systemctl.start(unit).ok
        except Exception as exc:  # noqa: BLE001 - report, never fail a good launch
            ok = False
            deps.logger(f"  could not start {unit}: {exc}")
        (result.companions_restarted if ok else result.companions_failed).append(unit)
        if ok:
            deps.logger(f"  started {unit} again (vLLM holds its memory now)")
    if harbor:
        try:
            ok = bool(deps.harbor_start(container=DEFAULT_OLLAMA_CONTAINER, logger=deps.logger))
        except Exception as exc:  # noqa: BLE001
            ok = False
            deps.logger(f"  could not start {DEFAULT_OLLAMA_CONTAINER}: {exc}")
        target = result.companions_restarted if ok else result.companions_failed
        target.append(DEFAULT_OLLAMA_CONTAINER)


#: Sentinel for "the env file did not exist before the launch".
_ABSENT = object()


def _read_prior_env(env_path: Path | None) -> str | object | None:
    """Return the env file body, :data:`_ABSENT`, or ``None`` if unknown."""
    if env_path is None:
        return None
    try:
        return env_path.read_text()
    except FileNotFoundError:
        return _ABSENT
    except OSError:
        return None


def _rollback(
    result: OrchestratorResult,
    managed_unit: ManagedUnitConfig,
    systemctl: SystemctlRunner,
    deps: Dependencies,
    *,
    env_written: bool,
    env_path: Path | None = None,
    prior_env: str | object | None = None,
) -> RollbackReport:
    """Best-effort undo of a failed launch; every step runs and is reported.

    Order: stop the half-started TP unit (it holds the VRAM the fleet needs
    and is running the spec we are about to un-write), restore the previous
    env file, then start every unit the preflight stopped in its original
    order. If ``vllm-tp`` itself was among them, starting it relaunches the
    *previous* preset from the restored env — the state before the swap.
    """
    log = deps.logger
    report = RollbackReport(env_path=env_path)
    log("  rollback: launch failed, restoring the previous state")

    if env_written:
        try:
            report.tp_stopped = systemctl.try_stop(managed_unit.unit_name)
            if report.tp_stopped:
                log(f"  rollback: stopped {managed_unit.unit_name}")
        except Exception as exc:  # noqa: BLE001 - rollback must keep going
            log(f"  rollback: could not stop {managed_unit.unit_name}: {exc}")

        if env_path is not None and prior_env is not None:
            try:
                if prior_env is _ABSENT:
                    env_path.unlink(missing_ok=True)
                else:
                    atomic_write_text(env_path, str(prior_env))
                report.env_restored = True
                log(f"  rollback: restored {env_path}")
            except OSError as exc:
                report.env_restored = False
                log(f"  rollback: FAILED to restore {env_path}: {exc}")
        else:
            report.env_restored = False
            log("  rollback: previous env file unknown; not restored")

    for unit in result.fleet_stopped:
        try:
            ok = systemctl.start(unit).ok
        except Exception as exc:  # noqa: BLE001 - rollback must keep going
            ok = False
            log(f"  rollback: could not start {unit}: {exc}")
        if ok:
            report.restarted.append(unit)
            log(f"  rollback: started {unit}")
        else:
            report.restart_failed.append(unit)
            log(f"  rollback: FAILED to start {unit}")

    if result.harbor_outcome is StopOutcome.STOPPED:
        try:
            report.harbor_restarted = bool(
                deps.harbor_start(container=DEFAULT_OLLAMA_CONTAINER, logger=log)
            )
        except Exception as exc:  # noqa: BLE001 - rollback must keep going
            report.harbor_restarted = False
            log(f"  rollback: could not start {DEFAULT_OLLAMA_CONTAINER}: {exc}")

    return report
