"""High-level vLLM start orchestration.

Glues the pieces from Phases 1–4 into the two operations the daily
driver actually wants:

* :func:`start_vllm_tp` — start the TP-fleet unit on a preset.
* :func:`start_slot` — start a per-GPU slot unit on a preset.

Each composes: preset loading -> TQ override -> spec build ->
fleet preflight -> Harbor preflight -> adapter restart -> readiness
poll -> Hermes verify. Everything is injectable so the CLI tests can
substitute fakes without touching real systemd / docker / hermes.

The CLI commands in ``llmctl/cli.py`` are thin wrappers around these
two functions plus :func:`preset_choices` for tab-completion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from llmctl.adapters.vllm_systemd import ManagedRestartResult, VLLMSystemdAdapter
from llmctl.config import (
    ManagedUnitConfig,
    SlotConfig,
    VLLMDefaultsConfig,
)
from llmctl.integrations.fleet import FleetRole, FleetUnitsConfig, preflight_stop
from llmctl.integrations.harbor import (
    DEFAULT_OLLAMA_CONTAINER,
    StopOutcome,
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
    VLLMSlotInfo,
    render_slot_env,
    render_vllm_env,
)
from llmctl.services.preset_loader import load_presets, model_to_launch_spec
from llmctl.services.preset_store import PresetStore, default_store


class UnknownPresetError(KeyError):
    """Raised when the requested preset alias isn't on disk."""


@dataclass
class OrchestratorOptions:
    """Everything the daily-driver orchestrator can be told to do or skip.

    All defaults match production: enable fleet preflight + Harbor
    preflight + Hermes verify, no TQ override, do the real systemctl
    restart and wait for readiness.
    """

    tq_override: bool | None = None
    dry_run: bool = False
    wait_for_ready: bool = True
    timeout_s: float = 300.0
    enable_fleet_preflight: bool = True
    enable_harbor_preflight: bool = True
    enable_hermes_verify: bool = True
    hermes_provider: str = "vllm"


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

    @property
    def ok(self) -> bool:
        """``True`` when the launch succeeded end-to-end."""
        if self.dry_run:
            return True
        if not self.fleet_failed and self.restart is not None and self.restart.ready:
            return True
        return False


# Injection seam for tests — production code uses the real defaults.
@dataclass
class Dependencies:
    """Injected dependencies. Production callers pass nothing; tests
    construct one with fakes for each external interface."""

    # Single source of truth for presets. Everything else (presets_loader,
    # _build_spec) routes through this so tests inject one stub instead
    # of three.
    preset_store: PresetStore = field(default_factory=default_store)
    adapter_factory: Callable[..., VLLMSystemdAdapter] = VLLMSystemdAdapter
    systemctl: SystemctlRunner | None = None
    harbor_stop: Callable[..., StopOutcome] = stop_ollama_container
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
    3. Stop competing systemd units (agents.target, slots, ollama).
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

    return _run_lifecycle(
        spec=spec,
        managed_unit=managed_unit,
        fleet=fleet,
        fleet_role=FleetRole.TP,
        options=options,
        deps=deps,
        renderer=render_vllm_env,
    )


def start_slot(
    slot_name: str,
    preset_name: str,
    *,
    slot_config: SlotConfig,
    managed_unit: ManagedUnitConfig | None = None,
    defaults: VLLMDefaultsConfig | None = None,
    fleet: FleetUnitsConfig | None = None,
    options: OrchestratorOptions | None = None,
    deps: Dependencies | None = None,
) -> OrchestratorResult:
    """Start a per-GPU slot unit on ``preset_name``.

    ``slot_name`` becomes the served name (downstream client configs
    keep pointing at ``coder``/``reasoner`` regardless of the model
    swap). The slot's GPU, port, and unit name come from
    ``slot_config``.

    The fleet role used for preflight is :attr:`FleetRole.CODER` for
    the coder slot and :attr:`FleetRole.REASONER` otherwise — slots
    are designed to coexist with their sibling slot but conflict with
    the TP unit + ollama.
    """
    deps = deps or Dependencies()
    options = options or OrchestratorOptions()
    fleet = fleet or FleetUnitsConfig()
    options = _slot_default_options(options)
    managed_unit = managed_unit or ManagedUnitConfig(
        unit_name=slot_config.unit_name,
        env_file_path=slot_config.resolve_env_file(slot_name),
        default_port=slot_config.port,
    )

    spec = _build_spec(
        preset_name,
        deps=deps,
        defaults=defaults,
        port_override=slot_config.port,
        tq_override=options.tq_override,
    )

    role = FleetRole.CODER if slot_name == "coder" else FleetRole.REASONER
    slot_info = VLLMSlotInfo(name=slot_name, gpu=slot_config.gpu, port=slot_config.port)
    renderer = partial(render_slot_env, slot=slot_info)

    return _run_lifecycle(
        spec=spec,
        managed_unit=managed_unit,
        fleet=fleet,
        fleet_role=role,
        options=options,
        deps=deps,
        renderer=renderer,
        hermes_provider_override=f"vllm-{slot_name}",
    )


def preset_choices(
    *,
    defaults: VLLMDefaultsConfig | None = None,
    store: PresetStore | None = None,
) -> list[str]:
    """Return sorted preset aliases (for CLI tab-completion)."""
    return sorted(load_presets(defaults=defaults, store=store).keys())


def _build_spec(
    preset_name: str,
    *,
    deps: Dependencies,
    defaults: VLLMDefaultsConfig | None,
    port_override: int,
    tq_override: bool | None,
) -> VLLMLaunchSpec:
    """Resolve preset -> spec, applying TQ override if requested.

    All preset access flows through ``deps.preset_store`` so the same
    fixture seeds every code path in a test — no more separate hooks
    for "list presets" vs "load one preset for rendering."
    """
    models = deps.preset_store.load()
    if preset_name not in models:
        available = ", ".join(sorted(models)) or "(none — write one to ~/.config/llm-models/)"
        raise UnknownPresetError(
            f"unknown preset {preset_name!r}. Available: {available}"
        )

    base = model_to_launch_spec(
        models[preset_name], defaults, port_override=port_override
    ).model_dump()
    base = apply_to_spec_dict(base, override=tq_override)
    return VLLMLaunchSpec.model_validate(base)


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
    """Shared start lifecycle for TP and slot units."""
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
    result.restart = adapter.restart_with_spec(
        spec,
        wait_for_ready=options.wait_for_ready,
        timeout_s=options.timeout_s,
    )

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


def _slot_default_options(options: OrchestratorOptions) -> OrchestratorOptions:
    """Slot starts default to verifying the slot-specific provider name."""
    if options.hermes_provider == "vllm":  # default — caller didn't override
        options.hermes_provider = ""  # let _run_lifecycle pick the override
    return options
