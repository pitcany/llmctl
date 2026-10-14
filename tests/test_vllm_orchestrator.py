"""Unit tests for :mod:`llmctl.services.vllm_orchestrator`.

Tests seed a real preset directory and inject ``Dependencies.config_dir``
so preset parsing follows the production path without process-wide XDG
patching.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llmctl.adapters.vllm_systemd import ManagedRestartResult
from llmctl.config import (
    FleetUnitsConfig,
    ManagedUnitConfig,
)
from llmctl.integrations.fleet import FleetRole
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.integrations.vllm_env import VLLMLaunchSpec
from llmctl.services.vllm_orchestrator import (
    Dependencies,
    OrchestratorOptions,
    OrchestratorResult,
    UnknownPresetError,
    start_vllm_tp,
)


def _write_preset(
    config_dir: Path,
    alias: str = "llama-3.3-70b",
    *,
    model_id: str = "casperhansen/llama-3.3-70b-instruct-awq",
    tp: int = 2,
    kv: str = "fp8",
    reasoning_parser: str | None = None,
) -> Path:
    """Write a preset YAML for orchestrator tests."""
    config_dir.mkdir(parents=True, exist_ok=True)
    reasoning_line = (
        f"reasoning_parser: {reasoning_parser}\n"
        if reasoning_parser
        else "reasoning_parser: null\n"
    )
    body = f"""
    alias: {alias}
    served_name: {alias}
    model_id: {model_id}
    quantization: awq
    vllm_quantization_flag: awq_marlin
    tensor_parallel_size: {tp}
    max_model_len: 32768
    max_num_seqs: 64
    kv_cache_dtype: {kv}
    tool_parser: llama3_json
    {reasoning_line.rstrip()}
    """
    path = config_dir / f"{alias}.yaml"
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path


@dataclass
class _FleetReport:
    """Stand-in for fleet.StopReport (free constructor for tests)."""

    stopped: list[str]
    skipped: list[str]
    failed: list[str]

    @property
    def all_clean(self) -> bool:
        return not self.failed


class _FakeCompanion:
    """A plugin companion (e.g. a container) that records stops and starts."""

    name = "ollama-container"
    start_hint = "docker start ollama-container"

    def __init__(self, log: dict[str, Any], *, running: bool) -> None:
        self._log = log
        self._running = running

    def stop(self, *, logger: Any) -> bool | None:
        self._log["companion_stops"].append(self.name)
        return True if self._running else None

    def start(self, *, logger: Any) -> bool:
        self._log["companion_starts"].append(self.name)
        return True

    def describe(self) -> str:
        return f"stops the container {self.name}, if running"


class _FakeCheck:
    """A plugin post-start check that records what it was asked to verify."""

    name = "agent-config"

    def __init__(self, log: dict[str, Any]) -> None:
        self._log = log

    def run(self, spec: Any, *, logger: Any) -> object:
        self._log["check_calls"].append({"port": spec.port})
        return "ok"

    def describe(self) -> str:
        return "checks the agent config"


class _AdapterStub:
    """Drop-in for VLLMSystemdAdapter capturing all side effects."""

    def __init__(self, *, ready: bool | None = True, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.spec_received: VLLMLaunchSpec | None = None
        self.wait_for_ready_arg: bool | None = None
        self.timeout_arg: float | None = None
        #: Readiness verdict this stub reports back (see the tri-state note
        #: at the bottom of this file).
        self._ready = ready

    def restart_with_spec(
        self,
        spec: VLLMLaunchSpec,
        *,
        wait_for_ready: bool = True,
        timeout_s: float = 300.0,
        poll_interval_s: float = 5.0,
    ) -> ManagedRestartResult:
        self.spec_received = spec
        self.wait_for_ready_arg = wait_for_ready
        self.timeout_arg = timeout_s
        return ManagedRestartResult(
            env_path=Path("/tmp/fake.env"),
            env_body="rendered\n",
            ready=self._ready,
        )


def _build_deps(
    *,
    config_dir: Path,
    adapter: _AdapterStub | None = None,
    companion_running: bool = False,
    fleet_report: _FleetReport | None = None,
) -> tuple[Dependencies, dict[str, Any]]:
    """Return Dependencies with all externals stubbed + a call log."""
    log: dict[str, Any] = {
        "preflight_calls": [],
        "companion_stops": [],
        "companion_starts": [],
        "check_calls": [],
        "adapters_built": [],
    }
    adapter = adapter or _AdapterStub()
    fleet_report = fleet_report or _FleetReport(stopped=[], skipped=[], failed=[])

    def fake_adapter(*args: Any, **kwargs: Any) -> _AdapterStub:
        log["adapters_built"].append({"args": args, "kwargs": kwargs})
        return adapter

    def fake_preflight(
        role: FleetRole,
        fleet: FleetUnitsConfig,
        sysctl: SystemctlRunner,
        **kw: Any,
    ) -> _FleetReport:
        log["preflight_calls"].append({"role": role, "fleet": fleet})
        return fleet_report

    deps = Dependencies(
        config_dir=config_dir,
        adapter_factory=fake_adapter,
        systemctl=SystemctlRunner(runner=lambda argv: None),  # type: ignore[arg-type]
        companions=[_FakeCompanion(log, running=companion_running)],
        post_start_checks=[_FakeCheck(log)],
        fleet_preflight=fake_preflight,
        logger=lambda _: None,
    )
    return deps, log


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/opt/python")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/tmp/hf")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


def test_unknown_preset_raises(tmp_path: Path) -> None:
    deps, _ = _build_deps(config_dir=tmp_path)
    with pytest.raises(UnknownPresetError, match="unknown preset"):
        start_vllm_tp(
            "nonexistent",
            managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
            deps=deps,
        )


def test_happy_path_lifecycle_order(tmp_path: Path) -> None:
    """Full lifecycle: preflight -> companions -> adapter -> checks."""
    adapter = _AdapterStub()
    _write_preset(tmp_path)
    deps, log = _build_deps(config_dir=tmp_path, adapter=adapter)

    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        deps=deps,
    )

    assert result.ok is True
    assert len(log["preflight_calls"]) == 1
    assert log["preflight_calls"][0]["role"] is FleetRole.TP
    assert log["companion_stops"] == ["ollama-container"]
    assert log["check_calls"] == [{"port": 8003}]
    assert result.checks == {"agent-config": "ok"}
    assert len(log["adapters_built"]) == 1
    assert adapter.spec_received is not None
    assert adapter.spec_received.served_name == "llama-3.3-70b"
    assert adapter.spec_received.port == 8003


def test_dry_run_skips_all_side_effects(tmp_path: Path) -> None:
    _write_preset(tmp_path)
    deps, log = _build_deps(config_dir=tmp_path)

    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        options=OrchestratorOptions(dry_run=True),
        deps=deps,
    )

    assert result.dry_run is True
    assert result.ok is True
    assert log["preflight_calls"] == []
    assert log["companion_stops"] == []
    assert log["check_calls"] == []
    assert log["adapters_built"] == []


def test_failed_fleet_preflight_aborts_before_adapter(tmp_path: Path) -> None:
    _write_preset(tmp_path)
    deps, log = _build_deps(
        config_dir=tmp_path,
        fleet_report=_FleetReport(
            stopped=["ollama"],
            skipped=[],
            failed=["vllm-tp"],
        ),
    )

    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        deps=deps,
    )

    assert result.ok is False
    assert result.fleet_failed == ["vllm-tp"]
    assert log["companion_stops"] == []
    assert log["adapters_built"] == []
    assert log["check_calls"] == []


def test_tq_override_on_sets_turboquant_kv_dtype(tmp_path: Path) -> None:
    adapter = _AdapterStub()
    _write_preset(tmp_path)
    deps, _ = _build_deps(config_dir=tmp_path, adapter=adapter)

    start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        options=OrchestratorOptions(tq_override=True),
        deps=deps,
    )

    assert adapter.spec_received is not None
    assert adapter.spec_received.kv_cache_type is not None
    assert adapter.spec_received.kv_cache_type.startswith("turboquant_")


def test_tq_override_off_clears_kv_dtype(tmp_path: Path) -> None:
    adapter = _AdapterStub()
    _write_preset(tmp_path)
    deps, _ = _build_deps(config_dir=tmp_path, adapter=adapter)

    start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        options=OrchestratorOptions(tq_override=False),
        deps=deps,
    )

    assert adapter.spec_received is not None
    assert adapter.spec_received.kv_cache_type is None


def test_disabling_integrations_skips_them(tmp_path: Path) -> None:
    _write_preset(tmp_path)
    deps, log = _build_deps(config_dir=tmp_path)

    start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        options=OrchestratorOptions(
            enable_fleet_preflight=False,
            enable_companions=False,
            enable_post_start_checks=False,
        ),
        deps=deps,
    )

    assert log["preflight_calls"] == []
    assert log["companion_stops"] == []
    assert log["check_calls"] == []
    assert len(log["adapters_built"]) == 1


def test_wait_for_ready_false_skips_post_restart_polling(tmp_path: Path) -> None:
    adapter = _AdapterStub()
    _write_preset(tmp_path)
    deps, _ = _build_deps(config_dir=tmp_path, adapter=adapter)

    start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        options=OrchestratorOptions(wait_for_ready=False),
        deps=deps,
    )

    assert adapter.wait_for_ready_arg is False


# --- readiness tri-state ------------------------------------------------------
#
# `ready` is bool | None: True = probed and serving, False = probed and not
# serving, None = `--no-wait`, nobody looked. `ok` must not confuse "I did not
# check" with "it failed", or the CLI exits 1 on a restart the operator
# deliberately chose not to wait for.


def _result(ready: bool | None, *, error: str | None = None) -> OrchestratorResult:
    return OrchestratorResult(
        spec=VLLMLaunchSpec(model="m", served_name="s"),
        restart=ManagedRestartResult(
            env_path=Path("/tmp/fake.env"), env_body="", ready=ready, error=error
        ),
    )


def test_ok_is_true_when_probed_and_ready() -> None:
    assert _result(True).ok is True


def test_ok_is_false_when_probed_and_not_ready() -> None:
    assert _result(False, error="vLLM did not become ready within 300s").ok is False


def test_ok_is_true_when_readiness_was_not_checked() -> None:
    """`--no-wait` is a successful restart with an unknown endpoint, not a failure."""
    assert _result(None).ok is True


def test_ok_is_false_when_the_restart_itself_errored_unchecked() -> None:
    """A failed `systemctl restart` reports ready=False + error; stays a failure."""
    assert _result(False, error="systemctl restart vllm-tp failed (exit 1)").ok is False


def test_ok_is_false_when_the_fleet_preflight_failed() -> None:
    result = _result(None)
    result.fleet_failed = ["ollama"]
    assert result.ok is False


def test_post_start_checks_are_skipped_when_readiness_was_not_checked(tmp_path: Path) -> None:
    """Checking anything against an unprobed endpoint reports nothing useful."""
    adapter = _AdapterStub(ready=None)
    _write_preset(tmp_path)
    deps, _ = _build_deps(config_dir=tmp_path, adapter=adapter)

    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        options=OrchestratorOptions(wait_for_ready=False),
        deps=deps,
    )

    assert result.restart is not None and result.restart.ready is None
    assert result.checks == {}


# --- rollback on a failed swap (FIXPLAN 2026-09-18 §1.1) ------------------


class _Completed:
    def __init__(self, returncode: int = 0, stdout: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


class _RecordingSystemctl:
    """argv recorder; ``vllm-tp`` reads active so the rollback stops it."""

    def __init__(self, *, fail_start: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_start = fail_start or set()

    def __call__(self, argv: list[str]) -> _Completed:
        self.calls.append(argv)
        verb, unit = argv[-2], argv[-1]
        if verb == "is-active":
            return _Completed(stdout="active" if unit == "vllm-tp" else "inactive")
        if verb == "start" and unit in self.fail_start:
            return _Completed(returncode=1)
        return _Completed()

    def verbs(self) -> list[tuple[str, str]]:
        return [(a[-2], a[-1]) for a in self.calls]


class _EnvWritingAdapter(_AdapterStub):
    """Writes the new env like the real adapter, then reports a failure."""

    def __init__(self, env_file_path: Path, *, raises: bool = False, **kw: Any) -> None:
        super().__init__(ready=False, **kw)
        self.env_file_path = env_file_path
        self.raises = raises

    def restart_with_spec(self, spec: VLLMLaunchSpec, **kw: Any) -> ManagedRestartResult:
        self.env_file_path.write_text("NEW-PRESET\n")
        if self.raises:
            raise RuntimeError("pre-start hook blew up")
        return ManagedRestartResult(
            env_path=self.env_file_path,
            env_body="NEW-PRESET\n",
            ready=False,
            error="vLLM did not become ready within 300s",
        )


def _rollback_setup(
    tmp_path: Path, *, raises: bool = False, fail_start: set[str] | None = None
) -> tuple[Dependencies, dict[str, Any], _RecordingSystemctl, Path, list[str]]:
    presets = tmp_path / "presets"
    _write_preset(presets)
    env = tmp_path / "vllm-tp.env"
    env.write_text("OLD-PRESET\n")
    adapter = _EnvWritingAdapter(env, raises=raises)
    deps, log = _build_deps(
        config_dir=presets,
        adapter=adapter,
        companion_running=True,
        fleet_report=_FleetReport(stopped=["ollama", "vllm-tp"], skipped=[], failed=[]),
    )
    sysctl = _RecordingSystemctl(fail_start=fail_start)
    deps.systemctl = SystemctlRunner(runner=sysctl)
    return deps, log, sysctl, env, log["companion_starts"]


def test_failed_readiness_rolls_back_env_and_restarts_fleet(tmp_path: Path) -> None:
    deps, log, sysctl, env, harbor_started = _rollback_setup(tmp_path)

    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        deps=deps,
    )

    assert result.ok is False
    assert env.read_text() == "OLD-PRESET\n"
    rb = result.rollback
    assert rb is not None and rb.clean
    assert rb.env_restored is True
    assert rb.tp_stopped is True
    assert rb.restarted == ["ollama", "vllm-tp", "ollama-container"]
    assert harbor_started == ["ollama-container"]
    # The half-started unit is stopped before the env is restored and the
    # previous units are started, in preflight order.
    assert sysctl.verbs()[-3:] == [("stop", "vllm-tp"), ("start", "ollama"), ("start", "vllm-tp")]
    assert log["check_calls"] == []


def test_adapter_exception_still_rolls_back(tmp_path: Path) -> None:
    deps, _, _, env, _ = _rollback_setup(tmp_path, raises=True)

    with pytest.raises(RuntimeError, match="blew up"):
        start_vllm_tp(
            "llama-3.3-70b",
            managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
            deps=deps,
        )

    assert env.read_text() == "OLD-PRESET\n"


def test_rollback_reports_units_it_could_not_start(tmp_path: Path) -> None:
    deps, _, _, _, _ = _rollback_setup(tmp_path, fail_start={"ollama"})

    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        deps=deps,
    )

    rb = result.rollback
    assert rb is not None and not rb.clean
    assert rb.restart_failed == ["ollama"]
    assert rb.restarted == ["vllm-tp", "ollama-container"]


def test_partial_preflight_failure_restarts_what_it_stopped(tmp_path: Path) -> None:
    _write_preset(tmp_path)
    deps, log = _build_deps(
        config_dir=tmp_path,
        fleet_report=_FleetReport(stopped=["ollama"], skipped=[], failed=["vllm-tp"]),
    )
    sysctl = _RecordingSystemctl()
    deps.systemctl = SystemctlRunner(runner=sysctl)

    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        deps=deps,
    )

    assert result.rollback is not None
    assert result.rollback.env_restored is None  # env was never written
    assert result.rollback.restarted == ["ollama"]
    assert ("stop", "vllm-tp") not in sysctl.verbs()
    assert log["adapters_built"] == []


def test_successful_launch_does_not_roll_back(tmp_path: Path) -> None:
    _write_preset(tmp_path)
    deps, _ = _build_deps(
        config_dir=tmp_path,
        fleet_report=_FleetReport(stopped=["ollama"], skipped=[], failed=[]),
    )
    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        deps=deps,
    )
    assert result.ok is True
    assert result.rollback is None


# --- companions: what the preflight paused comes back after a verified launch ---


def _companion_setup(tmp_path: Path, *, ready: bool | None, fail_start: set[str] | None = None):
    _write_preset(tmp_path)
    deps, log = _build_deps(
        config_dir=tmp_path,
        adapter=_AdapterStub(ready=ready),
        companion_running=True,
        fleet_report=_FleetReport(stopped=["ollama", "vllm-tp"], skipped=[], failed=[]),
    )
    sysctl = _RecordingSystemctl(fail_start=fail_start)
    deps.systemctl = SystemctlRunner(runner=sysctl)
    harbor_started = log["companion_starts"]
    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        deps=deps,
    )
    return result, sysctl, harbor_started


def test_ready_launch_starts_paused_services_again_but_not_the_tp_unit(tmp_path: Path) -> None:
    """A successful swap used to leave ollama down (WebUI retrieval 400s)."""
    result, sysctl, harbor_started = _companion_setup(tmp_path, ready=True)

    assert result.ok is True
    assert result.companions_restarted == ["ollama", "ollama-container"]
    assert ("start", "ollama") in sysctl.verbs()
    # vllm-tp was in fleet_stopped too; the restart already brought it back.
    assert ("start", "vllm-tp") not in sysctl.verbs()
    assert harbor_started == ["ollama-container"]
    assert result.start_hints["ollama-container"] == "docker start ollama-container"


def test_unverified_launch_leaves_them_paused_and_says_so(tmp_path: Path) -> None:
    """--no-wait: vLLM may still be sizing its KV cache from free VRAM."""
    result, sysctl, harbor_started = _companion_setup(tmp_path, ready=None)

    assert result.ok is True
    assert result.companions_restarted == []
    assert result.companions_pending == ["ollama", "ollama-container"]
    assert ("start", "ollama") not in sysctl.verbs()
    assert harbor_started == []


def test_a_companion_that_will_not_start_is_reported_not_fatal(tmp_path: Path) -> None:
    result, _, _ = _companion_setup(tmp_path, ready=True, fail_start={"ollama"})

    assert result.ok is True
    assert result.companions_failed == ["ollama"]
    assert result.companions_restarted == ["ollama-container"]


def test_failed_launch_restores_via_rollback_not_companions(tmp_path: Path) -> None:
    deps, _, _, _, _ = _rollback_setup(tmp_path)
    result = start_vllm_tp(
        "llama-3.3-70b",
        managed_unit=ManagedUnitConfig(unit_name="vllm-tp", default_port=8003),
        deps=deps,
    )
    assert result.ok is False
    assert result.companions_restarted == [] and result.companions_pending == []
