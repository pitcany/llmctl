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
from llmctl.integrations.harbor import StopOutcome
from llmctl.integrations.hermes import HermesStatus
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.integrations.vllm_env import VLLMLaunchSpec
from llmctl.services.vllm_orchestrator import (
    Dependencies,
    OrchestratorOptions,
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


class _AdapterStub:
    """Drop-in for VLLMSystemdAdapter capturing all side effects."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.spec_received: VLLMLaunchSpec | None = None
        self.wait_for_ready_arg: bool | None = None
        self.timeout_arg: float | None = None

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
            ready=True,
        )


def _build_deps(
    *,
    config_dir: Path,
    adapter: _AdapterStub | None = None,
    harbor_outcome: StopOutcome = StopOutcome.NOT_RUNNING,
    hermes_status: HermesStatus = HermesStatus.OK,
    fleet_report: _FleetReport | None = None,
) -> tuple[Dependencies, dict[str, Any]]:
    """Return Dependencies with all externals stubbed + a call log."""
    log: dict[str, Any] = {
        "preflight_calls": [],
        "harbor_calls": [],
        "hermes_calls": [],
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

    def fake_harbor(*args: Any, **kwargs: Any) -> StopOutcome:
        log["harbor_calls"].append(kwargs)
        return harbor_outcome

    def fake_hermes(provider: str, **kwargs: Any) -> HermesStatus:
        log["hermes_calls"].append({"provider": provider, **kwargs})
        return hermes_status

    deps = Dependencies(
        config_dir=config_dir,
        adapter_factory=fake_adapter,
        systemctl=SystemctlRunner(runner=lambda argv: None),  # type: ignore[arg-type]
        harbor_stop=fake_harbor,
        hermes_verify=fake_hermes,
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
    """Full lifecycle: preflight -> harbor -> adapter -> hermes."""
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
    assert len(log["harbor_calls"]) == 1
    assert len(log["hermes_calls"]) == 1
    assert log["hermes_calls"][0]["provider"] == "vllm"
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
    assert log["harbor_calls"] == []
    assert log["hermes_calls"] == []
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
    assert log["harbor_calls"] == []
    assert log["adapters_built"] == []
    assert log["hermes_calls"] == []


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
            enable_harbor_preflight=False,
            enable_hermes_verify=False,
        ),
        deps=deps,
    )

    assert log["preflight_calls"] == []
    assert log["harbor_calls"] == []
    assert log["hermes_calls"] == []
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
