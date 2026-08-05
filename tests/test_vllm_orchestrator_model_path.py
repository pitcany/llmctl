"""Orchestrator refusal when a preset's local model path is gone.

``llmctl vllm <preset>`` rewrites the env file and restarts the SINGLE TP
unit, so a preset naming deleted weights evicts whatever is currently
serving and only then fails to load. The refusal must fire before any side
effect (fleet preflight stops ollama, Harbor stops the container, the
adapter writes the env file).

HuggingFace repo ids must fail OPEN — they resolve from the HF cache or are
fetched on demand, so an absent local directory says nothing about them.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from llmctl.config import FleetUnitsConfig, ManagedUnitConfig
from llmctl.integrations.fleet import FleetRole
from llmctl.integrations.harbor import StopOutcome
from llmctl.integrations.hermes import HermesStatus
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.services.vllm_orchestrator import (
    Dependencies,
    MissingModelPathError,
    OrchestratorOptions,
    start_vllm_tp,
)


def _write_preset(config_dir: Path, alias: str, model_id: str) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    body = f"""
    alias: {alias}
    served_name: {alias}
    model_id: {model_id}
    quantization: awq
    vllm_quantization_flag: awq_marlin
    tensor_parallel_size: 2
    max_model_len: 32768
    max_num_seqs: 64
    kv_cache_dtype: fp8
    tool_parser: llama3_json
    reasoning_parser: null
    """
    (config_dir / f"{alias}.yaml").write_text(textwrap.dedent(body).strip() + "\n")


class _ExplodingAdapter:
    """Any use of this adapter means the refusal came too late."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def restart_with_spec(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("adapter must not run when the model path is missing")


def _deps(config_dir: Path) -> tuple[Dependencies, dict[str, list]]:
    """Dependencies whose side-effecting stubs record every invocation."""
    log: dict[str, list] = {"preflight": [], "harbor": [], "hermes": []}

    def fake_preflight(
        role: FleetRole, fleet: FleetUnitsConfig, sysctl: SystemctlRunner, **kw: Any
    ) -> Any:
        log["preflight"].append(role)

        class _R:
            stopped: list[str] = []
            skipped: list[str] = []
            failed: list[str] = []
            all_clean = True

        return _R()

    def fake_harbor(*args: Any, **kwargs: Any) -> StopOutcome:
        log["harbor"].append(kwargs)
        return StopOutcome.NOT_RUNNING

    def fake_hermes(provider: str, **kwargs: Any) -> HermesStatus:
        log["hermes"].append(provider)
        return HermesStatus.OK

    deps = Dependencies(
        config_dir=config_dir,
        adapter_factory=_ExplodingAdapter,
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


def _unit() -> ManagedUnitConfig:
    return ManagedUnitConfig(unit_name="vllm-tp", default_port=8003)


def test_missing_local_path_refuses(tmp_path: Path) -> None:
    """The ornith-35b-refusal scenario: preset points at deleted weights."""
    gone = tmp_path / "models" / "Ornith-1.0-35B-FP8-refusal-v6"
    _write_preset(tmp_path / "presets", "ornith-refusal", str(gone))
    deps, log = _deps(tmp_path / "presets")

    with pytest.raises(MissingModelPathError) as excinfo:
        start_vllm_tp("ornith-refusal", managed_unit=_unit(), deps=deps)

    assert str(gone) in str(excinfo.value)
    assert "--force" in str(excinfo.value)
    # Nothing was touched: no ollama stop, no container stop, no restart.
    assert log == {"preflight": [], "harbor": [], "hermes": []}


def test_present_local_path_passes(tmp_path: Path) -> None:
    present = tmp_path / "models" / "Qwen__Qwen3.6-35B-A3B-FP8"
    present.mkdir(parents=True)
    _write_preset(tmp_path / "presets", "qwen-local", str(present))
    deps, log = _deps(tmp_path / "presets")

    with pytest.raises(AssertionError, match="adapter must not run"):
        start_vllm_tp("qwen-local", managed_unit=_unit(), deps=deps)
    # Reaching the adapter at all proves the path check let it through.
    assert log["preflight"] == [FleetRole.TP]


def test_hf_repo_id_fails_open(tmp_path: Path) -> None:
    """``org/name`` is not a filesystem path — never refuse on it."""
    _write_preset(tmp_path / "presets", "qwen-hf", "Qwen/Qwen3.6-27B-FP8")
    deps, log = _deps(tmp_path / "presets")

    with pytest.raises(AssertionError, match="adapter must not run"):
        start_vllm_tp("qwen-hf", managed_unit=_unit(), deps=deps)
    assert log["preflight"] == [FleetRole.TP]


def test_force_bypasses_the_refusal(tmp_path: Path) -> None:
    gone = tmp_path / "models" / "gone"
    _write_preset(tmp_path / "presets", "broken", str(gone))
    deps, _ = _deps(tmp_path / "presets")

    with pytest.raises(AssertionError, match="adapter must not run"):
        start_vllm_tp(
            "broken",
            managed_unit=_unit(),
            options=OrchestratorOptions(force=True),
            deps=deps,
        )


def test_dry_run_still_refuses(tmp_path: Path) -> None:
    """A dry-run that reports 'would restart' on missing weights is a lie."""
    gone = tmp_path / "models" / "gone"
    _write_preset(tmp_path / "presets", "broken", str(gone))
    deps, _ = _deps(tmp_path / "presets")

    with pytest.raises(MissingModelPathError):
        start_vllm_tp(
            "broken",
            managed_unit=_unit(),
            options=OrchestratorOptions(dry_run=True),
            deps=deps,
        )


def test_tilde_path_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    present = tmp_path / "weights"
    present.mkdir()
    _write_preset(tmp_path / "presets", "tilde-ok", "~/weights")
    deps, log = _deps(tmp_path / "presets")

    with pytest.raises(AssertionError, match="adapter must not run"):
        start_vllm_tp("tilde-ok", managed_unit=_unit(), deps=deps)
    assert log["preflight"] == [FleetRole.TP]


def test_tilde_path_missing_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_preset(tmp_path / "presets", "tilde-gone", "~/no-such-weights")
    deps, _ = _deps(tmp_path / "presets")

    with pytest.raises(MissingModelPathError, match="no-such-weights"):
        start_vllm_tp("tilde-gone", managed_unit=_unit(), deps=deps)
