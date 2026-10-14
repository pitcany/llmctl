"""`llmctl run` front end: list, preview, lanes, and never stopping on a typo."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmctl import cli, cli_swap
from llmctl.adapters.vllm_systemd import ManagedRestartResult
from llmctl.integrations.vllm_env import VLLMLaunchSpec
from llmctl.services.lane_swap import LaneSwapResult
from llmctl.services.lanes import Lane, SwapState, build_catalog
from llmctl.services.vllm_orchestrator import OrchestratorResult

FT = Lane(
    name="flash",
    served_model="ft-model",
    upstream="http://127.0.0.1:1919",
    start_cmd=("ft-start", "--take-gpus"),
    unit="ft.service",
    runtime="freetoken",
)


class _Spec:
    served_name = "qwen3.8-27b"


def _state(serving: set[str], active: set[str]) -> SwapState:
    return SwapState(
        catalog=build_catalog({"qwen3.8-27b": _Spec()}, [FT], vllm_port=8003),
        serving=serving,
        active_units=active,
        vllm_unit="vllm-tp",
        vllm_active="qwen3.8-27b" in serving,
    )


@pytest.fixture
def calls(monkeypatch):
    seen: dict[str, list] = {"lane": [], "vllm": [], "stop": [], "restore": []}
    monkeypatch.setattr(cli, "_confirm_state_change", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_swap,
        "swap_to_lane",
        lambda lane, **k: seen["lane"].append(lane.name) or LaneSwapResult(ok=True, seconds=99),
    )

    def fake_stop(state, console):
        seen["stop"].append(True)
        return state.active_lanes()

    monkeypatch.setattr(cli_swap, "stop_lanes_for_vllm", fake_stop)
    monkeypatch.setattr(cli_swap, "restore_lanes", lambda lanes, c: seen["restore"].append(lanes))
    return seen


def _run(monkeypatch, state, *args, vllm_ok=True):
    monkeypatch.setattr(cli, "gather_state", lambda settings: state)
    result = OrchestratorResult(
        spec=VLLMLaunchSpec(model="m", served_name="qwen3.8-27b", port=8003),
        restart=ManagedRestartResult(
            env_path=Path("/tmp/x.env"),
            env_body="",
            ready=vllm_ok,
            error=None if vllm_ok else "boom",
        ),
    )
    monkeypatch.setattr(cli, "start_vllm_tp", lambda *a, **k: result)
    return CliRunner().invoke(cli.app, ["run", *args])


def test_list_shows_presets_and_lanes(monkeypatch, calls):
    res = _run(monkeypatch, _state({"flash"}, {"ft.service"}), "--list")
    assert res.exit_code == 0, res.output
    assert "qwen3.8-27b" in res.output and "FreeToken" in res.output
    assert "serving" in res.output
    assert calls["lane"] == [] and calls["stop"] == []


def test_lane_swap_previews_then_swaps(monkeypatch, calls):
    res = _run(monkeypatch, _state({"qwen3.8-27b"}, set()), "flash")
    assert res.exit_code == 0, res.output
    assert "stops:" in res.output and "vllm-tp (qwen3.8-27b)" in res.output
    assert "Ready" in res.output
    assert calls["lane"] == ["flash"]


def test_lane_dry_run_changes_nothing(monkeypatch, calls):
    res = _run(monkeypatch, _state({"qwen3.8-27b"}, set()), "flash", "--dry-run")
    assert res.exit_code == 0, res.output
    assert "starts:" in res.output
    assert calls["lane"] == []


def test_already_serving_lane_is_a_no_op(monkeypatch, calls):
    res = _run(monkeypatch, _state({"flash"}, {"ft.service"}), "flash")
    assert res.exit_code == 0, res.output
    assert "already serving" in res.output
    assert calls["lane"] == []


def test_vllm_tuning_flags_are_rejected_for_lanes(monkeypatch, calls):
    res = _run(monkeypatch, _state(set(), set()), "flash", "--tq")
    assert res.exit_code == 2
    assert calls["lane"] == []


def test_vllm_swap_stops_lanes_first_and_restores_on_failure(monkeypatch, calls):
    res = _run(monkeypatch, _state({"flash"}, {"ft.service"}), "qwen3.8-27b", vllm_ok=False)
    assert res.exit_code == 1, res.output
    assert calls["stop"] == [True]
    assert [lane.name for lane in calls["restore"][0]] == ["flash"]


def test_a_typo_never_stops_a_lane(monkeypatch, calls):
    """An unlisted name goes to the vLLM path untouched; lanes keep serving."""
    res = _run(monkeypatch, _state({"flash"}, {"ft.service"}), "qwen3.8-27", vllm_ok=True)
    assert calls["stop"] == []
    assert res.exit_code == 0, res.output


def test_vllm_swap_reports_paused_services(monkeypatch, calls):
    """Ollama coming back (or the command to bring it back) is printed."""
    from llmctl.services import vllm_orchestrator

    real = vllm_orchestrator.OrchestratorResult

    def with_companions(*a, **k):
        result = real(
            spec=VLLMLaunchSpec(model="m", served_name="qwen3.8-27b", port=8003),
            restart=ManagedRestartResult(env_path=Path("/tmp/x.env"), env_body="", ready=True),
        )
        result.companions_restarted = ["ollama"]
        result.companions_failed = ["harbor.ollama"]
        result.start_hints = {"harbor.ollama": "docker start harbor.ollama"}
        return result

    monkeypatch.setattr(cli, "gather_state", lambda settings: _state(set(), set()))
    monkeypatch.setattr(cli, "start_vllm_tp", with_companions)
    res = CliRunner().invoke(cli.app, ["run", "qwen3.8-27b"])
    assert res.exit_code == 0, res.output
    assert "started again: ollama" in res.output
    assert "docker start harbor.ollama" in res.output
