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
    # Never reach the live llm-router from the suite (2026-10-20: a run without this
    # posted 20 real /admin/switch calls). Router-path tests re-enable it explicitly.
    monkeypatch.setattr(cli_swap, "router_up", lambda url: False)
    monkeypatch.setattr(
        cli_swap,
        "swap_to_lane",
        lambda lane, **k: seen["lane"].append(lane.name) or LaneSwapResult(ok=True, seconds=99),
    )

    def fake_stop(state, console, target=None):
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


class _Rec:
    def __init__(self):
        self.out: list[str] = []

    def print(self, msg="", *a, **k):
        self.out.append(str(msg))


class _Sess:
    """Shaped like services.session_swap.GpuSession (no endpoint_url field)."""

    def __init__(self, name, port):
        self.name = name
        self.port = port
        self.gpu_ids = (0,)

    def label(self):
        return self.name


def _session_target(port=None):
    from llmctl.services.lanes import SwapTarget

    return SwapTarget(
        key="gemma", kind="session", model="gemma", engine="llama.cpp", upstream="", port=port
    )


def _session_state(sessions):
    st = _state(set(), set())
    st.gpu_sessions = sessions
    return st


def test_serving_lane_says_where_it_answers():
    st = _state({"flash"}, {"ft.service"})
    target = next(r for r in st.catalog if r.key == "flash")
    con = _Rec()
    assert cli_swap.print_preview(st, target, con) is False
    assert any(f"{target.upstream}/v1" in line for line in con.out)


def test_serving_session_port_beats_the_profile_port():
    st = _session_state([_Sess("gemma", 9000)])
    con = _Rec()
    assert cli_swap.print_preview(st, _session_target(port=8080), con) is False
    assert any("http://127.0.0.1:9000/v1" in line for line in con.out)


def test_serving_session_falls_back_to_port():
    st = _session_state([_Sess("gemma", None)])
    con = _Rec()
    assert cli_swap.print_preview(st, _session_target(port=8080), con) is False
    assert any("http://127.0.0.1:8080/v1" in line for line in con.out)


def test_serving_session_unknown_endpoint_prints_nothing_extra():
    st = _session_state([_Sess("gemma", None)])
    con = _Rec()
    assert cli_swap.print_preview(st, _session_target(), con) is False
    assert not any("answers at" in line for line in con.out)


def test_serving_real_gpu_session_uses_its_own_port():
    from llmctl.services.session_swap import GpuSession

    sess = GpuSession(id="abcdef123", name="gemma", runtime="llama_cpp", port=9001, gpu_ids=(0,))
    st = _session_state([sess])
    con = _Rec()
    assert cli_swap.print_preview(st, _session_target(), con) is False
    assert any("http://127.0.0.1:9001/v1" in line for line in con.out)


# --- one switch engine: llm-router does the swap when it is up (2026-10-20) ---------
from llmctl.services.lane_switcher import RouterRefused, RouterSwitch  # noqa: E402


def _router(monkeypatch, answer):
    """llm-router is up with a key; `answer` is a RouterSwitch, None (404) or an exception."""
    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(cli_swap, "router_up", lambda url: True)
    monkeypatch.setattr(cli_swap, "router_key", lambda env_file: "k")

    def fake_switch(model, *, url, key, force, timeout_s):
        seen.append((model, force))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(cli_swap, "switch_via_router", fake_switch)
    return seen


def test_lane_swap_goes_through_the_router(monkeypatch, calls):
    seen = _router(
        monkeypatch,
        RouterSwitch("flash", "ft-model", "http://127.0.0.1:1919", True, False, "qwen3.8-27b"),
    )
    res = _run(monkeypatch, _state({"qwen3.8-27b"}, set()), "flash")
    assert res.exit_code == 0, res.output
    assert seen == [("ft-model", True)]
    assert calls["lane"] == []  # llmctl did not swap on its own
    assert "Ready" in res.output and "via llm-router" in res.output


def test_vllm_preset_goes_through_the_router(monkeypatch, calls):
    seen = _router(
        monkeypatch,
        RouterSwitch("qwen38", "qwen3.8-27b", "http://127.0.0.1:8003", True, False, "ft-model"),
    )
    res = _run(monkeypatch, _state({"flash"}, {"ft.service"}), "qwen3.8-27b")
    assert res.exit_code == 0, res.output
    assert seen == [("qwen3.8-27b", True)]
    assert calls["stop"] == []  # no in-process stop of lanes either


def test_router_refusal_ends_the_command_without_a_direct_swap(monkeypatch, calls):
    _router(monkeypatch, RouterRefused("leased by mathlab", retry_after=120))
    res = _run(monkeypatch, _state({"qwen3.8-27b"}, set()), "flash")
    assert res.exit_code == 1
    assert "refused" in res.output and "--direct" in res.output
    assert calls["lane"] == []


def test_unknown_to_router_falls_back_to_the_direct_path(monkeypatch, calls):
    _router(monkeypatch, None)
    res = _run(monkeypatch, _state({"qwen3.8-27b"}, set()), "flash")
    assert res.exit_code == 0, res.output
    assert calls["lane"] == ["flash"]


def test_router_down_falls_back_to_the_direct_path(monkeypatch, calls):
    monkeypatch.setattr(cli_swap, "router_up", lambda url: False)
    monkeypatch.setattr(cli_swap, "router_key", lambda env_file: "k")
    res = _run(monkeypatch, _state({"qwen3.8-27b"}, set()), "flash")
    assert res.exit_code == 0, res.output
    assert calls["lane"] == ["flash"]


def test_direct_flag_skips_the_router(monkeypatch, calls):
    seen = _router(
        monkeypatch, RouterSwitch("flash", "ft-model", "http://127.0.0.1:1919", True, False, None)
    )
    res = _run(monkeypatch, _state({"qwen3.8-27b"}, set()), "flash", "--direct")
    assert res.exit_code == 0, res.output
    assert seen == [] and calls["lane"] == ["flash"]
