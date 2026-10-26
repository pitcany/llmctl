"""Lanes behind llama-swap: no systemd unit, judged by /running, stopped by a command.

llama-swap (the router on macOS hosts) starts and stops its backends itself, so
its lanes have no unit. Its /v1/models lists every configured model, loaded or
not, so a lane reads serving only while /running reports it ready; and with no
unit to stop, the lane's ``stop_cmd`` is how ``llmctl stop`` unloads it.
"""

from __future__ import annotations

import io
import json

from llmctl.services.backends import probe_llama_swap_running
from llmctl.services.lanes import Lane, SwapState, SwapTarget, serving_keys
from llmctl.services.stop_serving import StopDeps, pick, serving_now, stop_items

ROUTER = "http://127.0.0.1:8099"


def _row(key: str, *, stop_cmd: tuple[str, ...] = ()) -> SwapTarget:
    lane = Lane(
        name=key,
        served_model=key,
        upstream=ROUTER,
        start_cmd=("llmctl-mac", "load", key),
        stop_cmd=stop_cmd,
        runtime="llama_swap",
    )
    return SwapTarget(
        key=key, kind="lane", model=key, engine="llama-swap", upstream=ROUTER, lane=lane
    )


QWEN = _row("qwen27b-m5", stop_cmd=("llmctl-mac", "unload", "qwen27b-m5"))
TINY = _row("qwen4b-m5")


def _configured(url: str) -> list[str]:
    return ["qwen27b-m5", "qwen4b-m5"]  # /v1/models: every lane in the config


def _running(url: str) -> list[str]:
    return ["qwen27b-m5"]  # /running: what is loaded and ready


def _state(serving: set[str], catalog: list[SwapTarget] | None = None) -> SwapState:
    return SwapState(
        catalog=catalog or [QWEN, TINY],
        serving=serving,
        active_units=set(),
        vllm_unit="vllm-tp.service",
        vllm_active=False,
    )


def test_a_lane_has_no_stop_cmd_by_default() -> None:
    assert TINY.lane is not None
    assert TINY.lane.stop_cmd == ()


def test_probe_llama_swap_running_lists_only_ready_models() -> None:
    seen: list[str] = []

    def http_get(url: str, timeout: float):
        seen.append(url)
        body = {
            "running": [
                {"model": "qwen27b-m5", "state": "ready"},
                {"model": "qwen4b-m5", "state": "starting"},
            ]
        }
        return io.BytesIO(json.dumps(body).encode())

    assert probe_llama_swap_running(ROUTER, http_get=http_get) == ["qwen27b-m5"]
    assert seen == [f"{ROUTER}/running"]


def test_an_idle_router_reads_as_nothing_running() -> None:
    def body(payload: object):
        return lambda url, timeout: io.BytesIO(json.dumps(payload).encode())

    assert probe_llama_swap_running(ROUTER, http_get=body({"running": None})) == []
    assert probe_llama_swap_running(ROUTER, http_get=body({"running": []})) == []
    assert probe_llama_swap_running(ROUTER, http_get=body({})) == []


def test_probe_llama_swap_running_is_none_when_unreadable() -> None:
    def refused(url: str, timeout: float):
        raise OSError("connection refused")

    def not_json(url: str, timeout: float):
        return io.BytesIO(b"<html>")

    assert probe_llama_swap_running(ROUTER, http_get=refused) is None
    assert probe_llama_swap_running(ROUTER, http_get=not_json) is None


def test_llama_swap_lanes_are_judged_by_what_is_running() -> None:
    assert serving_keys([QWEN, TINY], _configured) == {"qwen27b-m5", "qwen4b-m5"}
    loaded = {"llama_swap": _running}
    assert serving_keys([QWEN, TINY], _configured, loaded) == {"qwen27b-m5"}


def test_a_serving_lane_with_a_stop_cmd_is_listed_for_stop() -> None:
    items = serving_now(_state({"qwen27b-m5"}))
    assert [(i.kind, i.ref, i.cmd) for i in items] == [
        ("lane", "qwen27b-m5", ("llmctl-mac", "unload", "qwen27b-m5"))
    ]
    assert pick(items, "qwen27b-m5") == items


def test_a_unitless_lane_without_a_stop_cmd_is_not_listed() -> None:
    assert serving_now(_state({"qwen4b-m5"})) == []


def test_an_idle_lane_with_a_stop_cmd_is_not_listed() -> None:
    assert serving_now(_state(set())) == []


def test_stop_items_runs_the_lanes_stop_cmd() -> None:
    ran: list[tuple[str, ...]] = []

    def run_cmd(argv) -> bool:
        ran.append(tuple(argv))
        return True

    def never(ref: str) -> bool:
        raise AssertionError(f"systemd stop called for {ref}")

    deps = StopDeps(stop_vllm=never, stop_lane=never, stop_session=never, run_cmd=run_cmd)
    results = stop_items(serving_now(_state({"qwen27b-m5"})), deps)
    assert [ok for _, ok in results] == [True]
    assert ran == [("llmctl-mac", "unload", "qwen27b-m5")]


def test_a_failing_stop_cmd_is_reported_not_raised() -> None:
    def run_cmd(argv) -> bool:
        return False

    def never(ref: str) -> bool:
        raise AssertionError(ref)

    deps = StopDeps(stop_vllm=never, stop_lane=never, stop_session=never, run_cmd=run_cmd)
    results = stop_items(serving_now(_state({"qwen27b-m5"})), deps)
    assert [ok for _, ok in results] == [False]


# -- preview: the router decides what a start unloads ---------------------------


def _planned(plan) -> SwapTarget:
    lane = Lane(
        name="glm53-m5",
        served_model="glm53-m5",
        upstream=ROUTER,
        start_cmd=("llmctl-mac", "start", "glm53-m5"),
        runtime="llama_swap",
        plan=plan,
    )
    return SwapTarget(
        key="glm53-m5", kind="lane", model="glm53-m5", engine="llama-swap", upstream=ROUTER,
        lane=lane,
    )


def test_a_lanes_plan_says_what_its_start_unloads() -> None:
    target = _planned(lambda: ["qwen35b-m5 (exclusive group big)", "mimo9b-m5 (swap group a)"])
    lines = _state({"qwen35b-m5"}, [QWEN, TINY, target]).preview(target).lines()

    assert lines[0] == "stops:  qwen35b-m5 (exclusive group big), mimo9b-m5 (swap group a)"
    assert lines[1] == "starts: llmctl-mac start glm53-m5"  # no unit: the command itself


def test_an_empty_plan_says_nothing_stops_not_that_nothing_runs() -> None:
    target = _planned(lambda: [])
    lines = _state({"qwen35b-m5"}, [QWEN, target]).preview(target).lines()
    assert lines[0] == "stops:  nothing"


def test_a_failing_plan_is_shown_not_raised() -> None:
    def broken() -> list[str]:
        raise RuntimeError("router down")

    target = _planned(broken)
    lines = _state(set(), [target]).preview(target).lines()
    assert lines[0] == "stops:  unknown (router down); the start command checks again"


def test_the_tui_runner_gives_the_start_command_no_stdin(monkeypatch) -> None:
    """Its output is captured, so a prompt there would wait unseen."""
    import subprocess

    from llmctl.services import lane_swap

    seen: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert lane_swap._run_captured(["true"], 5.0) == 0
    assert seen["stdin"] is subprocess.DEVNULL


def test_two_loads_of_a_planned_lane_are_equal() -> None:
    """Each lane source call makes a new plan closure; the lane is still the same lane."""
    assert _planned(lambda: []).lane == _planned(lambda: []).lane
