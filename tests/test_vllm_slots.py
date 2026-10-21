"""Placing a vLLM preset: one card for tp=1, the second engine when the first is busy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llmctl.config import ManagedUnitConfig
from llmctl.services.lanes import Lane, SwapState, build_catalog, resolve
from llmctl.services.vllm_slots import (
    cuda_devices,
    engine_cards,
    engine_config,
    engine_lane,
    parse_cards,
    place,
)

ENGINE2 = ManagedUnitConfig(unit_name="vllm-engine2", default_port=8010, user_scope=True)
FLASH = Lane(
    name="flash",
    served_model="flash-model",
    upstream="http://127.0.0.1:8007",
    start_cmd=("/bin/flash-start", "--take-gpus"),
    unit="flash.service",
    gpus=frozenset({0}),
)
PRESETS = {
    "small": SimpleNamespace(served_name="small-model", tensor_parallel=1),
    "big": SimpleNamespace(served_name="big-model", tensor_parallel=2),
}


def _state(
    *,
    lanes_up=(),
    vllm_cards=None,
    vllm_active=False,
    free=None,
    engine2=ENGINE2,
    engine_lanes=(),
) -> SwapState:
    return SwapState(
        catalog=build_catalog(PRESETS, [FLASH], vllm_port=8003),
        serving={"flash"} if "flash.service" in lanes_up else set(),
        active_units=set(lanes_up),
        vllm_unit="vllm-engine",
        vllm_active=vllm_active,
        vllm_cards=vllm_cards,
        free_mib=free or {0: 2000, 1: 30000},
        card_count=2,
        engine2=engine2,
        engine_lanes=list(engine_lanes),
    )


def test_parse_cards():
    assert parse_cards("1") == frozenset({1})
    assert parse_cards("0,1") == frozenset({0, 1})
    assert parse_cards("") is None
    assert parse_cards("x") is None


def test_engine_cards_reads_env(tmp_path: Path):
    env = tmp_path / "e.env"
    env.write_text("PATH=/bin\nCUDA_VISIBLE_DEVICES=1\n")
    unit = ManagedUnitConfig(unit_name="e", env_file_path=env)
    assert engine_cards(unit) == frozenset({1})
    missing = ManagedUnitConfig(unit_name="e", env_file_path=tmp_path / "none.env")
    assert engine_cards(missing) is None


def test_two_card_preset_stays_unplaced():
    state = _state()
    big = resolve("big", state.catalog)
    assert place(state, big, big.tp) is big
    assert big.gpus is None


def test_one_card_preset_takes_the_free_card_and_keeps_the_lane():
    state = _state(lanes_up={"flash.service"})
    small = place(state, resolve("small", state.catalog), 1)
    assert small.cards == frozenset({1})
    assert small.engine_unit == "vllm-engine"
    assert state.lanes_in_the_way(small) == []
    assert cuda_devices(small) == "1"
    lines = state.preview(small).lines()
    assert "flash" not in lines[0]
    assert lines[1] == "starts: vllm-engine on preset small on GPU 1"


def test_main_engine_on_the_other_card_sends_it_to_engine2():
    state = _state(vllm_active=True, vllm_cards=frozenset({0}), free={0: 1000, 1: 30000})
    small = place(state, resolve("small", state.catalog), 1)
    assert small.cards == frozenset({1})
    assert small.engine_unit == "vllm-engine2.service"
    assert not state.vllm_in_the_way(small)  # vllm-engine keeps serving
    settings = SimpleNamespace(
        managed_units=SimpleNamespace(vllm_engine2=ENGINE2, vllm_engine="main")
    )
    assert engine_config(settings, small) is ENGINE2
    assert "vllm-engine2.service on preset small on GPU 1" in state.preview(small).lines()[1]


def test_without_engine2_the_main_engine_is_restarted():
    state = _state(vllm_active=True, vllm_cards=frozenset({0}), engine2=None)
    small = place(state, resolve("small", state.catalog), 1)
    assert small.engine_unit == "vllm-engine"
    assert state.vllm_in_the_way(small)
    assert "restarted on preset small" in state.preview(small).lines()[0]


def test_two_card_preset_stops_a_running_engine2():
    running = engine_lane(ENGINE2, "small-model")
    running = Lane(**{**running.__dict__, "gpus": frozenset({1})})
    state = _state(engine_lanes=[running])
    big = resolve("big", state.catalog)
    assert [lane.unit for lane in state.lanes_in_the_way(big)] == ["vllm-engine2.service"]
    # ...but not when the swap restarts engine2 itself.
    small = place(state, resolve("small", state.catalog), 1)
    assert small.engine_unit == "vllm-engine"  # main engine is down: use it
    assert small.cards == frozenset({0})  # card 1 is held by engine2
    assert state.lanes_in_the_way(small) == []


def test_lane_swap_keeps_vllm_on_the_other_card():
    state = _state(vllm_active=True, vllm_cards=frozenset({1}))
    flash = resolve("flash", state.catalog)
    assert not state.vllm_in_the_way(flash)
    assert "vllm-engine" not in state.preview(flash).lines()[0]
    state.vllm_cards = frozenset({0})
    assert state.vllm_in_the_way(flash)


def test_engine2_lane_restarts_its_unit():
    lane = engine_lane(ENGINE2, "small-model")
    assert lane.unit == "vllm-engine2.service"
    assert lane.start_cmd == ("systemctl", "--user", "start", "vllm-engine2.service")
    assert lane.upstream == "http://127.0.0.1:8010"


def test_both_cards_held_prefers_engine2_over_restarting_the_main_engine():
    # flash on GPU 0, the main engine on GPU 1 with MORE free VRAM than GPU 0:
    # the new preset joins on engine2 (replacing flash) instead of evicting vLLM.
    state = _state(
        lanes_up={"flash.service"},
        vllm_active=True,
        vllm_cards=frozenset({1}),
        free={0: 800, 1: 3000},
    )
    small = place(state, resolve("small", state.catalog), 1)
    assert small.cards == frozenset({0})
    assert small.engine_unit == "vllm-engine2.service"
    assert [lane.name for lane in state.lanes_in_the_way(small)] == ["flash"]
    assert not state.vllm_in_the_way(small)


def test_user_runner_puts_every_call_in_user_scope():
    import subprocess

    from llmctl.integrations.systemctl import UserSystemctlRunner

    seen: list[list[str]] = []

    def runner(argv, **_kw):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "active\n", "")

    sc = UserSystemctlRunner(runner)
    sc.restart("vllm-engine2")
    assert sc.is_active("vllm-engine2")
    assert all("--user" in argv and "sudo" not in argv for argv in seen)
