"""Starting a lane beside what runs when its measured VRAM fits."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llmctl.services.lanes import (
    Lane,
    SwapState,
    build_catalog,
    record_load,
    resolve,
    share_start_cmd,
)
from llmctl.services.vram_fit import (
    HEADROOM_MIB,
    fits,
    last_vram_mib,
    pids_vram_mib,
    record_vram,
)


def _lane(name: str, gpus: frozenset[int], unit: str, take: bool = True) -> Lane:
    return Lane(
        name=name,
        served_model=f"{name}-model",
        upstream="http://127.0.0.1:1",
        start_cmd=(f"/bin/{name}-start", *(("--take-gpus",) if take else ())),
        unit=unit,
        gpus=gpus,
    )


BIG = _lane("big", frozenset({0}), "big.service")
SMALL = _lane("small", frozenset({0}), "small.service")
OTHER = _lane("other", frozenset({0}), "other.service")
NOFLAG = _lane("noflag", frozenset({0}), "noflag.service", take=False)


def _state(free, needs, *, active=("big.service",), conflicts=None, vllm=False) -> SwapState:
    return SwapState(
        catalog=build_catalog({}, [BIG, SMALL, OTHER, NOFLAG], vllm_port=8003),
        serving={"big"},
        active_units=set(active),
        vllm_unit="vllm-tp",
        vllm_active=vllm,
        free_mib=free,
        vram_need=lambda key: needs.get(key),
        conflicts_of=lambda unit: (conflicts or {}).get(unit, set()),
    )


def test_fits_needs_headroom_and_fails_closed():
    assert fits({0: 8000}, {0: 8000 + HEADROOM_MIB})
    assert not fits({0: 8000}, {0: 8000 + HEADROOM_MIB - 1})
    assert not fits({0: 8000, 1: 10}, {0: 20000})  # card 1 unreadable
    assert not fits(None, {0: 20000})  # never measured
    assert not fits({0: 1}, None)  # NVML down


def test_record_vram_keeps_load_time_and_back(tmp_path: Path):
    path = tmp_path / "load-times.json"
    record_load("small", 12.0, path)
    record_vram("small", {0: 8000}, path)
    record_load("small", 14.0, path)  # a later load keeps the measure
    assert last_vram_mib("small", path) == {0: 8000}
    assert last_vram_mib("never", path) is None


def test_pids_vram_sums_per_card():
    gpus = [
        SimpleNamespace(index=0, processes=[{"pid": 1, "used_memory_mb": 100}, {"pid": 9}]),
        SimpleNamespace(index=1, processes=[{"pid": 2, "used_memory_mb": 50}]),
    ]
    assert pids_vram_mib({1, 2}, gpus) == {0: 100, 1: 50}
    assert pids_vram_mib({7}, gpus) == {}


def test_lane_that_fits_keeps_what_runs():
    state = _state({0: 12000}, {"small": {0: 8000}}, vllm=True)
    small = resolve("small", state.catalog)
    fit = state.fits_alongside(small)
    assert fit is not None
    assert state.lanes_in_the_way(small, fit) == []
    assert state.sessions_in_the_way(small, fit) == []
    lines = state.preview(small).lines()
    assert lines[0].startswith("fits beside what is running: needs 7.8 GiB on GPU 0")
    assert lines[1] == "stops:  nothing; everything running stays up"
    assert lines[2].endswith("/bin/small-start --share-gpus")


def test_lane_that_does_not_fit_stops_overlap_as_before():
    state = _state({0: 5000}, {"small": {0: 8000}})
    small = resolve("small", state.catalog)
    assert state.fits_alongside(small) is None
    assert [lane.name for lane in state.lanes_in_the_way(small)] == ["big"]
    assert "big.service" in state.preview(small).lines()[0]


def test_never_measured_or_no_take_flag_fails_closed():
    state = _state({0: 30000}, {"noflag": {0: 1}})
    assert state.fits_alongside(resolve("small", state.catalog)) is None
    assert state.fits_alongside(resolve("noflag", state.catalog)) is None


def test_need_on_a_card_outside_the_footprint_fails_closed():
    state = _state({0: 30000, 1: 30000}, {"small": {1: 8000}})
    assert state.fits_alongside(resolve("small", state.catalog)) is None


def test_conflicting_unit_is_still_stopped():
    state = _state(
        {0: 30000},
        {"small": {0: 8000}},
        active=("big.service", "other.service"),
        conflicts={"small.service": {"other.service"}},
    )
    state.serving.add("other")
    small = resolve("small", state.catalog)
    fit = state.fits_alongside(small)
    assert fit is not None
    assert [lane.name for lane in state.lanes_in_the_way(small, fit)] == ["other"]
    assert (
        "other.service (other-model, its unit Conflicts= with small.service)"
        in (state.preview(small).lines()[1])
    )


def test_share_start_cmd():
    assert share_start_cmd(("/bin/x", "--take-gpus")) == ("/bin/x", "--share-gpus")
    assert share_start_cmd(("strata-uncensored", "start", "--take-gpus"))[-1] == "--share-gpus"
