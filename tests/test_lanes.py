"""The swap list: plugin lanes + vLLM presets, state, preview, load times."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig
from llmctl.services.lanes import (
    Lane,
    SwapState,
    build_catalog,
    format_duration,
    format_gpus,
    gpus_overlap,
    last_load_s,
    load_lanes,
    parse_gpus,
    preview_swap,
    record_load,
    resolve,
    serving_keys,
)


def _units() -> ManagedUnitsConfig:
    return ManagedUnitsConfig(
        units={
            "flash-next-ft": ManagedUnitConfig(
                unit_name="qwen3.8-flash-next-ft", default_port=1919, runtime="freetoken"
            )
        }
    )


FLASH = Lane(
    name="flash",
    served_model="qwen3.8-flash-next-freetoken",
    upstream="http://127.0.0.1:1919",
    start_cmd=("/bin/qwen38-ft-start", "--take-gpus"),
    health_timeout_s=900,
    aliases=("flash-next",),
    unit="qwen3.8-flash-next-ft.service",
    runtime="freetoken",
)
GLM53 = Lane(
    name="glm53",
    served_model="glm-5.3-flash",
    upstream="http://127.0.0.1:8006",
    start_cmd=("/bin/glm53-start",),
)
GLM53_SELF_DESCRIBED = Lane(
    name="glm53",
    served_model="glm-5.3-flash",
    upstream="http://127.0.0.1:8006",
    start_cmd=("/bin/glm53-start",),
    unit="glm-5.3-flash.service",
    runtime="llama_cpp",
)


def _lanes(tmp_path: Path | None = None) -> list[Lane]:
    return [FLASH, GLM53]


def _catalog(tmp_path: Path):
    presets = {"qwen3.8-27b": SimpleNamespace(served_name="qwen3.8-27b")}
    return build_catalog(presets, _lanes(tmp_path), vllm_port=8003)


def test_lanes_come_from_the_registered_sources_in_order(tmp_path):
    """A source that raises is skipped; the others still list."""

    def broken(units):
        raise OSError("config unreadable")

    assert load_lanes(_units(), sources=[lambda u: [FLASH], broken, lambda u: [GLM53]]) == [
        FLASH,
        GLM53,
    ]


def test_no_plugins_means_no_lanes():
    assert load_lanes(_units(), sources=[]) == []


def test_catalog_lists_presets_then_lanes_and_resolves_aliases(tmp_path):
    catalog = _catalog(tmp_path)
    assert [(r.key, r.kind, r.engine) for r in catalog] == [
        ("qwen3.8-27b", "vllm", "vLLM"),
        ("flash", "lane", "FreeToken"),
        ("glm53", "lane", "?"),
    ]
    assert resolve("flash-next", catalog).key == "flash"
    assert resolve("QWEN3.8-FLASH-NEXT-FREETOKEN", catalog).key == "flash"
    assert resolve("nope", catalog) is None


def test_serving_probes_each_upstream_once(tmp_path):
    catalog = _catalog(tmp_path)
    calls: list[str] = []

    def probe(url: str):
        calls.append(url)
        return ["qwen3.8-flash-next-freetoken"] if url.endswith(":1919") else None

    assert serving_keys(catalog, probe) == {"flash"}
    assert sorted(calls) == sorted(set(calls))


def test_preview_lane_lists_everything_it_stops(tmp_path):
    catalog = _catalog(tmp_path)
    target = resolve("flash", catalog)
    preview = preview_swap(
        target,
        catalog,
        serving={"qwen3.8-27b"},
        active_units=set(),
        vllm_unit="vllm-tp",
        vllm_active=True,
        load_times=tmp_path / "t.json",
    )
    text = "\n".join(preview.lines())
    assert "vllm-tp (qwen3.8-27b)" in text
    assert "Ollama models on its GPUs" in text
    assert "qwen38-ft-start --take-gpus" in text
    assert "never measured" in text


def test_preview_vllm_stops_active_lanes(tmp_path):
    catalog = _catalog(tmp_path)
    preview = preview_swap(
        resolve("qwen3.8-27b", catalog),
        catalog,
        serving={"flash"},
        active_units={"qwen3.8-flash-next-ft.service"},
        vllm_unit="vllm-tp",
        vllm_active=False,
        load_times=tmp_path / "t.json",
    )
    assert any("qwen3.8-flash-next-ft.service" in s for s in preview.stops)
    assert preview.starts == "vllm-tp on preset qwen3.8-27b"


def test_preview_already_serving_lane_is_a_no_op(tmp_path):
    catalog = _catalog(tmp_path)
    preview = preview_swap(
        resolve("flash", catalog),
        catalog,
        serving={"flash"},
        active_units={"qwen3.8-flash-next-ft.service"},
        vllm_unit="vllm-tp",
        vllm_active=False,
        load_times=tmp_path / "t.json",
    )
    assert preview.already_serving
    assert preview.lines() == ["qwen3.8-flash-next-freetoken (FreeToken) is already serving."]


def test_load_times_round_trip(tmp_path):
    path = tmp_path / "state" / "load-times.json"
    assert last_load_s("flash", path) is None
    record_load("flash", 130.44, path)
    assert last_load_s("flash", path) == 130.4
    assert format_duration(130.4) == "2m 10s"
    assert format_duration(42) == "42s"
    assert format_duration(None) == "never measured"


# -- GPU footprints -------------------------------------------------------------

_UNIT_OF = {
    "flash": "ft.service",
    "gpu1": "gpu1.service",
    "pair": "pair.service",
    "undeclared": "legacy.service",
    "malformed": "odd.service",
}


def _lane(name: str, port: int, gpus) -> Lane:
    return Lane(
        name=name,
        served_model=name,
        upstream=f"http://127.0.0.1:{port}",
        start_cmd=(f"/bin/{name}-start",),
        unit=_UNIT_OF[name],
        gpus=parse_gpus(gpus),
    )


def _footprint_state(tmp_path: Path, active: set[str]) -> SwapState:
    lanes = [
        _lane("flash", 1919, [0]),
        _lane("gpu1", 11435, [1]),
        _lane("pair", 8006, [0, 1]),
        _lane("undeclared", 8080, None),
        _lane("malformed", 8081, ["0"]),
    ]
    presets = {"qwen3.8-27b": SimpleNamespace(served_name="qwen3.8-27b")}
    return SwapState(
        catalog=build_catalog(presets, lanes, vllm_port=8003),
        serving=set(),
        active_units={_UNIT_OF[name] for name in active},
        vllm_unit="vllm-tp",
        vllm_active=False,
    )


def test_gpus_parse_and_fail_closed(tmp_path):
    state = _footprint_state(tmp_path, set())
    gpus = {row.key: row.gpus for row in state.catalog}
    assert gpus == {
        "qwen3.8-27b": None,  # vLLM preset: TP=2, every card
        "flash": frozenset({0}),
        "gpu1": frozenset({1}),
        "pair": frozenset({0, 1}),
        "undeclared": None,  # no gpus: -> every card
        "malformed": None,  # strings are not card indices -> every card
    }
    assert [format_gpus(g) for g in (frozenset({0}), frozenset({1, 0}), None)] == [
        "0",
        "0+1",
        "all",
    ]


def test_gpus_overlap():
    zero, one, both = frozenset({0}), frozenset({1}), frozenset({0, 1})
    assert not gpus_overlap(zero, one)
    assert gpus_overlap(zero, both)
    assert gpus_overlap(one, both)
    assert gpus_overlap(zero, zero)
    assert gpus_overlap(None, one)
    assert gpus_overlap(zero, None)


def test_lane_on_other_card_is_not_in_the_way(tmp_path):
    state = _footprint_state(tmp_path, {"flash"})
    gpu1 = resolve("gpu1", state.catalog)
    assert state.lanes_in_the_way(gpu1) == []
    preview = state.preview(gpu1, load_times=tmp_path / "t.json")
    assert not any("ft.service" in stop for stop in preview.stops)


def test_overlapping_lanes_are_in_the_way(tmp_path):
    state = _footprint_state(tmp_path, {"flash", "gpu1", "undeclared"})

    def names(target):
        return sorted(lane.name for lane in state.lanes_in_the_way(target))

    # A GPU0 lane stops the other GPU0 user and the undeclared one, not GPU1's.
    assert names(resolve("flash", state.catalog)) == ["undeclared"]
    # A two-card lane and a vLLM preset stop everything.
    assert names(resolve("pair", state.catalog)) == ["flash", "gpu1", "undeclared"]
    assert names(resolve("qwen3.8-27b", state.catalog)) == ["flash", "gpu1", "undeclared"]
    preview = state.preview(resolve("pair", state.catalog), load_times=tmp_path / "t.json")
    assert sum("service" in stop for stop in preview.stops) == 3


def test_units_from_lanes_skips_covered_ports_and_unitless_lanes(tmp_path):
    """Derive a row only for a lane that names a unit on a free port."""
    from llmctl.services.lanes import units_from_lanes

    lanes = _lanes() + [GLM53_SELF_DESCRIBED]

    derived = units_from_lanes(lanes, _units())

    # flash sits on 1919, which a managed_units row already owns; the first
    # glm53 names no unit, so only the self-described one is.
    assert set(derived) == {"glm53"}
    assert derived["glm53"].unit_name == "glm-5.3-flash"
    assert derived["glm53"].default_port == 8006
    assert derived["glm53"].runtime == "llama_cpp"
