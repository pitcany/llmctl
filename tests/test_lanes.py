"""The swap list: router.yaml lanes + vLLM presets, state, preview, load times."""

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
    preview_swap,
    record_load,
    resolve,
    serving_keys,
)

ROUTER_YAML = """
switch:
  health_timeout_s: 300
backends:
  - name: qwen38
    served_model: qwen3.8-27b
    profile_env: vllm-tp.env.qwen38
    upstream: http://127.0.0.1:8003
  - name: flash
    served_model: qwen3.8-flash-next-freetoken
    upstream: http://127.0.0.1:1919/
    start_cmd: ["/bin/qwen38-ft-start", "--take-gpus"]
    health_timeout_s: 900
    aliases: ["flash-next"]
  - name: glm53
    served_model: glm-5.3-flash
    upstream: http://127.0.0.1:8006
    start_cmd: ["/bin/glm53-start"]
"""


def _units() -> ManagedUnitsConfig:
    return ManagedUnitsConfig(
        units={
            "flash-next-ft": ManagedUnitConfig(
                unit_name="qwen3.8-flash-next-ft", default_port=1919, runtime="freetoken"
            )
        }
    )


def _lanes(tmp_path: Path) -> list[Lane]:
    path = tmp_path / "router.yaml"
    path.write_text(ROUTER_YAML)
    return load_lanes(path, _units())


def _catalog(tmp_path: Path):
    presets = {"qwen3.8-27b": SimpleNamespace(served_name="qwen3.8-27b")}
    return build_catalog(presets, _lanes(tmp_path), vllm_port=8003)


def test_only_command_launched_backends_are_lanes(tmp_path):
    lanes = _lanes(tmp_path)
    assert [lane.name for lane in lanes] == ["flash", "glm53"]
    flash, glm = lanes
    assert flash.upstream == "http://127.0.0.1:1919"
    assert flash.start_cmd == ("/bin/qwen38-ft-start", "--take-gpus")
    assert flash.health_timeout_s == 900
    # Unit and runtime come from llmctl's managed_units, matched by port.
    assert flash.unit == "qwen3.8-flash-next-ft.service"
    assert flash.runtime == "freetoken"
    # No managed unit on :8006 -> unknown unit, router's default timeout.
    assert glm.unit is None
    assert glm.health_timeout_s == 300


def test_missing_router_config_means_no_lanes(tmp_path):
    assert load_lanes(tmp_path / "absent.yaml") == []


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

FOOTPRINT_YAML = """
backends:
  - name: flash
    served_model: ft
    upstream: http://127.0.0.1:1919
    start_cmd: ["/bin/ft-start"]
    gpus: [0]
  - name: gpu1
    served_model: coder
    upstream: http://127.0.0.1:11435
    start_cmd: ["/bin/gpu1-start"]
    gpus: [1]
  - name: pair
    served_model: big
    upstream: http://127.0.0.1:8006
    start_cmd: ["/bin/pair-start"]
    gpus: [0, 1]
  - name: undeclared
    served_model: legacy
    upstream: http://127.0.0.1:8080
    start_cmd: ["/bin/legacy-start"]
  - name: malformed
    served_model: odd
    upstream: http://127.0.0.1:8081
    start_cmd: ["/bin/odd-start"]
    gpus: ["0"]
"""

_UNIT_OF = {
    "flash": "ft.service",
    "gpu1": "gpu1.service",
    "pair": "pair.service",
    "undeclared": "legacy.service",
    "malformed": "odd.service",
}


def _footprint_state(tmp_path: Path, active: set[str]) -> SwapState:
    path = tmp_path / "router.yaml"
    path.write_text(FOOTPRINT_YAML)
    lanes = [Lane(**{**lane.__dict__, "unit": _UNIT_OF[lane.name]}) for lane in load_lanes(path)]
    presets = {"qwen3.8-27b": SimpleNamespace(served_name="qwen3.8-27b")}
    return SwapState(
        catalog=build_catalog(presets, lanes, vllm_port=8003),
        serving=set(),
        active_units={_UNIT_OF[name] for name in active},
        vllm_unit="vllm-tp",
        vllm_active=False,
    )


def test_gpus_parse_from_router_yaml_and_fail_closed(tmp_path):
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


ROUTER_YAML_SELF_DESCRIBED = """
backends:
  - name: glm53
    served_model: glm-5.3-flash
    upstream: http://127.0.0.1:8006
    start_cmd: ["/bin/glm53-start"]
    unit: glm-5.3-flash
    runtime: llama_cpp
"""


def test_lane_takes_its_unit_from_router_yaml_when_settings_has_no_row(tmp_path):
    path = tmp_path / "router.yaml"
    path.write_text(ROUTER_YAML_SELF_DESCRIBED)
    lane = load_lanes(path, _units())[0]
    assert lane.unit == "glm-5.3-flash.service"  # .service suffix added
    assert lane.runtime == "llama_cpp"


def test_a_settings_row_still_wins_over_router_yaml(tmp_path):
    """The managed_units row is the override; router.yaml is the fallback."""
    path = tmp_path / "router.yaml"
    path.write_text(
        ROUTER_YAML_SELF_DESCRIBED.replace("8006", "1919").replace(
            "unit: glm-5.3-flash", "unit: wrong-unit"
        )
    )
    lane = load_lanes(path, _units())[0]
    assert lane.unit == "qwen3.8-flash-next-ft.service"
    assert lane.runtime == "freetoken"


def test_units_from_lanes_skips_covered_ports_and_unitless_lanes(tmp_path):
    """Derive a row only for a lane that names a unit on a free port."""
    from llmctl.services.lanes import units_from_lanes

    self_described = tmp_path / "self.yaml"
    self_described.write_text(ROUTER_YAML_SELF_DESCRIBED)
    lanes = _lanes(tmp_path) + load_lanes(self_described, _units())

    derived = units_from_lanes(lanes, _units())

    # flash sits on 1919, which a managed_units row already owns; the glm53
    # entry from ROUTER_YAML names no unit, so only the self-described one is.
    assert set(derived) == {"glm53"}
    assert derived["glm53"].unit_name == "glm-5.3-flash"
    assert derived["glm53"].default_port == 8006
    assert derived["glm53"].runtime == "llama_cpp"
