"""The swap list: router.yaml lanes + vLLM presets, state, preview, load times."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llmctl.config import ManagedUnitConfig, ManagedUnitsConfig
from llmctl.services.lanes import (
    Lane,
    build_catalog,
    format_duration,
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
