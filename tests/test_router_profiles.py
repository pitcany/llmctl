"""``llmctl router-profiles``: router profile envs are checked against presets.

llm-router swaps vLLM by copying ``services/vllm-tp.env.<profile>``; llmctl
renders the same file from a preset. The preset is the source of truth, and a
sync must keep the history comments that live in the profile files.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from llmctl.services.router_profiles import (
    check_profiles,
    diff_env,
    merge_env,
    write_profile,
)
from llmctl.services.vllm_orchestrator import render_preset_env


def _write_preset(config_dir: Path, alias: str, served: str, max_len: int = 32768) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    body = f"""
    alias: {alias}
    served_name: {served}
    model_id: Org/{alias}
    quantization: fp8
    vllm_quantization_flag: fp8
    tensor_parallel_size: 2
    max_model_len: {max_len}
    max_num_seqs: 32
    kv_cache_dtype: fp8
    tool_parser: qwen3_xml
    reasoning_parser: null
    """
    (config_dir / f"{alias}.yaml").write_text(textwrap.dedent(body).strip() + "\n")


def _write_router(path: Path, services: Path, backends: list[tuple[str, str, str]]) -> None:
    lines = ["switch:", f"  services_dir: {services}", "backends:"]
    for name, served, profile in backends:
        lines += [f"  - name: {name}", f"    served_model: {served}", f"    profile_env: {profile}"]
    lines += [
        "  - name: lane",
        "    served_model: lane-model",
        "    start_cmd: [lane-start]",
        "    upstream: http://127.0.0.1:1919",
    ]
    path.write_text("\n".join(lines) + "\n")


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    presets, services = tmp_path / "presets", tmp_path / "services"
    router = tmp_path / "router.yaml"
    services.mkdir()
    _write_preset(presets, "m1", "served-1")
    _write_router(router, services, [("b1", "served-1", "vllm-tp.env.b1")])
    return presets, services, router


def _check(presets: Path, router: Path):
    return check_profiles(port=8003, config_dir=presets, router_config=router)


def test_matching_profile_is_in_sync(tmp_path: Path) -> None:
    presets, services, router = _setup(tmp_path)
    _, body = render_preset_env("m1", port=8003, config_dir=presets)
    (services / "vllm-tp.env.b1").write_text("# history\n" + body)

    [drift] = _check(presets, router)  # the start_cmd lane is not a profile backend

    assert drift.preset == "m1"
    assert drift.in_sync


def test_changed_value_is_reported_and_write_keeps_comments(tmp_path: Path) -> None:
    presets, services, router = _setup(tmp_path)
    _, body = render_preset_env("m1", port=8003, config_dir=presets)
    profile = services / "vllm-tp.env.b1"
    profile.write_text("# why 32768\n" + body)
    _write_preset(presets, "m1", "served-1", max_len=65536)  # the preset moves on

    [drift] = _check(presets, router)
    assert drift.changes == {"VLLM_MAX_LEN": ("32768", "65536")}

    write_profile(drift)

    assert profile.read_text().startswith("# why 32768\n")
    assert "VLLM_MAX_LEN=65536" in profile.read_text()
    assert _check(presets, router)[0].in_sync


def test_backend_without_a_preset_is_an_error(tmp_path: Path) -> None:
    presets, services, router = _setup(tmp_path)
    _write_router(router, services, [("b2", "nobody-serves-this", "vllm-tp.env.b2")])

    [drift] = _check(presets, router)

    assert drift.error and "nobody-serves-this" in drift.error
    assert not drift.in_sync


def test_two_presets_serving_one_name_is_an_error(tmp_path: Path) -> None:
    presets, _services, router = _setup(tmp_path)
    _write_preset(presets, "m1-copy", "served-1")

    [drift] = _check(presets, router)

    assert drift.error and "m1, m1-copy" in drift.error


def test_missing_profile_file_is_drift_and_write_creates_it(tmp_path: Path) -> None:
    presets, services, router = _setup(tmp_path)

    [drift] = _check(presets, router)
    assert not drift.in_sync

    write_profile(drift)

    assert _check(presets, router)[0].in_sync
    assert (services / "vllm-tp.env.b1").is_file()


def test_merge_drops_removed_keys_and_appends_new_ones() -> None:
    current = "# keep\nA=1\nGONE=x\n\n# b\nB=2\n"
    rendered = "A=1\nB=3\nNEW=y\n"

    merged = merge_env(current, rendered)

    assert merged == "# keep\nA=1\n\n# b\nB=3\nNEW=y\n"
    assert diff_env(merged, rendered) == {}
