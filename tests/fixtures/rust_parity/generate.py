"""Generate the Python half of the llmctl-next preset parity fixtures (M5c item 2).

Both implementations read the same sanitised preset files
(``tests/fixtures/rust_parity/presets/``); this script records what the
Python side renders and edits into ``parity.json``. The Rust suite
(``rust/crates/llmctl-core/tests/preset_parity.rs``) must reproduce every
entry byte for byte, and ``tests/test_rust_preset_parity.py`` re-runs this
generator in memory and fails when Python's output moves off the file.

    python tests/fixtures/rust_parity/generate.py            # rewrite parity.json
    python tests/fixtures/rust_parity/generate.py --from-live  # refresh presets/ first

``--from-live`` copies ``~/.config/llmctl/presets/*.yaml`` with the home
directory replaced by ``/home/owner``. Nothing else is read from the host:
the launcher facts are pinned below.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import textwrap
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRESETS = HERE / "presets"
OUT = HERE / "parity.json"

LAUNCHER = {
    "python_root": "/home/owner/miniconda3/envs/vllm-serve",
    "cuda_root": "/usr/local/cuda",
    "hf_home": "/home/owner/AI/cache/huggingface",
}
#: The env the frozen ``vllm_env__*.txt`` fixtures were captured with.
SPEC_LAUNCHER = {
    "python_root": "/home/yannik/miniconda3/envs/vllm-serve",
    "cuda_root": "/usr/local/cuda",
    "hf_home": "/home/yannik/AI/cache/huggingface",
}
PORT = 8003
CUSTOM_DEFAULTS = {
    "gpus": "1",
    "port": 8010,
    "host": "0.0.0.0",
    "gpu_memory_utilization": 0.8,
    "prefix_cache": False,
    "chunked_prefill": False,
    "nccl_p2p_disable": True,
    "max_num_seqs": 32,
    "max_batched_tokens": 4096,
}


def _pin_env(launcher: dict[str, str]) -> None:
    os.environ["LLMCTL_PYTHON_ROOT"] = launcher["python_root"]
    os.environ["LLMCTL_CUDA_ROOT"] = launcher["cuda_root"]
    os.environ["HF_HOME"] = launcher["hf_home"]
    for var in ("CONDA_PREFIX", "VIRTUAL_ENV", "GPU_MODELS_PYTHON_ROOT", "GPU_MODELS_CUDA_ROOT"):
        os.environ.pop(var, None)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh_from_live() -> None:
    home = str(Path.home())
    live = Path.home() / ".config" / "llmctl" / "presets"
    PRESETS.mkdir(parents=True, exist_ok=True)
    for src in sorted(live.glob("*.yaml")):
        if src.name.startswith("_"):
            continue
        text = src.read_text().replace(home, "/home/owner")
        (PRESETS / f"live-{src.name}").write_text(text)
    # The shapes the loader tests freeze (gpu-models era presets).
    loader = _load_module(REPO / "tests" / "test_preset_loader.py", "_parity_loader")
    for alias, body in loader.PRESET_FIXTURES:
        (PRESETS / f"shape-{alias}.yaml").write_text(textwrap.dedent(body).strip() + "\n")


def _edits(raw: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Intended changes applied to every fixture preset."""
    out: list[tuple[str, dict[str, Any]]] = [("identity", dict(raw))]
    seqs = dict(raw)
    seqs["max_num_seqs"] = 16
    out.append(("set-max_num_seqs", seqs))
    dropped = {k: v for k, v in raw.items() if k not in ("tq", "shortcuts", "port")}
    out.append(("drop-vestigial", dropped))
    added = dict(raw)
    added["model_ref"] = "registry-row-1"
    out.append(("add-model_ref", added))
    spec = dict(raw)
    spec["spec_config"] = {"method": "mtp", "num_speculative_tokens": 5}
    out.append(("set-spec_config", spec))
    gpu = dict(raw)
    gpu["gpu_memory_utilization"] = 0.88
    gpu["extra_args"] = "--shutdown-timeout 30 --enable-expert-parallel"
    out.append(("set-gpu-and-extra", gpu))
    return out


#: test_preset_save_keeps.py's inputs (text, desired mapping).
REQ = (
    "alias: x\nserved_name: x\nmodel_id: org/m\nquantization: awq\n"
    "vllm_quantization_flag: awq\ntensor_parallel_size: 2\nmax_model_len: 4096\n"
)
REQ_KW: dict[str, Any] = {
    "alias": "x", "served_name": "x", "model_id": "org/m", "quantization": "awq",
    "vllm_quantization_flag": "awq", "tensor_parallel_size": 2, "max_model_len": 4096,
}
SAVE_KEEPS_TEXT = (
    "# header\n# more\n" + REQ.replace("awq\nvllm", "awq  # q\nvllm", 1)
    + "# between\nmax_num_seqs: 4  # seqs\nspec_config:\n  model: a\n  n: 2\n"
    "shortcuts:\n- s1\n- s2\nport: 9\n"
)


def generate() -> dict[str, Any]:
    import yaml

    from llmctl.config import VLLMDefaultsConfig
    from llmctl.integrations.vllm_env import VLLMLaunchSpec, render_vllm_env
    from llmctl.presets.yaml_edit import edit_preset_text
    from llmctl.services.vllm_orchestrator import render_preset_env

    _pin_env(LAUNCHER)
    renders: list[dict[str, Any]] = []
    edits: list[dict[str, Any]] = []
    files = sorted(p for p in PRESETS.glob("*.yaml") if not p.name.startswith("_"))
    for path in files:
        raw = yaml.safe_load(path.read_text())
        alias = raw["alias"]
        for label, defaults in (("default", None), ("custom", CUSTOM_DEFAULTS)):
            cfg = VLLMDefaultsConfig(**defaults) if defaults else VLLMDefaultsConfig()
            port = PORT if defaults is None else int(defaults["port"])
            _, body = render_preset_env(alias, port=port, defaults=cfg, config_dir=PRESETS)
            renders.append({
                "file": path.name, "alias": alias, "defaults": label, "port": port, "env": body,
            })
        text = path.read_text()
        for label, desired in _edits(raw):
            try:
                expected: str | None = edit_preset_text(text, desired)
            except ValueError:
                expected = None
            edits.append({"name": f"{path.name}:{label}", "text": text,
                          "desired": desired, "expected": expected})

    # test_preset_save_keeps.py cases, through the same editor.
    keeps_desired = {k: v for k, v in yaml.safe_load(SAVE_KEEPS_TEXT).items() if k != "port"}
    keeps_desired["max_num_seqs"] = 8
    keeps_desired["spec_config"] = {"model": "b"}
    for name, text, desired in [
        ("save-keeps:unset-stays-absent", REQ + "port: 8000\n", {**REQ_KW, "port": 8000}),
        ("save-keeps:new-file", "", dict(REQ_KW)),
        ("save-keeps:comments-survive", SAVE_KEEPS_TEXT, keeps_desired),
        ("save-keeps:max_model_len", REQ, {**REQ_KW, "max_model_len": 1024}),
    ]:
        try:
            expected = edit_preset_text(text, desired)
        except ValueError:
            expected = None
        edits.append({"name": name, "text": text, "desired": desired, "expected": expected})

    # Spec-level cases: the frozen gpu-models captures.
    _pin_env(SPEC_LAUNCHER)
    parity = _load_module(REPO / "tests" / "test_vllm_env_parity.py", "_parity_specs")
    specs: list[dict[str, Any]] = []
    for stem, kwargs in parity.PARITY_CASES:
        body = render_vllm_env(VLLMLaunchSpec(**kwargs))
        frozen = (REPO / "tests" / "fixtures" / "env_renders" / f"vllm_env__{stem}.txt").read_text()
        assert body == frozen, stem
        specs.append({"name": stem, "spec": kwargs, "env": body})

    return {
        "comment": "generated by tests/fixtures/rust_parity/generate.py; do not edit",
        "launcher": LAUNCHER,
        "spec_launcher": SPEC_LAUNCHER,
        "custom_defaults": CUSTOM_DEFAULTS,
        "renders": renders,
        "edits": edits,
        "specs": specs,
    }


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-live", action="store_true")
    args = ap.parse_args()
    if args.from_live:
        refresh_from_live()
    OUT.write_text(dumps(generate()))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
