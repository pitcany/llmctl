"""The Python half of the llmctl-next preset parity check (Rust M5c item 2).

``tests/fixtures/rust_parity/parity.json`` records what this code renders
and edits for the sanitised presets next to it; the Rust suite
(``rust/crates/llmctl-core/tests/preset_parity.rs``) must match it byte for
byte. This test fails when Python's output moves off the file, so the two
cannot drift apart silently: regenerate with
``python tests/fixtures/rust_parity/generate.py`` and re-run both suites.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

GEN = Path(__file__).resolve().parent / "fixtures" / "rust_parity" / "generate.py"


def _generator():
    spec = importlib.util.spec_from_file_location("_rust_parity_generate", GEN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_renders_and_edits_match_the_parity_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # The generator pins/pops these; monkeypatch restores them afterwards.
    for var in (
        "LLMCTL_PYTHON_ROOT", "LLMCTL_CUDA_ROOT", "HF_HOME", "CONDA_PREFIX",
        "VIRTUAL_ENV", "GPU_MODELS_PYTHON_ROOT", "GPU_MODELS_CUDA_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    gen = _generator()
    recorded = json.loads(gen.OUT.read_text())
    fresh = gen.generate()
    assert len(fresh["renders"]) == len(recorded["renders"]) >= 18
    assert len(fresh["edits"]) == len(recorded["edits"])
    for got, want in zip(fresh["renders"], recorded["renders"], strict=True):
        assert got == want, got["file"]
    for got, want in zip(fresh["edits"], recorded["edits"], strict=True):
        assert got == want, got["name"]
    assert fresh["specs"] == recorded["specs"]
    assert gen.dumps(fresh) == gen.OUT.read_text()


def test_fixture_presets_carry_no_home_path() -> None:
    presets = GEN.parent / "presets"
    for path in presets.glob("*.yaml"):
        assert "/home/yannik" not in path.read_text(), path
