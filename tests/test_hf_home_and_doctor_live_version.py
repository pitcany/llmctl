"""HF_HOME renders canonically and `doctor engine:vllm` names the serving version (2026-11-01)."""
from __future__ import annotations

import os
from pathlib import Path

from llmctl.integrations.launcher_env import resolve_hf_home
from llmctl.services.doctor import DoctorCheck, _with_live_vllm_version
from llmctl.services.engine_load import EngineLoadRow
from llmctl.telemetry.engine_metrics import EngineLoad


def test_hf_home_symlink_resolves_to_its_target(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "AI" / "cache" / "huggingface"
    target.mkdir(parents=True)
    link = tmp_path / ".cache-huggingface"
    link.symlink_to(target)
    monkeypatch.setenv("HF_HOME", str(link))
    assert resolve_hf_home() == str(target.resolve())


def test_hf_home_plain_dir_is_returned_as_given(tmp_path: Path, monkeypatch) -> None:
    """Only the path itself is resolved: a symlinked parent (macOS /tmp) must not change the value."""
    plain = tmp_path / "hf"
    plain.mkdir()
    monkeypatch.setenv("HF_HOME", str(plain))
    assert resolve_hf_home() == str(plain)
    monkeypatch.setenv("HF_HOME", "/nonexistent/hf-home")
    assert resolve_hf_home() == "/nonexistent/hf-home"


def test_hf_home_unset_defaults_under_home(monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    value = resolve_hf_home()
    assert value.endswith(os.path.join(".cache", "huggingface")) or Path(value).is_absolute()


def _row(name: str, runtime: str | None, version: str | None) -> EngineLoadRow:
    load = EngineLoad(runtime=runtime or "vllm", upstream="http://127.0.0.1:8003", version=version) if runtime else None
    return EngineLoadRow(name=name, runtime=runtime, upstream="http://127.0.0.1:8003", source="unit", load=load)


def test_doctor_engine_row_names_the_serving_vllm_version(monkeypatch) -> None:
    rows = [_row("vllm-engine", "vllm", "0.30.0"), _row("gpt-oss-120b", "llama_cpp", "b6000"), _row("idle", None, None)]
    import llmctl.services.engine_load as engine_load

    monkeypatch.setattr(engine_load, "collect_engine_loads", lambda cfg: rows)
    check = DoctorCheck("engine:vllm", "pass", "found 0.25.1: /x/bin/vllm")
    out = _with_live_vllm_version(object(), check)
    assert out.detail == "found 0.25.1: /x/bin/vllm; serving vLLM 0.30.0 on vllm-engine"
    assert out.verdict == "pass" and out.name == "engine:vllm"


def test_doctor_engine_row_unchanged_when_nothing_serves(monkeypatch) -> None:
    import llmctl.services.engine_load as engine_load

    monkeypatch.setattr(engine_load, "collect_engine_loads", lambda cfg: [_row("idle", None, None)])
    check = DoctorCheck("engine:vllm", "pass", "found 0.25.1: /x/bin/vllm")
    assert _with_live_vllm_version(object(), check) is check

    def boom(cfg):
        raise RuntimeError("no registry")

    monkeypatch.setattr(engine_load, "collect_engine_loads", boom)
    assert _with_live_vllm_version(object(), check) is check
