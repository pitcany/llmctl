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
    """Only the path itself is resolved: a symlinked parent (macOS /tmp) keeps its value."""
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
    url = "http://127.0.0.1:8003"
    load = EngineLoad(runtime=runtime, upstream=url, version=version) if runtime else None
    return EngineLoadRow(name=name, runtime=runtime, upstream=url, source="unit", load=load)


def test_doctor_engine_row_names_the_serving_vllm_version(monkeypatch) -> None:
    rows = [
        _row("vllm-engine", "vllm", "0.30.0"),
        _row("gpt-oss-120b", "llama_cpp", "b6000"),
        _row("idle", None, None),
    ]
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


def test_hf_home_tilde_is_expanded_for_the_environment_file(monkeypatch) -> None:
    """systemd does not expand ``~``, so the rendered value must not contain one."""
    monkeypatch.setenv("HF_HOME", "~/some-hf-cache")
    assert resolve_hf_home() == str(Path.home() / "some-hf-cache")


def _engine_rows(monkeypatch, rows, users) -> list[DoctorCheck]:
    import llmctl.services.engine_load as engine_load
    import llmctl.services.engines as engines
    from llmctl.services import doctor

    monkeypatch.setattr(engines, "engine_statuses", lambda cfg: rows)
    monkeypatch.setattr(doctor, "_model_users", lambda cfg: users)
    monkeypatch.setattr(
        engine_load, "collect_engine_loads", lambda cfg: [_row("vllm-engine", "vllm", "0.30.0")]
    )
    return doctor._check_engines(object())


def test_check_engines_gives_each_engine_exactly_one_row(monkeypatch) -> None:
    """An installed engine must not also get a `not used` or `not installed` row."""
    from llmctl.services.engines import EngineStatus

    rows = [
        EngineStatus("vllm", "path", "/x/bin/vllm", "0.25.1"),
        EngineStatus("llama_cpp", "path", "/x/bin/llama-server", "b6000"),
        EngineStatus("ollama", "lane", "qwen3coder30b"),
        EngineStatus("sglang", "missing", None),
        EngineStatus("strata", "missing", None),
    ]
    users = {"llama_cpp": ["m1"], "strata": ["m2"]}
    checks = _engine_rows(monkeypatch, rows, users)
    assert [c.name for c in checks] == [f"engine:{r.engine}" for r in rows]
    by_name = {c.name: c for c in checks}
    assert by_name["engine:vllm"].detail.endswith("serving vLLM 0.30.0 on vllm-engine")
    assert by_name["engine:llama_cpp"].verdict == "pass"
    assert by_name["engine:sglang"].detail == "not used"
    assert by_name["engine:strata"].verdict == "warn"


def test_missing_vllm_row_does_not_claim_a_serving_version(monkeypatch) -> None:
    from llmctl.services.engines import EngineStatus

    checks = _engine_rows(monkeypatch, [EngineStatus("vllm", "missing", None)], {})
    assert checks[0].detail == "not used"


def _blocked_preview_state():
    from types import SimpleNamespace

    preview = SimpleNamespace(blocked=True, already_serving=False, lines=lambda: ["weights gone"])
    return SimpleNamespace(preview=lambda target: preview)


def test_run_force_passes_a_blocked_vllm_preset_only() -> None:
    import pytest
    import typer
    from rich.console import Console

    from llmctl.cli_swap import SwapTarget, print_preview

    state, console = _blocked_preview_state(), Console(quiet=True)
    vllm = SwapTarget(key="m", kind="vllm", model="m", engine="vLLM", upstream="")
    lane = SwapTarget(key="m", kind="lane", model="m", engine="Ollama", upstream="")
    assert print_preview(state, vllm, console, force=True) is True
    with pytest.raises(typer.Exit):
        print_preview(state, vllm, console)
    with pytest.raises(typer.Exit):
        print_preview(state, lane, console, force=True)


def test_unset_hf_home_resolves_a_symlinked_default(tmp_path: Path, monkeypatch) -> None:
    """From systemd or a bare ssh (unset), the render matches a shell that sets the target."""
    target = tmp_path / "AI" / "cache" / "huggingface"
    target.mkdir(parents=True)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "huggingface").symlink_to(target)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    assert resolve_hf_home() == str(target.resolve())
