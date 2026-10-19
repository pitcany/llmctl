"""`llmctl run --tp/--max-len/--gpu-mem [--save]` and `llmctl settings`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from sqlmodel import Session
from typer.testing import CliRunner

from llmctl import cli, cli_swap
from llmctl.adapters.vllm_systemd import ManagedRestartResult
from llmctl.db import ModelRecord, ProfileRecord, RuntimeName, get_engine, init_db
from llmctl.integrations.vllm_env import VLLMLaunchSpec
from llmctl.presets.store import PresetRecord, load_all_records
from llmctl.services.lanes import Lane, SwapState, build_catalog
from llmctl.services.run_overrides import (
    OverrideError,
    RunOverrides,
    save_to_preset,
    save_to_profile,
    session_parameters,
)
from llmctl.services.session_swap import start_request_fields
from llmctl.services.vllm_orchestrator import Dependencies, OrchestratorResult, _build_spec

PRESET = """\
# A hand-written preset: these comments must survive --save.
alias: tiny
served_name: tiny
model_id: org/tiny
quantization: none
vllm_quantization_flag: none
tensor_parallel_size: 2   # both cards
max_model_len: 32768
"""


def _preset(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(PRESET)
    return path


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'run.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


# --- translating the flags -------------------------------------------------------------


def test_overrides_list_only_the_flags_given() -> None:
    assert not RunOverrides()
    ov = RunOverrides(max_len=8192, gpu_mem=0.9)
    assert ov.fields() == {"max_model_len": 8192, "gpu_memory_utilization": 0.9}
    assert ov.describe() == "--max-len 8192 --gpu-mem 0.9"


def test_llama_cpp_calls_the_context_ctx_size() -> None:
    params = session_parameters(RuntimeName.LLAMA_CPP, RunOverrides(max_len=65536))
    assert params == {"ctx_size": 65536}


@pytest.mark.parametrize(
    ("runtime", "overrides", "fragment"),
    [
        (RuntimeName.LLAMA_CPP, RunOverrides(gpu_mem=0.9), "no memory-fraction"),
        (RuntimeName.FREETOKEN, RunOverrides(tp=2), "one GPU"),
        (RuntimeName.STRATA, RunOverrides(tp=2), "one GPU"),
    ],
)
def test_flags_an_engine_cannot_take_are_refused(runtime, overrides, fragment) -> None:
    with pytest.raises(OverrideError, match=fragment):
        session_parameters(runtime, overrides)


def test_sglang_takes_all_three() -> None:
    ov = RunOverrides(tp=2, max_len=4096, gpu_mem=0.8)
    assert session_parameters(RuntimeName.SGLANG, ov) == ov.fields()


def test_session_request_merges_parameters_but_cards_win() -> None:
    record = ModelRecord(id="m1", name="x", runtime=RuntimeName.LLAMA_CPP)
    fields = start_request_fields(record, 1, {"ctx_size": 4096, "tensor_parallel_size": 2})
    assert fields["parameters"] == {"ctx_size": 4096, "tensor_parallel_size": 1}


def test_vllm_spec_uses_the_overrides_for_this_run_only(tmp_path: Path) -> None:
    _preset(tmp_path)
    deps = Dependencies(config_dir=tmp_path)
    spec = _build_spec(
        "tiny",
        deps=deps,
        defaults=None,
        port_override=8003,
        tq_override=None,
        preset_overrides={"tensor_parallel_size": 1, "max_model_len": 4096},
    )
    assert (spec.tensor_parallel, spec.max_model_len, spec.gpus) == (1, 4096, "0")
    plain = _build_spec("tiny", deps=deps, defaults=None, port_override=8003, tq_override=None)
    assert (plain.tensor_parallel, plain.max_model_len) == (2, 32768)
    assert (tmp_path / "tiny.yaml").read_text() == PRESET


# --- saving to a preset ----------------------------------------------------------------


def test_save_to_preset_edits_lines_and_keeps_comments(tmp_path: Path) -> None:
    path = _preset(tmp_path)
    changes = save_to_preset(
        path, {"tensor_parallel_size": 1, "max_model_len": 4096, "gpu_memory_utilization": 0.9}
    )
    text = path.read_text()
    assert "# A hand-written preset" in text
    assert "tensor_parallel_size: 1   # both cards\n" in text
    assert "max_model_len: 4096\n" in text
    assert text.endswith("gpu_memory_utilization: 0.9\n")
    assert changes == [
        "tensor_parallel_size 2 -> 1",
        "max_model_len 32768 -> 4096",
        "gpu_memory_utilization unset -> 0.9",
    ]
    assert load_all_records(config_dir=tmp_path)["tiny"].model.max_model_len == 4096


def test_save_to_preset_with_the_same_values_changes_nothing(tmp_path: Path) -> None:
    path = _preset(tmp_path)
    assert save_to_preset(path, {"max_model_len": 32768}) == []
    assert path.read_text() == PRESET


def test_save_to_preset_refuses_an_invalid_result(tmp_path: Path) -> None:
    path = _preset(tmp_path)
    path.write_text(PRESET.replace("quantization: none\n", ""))  # already incomplete
    with pytest.raises(OverrideError, match="invalid"):
        save_to_preset(path, {"max_model_len": 4096})
    assert "max_model_len: 32768" in path.read_text()


# --- saving to a profile ---------------------------------------------------------------


def test_a_model_without_a_profile_gets_its_own(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(name="glm", runtime=RuntimeName.LLAMA_CPP)
        db.add(model)
        db.commit()
        name, changes = save_to_profile(db, model, {"ctx_size": 65536})
        assert name == "glm" and changes == ["ctx_size unset -> 65536"]
        profile = db.get(ProfileRecord, model.default_profile_id)
        assert profile.parameters == {"ctx_size": 65536}
        assert profile.runtime == RuntimeName.LLAMA_CPP


def test_saving_updates_the_models_profile_and_keeps_other_keys(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        profile = ProfileRecord(
            name="glm-cpumoe",
            runtime=RuntimeName.LLAMA_CPP,
            parameters={"ctx_size": 131072, "n_gpu_layers": 99},
        )
        db.add(profile)
        db.commit()
        model = ModelRecord(
            name="glm", runtime=RuntimeName.LLAMA_CPP, default_profile_id=profile.id
        )
        db.add(model)
        db.commit()
        name, changes = save_to_profile(db, model, {"ctx_size": 65536})
        assert name == "glm-cpumoe" and changes == ["ctx_size 131072 -> 65536"]
        db.refresh(profile)
        assert profile.parameters == {"ctx_size": 65536, "n_gpu_layers": 99}


def test_a_shared_profile_is_not_changed(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        profile = ProfileRecord(name="shared", runtime=RuntimeName.SGLANG, parameters={})
        db.add(profile)
        db.commit()
        a = ModelRecord(name="a", runtime=RuntimeName.SGLANG, default_profile_id=profile.id)
        b = ModelRecord(name="b", runtime=RuntimeName.SGLANG, default_profile_id=profile.id)
        db.add_all([a, b])
        db.commit()
        with pytest.raises(OverrideError, match="shared by 2 models"):
            save_to_profile(db, a, {"max_model_len": 4096})
        db.refresh(profile)
        assert profile.parameters == {}


# --- the CLI ---------------------------------------------------------------------------


class _Spec:
    served_name = "tiny"


FT = Lane(
    name="flash",
    served_model="ft-model",
    upstream="http://127.0.0.1:1919",
    start_cmd=("ft-start",),
    unit="ft.service",
    runtime="freetoken",
)


def _invoke(monkeypatch, tmp_path: Path, *args: str, ok: bool = True):
    state = SwapState(
        catalog=build_catalog({"tiny": _Spec()}, [FT], vllm_port=8003),
        serving=set(),
        active_units=set(),
        vllm_unit="vllm-tp",
        vllm_active=False,
    )
    seen: dict[str, object] = {}

    def fake_start(preset, **kwargs):
        seen["overrides"] = kwargs["options"].preset_overrides
        return OrchestratorResult(
            spec=VLLMLaunchSpec(model="m", served_name="tiny", port=8003),
            dry_run=kwargs["options"].dry_run,
            restart=ManagedRestartResult(
                env_path=Path("/tmp/x.env"),
                env_body="",
                ready=ok,
                error=None if ok else "boom",
            ),
        )

    path = _preset(tmp_path)
    records = {
        "tiny": PresetRecord(
            model=load_all_records(config_dir=tmp_path)["tiny"].model, source_path=path
        )
    }
    monkeypatch.setattr(cli, "gather_state", lambda settings: state)
    monkeypatch.setattr(cli, "start_vllm_tp", fake_start)
    monkeypatch.setattr(cli, "_confirm_state_change", lambda *a, **k: None)
    monkeypatch.setattr(cli, "record_load", lambda *a, **k: None)
    monkeypatch.setattr(cli_swap, "stop_lanes_for_vllm", lambda state, console: [])
    monkeypatch.setattr(cli_swap, "restore_lanes", lambda lanes, console: None)
    monkeypatch.setattr("llmctl.presets.load_all_records", lambda *a, **k: records)
    result = CliRunner().invoke(cli.app, ["run", *args])
    return result, seen, path


def test_run_passes_the_flags_and_save_writes_after_success(monkeypatch, tmp_path) -> None:
    res, seen, path = _invoke(monkeypatch, tmp_path, "tiny", "--max-len", "4096", "--save")
    assert res.exit_code == 0, res.output
    assert seen["overrides"] == {"max_model_len": 4096}
    assert "Saved" in res.output
    assert "max_model_len: 4096" in path.read_text()


def test_a_failed_run_saves_nothing(monkeypatch, tmp_path) -> None:
    res, _, path = _invoke(monkeypatch, tmp_path, "tiny", "--max-len", "4096", "--save", ok=False)
    assert res.exit_code == 1
    assert path.read_text() == PRESET


def test_dry_run_says_what_it_would_save(monkeypatch, tmp_path) -> None:
    res, _, path = _invoke(monkeypatch, tmp_path, "tiny", "--tp", "1", "--save", "--dry-run")
    assert res.exit_code == 0, res.output
    assert "Would save --tp 1" in res.output
    assert path.read_text() == PRESET


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        (("tiny", "--save"), "give at least one"),
        (("flash", "--max-len", "4096"), "router lane"),
    ],
)
def test_run_refuses_flags_that_cannot_apply(monkeypatch, tmp_path, args, fragment) -> None:
    res, seen, _ = _invoke(monkeypatch, tmp_path, *args)
    assert res.exit_code == 2
    assert fragment in " ".join(res.output.split())
    assert "overrides" not in seen


def test_settings_shows_the_preset_and_where_it_lives(monkeypatch, tmp_path) -> None:
    path = _preset(tmp_path)
    real = load_all_records(config_dir=tmp_path)
    monkeypatch.setattr("llmctl.presets.load_all_records", lambda *a, **k: real)
    res = CliRunner().invoke(cli.app, ["settings", "tiny"])
    assert res.exit_code == 0, res.output
    flat = " ".join(res.output.split())
    assert "32768" in flat and "--max-len" in flat
    assert str(path.name) in flat
    assert yaml.safe_load(path.read_text())["max_model_len"] == 32768


def test_settings_edit_opens_the_preset_in_the_editor(monkeypatch, tmp_path) -> None:
    path = _preset(tmp_path)
    real = load_all_records(config_dir=tmp_path)
    monkeypatch.setattr("llmctl.presets.load_all_records", lambda *a, **k: real)
    opened: list[list[str]] = []
    monkeypatch.setenv("VISUAL", "myeditor --wait")
    monkeypatch.setattr("subprocess.call", lambda argv: opened.append(argv) or 0)
    res = CliRunner().invoke(cli.app, ["settings", "tiny", "--edit"])
    assert res.exit_code == 0, res.output
    assert opened == [["myeditor", "--wait", str(path)]]


# --- `llmctl settings` without NAME, or with a row number -----------------------------


def _settings_cli(monkeypatch, tmp_path: Path, *args: str, stdin: str | None = None):
    """`llmctl settings ...` over the same list `ls` shows: preset tiny (#1), lane flash (#2)."""
    _preset(tmp_path)
    real = load_all_records(config_dir=tmp_path)
    monkeypatch.setattr("llmctl.presets.load_all_records", lambda *a, **k: real)
    state = SwapState(
        catalog=build_catalog({"tiny": _Spec()}, [FT], vllm_port=8003),
        serving=set(),
        active_units=set(),
        vllm_unit="vllm-tp",
        vllm_active=False,
    )
    monkeypatch.setattr(cli, "gather_state", lambda settings: state)
    if stdin is not None:
        # CliRunner swaps sys.stdin during invoke, so fake the terminal check itself.
        tty = SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True))
        monkeypatch.setattr(cli_swap, "sys", tty)
    res = CliRunner().invoke(cli.app, ["settings", *args], input=stdin)
    return res, " ".join(res.output.split())


def test_settings_without_name_off_a_terminal_lists_and_says_how(monkeypatch, tmp_path) -> None:
    res, flat = _settings_cli(monkeypatch, tmp_path)
    assert res.exit_code == 2, res.output
    assert "Models you can run" in flat
    assert "llmctl settings NAME" in flat and "llmctl run NAME`" not in flat
    assert "llmctl settings NAME (or: llmctl settings)" in flat  # the table caption
    assert "llmctl run" not in flat


def test_settings_without_name_on_a_terminal_picks_a_row(monkeypatch, tmp_path) -> None:
    res, flat = _settings_cli(monkeypatch, tmp_path, stdin="1\n")
    assert res.exit_code == 0, res.output
    assert "Show settings for #" in flat
    assert "tiny settings" in flat and "32768" in flat


def test_settings_takes_the_row_number_from_ls(monkeypatch, tmp_path) -> None:
    res, flat = _settings_cli(monkeypatch, tmp_path, "1")
    assert res.exit_code == 0, res.output
    assert "tiny settings" in flat and "--max-len" in flat


def test_settings_for_a_lane_says_llmctl_does_not_own_them(monkeypatch, tmp_path) -> None:
    res, flat = _settings_cli(monkeypatch, tmp_path, "2")
    assert res.exit_code == 0, res.output
    assert "flash is a router lane" in flat


def test_settings_out_of_range_row_number(monkeypatch, tmp_path) -> None:
    res, flat = _settings_cli(monkeypatch, tmp_path, "9")
    assert res.exit_code == 2, res.output
    assert "No row 9" in flat and "1-2" in flat
