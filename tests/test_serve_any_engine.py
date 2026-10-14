"""An added model is servable, and `swap NAME` serves any engine.

Two gaps closed together: a vLLM checkpoint added with `model add` could not
be swapped to (swap only lists presets), and "serve it" was a different verb
per engine. These tests pin the starter preset (what it infers, what it
refuses to guess, that it never overwrites a hand-written preset, that purge
takes only its own) and the per-engine routing behind `llmctl run NAME`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from sqlmodel import Session
from typer.testing import CliRunner

from llmctl.config import VLLMDefaultsConfig
from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.presets import load_all_records
from llmctl.schemas import Model, ModelCreate
from llmctl.services import model_ops
from llmctl.services.lanes import SwapTarget
from llmctl.services.model_ops import ModelOps, plan_add, plan_remove
from llmctl.services.preset_scaffold import (
    STARTER_KEY,
    alias_for,
    build_starter_preset,
    ensure_starter_preset,
)
from llmctl.services.registry import RegistryService
from llmctl.services.serve import AmbiguousModel, find_model, route


@pytest.fixture(autouse=True)
def _presets_in_tmp(tmp_path, monkeypatch) -> None:
    """Every preset read/write lands under tmp, never ~/.config; two fake 32 GB GPUs."""
    from llmctl.schemas import GPUInfo

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cards = [GPUInfo(index=i, name="fake", memory_total_mb=32 * 1024) for i in (0, 1)]
    monkeypatch.setattr("llmctl.telemetry.gpu.get_gpu_info", lambda: cards)


@pytest.fixture
def db(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'serve.db'}"
    init_db(url)
    with Session(get_engine(url)) as session:
        yield session


def _checkpoint(root: Path, cfg: dict) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(cfg))
    return root


DEFAULTS = VLLMDefaultsConfig(tensor_parallel=2, max_model_len=32768)


# --------------------------------------------------------------------------
# Starter preset
# --------------------------------------------------------------------------


def test_starter_preset_reads_config_and_guesses_nothing_risky(tmp_path) -> None:
    ckpt = _checkpoint(
        tmp_path / "ckpt",
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "quantization_config": {"quant_method": "fp8"},
            "text_config": {"max_position_embeddings": 262144, "num_attention_heads": 24},
        },
    )
    model = Model(id="m1", name="Qwen/Qwen3.8-27B-FP8", runtime=RuntimeName.VLLM,
                  source="Qwen/Qwen3.8-27B-FP8", path=str(ckpt))
    preset, notes = build_starter_preset(model, "qwen3.8-27b-fp8", DEFAULTS)
    assert preset.model_id == "Qwen/Qwen3.8-27B-FP8"  # repo id, resolved via HF_HOME
    assert preset.max_model_len == 32768  # capped at settings, not native 262144
    assert preset.tensor_parallel_size == 2
    assert preset.quantization == "fp8"
    # Empty flag: vLLM reads quantization_config; "none" would be a bad --quantization.
    assert preset.vllm_quantization_flag == ""
    assert preset.tool_parser is None and preset.reasoning_parser is None
    assert preset.family == "qwen" and preset.model_ref == "m1"
    assert any("native 262144" in n for n in notes)
    assert any("tool_parser" in n for n in notes)


def test_starter_preset_drops_tp_when_heads_do_not_split(tmp_path) -> None:
    ckpt = _checkpoint(tmp_path / "c", {"num_attention_heads": 7, "max_position_embeddings": 4096})
    model = Model(name="x/y", runtime=RuntimeName.VLLM, source="x/y", path=str(ckpt))
    preset, _ = build_starter_preset(model, "y", DEFAULTS)
    assert preset.tensor_parallel_size == 1
    assert preset.max_model_len == 4096


def test_alias_for() -> None:
    assert alias_for("Qwen/Qwen3-8B") == "qwen3-8b"
    assert alias_for("/models/My Model_v2/") == "my-model-v2"


def _vllm_row(db, tmp_path: Path, repo="org/Tiny") -> Model:
    ckpt = _checkpoint(tmp_path / repo.replace("/", "_"), {"max_position_embeddings": 8192})
    return RegistryService(db).add_model(
        ModelCreate(name=repo, runtime=RuntimeName.VLLM, source=repo, path=str(ckpt))
    )


def test_ensure_writes_once_records_the_alias_and_reuses(db, tmp_path) -> None:
    model = _vllm_row(db, tmp_path)
    first = ensure_starter_preset(db, model, defaults=DEFAULTS)
    assert first.created and first.alias == "tiny"
    assert load_all_records()["tiny"].model.model_id == "org/Tiny"
    assert RegistryService(db).get_model(model.id).metadata[STARTER_KEY] == "tiny"
    again = ensure_starter_preset(db, model, defaults=DEFAULTS)
    assert (again.created, again.alias) == (False, "tiny")


def test_ensure_never_overwrites_a_hand_written_preset(db, tmp_path) -> None:
    from llmctl.presets import default_preset_dir

    hand = default_preset_dir()
    hand.mkdir(parents=True)
    (hand / "tiny.yaml").write_text(yaml.safe_dump({
        "alias": "tiny", "served_name": "tiny", "model_id": "someone/Else",
        "quantization": "none", "vllm_quantization_flag": "",
        "tensor_parallel_size": 2, "max_model_len": 4096,
    }))
    model = _vllm_row(db, tmp_path)
    starter = ensure_starter_preset(db, model, defaults=DEFAULTS)
    assert starter.alias == "tiny-2"
    assert load_all_records()["tiny"].model.model_id == "someone/Else"


def test_add_vllm_writes_the_preset_and_says_how_to_serve(db, tmp_path, monkeypatch) -> None:
    ckpt = _checkpoint(tmp_path / "local-ckpt", {"max_position_embeddings": 8192})
    monkeypatch.setattr(
        "llmctl.services.preset_scaffold.load_settings",
        lambda: type("S", (), {"vllm": type("V", (), {"defaults": DEFAULTS})()})(),
    )
    result = ModelOps(db).add(plan_add(str(ckpt)))
    assert result.preset == "local-ckpt"
    assert result.serve == "llmctl run local-ckpt"
    assert "local-ckpt" in load_all_records()


def test_purge_takes_the_starter_preset_but_a_hand_written_one_blocks(
    db, tmp_path, monkeypatch
) -> None:
    root = tmp_path / "models"
    ckpt = _checkpoint(root / "ckpt", {"max_position_embeddings": 8192})
    monkeypatch.setattr(model_ops, "_model_roots", lambda: [root.resolve()])
    model = RegistryService(db).add_model(
        ModelCreate(name="ckpt", runtime=RuntimeName.VLLM, source=str(ckpt), path=str(ckpt))
    )
    ensure_starter_preset(db, model, defaults=DEFAULTS)
    model = RegistryService(db).get_model(model.id)

    plan = plan_remove(model, db=db, roots=[root], hub_cache=tmp_path / "hub")
    assert plan.presets_to_delete == ("ckpt",) and not plan.blockers
    assert "starter preset ckpt" in plan.purge_label

    result = ModelOps(db, hub_cache=tmp_path / "hub").remove(model.id, purge=True)
    assert "preset ckpt deleted" in result.message
    assert not ckpt.exists() and "ckpt" not in load_all_records()


# --------------------------------------------------------------------------
# Routing: `swap NAME` for every engine
# --------------------------------------------------------------------------


def _catalog(*aliases: str) -> list[SwapTarget]:
    return [SwapTarget(key=a, kind="vllm", model=a, engine="vLLM", upstream="") for a in aliases]


@pytest.mark.parametrize(
    ("runtime", "kind"),
    [
        (RuntimeName.OLLAMA, "on-demand"),
        (RuntimeName.LMSTUDIO, "on-demand"),
        (RuntimeName.LLAMA_CPP, "session"),
        (RuntimeName.PYTHON_SCRIPT, "start"),
        (RuntimeName.SGLANG, "session"),
        (RuntimeName.FREETOKEN, "session"),
    ],
)
def test_route_by_engine(db, runtime, kind) -> None:
    model = RegistryService(db).add_model(ModelCreate(name="m", runtime=runtime, source="m"))
    assert route(db, model, []).kind == kind


def test_route_vllm_to_its_preset_or_offers_one(db, tmp_path) -> None:
    model = _vllm_row(db, tmp_path)
    assert route(db, model, _catalog()).kind == "no-preset"
    ensure_starter_preset(db, model, defaults=DEFAULTS)
    model = RegistryService(db).get_model(model.id)
    served = route(db, model, _catalog("tiny"))
    assert served.kind == "preset" and served.target.key == "tiny"


def test_route_refuses_rows_from_another_machine(db) -> None:
    model = Model(id="x", name="m", runtime=RuntimeName.OLLAMA, host="a-mac.lan")
    assert route(db, model, []).kind == "cannot"


def test_find_model_prefers_this_machine_and_flags_real_ambiguity(db) -> None:
    reg = RegistryService(db)
    reg.add_model(ModelCreate(name="dup", runtime=RuntimeName.LMSTUDIO, source="dup"))
    remote = reg.add_model(ModelCreate(name="dup", runtime=RuntimeName.LMSTUDIO, source="dup2"))
    from llmctl.db import ModelRecord

    record = db.get(ModelRecord, remote.id)
    record.host = "a-mac.lan"
    db.add(record)
    db.commit()
    assert find_model(db, "dup").host != "a-mac.lan"
    reg.add_model(ModelCreate(name="dup", runtime=RuntimeName.OLLAMA, source="dup"))
    with pytest.raises(AmbiguousModel, match="Use the id"):
        find_model(db, "dup")


# --------------------------------------------------------------------------
# CLI: llmctl run NAME
# --------------------------------------------------------------------------


@pytest.fixture
def cli(tmp_path, monkeypatch):
    import llmctl.cli as cli_mod
    from llmctl.services.lanes import SwapState

    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'cli.sqlite3'}")
    empty = SwapState(
        catalog=[], serving=set(), active_units=set(), vllm_unit="vllm-tp.service",
        vllm_active=False,
    )
    monkeypatch.setattr(cli_mod, "gather_state", lambda settings: empty)
    return cli_mod


def _add(runtime, name, path=None) -> Model:
    from llmctl.cli_registry import _session

    with _session() as db:
        return RegistryService(db).add_model(
            ModelCreate(name=name, runtime=runtime, source=path or name, path=path)
        )


def test_cli_swap_ollama_model_says_it_loads_on_demand(cli) -> None:
    _add(RuntimeName.OLLAMA, "qwen3:8b")
    result = CliRunner().invoke(cli.app, ["run", "qwen3:8b"])
    assert result.exit_code == 0, result.output
    assert "Nothing to start" in result.output
    assert "ollama run qwen3:8b" in result.output


def test_cli_swap_llama_cpp_row_frees_the_gpus_for_it(cli, monkeypatch, tmp_path) -> None:
    import llmctl.cli_swap as cli_swap
    from llmctl.services.session_swap import SessionSwapResult

    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x")
    _add(RuntimeName.LLAMA_CPP, "tiny-gguf", str(gguf))
    ran: list = []
    monkeypatch.setattr(
        cli_swap,
        "run_session_swap_service",
        lambda state, target, **kw: ran.append(target) or SessionSwapResult(ok=True),
    )
    result = CliRunner().invoke(cli.app, ["run", "tiny-gguf", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "llama-server for tiny-gguf on one GPU" in result.output
    assert ran == []  # a dry run stops and starts nothing

    result = CliRunner().invoke(cli.app, ["run", "tiny-gguf", "--yes"])
    assert result.exit_code == 0, result.output
    assert [t.kind for t in ran] == ["session"] and "Ready" in result.output


def test_cli_swap_vllm_checkpoint_without_preset_offers_one(cli, tmp_path) -> None:
    ckpt = _checkpoint(tmp_path / "ck", {"max_position_embeddings": 8192})
    _add(RuntimeName.VLLM, "ck", str(ckpt))
    result = CliRunner().invoke(cli.app, ["run", "ck", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "vLLM needs a preset" in result.output
    assert "Would write a starter preset" in result.output
    assert load_all_records() == {}  # dry run wrote nothing


def test_cli_swap_unknown_name_still_lists_presets(cli) -> None:
    result = CliRunner().invoke(cli.app, ["run", "no-such-thing", "--dry-run"])
    assert result.exit_code == 2
