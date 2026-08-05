"""Scheduler refusal for incomplete sharded GGUFs and in-flight downloads.

Ports the ``llama-server-guarded`` wrapper's preflight into the scheduler's
refusal chain: a sharded model (``*-NNNNN-of-MMMMM.gguf``) with an absent
sibling shard, or HuggingFace ``*.incomplete`` staging files next to the
model, refuses at plan time instead of dying ~2 minutes into loading.
Anything unparseable fails OPEN — it must never block a valid launch.
"""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from llmctl.db import ModelRecord, RuntimeName, get_engine, init_db
from llmctl.schemas import SessionStartRequest
from llmctl.services.scheduler import SchedulerService


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'shards.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def _plan_for(db: Session, model_path: Path):
    """Register a llama.cpp model at ``model_path`` and build its dry-run plan."""
    model = ModelRecord(
        name="sharded-test",
        runtime=RuntimeName.LLAMA_CPP,
        path=str(model_path),
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    scheduler = SchedulerService(db)
    return scheduler.create_launch_plan(
        SessionStartRequest(
            model_id=model.id,
            runtime=RuntimeName.LLAMA_CPP,
            allow_cpu=True,
            dry_run=True,
        )
    )


def _shard_refusals(plan) -> list[str]:
    return [r for r in plan.refusal_reasons if "shard" in r.lower() or "downloading" in r]


def _make_shards(directory: Path, stem: str, total: int, *, skip: set[int] = frozenset()):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(1, total + 1):
        if i in skip:
            continue
        (directory / f"{stem}-{i:05d}-of-{total:05d}.gguf").write_bytes(b"GGUF")
    return directory / f"{stem}-00001-of-{total:05d}.gguf"


def test_complete_sharded_model_passes(tmp_path: Path) -> None:
    first = _make_shards(tmp_path / "model", "m", 4)
    with _db(tmp_path) as db:
        plan = _plan_for(db, first)
    assert _shard_refusals(plan) == []


def test_missing_tail_shard_refuses_naming_it(tmp_path: Path) -> None:
    first = _make_shards(tmp_path / "model", "m", 4, skip={4})
    with _db(tmp_path) as db:
        plan = _plan_for(db, first)
    refusals = _shard_refusals(plan)
    assert len(refusals) == 1
    assert "m-00004-of-00004.gguf" in refusals[0]
    assert "4 shard(s) expected" in refusals[0]


def test_missing_middle_shard_refuses(tmp_path: Path) -> None:
    first = _make_shards(tmp_path / "model", "m", 3, skip={2})
    with _db(tmp_path) as db:
        plan = _plan_for(db, first)
    refusals = _shard_refusals(plan)
    assert len(refusals) == 1
    assert "m-00002-of-00003.gguf" in refusals[0]


def test_non_sharded_model_passes(tmp_path: Path) -> None:
    single = tmp_path / "model" / "plain-model.gguf"
    single.parent.mkdir()
    single.write_bytes(b"GGUF")
    with _db(tmp_path) as db:
        plan = _plan_for(db, single)
    assert _shard_refusals(plan) == []


def test_unparseable_shard_name_fails_open(tmp_path: Path) -> None:
    """Shard-ish but not the llama.cpp pattern (4-digit counters): pass."""
    odd = tmp_path / "model" / "m-001-of-004.gguf"
    odd.parent.mkdir()
    odd.write_bytes(b"GGUF")
    with _db(tmp_path) as db:
        plan = _plan_for(db, odd)
    assert _shard_refusals(plan) == []


def test_hf_incomplete_staging_refuses(tmp_path: Path) -> None:
    single = tmp_path / "root" / "sub" / "plain-model.gguf"
    single.parent.mkdir(parents=True)
    single.write_bytes(b"GGUF")
    staging = tmp_path / "root" / ".cache" / "huggingface" / "download" / "sub"
    staging.mkdir(parents=True)
    (staging / "plain-model.gguf.incomplete").write_bytes(b"partial")
    with _db(tmp_path) as db:
        plan = _plan_for(db, single)
    refusals = _shard_refusals(plan)
    assert len(refusals) == 1
    assert "still downloading" in refusals[0]


def test_symlinked_shards_count_as_present(tmp_path: Path) -> None:
    """The documented verification recipe symlinks shards into a temp dir."""
    real_first = _make_shards(tmp_path / "real", "m", 2)
    linkdir = tmp_path / "linked"
    linkdir.mkdir()
    for shard in real_first.parent.iterdir():
        (linkdir / shard.name).symlink_to(shard)
    with _db(tmp_path) as db:
        plan = _plan_for(db, linkdir / real_first.name)
    assert _shard_refusals(plan) == []


def test_missing_model_path_does_not_double_refuse(tmp_path: Path) -> None:
    """A nonexistent path is _check_model_path's refusal; the shard check stays quiet."""
    with _db(tmp_path) as db:
        plan = _plan_for(db, tmp_path / "nope" / "m-00001-of-00004.gguf")
    assert _shard_refusals(plan) == []
    assert any("does not exist" in r for r in plan.refusal_reasons)


def test_vllm_runtime_is_exempt(tmp_path: Path) -> None:
    """The check is llama.cpp-only; HF-repo sources for vLLM are untouched."""
    with _db(tmp_path) as db:
        model = ModelRecord(
            name="vllm-model",
            runtime=RuntimeName.VLLM,
            source="Qwen/Qwen3.6-27B-FP8",
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id,
                runtime=RuntimeName.VLLM,
                allow_cpu=True,
                dry_run=True,
            )
        )
    assert _shard_refusals(plan) == []
