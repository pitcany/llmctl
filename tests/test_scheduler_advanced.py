"""Tests for advanced scheduler placement, VRAM control, ports, and refusals."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session

from llmctl.db import (
    ModelRecord,
    ProfileRecord,
    RuntimeName,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import GPUInfo, SessionStartRequest
from llmctl.services import scheduler as scheduler_module
from llmctl.services.scheduler import SchedulerError, SchedulerService


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'adv.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def _gpus() -> list[GPUInfo]:
    return [
        GPUInfo(index=0, name="GPU0", memory_free_mb=8000, utilization_gpu_percent=10),
        GPUInfo(index=1, name="GPU1", memory_free_mb=8000, utilization_gpu_percent=80),
        GPUInfo(index=2, name="GPU2", memory_free_mb=8000, utilization_gpu_percent=40),
    ]


def _py_model(db: Session, tmp_path: Path, **kwargs: object) -> ModelRecord:
    script = tmp_path / "run.py"
    script.write_text("pass\n")
    model = ModelRecord(
        name="m", runtime=RuntimeName.PYTHON_SCRIPT, path=str(script), **kwargs
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


# -- GPU selection modes ----------------------------------------------------


def test_most_free_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gpus = [
        GPUInfo(index=0, name="a", memory_free_mb=2000),
        GPUInfo(index=1, name="b", memory_free_mb=9000),
        GPUInfo(index=2, name="c", memory_free_mb=5000),
    ]
    monkeypatch.setattr(scheduler_module, "get_gpu_info", lambda: gpus)
    with _db(tmp_path) as db:
        model = _py_model(db, tmp_path)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, gpu_mode="most-free"
            )
        )
    assert plan.gpu_ids == [1]


def test_least_used_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module, "get_gpu_info", _gpus)
    with _db(tmp_path) as db:
        model = _py_model(db, tmp_path)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, gpu_mode="least-used"
            )
        )
    assert plan.gpu_ids == [0]  # lowest utilization (10%)


def test_balanced_mode_prefers_unused_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler_module, "get_gpu_info", _gpus)
    with _db(tmp_path) as db:
        # Pre-existing active session occupies GPU 0.
        db.add(
            SessionRecord(
                runtime=RuntimeName.PYTHON_SCRIPT,
                status=SessionStatus.RUNNING,
                gpu_ids=[0],
                pid=1,
            )
        )
        db.commit()
        model = _py_model(db, tmp_path)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, gpu_mode="balanced"
            )
        )
    assert plan.gpu_ids and plan.gpu_ids[0] != 0  # avoids the busy GPU


def test_explicit_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module, "get_gpu_info", _gpus)
    with _db(tmp_path) as db:
        model = _py_model(db, tmp_path)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, gpu_ids=[2]
            )
        )
    assert plan.gpu_ids == [2]
    assert plan.gpu_selection_mode == "explicit"


# -- port allocation --------------------------------------------------------


def test_port_allocation_avoids_used(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        db.add(
            SessionRecord(
                runtime=RuntimeName.PYTHON_SCRIPT,
                status=SessionStatus.RUNNING,
                port=8200,
                pid=1,
            )
        )
        db.commit()
        model = _py_model(db, tmp_path)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT)
        )
    assert plan.port is not None
    assert plan.port != 8200
    assert 8200 <= plan.port <= 8299


# -- VRAM admission control -------------------------------------------------


def test_vram_refusal_when_too_large(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scheduler_module,
        "get_gpu_info",
        lambda: [GPUInfo(index=0, name="a", memory_free_mb=10000)],
    )
    with _db(tmp_path) as db:
        model = _py_model(db, tmp_path, estimated_vram_gb=20.0)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, gpu_mode="most-free"
            )
        )
    assert any("exceeds free" in r for r in plan.refusal_reasons)


def test_vram_ok_when_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scheduler_module,
        "get_gpu_info",
        lambda: [GPUInfo(index=0, name="a", memory_free_mb=24000)],
    )
    with _db(tmp_path) as db:
        model = _py_model(db, tmp_path, estimated_vram_gb=5.0)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT, gpu_mode="most-free"
            )
        )
    assert not any("exceeds free" in r for r in plan.refusal_reasons)
    assert plan.estimated_vram_gb == 5.0


def test_unknown_vram_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scheduler_module,
        "get_gpu_info",
        lambda: [GPUInfo(index=0, name="a", memory_free_mb=24000)],
    )
    with _db(tmp_path) as db:
        model = _py_model(db, tmp_path)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.PYTHON_SCRIPT)
        )
    assert any("VRAM unknown" in w for w in plan.warnings)


# -- safety refusals --------------------------------------------------------


def test_tensor_parallel_exceeds_gpu_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module, "get_gpu_info", _gpus)
    with _db(tmp_path) as db:
        model = _py_model(db, tmp_path)
        profile = ProfileRecord(
            name="tp2", runtime=RuntimeName.PYTHON_SCRIPT, parameters={"tensor_parallel_size": 2}
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id,
                profile_id=profile.id,
                runtime=RuntimeName.PYTHON_SCRIPT,
                gpu_ids=[0],
            )
        )
    assert any("tensor_parallel_size" in r for r in plan.refusal_reasons)


def test_profile_incompatible_with_backend(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.VLLM, source="org/m")
        profile = ProfileRecord(name="cpp", runtime=RuntimeName.LLAMA_CPP)
        db.add(model)
        db.add(profile)
        db.commit()
        db.refresh(model)
        db.refresh(profile)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, profile_id=profile.id, runtime=RuntimeName.VLLM
            )
        )
    assert any("incompatible" in r for r in plan.refusal_reasons)


def test_missing_binary_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scheduler_module,
        "get_gpu_info",
        lambda: [GPUInfo(index=0, name="a", memory_free_mb=24000)],
    )
    # Force `vllm` to appear absent regardless of the host env (the dev box's
    # vllm-serve conda env ships vllm on PATH; CI does not).
    monkeypatch.setattr(scheduler_module.shutil, "which", lambda _binary: None)
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.VLLM, source="org/m")
        db.add(model)
        db.commit()
        db.refresh(model)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.VLLM, gpu_mode="most-free")
        )
    assert any("not found on PATH" in r for r in plan.refusal_reasons)


def test_missing_model_path_refusal(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(
            name="m", runtime=RuntimeName.LLAMA_CPP, path="/nonexistent/model.gguf"
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.LLAMA_CPP, allow_cpu=True)
        )
    assert any("does not exist" in r for r in plan.refusal_reasons)


# -- validation / enforcement ----------------------------------------------


def test_validate_raises_then_force_bypasses(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(
            name="m", runtime=RuntimeName.LLAMA_CPP, path="/nonexistent/model.gguf"
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        scheduler = SchedulerService(db)
        plan = scheduler.create_launch_plan(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.LLAMA_CPP, allow_cpu=True)
        )
        with pytest.raises(SchedulerError):
            scheduler.validate(plan, force=False, dry_run=False)
        scheduler.validate(plan, force=True, dry_run=False)  # no raise
        scheduler.validate(plan, force=False, dry_run=True)  # no raise


def test_command_preview_present(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.VLLM, source="org/m")
        db.add(model)
        db.commit()
        db.refresh(model)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.VLLM, allow_cpu=True)
        )
    assert "serve" in plan.command_preview
    assert plan.gpu_selection_mode == "cpu"
