"""Tests for GPU placement and vLLM argument construction in the scheduler."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session

from llmctl.db import ModelRecord, ProfileRecord, RuntimeName, get_engine, init_db
from llmctl.schemas import GPUInfo, SessionStartRequest
from llmctl.services import scheduler as scheduler_module
from llmctl.services.scheduler import SchedulerError, SchedulerService


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'gpu.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def _fake_gpus() -> list[GPUInfo]:
    return [
        GPUInfo(index=0, name="GPU0", memory_free_mb=2000),
        GPUInfo(index=1, name="GPU1", memory_free_mb=9000),
        GPUInfo(index=2, name="GPU2", memory_free_mb=5000),
    ]


def test_auto_picks_most_free_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module, "get_gpu_info", _fake_gpus)
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.VLLM, source="org/model")
        db.add(model)
        db.commit()
        db.refresh(model)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.VLLM, gpus_auto=True, dry_run=False
            )
        )
    assert plan.gpu_ids == [1]
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "1"


def test_auto_tensor_parallel_picks_multiple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scheduler_module, "get_gpu_info", _fake_gpus)
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.VLLM, source="org/model")
        profile = ProfileRecord(
            name="tp2", runtime=RuntimeName.VLLM, parameters={"tensor_parallel_size": 2}
        )
        db.add(model)
        db.add(profile)
        db.commit()
        db.refresh(model)
        db.refresh(profile)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id,
                profile_id=profile.id,
                runtime=RuntimeName.VLLM,
                gpus_auto=True,
                dry_run=False,
            )
        )
    assert plan.gpu_ids == [1, 2]  # two most-free GPUs in order


def test_refuse_when_no_gpu_for_vllm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module, "get_gpu_info", lambda: [])
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.VLLM, source="org/model")
        db.add(model)
        db.commit()
        db.refresh(model)
        scheduler = SchedulerService(db)
        request = SessionStartRequest(model_id=model.id, runtime=RuntimeName.VLLM, dry_run=False)
        plan = scheduler.create_launch_plan(request)
        assert any("No NVIDIA GPUs" in reason for reason in plan.refusal_reasons)
        with pytest.raises(SchedulerError):
            scheduler.validate(plan, force=False, dry_run=False)
        # --force bypasses enforcement; --dry-run also bypasses.
        scheduler.validate(plan, force=True, dry_run=False)


def test_cpu_flag_hides_gpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler_module, "get_gpu_info", lambda: [])
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.VLLM, source="org/model")
        db.add(model)
        db.commit()
        db.refresh(model)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id, runtime=RuntimeName.VLLM, allow_cpu=True, dry_run=False
            )
        )
    assert plan.gpu_ids == []
    assert plan.env["CUDA_VISIBLE_DEVICES"] == ""


def test_vllm_extended_args(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.VLLM, source="org/model")
        profile = ProfileRecord(
            name="x",
            runtime=RuntimeName.VLLM,
            parameters={
                "dtype": "float16",
                "quantization": "awq",
                "served_model_name": "my-llm",
                "extra_args": ["--enforce-eager"],
            },
        )
        db.add(model)
        db.add(profile)
        db.commit()
        db.refresh(model)
        db.refresh(profile)
        plan = SchedulerService(db).create_launch_plan(
            SessionStartRequest(
                model_id=model.id,
                profile_id=profile.id,
                runtime=RuntimeName.VLLM,
                gpu_ids=[0],
                dry_run=False,
            )
        )
    assert "--dtype" in plan.command and "float16" in plan.command
    assert "--quantization" in plan.command and "awq" in plan.command
    assert "--served-model-name" in plan.command and "my-llm" in plan.command
    assert "--enforce-eager" in plan.command
    assert plan.health_url is not None and plan.health_url.endswith("/v1/models")
