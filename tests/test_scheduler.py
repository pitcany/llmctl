"""Tests for the scheduler launch-plan builder."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from llmctl.db import ModelRecord, ProfileRecord, RuntimeName, get_engine, init_db
from llmctl.schemas import SessionStartRequest
from llmctl.services.scheduler import SchedulerService


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'sched.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def test_vllm_command_includes_model_and_port(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(name="llama", runtime=RuntimeName.VLLM, source="meta/Llama-3-8B")
        profile = ProfileRecord(
            name="p1", runtime=RuntimeName.VLLM, parameters={"tensor_parallel_size": 2}
        )
        db.add(model)
        db.add(profile)
        db.commit()
        db.refresh(model)
        db.refresh(profile)

        scheduler = SchedulerService(db)
        plan = scheduler.create_launch_plan(
            SessionStartRequest(
                model_id=model.id,
                profile_id=profile.id,
                runtime=RuntimeName.VLLM,
                gpu_ids=[0, 1],
                dry_run=False,
            )
        )
    assert "serve" in plan.command
    assert "meta/Llama-3-8B" in plan.command
    assert "--tensor-parallel-size" in plan.command
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert plan.endpoint_url is not None and "http://" in plan.endpoint_url


def test_vllm_command_honors_promoted_columns(tmp_path: Path) -> None:
    """Profile knobs set as typed columns (e.g. via ``llmctl profile create``)
    must reach the launch command, not only those nested in ``parameters``.

    Regression: the scheduler read ``profile.parameters`` directly, so a profile
    whose tensor_parallel_size/quantization lived only in the promoted columns
    silently fell back to tp=1 with a bare ``vllm serve`` command.
    """
    with _db(tmp_path) as db:
        model = ModelRecord(
            name="llama70b",
            runtime=RuntimeName.VLLM,
            source="casperhansen/llama-3.3-70b-instruct-awq",
        )
        profile = ProfileRecord(
            name="tp2-awq",
            runtime=RuntimeName.VLLM,
            tensor_parallel_size=2,
            quantization="awq_marlin",
            max_model_len=40960,
            # parameters dict intentionally empty: values live only in columns.
        )
        db.add(model)
        db.add(profile)
        db.commit()
        db.refresh(model)
        db.refresh(profile)

        scheduler = SchedulerService(db)
        plan = scheduler.create_launch_plan(
            SessionStartRequest(
                model_id=model.id,
                profile_id=profile.id,
                runtime=RuntimeName.VLLM,
                gpu_ids=[0, 1],
                dry_run=False,
            )
        )
    assert plan.tensor_parallel_size == 2
    assert "--tensor-parallel-size" in plan.command
    assert plan.command[plan.command.index("--tensor-parallel-size") + 1] == "2"
    assert "--quantization" in plan.command
    assert "awq_marlin" in plan.command
    assert "--max-model-len" in plan.command


def test_llama_cpp_command_uses_path(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(
            name="local", runtime=RuntimeName.LLAMA_CPP, path="/models/local.gguf"
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        scheduler = SchedulerService(db)
        plan = scheduler.create_launch_plan(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.LLAMA_CPP, dry_run=False)
        )
    assert "-m" in plan.command
    assert "/models/local.gguf" in plan.command


def test_http_runtime_has_no_command(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        model = ModelRecord(name="m", runtime=RuntimeName.OLLAMA, source="llama3")
        db.add(model)
        db.commit()
        db.refresh(model)
        scheduler = SchedulerService(db)
        plan = scheduler.create_launch_plan(
            SessionStartRequest(model_id=model.id, runtime=RuntimeName.OLLAMA, dry_run=False)
        )
    assert plan.command == []
    assert plan.endpoint_url == "http://127.0.0.1:11434"


def test_dry_run_adds_safety_check(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        scheduler = SchedulerService(db)
        plan = scheduler.create_launch_plan(
            SessionStartRequest(model_id="x", runtime=RuntimeName.VLLM, dry_run=True)
        )
    assert "dry_run_no_process_launch" in plan.safety_checks
