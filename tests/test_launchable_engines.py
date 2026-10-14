"""SGLang and FreeToken are engines llmctl starts itself, like llama.cpp.

Before this they were "adopt-only": llmctl could watch a server somebody else
started but `llmctl run` answered "cannot". Now the scheduler builds their
launch commands, `run` routes them to a session, and health distinguishes
serving / installed-but-idle / not installed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlmodel import Session

from llmctl.adapters.freetoken import FreeTokenAdapter
from llmctl.adapters.sglang import SGLangAdapter
from llmctl.config import RuntimeConfig
from llmctl.db import ModelRecord, RuntimeName, get_engine, init_db
from llmctl.schemas import HealthState, SessionStartRequest
from llmctl.services import scheduler as scheduler_mod
from llmctl.services.lanes import session_rows_for
from llmctl.services.scheduler import SchedulerService


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'sched.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def _plan(db: Session, runtime: RuntimeName, parameters: dict | None = None, **req):
    model = ModelRecord(name="m", runtime=runtime, source="Qwen/Qwen3-8B")
    db.add(model)
    db.commit()
    db.refresh(model)
    scheduler = SchedulerService(db)
    scheduler.settings.runtimes[runtime.value] = RuntimeConfig(binary="/definitely/missing")
    return scheduler.create_launch_plan(
        SessionStartRequest(
            model_id=model.id,
            runtime=runtime,
            gpu_ids=req.pop("gpu_ids", [0]),
            parameters=parameters or {},
            dry_run=True,
            **req,
        )
    )


def test_sglang_command_is_the_launch_server_module(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        plan = _plan(
            db,
            RuntimeName.SGLANG,
            {"tensor_parallel_size": 2, "max_model_len": 8192, "served_model_name": "q"},
            gpu_ids=[0, 1],
        )
    assert plan.command[:3] == ["/definitely/missing", "-m", "sglang.launch_server"]
    assert "--model-path" in plan.command and "Qwen/Qwen3-8B" in plan.command
    assert plan.command[plan.command.index("--tp") + 1] == "2"
    assert plan.command[plan.command.index("--context-length") + 1] == "8192"
    assert plan.command[plan.command.index("--served-model-name") + 1] == "q"
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert plan.port is not None and plan.health_url is not None


def test_freetoken_command_is_ft_serve(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        plan = _plan(db, RuntimeName.FREETOKEN, {"gpu_memory_utilization": 0.8})
    assert plan.command[:4] == ["/definitely/missing", "serve", "--model", "Qwen/Qwen3-8B"]
    assert plan.command[plan.command.index("--memory-ratio") + 1] == "0.8"
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "0"


def test_freetoken_refuses_tensor_parallel(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        plan = _plan(db, RuntimeName.FREETOKEN, {"tensor_parallel_size": 2}, gpu_ids=[0, 1])
    assert any("one GPU" in reason for reason in plan.refusal_reasons)


def test_missing_engine_says_how_to_install_it(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        plan = _plan(db, RuntimeName.SGLANG)
    reasons = " ".join(plan.refusal_reasons)
    assert "sglang is not installed" in reasons
    assert "llmctl setup sglang" in reasons


def test_sglang_needs_the_module_not_just_a_python(tmp_path: Path, monkeypatch) -> None:
    """A python on PATH is not enough: the env must have sglang in it."""
    monkeypatch.setattr(scheduler_mod.shutil, "which", lambda b: "/usr/bin/python3")
    monkeypatch.setattr(scheduler_mod, "_python_has_module", lambda python, module: False)
    with _db(tmp_path) as db:
        plan = _plan(db, RuntimeName.SGLANG)
    assert any("not installed in /usr/bin/python3" in r for r in plan.refusal_reasons)


def test_gpu_is_required_for_both(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(scheduler_mod, "get_gpu_info", lambda: [])
    with _db(tmp_path) as db:
        for runtime in (RuntimeName.SGLANG, RuntimeName.FREETOKEN):
            plan = _plan(db, runtime, gpu_ids=[])
            assert any("No NVIDIA GPUs" in r for r in plan.refusal_reasons), runtime


# --- adapters -----------------------------------------------------------------------


@pytest.mark.parametrize("adapter_cls", [SGLangAdapter, FreeTokenAdapter])
def test_adapters_can_launch_and_stop(adapter_cls) -> None:
    caps = adapter_cls().capabilities()
    assert caps["launch_process"] and caps["stop_process"] and caps["logs"]


@pytest.mark.parametrize("adapter_cls", [SGLangAdapter, FreeTokenAdapter])
async def test_health_is_three_state(adapter_cls, monkeypatch) -> None:
    adapter = adapter_cls(RuntimeConfig(endpoint="http://127.0.0.1:1", binary="nope-bin"))

    async def unreachable():
        from llmctl.schemas import AdapterStatus

        return AdapterStatus(runtime=adapter.runtime, state=HealthState.UNAVAILABLE, message="x")

    monkeypatch.setattr(adapter, "_http_health", unreachable)
    monkeypatch.setattr(adapter, "installed_binary", lambda: None)
    status = await adapter.health_check()
    assert status.state == HealthState.UNAVAILABLE
    assert f"llmctl setup {adapter.runtime.value}" in status.message

    monkeypatch.setattr(adapter, "installed_binary", lambda: "/opt/engine/bin/x")
    status = await adapter.health_check()
    assert status.state == HealthState.OK
    assert "not running" in status.message and "/opt/engine/bin/x" in status.message


def test_old_endpoint_signature_still_works() -> None:
    adapter = SGLangAdapter("http://sglang.test")
    assert adapter.endpoint == "http://sglang.test"
    assert adapter.config.binary == "python3"


# --- the run list ----------------------------------------------------------------------


def test_run_list_includes_sglang_and_freetoken_models() -> None:
    models = [
        SimpleNamespace(name="a", runtime=RuntimeName.SGLANG, path=None),
        SimpleNamespace(name="b", runtime=RuntimeName.FREETOKEN, path=None),
        SimpleNamespace(name="c", runtime=RuntimeName.OLLAMA, path=None),
    ]
    from llmctl.services.lanes import SwapTarget

    def target_for(m):
        return SwapTarget(key=m.name, kind="session", model=m.name, engine="x", upstream="")

    rows = session_rows_for(models, [], target_for=target_for, is_local=lambda m: True)
    assert [r.key for r in rows] == ["a", "b"]
