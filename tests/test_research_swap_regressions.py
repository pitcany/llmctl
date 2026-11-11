"""Exercise operation tracking through the actual CLI and TUI callers."""

from __future__ import annotations

import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console
from sqlmodel import Session, select

from llmctl import cli_swap
from llmctl.config import Settings
from llmctl.db import OperationRecord, get_engine
from llmctl.services.op_lock import LockBusy, gpu_operation_lock
from llmctl.services.operations import OperationStore, cli_operation_guard
from llmctl.tui import screens_swap


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings()
    settings.database.url = f"sqlite:///{tmp_path / 'operations.db'}"
    monkeypatch.setattr(screens_swap, "load_settings", lambda: settings)
    monkeypatch.setattr(cli_swap, "load_settings", lambda: settings)
    return settings


def _only_operation(settings: Settings) -> OperationRecord:
    with Session(get_engine(settings.database_url)) as db:
        return db.exec(select(OperationRecord)).one()


def test_tui_keeps_exclusive_ownership_between_swap_steps(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A competing thread cannot enter between pause and replacement launch."""
    blocked: list[bool] = []

    def compete() -> None:
        try:
            with gpu_operation_lock(purpose="competing stop", interface="cli", timeout=0):
                blocked.append(False)
        except LockBusy:
            blocked.append(True)

    def launch_after_pause(*args: object, **kwargs: object) -> tuple[object, object]:
        thread = threading.Thread(target=compete)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        return SimpleNamespace(ok=True, companions_failed=[]), SimpleNamespace()

    state = SimpleNamespace(sessions_in_the_way=lambda _: [], lanes_in_the_way=lambda _: [])
    target = SimpleNamespace(kind="vllm", key="demo", model="demo", tp=1)
    monkeypatch.setattr(screens_swap, "place", lambda s, t, tp: t)
    monkeypatch.setattr(screens_swap, "swap_to_vllm", launch_after_pause)
    monkeypatch.setattr(screens_swap, "record_load", lambda *args: None)
    assert screens_swap.run_swap(state, target)[0]
    assert blocked == [True]
    assert _only_operation(settings).phase == "completed"


@pytest.mark.parametrize("exception", [RuntimeError("load failed"), KeyboardInterrupt()])
def test_tui_exception_finalizes_operation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, exception: BaseException,
) -> None:
    def crash(*args: object) -> None:
        raise exception

    monkeypatch.setattr(screens_swap, "_do_swap", crash)
    with pytest.raises(type(exception)):
        screens_swap.run_swap(None, SimpleNamespace(key="demo", model="demo"))
    assert OperationStore(settings.database_url).unfinished() == []
    record = _only_operation(settings)
    assert record.phase == "interrupted"
    assert record.owner["interface"] == "tui"
    assert "guidance" in record.recovery
    with gpu_operation_lock(purpose="next run", interface="cli", timeout=0):
        pass


@pytest.mark.parametrize("caller", ["session", "lane", "tui"])
def test_real_swap_callers_persist_failed_restoration(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, caller: str,
) -> None:
    failed = SimpleNamespace(
        ok=False, error="load failed", restored=[], restore_failed=["previous-model"],
    )
    target = SimpleNamespace(kind="session", key="demo", model="demo", lane=object())
    if caller == "tui":
        monkeypatch.setattr(screens_swap, "run_session_swap", lambda *a, **k: failed)
        assert not screens_swap.run_swap(None, target)[0]
    else:
        monkeypatch.setattr(cli_swap, "run_session_swap_service", lambda *a, **k: failed)
        monkeypatch.setattr(cli_swap, "swap_to_lane", lambda *a, **k: failed)
        state = SimpleNamespace(
            sessions_in_the_way=lambda *a: [], lanes_in_the_way=lambda *a: [],
            vllm_in_the_way=lambda *a: False, vllm_unit="vllm-test",
        )
        with pytest.raises(typer.Exit):
            with cli_operation_guard(
                db_url=settings.database_url, kind="swap", target="demo", model="demo",
            ):
                run = cli_swap.run_session_swap if caller == "session" else cli_swap.run_lane_swap
                run(state, target, Console(file=StringIO()))
    record = _only_operation(settings)
    assert record.phase == "needs_attention"
    assert record.recovery["not_restored"] == ["previous-model"]


def test_tui_interruption_during_restoration_needs_attention(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llmctl.services.operations import OperationPhase, mark_phase, note_restore_failure

    def interrupted_restore(*args: object) -> None:
        mark_phase(OperationPhase.RESTORING)
        note_restore_failure(["previous-model"])
        raise KeyboardInterrupt()

    monkeypatch.setattr(screens_swap, "_do_swap", interrupted_restore)
    with pytest.raises(KeyboardInterrupt):
        screens_swap.run_swap(None, SimpleNamespace(key="demo", model="demo"))
    record = _only_operation(settings)
    assert record.phase == "needs_attention"
    assert record.recovery["not_restored"] == ["previous-model"]


def test_busy_tui_swap_does_not_create_an_unfinished_record(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with gpu_operation_lock(purpose="other run", interface="cli"):
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(LockBusy):
            screens_swap.run_swap(None, SimpleNamespace(key="demo", model="demo"))
        assert OperationStore(settings.database_url).unfinished() == []
    finally:
        release.set()
        thread.join(timeout=2)


def _vllm_swap_with_a_stopped_lane(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, start: object,
) -> tuple[SimpleNamespace, SimpleNamespace, list[str]]:
    """A vLLM target with one lane in the way; the real ``swap_to_vllm`` runs."""
    from llmctl.services import lane_swap

    lane = SimpleNamespace(unit="flash.service", name="flash")
    resumed: list[str] = []

    def restore(stopped, *, result, **kwargs):  # type: ignore[no-untyped-def]
        result.restored += [item.unit for item in stopped]

    def resume(paused, **kwargs):  # type: ignore[no-untyped-def]
        resumed.extend(paused)
        return SimpleNamespace(restored=list(paused), restore_failed=[])

    settings.managed_units.fleet = None
    monkeypatch.setattr(lane_swap, "stop_user_unit", lambda unit, deps: True)
    monkeypatch.setattr(lane_swap, "restore_previous", restore)
    monkeypatch.setattr(screens_swap, "place", lambda s, t, tp: t)
    monkeypatch.setattr(screens_swap, "engine_config", lambda s, t: "main")
    monkeypatch.setattr(screens_swap, "cuda_devices", lambda t: None)
    monkeypatch.setattr(screens_swap, "pause_for_swap", lambda *a, **k: (["llama-8080"], None))
    monkeypatch.setattr(screens_swap, "resume_after_failed_swap", resume)
    monkeypatch.setattr(screens_swap, "start_vllm_tp", start)
    state = SimpleNamespace(lanes_in_the_way=lambda _: [lane])
    target = SimpleNamespace(kind="vllm", key="demo", model="demo", tp=1)
    return state, target, resumed


def test_tui_launch_that_raises_restores_lanes_and_sessions(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising launch is a failed swap, not an interrupted one."""
    from llmctl.services.vllm_orchestrator import RollbackReport, attach_rollback

    def start(key: str, **kwargs: object) -> object:
        exc = FileNotFoundError("weights are gone")
        attach_rollback(
            exc,
            RollbackReport(
                env_restored=False, env_path=Path("/etc/vllm-tp.env"), restart_failed=["ollama"]
            ),
        )
        raise exc

    state, target, resumed = _vllm_swap_with_a_stopped_lane(settings, monkeypatch, start)
    ok, message = screens_swap.run_swap(state, target)

    assert not ok
    assert "FileNotFoundError: weights are gone" in message
    assert "Started again: flash.service, llama-8080" in message
    assert "Still stopped: ollama, env file /etc/vllm-tp.env" in message
    assert resumed == ["llama-8080"]
    assert _only_operation(settings).phase == "needs_attention"


def test_tui_failed_launch_names_what_its_rollback_left_down(
    settings: Settings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llmctl.services.vllm_orchestrator import RollbackReport

    def start(key: str, **kwargs: object) -> object:
        return SimpleNamespace(
            ok=False,
            restart=SimpleNamespace(error="health check timed out"),
            rollback=RollbackReport(restart_failed=["ollama"]),
        )

    state, target, _ = _vllm_swap_with_a_stopped_lane(settings, monkeypatch, start)
    ok, message = screens_swap.run_swap(state, target)

    assert not ok
    assert "health check timed out" in message
    assert "Still stopped: ollama" in message
