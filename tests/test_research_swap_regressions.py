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
