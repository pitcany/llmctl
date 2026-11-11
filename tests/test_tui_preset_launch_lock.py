"""The Presets screen's launch takes the GPU lock and records an operation.

``llmctl run`` and the Swap screen both hold the host-wide GPU operation
lock and write an operation record while they restart vLLM. The Presets
screen's launch did neither, so it could run next to another swap and left
no trace when the TUI died halfway.
"""

from __future__ import annotations

import pytest

from llmctl.config import load_settings
from llmctl.services import op_lock, operations
from llmctl.services.vllm_orchestrator import OrchestratorOptions
from llmctl.tui import screens_presets


class _State:
    def active_lanes(self) -> list:
        return []


class _Result:
    fleet_failed: list[str] = []
    restart = None
    rollback = None
    fleet_stopped: list[str] = []
    spec = type("S", (), {"port": 8003})()

    def __init__(self, ok: bool) -> None:
        self.ok = ok


@pytest.fixture
def seen(monkeypatch) -> dict:
    seen: dict = {}
    monkeypatch.setattr(screens_presets, "gather_state", lambda settings: _State())
    monkeypatch.setattr(
        screens_presets, "swap_to_vllm", lambda active, start, deps: (start(), None)
    )
    return seen


def _launch(monkeypatch, seen: dict, *, ok: bool):
    def _start(alias, **kwargs):
        seen["lock_held"] = op_lock.held_by_us()
        seen["operation_id"] = operations.current_operation_id()
        return _Result(ok)

    monkeypatch.setattr(screens_presets, "start_vllm_tp", _start)
    settings = load_settings()
    result = screens_presets.launch_preset(
        "ornith-35b", settings, OrchestratorOptions(dry_run=False)
    )
    store = operations.OperationStore(settings.database_url)
    return result, store.get(seen["operation_id"] or "")


def test_launch_holds_the_gpu_lock_while_vllm_starts(monkeypatch, seen) -> None:
    result, _ = _launch(monkeypatch, seen, ok=True)
    assert result.ok
    assert seen["lock_held"] is True
    assert op_lock.held_by_us() is False, "the lock must be released afterwards"


def test_successful_launch_is_recorded_as_completed(monkeypatch, seen) -> None:
    _, record = _launch(monkeypatch, seen, ok=True)
    assert record is not None, "no operation record was written"
    assert record.kind == "swap"
    assert record.target == "ornith-35b"
    assert record.owner.get("interface") == "tui"
    assert record.phase == operations.OperationPhase.COMPLETED.value


def test_failed_launch_is_recorded_as_failed(monkeypatch, seen) -> None:
    result, record = _launch(monkeypatch, seen, ok=False)
    assert not result.ok
    assert record is not None
    assert record.phase == operations.OperationPhase.FAILED.value
    assert operations.current_operation_id() is None
