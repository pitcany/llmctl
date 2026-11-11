"""The Presets screen's launch takes the GPU lock and records an operation.

``llmctl run`` and the Swap screen both hold the host-wide GPU operation
lock and write an operation record while they restart vLLM. The Presets
screen's launch did neither, so it could run next to another swap and left
no trace when the TUI died halfway.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmctl.config import load_settings
from llmctl.services import op_lock, operations
from llmctl.services.lane_swap import LaneSwapResult
from llmctl.services.vllm_orchestrator import OrchestratorOptions, RollbackReport
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


# --- What the record says about a launch that did not go cleanly ---------


def _record_for(seen: dict):
    store = operations.OperationStore(load_settings().database_url)
    return store.get(seen["operation_id"])


def _launch_with(monkeypatch, seen: dict, result, lanes=None):
    """Launch with a canned vLLM result; ``result=None`` means a lane refused to stop."""

    def _swap(active, start, deps):
        seen["operation_id"] = operations.current_operation_id()
        return (start() if result is not None else None), lanes

    monkeypatch.setattr(screens_presets, "swap_to_vllm", _swap)
    monkeypatch.setattr(screens_presets, "start_vllm_tp", lambda alias, **kw: result)
    launch = screens_presets.launch_preset(
        "ornith-35b", load_settings(), OrchestratorOptions(dry_run=False)
    )
    return launch, _record_for(seen)


def test_phase_is_loading_while_vllm_starts(monkeypatch, seen) -> None:
    def _start(alias, **kwargs):
        seen["operation_id"] = operations.current_operation_id()
        seen["phase"] = _record_for(seen).phase
        return _Result(True)

    monkeypatch.setattr(screens_presets, "start_vllm_tp", _start)
    screens_presets.launch_preset(
        "ornith-35b", load_settings(), OrchestratorOptions(dry_run=False)
    )
    assert seen["phase"] == operations.OperationPhase.LOADING.value


def test_refused_lane_stop_is_a_plain_failure_when_lanes_came_back(monkeypatch, seen) -> None:
    lanes = LaneSwapResult(ok=False, error="could not stop a.service", restored=["b.service"])
    launch, record = _launch_with(monkeypatch, seen, None, lanes)
    assert not launch.ok
    assert launch.result is None
    assert record.phase == operations.OperationPhase.FAILED.value
    assert "could not stop a.service" in record.error
    assert "not_restored" not in record.recovery


def test_refused_lane_stop_records_the_lane_that_stayed_down(monkeypatch, seen) -> None:
    lanes = LaneSwapResult(
        ok=False, error="could not stop a.service", restore_failed=["b.service"]
    )
    _, record = _launch_with(monkeypatch, seen, None, lanes)
    assert record.phase == operations.OperationPhase.NEEDS_ATTENTION.value
    assert record.recovery["not_restored"] == ["b.service"]


def test_failed_launch_records_what_rollback_could_not_restore(monkeypatch, seen) -> None:
    result = _Result(False)
    result.rollback = RollbackReport(
        env_restored=False, env_path=Path("/etc/vllm/tp.env"), restart_failed=["ollama"]
    )
    lanes = LaneSwapResult(ok=False, restore_failed=["b.service"])
    _, record = _launch_with(monkeypatch, seen, result, lanes)
    assert record.phase == operations.OperationPhase.NEEDS_ATTENTION.value
    assert record.recovery["not_restored"] == [
        "ollama", "env file /etc/vllm/tp.env", "b.service",
    ]


def test_ready_launch_records_a_companion_that_was_not_started_again(monkeypatch, seen) -> None:
    result = _Result(True)
    result.companions_failed = ["comfyui.service"]
    launch, record = _launch_with(monkeypatch, seen, result, LaneSwapResult(ok=True))
    assert launch.ok
    assert record.phase == operations.OperationPhase.COMPLETED.value
    assert record.recovery["not_restored"] == ["comfyui.service"]


# --- What the operator is told ------------------------------------------


def _notify(launch) -> tuple[str, dict]:
    said: list[tuple[str, dict]] = []
    screen = type("Screen", (), {})()
    screen.app = type("App", (), {"notify": lambda self, msg, **kw: said.append((msg, kw))})()
    screens_presets.PresetsScreen._after_launch(screen, "vLLM TP: x", launch)
    assert len(said) == 1
    return said[0]


def test_ready_notification_names_the_companion_that_stayed_down() -> None:
    result = _Result(True)
    result.companions_failed = ["comfyui.service"]
    message, options = _notify(screens_presets.PresetLaunch(result, LaneSwapResult(ok=True)))
    assert "ready" in message
    assert "comfyui.service" in message
    assert options["severity"] == "warning"


def test_refused_lane_stop_notification_says_vllm_was_not_started() -> None:
    lanes = LaneSwapResult(
        ok=False, error="could not stop a.service",
        restored=["b.service"], restore_failed=["c.service"],
    )
    message, options = _notify(screens_presets.PresetLaunch(None, lanes))
    assert "could not stop a.service; vLLM not started" in message
    assert "Started again: b.service" in message
    assert "Still stopped: c.service" in message
    assert options["severity"] == "error"


# --- A launch that raises (a preset whose weights are gone) ----------------


def _launch_that_raises(monkeypatch, seen: dict, lanes: LaneSwapResult):
    """Run the real ``swap_to_vllm`` contract: a failed start restores the lanes."""

    def _swap(active, start, deps):
        seen["operation_id"] = operations.current_operation_id()
        result = start()
        seen["restored"] = not getattr(result, "ok", False)
        return result, lanes

    def _start(alias, **kwargs):
        raise FileNotFoundError("preset model path /models/gone is missing")

    monkeypatch.setattr(screens_presets, "swap_to_vllm", _swap)
    monkeypatch.setattr(screens_presets, "start_vllm_tp", _start)
    launch = screens_presets.launch_preset(
        "ornith-35b", load_settings(), OrchestratorOptions(dry_run=False)
    )
    return launch, _record_for(seen)


def test_launch_that_raises_restores_the_lanes_and_is_recorded_as_failed(
    monkeypatch, seen
) -> None:
    lanes = LaneSwapResult(ok=False, restored=["freetoken.service"])
    launch, record = _launch_that_raises(monkeypatch, seen, lanes)
    assert seen["restored"] is True  # swap_to_vllm saw a failed start, not an exception
    assert not launch.ok
    assert "/models/gone is missing" in launch.reason
    assert record.phase == operations.OperationPhase.FAILED.value
    assert "/models/gone is missing" in record.error


def test_launch_that_raises_records_the_lane_that_stayed_down(monkeypatch, seen) -> None:
    lanes = LaneSwapResult(ok=False, restore_failed=["freetoken.service"])
    launch, record = _launch_that_raises(monkeypatch, seen, lanes)
    assert launch.not_restored == ["freetoken.service"]
    assert record.phase == operations.OperationPhase.NEEDS_ATTENTION.value
    assert record.recovery["not_restored"] == ["freetoken.service"]
    assert "/models/gone is missing" in record.error


def test_notification_for_a_launch_that_raised_gives_the_error() -> None:
    lanes = LaneSwapResult(ok=False, restored=["b.service"], restore_failed=["c.service"])
    launch = screens_presets.PresetLaunch(None, lanes, error="FileNotFoundError: gone")
    message, options = _notify(launch)
    assert "FileNotFoundError: gone" in message
    assert "vLLM not started" not in message
    assert "Still stopped: c.service" in message
    assert options["severity"] == "error"
