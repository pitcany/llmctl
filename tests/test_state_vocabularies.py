"""The named state vocabularies agree with their predicates, each other and the doc.

See ``docs/STATE-MACHINES.md``. These tests keep the lifecycle tables honest:
every member is documented, the active/terminal sets do not overlap, and the
swap-job projection of operation phases stays total.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmctl.db import (
    SESSION_ACTIVE_STATES,
    SESSION_TERMINAL_STATES,
    SESSION_TRANSITIONS,
    SessionStatus,
)
from llmctl.services.lanes import LaneState
from llmctl.services.operations import (
    TERMINAL_PHASE_VALUES,
    TERMINAL_PHASES,
    OperationPhase,
)
from llmctl.services.sessions import set_status
from llmctl.services.swap_jobs import SwapJobStatus

DOC = Path(__file__).resolve().parents[1] / "docs" / "STATE-MACHINES.md"


def test_session_transition_table_covers_every_status() -> None:
    assert set(SESSION_TRANSITIONS) == set(SessionStatus)
    for targets in SESSION_TRANSITIONS.values():
        assert targets <= set(SessionStatus)


def test_session_active_and_terminal_sets_are_disjoint_and_match_predicates() -> None:
    assert not (SESSION_ACTIVE_STATES & SESSION_TERMINAL_STATES)
    for status in SessionStatus:
        assert status.is_active == (status in SESSION_ACTIVE_STATES)
        assert status.is_terminal == (status in SESSION_TERMINAL_STATES)


def test_session_status_members_compare_equal_to_their_strings() -> None:
    # The CLI, TUI and API all rely on this; it is why the refactor changed no output.
    assert SessionStatus.RUNNING == "running"
    assert {"stopped", "failed"} == set(SESSION_TERMINAL_STATES)


def test_set_status_returns_old_and_logs_only_off_table_moves(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = SimpleNamespace(id="s1", status=SessionStatus.STARTING)
    with caplog.at_level(logging.DEBUG, logger="llmctl.services.sessions"):
        old = set_status(record, SessionStatus.RUNNING, why="test")
        assert old is SessionStatus.STARTING
        assert record.status is SessionStatus.RUNNING
        assert not caplog.records

        # RUNNING -> FAILED is not a move the code makes today.
        set_status(record, SessionStatus.FAILED, why="test")
        assert record.status is SessionStatus.FAILED
        assert len(caplog.records) == 1
        assert "running -> failed" in caplog.records[0].getMessage()


def test_every_operation_phase_has_a_view_that_agrees_with_it() -> None:
    from llmctl.services.operations import PHASE_VIEWS, Tone, describe_phase

    assert set(PHASE_VIEWS) == set(OperationPhase)
    for phase in OperationPhase:
        view = describe_phase(phase.value)
        assert view is PHASE_VIEWS[phase]
        assert view.phase == phase.value
        assert view.terminal == phase.is_terminal
        assert view.label and view.icon
        if phase.is_terminal and phase is not OperationPhase.COMPLETED:
            assert view.hint and view.tone is Tone.ERROR, "a bad ending needs a next step"
        if not phase.is_terminal:
            assert view.tone is Tone.ACTIVE
    assert describe_phase(OperationPhase.COMPLETED).hint == ""
    assert describe_phase("no-such-phase").tone is Tone.MUTED
    assert describe_phase(None).label == "unknown"
    assert describe_phase(OperationPhase.LOADING).as_dict()["tone"] == "active"


def test_operation_phase_terminal_predicate_matches_sets() -> None:
    for phase in OperationPhase:
        assert phase.is_terminal == (phase in TERMINAL_PHASES)
    assert TERMINAL_PHASE_VALUES == {p.value for p in TERMINAL_PHASES}


@pytest.mark.parametrize("phase", list(OperationPhase))
def test_swap_job_status_projection_is_total(phase: OperationPhase) -> None:
    status = SwapJobStatus.from_phase(phase.value)
    if phase is OperationPhase.COMPLETED:
        assert status is SwapJobStatus.SUCCEEDED
    elif phase.is_terminal:
        assert status is SwapJobStatus.FAILED
    else:
        assert status is SwapJobStatus.RUNNING


def test_swap_job_status_from_exit_code() -> None:
    assert SwapJobStatus.from_exit_code(0) is SwapJobStatus.SUCCEEDED
    assert SwapJobStatus.from_exit_code(2) is SwapJobStatus.FAILED
    assert SwapJobStatus.from_exit_code(None) is SwapJobStatus.FAILED
    assert SwapJobStatus.SUCCEEDED == "succeeded"


def test_gpu_mode_parse_results_are_members_equal_to_their_strings() -> None:
    from llmctl.gpu_arg import GPU_MODES, RANKING_MODES, GpuMode, parse_gpus

    assert GPU_MODES == RANKING_MODES
    assert set(RANKING_MODES) == set(GpuMode) - {GpuMode.EXPLICIT, GpuMode.CPU}
    assert parse_gpus("cpu") == ([], GpuMode.CPU, True)
    assert parse_gpus("0,1")[1] is GpuMode.EXPLICIT
    assert parse_gpus("least-used") == ([], "least-used", False)


def test_lane_state_idle_is_blank() -> None:
    assert LaneState.IDLE == ""
    assert not LaneState.IDLE
    assert str(LaneState.WEIGHTS_GONE) == "weights gone"


def test_doc_names_every_lifecycle_member() -> None:
    text = DOC.read_text(encoding="utf-8")
    from llmctl.gpu_arg import GpuMode

    for enum in (SessionStatus, OperationPhase, SwapJobStatus, LaneState, GpuMode):
        for member in enum:
            needle = f"`{member.value}`" if member.value else "idle"
            assert needle in text, f"{enum.__name__}.{member.name} missing from {DOC.name}"
