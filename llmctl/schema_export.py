"""llmctl's vocabularies and models as one JSON document.

``llmctl schema`` and ``GET /schema`` serve :func:`build_schema` so a script
or an agent can read every state name, the session transition table, the
phase renderings and the API models from the running version instead of
from this source tree. The enums are the source of truth; nothing here is
hand-maintained.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from llmctl import __version__
from llmctl.db import (
    SESSION_ACTIVE_STATES,
    SESSION_TERMINAL_STATES,
    SESSION_TRANSITIONS,
    STATUS_HISTORY_LIMIT,
    BenchmarkKind,
    EventLevel,
    ModelStatus,
    RuntimeName,
    SessionKind,
    SessionStatus,
)
from llmctl.gpu_arg import RANKING_MODES, GpuMode
from llmctl.schemas import (
    HealthState,
    LaunchPlan,
    Model,
    Session,
    SessionStartRequest,
    StatusChange,
)
from llmctl.services.lanes import LaneState
from llmctl.services.operations import (
    PHASE_VIEWS,
    TERMINAL_PHASES,
    OperationPhase,
    Tone,
)
from llmctl.services.swap_jobs import SwapJobStatus

#: Every vocabulary the schema lists, by its class name.
ENUMS: tuple[type[StrEnum], ...] = (
    SessionStatus,
    OperationPhase,
    SwapJobStatus,
    LaneState,
    GpuMode,
    Tone,
    HealthState,
    RuntimeName,
    ModelStatus,
    SessionKind,
    BenchmarkKind,
    EventLevel,
)


def _values(enum: type[StrEnum]) -> list[str]:
    return [member.value for member in enum]


def _sorted(statuses: frozenset[SessionStatus]) -> list[str]:
    # Enum definition order, not alphabetical: it reads as the lifecycle.
    return [s.value for s in SessionStatus if s in statuses]


def build_schema() -> dict[str, Any]:
    """The schema document; see the module docstring for who reads it."""
    from llmctl.api.routes_swap import SwapJobDetail, SwapJobSummary, SwapList

    models = {
        model.__name__: model.model_json_schema()
        for model in (
            Session,
            StatusChange,
            SessionStartRequest,
            LaunchPlan,
            Model,
            SwapJobSummary,
            SwapJobDetail,
            SwapList,
        )
    }
    return {
        "llmctl": __version__,
        "doc": "docs/STATE-MACHINES.md",
        "enums": {enum.__name__: _values(enum) for enum in ENUMS},
        "sessions": {
            "active": _sorted(SESSION_ACTIVE_STATES),
            "terminal": _sorted(SESSION_TERMINAL_STATES),
            "transitions": {
                status.value: _sorted(SESSION_TRANSITIONS[status]) for status in SessionStatus
            },
            "history_limit": STATUS_HISTORY_LIMIT,
        },
        "operations": {
            "terminal_phases": [p.value for p in OperationPhase if p in TERMINAL_PHASES],
            "phase_views": {phase.value: view.as_dict() for phase, view in PHASE_VIEWS.items()},
        },
        "gpu_modes": {"ranking": [m.value for m in RANKING_MODES]},
        "models": models,
    }
