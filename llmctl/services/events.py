"""Event/audit log service.

Lightweight helpers for writing and reading :class:`~llmctl.db.EventRecord`
entries. Services use :func:`log_event` to record lifecycle actions so the CLI,
API, and TUI can surface a unified audit trail.
"""

from __future__ import annotations

from typing import Any

from sqlmodel import Session as DBSession
from sqlmodel import col, select

from llmctl.db import EventLevel, EventRecord


def log_event(
    db: DBSession,
    level: EventLevel,
    category: str,
    message: str,
    *,
    session_id: str | None = None,
    model_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> EventRecord:
    """Persist an event record and return it."""
    record = EventRecord(
        level=level,
        category=category,
        message=message,
        session_id=session_id,
        model_id=model_id,
        data=data or {},
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_events(db: DBSession, limit: int = 50) -> list[EventRecord]:
    """Return the most recent event records, newest first."""
    statement = select(EventRecord).order_by(col(EventRecord.created_at).desc()).limit(limit)
    return list(db.exec(statement).all())
