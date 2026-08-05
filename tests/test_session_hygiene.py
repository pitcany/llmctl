"""Session hygiene: dry-run PLANNED rows must be clearable, adopt errors actionable.

A ``llmctl start --dry-run`` records a PLANNED session (with ``dry_run: true``
in its stored plan) and never launches a process, so nothing ever transitions
the row — historically it blocked ``llmctl adopt`` on that endpoint
indefinitely. These tests pin the fix: ``cleanup --remove-stale`` purges such
rows (and only such rows), and the adopt refusal names the blocking session,
its status, and the exact command that clears it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlmodel import Session, select

from llmctl.db import RuntimeName, SessionKind, SessionRecord, SessionStatus, get_engine, init_db
from llmctl.services.sessions import AdoptError, SessionService


def _make_service(
    tmp_path: Path,
    probe: Callable[[str, float], list[str] | None],
) -> tuple[Session, SessionService]:
    """Wire up an isolated DB-backed SessionService with the supplied probe."""
    url = f"sqlite:///{tmp_path / 'hygiene.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    service = SessionService(db, probe=probe)
    return db, service


def _planned_row(endpoint: str, *, dry_run: bool | None) -> SessionRecord:
    """Build a PLANNED OWNED row; ``dry_run=None`` stores no launch plan at all."""
    plan = None if dry_run is None else {"dry_run": dry_run}
    return SessionRecord(
        runtime=RuntimeName.VLLM,
        status=SessionStatus.PLANNED,
        kind=SessionKind.OWNED,
        endpoint_url=endpoint,
        port=8000,
        launch_plan=plan,
    )


def test_cleanup_purges_dry_run_planned_rows(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        db.add(_planned_row("http://127.0.0.1:8000", dry_run=True))
        db.commit()
        report = service.cleanup(remove_stale=True)
        assert report["stale_removed"] == 1
        assert db.exec(select(SessionRecord)).all() == []
    finally:
        db.close()


def test_cleanup_keeps_real_planned_rows(tmp_path: Path) -> None:
    """A real start passes through PLANNED with dry_run=false — never purged."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        db.add(_planned_row("http://127.0.0.1:8000", dry_run=False))
        db.commit()
        report = service.cleanup(remove_stale=True)
        assert report["stale_removed"] == 0
        rows = db.exec(select(SessionRecord)).all()
        assert len(rows) == 1
        assert rows[0].status == SessionStatus.PLANNED
    finally:
        db.close()


def test_cleanup_keeps_planned_rows_without_a_plan(tmp_path: Path) -> None:
    """No stored plan means we cannot prove it was a dry-run — leave it alone."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        db.add(_planned_row("http://127.0.0.1:8000", dry_run=None))
        db.commit()
        report = service.cleanup(remove_stale=True)
        assert report["stale_removed"] == 0
        assert len(db.exec(select(SessionRecord)).all()) == 1
    finally:
        db.close()


def test_cleanup_without_remove_stale_keeps_dry_run_planned(tmp_path: Path) -> None:
    """The purge stays opt-in: plain ``cleanup`` only reports, never deletes."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        db.add(_planned_row("http://127.0.0.1:8000", dry_run=True))
        db.commit()
        report = service.cleanup(remove_stale=False)
        assert report["stale_removed"] == 0
        assert len(db.exec(select(SessionRecord)).all()) == 1
    finally:
        db.close()


def test_adopt_blocked_by_dry_run_plan_names_cause_and_fix(tmp_path: Path) -> None:
    """Reproduce the historical block end-to-end, then clear it and re-adopt.

    This is the exact failure that once blocked ``llmctl adopt`` on
    :8000 for months: a dry-run PLANNED row reserving the endpoint with an
    error that never explained itself.
    """
    db, service = _make_service(tmp_path, lambda u, _t: ["deepseek-v4-flash"])
    try:
        blocker = _planned_row("http://127.0.0.1:8000", dry_run=True)
        db.add(blocker)
        db.commit()

        with pytest.raises(AdoptError) as excinfo:
            service.adopt(RuntimeName.LLAMA_CPP, "http://127.0.0.1:8000")
        message = str(excinfo.value)
        assert blocker.id in message
        assert "status=planned" in message
        assert "dry-run" in message
        assert "llmctl cleanup --remove-stale" in message

        service.cleanup(remove_stale=True)
        session = service.adopt(RuntimeName.LLAMA_CPP, "http://127.0.0.1:8000")
        assert session.status == SessionStatus.RUNNING
        assert session.kind == SessionKind.ADOPTED
    finally:
        db.close()


def test_adopt_blocked_by_owned_active_names_stop_command(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        running = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.RUNNING,
            kind=SessionKind.OWNED,
            endpoint_url="http://127.0.0.1:8000",
            port=8000,
            pid=12345,
        )
        db.add(running)
        db.commit()
        with pytest.raises(AdoptError) as excinfo:
            service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8000")
        message = str(excinfo.value)
        assert running.id in message
        assert f"llmctl stop {running.id}" in message
    finally:
        db.close()


def test_adopt_blocked_by_adopted_active_names_detach_command(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        adopted = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.RUNNING,
            kind=SessionKind.ADOPTED,
            endpoint_url="http://127.0.0.1:8000",
            port=8000,
        )
        db.add(adopted)
        db.commit()
        with pytest.raises(AdoptError) as excinfo:
            service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8000")
        assert f"llmctl detach {adopted.id}" in str(excinfo.value)
    finally:
        db.close()
