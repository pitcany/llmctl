"""Phase-1 adopt schema: SessionKind enum, new SessionRecord columns, and migration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import inspect, text
from sqlmodel import Session, select

from llmctl.db import (
    RuntimeName,
    SessionKind,
    SessionRecord,
    SessionStatus,
    apply_migrations,
    get_engine,
    init_db,
)


def _make_db_url(tmp_path: Path, name: str = "test.sqlite3") -> str:
    return f"sqlite:///{tmp_path / name}"


def test_session_kind_enum_values() -> None:
    """Sanity: enum exposes exactly OWNED and ADOPTED."""
    assert SessionKind.OWNED.value == "owned"
    assert SessionKind.ADOPTED.value == "adopted"
    assert set(SessionKind) == {SessionKind.OWNED, SessionKind.ADOPTED}


def test_new_db_has_adopt_columns(tmp_path: Path) -> None:
    """A fresh init_db should create the new columns out of the gate."""
    db_url = _make_db_url(tmp_path)
    init_db(db_url)
    engine = get_engine(db_url)
    cols = {col["name"] for col in inspect(engine).get_columns("sessions")}
    assert {"kind", "systemd_unit", "served_name", "adopted_at"}.issubset(cols)


def test_apply_migrations_adds_columns_to_pre_phase1_db(tmp_path: Path) -> None:
    """A pre-Phase-1 sessions table (no new columns) gains them via apply_migrations."""
    db_url = _make_db_url(tmp_path, "pre_phase1.sqlite3")
    engine = get_engine(db_url)
    legacy_ddl = """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            model_id TEXT,
            profile_id TEXT,
            runtime TEXT NOT NULL,
            status TEXT NOT NULL,
            pid INTEGER,
            port INTEGER,
            endpoint_url TEXT,
            health_url TEXT,
            log_path TEXT,
            gpu_ids TEXT,
            env TEXT,
            command TEXT,
            launch_plan TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            stopped_at TEXT,
            updated_at TEXT NOT NULL
        )
    """
    with engine.begin() as conn:
        conn.execute(text(legacy_ddl))
        conn.execute(
            text(
                "INSERT INTO sessions (id, runtime, status, created_at, updated_at) "
                "VALUES ('legacy-1', 'vllm', 'running', '2026-05-30T00:00:00+00:00', "
                "'2026-05-30T00:00:00+00:00')"
            )
        )

    pre_cols = {col["name"] for col in inspect(engine).get_columns("sessions")}
    assert "kind" not in pre_cols
    assert "systemd_unit" not in pre_cols
    assert "served_name" not in pre_cols
    assert "adopted_at" not in pre_cols

    apply_migrations(engine)

    post_cols = {col["name"] for col in inspect(engine).get_columns("sessions")}
    assert {"kind", "systemd_unit", "served_name", "adopted_at"}.issubset(post_cols)

    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT kind, systemd_unit, served_name, adopted_at "
                "FROM sessions WHERE id = 'legacy-1'"
            )
        ).one()
    assert row == (None, None, None, None)


def test_legacy_row_reads_back_as_owned(tmp_path: Path) -> None:
    """After migration, a NULL kind should not fail to load; SQLModel surfaces it as None."""
    db_url = _make_db_url(tmp_path, "legacy_read.sqlite3")
    engine = get_engine(db_url)
    init_db(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions "
                "(id, runtime, status, kind, gpu_ids, env, command, launch_plan, "
                "created_at, updated_at) "
                "VALUES ('legacy-2', 'VLLM', 'RUNNING', NULL, '[]', '{}', '[]', '{}', "
                "'2026-05-30T00:00:00+00:00', '2026-05-30T00:00:00+00:00')"
            )
        )
    with Session(engine) as session:
        record = session.exec(select(SessionRecord).where(SessionRecord.id == "legacy-2")).one()
    assert record.kind is None


def test_owned_session_defaults(tmp_path: Path) -> None:
    """Newly created records default to kind=OWNED with nullable adopt fields empty."""
    db_url = _make_db_url(tmp_path, "owned.sqlite3")
    init_db(db_url)
    engine = get_engine(db_url)
    record = SessionRecord(runtime=RuntimeName.VLLM, status=SessionStatus.PLANNED)
    with Session(engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        reloaded = session.exec(select(SessionRecord).where(SessionRecord.id == record.id)).one()
    assert reloaded.kind == SessionKind.OWNED
    assert reloaded.systemd_unit is None
    assert reloaded.served_name is None
    assert reloaded.adopted_at is None


def test_adopted_session_round_trip(tmp_path: Path) -> None:
    """An ADOPTED record persists and reloads with all four adopt fields intact."""
    db_url = _make_db_url(tmp_path, "adopted.sqlite3")
    init_db(db_url)
    engine = get_engine(db_url)
    adopted_at = datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC)
    record = SessionRecord(
        runtime=RuntimeName.VLLM,
        status=SessionStatus.RUNNING,
        kind=SessionKind.ADOPTED,
        endpoint_url="http://127.0.0.1:8003",
        port=8003,
        systemd_unit="vllm-tp.service",
        served_name="llama-3.3-70b",
        adopted_at=adopted_at,
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        reloaded = session.exec(select(SessionRecord).where(SessionRecord.id == record.id)).one()
    assert reloaded.kind == SessionKind.ADOPTED
    assert reloaded.systemd_unit == "vllm-tp.service"
    assert reloaded.served_name == "llama-3.3-70b"
    assert reloaded.adopted_at is not None
    assert reloaded.adopted_at.replace(tzinfo=UTC) == adopted_at
    assert reloaded.endpoint_url == "http://127.0.0.1:8003"
    assert reloaded.port == 8003
