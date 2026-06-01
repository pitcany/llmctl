"""Database models and helpers.

The schema uses SQLModel on top of SQLite. Helpers are intentionally lightweight
so tests and future services can create isolated databases easily.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Column, inspect, text
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine

from llmctl.config import load_settings


class RuntimeName(StrEnum):
    """Supported runtime adapter identifiers."""

    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    PYTHON_SCRIPT = "python_script"


class ModelStatus(StrEnum):
    """Model registry status."""

    DISCOVERED = "discovered"
    REGISTERED = "registered"
    MISSING = "missing"
    DELETED = "deleted"


class SessionStatus(StrEnum):
    """Session lifecycle state."""

    PLANNED = "planned"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"


class EventLevel(StrEnum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(UTC)


def new_id() -> str:
    """Return a stable string UUID for primary keys."""
    return str(uuid4())


class ModelRecord(SQLModel, table=True):
    """Registered or discovered local model."""

    __tablename__ = "models"

    id: str = Field(default_factory=new_id, primary_key=True)
    name: str = Field(index=True)
    runtime: RuntimeName = Field(index=True)
    source: str | None = Field(
        default=None,
        description="Path, runtime ID, or remote-like source URI.",
    )
    path: str | None = Field(default=None, description="Optional local filesystem path.")
    format: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    estimated_vram_gb: float | None = None
    max_context: int | None = None
    parameter_count: int | None = Field(
        default=None,
        description="Approximate parameter count (raw integer, not billions).",
    )
    notes: str | None = None
    default_profile_id: str | None = Field(
        default=None, foreign_key="profiles.id", index=True
    )
    # ``active`` is independent of ``status``: a registered model can be
    # disabled without losing its row. Nullable for forward-compat with
    # databases created before this column existed — the service layer
    # treats NULL as active.
    active: bool | None = Field(default=True, index=True)
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    metadata_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: ModelStatus = Field(default=ModelStatus.REGISTERED, index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow)


class ProfileRecord(SQLModel, table=True):
    """Reusable launch/runtime profile."""

    __tablename__ = "profiles"

    id: str = Field(default_factory=new_id, primary_key=True)
    name: str = Field(index=True, unique=True)
    runtime: RuntimeName = Field(index=True)
    description: str | None = None
    # Promoted top-level launch knobs. ``parameters`` remains the canonical
    # store for everything else (backward compatible with profiles.yaml).
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    dtype: str | None = None
    quantization: str | None = None
    extra_args: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    environment_variables: dict[str, str] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    scheduler_preferences: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    parameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    gpu_policy: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    safety: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow)


class SessionRecord(SQLModel, table=True):
    """A launched or planned runtime session."""

    __tablename__ = "sessions"

    id: str = Field(default_factory=new_id, primary_key=True)
    model_id: str | None = Field(default=None, foreign_key="models.id", index=True)
    profile_id: str | None = Field(default=None, foreign_key="profiles.id", index=True)
    runtime: RuntimeName = Field(index=True)
    status: SessionStatus = Field(default=SessionStatus.PLANNED, index=True)
    pid: int | None = Field(default=None, index=True)
    port: int | None = Field(default=None, index=True)
    endpoint_url: str | None = None
    health_url: str | None = None
    log_path: str | None = None
    gpu_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    env: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    command: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    launch_plan: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow, index=True)
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)


class BenchmarkRecord(SQLModel, table=True):
    """Benchmark result history."""

    __tablename__ = "benchmarks"

    id: str = Field(default_factory=new_id, primary_key=True)
    model_id: str | None = Field(default=None, foreign_key="models.id", index=True)
    session_id: str | None = Field(default=None, foreign_key="sessions.id", index=True)
    profile_id: str | None = Field(default=None, foreign_key="profiles.id", index=True)
    name: str = Field(index=True)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    tokens_per_second: float | None = None
    ttft_ms: float | None = None
    gpu_snapshot: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    parameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    samples: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    success: bool = Field(default=True, index=True)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow, index=True)


class EventRecord(SQLModel, table=True):
    """Audit/event/log record."""

    __tablename__ = "events"

    id: str = Field(default_factory=new_id, primary_key=True)
    level: EventLevel = Field(default=EventLevel.INFO, index=True)
    category: str = Field(index=True)
    message: str
    session_id: str | None = Field(default=None, foreign_key="sessions.id", index=True)
    model_id: str | None = Field(default=None, foreign_key="models.id", index=True)
    data: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utcnow, index=True)


def get_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine for the configured database URL."""
    url = database_url or load_settings().database_url
    if url.startswith("sqlite") and "/" in url:
        db_path = url.replace("sqlite:///", "")
        if db_path and db_path != ":memory:":
            Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine_kwargs: dict[str, Any] = {"connect_args": connect_args}
    if url in {"sqlite:///:memory:", "sqlite://"}:
        engine_kwargs["poolclass"] = StaticPool
    return create_engine(url, **engine_kwargs)


def apply_migrations(engine) -> None:
    """Add any model-declared columns missing from existing tables.

    SQLModel's ``create_all`` creates new tables but never alters existing ones.
    This keeps older SQLite databases forward-compatible by adding new nullable
    columns in place, avoiding destructive recreation on upgrade.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table_name, table in SQLModel.metadata.tables.items():
        if table_name not in existing_tables:
            continue
        present = {column["name"] for column in inspector.get_columns(table_name)}
        for column in table.columns:
            if column.name in present:
                continue
            ddl_type = column.type.compile(dialect=engine.dialect)
            with engine.begin() as conn:
                conn.execute(
                    text(f'ALTER TABLE "{table_name}" ADD COLUMN "{column.name}" {ddl_type}')
                )


def init_db(database_url: str | None = None) -> None:
    """Create all tables for the configured database and apply migrations."""
    engine = get_engine(database_url)
    SQLModel.metadata.create_all(engine)
    apply_migrations(engine)


def get_session(database_url: str | None = None) -> Generator[Session, None, None]:
    """Yield a database session suitable for FastAPI dependencies."""
    engine = get_engine(database_url)
    with Session(engine) as session:
        yield session
