"""Database models and helpers.

The schema uses SQLModel on top of SQLite. Helpers are intentionally lightweight
so tests and future services can create isolated databases easily.
"""

from __future__ import annotations

import threading
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
    """Supported runtime adapter identifiers.

    ``OPENAI`` is adopt-only: a generic OpenAI-compatible endpoint llmctl
    routes to but cannot launch, stop, or inventory (no adapter, no launch
    plan). Everything else is a local runtime with a full adapter.
    """

    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    LMSTUDIO = "lmstudio"
    OLLAMA = "ollama"
    PYTHON_SCRIPT = "python_script"
    SGLANG = "sglang"
    FREETOKEN = "freetoken"
    STRATA = "strata"
    OPENAI = "openai"


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
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @property
    def is_active(self) -> bool:
        """A process or endpoint llmctl still expects to answer (reconcile probes it)."""
        return self in SESSION_ACTIVE_STATES

    @property
    def is_terminal(self) -> bool:
        """Nothing further happens to the row unless ``restart`` re-plans it."""
        return self in SESSION_TERMINAL_STATES


#: Statuses with a live process or endpoint: probed by reconcile, routed by the gateway.
SESSION_ACTIVE_STATES: frozenset[SessionStatus] = frozenset(
    {SessionStatus.STARTING, SessionStatus.RUNNING, SessionStatus.DEGRADED}
)

#: Statuses ``cleanup`` may purge; only ``restart`` leaves them.
SESSION_TERMINAL_STATES: frozenset[SessionStatus] = frozenset(
    {SessionStatus.STOPPED, SessionStatus.FAILED}
)

#: The session lifecycle as data: status -> the statuses the code moves it to.
#:
#: ``SessionService`` writes every status through ``set_status``, which logs (never
#: refuses) a move outside this table, so a new path shows up in the debug log
#: instead of silently widening the machine. ``docs/STATE-MACHINES.md`` renders it.
#:
#: ``STOPPING`` is the window between a stop (or restart) being requested and
#: the process or unit actually going away; a stop that cannot run puts the
#: row back where it was. ``UNKNOWN`` is the escape hatch for rows reconcile
#: cannot classify.
SESSION_TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.PLANNED: frozenset(
        {
            SessionStatus.STARTING,
            SessionStatus.STOPPING,
            SessionStatus.STOPPED,
            SessionStatus.FAILED,
        }
    ),
    SessionStatus.STARTING: frozenset(
        {
            SessionStatus.RUNNING,
            SessionStatus.DEGRADED,
            SessionStatus.STOPPING,
            SessionStatus.STOPPED,
            SessionStatus.FAILED,
            SessionStatus.PLANNED,
        }
    ),
    SessionStatus.RUNNING: frozenset(
        {
            SessionStatus.DEGRADED,
            SessionStatus.STOPPING,
            SessionStatus.STOPPED,
            SessionStatus.STARTING,
            SessionStatus.PLANNED,
        }
    ),
    SessionStatus.DEGRADED: frozenset(
        {
            SessionStatus.RUNNING,
            SessionStatus.STOPPING,
            SessionStatus.STOPPED,
            SessionStatus.STARTING,
            SessionStatus.PLANNED,
        }
    ),
    SessionStatus.STOPPING: frozenset(
        {
            SessionStatus.STOPPED,
            SessionStatus.FAILED,
            SessionStatus.DEGRADED,
            SessionStatus.PLANNED,
            # A stop that could not run restores the status it found.
            SessionStatus.RUNNING,
            SessionStatus.STARTING,
        }
    ),
    SessionStatus.STOPPED: frozenset(
        {SessionStatus.RUNNING, SessionStatus.STARTING, SessionStatus.PLANNED}
    ),
    SessionStatus.FAILED: frozenset({SessionStatus.STARTING, SessionStatus.PLANNED}),
    SessionStatus.UNKNOWN: frozenset(SessionStatus),
}


class BenchmarkKind(StrEnum):
    """Benchmark workload kind.

    ``CHAT`` streams /v1/chat/completions (default; measures TTFT + tok/s).
    ``COMPLETION`` uses /v1/completions for backends without chat support.
    ``HEALTH`` times /v1/models or backend health endpoint.
    ``LONG_CONTEXT`` sends a configurable long prompt to verify the model
    survives near its context limit.
    """

    CHAT = "chat"
    COMPLETION = "completion"
    HEALTH = "health"
    LONG_CONTEXT = "long_context"


class SessionKind(StrEnum):
    """Whether llmctl owns the process or only tracks an external endpoint.

    ``OWNED`` sessions are spawned and supervised by llmctl (the historical
    default). ``ADOPTED`` sessions point at an externally-managed endpoint
    (typically a systemd unit such as ``vllm-tp``) and are tracked for
    routing/visibility only — llmctl never spawns or kills them.
    """

    OWNED = "owned"
    ADOPTED = "adopted"


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
    # Device the model lives on: the LM Link device name for LM Studio models
    # (e.g. ``yannik-desktop``, ``macbookpro.lan``), the local hostname for
    # loopback runtimes. Nullable for forward-compat with databases created
    # before this column existed; the service layer treats NULL as the local
    # host. Part of the discovery identity key, so the same model on two
    # devices is two rows.
    host: str | None = Field(default=None, index=True)
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
    kind: SessionKind | None = Field(default=SessionKind.OWNED, index=True)
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
    systemd_unit: str | None = Field(default=None, index=True)
    served_name: str | None = Field(default=None, index=True)
    adopted_at: datetime | None = None
    #: When a termination attempt was made and the process outlived it
    #: (SIGTERM *and* SIGKILL). Deliberately separate from ``status``: status
    #: describes the world right now, and a survivor that keeps serving really
    #: is ``RUNNING`` — reconcile is right to say so. This records that an
    #: *operation* did not take, which reconcile has no business clearing.
    #: Cleared by a stop that succeeds, or by a fresh launch.
    terminate_failed_at: datetime | None = None
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
    kind: str | None = Field(default=None, index=True)
    backend: str | None = Field(default=None, index=True)
    context_length: int | None = None
    peak_vram_mb: int | None = None
    avg_gpu_util_pct: float | None = None
    max_gpu_util_pct: float | None = None
    gpu_snapshot: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    parameters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    samples: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    #: Immutable experiment manifest (FIXPLAN 2026-11-03 §1C): a versioned,
    #: allowlisted snapshot of the resolved model/engine/launch/hardware/
    #: workload identity at run time. ``NULL`` on legacy rows -- readers
    #: treat that as "unknown provenance", never fabricate one.
    manifest: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    success: bool = Field(default=True, index=True)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow, index=True)


class OperationRecord(SQLModel, table=True):
    """A managed GPU lifecycle operation (swap/stop/start), persisted for
    coordination and recovery across processes and restarts (FIXPLAN §2B).

    Intent is written before side effects; phases and the final outcome are
    updated as the executor progresses. See ``services/operations.py``.
    """

    __tablename__ = "operations"

    id: str = Field(default_factory=new_id, primary_key=True)
    kind: str = Field(index=True)
    target: str
    phase: str = Field(index=True)
    #: Optional caller-supplied idempotency key; repeating it with the same
    #: request returns the same operation, a different request is refused.
    idempotency_key: str | None = Field(default=None, index=True)
    #: Browser polling identity and child outcome, retained across API restarts.
    job_id: str | None = Field(default=None, index=True)
    job_exit_code: int | None = None
    request_fingerprint: str
    request: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    previous_state: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    owner: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    recovery: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    output: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


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
    in_memory = url == "sqlite://" or (url.startswith("sqlite") and ":memory:" in url)
    if in_memory:
        engine_kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **engine_kwargs)
    if url.startswith("sqlite") and not in_memory:
        # The CLI, TUI, control-plane API, and gateway reconcile loop all open
        # this file independently. WAL lets readers proceed during a write and
        # busy_timeout makes writers wait instead of failing with
        # "database is locked".
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _sqlite_concurrency_pragmas(dbapi_connection, _record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
            finally:
                cursor.close()

    return engine


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


#: Table creation is not atomic across threads (create_all inspects then
#: creates, and SQLite DDL is not concurrent). Every init path — create_app,
#: the TUI's db_session, OperationStore — funnels through this one guard so
#: two threads cannot race "CREATE TABLE" against each other.
_init_lock = threading.Lock()
_initialised_urls: set[str] = set()


def ensure_tables(engine: Any, database_url: str | None = None) -> None:
    """Create all tables on ``engine`` and apply migrations.

    Idempotent per file-backed URL within the process: a second call
    returns immediately (nothing drops tables in-process, so this cannot
    skip needed work). In-memory URLs are always (re)created: every
    ``get_engine(":memory:")`` is a distinct database, so caching by URL
    string would leave later engines tableless.
    """
    url = database_url or load_settings().database_url
    in_memory = url == "sqlite://" or ":memory:" in url
    with _init_lock:
        if not in_memory and url in _initialised_urls:
            return
        SQLModel.metadata.create_all(engine)
        apply_migrations(engine)
        if not in_memory:
            _initialised_urls.add(url)


def init_db(database_url: str | None = None) -> None:
    """Create all tables for the configured database and apply migrations."""
    url = database_url or load_settings().database_url
    ensure_tables(get_engine(url), url)


def get_session(database_url: str | None = None) -> Generator[Session, None, None]:
    """Yield a database session suitable for FastAPI dependencies."""
    engine = get_engine(database_url)
    with Session(engine) as session:
        yield session
