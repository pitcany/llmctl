//! Durable operation journal and intent store: SQLite behind one dedicated
//! worker thread with a bounded queue. No transaction is ever held open
//! across an external effect; each state move is its own committed write.

use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::thread;

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::oneshot;

use llmctl_client::wire::{OpStatus, PhaseStatus};
use llmctl_core::{IdempotencyKey, OpKind, OperationId, Plan, WorkloadName};

/// Queue depth for database work. A full queue is a `Busy` error, surfaced
/// honestly — never an unbounded pile-up inside the daemon.
const QUEUE_DEPTH: usize = 64;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("database queue is saturated; retry")]
    Busy,
    #[error("database worker stopped")]
    Closed,
    #[error("database error: {0}")]
    Sql(#[from] rusqlite::Error),
    #[error("corrupt record: {0}")]
    Corrupt(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseRecord {
    pub index: usize,
    pub label: String,
    pub status: PhaseStatus,
    pub detail: Option<String>,
}

/// One durable operation row. Written before external effects and updated
/// as the executor moves through phases, so a crash always leaves a
/// discoverable, honest trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpRow {
    pub id: OperationId,
    pub idempotency_key: IdempotencyKey,
    pub kind: OpKind,
    pub workload: WorkloadName,
    pub status: OpStatus,
    pub plan: Plan,
    pub phases: Vec<PhaseRecord>,
    pub result: Option<serde_json::Value>,
    pub recovery: Option<serde_json::Value>,
    pub acknowledged: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Full overwrite of the mutable fields; the executor owns the row during
/// its lifetime and writes complete states, each a separate transaction.
#[derive(Debug, Clone)]
pub struct OpMutation {
    pub status: OpStatus,
    pub phases: Vec<PhaseRecord>,
    pub result: Option<serde_json::Value>,
    pub recovery: Option<serde_json::Value>,
    pub acknowledged: bool,
}

enum DbMsg {
    MetaGet(String, oneshot::Sender<Result<Option<String>, DbError>>),
    MetaSet(String, String, oneshot::Sender<Result<(), DbError>>),
    InsertOp(Box<OpRow>, oneshot::Sender<Result<(), DbError>>),
    GetOp(String, oneshot::Sender<Result<Option<OpRow>, DbError>>),
    FindByKey(String, oneshot::Sender<Result<Option<OpRow>, DbError>>),
    ListOps(usize, oneshot::Sender<Result<Vec<OpRow>, DbError>>),
    MutateOp(String, OpMutation, oneshot::Sender<Result<(), DbError>>),
    SetIntent(String, String, oneshot::Sender<Result<(), DbError>>),
    GetIntents(oneshot::Sender<Result<Vec<(String, String)>, DbError>>),
}

#[derive(Clone)]
pub struct Db {
    tx: SyncSender<DbMsg>,
}

impl Db {
    /// Opens (creating if needed) the journal at `path` and starts the
    /// single worker thread that owns the connection.
    pub fn open(path: &std::path::Path) -> Result<Self, DbError> {
        let mut conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        let (tx, rx) = sync_channel(QUEUE_DEPTH);
        thread::Builder::new()
            .name("llmctl-db".into())
            .spawn(move || {
                for msg in rx {
                    if handle(&mut conn, msg).is_err() {
                        // reply channel dropped; keep serving others
                    }
                }
            })
            .map_err(|e| DbError::Corrupt(format!("spawn db worker: {e}")))?;
        Ok(Self { tx })
    }

    fn send(&self, msg: DbMsg) -> Result<(), DbError> {
        match self.tx.try_send(msg) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(DbError::Busy),
            Err(TrySendError::Disconnected(_)) => Err(DbError::Closed),
        }
    }

    pub async fn meta_get(&self, key: &str) -> Result<Option<String>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::MetaGet(key.into(), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn meta_set(&self, key: &str, value: &str) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::MetaSet(key.into(), value.into(), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn insert_op(&self, row: OpRow) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::InsertOp(Box::new(row), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn get_op(&self, id: &str) -> Result<Option<OpRow>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::GetOp(id.into(), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn find_by_key(&self, key: &str) -> Result<Option<OpRow>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::FindByKey(key.into(), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn list_ops(&self, limit: usize) -> Result<Vec<OpRow>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::ListOps(limit, tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn mutate_op(&self, id: &str, mutation: OpMutation) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::MutateOp(id.into(), mutation, tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn set_intent(&self, workload: &str, intent: &str) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::SetIntent(workload.into(), intent.into(), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn get_intents(&self) -> Result<Vec<(String, String)>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::GetIntents(tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS operations(
  id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  workload TEXT NOT NULL,
  status TEXT NOT NULL,
  plan TEXT NOT NULL,
  phases TEXT NOT NULL,
  result TEXT,
  recovery TEXT,
  acknowledged INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_operations_status ON operations(status);
CREATE TABLE IF NOT EXISTS intents(
  workload TEXT PRIMARY KEY,
  intent TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"#;

fn handle(conn: &mut Connection, msg: DbMsg) -> Result<(), Box<dyn std::error::Error>> {
    // Each message handler is one committed transaction; none of them wait
    // on anything external while holding the database.
    match msg {
        DbMsg::MetaGet(key, tx) => {
            let out = conn
                .query_row("SELECT value FROM meta WHERE key = ?1", params![key], |r| {
                    r.get::<_, String>(0)
                })
                .optional()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::MetaSet(key, value, tx) => {
            let out = conn
                .execute(
                    "INSERT INTO meta(key, value) VALUES(?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    params![key, value],
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::InsertOp(boxed, tx) => {
            let row = *boxed;
            let out = conn
                .execute(
                    "INSERT INTO operations(
                       id, idempotency_key, kind, workload, status,
                       plan, phases, result, recovery, acknowledged,
                       created_at, updated_at)
                     VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                    params![
                        row.id.as_str(),
                        row.idempotency_key.as_str(),
                        serde_json::to_string(&row.kind).unwrap_or_default(),
                        row.workload.as_str(),
                        serde_json::to_string(&row.status).unwrap_or_default(),
                        serde_json::to_string(&row.plan).unwrap_or_default(),
                        serde_json::to_string(&row.phases).unwrap_or_default(),
                        row.result
                            .as_ref()
                            .map(|v| serde_json::to_string(v).unwrap_or_default()),
                        row.recovery
                            .as_ref()
                            .map(|v| serde_json::to_string(v).unwrap_or_default()),
                        row.acknowledged as i64,
                        row.created_at.to_rfc3339(),
                        row.updated_at.to_rfc3339(),
                    ],
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::GetOp(id, tx) => {
            let out = conn
                .query_row(
                    "SELECT id, idempotency_key, kind, workload, status,
                            plan, phases, result, recovery, acknowledged,
                            created_at, updated_at
                     FROM operations WHERE id = ?1",
                    params![id],
                    op_row_from,
                )
                .optional()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::FindByKey(key, tx) => {
            let out = conn
                .query_row(
                    "SELECT id, idempotency_key, kind, workload, status,
                            plan, phases, result, recovery, acknowledged,
                            created_at, updated_at
                     FROM operations WHERE idempotency_key = ?1",
                    params![key],
                    op_row_from,
                )
                .optional()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::ListOps(limit, tx) => {
            let mut stmt = conn.prepare(
                "SELECT id, idempotency_key, kind, workload, status,
                        plan, phases, result, recovery, acknowledged,
                        created_at, updated_at
                 FROM operations ORDER BY created_at DESC LIMIT ?1",
            )?;
            let out = stmt
                .query_map(params![limit as i64], op_row_from)
                .map_err(DbError::from)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::MutateOp(id, m, tx) => {
            let out = conn
                .execute(
                    "UPDATE operations SET status=?2, phases=?3, result=?4,
                                            recovery=?5, acknowledged=?6, updated_at=?7
                     WHERE id = ?1",
                    params![
                        id,
                        serde_json::to_string(&m.status).unwrap_or_default(),
                        serde_json::to_string(&m.phases).unwrap_or_default(),
                        m.result
                            .as_ref()
                            .map(|v| serde_json::to_string(v).unwrap_or_default()),
                        m.recovery
                            .as_ref()
                            .map(|v| serde_json::to_string(v).unwrap_or_default()),
                        m.acknowledged as i64,
                        Utc::now().to_rfc3339(),
                    ],
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::SetIntent(workload, intent, tx) => {
            let out = conn
                .execute(
                    "INSERT INTO intents(workload, intent, updated_at) VALUES(?1,?2,?3)
                     ON CONFLICT(workload) DO UPDATE SET
                       intent = excluded.intent, updated_at = excluded.updated_at",
                    params![workload, intent, Utc::now().to_rfc3339()],
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::GetIntents(tx) => {
            let mut stmt = conn.prepare("SELECT workload, intent FROM intents")?;
            let out = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
                .map_err(DbError::from)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
    }
    Ok(())
}

fn op_row_from(r: &rusqlite::Row<'_>) -> rusqlite::Result<OpRow> {
    fn parse<T: serde::de::DeserializeOwned>(v: String, what: &str) -> rusqlite::Result<T> {
        serde_json::from_str(&v).map_err(|e| corrupt_row(what, &e.to_string()))
    }

    /// Plain TEXT columns (id, key, workload) are validated through the
    /// newtype constructors, not JSON-parsed.
    fn typed<T, F>(value: String, what: &str, build: F) -> rusqlite::Result<T>
    where
        F: FnOnce(String) -> Result<T, llmctl_core::CoreError>,
    {
        build(value).map_err(|e| corrupt_row(what, &e.to_string()))
    }

    fn corrupt_row(what: &str, msg: &str) -> rusqlite::Error {
        rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("corrupt {what}: {msg}"),
            )),
        )
    }

    Ok(OpRow {
        id: typed(
            r.get::<_, String>("id")?,
            "operation id",
            OperationId::try_from,
        )?,
        idempotency_key: typed(
            r.get::<_, String>("idempotency_key")?,
            "idempotency key",
            IdempotencyKey::try_from,
        )?,
        kind: parse(r.get::<_, String>("kind")?, "kind")?,
        workload: typed(
            r.get::<_, String>("workload")?,
            "workload",
            WorkloadName::try_from,
        )?,
        status: parse(r.get::<_, String>("status")?, "status")?,
        plan: parse(r.get::<_, String>("plan")?, "plan")?,
        phases: parse(r.get::<_, String>("phases")?, "phases")?,
        result: r
            .get::<_, Option<String>>("result")?
            .map(|v| parse(v, "result"))
            .transpose()?,
        recovery: r
            .get::<_, Option<String>>("recovery")?
            .map(|v| parse(v, "recovery"))
            .transpose()?,
        acknowledged: r.get::<_, i64>("acknowledged")? != 0,
        created_at: parse_rfc3339(r.get::<_, String>("created_at")?),
        updated_at: parse_rfc3339(r.get::<_, String>("updated_at")?),
    })
}

fn parse_rfc3339(v: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&v)
        .map(|t| t.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}
