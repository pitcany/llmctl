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
    #[serde(default)]
    pub request_fingerprint: Option<String>,
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
    RecordRecovery(String, String, oneshot::Sender<Result<(), DbError>>),
    RecoveryAttempts(
        String,
        DateTime<Utc>,
        oneshot::Sender<Result<(Vec<RecoveryAttempt>, u64), DbError>>,
    ),
    InsertUpgrade(Box<UpgradeRow>, oneshot::Sender<Result<(), DbError>>),
    UpdateUpgradeStatus(
        String,
        String,
        Option<String>,
        Option<String>,
        oneshot::Sender<Result<(), DbError>>,
    ),
    ListUpgrades(String, oneshot::Sender<Result<Vec<UpgradeRow>, DbError>>),
    /// Compare-and-set: (id, expected status, new status, detail).
    SetUpgradeStatusIf(
        String,
        String,
        String,
        Option<String>,
        oneshot::Sender<Result<bool, DbError>>,
    ),
    InsertFinding(Box<FindingRow>, oneshot::Sender<Result<(), DbError>>),
    /// (since, limit, unexplained): findings at or after `since`, newest first.
    ListFindings(
        Option<DateTime<Utc>>,
        usize,
        bool,
        oneshot::Sender<Result<Vec<FindingRow>, DbError>>,
    ),
    /// Operations updated at or after `since`, plus all active operations.
    /// Audit needs active operations even when the baseline is newer.
    OpsUpdatedSince(
        DateTime<Utc>,
        oneshot::Sender<Result<Vec<OpRow>, DbError>>,
    ),
    SeenAll(oneshot::Sender<Result<Vec<SeenRow>, DbError>>),
    LegacyAll(oneshot::Sender<Result<Vec<LegacyRow>, DbError>>),
    LegacyInsert(Vec<LegacyRow>, oneshot::Sender<Result<(), DbError>>),
    SetSeen(String, String, DateTime<Utc>, oneshot::Sender<Result<(), DbError>>),
    /// Keep `audit_seen` rows only for these workloads (inventory left one).
    PruneSeen(Vec<String>, oneshot::Sender<Result<(), DbError>>),
}

/// One foreign-write finding (M5a, decision 1): a managed workload's
/// state changed and no admitted operation explains it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FindingRow {
    pub id: i64,
    pub host: String,
    pub workload: String,
    /// `foreign_transition` or `engine_scheduler`.
    pub kind: String,
    /// What changed: `unit` (process/pid/served model) or `lanes`.
    pub what: String,
    pub before_state: String,
    pub after_state: String,
    /// Who did it when the manager names an actor; `None` = unknown.
    pub actor: Option<String>,
    pub observed_at: DateTime<Utc>,
}

/// Last state an audit pass saw for one workload (its comparison baseline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeenRow {
    pub workload: String,
    pub state: String,
    pub at: DateTime<Utc>,
}

/// Findings older than this are pruned as each new one lands.
pub const FINDING_RETENTION_DAYS: i64 = 30;

/// One staged/promoted upgrade of one workload (plan D.6: history).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeRow {
    pub id: String,
    pub workload: String,
    pub from_version: String,
    pub to_version: String,
    pub env_path: String,
    pub interpreter: String,
    /// staged → validated → promoting → promoted | failed | rolled_back | discarded
    pub status: String,
    pub detail: Option<serde_json::Value>,
    pub operation_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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
        // Older journals have no request identity. Keep those rows readable,
        // but never guess that a newly supplied request is identical.
        let has_fingerprint = conn
            .prepare("PRAGMA table_info(operations)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .any(|name| name == "request_fingerprint");
        if !has_fingerprint {
            conn.execute(
                "ALTER TABLE operations ADD COLUMN request_fingerprint TEXT",
                [],
            )?;
        }
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

    /// Record one admitted recovery attempt for `workload`.
    pub async fn record_recovery(&self, workload: &str, op_id: &str) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::RecordRecovery(workload.into(), op_id.into(), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    /// All recovery attempts of `workload` at or after `since`, oldest
    /// first, plus the total attempt count ever (for monotonic keys).
    pub async fn recovery_attempts(
        &self,
        workload: &str,
        since: &DateTime<Utc>,
    ) -> Result<(Vec<RecoveryAttempt>, u64), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::RecoveryAttempts(workload.into(), *since, tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn insert_upgrade(&self, row: UpgradeRow) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::InsertUpgrade(Box::new(row), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn update_upgrade_status(
        &self,
        id: &str,
        status: &str,
        detail: Option<serde_json::Value>,
        operation_id: Option<String>,
    ) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        let detail_str = detail
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default());
        self.send(DbMsg::UpdateUpgradeStatus(
            id.into(),
            status.into(),
            detail_str,
            operation_id,
            tx,
        ))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn list_upgrades(&self, workload: &str) -> Result<Vec<UpgradeRow>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::ListUpgrades(workload.into(), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    /// Move a row from `from` to `to` only if it is still in `from`
    /// (replacing `detail` when given). Returns whether the row moved, so
    /// concurrent settlers and late writers never overwrite a newer state.
    pub async fn set_upgrade_status_if(
        &self,
        id: &str,
        from: &str,
        to: &str,
        detail: Option<serde_json::Value>,
    ) -> Result<bool, DbError> {
        let (tx, rx) = oneshot::channel();
        let detail_str = detail
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default());
        self.send(DbMsg::SetUpgradeStatusIf(
            id.into(),
            from.into(),
            to.into(),
            detail_str,
            tx,
        ))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    /// Record one finding, pruning rows past the retention bound in the
    /// same transaction.
    pub async fn insert_finding(&self, row: FindingRow) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::InsertFinding(Box::new(row), tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    /// Findings at or after `since` (all when `None`), newest first.
    pub async fn list_findings(
        &self,
        since: Option<DateTime<Utc>>,
        limit: usize,
        unexplained: bool,
    ) -> Result<Vec<FindingRow>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::ListFindings(since, limit, unexplained, tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    /// Operations updated at or after `since`, oldest first.
    pub async fn ops_updated_since(&self, since: DateTime<Utc>) -> Result<Vec<OpRow>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::OpsUpdatedSince(since, tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    /// Every imported legacy row, by id.
    pub async fn legacy_all(&self) -> Result<Vec<LegacyRow>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::LegacyAll(tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    /// Insert rows in one transaction; an id that exists is an error (the
    /// importer compares first), so nothing is ever overwritten.
    pub async fn legacy_insert(&self, rows: Vec<LegacyRow>) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::LegacyInsert(rows, tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn seen_all(&self) -> Result<Vec<SeenRow>, DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::SeenAll(tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn set_seen(
        &self,
        workload: &str,
        state: &str,
        at: DateTime<Utc>,
    ) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::SetSeen(workload.into(), state.into(), at, tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }

    pub async fn prune_seen(&self, keep: Vec<String>) -> Result<(), DbError> {
        let (tx, rx) = oneshot::channel();
        self.send(DbMsg::PruneSeen(keep, tx))?;
        rx.await.map_err(|_| DbError::Closed)?
    }
}

/// One recorded recovery attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAttempt {
    pub op_id: String,
    pub attempted_at: DateTime<Utc>,
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS operations(
  id TEXT PRIMARY KEY,
  idempotency_key TEXT NOT NULL UNIQUE,
  request_fingerprint TEXT,
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
CREATE TABLE IF NOT EXISTS recoveries(
  workload TEXT NOT NULL,
  op_id TEXT NOT NULL,
  attempted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recoveries_workload ON recoveries(workload, attempted_at);
CREATE TABLE IF NOT EXISTS upgrades(
  id TEXT PRIMARY KEY,
  workload TEXT NOT NULL,
  from_version TEXT NOT NULL,
  to_version TEXT NOT NULL,
  env_path TEXT NOT NULL,
  interpreter TEXT NOT NULL,
  status TEXT NOT NULL,
  detail TEXT,
  operation_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_upgrades_workload ON upgrades(workload, created_at);
CREATE TABLE IF NOT EXISTS audit_findings(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  host TEXT NOT NULL,
  workload TEXT NOT NULL,
  kind TEXT NOT NULL,
  what TEXT NOT NULL,
  before_state TEXT NOT NULL,
  after_state TEXT NOT NULL,
  actor TEXT,
  observed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_findings_time ON audit_findings(observed_at);
CREATE TABLE IF NOT EXISTS audit_seen(
  workload TEXT PRIMARY KEY,
  state TEXT NOT NULL,
  at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS legacy_history(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  host TEXT NOT NULL,
  row_json TEXT NOT NULL,
  row_sha TEXT NOT NULL,
  snapshot_sha TEXT NOT NULL,
  imported_at TEXT NOT NULL
);
"#;

/// One row of the frozen Python history (M6 decision 9): read-only, never
/// read by admission or recovery. `row_json` is the source row verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegacyRow {
    pub id: String,
    pub kind: String,
    pub host: String,
    pub row_json: String,
    pub row_sha: String,
    pub snapshot_sha: String,
    pub imported_at: DateTime<Utc>,
}

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
                       created_at, updated_at, request_fingerprint)
                     VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
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
                        row.request_fingerprint,
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
                            created_at, updated_at, request_fingerprint
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
                            created_at, updated_at, request_fingerprint
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
                        created_at, updated_at, request_fingerprint
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
                    "UPDATE operations SET status=CASE
                         WHEN status='\"cancel_requested\"' AND ?2='\"executing\"'
                         THEN status ELSE ?2 END, phases=?3, result=?4,
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
        DbMsg::RecordRecovery(workload, op_id, tx) => {
            let out = conn
                .execute(
                    "INSERT INTO recoveries(workload, op_id, attempted_at) VALUES(?1,?2,?3)",
                    params![workload, op_id, Utc::now().to_rfc3339()],
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::RecoveryAttempts(workload, since, tx) => {
            let mut stmt = conn.prepare(
                "SELECT op_id, attempted_at FROM recoveries WHERE workload = ?1 AND attempted_at >= ?2
                 ORDER BY attempted_at",
            )?;
            let recent = stmt
                .query_map(params![workload, since.to_rfc3339()], |r| {
                    Ok(RecoveryAttempt {
                        op_id: r.get(0)?,
                        attempted_at: parse_rfc3339(r.get(1)?),
                    })
                })
                .map_err(DbError::from)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(DbError::from);
            let total = conn
                .query_row(
                    "SELECT COUNT(*) FROM recoveries WHERE workload = ?1",
                    params![workload],
                    |r| r.get::<_, i64>(0),
                )
                .map_err(DbError::from);
            let out = match (recent, total) {
                (Ok(recent), Ok(total)) => Ok((recent, total as u64)),
                (Err(e), _) | (Ok(_), Err(e)) => Err(e),
            };
            let _ = tx.send(out);
        }
        DbMsg::InsertUpgrade(boxed, tx) => {
            let row = *boxed;
            let out = conn
                .execute(
                    "INSERT INTO upgrades(id, workload, from_version, to_version, env_path,
                                          interpreter, status, detail, operation_id,
                                          created_at, updated_at)
                     VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                    params![
                        row.id,
                        row.workload,
                        row.from_version,
                        row.to_version,
                        row.env_path,
                        row.interpreter,
                        row.status,
                        row.detail
                            .as_ref()
                            .map(|v| serde_json::to_string(v).unwrap_or_default()),
                        row.operation_id,
                        row.created_at.to_rfc3339(),
                        row.updated_at.to_rfc3339(),
                    ],
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::UpdateUpgradeStatus(id, status, detail, op_id, tx) => {
            // `detail` arrives pre-serialized by the method.
            let out = conn
                .execute(
                    "UPDATE upgrades SET status=?2, detail=COALESCE(?3, detail),
                                        operation_id=COALESCE(?4, operation_id),
                                        updated_at=?5
                     WHERE id = ?1",
                    params![id, status, detail, op_id, Utc::now().to_rfc3339()],
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::ListUpgrades(workload, tx) => {
            let mut stmt = conn.prepare(
                "SELECT id, workload, from_version, to_version, env_path, interpreter,
                        status, detail, operation_id, created_at, updated_at
                 FROM upgrades WHERE workload = ?1 ORDER BY created_at, rowid",
            )?;
            let out = stmt
                .query_map(params![workload], upgrade_row_from)
                .map_err(DbError::from)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::SetUpgradeStatusIf(id, from, to, detail, tx) => {
            let out = conn
                .execute(
                    "UPDATE upgrades SET status=?3, detail=COALESCE(?4, detail), updated_at=?5
                     WHERE id = ?1 AND status = ?2",
                    params![id, from, to, detail, Utc::now().to_rfc3339()],
                )
                .map(|n| n == 1)
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::InsertFinding(boxed, tx) => {
            let row = *boxed;
            let out = conn
                .execute_batch("BEGIN")
                .and_then(|_| {
                    let cutoff = (Utc::now() - chrono::Duration::days(FINDING_RETENTION_DAYS))
                        .to_rfc3339();
                    conn.execute("DELETE FROM audit_findings WHERE observed_at < ?1", params![cutoff])?;
                    conn.execute(
                        "INSERT INTO audit_findings(
                           host, workload, kind, what, before_state, after_state, actor, observed_at)
                         VALUES(?1,?2,?3,?4,?5,?6,?7,?8)",
                        params![
                            row.host,
                            row.workload,
                            row.kind,
                            row.what,
                            row.before_state,
                            row.after_state,
                            row.actor,
                            row.observed_at.to_rfc3339(),
                        ],
                    )?;
                    conn.execute_batch("COMMIT")?;
                    Ok(())
                })
                .map_err(|e| {
                    let _ = conn.execute_batch("ROLLBACK");
                    DbError::Sql(e)
                });
            let _ = tx.send(out);
        }
        DbMsg::ListFindings(since, limit, unexplained, tx) => {
            let out = (|| -> Result<Vec<FindingRow>, DbError> {
                let mut stmt = conn.prepare(
                    "SELECT id, host, workload, kind, what, before_state, after_state, actor, observed_at
                     FROM audit_findings WHERE observed_at >= ?1
                     ORDER BY observed_at DESC, id DESC LIMIT ?2",
                )?;
                // Filter before the result limit so a burst of scheduler
                // events cannot hide older unexplained findings.
                let cutoff = since.map(|at| at.to_rfc3339()).unwrap_or_default();
                let sql_limit = if unexplained { -1 } else { limit as i64 };
                let rows = stmt.query_map(params![cutoff, sql_limit], finding_row_from)?;
                rows.map(|row| row.map(crate::audit::classify_finding))
                    .filter(|row| {
                        !unexplained
                            || !row.as_ref().is_ok_and(|f| f.kind == crate::audit::ENGINE_SCHEDULER)
                    })
                    .take(limit)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(DbError::from)
            })();
            let _ = tx.send(out);
        }
        DbMsg::OpsUpdatedSince(since, tx) => {
            let mut stmt = conn.prepare(
                "SELECT id, idempotency_key, kind, workload, status,
                        plan, phases, result, recovery, acknowledged,
                        created_at, updated_at, request_fingerprint
                 FROM operations WHERE updated_at >= ?1
                    OR status IN ('\"admitted\"', '\"executing\"', '\"cancel_requested\"')
                 ORDER BY updated_at",
            )?;
            let out = stmt
                .query_map(params![since.to_rfc3339()], op_row_from)
                .map_err(DbError::from)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::LegacyAll(tx) => {
            let mut stmt = conn.prepare(
                "SELECT id, kind, host, row_json, row_sha, snapshot_sha, imported_at
                 FROM legacy_history ORDER BY id",
            )?;
            let out = stmt
                .query_map([], |r| {
                    Ok(LegacyRow {
                        id: r.get(0)?,
                        kind: r.get(1)?,
                        host: r.get(2)?,
                        row_json: r.get(3)?,
                        row_sha: r.get(4)?,
                        snapshot_sha: r.get(5)?,
                        imported_at: parse_rfc3339(r.get(6)?),
                    })
                })?
                .collect::<Result<Vec<_>, _>>()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::LegacyInsert(rows, tx) => {
            let out = (|| -> Result<(), rusqlite::Error> {
                let t = conn.transaction()?;
                for r in &rows {
                    t.execute(
                        "INSERT INTO legacy_history(id, kind, host, row_json, row_sha, snapshot_sha, imported_at)
                         VALUES(?1,?2,?3,?4,?5,?6,?7)",
                        params![r.id, r.kind, r.host, r.row_json, r.row_sha, r.snapshot_sha, r.imported_at.to_rfc3339()],
                    )?;
                }
                t.commit()
            })()
            .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::SeenAll(tx) => {
            let mut stmt =
                conn.prepare("SELECT workload, state, at FROM audit_seen ORDER BY workload")?;
            let out = stmt
                .query_map([], |r| {
                    Ok(SeenRow {
                        workload: r.get(0)?,
                        state: r.get(1)?,
                        at: parse_rfc3339(r.get(2)?),
                    })
                })?
                .collect::<Result<Vec<_>, _>>()
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::SetSeen(workload, state, at, tx) => {
            let out = conn
                .execute(
                    "INSERT INTO audit_seen(workload, state, at) VALUES(?1,?2,?3)
                     ON CONFLICT(workload) DO UPDATE SET state = excluded.state, at = excluded.at",
                    params![workload, state, at.to_rfc3339()],
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
        DbMsg::PruneSeen(keep, tx) => {
            // A short IN list of enrolled workload names; an empty keep
            // list clears the table (an inventory with no managed units).
            let placeholders = keep.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = if keep.is_empty() {
                "DELETE FROM audit_seen".to_string()
            } else {
                format!("DELETE FROM audit_seen WHERE workload NOT IN ({placeholders})")
            };
            let out = conn
                .execute(
                    &sql,
                    rusqlite::params_from_iter(keep.iter()),
                )
                .map(|_| ())
                .map_err(DbError::from);
            let _ = tx.send(out);
        }
    }
    Ok(())
}

fn finding_row_from(r: &rusqlite::Row<'_>) -> rusqlite::Result<FindingRow> {
    Ok(FindingRow {
        id: r.get(0)?,
        host: r.get(1)?,
        workload: r.get(2)?,
        kind: r.get(3)?,
        what: r.get(4)?,
        before_state: r.get(5)?,
        after_state: r.get(6)?,
        actor: r.get(7)?,
        observed_at: parse_rfc3339(r.get(8)?),
    })
}

fn upgrade_row_from(r: &rusqlite::Row<'_>) -> rusqlite::Result<UpgradeRow> {
    Ok(UpgradeRow {
        id: r.get(0)?,
        workload: r.get(1)?,
        from_version: r.get(2)?,
        to_version: r.get(3)?,
        env_path: r.get(4)?,
        interpreter: r.get(5)?,
        status: r.get(6)?,
        detail: r
            .get::<_, Option<String>>(7)?
            .and_then(|v| serde_json::from_str(&v).ok()),
        operation_id: r.get(8)?,
        created_at: parse_rfc3339(r.get(9)?),
        updated_at: parse_rfc3339(r.get(10)?),
    })
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
        request_fingerprint: r.get("request_fingerprint")?,
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

#[cfg(test)]
mod migration_tests {
    use super::*;

    #[tokio::test]
    async fn old_journals_keep_rows_without_inventing_request_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("journal.sqlite");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(&SCHEMA.replace("  request_fingerprint TEXT,\n", ""))
            .unwrap();
        let id = OperationId::generate();
        let plan = Plan {
            id: id.clone(),
            kind: OpKind::Run,
            target: WorkloadName::new("coder").unwrap(),
            revision: 1,
            expected_model: "fake/coder".into(),
            affected_gpus: vec![],
            steps: vec![],
            rollback: vec![],
        };
        conn.execute(
            "INSERT INTO operations(id, idempotency_key, kind, workload, status,
            plan, phases, created_at, updated_at) VALUES(?1, 'legacy', '\"run\"', 'coder',
            '\"succeeded\"', ?2, '[]', ?3, ?3)",
            params![
                id.as_str(),
                serde_json::to_string(&plan).unwrap(),
                Utc::now().to_rfc3339()
            ],
        )
        .unwrap();
        drop(conn);
        let db = Db::open(&path).unwrap();
        let row = db.find_by_key("legacy").await.unwrap().unwrap();
        assert_eq!(row.id, id);
        assert_eq!(row.request_fingerprint, None);
        // Reopening an already migrated journal is safe too.
        let reopened = Db::open(&path).unwrap();
        assert!(reopened.get_op(id.as_str()).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn stale_progress_cannot_overwrite_a_persisted_cancel_request() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(&dir.path().join("journal.sqlite")).unwrap();
        let id = OperationId::generate();
        let now = Utc::now();
        let name = WorkloadName::new("coder").unwrap();
        db.insert_op(OpRow {
            id: id.clone(),
            idempotency_key: IdempotencyKey::new("cancel-race").unwrap(),
            request_fingerprint: None,
            kind: OpKind::Run,
            workload: name.clone(),
            status: OpStatus::CancelRequested,
            plan: Plan {
                id: id.clone(),
                kind: OpKind::Run,
                target: name,
                revision: 1,
                expected_model: "fake/coder".into(),
                affected_gpus: vec![],
                steps: vec![],
                rollback: vec![],
            },
            phases: vec![],
            result: None,
            recovery: None,
            acknowledged: false,
            created_at: now,
            updated_at: now,
        })
        .await
        .unwrap();
        // The executor prepared this write before the cancellation committed.
        db.mutate_op(
            id.as_str(),
            OpMutation {
                status: OpStatus::Executing,
                phases: vec![],
                result: None,
                recovery: None,
                acknowledged: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            db.get_op(id.as_str()).await.unwrap().unwrap().status,
            OpStatus::CancelRequested
        );
    }
}
