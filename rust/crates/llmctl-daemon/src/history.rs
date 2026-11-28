//! Legacy history import (llmctl M6 decision 9, M6b item 7).
//!
//! The Python DB is frozen at the cutover. `history import <snapshot>` reads
//! a copy of it on the CLI's host ([`read_snapshot`]) and sends the
//! `sessions` and `operations` rows to the agent, which keeps them verbatim
//! and read-only in the journal table `legacy_history`, as
//! `py:<host>:<source id>`. Nothing reads that table for admission,
//! recovery or audit: an imported row never authorizes an action.
//!
//! The owner host of an operation is its `owner.hostname`; a session has no
//! host column and belongs to the host that imports it. A row of another
//! host is reported, not imported. A first call is a report; only its
//! `confirm` token applies exactly that report.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value};

use llmctl_client::catalog::{
    HistoryEntry, HistoryImportRequest, HistoryImportResponse, HistoryListResponse, HistoryRow,
    HistoryRowReport,
};

use crate::app::AppState;
use crate::catalog::CatalogError;
use crate::db::LegacyRow;

/// The tables read, and the kind each becomes.
const TABLES: [(&str, &str); 2] = [("sessions", "session"), ("operations", "operation")];

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// Read a Python DB snapshot: `(sha256 of the file, rows)`. Refuses a file
/// with a `-wal` beside it (a live DB: take `sqlite3 <db> ".backup <copy>"`).
///
/// The copy is opened immutable. A `.backup` copy of a WAL-mode DB is in WAL
/// mode too, and a plain read-only open creates `-wal` and `-shm` beside it:
/// the next call (`--confirm`) then refused the same copy as a live DB.
/// Immutable also means SQLite reads the same bytes that the sha256 is of.
pub fn read_snapshot(path: &Path) -> Result<(String, Vec<HistoryRow>), String> {
    let wal = path.with_file_name(format!(
        "{}-wal",
        path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()
    ));
    if wal.exists() {
        return Err(format!(
            "{} has a write-ahead log beside it: it is a live DB. Take a copy first: sqlite3 {} \".backup <copy>\"",
            path.display(),
            path.display()
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // The path goes in a URI: escape the three characters a URI gives a meaning.
    let uri = format!(
        "file:{}?mode=ro&immutable=1",
        path.to_string_lossy().replace('%', "%25").replace('?', "%3f").replace('#', "%23")
    );
    let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_URI;
    let conn = rusqlite::Connection::open_with_flags(&uri, flags)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (table, kind) in TABLES {
        let exists: bool = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                [table],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n > 0)
            .map_err(|e| e.to_string())?;
        if !exists {
            continue;
        }
        let mut stmt = conn.prepare(&format!("SELECT * FROM \"{table}\" ORDER BY rowid")).map_err(|e| e.to_string())?;
        let names: Vec<String> = stmt.column_names().map(str::to_owned).collect();
        let mut q = stmt.query([]).map_err(|e| e.to_string())?;
        while let Some(r) = q.next().map_err(|e| e.to_string())? {
            let mut row = Map::new();
            for (i, name) in names.iter().enumerate() {
                use rusqlite::types::ValueRef;
                let v = match r.get_ref(i).map_err(|e| e.to_string())? {
                    ValueRef::Null => Value::Null,
                    ValueRef::Integer(n) => Value::from(n),
                    ValueRef::Real(x) => Value::from(x),
                    ValueRef::Text(t) => Value::String(String::from_utf8_lossy(t).into_owned()),
                    ValueRef::Blob(b) => Value::String(format!("blob:sha256:{}", sha256_hex(b))),
                };
                row.insert(name.clone(), v);
            }
            rows.push(HistoryRow { kind: kind.into(), row });
        }
    }
    Ok((sha256_hex(&bytes), rows))
}

/// The host a row belongs to: an operation's `owner.hostname`, else none.
fn owner_host(row: &HistoryRow) -> Option<String> {
    if row.kind != "operation" {
        return None;
    }
    let owner = row.row.get("owner")?.as_str()?;
    let v: Value = serde_json::from_str(owner).ok()?;
    v.get("hostname")?.as_str().map(str::to_owned)
}

/// One row's id and outcome against what the journal holds.
fn plan_row(
    row: &HistoryRow,
    names: &[String],
    label: &str,
    have: &BTreeMap<String, String>,
) -> (HistoryRowReport, Option<(String, String)>) {
    let source = row.row.get("id").and_then(Value::as_str).unwrap_or("").to_string();
    let report = |id: String, outcome: &str, reason: Option<String>| HistoryRowReport {
        id,
        kind: row.kind.clone(),
        outcome: outcome.into(),
        reason,
    };
    if source.is_empty() || !TABLES.iter().any(|(_, k)| *k == row.kind) {
        return (report(source, "unsupported", Some("no text id, or not a session or operation row".into())), None);
    }
    let host = match owner_host(row) {
        Some(h) if !names.contains(&h) => {
            let reason = format!("owner.hostname {h} is another host (--as names this host)");
            return (report(format!("py:{h}:{source}"), "other_host", Some(reason)), None);
        }
        Some(h) => h,
        None => label.to_string(),
    };
    let id = format!("py:{host}:{source}");
    let text = serde_json::to_string(&row.row).unwrap_or_default();
    let sha = sha256_hex(text.as_bytes());
    match have.get(&id) {
        Some(old) if *old == sha => (report(id, "present", None), None),
        Some(_) => (
            report(id, "conflicting", Some("imported before with other content; the first import is kept".into())),
            None,
        ),
        None => (report(id.clone(), "new", None), Some((host, text))),
    }
}

pub async fn import(state: &AppState, req: &HistoryImportRequest) -> Result<HistoryImportResponse, CatalogError> {
    let _one = state.adopt_lock.lock().await;
    let db = state.ops.db();
    let existing = db.legacy_all().await.map_err(|e| CatalogError::Io(e.to_string()))?;
    let have: BTreeMap<String, String> = existing.iter().map(|r| (r.id.clone(), r.row_sha.clone())).collect();
    let mut names = vec![state.host.to_string()];
    for n in &req.host_names {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    let label = req.host_names.first().cloned().unwrap_or_else(|| state.host.to_string());
    let mut reports = Vec::new();
    let mut new_rows = Vec::new();
    for row in &req.rows {
        let (rep, insert) = plan_row(row, &names, &label, &have);
        if let Some((host, text)) = insert {
            new_rows.push(LegacyRow {
                id: rep.id.clone(),
                kind: row.kind.clone(),
                host,
                row_sha: sha256_hex(text.as_bytes()),
                row_json: text,
                snapshot_sha: req.snapshot_sha256.clone(),
                imported_at: chrono::Utc::now(),
            });
        }
        reports.push(rep);
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for r in &reports {
        *counts.entry(r.outcome.clone()).or_default() += 1;
    }
    let confirm = format!(
        "hist-{}",
        &sha256_hex(format!("{}\n{}\n{}", req.snapshot_sha256, names.join(","), have.len()).as_bytes())[..16]
    );
    let mut applied = false;
    if let Some(token) = &req.confirm {
        if token != &confirm {
            return Err(CatalogError::Refused(
                "the confirm token does not match this snapshot, these host names and the journal now; report again"
                    .into(),
            ));
        }
        db.legacy_insert(new_rows.clone()).await.map_err(|e| CatalogError::Io(e.to_string()))?;
        applied = true;
    }
    let total = have.len() + if applied { new_rows.len() } else { 0 };
    Ok(HistoryImportResponse { rows: reports, counts, confirm, applied, total })
}

/// The fields `history list` shows: no env, no command, no request body.
const SUMMARY: [&str; 9] =
    ["kind", "target", "phase", "runtime", "status", "systemd_unit", "served_name", "created_at", "finished_at"];

pub async fn list(state: &AppState) -> Result<HistoryListResponse, CatalogError> {
    let rows = state.ops.db().legacy_all().await.map_err(|e| CatalogError::Io(e.to_string()))?;
    let entries = rows
        .into_iter()
        .map(|r| {
            let src: Map<String, Value> = serde_json::from_str(&r.row_json).unwrap_or_default();
            let summary = SUMMARY
                .iter()
                .filter_map(|k| src.get(*k).filter(|v| !v.is_null()).map(|v| (k.to_string(), v.clone())))
                .collect();
            HistoryEntry { id: r.id, kind: r.kind, host: r.host, imported_at: r.imported_at, summary }
        })
        .collect();
    Ok(HistoryListResponse { host: state.host.to_string(), entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(id: &str, host: &str) -> HistoryRow {
        let mut row = Map::new();
        row.insert("id".into(), Value::from(id));
        row.insert("owner".into(), Value::from(format!(r#"{{"pid": 1, "hostname": "{host}"}}"#)));
        HistoryRow { kind: "operation".into(), row }
    }

    #[test]
    fn rows_get_their_owner_host_and_compare_by_content() {
        let names = vec!["workstation".to_string(), "yannik-desktop".to_string()];
        let mut have = BTreeMap::new();
        let (r, ins) = plan_row(&op("a", "yannik-desktop"), &names, "yannik-desktop", &have);
        assert_eq!((r.id.as_str(), r.outcome.as_str()), ("py:yannik-desktop:a", "new"));
        let (host, text) = ins.unwrap();
        assert_eq!(host, "yannik-desktop");
        have.insert(r.id.clone(), sha256_hex(text.as_bytes()));
        assert_eq!(plan_row(&op("a", "yannik-desktop"), &names, "x", &have).0.outcome, "present");
        let mut changed = op("a", "yannik-desktop");
        changed.row.insert("phase".into(), Value::from("failed"));
        assert_eq!(plan_row(&changed, &names, "x", &have).0.outcome, "conflicting");
        let (other, none) = plan_row(&op("b", "macbookpro.lan"), &names, "x", &have);
        assert_eq!((other.outcome.as_str(), none.is_none()), ("other_host", true));
        let mut session = HistoryRow { kind: "session".into(), row: Map::new() };
        session.row.insert("id".into(), Value::from("s1"));
        assert_eq!(plan_row(&session, &names, "yannik-desktop", &have).0.id, "py:yannik-desktop:s1");
        session.row.insert("id".into(), Value::from(7));
        assert_eq!(plan_row(&session, &names, "x", &have).0.outcome, "unsupported");
    }
}
