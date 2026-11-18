//! The bounded multi-host status call behind the Hosts screen.
//!
//! Same rules as `llmctl-next status --all` (M3-E.2 and the sleep drill,
//! `docs/RUST-M3-SLEEP-DRILL-2026-11-18.md` finding 1): every host gets at
//! most `timeout` (`LLMCTL_NEXT_STATUS_TIMEOUT_S`, default 10 s); a host
//! that does not answer in time shows its last cached observations marked
//! STALE, never "stopped"; a fresh answer refreshes the same cache file
//! `status --all` writes. Unlike the CLI the hosts are asked concurrently,
//! so one sleeping Mac costs the screen one timeout, not one per host.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use llmctl_client::wire::StatusResponse;

use crate::api::AgentApi;

/// One agent the TUI talks to.
#[derive(Clone)]
pub struct HostTarget {
    /// The enrolled name (`m5`, `mbp2`) or the local agent's host id.
    pub name: String,
    pub api: Arc<dyn AgentApi>,
    /// The workstation's own agent (local socket).
    pub local: bool,
    /// `status --all`'s cache of the last fresh status, for STALE display.
    pub cache_path: Option<PathBuf>,
    /// Last successful contact before this TUI started (hosts.json).
    pub last_seen: Option<DateTime<Utc>>,
}

/// One host's answer (or the lack of one).
#[derive(Debug, Clone)]
pub struct HostSnapshot {
    pub name: String,
    pub local: bool,
    /// The status from this refresh; `None` = no answer in time.
    pub fresh: Option<StatusResponse>,
    /// The cached status shown, marked STALE, when there is no fresh one.
    pub cached: Option<StatusResponse>,
    /// Why there is no fresh status.
    pub error: Option<String>,
    pub agent_version: Option<String>,
    /// Last time this host answered (this refresh when fresh).
    pub last_seen: Option<DateTime<Utc>>,
    pub fetched_at: DateTime<Utc>,
}

impl HostSnapshot {
    /// The status to draw: fresh, else the cached one.
    pub fn shown(&self) -> Option<&StatusResponse> {
        self.fresh.as_ref().or(self.cached.as_ref())
    }

    pub fn is_stale(&self) -> bool {
        self.fresh.is_none()
    }
}

/// `LLMCTL_NEXT_STATUS_TIMEOUT_S` (whole seconds, at least 1), default 10 —
/// the same parse as the CLI's `status --all`.
pub fn status_timeout_from_env() -> Duration {
    parse_status_timeout(std::env::var("LLMCTL_NEXT_STATUS_TIMEOUT_S").ok().as_deref())
}

pub fn parse_status_timeout(v: Option<&str>) -> Duration {
    let secs = v
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|s| *s >= 1)
        .unwrap_or(10);
    Duration::from_secs(secs)
}

/// Ask every target for `/v1/status` (and `/v1/health` for the agent
/// version) at once, each bounded by `timeout`. Returns one snapshot per
/// target, in target order, after at most about `timeout`.
pub async fn fetch_status_all(targets: &[HostTarget], timeout: Duration) -> Vec<HostSnapshot> {
    let mut tasks = Vec::with_capacity(targets.len());
    for t in targets {
        let t = t.clone();
        tasks.push(tokio::spawn(async move { fetch_one(&t, timeout).await }));
    }
    let mut out = Vec::with_capacity(tasks.len());
    for (task, t) in tasks.into_iter().zip(targets) {
        out.push(match task.await {
            Ok(s) => s,
            Err(e) => stale(t, format!("status task failed: {e}")),
        });
    }
    out
}

/// One host's bounded status (the unit [`fetch_status_all`] runs per host).
pub async fn fetch_one(t: &HostTarget, timeout: Duration) -> HostSnapshot {
    let api = t.api.clone();
    let asked = tokio::time::timeout(timeout, async move {
        let (health, status) = tokio::join!(api.health(), api.status());
        (health.ok(), status)
    })
    .await;
    match asked {
        Ok((health, Ok(status))) => {
            if let Some(path) = &t.cache_path {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(path, serde_json::to_string(&status).unwrap_or_default());
            }
            let now = Utc::now();
            HostSnapshot {
                name: t.name.clone(),
                local: t.local,
                fresh: Some(status),
                cached: None,
                error: None,
                agent_version: health.map(|h| h.daemon_version),
                last_seen: Some(now),
                fetched_at: now,
            }
        }
        Ok((_, Err(e))) => stale(t, e.to_string()),
        Err(_) => stale(
            t,
            format!(
                "no answer within {} s: asleep or unreachable (LLMCTL_NEXT_STATUS_TIMEOUT_S sets the limit)",
                timeout.as_secs_f64()
            ),
        ),
    }
}

fn stale(t: &HostTarget, error: String) -> HostSnapshot {
    let cached = t
        .cache_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|text| serde_json::from_str::<StatusResponse>(&text).ok());
    HostSnapshot {
        name: t.name.clone(),
        local: t.local,
        fresh: None,
        cached,
        error: Some(error),
        agent_version: None,
        last_seen: t.last_seen,
        fetched_at: Utc::now(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_parse_matches_the_cli() {
        assert_eq!(parse_status_timeout(None), Duration::from_secs(10));
        assert_eq!(parse_status_timeout(Some("3")), Duration::from_secs(3));
        assert_eq!(parse_status_timeout(Some(" 7 ")), Duration::from_secs(7));
        assert_eq!(parse_status_timeout(Some("0")), Duration::from_secs(10));
        assert_eq!(parse_status_timeout(Some("x")), Duration::from_secs(10));
    }
}
