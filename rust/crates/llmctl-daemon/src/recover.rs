//! Bounded automatic recovery — the only background loop. A watcher
//! observes enrolled workloads on an interval and may admit an ordinary
//! `run` (key `recover-<workload>-<n>`) so recovery inherits admission,
//! the drift guard, the GPU guard, the journal and rollback for free.
//!
//! Policy (docs/RUST-M2-PLAN.md, B):
//!
//! * systemd retries first: while the manager is retrying (`auto-restart`)
//!   llmctl does nothing; it acts only when the manager gave up or the unit
//!   has `Restart=no`.
//! * Slow budget per workload, persisted in SQLite: at most
//!   `recovery.max_attempts` runs per rolling `window_secs`; past that the
//!   workload goes `needs_attention` (blocking mutations until acked)
//!   instead of being retried again.
//! * Never recover when the intent is a manual stop or unmanaged, the
//!   observation is `Unknown`, the definition drifted, a foreign process
//!   holds the card, an operation is active or unacknowledged, or the
//!   daemon just started (one interval of grace).

use std::sync::Arc;

use chrono::{Duration, Utc};
use serde_json::json;

use llmctl_client::wire::{OpStatus, PhaseStatus, PROTOCOL_VERSION};
use llmctl_core::{DesiredIntent, OpKind, OperationId, Plan};

use crate::app::AppState;
use crate::db::PhaseRecord;
use crate::engine::RecoveryState;
use crate::ops::{AdmitOutcome, AdmitParams, OpManager};

/// Default observation interval for the watcher.
const DEFAULT_INTERVAL_MS: u64 = 15_000;

fn watch_interval() -> std::time::Duration {
    std::env::var("LLMCTL_NEXT_WATCH_INTERVAL_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|ms| *ms >= 50)
        .map(std::time::Duration::from_millis)
        .unwrap_or_else(|| std::time::Duration::from_millis(DEFAULT_INTERVAL_MS))
}

/// Spawn the watcher (called once from `serve`).
pub fn spawn(state: Arc<AppState>) {
    tokio::spawn(async move {
        let interval = watch_interval();
        tracing::info!(?interval, "recovery watcher started");
        loop {
            tokio::time::sleep(interval).await;
            // One interval of grace after daemon start: observations must
            // be older than the daemon before they authorize anything.
            if Utc::now() < state.started_at + Duration::from_std(interval).expect("interval") {
                continue;
            }
            if let Err(e) = tick(&state).await {
                tracing::error!(error = %e, "recovery watcher tick failed");
            }
        }
    });
}

/// One observation pass. Errors are logged by the caller, never fatal.
async fn tick(state: &Arc<AppState>) -> Result<(), crate::ops::OpsError> {
    let inv = state.ops.inventory().await;

    // The foreign-write record rides on every pass (M5a decision 1): a
    // transition no admitted operation explains becomes a finding, even
    // when nobody asks for status.
    if let Ok(observations) = state.ops.engine().observe(&inv).await {
        let _ = crate::audit::compare(state, &observations).await;
    }

    let intents: std::collections::HashMap<String, DesiredIntent> = state
        .ops
        .db()
        .get_intents()
        .await?
        .into_iter()
        .filter_map(|(w, raw)| {
            serde_json::from_str::<DesiredIntent>(&raw)
                .ok()
                .map(|i| (w, i))
        })
        .collect();

    for def in &inv.workloads {
        let Some(policy) = def.recovery else {
            continue;
        };
        // Manual stop, plan-evicted or unmanaged: not ours to flip back on.
        match intents.get(def.name.as_str()) {
            Some(DesiredIntent::Running) => {}
            Some(_) => continue,
            None => continue,
        }

        let state_now = match state.ops.engine().recovery_state(def).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(workload = %def.name, error = %e, "recovery state unknown");
                continue;
            }
        };
        match state_now {
            RecoveryState::GaveUp => {}
            RecoveryState::Alive | RecoveryState::ManagerRetrying | RecoveryState::Unknown => {
                continue
            }
        }

        if let Err(e) = maybe_recover(
            &state.ops,
            def,
            policy.max_attempts,
            policy.window_secs,
            policy.min_interval_secs,
        )
        .await
        {
            tracing::warn!(workload = %def.name, error = %e, "recovery refused");
        }
    }
    Ok(())
}

/// Budget check, then one admitted recovery run (or the exhaustion marker).
async fn maybe_recover(
    ops: &Arc<OpManager>,
    def: &llmctl_core::WorkloadDefinition,
    max_attempts: u32,
    window_secs: u64,
    min_interval_secs: u64,
) -> Result<(), crate::ops::OpsError> {
    let db = ops.db();
    let since = Utc::now() - Duration::seconds(window_secs as i64);
    let (recent, total) = db.recovery_attempts(def.name.as_str(), &since).await?;

    // Spacing: a failing unit (and the manager's own retry loop) gets real
    // time between llmctl attempts.
    if let Some(last) = recent.last() {
        let next_allowed = last.attempted_at + Duration::seconds(min_interval_secs as i64);
        if Utc::now() < next_allowed {
            return Ok(());
        }
    }

    if (recent.len() as u32) < max_attempts {
        let key = format!("recover-{}-{}", def.name, total + 1);
        match ops
            .admit(AdmitParams {
                protocol: PROTOCOL_VERSION,
                idempotency_key: key,
                kind: OpKind::Run,
                workload: def.name.to_string(),
                expected_revision: None,
                expected_stop_targets: None,
                force: false,
                profile: None,
                plan_override: None,
            })
            .await
        {
            Ok(AdmitOutcome::Admitted(row)) => {
                db.record_recovery(def.name.as_str(), row.id.as_str())
                    .await?;
                tracing::warn!(
                    workload = %def.name,
                    operation = %row.id,
                    attempt = total + 1,
                    "manager gave up; admitted a recovery run"
                );
            }
            // Already admitted under this key (a racing tick): nothing to do.
            Ok(AdmitOutcome::Replayed(_)) => {}
            // Overlapping operation, drift, foreign GPU occupant, blocked
            // needs_attention: honest refusals, not watcher failures.
            Err(e) => return Err(e),
        }
        return Ok(());
    }

    // Budget exhausted: one unacknowledged needs_attention marker per
    // exhaustion episode (keyed by the last attempt), blocking mutations.
    let last = recent.last().map(|a| a.op_id.clone()).unwrap_or_default();
    let key = format!("recover-stalled-{}-{}", def.name, last);
    if db.find_by_key(&key).await?.is_some() {
        return Ok(());
    }
    let now = Utc::now();
    let row = crate::db::OpRow {
        id: OperationId::generate(),
        idempotency_key: llmctl_core::IdempotencyKey::new(&key)
            .map_err(|e| crate::ops::OpsError::Engine(format!("stall marker key: {e}")))?,
        kind: OpKind::Run,
        workload: def.name.clone(),
        status: OpStatus::NeedsAttention,
        plan: Plan {
            id: OperationId::generate(),
            kind: OpKind::Run,
            target: def.name.clone(),
            revision: ops.inventory().await.revision,
            expected_model: def.model.clone(),
            affected_gpus: def.gpus.clone(),
            steps: Vec::new(),
            rollback: Vec::new(),
        },
        phases: vec![PhaseRecord {
            index: 0,
            label: "recovery budget exhausted".into(),
            status: PhaseStatus::Failed,
            detail: Some(format!(
                "{} recoveries in the last {window_secs}s (budget {max_attempts}); no further attempts",
                recent.len()
            )),
        }],
        result: Some(json!({
            "outcome": "needs_attention",
            "reason": "recovery budget exhausted",
            "attempts_in_window": recent.len(),
            "window_secs": window_secs,
            "max_attempts": max_attempts,
            "last_attempt": last,
        })),
        recovery: None,
        acknowledged: false,
        created_at: now,
        updated_at: now,
    };
    db.insert_op(row).await?;
    tracing::error!(
        workload = %def.name,
        "recovery budget exhausted; workload needs attention"
    );
    Ok(())
}
