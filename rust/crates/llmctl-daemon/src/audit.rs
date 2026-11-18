//! The foreign-write record (M5a, plan decision 1).
//!
//! Every observation pass compares each managed workload's state with the
//! last state the agent recorded. A change of running/stopped, pid or
//! served model that no admitted operation explains becomes a
//! `foreign_transition` finding in the journal DB: the evidence behind
//! "no legacy lifecycle writes" and the pilot day's second condition.
//!
//! What counts as explained: an operation whose `workload` is the changed
//! one, or whose plan has any step naming it (an eviction's `stop` step
//! explains the target's disappearance). Healer and recovery repairs that
//! went through llmctl-next are such operations; a repair that bypassed
//! the agent is a finding, exactly as intended.
//!
//! Honest limits, on purpose:
//!
//! * `process == Unknown` never concludes anything — the baseline is kept.
//! * The served model is compared only when both sides name one: a probe
//!   that could not verify (transient HTTP flake) must not become a
//!   finding. A silent engine-side model unload is therefore not caught
//!   here; the engines' own loaded-list checks cover it at swap time.
//! * The actor is `None` unless the engine names one. A manager
//!   crash-restart (same unit, new pid) records as a pid change with no
//!   actor; the before/after states say what happened.
//! * llama-swap lanes: a lane that vanished names `llama-swap (ttl)` as
//!   the actor when the lane's own record declares a TTL; a lane that
//!   appeared has no actor to name. A router that cannot be asked skips
//!   the lane comparison (empty is never "unloaded").

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use llmctl_core::{Observation, Ownership, PlanStep};

use crate::app::AppState;
use crate::db::FindingRow;

/// Findings returned per listing call.
pub const LIST_LIMIT: usize = 500;

/// The comparison signature of one workload at one pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SeenState {
    pub process: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Loaded llama-swap lanes (workloads with a lane store only). `None`
    /// when this pass could not ask the router.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lanes: Option<Vec<String>>,
}

fn model_of(o: &Observation) -> Option<String> {
    match &o.served_model {
        llmctl_core::ServedModel::Verified { model }
        | llmctl_core::ServedModel::Unverified { model } => Some(model.clone()),
        llmctl_core::ServedModel::Unknown | llmctl_core::ServedModel::NotServing => None,
    }
}

/// A unit observation as a comparable state. `None` when the observation
/// cannot support a conclusion (process unknown).
fn unit_state(o: &Observation) -> Option<SeenState> {
    if o.process == llmctl_core::ProcessState::Unknown {
        return None;
    }
    Some(SeenState {
        process: format!("{:?}", o.process).to_lowercase(),
        pid: o.pid,
        model: model_of(o),
        lanes: None,
    })
}

/// True when `op` could have changed `workload`: it is the op's own
/// workload, or any plan step names it (an eviction stops another target).
fn op_touches(op: &crate::db::OpRow, workload: &str) -> bool {
    op.workload.as_str() == workload
        || op.plan.steps.iter().any(|s| match s {
            PlanStep::PersistIntent { .. } => false,
            _ => s.workload().as_str() == workload,
        })
}

/// One comparison pass. Records findings for unexplained transitions and
/// moves every compared workload's baseline forward. Serialized by the
/// app's audit lock, so a status call and the watcher tick never
/// double-report the same transition.
pub async fn compare(state: &Arc<AppState>, observations: &[Observation]) -> Vec<FindingRow> {
    let _guard = state.audit_lock.lock().await;
    let inv = state.ops.inventory().await;
    let db = state.ops.db();
    let now = Utc::now();

    let baselines: BTreeMap<String, (SeenState, DateTime<Utc>)> = db
        .seen_all()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter_map(|row| {
            serde_json::from_str::<SeenState>(&row.state)
                .ok()
                .map(|s| (row.workload, (s, row.at)))
        })
        .collect();

    // One ops pull covers every baseline window that matters (the oldest
    // baseline in the table bounds it).
    let oldest = baselines.values().map(|(_, at)| *at).min();
    let ops = match oldest {
        Some(at) => db.ops_updated_since(at).await.unwrap_or_default(),
        None => Vec::new(),
    };

    // Workloads with a lane store get the lane comparison too.
    let mut lane_stores: BTreeMap<String, crate::lanes::LaneStore> = BTreeMap::new();
    for def in &inv.workloads {
        let path = crate::lanes::store_path(state, def.name.as_str());
        if let Ok(Some(store)) = crate::lanes::read_store(&path).await {
            lane_stores.insert(def.name.to_string(), store);
        }
    }

    let mut findings = Vec::new();
    let mut managed: Vec<String> = Vec::new();
    for def in &inv.workloads {
        if !matches!(def.ownership, Ownership::Managed { .. }) {
            continue;
        }
        managed.push(def.name.to_string());
        let name = def.name.as_str();
        let Some(obs) = observations.iter().find(|o| o.workload.as_str() == name) else {
            continue; // not observed this pass: keep the baseline
        };
        let Some(mut current) = unit_state(obs) else {
            continue; // unknown never concludes
        };

        // Lanes: ask the router this workload's llama-swap (if any).
        if lane_stores.contains_key(name) {
            match crate::lanes::running_on_checked(def.port).await {
                Some(running) => current.lanes = Some(running),
                None => continue, // cannot ask: no lane conclusion this pass
            }
        }

        let (baseline, baseline_at) = match baselines.get(name) {
            Some((seen, at)) => (seen.clone(), *at),
            None => {
                // First sight of this workload: record, never conclude.
                let _ = db.set_seen(name, &encode(&current), now).await;
                continue;
            }
        };

        let changed = unit_change(&baseline, &current) || lane_change(&baseline, &current);
        if changed {
            let explained = ops
                .iter()
                .any(|op| op_touches(op, name) && op.updated_at >= baseline_at);
            if !explained {
                let actor = match (&baseline.lanes, &current.lanes) {
                    (Some(before), Some(after)) => {
                        lane_actor(&lane_stores[name], before, after)
                    }
                    _ => None,
                };
                findings.push(FindingRow {
                    id: 0,
                    host: inv.host.to_string(),
                    workload: name.to_string(),
                    kind: "foreign_transition".into(),
                    what: if baseline.lanes.is_some() || current.lanes.is_some() {
                        "unit+lanes".into()
                    } else {
                        "unit".into()
                    },
                    before_state: encode(&baseline),
                    after_state: encode(&current),
                    actor,
                    observed_at: now,
                });
            }
        }
        let _ = db.set_seen(name, &encode(&current), now).await;
    }

    for f in &findings {
        tracing::warn!(
            workload = %f.workload,
            what = %f.what,
            actor = ?f.actor,
            "foreign transition recorded"
        );
        if let Err(e) = db.insert_finding(f.clone()).await {
            tracing::error!(error = %e, "finding not recorded");
        }
    }

    // Baselines for workloads that left the inventory go away.
    let stale = baselines.keys().any(|w| !managed.contains(w));
    if stale {
        let _ = db.prune_seen(managed).await;
    }

    findings
}

fn encode(s: &SeenState) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "{}".into())
}

/// Unit-level comparison: process flip, pid change (when both name one),
/// or served-model change (when both name one).
fn unit_change(before: &SeenState, after: &SeenState) -> bool {
    if before.process != after.process {
        return true;
    }
    if before.pid.is_some() && after.pid.is_some() && before.pid != after.pid {
        return true;
    }
    if before.model.is_some() && after.model.is_some() && before.model != after.model {
        return true;
    }
    false
}

/// Lane-set comparison (workloads with lanes on both sides).
fn lane_change(before: &SeenState, after: &SeenState) -> bool {
    match (&before.lanes, &after.lanes) {
        (Some(b), Some(a)) => b != a,
        _ => false,
    }
}

/// Name the actor for a lane-set change when the records can: a lane that
/// vanished and declared a TTL was unloaded by llama-swap's own timer.
fn lane_actor(
    store: &crate::lanes::LaneStore,
    before: &[String],
    after: &[String],
) -> Option<String> {
    let vanished = before.iter().any(|l| !after.contains(l));
    let appeared = after.iter().any(|l| !before.contains(l));
    if vanished && !appeared {
        let ttl_lanes: Vec<String> = store
            .records
            .lanes
            .iter()
            .filter(|l| l.ttl().is_some() && before.contains(&l.id))
            .map(|l| l.id.clone())
            .collect();
        if !ttl_lanes.is_empty() {
            return Some(format!("llama-swap (ttl: {})", ttl_lanes.join(", ")));
        }
    }
    None
}

/// Findings at or after `since` (all when `None`), newest first.
pub async fn list(
    state: &Arc<AppState>,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<FindingRow>, crate::ops::OpsError> {
    state
        .ops
        .db()
        .list_findings(since, LIST_LIMIT)
        .await
        .map_err(crate::ops::OpsError::Db)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seen(process: &str, pid: Option<u32>, model: Option<&str>) -> SeenState {
        SeenState {
            process: process.into(),
            pid,
            model: model.map(str::to_owned),
            lanes: None,
        }
    }

    #[test]
    fn process_pid_and_model_changes_are_found() {
        let a = seen("running", Some(100), Some("qwen"));
        assert!(!unit_change(&a, &seen("running", Some(100), Some("qwen"))));
        assert!(unit_change(&a, &seen("stopped", Some(100), None)));
        assert!(unit_change(&a, &seen("running", Some(101), Some("qwen"))));
        assert!(unit_change(&a, &seen("running", Some(100), Some("llama"))));
    }

    #[test]
    fn unknown_sides_never_conclude() {
        let a = seen("running", Some(100), Some("qwen"));
        // A probe that could not verify the model or name a pid is not a
        // transition, and neither is a model going unknown.
        assert!(!unit_change(&a, &seen("running", None, Some("qwen"))));
        assert!(!unit_change(&a, &seen("running", Some(100), None)));
        assert!(!unit_change(&seen("stopped", None, None), &seen("stopped", None, None)));
    }

    #[test]
    fn lane_sets_compare_only_when_both_sides_answered() {
        let mut a = seen("running", None, None);
        a.lanes = Some(vec!["x".into()]);
        let mut b = seen("running", None, None);
        assert!(!lane_change(&a, &b));
        b.lanes = Some(vec!["x".into(), "y".into()]);
        assert!(lane_change(&a, &b));
        assert!(!lane_change(&b, &b));
    }

    #[test]
    fn a_ttl_vanish_names_llamaswap_but_an_appear_does_not() {
        use llmctl_core::swap::{Lane, SwapConfig};
        use llmctl_core::EngineKind;
        let yaml_str = |s: &str| serde_yaml::Value::String(s.to_string());
        let lane = |id: &str, ttl: Option<i64>| Lane {
            id: id.into(),
            engine_kind: EngineKind::Other("llama.cpp".into()),
            fields: match ttl {
                Some(t) => vec![
                    ("cmd".into(), yaml_str("/bin/x")),
                    ("ttl".into(), serde_yaml::Value::Number(t.into())),
                ],
                None => vec![("cmd".into(), yaml_str("/bin/y"))],
            },
        };
        let store = crate::lanes::LaneStore {
            workload: "llamaswap".into(),
            config_path: "/tmp/x".into(),
            log_path: None,
            records: SwapConfig {
                top_order: vec!["models".into()],
                globals: Vec::new(),
                macros: Vec::new(),
                matrix: None,
                lanes: vec![lane("ttl-lane", Some(300)), lane("forever", None)],
                notes: Default::default(),
            },
            file_digest: "sha256:0".into(),
            revision: 1,
            updated_at: Utc::now(),
            versions: Default::default(),
            budget_gib: None,
            peaks_gib: Default::default(),
            engine_recipes: Default::default(),
        };
        assert_eq!(
            lane_actor(&store, &["ttl-lane".into()], &[]),
            Some("llama-swap (ttl: ttl-lane)".into())
        );
        assert_eq!(lane_actor(&store, &["forever".into()], &[]), None);
        assert_eq!(lane_actor(&store, &[], &["ttl-lane".into()]), None);
    }
}
