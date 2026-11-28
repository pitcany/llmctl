//! The foreign-write record (M5a, plan decision 1).
//!
//! Every observation pass compares each managed workload's state with the
//! last state the agent recorded. A change of running/stopped, pid or
//! served model that no admitted operation explains becomes a
//! finding in the journal DB. Llama-swap lane changes with a known,
//! unchanged service PID are classified separately as `engine_scheduler`.
//!
//! The reference for a pass is the state the workload's newest post-effect
//! observation (recorded by an executor in this baseline window) left it
//! in, else the last pass's state. Any change from the reference is a
//! finding. Merely admitting or acknowledging an operation is not evidence
//! that it changed anything.
//! Active operations defer comparison until their effects are settled.
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
use serde_json::json;
use llmctl_client::wire::PhaseStatus;

use llmctl_core::{Observation, Ownership, PlanStep};

use crate::app::AppState;
use crate::db::FindingRow;

/// Findings returned per listing call.
pub const LIST_LIMIT: usize = 500;
pub(crate) const ENGINE_SCHEDULER: &str = "engine_scheduler";
const FOREIGN_TRANSITION: &str = "foreign_transition";

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
pub(crate) fn unit_state(o: &Observation) -> Option<SeenState> {
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
        || op.plan.steps.iter().chain(&op.plan.rollback).any(|s| match s {
            PlanStep::PersistIntent { .. } => false,
            _ => s.workload().as_str() == workload,
        })
}

/// Evidence captured by an executor after actual state-changing phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EffectObservation {
    pub workload: String,
    pub state: SeenState,
    pub at: DateTime<Utc>,
}

fn effects(op: &crate::db::OpRow) -> Vec<EffectObservation> {
    op.result
        .as_ref()
        .and_then(|r| r.get("audit_observations"))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default()
}

/// What `current` is compared with: the state the workload's newest effect
/// since the baseline left it in, else the baseline. The agent's own
/// operations move no baseline, so an agent stop, run and a raw stop
/// before the next pass looked like "stopped -> stopped" (M6c finding 3).
fn reference_state(
    baseline: &SeenState,
    baseline_at: DateTime<Utc>,
    effects: &[EffectObservation],
    workload: &str,
) -> SeenState {
    match effects
        .iter()
        .filter(|e| e.workload == workload && e.at >= baseline_at)
        .max_by_key(|e| e.at)
    {
        Some(e) => SeenState {
            lanes: e.state.lanes.clone().or_else(|| baseline.lanes.clone()),
            ..e.state.clone()
        },
        None => baseline.clone(),
    }
}

/// Capture the result of completed effects while this executor still
/// holds the host lock. No-op plans and later acknowledgements get no
/// new evidence. Unknown observations cannot explain a transition.
pub(crate) async fn record_effects(manager: &crate::ops::OpManager, row: &mut crate::db::OpRow) {
    use std::collections::BTreeSet;
    let changed: BTreeSet<_> = row
        .plan
        .steps
        .iter()
        .chain(&row.plan.rollback)
        .zip(&row.phases)
        .filter(|(step, phase)| {
            phase.status == PhaseStatus::Done
                && !phase
                    .detail
                    .as_deref()
                    .is_some_and(|s| s.starts_with("skipped:"))
                && matches!(
                    step,
                    PlanStep::StartUnit { .. }
                        | PlanStep::StopUnit { .. }
                        | PlanStep::UnloadModel { .. }
                        | PlanStep::LoadModel { .. }
                        | PlanStep::LoadLane { .. }
                        | PlanStep::UnloadLane { .. }
                        | PlanStep::WriteSwapConfig { .. }
                        | PlanStep::LoadRouterModel { .. }
                        | PlanStep::UnloadRouterModel { .. }
                        | PlanStep::WriteRouterPreset { .. }
                )
        })
        .map(|(step, _)| step.workload().to_string())
        .collect();
    if changed.is_empty() {
        return;
    }
    let inv = manager.inventory().await;
    let mut affected = (*inv).clone();
    affected
        .workloads
        .retain(|def| changed.contains(def.name.as_str()));
    let observations = match manager.engine().observe(&affected).await {
        Ok(observations) => observations,
        Err(error) => {
            tracing::warn!(%error, "could not capture audit effect evidence");
            return;
        }
    };
    let mut evidence = Vec::new();
    for obs in observations
        .iter()
        .filter(|o| changed.contains(o.workload.as_str()))
    {
        let Some(mut state) = crate::audit::unit_state(obs) else {
            continue;
        };
        if let (Some(ctx), Some(def)) = (&manager.own_ctx, inv.workload(&obs.workload)) {
            let path = ctx
                .paths
                .data_dir
                .join("lanes")
                .join(format!("{}.json", obs.workload));
            if matches!(crate::lanes::read_store(&path).await, Ok(Some(_))) {
                state.lanes = crate::lanes::running_on_checked(def.port).await;
            }
        }
        evidence.push(crate::audit::EffectObservation {
            workload: obs.workload.to_string(),
            state,
            at: obs.observed_at,
        });
    }
    if let Some(result) = row.result.as_mut().and_then(|r| r.as_object_mut()) {
        result.insert("audit_observations".into(), json!(evidence));
    }
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

    let done: Vec<EffectObservation> = ops
        .iter()
        .filter(|op| op.status.is_terminal())
        .flat_map(effects)
        .collect();
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

        // A status probe can overlap an executor. Preserve the old baseline
        // until completion; never credit its intent or a half-observed effect.
        if ops.iter().any(|op| {
            op_touches(op, name)
                && (!op.status.is_terminal()
                    || effects(op).iter().any(|e| e.workload == name && e.at > obs.observed_at))
        }) {
            continue;
        }
        let baseline = match baselines.get(name) {
            Some((seen, at)) => reference_state(seen, *at, &done, name),
            None => {
                // First sight of this workload: record, never conclude.
                let _ = db.set_seen(name, &encode(&current), obs.observed_at).await;
                continue;
            }
        };

        let unit_changed = unit_change(&baseline, &current);
        let lanes_changed = lane_change(&baseline, &current);
        // `baseline` is already the state the agent's newest effect left:
        // nothing the agent did explains a change from it.
        if unit_changed || lanes_changed {
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
                kind: if engine_scheduler(&baseline, &current) {
                    ENGINE_SCHEDULER.into()
                } else {
                    FOREIGN_TRANSITION.into()
                },
                what: match (unit_changed, lanes_changed) {
                    (true, true) => "unit+lanes".into(),
                    (true, false) => "unit".into(),
                    _ => "lanes".into(),
                },
                before_state: encode(&baseline),
                after_state: encode(&current),
                actor,
                observed_at: now,
            });
        }
        let _ = db.set_seen(name, &encode(&current), obs.observed_at).await;
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

fn engine_scheduler(before: &SeenState, after: &SeenState) -> bool {
    before.process == "running"
        && before.pid.is_some_and(|pid| pid > 0)
        && before.pid == after.pid
        && !unit_change(before, after)
        && lane_change(before, after)
}

/// M5 recorded all changes as foreign transitions. The two lane
/// observations come from a llama-swap lane store, even if that workload
/// has since been removed from the inventory.
pub(crate) fn classify_finding(mut row: FindingRow) -> FindingRow {
    if row.kind == FOREIGN_TRANSITION && matches!(row.what.as_str(), "unit+lanes" | "lanes") {
        if let (Ok(before), Ok(after)) = (
            serde_json::from_str::<SeenState>(&row.before_state),
            serde_json::from_str::<SeenState>(&row.after_state),
        ) {
            if engine_scheduler(&before, &after) {
                row.kind = ENGINE_SCHEDULER.into();
                row.what = "lanes".into();
            }
        }
    }
    row
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
    unexplained: bool,
) -> Result<Vec<FindingRow>, crate::ops::OpsError> {
    state
        .ops
        .db()
        .list_findings(since, LIST_LIMIT, unexplained)
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

    /// The finding rule: `unit_change(reference_state(..), current)`.
    fn finds(baseline: &SeenState, at: DateTime<Utc>, effects: &[EffectObservation], current: &SeenState) -> bool {
        unit_change(&reference_state(baseline, at, effects, "lane"), current)
    }

    #[test]
    fn effect_evidence_expires_and_must_match_the_observed_identity() {
        let at = Utc::now();
        let before = seen("stopped", None, None);
        let effect = EffectObservation {
            workload: "lane".into(),
            at,
            state: seen("running", Some(100), Some("qwen")),
        };
        let fx = std::slice::from_ref(&effect);
        assert!(!finds(&before, at, fx, &effect.state));
        // Updating/acknowledging the operation does not renew this timestamp.
        assert!(finds(&before, at + chrono::Duration::seconds(1), fx, &effect.state));
        assert!(finds(&before, at, fx, &seen("running", Some(101), Some("qwen"))));
        assert!(finds(&before, at, fx, &seen("running", Some(100), Some("different"))));
        // Another workload's effect explains nothing here.
        let other = EffectObservation { workload: "engine".into(), ..effect.clone() };
        assert!(finds(&before, at, &[other], &effect.state));
    }

    #[test]
    fn the_newest_effect_since_the_baseline_is_the_reference() {
        // M6c finding 3: the last pass saw the lane stopped (a raw stop);
        // then the agent stopped it and ran it, and a raw stop followed
        // before any pass. The reference is "running", so it is found.
        let base = Utc::now();
        let stopped = seen("stopped", None, None);
        let stop = EffectObservation {
            workload: "lane".into(),
            at: base + chrono::Duration::seconds(1),
            state: stopped.clone(),
        };
        let run = EffectObservation {
            workload: "lane".into(),
            at: base + chrono::Duration::seconds(2),
            state: seen("running", Some(200), Some("m")),
        };
        let fx = [run.clone(), stop.clone()];
        assert!(finds(&stopped, base, &fx, &stopped));
        assert!(!finds(&stopped, base, &fx, &run.state));
        // From running: the agent's stop alone explains a stop.
        let running = seen("running", Some(100), Some("m"));
        assert!(!finds(&running, base, std::slice::from_ref(&stop), &stopped));
        // No effect: the baseline is the reference.
        assert!(finds(&running, base, &[], &stopped));
        assert!(!finds(&stopped, base, &[], &stopped));
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
