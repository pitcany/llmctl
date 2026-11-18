//! llama-swap lane records on the agent (M4b step 4, owner decision 3).
//!
//! The agent keeps one lane store per enrolled llama-swap workload
//! (`<data>/lanes/<workload>.json`): the records imported from its config,
//! the config path, its log, and the digest of the file the records stand
//! for. Flow:
//!
//! - `import` previews (round-trip proof, validation, lanes) and, with the
//!   preview's digest as confirmation, persists the store. Nothing is
//!   written to the config.
//! - `plan` applies edits to the records, validates, proves the render, and
//!   lists what changes and which running lanes the reload restarts. A file
//!   edited by hand since the import (drift) refuses: re-import first.
//! - `apply` admits a `lane_write` operation whose one step writes the
//!   render atomically, confirms llama-swap reloaded it, and rolls back to
//!   the previous bytes (confirmed the same way) when it did not.

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use llmctl_client::wire::{
    LaneEdit, LaneImportRequest, LaneImportResponse, LanePlanResponse, LaneStatusResponse,
    LaneSummary,
};
use llmctl_core::swap::{self, SwapConfig};
use llmctl_core::{OpKind, Ownership, Plan, PlanStep, WorkloadName};

use crate::app::AppState;
use crate::ops::{AdmitOutcome, AdmitParams, OpsError};

/// Seconds a write waits for llama-swap to report the reload.
pub const RELOAD_TIMEOUT_SECS: u64 = 60;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneStore {
    pub workload: String,
    pub config_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_path: Option<String>,
    pub records: SwapConfig,
    /// `sha256:` of the config bytes the records stand for.
    pub file_digest: String,
    pub revision: u64,
    pub updated_at: DateTime<Utc>,
    /// Pinned version facts per lane id (`lanes version --pin`): what the
    /// lane's command ran when the owner last accepted it.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub versions: std::collections::BTreeMap<String, llmctl_core::VersionRef>,
    /// GiB of model peaks this host can hold at once (`lanes memory`). A
    /// Mac has one pool of unified memory, not cards to fit models on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_gib: Option<u64>,
    /// Peak memory per lane id, GiB.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub peaks_gib: std::collections::BTreeMap<String, f64>,
}

impl LaneStore {
    /// The budget against `cfg` (the records, or a planned change of
    /// them); `None` when no budget is set.
    pub fn budget_report(&self, cfg: &SwapConfig) -> Option<swap::BudgetReport> {
        self.budget_gib.map(|b| swap::budget_report(cfg, &self.peaks_gib, b))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LanesError {
    #[error(transparent)]
    Ops(#[from] OpsError),
    #[error("{0}")]
    Refused(String),
    #[error("{0}")]
    Io(String),
}

pub(crate) fn refuse(msg: impl Into<String>) -> LanesError {
    LanesError::Refused(msg.into())
}

pub fn store_path(state: &AppState, workload: &str) -> PathBuf {
    state.paths.data_dir.join("lanes").join(format!("{workload}.json"))
}

pub fn bytes_digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

pub(crate) async fn read_store(path: &Path) -> Result<Option<LaneStore>, LanesError> {
    match tokio::fs::read(path).await {
        Ok(b) => serde_json::from_slice(&b)
            .map(Some)
            .map_err(|e| LanesError::Io(format!("lane store {}: {e}", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(LanesError::Io(format!("lane store {}: {e}", path.display()))),
    }
}

async fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use tokio::io::AsyncWriteExt;
    let dir = path.parent().ok_or_else(|| format!("{} has no parent", path.display()))?;
    tokio::fs::create_dir_all(dir)
        .await
        .map_err(|e| format!("create {}: {e}", dir.display()))?;
    let tmp = dir.join(format!(
        ".{}.llmctl-tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("file")
    ));
    let mut f = tokio::fs::File::create(&tmp)
        .await
        .map_err(|e| format!("create {}: {e}", tmp.display()))?;
    f.write_all(bytes).await.map_err(|e| format!("write {}: {e}", tmp.display()))?;
    f.sync_all().await.map_err(|e| format!("sync {}: {e}", tmp.display()))?;
    drop(f);
    // Keep the mode of the file being replaced (a config may be 0600).
    if let Ok(meta) = tokio::fs::metadata(path).await {
        let _ = tokio::fs::set_permissions(&tmp, meta.permissions()).await;
    }
    tokio::fs::rename(&tmp, path)
        .await
        .map_err(|e| format!("rename onto {}: {e}", path.display()))
}

pub(crate) async fn save_store(path: &Path, store: &LaneStore) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(store).map_err(|e| e.to_string())?;
    write_atomic(path, &bytes).await
}

fn summaries(cfg: &SwapConfig) -> Vec<LaneSummary> {
    cfg.lanes
        .iter()
        .map(|l| LaneSummary {
            id: l.id.clone(),
            engine_kind: l.engine_kind.to_string(),
            name: l.str_field("name").map(str::to_owned),
            model_name: l.str_field("useModelName").map(str::to_owned),
            ttl: l.ttl(),
            pinned: None,
        })
        .collect()
}

/// The summaries with each lane's pinned version named (no probing).
fn summaries_with_pins(store: &LaneStore) -> Vec<LaneSummary> {
    let mut out = summaries(&store.records);
    for (s, lane) in out.iter_mut().zip(&store.records.lanes) {
        s.pinned = store
            .versions
            .get(&s.id)
            .map(|pin| crate::lane_version::label(pin, &lane.engine_kind));
    }
    out
}

/// The llama-swap workload, its config path and log path: from the request,
/// else from the pinned launch definition (`-config <path>`, plist stdout).
async fn locate(
    state: &AppState,
    workload: &str,
    config_path: Option<String>,
    log_path: Option<String>,
) -> Result<(llmctl_core::WorkloadDefinition, String, Option<String>), LanesError> {
    let name = WorkloadName::new(workload).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let def = inv
        .workload(&name)
        .cloned()
        .ok_or_else(|| OpsError::UnknownWorkload(workload.to_string()))?;
    let (argv, stdout) = match &def.ownership {
        Ownership::Managed { definition, .. } => (
            definition
                .exec_start
                .first()
                .map(|c| c.argv.clone())
                .unwrap_or_default(),
            definition.launchd.as_ref().and_then(|l| l.stdout_path.clone()),
        ),
        Ownership::External => (Vec::new(), None),
    };
    let from_argv = argv
        .iter()
        .position(|a| a == "-config" || a == "--config")
        .and_then(|i| argv.get(i + 1).cloned());
    let config = config_path.or(from_argv).ok_or_else(|| {
        refuse(format!(
            "{workload}: no -config in its pinned definition; pass --config <path>"
        ))
    })?;
    if !Path::new(&config).is_absolute() {
        return Err(refuse(format!("config path {config} is not absolute")));
    }
    Ok((def, config, log_path.or(stdout)))
}

// ---- import --------------------------------------------------------------------

pub async fn import(state: &AppState, req: LaneImportRequest) -> Result<LaneImportResponse, LanesError> {
    let (def, config_path, log_path) =
        locate(state, &req.workload, req.config_path.clone(), req.log_path.clone()).await?;
    let bytes = tokio::fs::read(&config_path)
        .await
        .map_err(|e| LanesError::Io(format!("reading {config_path}: {e}")))?;
    let file_digest = bytes_digest(&bytes);
    let text = String::from_utf8(bytes).map_err(|_| refuse(format!("{config_path} is not UTF-8")))?;
    let mut resp = LaneImportResponse {
        workload: def.name.to_string(),
        config_path: config_path.clone(),
        log_path: log_path.clone(),
        file_digest: file_digest.clone(),
        records_digest: None,
        roundtrip: String::new(),
        validation: Vec::new(),
        lanes: Vec::new(),
        notes_kept: 0,
        persisted: false,
    };
    let cfg = match swap::roundtrip(&text) {
        Ok((cfg, _)) => {
            resp.roundtrip = "equivalent".into();
            cfg
        }
        Err(e) => {
            resp.roundtrip = e.to_string();
            return Ok(resp);
        }
    };
    resp.validation = swap::validate(&cfg);
    resp.lanes = summaries(&cfg);
    resp.notes_kept = cfg.notes.count();
    // The confirmation binds the records *and* the file they came from.
    let digest = bytes_digest(format!("{}|{}", cfg.digest(), file_digest).as_bytes());
    resp.records_digest = Some(digest.clone());
    let Some(confirm) = req.confirm else {
        return Ok(resp);
    };
    if confirm != digest {
        return Err(refuse(format!(
            "stale confirmation: the preview digest is now {digest}"
        )));
    }
    if !resp.validation.is_empty() {
        return Err(refuse(format!(
            "llama-swap would refuse this config: {}",
            resp.validation.join("; ")
        )));
    }
    let path = store_path(state, &resp.workload);
    let existing = read_store(&path).await?;
    let revision = existing.as_ref().map(|s| s.revision + 1).unwrap_or(1);
    // A re-import follows a hand edit of the file. It changes what the
    // records say, not what the owner accepted: pins stay (for lanes that
    // still exist), so a moved engine path shows as drift, not as a pin.
    let (mut versions, budget_gib, mut peaks_gib) = existing
        .map(|s| (s.versions, s.budget_gib, s.peaks_gib))
        .unwrap_or_default();
    versions.retain(|id, _| cfg.lane(id).is_some());
    peaks_gib.retain(|id, _| cfg.lane(id).is_some());
    save_store(
        &path,
        &LaneStore {
            workload: resp.workload.clone(),
            config_path,
            log_path,
            records: cfg,
            file_digest,
            revision,
            updated_at: Utc::now(),
            versions,
            budget_gib,
            peaks_gib,
        },
    )
    .await
    .map_err(LanesError::Io)?;
    resp.persisted = true;
    Ok(resp)
}

// ---- status --------------------------------------------------------------------

pub async fn status(state: &AppState, workload: &str) -> Result<LaneStatusResponse, LanesError> {
    let store = read_store(&store_path(state, workload))
        .await?
        .ok_or_else(|| refuse(format!("{workload} has no lane records; run `lanes import {workload}`")))?;
    let now = tokio::fs::read(&store.config_path)
        .await
        .map(|b| bytes_digest(&b))
        .unwrap_or_else(|e| format!("unreadable: {e}"));
    Ok(LaneStatusResponse {
        workload: store.workload.clone(),
        config_path: store.config_path.clone(),
        log_path: store.log_path.clone(),
        revision: store.revision,
        file_digest: store.file_digest.clone(),
        drift: now != store.file_digest,
        file_digest_now: now,
        lanes: summaries_with_pins(&store),
        macros: store.records.macros.clone(),
        validation: swap::validate(&store.records),
        updated_at: store.updated_at,
        memory: store
            .budget_report(&store.records)
            .map(|r| r.lines)
            .unwrap_or_default(),
        matrix: store.records.matrix.clone(),
    })
}

// ---- plan / apply ----------------------------------------------------------------

fn apply_edits(cfg: &mut SwapConfig, edits: &[LaneEdit]) -> Result<Vec<String>, LanesError> {
    let mut summary = Vec::new();
    for e in edits {
        match e {
            LaneEdit::SetField { lane, field, value } => {
                let v: serde_yaml::Value = serde_yaml::to_value(value)
                    .map_err(|e| refuse(format!("value for {lane}.{field}: {e}")))?;
                let l = cfg
                    .lane_mut(lane)
                    .ok_or_else(|| refuse(format!("no lane {lane}")))?;
                l.set(field, v);
                summary.push(format!("{lane}.{field} = {value}"));
            }
            LaneEdit::RemoveField { lane, field } => {
                let l = cfg
                    .lane_mut(lane)
                    .ok_or_else(|| refuse(format!("no lane {lane}")))?;
                let before = l.fields.len();
                l.fields.retain(|(k, _)| k != field);
                if l.fields.len() == before {
                    return Err(refuse(format!("lane {lane} has no field {field}")));
                }
                summary.push(format!("{lane}: remove {field}"));
            }
            LaneEdit::SetMacro { name, value } => {
                cfg.set_macro(name, value);
                summary.push(format!("macro {name} = {value}"));
            }
            LaneEdit::Matrix { edit } => {
                summary.push(llmctl_core::matrix_edit::apply(cfg, edit).map_err(refuse)?);
            }
        }
    }
    if edits.is_empty() {
        summary.push("re-render (no record change)".into());
    }
    // Kinds follow the commands.
    let macros = cfg.macros.clone();
    for l in &mut cfg.lanes {
        l.engine_kind = swap::infer_lane_kind(l, &macros);
    }
    Ok(summary)
}

struct Planned {
    store: LaneStore,
    next: SwapConfig,
    rendered: String,
    current_bytes: Vec<u8>,
    resp: LanePlanResponse,
}

async fn plan_inner(
    state: &AppState,
    workload: &str,
    edits: &[LaneEdit],
) -> Result<Planned, LanesError> {
    let store = read_store(&store_path(state, workload))
        .await?
        .ok_or_else(|| refuse(format!("{workload} has no lane records; run `lanes import {workload}`")))?;
    let current_bytes = tokio::fs::read(&store.config_path)
        .await
        .map_err(|e| LanesError::Io(format!("reading {}: {e}", store.config_path)))?;
    if bytes_digest(&current_bytes) != store.file_digest {
        return Err(refuse(format!(
            "{} changed since the lane records were imported (edited by hand?); review it, then `lanes import {workload}` again",
            store.config_path
        )));
    }
    let mut next = store.records.clone();
    let summary = apply_edits(&mut next, edits)?;
    let mut validation = swap::validate(&next);
    // A write must not add a combination that does not fit the memory
    // budget. One that was over before this change is the owner's
    // standing decision and does not block an unrelated edit.
    if let (Some(now), Some(after)) = (store.budget_report(&store.records), store.budget_report(&next)) {
        let budget = store.budget_gib.unwrap_or(0);
        for (set, peak) in &after.over {
            if !now.over.iter().any(|(s, p)| s == set && p >= peak) {
                validation.push(format!(
                    "matrix set {set} would peak at {peak}G, over the {budget}G memory budget \
                     (see `lanes memory {workload}`; a set named {}* is exempt)",
                    swap::RISKY_SET_PREFIX
                ));
            }
        }
    }
    validation.extend(matrix_guard(&store, &next, edits, workload));
    let rendered = swap::check_render(&next).map_err(|e| refuse(e.to_string()))?;
    let changes = swap::value_diffs(&swap::to_value(&store.records), &swap::to_value(&next), "");
    let running = running_lanes(state, workload).await;
    let plan_digest = bytes_digest(
        format!("{}|{}|{}", next.digest(), store.file_digest, bytes_digest(rendered.as_bytes())).as_bytes(),
    );
    let resp = LanePlanResponse {
        workload: workload.to_string(),
        summary: summary.join("; "),
        changes,
        validation,
        reload_restarts: running,
        reload_preloads: next
            .globals
            .iter()
            .find(|(k, _)| k == "hooks")
            .and_then(|(_, h)| h.get("on_startup"))
            .and_then(|o| o.get("preload"))
            .and_then(serde_yaml::Value::as_sequence)
            .map(|s| s.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
            .unwrap_or_default(),
        plan_digest,
        bytes_changed: rendered.as_bytes() != current_bytes.as_slice(),
    };
    Ok(Planned {
        store,
        next,
        rendered,
        current_bytes,
        resp,
    })
}

/// The memory budget as a guard on matrix writes (M5c item 4): a matrix
/// edit needs a recorded budget, and every set it adds or changes (not
/// `risky*`) must have a peak figure for each of its lanes, so the sum
/// the budget check makes is a real one and not a lower bound.
fn matrix_guard(store: &LaneStore, next: &SwapConfig, edits: &[LaneEdit], workload: &str) -> Vec<String> {
    if !edits.iter().any(|e| matches!(e, LaneEdit::Matrix { .. })) {
        return Vec::new();
    }
    if store.budget_gib.is_none() {
        return vec![format!(
            "no memory budget recorded for {workload}: a matrix change is checked against it \
             (`lanes memory {workload} --budget <GiB>` first)"
        )];
    }
    let expand = |cfg: &SwapConfig| -> Vec<swap::ExpandedSet> {
        let ids: std::collections::BTreeSet<&str> = cfg.lanes.iter().map(|l| l.id.as_str()).collect();
        cfg.matrix
            .as_ref()
            .and_then(|m| swap::expand_matrix(m, &ids).ok())
            .unwrap_or_default()
    };
    let before = expand(&store.records);
    let mut out = Vec::new();
    for set in expand(next) {
        if set.name.starts_with(swap::RISKY_SET_PREFIX) || before.contains(&set) {
            continue;
        }
        for lane in &set.lanes {
            let forwarder = next
                .lane(lane)
                .is_some_and(|l| matches!(&l.engine_kind, llmctl_core::EngineKind::Other(k) if k == "forwarder"));
            if !forwarder && !store.peaks_gib.contains_key(lane) {
                let msg = format!(
                    "matrix set {} holds {lane}, which has no peak figure, so the budget cannot \
                     check it (`lanes memory {workload} --peak {lane}=<GiB>`)",
                    set.name
                );
                if !out.contains(&msg) {
                    out.push(msg);
                }
            }
        }
    }
    out
}

pub async fn plan(
    state: &AppState,
    workload: &str,
    edits: &[LaneEdit],
) -> Result<LanePlanResponse, LanesError> {
    Ok(plan_inner(state, workload, edits).await?.resp)
}

/// Lanes llama-swap has running now (`GET /running`); a reload restarts them.
async fn running_lanes(state: &AppState, workload: &str) -> Vec<String> {
    let Ok(name) = WorkloadName::new(workload) else {
        return Vec::new();
    };
    let Some(port) = state.ops.inventory().await.workload(&name).map(|d| d.port) else {
        return Vec::new();
    };
    running_on(port).await
}

/// Lanes the llama-swap on `port` has running (empty when it cannot say).
async fn running_on(port: u16) -> Vec<String> {
    running_on_checked(port).await.unwrap_or_default()
}

/// Lanes the llama-swap on `port` has running. `None` = the router could
/// not be asked — never read as "nothing loaded" (M5a audit baseline).
pub(crate) async fn running_on_checked(port: u16) -> Option<Vec<String>> {
    match crate::probe_http::request(port, hyper::Method::GET, "/running", None, Duration::from_secs(3)).await {
        Ok((200, body)) => serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v.get("running").and_then(|r| r.as_array()).cloned())
            .map(|a| {
                a.iter()
                    .filter_map(|m| m.get("model").and_then(|x| x.as_str()).map(str::to_owned))
                    .collect()
            }),
        _ => None,
    }
}

pub async fn apply(
    state: &AppState,
    workload: &str,
    edits: &[LaneEdit],
    confirm: &str,
    idempotency_key: String,
) -> Result<AdmitOutcome, LanesError> {
    let p = plan_inner(state, workload, edits).await?;
    if p.resp.plan_digest != confirm {
        return Err(refuse(format!(
            "stale confirmation: the plan digest is now {} (re-run the plan and review it)",
            p.resp.plan_digest
        )));
    }
    if !p.resp.validation.is_empty() {
        return Err(refuse(format!(
            "llama-swap would refuse this config: {}",
            p.resp.validation.join("; ")
        )));
    }
    if !p.resp.bytes_changed {
        return Err(refuse("the render equals the live file; nothing to write"));
    }
    let name = WorkloadName::new(workload).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let def = inv
        .workload(&name)
        .cloned()
        .ok_or_else(|| OpsError::UnknownWorkload(workload.to_string()))?;
    let store_file = store_path(state, workload).display().to_string();
    let next_lanes: Vec<String> = p.next.lanes.iter().map(|l| l.id.clone()).collect();
    let prev_lanes: Vec<String> = p.store.records.lanes.iter().map(|l| l.id.clone()).collect();
    let previous = String::from_utf8(p.current_bytes.clone())
        .map_err(|_| refuse("the live config is not UTF-8"))?;
    let step = |content: String, expect: String, lanes: Vec<String>, summary: String| {
        PlanStep::WriteSwapConfig {
            workload: name.clone(),
            path: p.store.config_path.clone(),
            content,
            expect_digest: expect,
            log_path: p.store.log_path.clone(),
            port: def.port,
            lanes,
            timeout_secs: RELOAD_TIMEOUT_SECS,
            store_path: store_file.clone(),
            summary,
        }
    };
    let plan = Plan {
        id: llmctl_core::OperationId::generate(),
        kind: OpKind::LaneWrite,
        target: name.clone(),
        revision: inv.revision,
        expected_model: def.model.clone(),
        affected_gpus: Vec::new(),
        steps: vec![step(
            p.rendered.clone(),
            p.store.file_digest.clone(),
            next_lanes,
            p.resp.summary.clone(),
        )],
        rollback: vec![step(
            previous,
            bytes_digest(p.rendered.as_bytes()),
            prev_lanes,
            format!("roll back: {}", p.resp.summary),
        )],
    };
    Ok(state
        .ops
        .admit(AdmitParams {
            protocol: llmctl_client::wire::PROTOCOL_VERSION,
            idempotency_key,
            kind: OpKind::LaneWrite,
            workload: workload.to_string(),
            expected_revision: None,
            expected_stop_targets: None,
            force: false,
            profile: None,
            plan_override: Some(plan),
        })
        .await?)
}

// ---- lane swap (M4c) -------------------------------------------------------------------

/// Longest one lane load may take: a 35B read from disk needs minutes.
const LANE_LOAD_TIMEOUT_SECS: u64 = 900;
/// Longest a lane may take to leave after llama-swap was asked.
const LANE_UNLOAD_TIMEOUT_SECS: u64 = 90;

/// Predict a lane swap from the records and, unless `dry_run`, admit it as
/// an operation. llama-swap decides the eviction when the lane is
/// requested; the prediction tells the owner beforehand and is compared
/// with what really happened afterwards.
pub async fn swap(
    state: &AppState,
    req: &llmctl_client::wire::LaneSwapRequest,
) -> Result<(llmctl_client::wire::LaneSwapResponse, Option<AdmitOutcome>), LanesError> {
    let name = WorkloadName::new(&req.workload).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let def = inv
        .workload(&name)
        .cloned()
        .ok_or_else(|| OpsError::UnknownWorkload(req.workload.clone()))?;
    let Some(store) = read_store(&store_path(state, &req.workload)).await? else {
        return Err(refuse(format!(
            "{} has no lane records; `lanes import {}` first",
            req.workload, req.workload
        )));
    };
    let running = running_on(def.port).await;
    let mut resp = if req.unload {
        // A stop: the lane itself leaves, nothing else. No matrix to ask.
        if store.records.lane(&req.lane).is_none() {
            return Err(refuse(format!("{} has no lane {}", req.workload, req.lane)));
        }
        let is_running = running.iter().any(|l| l == &req.lane);
        llmctl_client::wire::LaneSwapResponse {
            workload: req.workload.clone(),
            lane: req.lane.clone(),
            evicts: if is_running { vec![req.lane.clone()] } else { Vec::new() },
            running,
            set: String::new(),
            cost: 0,
            already_loaded: false,
            operation: None,
            replayed: false,
            unload: true,
        }
    } else {
        let p = llmctl_core::swap::predict_swap(&store.records, &req.lane, &running).map_err(refuse)?;
        llmctl_client::wire::LaneSwapResponse {
            workload: req.workload.clone(),
            lane: req.lane.clone(),
            running,
            evicts: p.evicts.clone(),
            set: p.set,
            cost: p.cost,
            already_loaded: p.already_loaded,
            operation: None,
            replayed: false,
            unload: false,
        }
    };
    // Nothing to stop: say so and admit nothing.
    if req.dry_run || (req.unload && resp.evicts.is_empty()) {
        return Ok((resp, None));
    }
    // Request safety (M4c item 6) for lanes: an eviction cuts whatever the
    // evicted lane is answering. llama-swap only reports a router-wide
    // count, so any request in flight holds an evicting swap: wait the
    // usual bound, then refuse unless forced. A swap that evicts nothing,
    // or a llama-swap that cannot say, is not held.
    if !req.force && !resp.evicts.is_empty() {
        let budget = crate::inflight::wait_budget();
        let deadline = tokio::time::Instant::now() + budget;
        while let Some(n) = llama_swap_inflight(def.port, Duration::from_secs(3)).await {
            if n == 0 {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(OpsError::RequestsInFlight {
                    workload: req.workload.clone(),
                    busy: vec![(format!("{} (all lanes)", req.workload), n)],
                    waited_ms: budget.as_millis() as u64,
                }
                .into());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
    let plan = Plan {
        id: llmctl_core::OperationId::generate(),
        kind: OpKind::LaneSwap,
        target: name.clone(),
        revision: inv.revision,
        expected_model: def.model.clone(),
        affected_gpus: Vec::new(),
        steps: vec![if req.unload {
            PlanStep::UnloadLane {
                workload: name.clone(),
                port: def.port,
                lane: req.lane.clone(),
                timeout_secs: LANE_UNLOAD_TIMEOUT_SECS,
            }
        } else {
            PlanStep::LoadLane {
                workload: name.clone(),
                port: def.port,
                lane: req.lane.clone(),
                evicts: resp.evicts.clone(),
                timeout_secs: LANE_LOAD_TIMEOUT_SECS,
            }
        }],
        // Nothing to undo: an evicted lane loads again on its next request.
        rollback: Vec::new(),
    };
    let key = req.idempotency_key.clone().unwrap_or_else(|| {
        format!(
            "{}-{}-{}-{}",
            if req.unload { "lanestop" } else { "laneswap" },
            req.workload,
            req.lane.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.', "_"),
            chrono::Utc::now().timestamp_millis()
        )
    });
    let outcome = state
        .ops
        .admit(AdmitParams {
            protocol: llmctl_client::wire::PROTOCOL_VERSION,
            idempotency_key: key,
            kind: OpKind::LaneSwap,
            workload: req.workload.clone(),
            expected_revision: None,
            expected_stop_targets: None,
            force: false,
            profile: None,
            plan_override: Some(plan),
        })
        .await?;
    resp.replayed = matches!(outcome, AdmitOutcome::Replayed(_));
    Ok((resp, Some(outcome)))
}

/// Requests llama-swap has in flight, all lanes together: the first
/// `inflight` event of its never-ending event stream (`GET /api/events`,
/// `data:{"type":"inflight","data":"{\"total\":N}"}`). `None` when it
/// cannot be read in time — unknown, which holds nothing.
pub async fn llama_swap_inflight(port: u16, timeout: Duration) -> Option<u64> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    let read = async {
        let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.ok()?;
        stream
            .write_all(
                format!("GET /api/events HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAccept: text/event-stream\r\n\r\n")
                    .as_bytes(),
            )
            .await
            .ok()?;
        let mut lines = tokio::io::BufReader::new(stream).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(total) = parse_inflight_event(&line) {
                return Some(total);
            }
        }
        None
    };
    tokio::time::timeout(timeout, read).await.ok().flatten()
}

/// `N` from one SSE line when it is llama-swap's `inflight` event.
fn parse_inflight_event(line: &str) -> Option<u64> {
    let event: serde_json::Value = serde_json::from_str(line.strip_prefix("data:")?.trim()).ok()?;
    if event.get("type")?.as_str()? != "inflight" {
        return None;
    }
    let data = event.get("data")?;
    let payload = match data.as_str() {
        Some(text) => serde_json::from_str(text).ok()?,
        None => data.clone(),
    };
    payload.get("total")?.as_u64()
}

/// The effect of a `LoadLane` step: a one-token request makes llama-swap
/// load the lane (and evict what must go); then `/running` must list it.
/// Reports what left memory and whether the records had predicted it.
pub async fn load_and_confirm(
    port: u16,
    lane: &str,
    predicted: &[String],
    timeout: Duration,
) -> Result<String, String> {
    let before = running_on(port).await;
    if before.iter().any(|l| l == lane) {
        return Ok(format!("{lane} was already running; nothing loaded or evicted"));
    }
    let body = serde_json::json!({
        "model": lane,
        "messages": [{"role": "user", "content": "ok"}],
        "max_tokens": 1,
        "stream": false,
    });
    let sent = tokio::time::timeout(
        timeout,
        crate::probe_http::request(port, hyper::Method::POST, "/v1/chat/completions", Some(&body), timeout),
    )
    .await;
    match sent {
        Ok(Ok((status, _))) if (200..300).contains(&status) => {}
        Ok(Ok((status, body))) => {
            return Err(format!(
                "llama-swap answered {status} for {lane}: {}",
                String::from_utf8_lossy(&body).chars().take(200).collect::<String>()
            ))
        }
        Ok(Err(())) => return Err(format!("llama-swap did not answer the request for {lane}")),
        Err(_) => return Err(format!("{lane} did not load within {}s", timeout.as_secs())),
    }
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let after = loop {
        let now = running_on(port).await;
        if now.iter().any(|l| l == lane) {
            break now;
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!("{lane} answered but llama-swap does not list it as running"));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    let mut evicted: Vec<String> = before.into_iter().filter(|l| !after.contains(l)).collect();
    evicted.sort();
    let mut expected = predicted.to_vec();
    expected.sort();
    let list = |v: &[String]| if v.is_empty() { "nothing".to_string() } else { v.join(", ") };
    Ok(if evicted == expected {
        format!("{lane} running; llama-swap evicted {}, as the records predicted", list(&evicted))
    } else {
        format!(
            "{lane} running; llama-swap evicted {} but the records predicted {} (PREDICTION MISMATCH)",
            list(&evicted),
            list(&expected)
        )
    })
}

/// The effect of an `UnloadLane` step: llama-swap's own unload call, then
/// the lane must be gone from `/running`.
pub async fn unload_and_confirm(port: u16, lane: &str, timeout: Duration) -> Result<String, String> {
    if !running_on(port).await.iter().any(|l| l == lane) {
        return Ok(format!("{lane} was not running; nothing unloaded"));
    }
    let path = format!("/api/models/unload/{lane}");
    match crate::probe_http::request(port, hyper::Method::POST, &path, None, Duration::from_secs(30)).await {
        Ok((status, _)) if (200..300).contains(&status) => {}
        Ok((status, body)) => {
            return Err(format!(
                "llama-swap answered {status} to the unload of {lane}: {}",
                String::from_utf8_lossy(&body).chars().take(200).collect::<String>()
            ))
        }
        Err(()) => return Err(format!("llama-swap did not answer the unload of {lane}")),
    }
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if !running_on(port).await.iter().any(|l| l == lane) {
            return Ok(format!("{lane} unloaded; llama-swap no longer lists it"));
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!(
                "{lane} is still listed as running {}s after the unload",
                timeout.as_secs()
            ));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// The largest `*_peak_memory_*_gb` figure of each registry record,
/// by record id (the Mac model registry: one record per lane).
fn registry_peaks(text: &str) -> Result<std::collections::BTreeMap<String, f64>, String> {
    let doc: serde_yaml::Value = serde_yaml::from_str(text).map_err(|e| format!("not YAML: {e}"))?;
    let records = doc
        .get("models")
        .and_then(serde_yaml::Value::as_sequence)
        .or_else(|| doc.as_sequence())
        .ok_or("no `models:` list of records")?;
    let mut out = std::collections::BTreeMap::new();
    for rec in records {
        let (Some(id), Some(map)) = (rec.get("id").and_then(serde_yaml::Value::as_str), rec.as_mapping()) else {
            continue;
        };
        let peak = map
            .iter()
            .filter_map(|(k, v)| {
                let key = k.as_str()?;
                let named = (key.starts_with("estimated_peak_memory_") || key.starts_with("measured_peak_memory_"))
                    && key.ends_with("_gb");
                if named { v.as_f64() } else { None }
            })
            .fold(None::<f64>, |best, v| Some(best.map_or(v, |b| b.max(v))));
        if let Some(p) = peak {
            out.insert(id.to_string(), p.ceil());
        }
    }
    Ok(out)
}

/// `lanes memory`: read, or change and read, the budget and the lanes'
/// peak figures, and sum every combination the matrix allows.
pub async fn memory(
    state: &AppState,
    req: &llmctl_client::wire::LaneMemoryRequest,
) -> Result<llmctl_client::wire::LaneMemoryResponse, LanesError> {
    let path = store_path(state, &req.workload);
    let mut store = read_store(&path).await?.ok_or_else(|| {
        refuse(format!(
            "{} has no lane records; `lanes import {}` first",
            req.workload, req.workload
        ))
    })?;
    let before = (store.budget_gib, store.peaks_gib.clone());
    if let Some(file) = &req.from_registry {
        let text = tokio::fs::read_to_string(file)
            .await
            .map_err(|e| refuse(format!("reading {file}: {e}")))?;
        let peaks = registry_peaks(&text).map_err(|e| refuse(format!("{file}: {e}")))?;
        for (id, gib) in peaks {
            if store.records.lane(&id).is_some() {
                store.peaks_gib.insert(id, gib);
            }
        }
    }
    for (lane, gib) in &req.peaks {
        if store.records.lane(lane).is_none() {
            return Err(refuse(format!("{} has no lane {lane}", req.workload)));
        }
        match gib {
            Some(g) if g.is_finite() && *g >= 0.0 => {
                store.peaks_gib.insert(lane.clone(), *g);
            }
            Some(g) => return Err(refuse(format!("{g} GiB is not a peak figure for {lane}"))),
            None => {
                store.peaks_gib.remove(lane);
            }
        }
    }
    if let Some(b) = req.budget_gib {
        store.budget_gib = (b > 0).then_some(b);
    }
    let changed = before != (store.budget_gib, store.peaks_gib.clone());
    if changed {
        save_store(&path, &store).await.map_err(LanesError::Io)?;
        tracing::info!(workload = %req.workload, budget = ?store.budget_gib, "lane memory figures changed");
    }
    let report = store.budget_report(&store.records);
    Ok(llmctl_client::wire::LaneMemoryResponse {
        workload: store.workload.clone(),
        budget_gib: store.budget_gib,
        peaks: store.peaks_gib.clone(),
        lines: match &report {
            Some(r) => r.lines.clone(),
            None => vec!["no memory budget is set: `lanes memory <workload> --budget <GiB>`".into()],
        },
        over: report.map(|r| r.over).unwrap_or_default(),
        changed,
    })
}

/// `lanes logs`: the lane's own output. llama-swap keeps a buffer per
/// lane and serves it on `/logs/stream/<lane>`: first what it has, then
/// new lines for as long as the connection stays open. This reads what
/// arrives in a short window and returns the newest `lines`.
pub async fn lane_logs(
    state: &AppState,
    req: &llmctl_client::wire::LaneLogsRequest,
) -> Result<llmctl_client::wire::LaneLogsResponse, LanesError> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let name = WorkloadName::new(&req.workload).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let def = inv
        .workload(&name)
        .cloned()
        .ok_or_else(|| OpsError::UnknownWorkload(req.workload.clone()))?;
    let Some(store) = read_store(&store_path(state, &req.workload)).await? else {
        return Err(refuse(format!(
            "{} has no lane records; `lanes import {}` first",
            req.workload, req.workload
        )));
    };
    if store.records.lane(&req.lane).is_none() {
        return Err(refuse(format!("{} has no lane {}", req.workload, req.lane)));
    }
    let port = def.port;
    // HTTP/1.0: the answer is not chunked and ends when we stop reading.
    let read = async {
        let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.ok()?;
        stream
            .write_all(format!("GET /logs/stream/{} HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n", req.lane).as_bytes())
            .await
            .ok()?;
        let mut all = Vec::new();
        let mut buf = vec![0u8; 64 * 1024];
        // The buffer comes at once; a quiet half second means it is all here.
        loop {
            match tokio::time::timeout(Duration::from_millis(500), stream.read(&mut buf)).await {
                Ok(Ok(0)) | Ok(Err(_)) | Err(_) => break,
                Ok(Ok(n)) => {
                    all.extend_from_slice(&buf[..n]);
                    if all.len() > 8 * 1024 * 1024 {
                        break;
                    }
                }
            }
        }
        Some(all)
    };
    let raw = tokio::time::timeout(Duration::from_secs(10), read)
        .await
        .ok()
        .flatten()
        .ok_or_else(|| LanesError::Io(format!("{} did not answer on port {port}", req.workload)))?;
    let text = String::from_utf8_lossy(&raw);
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((text.as_ref(), ""));
    let status = head.split_whitespace().nth(1).unwrap_or("");
    if status != "200" {
        return Err(refuse(format!(
            "llama-swap answered {status} for the log of {}; it keeps a log only for lanes it knows",
            req.lane
        )));
    }
    let all: Vec<&str> = body.lines().collect();
    let keep = (req.lines.max(1) as usize).min(5000);
    Ok(llmctl_client::wire::LaneLogsResponse {
        workload: req.workload.clone(),
        lane: req.lane.clone(),
        lines: all[all.len().saturating_sub(keep)..].iter().map(|l| l.to_string()).collect(),
        available: all.len(),
    })
}

// ---- the effect (run by the executor) ------------------------------------------------

/// Write `content` over `path` (refusing if the file is not `expect_digest`
/// any more), then confirm llama-swap reloaded it. On success, re-import
/// the store from `content` and commit to the config's git repo.
#[allow(clippy::too_many_arguments)]
pub async fn write_and_confirm(
    workload: &str,
    path: &str,
    content: &str,
    expect_digest: &str,
    log_path: Option<&str>,
    port: u16,
    lanes: &[String],
    timeout: Duration,
    store_path: &str,
    summary: &str,
) -> Result<String, String> {
    let current = tokio::fs::read(path)
        .await
        .map_err(|e| format!("reading {path}: {e}"))?;
    let current_digest = bytes_digest(&current);
    let target_digest = bytes_digest(content.as_bytes());
    let already = current_digest == target_digest;
    if !already && current_digest != expect_digest {
        return Err(format!(
            "{path} is {current_digest}, not the planned {expect_digest}: changed since the plan; refusing to overwrite"
        ));
    }
    let log_offset = match log_path {
        Some(l) => tokio::fs::metadata(l).await.map(|m| m.len()).ok(),
        None => None,
    };
    if !already {
        // The bytes being replaced stay next to the file for the owner.
        let prev = format!("{path}.llmctl-previous");
        write_atomic(Path::new(&prev), &current).await?;
        write_atomic(Path::new(path), content.as_bytes()).await?;
    }
    let mut report = confirm_reload(log_path, log_offset, port, lanes, timeout, already).await?;

    // The store follows what is now live.
    if let Ok(records) = swap::import(content) {
        let sp = Path::new(store_path);
        let mut store = read_store(sp).await.ok().flatten();
        if let Some(s) = store.as_mut() {
            let previous = std::mem::replace(&mut s.records, records);
            crate::lane_version::follow_write(s, &previous).await;
            let kept: Vec<String> = s.records.lanes.iter().map(|l| l.id.clone()).collect();
            s.peaks_gib.retain(|id, _| kept.contains(id));
            s.file_digest = target_digest.clone();
            s.revision += 1;
            s.updated_at = Utc::now();
            if let Err(e) = save_store(sp, s).await {
                report.push_str(&format!("; lane store not updated: {e}"));
            }
        }
    }
    if !already {
        report.push_str(&git_commit(path, &format!("llmctl-next({workload}): {summary}")).await);
    }
    Ok(report)
}

/// Wait for llama-swap's own word on the reload, then check it serves the
/// expected lanes. Without a log, `/v1/models` alone decides (weaker: an
/// unchanged lane set cannot prove the new file was read).
async fn confirm_reload(
    log_path: Option<&str>,
    log_offset: Option<u64>,
    port: u16,
    lanes: &[String],
    timeout: Duration,
    unchanged: bool,
) -> Result<String, String> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut notes = Vec::new();
    if let (Some(log), Some(offset), false) = (log_path, log_offset, unchanged) {
        loop {
            let tail = read_from(log, offset).await;
            let reloaded = tail.contains("configuration reloaded");
            if let Some(err) = tail.lines().find(|l| {
                let l = l.to_ascii_lowercase();
                l.contains("[error]") && (l.contains("config") || l.contains("yaml") || l.contains("matrix"))
            }) {
                return Err(format!("llama-swap rejected the config: {}", err.trim()));
            }
            if reloaded {
                notes.push("llama-swap logged `configuration reloaded`".to_string());
                for l in tail.lines().filter(|l| l.contains("[ERROR]")) {
                    notes.push(format!("after reload: {}", l.trim()));
                }
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(format!(
                    "llama-swap did not log a reload within {timeout:?} (log {log})"
                ));
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    } else if unchanged {
        notes.push("file already had the planned bytes; no reload expected".into());
    } else {
        notes.push("no log path: confirmed by /v1/models only".into());
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    // Serving check: every expected lane listed.
    loop {
        match crate::probe_http::probe_models(port, Duration::from_secs(3)).await {
            crate::probe_http::HttpProbe::Models(ids) => {
                let missing: Vec<&String> = lanes.iter().filter(|l| !ids.contains(l)).collect();
                if missing.is_empty() {
                    notes.push(format!("/v1/models lists all {} lanes", lanes.len()));
                    return Ok(notes.join("; "));
                }
                if tokio::time::Instant::now() >= deadline {
                    return Err(format!("llama-swap does not list {missing:?} after the reload"));
                }
            }
            other if tokio::time::Instant::now() >= deadline => {
                return Err(format!("llama-swap /v1/models after the reload: {other:?}"));
            }
            _ => {}
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

async fn read_from(path: &str, offset: u64) -> String {
    use tokio::io::{AsyncReadExt, AsyncSeekExt};
    let Ok(mut f) = tokio::fs::File::open(path).await else {
        return String::new();
    };
    let len = f.metadata().await.map(|m| m.len()).unwrap_or(0);
    // A rotated (shorter) log starts over.
    let start = if len < offset { 0 } else { offset };
    if f.seek(std::io::SeekFrom::Start(start)).await.is_err() {
        return String::new();
    }
    let mut buf = Vec::new();
    let _ = f.take(1 << 20).read_to_end(&mut buf).await;
    String::from_utf8_lossy(&buf).into_owned()
}

/// Commit just this file when it lives in a git work tree. Never pushes;
/// a failure is reported, not fatal (the journal is the record).
async fn git_commit(path: &str, message: &str) -> String {
    let p = Path::new(path);
    let (Some(dir), Some(file)) = (p.parent(), p.file_name()) else {
        return String::new();
    };
    let run = |args: Vec<String>| {
        let mut c = tokio::process::Command::new("git");
        c.arg("-C").arg(dir).args(args).kill_on_drop(true);
        c
    };
    let inside = run(vec!["rev-parse".into(), "--is-inside-work-tree".into()]).output().await;
    if !matches!(&inside, Ok(o) if o.status.success()) {
        return "; not in a git repo (no commit)".into();
    }
    let file = file.to_string_lossy().to_string();
    let out = run(vec![
        "commit".into(),
        "--quiet".into(),
        "-m".into(),
        message.into(),
        "--".into(),
        file.clone(),
    ])
    .output()
    .await;
    match out {
        Ok(o) if o.status.success() => "; committed to the config repo".into(),
        Ok(o) => {
            // An untracked file needs an add first; never stage anything else.
            let added = run(vec!["add".into(), "--".into(), file.clone()]).output().await;
            if matches!(&added, Ok(a) if a.status.success()) {
                if let Ok(o2) = run(vec![
                    "commit".into(),
                    "--quiet".into(),
                    "-m".into(),
                    message.into(),
                    "--".into(),
                    file,
                ])
                .output()
                .await
                {
                    if o2.status.success() {
                        return "; committed to the config repo".into();
                    }
                }
            }
            format!(
                "; git commit failed: {}",
                String::from_utf8_lossy(&o.stderr).trim().chars().take(200).collect::<String>()
            )
        }
        Err(e) => format!("; git not runnable: {e}"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_inflight_event_is_read_and_everything_else_is_skipped() {
        use super::parse_inflight_event as p;
        assert_eq!(p(r#"data:{"type":"inflight","data":"{\"total\":3}"}"#), Some(3));
        assert_eq!(p(r#"data: {"type":"inflight","data":{"total":0}}"#), Some(0));
        for other in [
            "event:message",
            "",
            r#"data:{"type":"logData","data":"{\"data\":\"x\"}"}"#,
            r#"data:{"type":"inflight","data":"{\"total\":-1}"}"#,
            r#"data:{"type":"inflight","data":"nonsense"}"#,
            "data:not json",
        ] {
            assert_eq!(p(other), None, "{other}");
        }
    }

    use super::*;

    #[tokio::test]
    async fn a_write_refuses_a_file_changed_since_the_plan() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.yaml");
        std::fs::write(&cfg, "models: {}\n").unwrap();
        let err = write_and_confirm(
            "swap",
            cfg.to_str().unwrap(),
            "models:\n  a:\n    cmd: x\n",
            "sha256:not-the-file",
            None,
            1,
            &[],
            Duration::from_millis(200),
            dir.path().join("store.json").to_str().unwrap(),
            "test",
        )
        .await
        .unwrap_err();
        assert!(err.contains("changed since the plan"), "{err}");
        assert_eq!(std::fs::read_to_string(&cfg).unwrap(), "models: {}\n");
    }

    #[tokio::test]
    async fn a_write_without_a_reload_fails_and_keeps_the_previous_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.yaml");
        let log = dir.path().join("swap.log");
        std::fs::write(&cfg, "models: {}\n").unwrap();
        std::fs::write(&log, "[INFO] started\n").unwrap();
        let before = bytes_digest(b"models: {}\n");
        let err = write_and_confirm(
            "swap",
            cfg.to_str().unwrap(),
            "models:\n  a:\n    cmd: x\n",
            &before,
            Some(log.to_str().unwrap()),
            1,
            &["a".into()],
            Duration::from_millis(600),
            dir.path().join("store.json").to_str().unwrap(),
            "test",
        )
        .await
        .unwrap_err();
        assert!(err.contains("did not log a reload"), "{err}");
        // The new bytes are on disk (the executor's rollback restores);
        // the replaced bytes sit beside the file.
        assert_eq!(
            std::fs::read_to_string(dir.path().join("config.yaml.llmctl-previous")).unwrap(),
            "models: {}\n"
        );
    }

    #[tokio::test]
    async fn a_logged_config_error_is_a_refusal() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.yaml");
        let log = dir.path().join("swap.log");
        std::fs::write(&cfg, "models: {}\n").unwrap();
        std::fs::write(&log, "").unwrap();
        let before = bytes_digest(b"models: {}\n");
        let log2 = log.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            let mut s = std::fs::read_to_string(&log2).unwrap();
            s.push_str("[INFO] reloading configuration\n[ERROR] unable to load config: bad matrix\n");
            std::fs::write(&log2, s).unwrap();
        });
        let err = write_and_confirm(
            "swap",
            cfg.to_str().unwrap(),
            "models:\n  a:\n    cmd: x\n",
            &before,
            Some(log.to_str().unwrap()),
            1,
            &["a".into()],
            Duration::from_secs(3),
            dir.path().join("store.json").to_str().unwrap(),
            "test",
        )
        .await
        .unwrap_err();
        assert!(err.contains("rejected the config"), "{err}");
    }
}
