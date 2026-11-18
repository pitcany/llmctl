//! llama.cpp router models on the agent (M5d items 1–3).
//!
//! One store per router workload (`<data>/llamacpp/<workload>.json`): the
//! preset records imported from its INI, the INI path, the digest of the
//! bytes the records stand for, and llmctl-only facts per model (measured
//! or declared VRAM need, host-RAM share, Mac peak, VRAM floor). Model
//! hooks live in `<data>/llamacpp/<workload>.hooks.json`, written by the
//! owner only.
//!
//! - `import` previews (round trip, validation, the router's command line)
//!   and persists with the preview's digest. The INI is not touched.
//! - `edit` plans preset changes and, confirmed, admits a `router_write`:
//!   the render is written atomically (previous bytes kept beside it),
//!   `GET /models?reload=1` re-reads it, and the model list must follow;
//!   a write that does not take is rolled back the same way.
//! - `load`/`unload` admit a `router_model` operation after the load
//!   decision (fit or budget, `--models-max`) and the pre-load guard said
//!   yes. The router runs with `--no-models-autoload`, so this is the only
//!   way a model loads. An `evict` load stops the at-start workloads on
//!   the router's cards first and starts them again if the load fails.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use llmctl_client::llamacpp::{
    ModelFacts, RouterEdit, RouterEditRequest, RouterFactsRequest, RouterImportRequest,
    RouterImportResponse, RouterLoadRequest, RouterLoadResponse, RouterModelView,
    RouterModelsResponse, RouterPlanResponse,
};
use llmctl_client::wire::AdmitResponse;
use llmctl_core::fit::CardMib;
use llmctl_core::llamacpp::{
    self as lc, LoadDecision, MemoryCheck, ModelHookWhen, ModelHooks, RouterModelStatus, RouterPreset,
};
use llmctl_core::{DesiredIntent, OpKind, Ownership, Plan, PlanStep, Platform, ProcessState, WorkloadName};

use crate::api::ApiError;
use crate::app::AppState;
use crate::lanes::{bytes_digest, git_commit, refuse, write_atomic, LanesError};
use crate::ops::{AdmitOutcome, AdmitParams, OpsError};

/// Seconds a preset write waits for the router to list the new models.
pub const RELOAD_TIMEOUT_SECS: u64 = 60;
/// How long the model list may take to follow an answered reload.
const RELOAD_SETTLE: Duration = Duration::from_secs(5);
/// Longest one model load may take (gpt-oss-120b: ~5 min from disk; its
/// unit allowed 20).
pub const LOAD_TIMEOUT_SECS: u64 = 1200;
/// Longest a model may take to leave after the unload request.
pub const UNLOAD_TIMEOUT_SECS: u64 = 120;
/// Test seam: free host RAM in GiB instead of `/proc/meminfo`.
pub const FREE_RAM_ENV: &str = "LLMCTL_NEXT_FREE_RAM_GIB";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStore {
    pub workload: String,
    pub preset_path: String,
    pub records: RouterPreset,
    /// `sha256:` of the INI bytes the records stand for.
    pub file_digest: String,
    pub revision: u64,
    pub updated_at: DateTime<Utc>,
    /// `--models-max` the router runs with.
    pub models_max: u32,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub facts: BTreeMap<String, ModelFacts>,
    /// GiB of model peaks a unified-memory host can hold at once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_gib: Option<u64>,
}

pub fn store_path(state: &AppState, workload: &str) -> PathBuf {
    state.paths.data_dir.join("llamacpp").join(format!("{workload}.json"))
}

pub fn hooks_path(state: &AppState, workload: &str) -> PathBuf {
    state.paths.data_dir.join("llamacpp").join(format!("{workload}.hooks.json"))
}

async fn read_store(path: &Path) -> Result<Option<RouterStore>, LanesError> {
    match tokio::fs::read(path).await {
        Ok(b) => serde_json::from_slice(&b)
            .map(Some)
            .map_err(|e| LanesError::Io(format!("router store {}: {e}", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(LanesError::Io(format!("router store {}: {e}", path.display()))),
    }
}

async fn save_store(path: &Path, store: &RouterStore) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(store).map_err(|e| e.to_string())?;
    write_atomic(path, &bytes).await
}

async fn require_store(state: &AppState, workload: &str) -> Result<RouterStore, LanesError> {
    read_store(&store_path(state, workload)).await?.ok_or_else(|| {
        refuse(format!(
            "{workload} has no router records; run `llamacpp import {workload}` first"
        ))
    })
}

/// The owner's model hooks; absent = none. A file that does not validate
/// refuses every operation that would run hooks.
async fn read_hooks(state: &AppState, workload: &str) -> Result<ModelHooks, LanesError> {
    let path = hooks_path(state, workload);
    let hooks: ModelHooks = match tokio::fs::read(&path).await {
        Ok(b) => serde_json::from_slice(&b)
            .map_err(|e| refuse(format!("{}: {e}", path.display())))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ModelHooks::default(),
        Err(e) => return Err(LanesError::Io(format!("{}: {e}", path.display()))),
    };
    let problems = hooks.validate();
    if !problems.is_empty() {
        return Err(refuse(format!("{}: {}", path.display(), problems.join("; "))));
    }
    Ok(hooks)
}

async fn router_def(state: &AppState, workload: &str) -> Result<(Arc<llmctl_core::HostInventory>, llmctl_core::WorkloadDefinition), LanesError> {
    let name = WorkloadName::new(workload).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let def = inv
        .workload(&name)
        .cloned()
        .ok_or_else(|| OpsError::UnknownWorkload(workload.to_string()))?;
    Ok((inv, def))
}

// ---- the router's API ------------------------------------------------------------

/// `GET /models` on the router. `None` = it did not answer in shape.
pub async fn list_models(port: u16, timeout: Duration) -> Option<Vec<RouterModelStatus>> {
    list_models_at(port, "/models", timeout).await
}

async fn list_models_at(port: u16, path: &str, timeout: Duration) -> Option<Vec<RouterModelStatus>> {
    match crate::probe_http::request(port, hyper::Method::GET, path, None, timeout).await {
        Ok((200, body)) => serde_json::from_slice::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| lc::parse_models(&v)),
        _ => None,
    }
}

fn holding(rows: &[RouterModelStatus]) -> Vec<String> {
    rows.iter().filter(|r| r.state.holds_memory()).map(|r| r.id.clone()).collect()
}

/// The effect of a `LoadRouterModel` step: `POST /models/load`, then the
/// router must list it `loaded`. A failed load is reported with its exit
/// code. A `failed` flag left from an earlier attempt counts only once
/// this load was seen `loading` (or after a grace period).
pub async fn load_and_confirm(port: u16, model: &str, timeout: Duration) -> Result<String, String> {
    let before = list_models(port, Duration::from_secs(5))
        .await
        .ok_or_else(|| "the router does not answer GET /models".to_string())?;
    match before.iter().find(|r| r.id == model) {
        None => return Err(format!("the router does not list {model}")),
        Some(r) if r.state == lc::ModelState::Loaded => {
            return Ok(format!("{model} was loaded already; nothing loaded"))
        }
        _ => {}
    }
    let body = serde_json::json!({"model": model});
    match crate::probe_http::request(port, hyper::Method::POST, "/models/load", Some(&body), Duration::from_secs(30)).await {
        Ok((status, _)) if (200..300).contains(&status) => {}
        Ok((status, body)) => {
            return Err(format!(
                "the router answered {status} to the load of {model}: {}",
                String::from_utf8_lossy(&body).chars().take(200).collect::<String>()
            ))
        }
        Err(()) => return Err(format!("the router did not answer the load of {model}")),
    }
    let started = tokio::time::Instant::now();
    let deadline = started + timeout;
    let mut seen_loading = false;
    loop {
        if let Some(rows) = list_models(port, Duration::from_secs(5)).await {
            match rows.iter().find(|r| r.id == model) {
                Some(r) if r.state == lc::ModelState::Loaded => {
                    return Ok(format!(
                        "{model} loaded after {}s; holding memory now: {}",
                        started.elapsed().as_secs(),
                        holding(&rows).join(", ")
                    ))
                }
                Some(r) if r.state == lc::ModelState::Loading => seen_loading = true,
                Some(r)
                    if r.state == lc::ModelState::Unloaded
                        && r.failed
                        && (seen_loading || started.elapsed() > Duration::from_secs(10)) =>
                {
                    return Err(format!(
                        "the router reports the load of {model} failed (exit code {})",
                        r.exit_code.map(|c| c.to_string()).unwrap_or_else(|| "unknown".into())
                    ))
                }
                None => return Err(format!("{model} left the router's list during the load")),
                _ => {}
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!("{model} did not load within {}s", timeout.as_secs()));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// The effect of an `UnloadRouterModel` step: the router's own unload, then
/// the model must be listed `unloaded`. Not loaded = nothing to do.
pub async fn unload_and_confirm(port: u16, model: &str, timeout: Duration) -> Result<String, String> {
    let rows = list_models(port, Duration::from_secs(5))
        .await
        .ok_or_else(|| "the router does not answer GET /models".to_string())?;
    match rows.iter().find(|r| r.id == model) {
        None => return Ok(format!("the router does not list {model}; nothing unloaded")),
        Some(r) if !r.state.holds_memory() => return Ok(format!("{model} was not loaded; nothing unloaded")),
        _ => {}
    }
    let body = serde_json::json!({"model": model});
    match crate::probe_http::request(port, hyper::Method::POST, "/models/unload", Some(&body), Duration::from_secs(30)).await {
        Ok((status, _)) if (200..300).contains(&status) => {}
        Ok((status, body)) => {
            return Err(format!(
                "the router answered {status} to the unload of {model}: {}",
                String::from_utf8_lossy(&body).chars().take(200).collect::<String>()
            ))
        }
        Err(()) => return Err(format!("the router did not answer the unload of {model}")),
    }
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if let Some(rows) = list_models(port, Duration::from_secs(5)).await {
            if rows.iter().find(|r| r.id == model).is_none_or(|r| !r.state.holds_memory()) {
                return Ok(format!("{model} unloaded; the router lists it unloaded"));
            }
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(format!("{model} still holds memory {}s after the unload", timeout.as_secs()));
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

/// The effect of a `WriteRouterPreset` step: write `content` over `path`
/// (only if it still is `expect_digest`), keep the bytes it replaces in
/// `<path>.llmctl-previous`, reload, and confirm the router lists every
/// model of the new records and none the write removed. The store
/// follows; the file is committed when it lives in a git work tree.
#[allow(clippy::too_many_arguments)]
pub async fn write_and_confirm(
    workload: &str,
    path: &str,
    content: &str,
    expect_digest: &str,
    port: u16,
    models: &[String],
    timeout: Duration,
    store_path: &str,
    summary: &str,
) -> Result<String, String> {
    let current = tokio::fs::read(path).await.map_err(|e| format!("reading {path}: {e}"))?;
    let current_digest = bytes_digest(&current);
    let target_digest = bytes_digest(content.as_bytes());
    let already = current_digest == target_digest;
    if !already && current_digest != expect_digest {
        return Err(format!(
            "{path} is {current_digest}, not the planned {expect_digest}: changed since the plan; refusing to overwrite"
        ));
    }
    let sp = Path::new(store_path);
    let mut store = read_store(sp).await.ok().flatten();
    let removed: Vec<String> = store
        .as_ref()
        .map(|s| s.records.model_ids().into_iter().filter(|id| !models.contains(id)).collect())
        .unwrap_or_default();
    if !already {
        write_atomic(Path::new(&format!("{path}.llmctl-previous")), &current).await?;
        write_atomic(Path::new(path), content.as_bytes()).await?;
    }
    let mut deadline = tokio::time::Instant::now() + timeout;
    let mut path_q = "/models?reload=1";
    let mut report = loop {
        // Polls re-read the sources until one reload is answered; later
        // polls only list. The reload is synchronous in llama.cpp, so the
        // list settles within seconds of that answer.
        let rows = list_models_at(port, path_q, Duration::from_secs(10)).await;
        if rows.is_some() && path_q != "/models" {
            path_q = "/models";
            deadline = deadline.min(tokio::time::Instant::now() + RELOAD_SETTLE);
        }
        if let Some(rows) = rows {
            let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
            let missing: Vec<&String> = models.iter().filter(|m| !ids.contains(&m.as_str())).collect();
            let lingering: Vec<&String> = removed.iter().filter(|m| ids.contains(&m.as_str())).collect();
            if missing.is_empty() && lingering.is_empty() {
                break format!(
                    "{}reloaded; the router lists all {} models{}",
                    if already { "file already had the planned bytes; " } else { "" },
                    models.len(),
                    if removed.is_empty() { String::new() } else { format!(", {} removed", removed.join(", ")) }
                );
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(format!(
                    "after the reload the router lists {} (missing {missing:?}, still listed {lingering:?})",
                    ids.join(", ")
                ));
            }
        } else if tokio::time::Instant::now() >= deadline {
            return Err("the router did not answer GET /models after the write".into());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    if let (Some(s), Ok(records)) = (store.as_mut(), lc::import(content)) {
        s.records = records;
        s.file_digest = target_digest;
        s.revision += 1;
        s.updated_at = Utc::now();
        if let Err(e) = save_store(sp, s).await {
            report.push_str(&format!("; router store not updated: {e}"));
        }
    }
    if !already {
        report.push_str(&git_commit(path, &format!("llmctl-next({workload}): {summary}")).await);
    }
    Ok(report)
}

/// After a load: what the cards lost is the model's measured need.
pub async fn record_measured(store_path: &str, model: &str, before: &CardMib, after: Option<&CardMib>) -> String {
    let Some(after) = after else {
        return "; VRAM after the load could not be read (need not measured)".into();
    };
    let need: CardMib = before
        .iter()
        .filter_map(|(c, b)| {
            let a = after.get(c)?;
            let d = b.saturating_sub(*a);
            (d > 0).then_some((*c, d))
        })
        .collect();
    if need.is_empty() {
        return "; no VRAM drop measured".into();
    }
    let sp = Path::new(store_path);
    let Ok(Some(mut store)) = read_store(sp).await else {
        return "; router store unreadable (need not recorded)".into();
    };
    let f = store.facts.entry(model.to_string()).or_default();
    f.need_mib = Some(need.clone());
    f.need_source = Some(format!("measured {}", Utc::now().to_rfc3339()));
    match save_store(sp, &store).await {
        Ok(()) => format!("; measured need {}", llmctl_core::fit::format_mib(&need)),
        Err(e) => format!("; measured need not recorded: {e}"),
    }
}

/// Free host RAM (GiB): `MemAvailable` on Linux, or the test seam.
fn free_ram_gib() -> Option<f64> {
    if let Ok(v) = std::env::var(FREE_RAM_ENV) {
        return v.parse().ok();
    }
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib: f64 = text
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kib / (1024.0 * 1024.0))
}

/// The pre-load guard's disk side: missing shards and live `.incomplete`
/// downloads (`llama-server-guarded`, now the router's check).
async fn disk_problems(model: &str) -> (Vec<String>, Vec<String>) {
    let mut missing = Vec::new();
    for shard in lc::shard_files(model) {
        if !tokio::fs::metadata(&shard).await.is_ok_and(|m| m.is_file()) {
            missing.push(
                Path::new(&shard)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or(shard.clone()),
            );
        }
    }
    let mut incomplete = Vec::new();
    if let Some(stage) = lc::download_stage_dir(model) {
        if let Ok(mut rd) = tokio::fs::read_dir(&stage).await {
            while let Ok(Some(e)) = rd.next_entry().await {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.ends_with(".incomplete") {
                    incomplete.push(n);
                }
            }
        }
    }
    (missing, incomplete)
}

// ---- import / models / facts ------------------------------------------------------

/// A router's argv and inline environment.
type Launch = (Vec<String>, Vec<(String, String)>);

/// The router's argv and env from its pinned definition (`None` when not
/// managed: the command line cannot be checked).
fn launch_of(def: &llmctl_core::WorkloadDefinition) -> Option<Launch> {
    match &def.ownership {
        Ownership::Managed { definition, .. } => Some((
            definition.exec_start.first().map(|c| c.argv.clone()).unwrap_or_default(),
            definition.environment.clone(),
        )),
        Ownership::External => None,
    }
}

pub async fn import(state: &AppState, req: RouterImportRequest) -> Result<RouterImportResponse, LanesError> {
    let (_, def) = router_def(state, &req.workload).await?;
    let launch = launch_of(&def);
    let (argv_problems, argv_note, models_max) = match &launch {
        Some((argv, env)) => (
            lc::check_router_argv(argv, env),
            None,
            lc::models_max_from_argv(argv).unwrap_or(lc::DEFAULT_MODELS_MAX),
        ),
        None => (
            Vec::new(),
            Some(format!(
                "{} is not managed: its command line was not checked (it must run with --no-models-autoload)",
                req.workload
            )),
            lc::DEFAULT_MODELS_MAX,
        ),
    };
    let preset_path = req
        .preset_path
        .clone()
        .or_else(|| launch.as_ref().and_then(|(argv, _)| lc::preset_path_from_argv(argv)))
        .ok_or_else(|| refuse(format!("{}: no --models-preset in its definition; pass --preset <path>", req.workload)))?;
    if !Path::new(&preset_path).is_absolute() {
        return Err(refuse(format!("preset path {preset_path} is not absolute")));
    }
    let bytes = tokio::fs::read(&preset_path)
        .await
        .map_err(|e| LanesError::Io(format!("reading {preset_path}: {e}")))?;
    let file_digest = bytes_digest(&bytes);
    let text = String::from_utf8(bytes).map_err(|_| refuse(format!("{preset_path} is not UTF-8")))?;
    let mut resp = RouterImportResponse {
        workload: def.name.to_string(),
        preset_path: preset_path.clone(),
        file_digest: file_digest.clone(),
        records_digest: None,
        roundtrip: String::new(),
        validation: Vec::new(),
        argv_problems,
        argv_note,
        models: Vec::new(),
        notes_kept: 0,
        persisted: false,
    };
    let records = match lc::roundtrip(&text) {
        Ok((r, _)) => {
            resp.roundtrip = "equivalent".into();
            r
        }
        Err(e) => {
            resp.roundtrip = e.to_string();
            return Ok(resp);
        }
    };
    resp.validation = lc::validate(&records);
    resp.models = lc::model_records(&resp.workload, &records);
    resp.notes_kept = records.notes_count();
    let digest = bytes_digest(format!("{}|{}", records.digest(), file_digest).as_bytes());
    resp.records_digest = Some(digest.clone());
    let Some(confirm) = req.confirm else {
        return Ok(resp);
    };
    if confirm != digest {
        return Err(refuse(format!("stale confirmation: the preview digest is now {digest}")));
    }
    if !resp.argv_problems.is_empty() {
        return Err(refuse(format!(
            "the router's command line: {}",
            resp.argv_problems.join("; ")
        )));
    }
    if !resp.validation.is_empty() {
        return Err(refuse(format!("the preset does not validate: {}", resp.validation.join("; "))));
    }
    let path = store_path(state, &resp.workload);
    let existing = read_store(&path).await?;
    let revision = existing.as_ref().map(|s| s.revision + 1).unwrap_or(1);
    let (mut facts, budget_gib) = existing.map(|s| (s.facts, s.budget_gib)).unwrap_or_default();
    facts.retain(|id, _| records.section(id).is_some());
    save_store(
        &path,
        &RouterStore {
            workload: resp.workload.clone(),
            preset_path,
            records,
            file_digest,
            revision,
            updated_at: Utc::now(),
            models_max,
            facts,
            budget_gib,
        },
    )
    .await
    .map_err(LanesError::Io)?;
    resp.persisted = true;
    Ok(resp)
}

pub async fn models(state: &AppState, workload: &str) -> Result<RouterModelsResponse, LanesError> {
    let (_, def) = router_def(state, workload).await?;
    let store = require_store(state, workload).await?;
    let now = tokio::fs::read(&store.preset_path)
        .await
        .map(|b| bytes_digest(&b))
        .unwrap_or_else(|e| format!("unreadable: {e}"));
    let rows = list_models(def.port, Duration::from_secs(5)).await;
    let ids = store.records.model_ids();
    let models = lc::model_records(workload, &store.records)
        .into_iter()
        .map(|record| {
            let row = rows.as_ref().and_then(|r| r.iter().find(|x| x.id == record.id));
            RouterModelView {
                state: row.map(|r| r.state.clone()),
                failed: row.is_some_and(|r| r.failed),
                exit_code: row.and_then(|r| r.exit_code),
                facts: store.facts.get(&record.id).cloned().unwrap_or_default(),
                record,
            }
        })
        .collect();
    let hooks = read_hooks(state, workload)
        .await
        .map(|h| {
            h.hooks
                .iter()
                .map(|x| format!("{} {} {}: {}", x.model, x.when, x.name, x.argv.join(" ")))
                .collect()
        })
        .unwrap_or_else(|e| vec![format!("hooks file refused: {e}")]);
    Ok(RouterModelsResponse {
        workload: workload.to_string(),
        preset_path: store.preset_path.clone(),
        revision: store.revision,
        drift: now != store.file_digest,
        router_answers: rows.is_some(),
        models_max: store.models_max,
        budget_gib: store.budget_gib,
        models,
        unknown_listed: rows
            .as_ref()
            .map(|r| r.iter().filter(|x| !ids.contains(&x.id)).map(|x| x.id.clone()).collect())
            .unwrap_or_default(),
        validation: lc::validate(&store.records),
        hooks,
    })
}

pub async fn facts(state: &AppState, req: RouterFactsRequest) -> Result<RouterModelsResponse, LanesError> {
    let path = store_path(state, &req.workload);
    let mut store = require_store(state, &req.workload).await?;
    let model_facts = req.need_mib.is_some()
        || req.host_ram_gib.is_some()
        || req.peak_gib.is_some()
        || req.min_free_mib.is_some()
        || req.clear.iter().any(|c| c != "budget");
    if model_facts {
        let model = req
            .model
            .as_deref()
            .ok_or_else(|| refuse("name the model these facts are about"))?;
        if store.records.model(model).is_none() {
            return Err(refuse(format!("{} has no model {model} in its records", req.workload)));
        }
        let f = store.facts.entry(model.to_string()).or_default();
        for c in &req.clear {
            match c.as_str() {
                "need" => {
                    f.need_mib = None;
                    f.need_source = None;
                }
                "host_ram" => f.host_ram_gib = None,
                "peak" => f.peak_gib = None,
                "min_free" => f.min_free_mib = None,
                "budget" => {}
                other => return Err(refuse(format!("unknown fact {other:?} to clear"))),
            }
        }
        if let Some(n) = req.need_mib {
            if n.is_empty() || n.values().any(|m| *m == 0) {
                return Err(refuse("a declared need names at least one card, each above 0 MiB"));
            }
            f.need_mib = Some(n);
            f.need_source = Some(format!("declared {}", Utc::now().to_rfc3339()));
        }
        if let Some(g) = req.host_ram_gib {
            f.host_ram_gib = Some(g);
        }
        if let Some(g) = req.peak_gib {
            f.peak_gib = Some(g);
        }
        if let Some(m) = req.min_free_mib {
            f.min_free_mib = Some(m);
        }
    }
    if let Some(b) = req.budget_gib {
        store.budget_gib = Some(b);
    }
    if req.clear.iter().any(|c| c == "budget") {
        store.budget_gib = None;
    }
    store.updated_at = Utc::now();
    save_store(&path, &store).await.map_err(LanesError::Io)?;
    models(state, &req.workload).await
}

// ---- preset edits ---------------------------------------------------------------------

fn apply_edits(p: &mut RouterPreset, edits: &[RouterEdit]) -> Result<Vec<String>, LanesError> {
    let mut summary = Vec::new();
    for e in edits {
        match e {
            RouterEdit::Set { model, key, value } => {
                p.set(model, key, value).map_err(refuse)?;
                summary.push(format!("set {model} {key}"));
            }
            RouterEdit::Remove { model, key } => {
                if !p.remove_key(model, key).map_err(refuse)? {
                    return Err(refuse(format!("[{model}] has no {key}")));
                }
                summary.push(format!("remove {model} {key}"));
            }
            RouterEdit::AddModel { model } => {
                p.add_model(model).map_err(refuse)?;
                summary.push(format!("add model {}", model.id));
            }
            RouterEdit::RemoveModel { id } => {
                p.remove_model(id).map_err(refuse)?;
                summary.push(format!("remove model {id}"));
            }
        }
    }
    if summary.is_empty() {
        return Err(refuse("no edits"));
    }
    Ok(summary)
}

struct Planned {
    store: RouterStore,
    next: RouterPreset,
    rendered: String,
    current_bytes: Vec<u8>,
    resp: RouterPlanResponse,
}

async fn plan_inner(state: &AppState, workload: &str, edits: &[RouterEdit]) -> Result<Planned, LanesError> {
    let (_, def) = router_def(state, workload).await?;
    let store = require_store(state, workload).await?;
    let current_bytes = tokio::fs::read(&store.preset_path)
        .await
        .map_err(|e| LanesError::Io(format!("reading {}: {e}", store.preset_path)))?;
    if bytes_digest(&current_bytes) != store.file_digest {
        return Err(refuse(format!(
            "{} changed since the records were imported (edited by hand?); review it, then `llamacpp import {workload}` again",
            store.preset_path
        )));
    }
    let mut next = store.records.clone();
    let summary = apply_edits(&mut next, edits)?;
    let validation = lc::validate(&next);
    let rendered = lc::check_render(&next).map_err(|e| refuse(e.to_string()))?;
    let changes = lc::changes(&store.records, &next);
    // The router unloads a loaded model whose section changed or went.
    let (before, after) = (store.records.effective(), next.effective());
    let reload_unloads = list_models(def.port, Duration::from_secs(5))
        .await
        .map(|rows| {
            holding(&rows)
                .into_iter()
                .filter(|id| before.get(id) != after.get(id))
                .collect()
        })
        .unwrap_or_default();
    let plan_digest = bytes_digest(
        format!("{}|{}|{}", next.digest(), store.file_digest, bytes_digest(rendered.as_bytes())).as_bytes(),
    );
    let resp = RouterPlanResponse {
        workload: workload.to_string(),
        summary: summary.join("; "),
        changes,
        validation,
        reload_unloads,
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

pub async fn plan(state: &AppState, workload: &str, edits: &[RouterEdit]) -> Result<RouterPlanResponse, LanesError> {
    Ok(plan_inner(state, workload, edits).await?.resp)
}

pub async fn apply(
    state: &AppState,
    workload: &str,
    edits: &[RouterEdit],
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
        return Err(refuse(format!("the preset would not validate: {}", p.resp.validation.join("; "))));
    }
    if !p.resp.bytes_changed {
        return Err(refuse("the render equals the live file; nothing to write"));
    }
    let (inv, def) = router_def(state, workload).await?;
    let store_file = store_path(state, workload).display().to_string();
    let previous = String::from_utf8(p.current_bytes.clone()).map_err(|_| refuse("the live preset is not UTF-8"))?;
    let step = |content: String, expect: String, models: Vec<String>, summary: String| PlanStep::WriteRouterPreset {
        workload: def.name.clone(),
        path: p.store.preset_path.clone(),
        content,
        expect_digest: expect,
        port: def.port,
        models,
        timeout_secs: RELOAD_TIMEOUT_SECS,
        store_path: store_file.clone(),
        summary,
    };
    let plan = Plan {
        id: llmctl_core::OperationId::generate(),
        kind: OpKind::RouterWrite,
        target: def.name.clone(),
        revision: inv.revision,
        expected_model: def.model.clone(),
        affected_gpus: def.gpus.clone(),
        steps: vec![step(
            p.rendered.clone(),
            p.store.file_digest.clone(),
            p.next.model_ids(),
            p.resp.summary.clone(),
        )],
        rollback: vec![step(
            previous,
            bytes_digest(p.rendered.as_bytes()),
            p.store.records.model_ids(),
            format!("roll back: {}", p.resp.summary),
        )],
    };
    Ok(state
        .ops
        .admit(AdmitParams {
            protocol: llmctl_client::wire::PROTOCOL_VERSION,
            idempotency_key,
            kind: OpKind::RouterWrite,
            workload: workload.to_string(),
            expected_revision: None,
            expected_stop_targets: None,
            force: false,
            profile: None,
            plan_override: Some(plan),
        })
        .await?)
}

// ---- load / unload ----------------------------------------------------------------------

fn hook_steps(hooks: &ModelHooks, workload: &WorkloadName, model: &str, when: ModelHookWhen) -> Vec<PlanStep> {
    hooks
        .at(model, when)
        .map(|h| PlanStep::RunHook {
            workload: workload.clone(),
            name: format!("{model}:{}", h.name),
            phase: when.to_string(),
            argv: h.argv.clone(),
            timeout_secs: h.timeout_secs,
            optional: h.optional,
        })
        .collect()
}

fn hook_labels(hooks: &ModelHooks, model: &str, whens: &[ModelHookWhen]) -> Vec<String> {
    whens
        .iter()
        .flat_map(|w| hooks.at(model, *w).map(move |h| format!("{w} {}", h.name)))
        .collect()
}

/// Running at-start workloads on the router's cards: what an `evict` load
/// stops (the router itself and on-demand services never).
async fn neighbours(
    state: &AppState,
    inv: &llmctl_core::HostInventory,
    def: &llmctl_core::WorkloadDefinition,
) -> Vec<llmctl_core::WorkloadDefinition> {
    let Ok(observed) = state.ops.engine().observe(inv).await else {
        return Vec::new();
    };
    let mut out: Vec<llmctl_core::WorkloadDefinition> = inv
        .workloads
        .iter()
        .filter(|w| w.name != def.name && w.gpu_claim.is_at_start())
        .filter(|w| w.gpus.iter().any(|g| def.gpus.contains(g)))
        .filter(|w| observed.iter().any(|o| o.workload == w.name && o.process == ProcessState::Running))
        .cloned()
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

pub async fn load(
    state: &AppState,
    req: &RouterLoadRequest,
) -> Result<(RouterLoadResponse, Option<AdmitOutcome>), LanesError> {
    let (inv, def) = router_def(state, &req.workload).await?;
    let store = require_store(state, &req.workload).await?;
    let hooks = read_hooks(state, &req.workload).await?;
    let Some(record) = store.records.model(&req.model) else {
        return Err(refuse(format!(
            "{} has no model {} in its records (`llamacpp models {}`)",
            req.workload, req.model, req.workload
        )));
    };
    let rows = list_models(def.port, Duration::from_secs(5)).await;
    let loaded_now = rows.as_deref().map(holding).unwrap_or_default();
    let mut resp = RouterLoadResponse {
        workload: req.workload.clone(),
        model: req.model.clone(),
        unload: req.unload,
        decision: String::new(),
        reason: String::new(),
        loaded_now: loaded_now.clone(),
        preload: Vec::new(),
        evicts: Vec::new(),
        hooks: Vec::new(),
        operation: None,
        replayed: false,
    };
    let (steps, rollback) = if req.unload {
        let Some(rows) = rows.as_deref() else {
            return Err(refuse("the router does not answer GET /models"));
        };
        if !rows.iter().any(|r| r.id == req.model && r.state.holds_memory()) {
            resp.decision = "not_loaded".into();
            resp.reason = format!("{} holds no memory; nothing to unload", req.model);
            return Ok((resp, None));
        }
        resp.decision = "unload".into();
        resp.reason = format!("unload {} from {}", req.model, req.workload);
        resp.hooks = hook_labels(&hooks, &req.model, &[ModelHookWhen::PreUnload, ModelHookWhen::PostUnload]);
        let mut steps = hook_steps(&hooks, &def.name, &req.model, ModelHookWhen::PreUnload);
        steps.push(PlanStep::UnloadRouterModel {
            workload: def.name.clone(),
            port: def.port,
            model: req.model.clone(),
            timeout_secs: UNLOAD_TIMEOUT_SECS,
        });
        steps.extend(hook_steps(&hooks, &def.name, &req.model, ModelHookWhen::PostUnload));
        // Nothing to undo: the model loads again by verb.
        (steps, Vec::new())
    } else {
        let facts = store.facts.get(&req.model).cloned().unwrap_or_default();
        let free = state.ops.engine().gpu_free_mib(&def.gpus).await;
        let memory = |free: Option<CardMib>| match inv.platform {
            Platform::Macos => MemoryCheck::Budget {
                budget_gib: store.budget_gib,
                need_gib: facts.peak_gib,
                loaded: loaded_now
                    .iter()
                    .filter(|id| **id != req.model)
                    .map(|id| (id.clone(), store.facts.get(id).and_then(|f| f.peak_gib)))
                    .collect(),
            },
            _ => MemoryCheck::Cards {
                need: facts.need_mib.clone(),
                free,
                host_ram_gib: facts.host_ram_gib,
                free_ram_gib: free_ram_gib(),
            },
        };
        let mut decision = lc::decide_load(&req.model, rows.as_deref(), store.models_max, &memory(free.clone()));
        let mut free_for_floor = free.clone();
        let mut evicted: Vec<llmctl_core::WorkloadDefinition> = Vec::new();
        if req.evict && inv.platform != Platform::Macos && matches!(&decision, LoadDecision::Refuse(r) if r.starts_with("does not fit")) {
            let near = neighbours(state, &inv, &def).await;
            if !near.is_empty() {
                // What the cards free when they go: what each holds now.
                let units: Vec<String> = near.iter().map(|w| w.unit.to_string()).collect();
                let after = match (free.clone(), state.ops.engine().gpu_occupancy(&def.gpus).await) {
                    (Some(mut f), Ok(occupants)) => {
                        for o in occupants.iter().filter(|o| o.unit.as_ref().is_some_and(|u| units.contains(u))) {
                            *f.entry(o.gpu).or_insert(0) += o.memory_mib;
                        }
                        Some(f)
                    }
                    _ => None,
                };
                let again = lc::decide_load(&req.model, rows.as_deref(), store.models_max, &memory(after.clone()));
                if matches!(again, LoadDecision::Load(_)) {
                    decision = LoadDecision::Load(format!(
                        "{} once {} {} stopped",
                        again.reason(),
                        near.iter().map(|w| w.name.to_string()).collect::<Vec<_>>().join(", "),
                        if near.len() == 1 { "is" } else { "are" }
                    ));
                    free_for_floor = after;
                    evicted = near;
                }
            }
        }
        resp.reason = decision.reason().to_string();
        resp.evicts = evicted.iter().map(|w| w.name.to_string()).collect();
        match &decision {
            LoadDecision::AlreadyLoaded(_) => {
                resp.decision = "already_loaded".into();
                return Ok((resp, None));
            }
            LoadDecision::Refuse(_) => {
                resp.decision = "refuse".into();
                return Ok((resp, None));
            }
            LoadDecision::Load(_) => {}
        }
        // The pre-load guard: every shard present, no live download, the
        // VRAM floor (after any eviction).
        if let Some(path) = &record.model {
            let (missing, incomplete) = disk_problems(path).await;
            resp.preload = lc::preload_problems(path, &missing, &incomplete, facts.min_free_mib, free_for_floor.as_ref());
        } else if facts.min_free_mib.is_some() {
            resp.preload = lc::preload_problems("", &[], &[], facts.min_free_mib, free_for_floor.as_ref());
        }
        if !resp.preload.is_empty() {
            resp.decision = "refuse".into();
            resp.reason = format!("pre-load check: {}", resp.preload.join("; "));
            return Ok((resp, None));
        }
        resp.decision = "load".into();
        resp.hooks = hook_labels(&hooks, &req.model, &[ModelHookWhen::PreLoad, ModelHookWhen::PostLoad]);
        let mut steps = Vec::new();
        let mut rollback = vec![PlanStep::UnloadRouterModel {
            workload: def.name.clone(),
            port: def.port,
            model: req.model.clone(),
            timeout_secs: UNLOAD_TIMEOUT_SECS,
        }];
        rollback.extend(hook_steps(&hooks, &def.name, &req.model, ModelHookWhen::PostUnload));
        for w in &evicted {
            steps.push(PlanStep::PersistIntent {
                workload: w.name.clone(),
                intent: DesiredIntent::Stopped { manual: false },
            });
            steps.push(PlanStep::StopUnit {
                workload: w.name.clone(),
                unit: w.unit.clone(),
            });
        }
        for w in evicted.iter().rev() {
            rollback.push(PlanStep::PersistIntent {
                workload: w.name.clone(),
                intent: DesiredIntent::Running,
            });
            rollback.push(PlanStep::StartUnit {
                workload: w.name.clone(),
                unit: w.unit.clone(),
            });
        }
        steps.extend(hook_steps(&hooks, &def.name, &req.model, ModelHookWhen::PreLoad));
        steps.push(PlanStep::LoadRouterModel {
            workload: def.name.clone(),
            port: def.port,
            model: req.model.clone(),
            timeout_secs: LOAD_TIMEOUT_SECS,
            measure_gpus: if inv.platform == Platform::Macos {
                Vec::new()
            } else {
                def.gpus.iter().map(|g| g.index).collect()
            },
            store_path: Some(store_path(state, &req.workload).display().to_string()),
            reason: resp.reason.clone(),
        });
        steps.extend(hook_steps(&hooks, &def.name, &req.model, ModelHookWhen::PostLoad));
        (steps, rollback)
    };
    if req.dry_run {
        return Ok((resp, None));
    }
    let plan = Plan {
        id: llmctl_core::OperationId::generate(),
        kind: OpKind::RouterModel,
        target: def.name.clone(),
        revision: inv.revision,
        expected_model: def.model.clone(),
        affected_gpus: def.gpus.clone(),
        steps,
        rollback,
    };
    let key = req.idempotency_key.clone().unwrap_or_else(|| {
        format!(
            "{}-{}-{}-{}",
            if req.unload { "routerunload" } else { "routerload" },
            req.workload,
            req.model.replace(|c: char| !c.is_ascii_alphanumeric() && c != '-' && c != '_' && c != '.', "_"),
            Utc::now().timestamp_millis()
        )
    });
    let outcome = state
        .ops
        .admit(AdmitParams {
            protocol: llmctl_client::wire::PROTOCOL_VERSION,
            idempotency_key: key,
            kind: OpKind::RouterModel,
            workload: req.workload.clone(),
            expected_revision: None,
            expected_stop_targets: None,
            force: req.force,
            profile: None,
            plan_override: Some(plan),
        })
        .await?;
    resp.replayed = matches!(outcome, AdmitOutcome::Replayed(_));
    Ok((resp, Some(outcome)))
}

// ---- routes -----------------------------------------------------------------------------

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/llamacpp/import", post(h_import))
        .route("/v1/llamacpp/edit", post(h_edit))
        .route("/v1/llamacpp/facts", post(h_facts))
        .route("/v1/llamacpp/loads", post(h_load))
        .route("/v1/llamacpp/install", post(h_install))
        .route("/v1/llamacpp/{workload}/models", get(h_models))
}

async fn h_import(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RouterImportRequest>,
) -> Result<Json<RouterImportResponse>, ApiError> {
    crate::api::check_protocol(req.protocol)?;
    let _one_at_a_time = state.adopt_lock.lock().await;
    Ok(Json(import(&state, req).await.map_err(ApiError::from_lanes)?))
}

async fn h_install(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::llamacpp::LaneEngineInstallRequest>,
) -> Result<Json<llmctl_client::llamacpp::LaneEngineInstallResponse>, ApiError> {
    crate::api::check_protocol(req.protocol)?;
    let _one_at_a_time = state.adopt_lock.lock().await;
    Ok(Json(crate::llamacpp_engine::install(&state, req).await.map_err(ApiError::from_lanes)?))
}

async fn h_models(
    State(state): State<Arc<AppState>>,
    AxPath(workload): AxPath<String>,
) -> Result<Json<RouterModelsResponse>, ApiError> {
    Ok(Json(models(&state, &workload).await.map_err(ApiError::from_lanes)?))
}

async fn h_facts(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RouterFactsRequest>,
) -> Result<Json<RouterModelsResponse>, ApiError> {
    crate::api::check_protocol(req.protocol)?;
    let _one_at_a_time = state.adopt_lock.lock().await;
    Ok(Json(facts(&state, req).await.map_err(ApiError::from_lanes)?))
}

async fn h_edit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RouterEditRequest>,
) -> Result<Response, ApiError> {
    crate::api::check_protocol(req.protocol)?;
    let _one_at_a_time = state.adopt_lock.lock().await;
    let Some(confirm) = req.confirm.clone() else {
        let plan = plan(&state, &req.workload, &req.edits).await.map_err(ApiError::from_lanes)?;
        return Ok(Json(plan).into_response());
    };
    let key = req.idempotency_key.clone().unwrap_or_else(|| {
        format!("llamacpp-{}-{}", req.workload, &confirm[confirm.len().saturating_sub(12)..])
    });
    let outcome = apply(&state, &req.workload, &req.edits, &confirm, key)
        .await
        .map_err(ApiError::from_lanes)?;
    let (row, replayed, status) = match outcome {
        AdmitOutcome::Admitted(row) => (row, false, StatusCode::ACCEPTED),
        AdmitOutcome::Replayed(row) => (row, true, StatusCode::OK),
    };
    Ok((
        status,
        Json(AdmitResponse {
            operation: crate::api::operation_view(&row),
            replayed,
        }),
    )
        .into_response())
}

async fn h_load(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RouterLoadRequest>,
) -> Result<Response, ApiError> {
    crate::api::check_protocol(req.protocol)?;
    let _one_at_a_time = state.adopt_lock.lock().await;
    let (mut resp, outcome) = load(&state, &req).await.map_err(ApiError::from_lanes)?;
    let status = match &outcome {
        Some(AdmitOutcome::Admitted(row)) => {
            resp.operation = Some(crate::api::operation_view(row));
            StatusCode::ACCEPTED
        }
        Some(AdmitOutcome::Replayed(row)) => {
            resp.operation = Some(crate::api::operation_view(row));
            StatusCode::OK
        }
        None => StatusCode::OK,
    };
    Ok((status, Json(resp)).into_response())
}
