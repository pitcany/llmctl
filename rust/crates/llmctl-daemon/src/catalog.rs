//! The model catalog on the agent (Rust M5c item 3, plan decision 4).
//!
//! One store per agent, `<data>/catalog.json`: the records of this host,
//! the bundle rows that did not map here (kept verbatim so the reverse
//! export is the whole bundle), the Python launch profiles (verbatim), and
//! the registry host names that mean this host.
//!
//! - `import` reads a Python registry bundle and reports, per row, what it
//!   becomes (new / no store / present / unmapped and why); only the
//!   report's `confirm` token applies it. The Python DB is never written.
//! - `export` writes a bundle `llmctl import-registry` accepts.
//! - `add/edit/enable/disable` change records; `scan` walks the declared
//!   stores for weights the catalog does not name and records whose
//!   weights are gone.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use llmctl_client::catalog::{
    CatalogImportRequest, CatalogImportResponse, CatalogListResponse, ModelAddRequest,
    ModelEditRequest, ModelEnableRequest, ModelWriteResponse, ScanFound, ScanRequest, ScanResponse,
};
use llmctl_core::catalog::{self as cat, CatalogModel, RowOutcome, RowReport};
use llmctl_core::{ModelStore, StoreKind};

use crate::app::AppState;

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("{0}")]
    Unknown(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Refused(String),
    #[error("{0}")]
    Io(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CatalogStore {
    #[serde(default)]
    pub revision: u64,
    /// Registry host names that mean this host (from `--as` on import).
    #[serde(default)]
    pub host_names: Vec<String>,
    #[serde(default)]
    pub models: Vec<CatalogModel>,
    /// Bundle rows that are not records here, verbatim.
    #[serde(default)]
    pub unmapped: Vec<Value>,
    /// Python launch profiles, verbatim.
    #[serde(default)]
    pub profiles: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
}

pub fn store_path(state: &AppState) -> PathBuf {
    state.paths.data_dir.join("catalog.json")
}

pub async fn load(state: &AppState) -> Result<CatalogStore, CatalogError> {
    load_at(&store_path(state)).await
}

/// The catalog store at a path: the agent's own (`load`) or a data dir
/// the operation manager reaches without an `AppState` (the `load` verb's
/// re-plan under the admission lock).
pub async fn load_at(path: &Path) -> Result<CatalogStore, CatalogError> {
    match tokio::fs::read(path).await {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| CatalogError::Io(format!("{}: {e}", path.display()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(CatalogStore::default()),
        Err(e) => Err(CatalogError::Io(format!("{}: {e}", path.display()))),
    }
}

async fn save(state: &AppState, mut store: CatalogStore) -> Result<u64, CatalogError> {
    store.revision += 1;
    store.updated_at = Some(Utc::now());
    crate::engine::write_json_atomic(&store_path(state), &store)
        .await
        .map_err(|e| CatalogError::Io(e.to_string()))?;
    Ok(store.revision)
}

fn host_id(state: &AppState) -> String {
    state.host.to_string()
}

/// Names a row's `host` may carry and still be this host.
fn names_for(state: &AppState, store: &CatalogStore, extra: &[String]) -> Vec<String> {
    let mut names = vec![host_id(state)];
    for n in store.host_names.iter().chain(extra) {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    names
}

async fn stores(state: &AppState) -> Vec<ModelStore> {
    state.ops.inventory().await.stores.clone()
}

/// A record by id, or by a name unique on this host.
fn find<'a>(store: &'a CatalogStore, key: &str) -> Result<&'a CatalogModel, CatalogError> {
    if let Some(m) = store.models.iter().find(|m| m.id == key) {
        return Ok(m);
    }
    let named: Vec<&CatalogModel> = store.models.iter().filter(|m| m.name == key).collect();
    match named.as_slice() {
        [one] => Ok(one),
        [] => Err(CatalogError::Unknown(format!("no catalog record {key:?} on this host"))),
        many => Err(CatalogError::Invalid(format!(
            "{key:?} names {} records; use an id: {}",
            many.len(),
            many.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", ")
        ))),
    }
}

pub async fn list(state: &AppState, all: bool) -> Result<CatalogListResponse, CatalogError> {
    let store = load(state).await?;
    let total = store.models.len();
    let models: Vec<CatalogModel> = store.models.into_iter().filter(|m| all || m.enabled).collect();
    Ok(CatalogListResponse {
        host: host_id(state),
        host_names: store.host_names,
        revision: store.revision,
        disabled_hidden: total - models.len(),
        models,
        unmapped: store.unmapped.len(),
        profiles: store.profiles.len(),
    })
}

pub async fn show(state: &AppState, key: &str) -> Result<CatalogModel, CatalogError> {
    let store = load(state).await?;
    find(&store, key).cloned()
}

fn digest(bytes: &[u8]) -> String {
    crate::profile::digest(bytes)
}

fn outcome_name(o: RowOutcome) -> &'static str {
    match o {
        RowOutcome::New => "new",
        RowOutcome::NoStore => "no_store",
        RowOutcome::Present => "present",
        RowOutcome::Unmapped => "unmapped",
    }
}

pub async fn import(state: &AppState, req: &CatalogImportRequest) -> Result<CatalogImportResponse, CatalogError> {
    let _one = state.adopt_lock.lock().await;
    let mut store = load(state).await?;
    let version = req.bundle.get("version").and_then(|v| v.as_i64()).unwrap_or(1);
    if version != 1 {
        return Err(CatalogError::Invalid(format!("registry bundle version {version}; this agent reads v1")));
    }
    let rows = req
        .bundle
        .get("models")
        .and_then(|m| m.as_array())
        .ok_or_else(|| CatalogError::Invalid("not a registry bundle: no `models` list".into()))?;
    let profiles: Vec<Value> = req
        .bundle
        .get("profiles")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();
    for n in &req.host_names {
        if n.trim().is_empty() {
            return Err(CatalogError::Invalid("--as needs a host name".into()));
        }
    }
    let names = names_for(state, &store, &req.host_names);
    let decl = stores(state).await;

    // The token binds the report to this bundle, these names, and the
    // catalog as it is now.
    let token = {
        let mut bytes = serde_json::to_vec(&req.bundle).unwrap_or_default();
        bytes.extend(names.join("\n").as_bytes());
        bytes.extend(store.revision.to_le_bytes());
        digest(&bytes)
    };

    let mut reports = Vec::new();
    let mut new_models: Vec<CatalogModel> = Vec::new();
    let mut new_unmapped: Vec<Value> = Vec::new();
    for row in rows {
        let name = row.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let runtime = row.get("runtime").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let host = row.get("host").and_then(|v| v.as_str()).map(str::to_string);
        let id = row.get("id").and_then(|v| v.as_str()).map(str::to_string);
        let mut report = RowReport { id: id.clone(), name, runtime, host, outcome: RowOutcome::New, store: None, reason: None };
        match cat::map_row(row, &names, &decl) {
            Ok(mut m) => {
                let key = m.registry_key();
                let present = store
                    .models
                    .iter()
                    .any(|r| (!m.id.is_empty() && r.id == m.id) || r.registry_key() == key);
                let twin = new_models.iter().find(|r| r.registry_key() == key).map(|r| r.name.clone());
                if present {
                    report.outcome = RowOutcome::Present;
                } else if let Some(twin) = twin {
                    // One record per engine and source on a host; the row
                    // stays in the store verbatim, so the export (and a
                    // Python import-registry of it) still has it.
                    report.outcome = RowOutcome::Unmapped;
                    report.reason = Some(format!(
                        "same runtime and source as {twin} earlier in this bundle (one record per engine and source; kept for the export)"
                    ));
                    if !store.unmapped.iter().chain(new_unmapped.iter()).any(|u| u == row) {
                        new_unmapped.push(row.clone());
                    }
                } else {
                    if m.id.is_empty() {
                        m.id = uuid::Uuid::new_v4().to_string();
                    }
                    report.store = m.store.clone();
                    if m.store.is_none() {
                        report.outcome = RowOutcome::NoStore;
                        report.reason = Some(match (&m.path, m.engine.as_str()) {
                            (_, "lmstudio") => "no lmstudio store declared".into(),
                            (_, "ollama") => "no ollama store declared".into(),
                            (Some(p), _) => format!("{p} is in no declared store (`stores add` one)"),
                            (None, _) => "no path, and no store by engine".into(),
                        });
                    }
                    new_models.push(m);
                }
            }
            Err(reason) => {
                report.outcome = RowOutcome::Unmapped;
                report.reason = Some(reason);
                let dup = store.unmapped.iter().chain(new_unmapped.iter()).any(|u| {
                    u == row || (id.is_some() && u.get("id").and_then(|v| v.as_str()) == id.as_deref())
                });
                if !dup {
                    new_unmapped.push(row.clone());
                }
            }
        }
        reports.push(report);
    }
    let mut counts = std::collections::BTreeMap::new();
    for r in &reports {
        *counts.entry(outcome_name(r.outcome).to_string()).or_insert(0) += 1;
    }
    let existing_profiles: Vec<String> = store
        .profiles
        .iter()
        .filter_map(|p| p.get("name").and_then(|n| n.as_str()).map(str::to_string))
        .collect();
    let fresh_profiles: Vec<Value> = profiles
        .into_iter()
        .filter(|p| {
            p.get("name")
                .and_then(|n| n.as_str())
                .is_some_and(|n| !existing_profiles.iter().any(|e| e == n))
        })
        .collect();
    let mut resp = CatalogImportResponse {
        rows: reports,
        counts,
        profiles: fresh_profiles.len(),
        confirm: token.clone(),
        applied: false,
        revision: store.revision,
    };
    let Some(confirm) = &req.confirm else { return Ok(resp) };
    if confirm != &token {
        return Err(CatalogError::Refused(
            "the confirm token is not this report's (the bundle, the host names or the catalog changed): run the dry run again".into(),
        ));
    }
    store.models.extend(new_models);
    store.unmapped.extend(new_unmapped);
    store.profiles.extend(fresh_profiles);
    for n in &req.host_names {
        if !store.host_names.contains(n) {
            store.host_names.push(n.clone());
        }
    }
    resp.revision = save(state, store).await?;
    resp.applied = true;
    Ok(resp)
}

pub async fn export(state: &AppState) -> Result<Value, CatalogError> {
    let store = load(state).await?;
    let default_host = store.host_names.first().cloned().unwrap_or_else(|| host_id(state));
    let mut models: Vec<Value> = store.models.iter().map(|m| cat::export_row(m, Some(&default_host))).collect();
    models.extend(store.unmapped.iter().cloned());
    Ok(serde_json::json!({
        "version": 1,
        "models": models,
        "profiles": store.profiles,
        "settings": {},
    }))
}

fn parse_sets(set: &[(String, String)]) -> Result<Vec<(String, Value)>, CatalogError> {
    set.iter()
        .map(|(k, raw)| {
            let y: serde_yaml::Value = serde_yaml::from_str(raw)
                .map_err(|e| CatalogError::Invalid(format!("{k}: {raw:?} is not YAML: {e}")))?;
            let v = serde_json::to_value(y).map_err(|e| CatalogError::Invalid(format!("{k}: {e}")))?;
            Ok((k.clone(), v))
        })
        .collect()
}

fn changes(a: &CatalogModel, b: &CatalogModel) -> Vec<(String, Option<String>, Option<String>)> {
    let (va, vb) = (serde_json::to_value(a).unwrap_or_default(), serde_json::to_value(b).unwrap_or_default());
    let flat = |v: &Value| -> std::collections::BTreeMap<String, String> {
        let mut out = std::collections::BTreeMap::new();
        if let Value::Object(m) = v {
            for (k, x) in m {
                if k == "extra" {
                    if let Value::Object(e) = x {
                        for (ek, ev) in e {
                            out.insert(ek.clone(), ev.to_string());
                        }
                    }
                } else {
                    out.insert(k.clone(), x.to_string());
                }
            }
        }
        out
    };
    let (fa, fb) = (flat(&va), flat(&vb));
    let mut keys: Vec<&String> = fa.keys().chain(fb.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter(|k| fa.get(*k) != fb.get(*k))
        .map(|k| (k.clone(), fa.get(k).cloned(), fb.get(k).cloned()))
        .collect()
}

/// Validate a record against the declared stores and the catalog.
fn check_record(m: &CatalogModel, decl: &[ModelStore], others: &[CatalogModel]) -> Result<(), CatalogError> {
    if !cat::ENGINES.contains(&m.engine.as_str()) {
        return Err(CatalogError::Invalid(format!("engine {:?} is not one of {}", m.engine, cat::ENGINES.join(", "))));
    }
    if m.name.trim().is_empty() {
        return Err(CatalogError::Invalid("a record needs a name".into()));
    }
    if m.source.is_none() && m.path.is_none() {
        return Err(CatalogError::Invalid("a record needs --source or --path (what the engine loads)".into()));
    }
    if let Some(s) = &m.store {
        if !decl.iter().any(|d| &d.name == s) {
            return Err(CatalogError::Invalid(format!("no declared store {s:?} (`stores` lists them)")));
        }
    }
    let key = m.registry_key();
    if let Some(dup) = others.iter().find(|o| o.id != m.id && o.registry_key() == key) {
        return Err(CatalogError::Refused(format!(
            "{} already catalogs {} {} (record {})",
            dup.name, key.0, key.1, dup.id
        )));
    }
    Ok(())
}

pub async fn add(state: &AppState, req: &ModelAddRequest) -> Result<ModelWriteResponse, CatalogError> {
    let _one = state.adopt_lock.lock().await;
    let mut store = load(state).await?;
    let decl = stores(state).await;
    let mut m = CatalogModel {
        id: uuid::Uuid::new_v4().to_string(),
        name: req.name.clone(),
        engine: req.engine.clone(),
        source: None,
        path: None,
        store: None,
        size_bytes: None,
        enabled: true,
        extra: Default::default(),
        origin: "add".into(),
    };
    let mut sets: Vec<(String, Value)> = Vec::new();
    if let Some(s) = &req.source {
        sets.push(("source".into(), Value::String(s.clone())));
    }
    if let Some(p) = &req.path {
        sets.push(("path".into(), Value::String(p.clone())));
    }
    sets.extend(parse_sets(&req.set)?);
    for (k, v) in sets {
        cat::set_field(&mut m, &k, v).map_err(CatalogError::Invalid)?;
    }
    m.store = match &req.store {
        Some(s) => Some(s.clone()),
        None => cat::store_for(&m.engine, m.path.as_deref(), m.source.as_deref(), &decl),
    };
    check_record(&m, &decl, &store.models)?;
    let before = m.clone();
    store.models.push(m.clone());
    let revision = save(state, store).await?;
    let mut empty = before;
    empty.name = String::new();
    Ok(ModelWriteResponse { changes: changes(&empty, &m), model: m, revision })
}

pub async fn edit(state: &AppState, req: &ModelEditRequest) -> Result<ModelWriteResponse, CatalogError> {
    let _one = state.adopt_lock.lock().await;
    let mut store = load(state).await?;
    let decl = stores(state).await;
    let old = find(&store, &req.model)?.clone();
    let mut m = old.clone();
    for (k, v) in parse_sets(&req.set)? {
        cat::set_field(&mut m, &k, v).map_err(CatalogError::Invalid)?;
    }
    for k in &req.unset {
        if k == "name" {
            return Err(CatalogError::Invalid("a record keeps its name; set another".into()));
        }
        cat::set_field(&mut m, k, Value::Null).map_err(CatalogError::Invalid)?;
    }
    check_record(&m, &decl, &store.models)?;
    let ch = changes(&old, &m);
    if ch.is_empty() {
        return Ok(ModelWriteResponse { model: m, changes: ch, revision: store.revision });
    }
    if let Some(slot) = store.models.iter_mut().find(|x| x.id == old.id) {
        *slot = m.clone();
    }
    let revision = save(state, store).await?;
    Ok(ModelWriteResponse { model: m, changes: ch, revision })
}

pub async fn enable(state: &AppState, req: &ModelEnableRequest) -> Result<ModelWriteResponse, CatalogError> {
    let _one = state.adopt_lock.lock().await;
    let mut store = load(state).await?;
    let old = find(&store, &req.model)?.clone();
    if old.enabled == req.enabled {
        return Ok(ModelWriteResponse { model: old, changes: vec![], revision: store.revision });
    }
    let mut m = old.clone();
    m.enabled = req.enabled;
    if let Some(slot) = store.models.iter_mut().find(|x| x.id == old.id) {
        *slot = m.clone();
    }
    let revision = save(state, store).await?;
    Ok(ModelWriteResponse { changes: changes(&old, &m), model: m, revision })
}

// ---- scan ----------------------------------------------------------------------------

fn dirs(p: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(p)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    out.sort();
    out
}

fn name_of(p: &Path) -> String {
    p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string()
}

/// Weights a store holds, as (engine, source, path).
pub fn scan_store(st: &ModelStore) -> Vec<(String, String, String)> {
    let root = Path::new(&st.path);
    let mut out = Vec::new();
    match st.kind {
        StoreKind::HfCache => {
            for d in dirs(root) {
                let n = name_of(&d);
                if let Some(rest) = n.strip_prefix("models--") {
                    if let Some((org, name)) = rest.split_once("--") {
                        out.push(("vllm".into(), format!("{org}/{name}"), d.display().to_string()));
                    }
                }
            }
        }
        StoreKind::Directory => {
            for d in dirs(root) {
                let n = name_of(&d);
                if n.starts_with('.') {
                    continue;
                }
                if d.is_file() && n.ends_with(".gguf") {
                    out.push(("llama_cpp".into(), d.display().to_string(), d.display().to_string()));
                } else if d.join("config.json").is_file() {
                    out.push(("vllm".into(), d.display().to_string(), d.display().to_string()));
                } else if d.is_dir() && dirs(&d).iter().any(|f| name_of(f).ends_with(".gguf")) {
                    out.push(("llama_cpp".into(), d.display().to_string(), d.display().to_string()));
                }
            }
        }
        StoreKind::Ollama => {
            // manifests/<registry>/<namespace>/<model>/<tag>
            for reg in dirs(&root.join("manifests")) {
                for ns in dirs(&reg) {
                    for model in dirs(&ns) {
                        for tag in dirs(&model) {
                            if !tag.is_file() {
                                continue;
                            }
                            let (r, n, m) = (name_of(&reg), name_of(&ns), name_of(&model));
                            let base = match (r.as_str(), n.as_str()) {
                                ("registry.ollama.ai", "library") => m,
                                ("registry.ollama.ai", _) => format!("{n}/{m}"),
                                _ => format!("{r}/{n}/{m}"),
                            };
                            out.push(("ollama".into(), format!("{base}:{}", name_of(&tag)), tag.display().to_string()));
                        }
                    }
                }
            }
        }
        StoreKind::Lmstudio => {
            for publisher in dirs(root) {
                if !publisher.is_dir() || name_of(&publisher).starts_with('.') {
                    continue;
                }
                for model in dirs(&publisher) {
                    if model.is_dir() {
                        out.push((
                            "lmstudio".into(),
                            format!("{}/{}", name_of(&publisher), name_of(&model)),
                            model.display().to_string(),
                        ));
                    }
                }
            }
        }
    }
    out
}

fn record_for<'a>(models: &'a [CatalogModel], engine: &str, source: &str, path: &str) -> Option<&'a CatalogModel> {
    let base = |s: &str| s.rsplit('/').next().unwrap_or(s).to_string();
    // The path or the source names it, whatever engine the record says
    // (a HF repo serves vLLM or SGLang alike); LM Studio keys are matched
    // on their last part as well (`publisher/model` vs `model`).
    models.iter().find(|m| {
        m.path.as_deref() == Some(path)
            || m.source.as_deref() == Some(source)
            || (engine == "lmstudio" && m.engine == "lmstudio" && m.source.as_deref().map(base) == Some(base(source)))
    })
}

pub async fn scan(state: &AppState, req: &ScanRequest) -> Result<ScanResponse, CatalogError> {
    let _one = state.adopt_lock.lock().await;
    let mut store = load(state).await?;
    let decl = stores(state).await;
    let chosen: Vec<ModelStore> = match &req.store {
        Some(n) => vec![decl
            .iter()
            .find(|s| &s.name == n)
            .cloned()
            .ok_or_else(|| CatalogError::Invalid(format!("no declared store {n:?}")))?],
        None => decl.clone(),
    };
    let mut found = Vec::new();
    for st in &chosen {
        let st2 = st.clone();
        let items = tokio::task::spawn_blocking(move || scan_store(&st2))
            .await
            .map_err(|e| CatalogError::Io(e.to_string()))?;
        for (engine, source, path) in items {
            let record = record_for(&store.models, &engine, &source, &path).map(|m| m.id.clone());
            found.push(ScanFound { store: st.name.clone(), engine, source, path, record });
        }
    }
    // Records in a scanned store whose absolute path is gone.
    let names: Vec<&str> = chosen.iter().map(|s| s.name.as_str()).collect();
    let missing: Vec<CatalogModel> = store
        .models
        .iter()
        .filter(|m| m.store.as_deref().is_some_and(|s| names.contains(&s)))
        .filter(|m| m.path.as_deref().is_some_and(|p| p.starts_with('/') && !Path::new(p).exists()))
        .cloned()
        .collect();
    let mut added = 0;
    if req.add {
        for f in found.iter_mut().filter(|f| f.record.is_none()) {
            let source_is_path = f.source == f.path;
            let m = CatalogModel {
                id: uuid::Uuid::new_v4().to_string(),
                name: f.source.rsplit('/').next().unwrap_or(&f.source).to_string(),
                engine: f.engine.clone(),
                source: (!source_is_path).then(|| f.source.clone()),
                path: Some(f.path.clone()),
                store: Some(f.store.clone()),
                size_bytes: None,
                enabled: true,
                extra: Default::default(),
                origin: "scan".into(),
            };
            if store.models.iter().any(|o| o.registry_key() == m.registry_key()) {
                continue;
            }
            f.record = Some(m.id.clone());
            store.models.push(m);
            added += 1;
        }
    }
    let revision = if added > 0 { save(state, store).await? } else { store.revision };
    Ok(ScanResponse {
        stores: names.iter().map(|s| s.to_string()).collect(),
        found,
        missing,
        added,
        revision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_reads_each_store_kind() {
        let d = tempfile::tempdir().unwrap();
        let r = d.path();
        let mk = |p: &str| std::fs::create_dir_all(r.join(p)).unwrap();
        let touch = |p: &str| {
            std::fs::create_dir_all(r.join(p).parent().unwrap()).unwrap();
            std::fs::write(r.join(p), "x").unwrap();
        };
        mk("hf/models--Qwen--Qwen3-8B/snapshots");
        mk("hf/datasets--x--y");
        touch("dir/Q-AWQ/config.json");
        touch("dir/gg/model-Q4.gguf");
        touch("dir/solo.gguf");
        mk("dir/empty");
        touch("ol/manifests/registry.ollama.ai/library/qwen3/8b");
        touch("ol/manifests/hf.co/bartowski/x/latest");
        mk("lms/qwen/qwen3-4b");
        let st = |name: &str, kind| ModelStore {
            name: name.into(),
            kind,
            path: r.join(name).display().to_string(),
            reserve_gib: 1,
            workload: None,
        };
        let got = |s: ModelStore| scan_store(&s).into_iter().map(|(e, src, _)| format!("{e} {src}")).collect::<Vec<_>>();
        assert_eq!(got(st("hf", StoreKind::HfCache)), vec!["vllm Qwen/Qwen3-8B"]);
        let dir = got(st("dir", StoreKind::Directory));
        assert_eq!(dir.len(), 3, "{dir:?}");
        assert!(dir[0].starts_with("vllm ") && dir[0].ends_with("Q-AWQ"), "{dir:?}");
        assert_eq!(got(st("ol", StoreKind::Ollama)), vec!["ollama hf.co/bartowski/x:latest", "ollama qwen3:8b"]);
        assert_eq!(got(st("lms", StoreKind::Lmstudio)), vec!["lmstudio qwen/qwen3-4b"]);
    }
}
