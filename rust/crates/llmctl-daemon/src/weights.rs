//! Declaring model stores, and removing weights from them.
//!
//! Removing weights is the one thing here that deletes data, so it is a
//! two-step verb: a preview that says what would go, how big it is and
//! who still references it, then the same call with the preview's token.
//! Weights a workload, a profile or a lane still names are never removed
//! (design decision 5: "removing weights refuses while a lane or preset
//! references them").

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use llmctl_client::wire::{ModelRemoveRequest, ModelRemoveResponse, StoreAddRequest};
use llmctl_core::{ModelStore, Ownership, ProcessState, Source, StoreKind, WorkloadName};

use crate::app::AppState;
use crate::ops::OpsError;

fn invalid(msg: impl Into<String>) -> OpsError {
    OpsError::Invalid(msg.into())
}

// ---- stores ------------------------------------------------------------------------

/// `stores add`: declare a store. Nothing is created on disk.
pub async fn add_store(state: &AppState, req: &StoreAddRequest) -> Result<serde_json::Value, OpsError> {
    let kind: StoreKind = serde_json::from_value(serde_json::Value::String(req.kind.clone()))
        .map_err(|_| invalid(format!("store kind {:?}: use hf_cache, directory, ollama or lmstudio", req.kind)))?;
    let workload = match &req.workload {
        Some(w) => Some(WorkloadName::new(w).map_err(|e| invalid(e.to_string()))?),
        None => None,
    };
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    if inv.stores.iter().any(|s| s.name == req.name) {
        return Err(OpsError::InvalidState(format!("a store named {} is already declared", req.name)));
    }
    let mut next = (*inv).clone();
    next.stores.push(ModelStore {
        name: req.name.clone(),
        kind,
        path: req.path.clone(),
        reserve_gib: req.reserve_gib.unwrap_or(20),
        workload,
    });
    let revision = crate::pin::commit_inventory(state, next, "store").await?;
    tracing::info!(store = %req.name, path = %req.path, revision, "store declared");
    Ok(serde_json::json!({ "store": req.name, "added": true, "inventory_revision": revision }))
}

/// `stores remove`: forget a store. The weights in it are not touched.
pub async fn remove_store(state: &AppState, name: &str) -> Result<serde_json::Value, OpsError> {
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    if !inv.stores.iter().any(|s| s.name == name) {
        return Err(invalid(format!("no store named {name}")));
    }
    let mut next = (*inv).clone();
    next.stores.retain(|s| s.name != name);
    let revision = crate::pin::commit_inventory(state, next, "store").await?;
    tracing::info!(store = %name, revision, "store declaration removed");
    Ok(serde_json::json!({ "store": name, "removed": true, "inventory_revision": revision }))
}

// ---- what a target names ---------------------------------------------------------

/// What `model rm` was asked to remove.
enum Target {
    /// A directory or file inside the store.
    Path { path: PathBuf, names: Vec<String> },
    /// A tag in an Ollama server's store.
    OllamaTag(String),
}

/// `hf:org/name`, `ollama:tag`, or `path:<relative path inside the store>`
/// (LM Studio names its folders after the repository, not the catalog
/// entry, so only a path says which one).
fn resolve(target: &str, store: &ModelStore) -> Result<Target, OpsError> {
    let root = Path::new(&store.path);
    if let Some(rel) = target.strip_prefix("path:") {
        let rel_path = Path::new(rel);
        let plain = !rel.is_empty()
            && rel_path.is_relative()
            && rel_path.components().all(|c| matches!(c, std::path::Component::Normal(_)));
        if !plain {
            return Err(invalid(format!(
                "{rel:?} is not a plain relative path inside store {}",
                store.name
            )));
        }
        if store.kind == StoreKind::Ollama {
            return Err(invalid("an Ollama store is emptied by tag: use ollama:<tag>"));
        }
        return Ok(Target::Path { path: root.join(rel_path), names: Vec::new() });
    }
    match Source::parse(target).map_err(invalid)? {
        Source::Hf { repo, .. } => {
            let (org, name) = repo.split_once('/').unwrap_or(("", repo.as_str()));
            let cache_dir = format!("models--{org}--{name}");
            let path = match store.kind {
                StoreKind::HfCache => root.join(&cache_dir),
                StoreKind::Directory => root.join(name),
                _ => {
                    return Err(invalid(format!(
                        "hf:{repo} does not name anything in the {} store {}",
                        kind_name(store.kind),
                        store.name
                    )))
                }
            };
            Ok(Target::Path { path, names: vec![repo.clone(), cache_dir] })
        }
        Source::Ollama { tag } => {
            if store.kind != StoreKind::Ollama {
                return Err(invalid(format!("ollama:{tag} needs an ollama store, not {}", store.name)));
            }
            Ok(Target::OllamaTag(tag))
        }
        Source::Lms { model, .. } => Err(invalid(format!(
            "LM Studio does not keep {model} under that name; name its folder: path:<publisher>/<folder>"
        ))),
    }
}

fn kind_name(kind: StoreKind) -> &'static str {
    match kind {
        StoreKind::HfCache => "hf_cache",
        StoreKind::Directory => "directory",
        StoreKind::Ollama => "ollama",
        StoreKind::Lmstudio => "lmstudio",
    }
}

// ---- who still references it ------------------------------------------------------

/// True when `text` names `needle` as a whole name: the characters on
/// either side are not ones a model name or path component is made of.
fn names(text: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let part = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-');
    let mut from = 0;
    while let Some(i) = text[from..].find(needle) {
        let start = from + i;
        let end = start + needle.len();
        let before_ok = text[..start].chars().next_back().is_none_or(|c| !part(c));
        let after_ok = text[end..].chars().next().is_none_or(|c| !part(c));
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

/// Profiles next to an env file: `vllm-engine.env.qwen38` beside
/// `vllm-engine.env`. Backups a tool left behind are not presets.
async fn sibling_profiles(env_file: &Path) -> Vec<PathBuf> {
    let (Some(dir), Some(base)) = (env_file.parent(), env_file.file_name().and_then(|n| n.to_str())) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
        return out;
    };
    while let Ok(Some(e)) = entries.next_entry().await {
        let name = e.file_name().to_string_lossy().into_owned();
        let backup = name.contains(".bak") || name.ends_with(".previous") || name.ends_with(".llmctl-previous");
        if name.starts_with(&format!("{base}.")) && !backup {
            out.push(e.path());
        }
    }
    out.sort();
    out
}

/// Everything that still names the target: workload records (model,
/// launch command, environment, env files and the profiles beside them)
/// and llama-swap lanes. File contents are matched, never returned.
async fn references(state: &AppState, needles: &[String]) -> Vec<String> {
    let hit = |text: &str| needles.iter().any(|n| names(text, n));
    let mut found: Vec<String> = Vec::new();
    let inv = state.ops.inventory().await;
    for w in &inv.workloads {
        if hit(&w.model) {
            found.push(format!("workload {} serves it (model {})", w.name, w.model));
        }
        let Ownership::Managed { definition, .. } = &w.ownership else {
            continue;
        };
        let argv = definition
            .exec_start_pre
            .iter()
            .chain(definition.exec_start.iter())
            .chain(definition.exec_start_post.iter())
            .flat_map(|c| c.argv.iter());
        if argv.clone().any(|a| hit(a)) {
            found.push(format!("workload {}: its launch command names it", w.name));
        }
        if definition.environment.iter().any(|(_, v)| hit(v)) {
            found.push(format!("workload {}: its environment names it", w.name));
        }
        for env in &definition.environment_files {
            let path = Path::new(&env.path);
            let mut files = vec![path.to_path_buf()];
            files.extend(sibling_profiles(path).await);
            for f in files {
                if let Ok(text) = tokio::fs::read_to_string(&f).await {
                    if hit(&text) {
                        found.push(format!("workload {}: profile {} names it", w.name, f.display()));
                    }
                }
            }
        }
    }
    let lanes_dir = state.paths.data_dir.join("lanes");
    if let Ok(mut entries) = tokio::fs::read_dir(&lanes_dir).await {
        while let Ok(Some(e)) = entries.next_entry().await {
            let Ok(bytes) = tokio::fs::read(e.path()).await else { continue };
            let Ok(store) = serde_json::from_slice::<crate::lanes::LaneStore>(&bytes) else { continue };
            for lane in &store.records.lanes {
                let cmd = store.records.expanded_cmd(lane).unwrap_or_default();
                if hit(&cmd) {
                    found.push(format!("lane {} of {} runs it", lane.id, store.workload));
                }
            }
        }
    }
    found.sort();
    found.dedup();
    found
}

// ---- removal ---------------------------------------------------------------------

async fn dir_bytes(path: &Path) -> Option<u64> {
    let out = tokio::process::Command::new("du").arg("-sk").arg(path).output().await.ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let kib: u64 = text.split_whitespace().next()?.parse().ok()?;
    Some(kib * 1024)
}

fn token(store: &str, what: &str, bytes: Option<u64>) -> String {
    let digest = Sha256::digest(format!("rm|{store}|{what}|{}", bytes.unwrap_or(0)).as_bytes());
    format!("{:x}", digest)[..16].to_string()
}

/// `model rm`: preview, or (with the preview's token) remove.
pub async fn remove(state: &AppState, req: &ModelRemoveRequest) -> Result<ModelRemoveResponse, OpsError> {
    // No download may be filling the store while something is taken out.
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    let store = inv
        .stores
        .iter()
        .find(|s| s.name == req.store)
        .cloned()
        .ok_or_else(|| invalid(format!("no store named {} (see `stores`)", req.store)))?;
    let mut resp = ModelRemoveResponse {
        target: req.target.clone(),
        store: store.name.clone(),
        path: None,
        bytes: None,
        references: Vec::new(),
        confirm: None,
        removed: false,
        freed_bytes: None,
    };
    match resolve(&req.target, &store)? {
        Target::Path { path, mut names } => {
            // The target must really lie inside the store, whatever links
            // are on the way.
            let root = tokio::fs::canonicalize(&store.path)
                .await
                .map_err(|e| invalid(format!("store {} ({}): {e}", store.name, store.path)))?;
            let meta = tokio::fs::symlink_metadata(&path)
                .await
                .map_err(|_| invalid(format!("{} is not in store {} ({})", req.target, store.name, path.display())))?;
            let parent = tokio::fs::canonicalize(path.parent().unwrap_or(&root))
                .await
                .map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            if !parent.starts_with(&root) {
                return Err(invalid(format!("{} is outside store {}", path.display(), store.name)));
            }
            let real = parent.join(path.file_name().unwrap_or_default());
            names.push(real.display().to_string());
            names.push(path.display().to_string());
            resp.path = Some(real.display().to_string());
            resp.bytes = if meta.is_dir() { dir_bytes(&real).await } else { Some(meta.len()) };
            resp.references = references(state, &names).await;
            let expect = token(&store.name, &real.display().to_string(), resp.bytes);
            if !resp.references.is_empty() {
                return Ok(resp);
            }
            let Some(given) = &req.confirm else {
                resp.confirm = Some(expect);
                return Ok(resp);
            };
            if given != &expect {
                return Err(OpsError::InvalidState(format!(
                    "stale confirmation: the preview token is now {expect} (the size changed; look again)"
                )));
            }
            let before = crate::acquire::free_bytes(&root).await;
            if meta.is_dir() {
                tokio::fs::remove_dir_all(&real).await
            } else {
                tokio::fs::remove_file(&real).await
            }
            .map_err(|e| OpsError::Engine(format!("removing {}: {e}", real.display())))?;
            let after = crate::acquire::free_bytes(&root).await;
            resp.removed = true;
            resp.freed_bytes = match (before, after) {
                (Some(b), Some(a)) => Some(a.saturating_sub(b)),
                _ => None,
            };
            tracing::info!(store = %store.name, path = %real.display(), bytes = ?resp.bytes, "weights removed");
        }
        Target::OllamaTag(tag) => {
            let Some(service) = store.workload.as_ref() else {
                return Err(invalid(format!("store {} names no Ollama workload", store.name)));
            };
            let Some(def) = inv.workload(service) else {
                return Err(invalid(format!("store {} names {service}, which is not enrolled", store.name)));
            };
            let running = state
                .ops
                .engine()
                .observe(&inv)
                .await
                .map_err(|e| OpsError::Engine(e.to_string()))?
                .iter()
                .any(|o| &o.workload == service && o.process == ProcessState::Running);
            if !running {
                return Err(OpsError::InvalidState(format!(
                    "{service} is not running, and Ollama removes a model through its server; \
                     start it first (llmctl-next run {service})"
                )));
            }
            resp.references = references(state, std::slice::from_ref(&tag)).await;
            let expect = token(&store.name, &tag, None);
            if !resp.references.is_empty() {
                return Ok(resp);
            }
            let Some(given) = &req.confirm else {
                resp.confirm = Some(expect);
                return Ok(resp);
            };
            if given != &expect {
                return Err(OpsError::InvalidState(format!("stale confirmation: the preview token is {expect}")));
            }
            let ollama = std::env::var(crate::acquire::OLLAMA_BIN_ENV)
                .ok()
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| crate::acquire::locate("ollama"));
            let out = tokio::process::Command::new(&ollama)
                .args(["rm", &tag])
                .env("OLLAMA_HOST", format!("127.0.0.1:{}", def.port))
                .stdin(std::process::Stdio::null())
                .output()
                .await
                .map_err(|e| OpsError::Engine(format!("running {ollama} rm: {e}")))?;
            if !out.status.success() {
                return Err(OpsError::Engine(format!(
                    "ollama rm {tag}: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                )));
            }
            resp.removed = true;
            tracing::info!(store = %store.name, %tag, "ollama model removed");
        }
    }
    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(kind: StoreKind) -> ModelStore {
        ModelStore {
            name: "s".into(),
            kind,
            path: "/data/store".into(),
            reserve_gib: 20,
            workload: None,
        }
    }

    #[test]
    fn a_target_names_one_place_inside_its_store() {
        let Ok(Target::Path { path, names }) = resolve("hf:Qwen/Qwen3.8-27B", &store(StoreKind::HfCache)) else {
            panic!("an hf target in an hf cache");
        };
        assert_eq!(path, Path::new("/data/store/models--Qwen--Qwen3.8-27B"));
        assert!(names.contains(&"Qwen/Qwen3.8-27B".to_string()));
        let Ok(Target::Path { path, .. }) = resolve("hf:Qwen/Qwen3.8-27B", &store(StoreKind::Directory)) else {
            panic!("an hf target in a directory store");
        };
        assert_eq!(path, Path::new("/data/store/Qwen3.8-27B"));
        let Ok(Target::Path { path, .. }) = resolve("path:lmstudio-community/gemma-GGUF", &store(StoreKind::Lmstudio))
        else {
            panic!("a path target");
        };
        assert_eq!(path, Path::new("/data/store/lmstudio-community/gemma-GGUF"));

        // Never out of the store, never the store itself, never by guess.
        for bad in ["path:../other", "path:/etc", "path:", "path:a/../../b", "lms:google/gemma-3-270m"] {
            assert!(resolve(bad, &store(StoreKind::Lmstudio)).is_err(), "{bad}");
        }
        assert!(resolve("hf:Qwen/Qwen3.8-27B", &store(StoreKind::Ollama)).is_err());
        assert!(resolve("ollama:qwen3:8b", &store(StoreKind::HfCache)).is_err());
        assert!(resolve("path:blobs", &store(StoreKind::Ollama)).is_err());
    }

    #[test]
    fn a_name_matches_whole_not_as_part_of_a_longer_one() {
        assert!(names("VLLM_MODEL=Qwen/Qwen3.8-27B\n", "Qwen/Qwen3.8-27B"));
        assert!(names("--model /hub/models--Qwen--Qwen3.8-27B/snapshots/abc", "models--Qwen--Qwen3.8-27B"));
        assert!(names("exec srv --model /m/Qwen3.8-27B --port 1", "/m/Qwen3.8-27B"));
        // A longer name that merely starts the same way is another model.
        assert!(!names("VLLM_MODEL=Qwen/Qwen3.8-27B-AWQ\n", "Qwen/Qwen3.8-27B"));
        assert!(!names("--model /m/Qwen3.8-27B-uncensored", "/m/Qwen3.8-27B"));
        assert!(!names("anything", ""));
    }

    #[test]
    fn the_token_binds_store_target_and_size() {
        let t = token("s", "/data/store/a", Some(10));
        assert_eq!(t.len(), 16);
        assert_ne!(t, token("s", "/data/store/a", Some(11)));
        assert_ne!(t, token("s", "/data/store/b", Some(10)));
        assert_ne!(t, token("t", "/data/store/a", Some(10)));
    }
}
