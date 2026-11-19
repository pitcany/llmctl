//! The model catalog (Rust M5c item 3, plan decision 4), the pure part:
//! records, the mapping from Python registry rows, and the reverse export.
//!
//! The Python registry (`~/.local/share/llmctl/llmctl.sqlite3`, table
//! `models`) is never written by Rust. Its rows come in as a registry
//! bundle (`llmctl export-registry`, `RegistryExport` v1) and become agent
//! records per host; every field the agent does not model is kept verbatim
//! in `extra`, so `catalog export` writes a bundle Python's
//! `import-registry` takes back with the same rows. Weights stay in the M4
//! stores: a record points at them (`store`, `path`/`source`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::acquire::{ModelStore, StoreKind};

/// Python `RuntimeName` values a record may carry. `openai` is not here:
/// an OpenAI-compatible endpoint is a route, not weights on a host.
pub const ENGINES: &[&str] = &[
    "vllm",
    "llama_cpp",
    "lmstudio",
    "ollama",
    "python_script",
    "sglang",
    "freetoken",
    "strata",
];

/// Registry fields kept verbatim in `extra` (everything a bundle row has
/// that is not modelled on the record).
pub const EXTRA_FIELDS: &[&str] = &[
    "host",
    "format",
    "quantization",
    "estimated_vram_gb",
    "max_context",
    "parameter_count",
    "notes",
    "default_profile_id",
    "tags",
    "metadata",
    "status",
    "created_at",
    "updated_at",
];

/// Fields an owner may set on a record (`model add/edit --set`).
pub const EDITABLE: &[&str] = &[
    "name",
    "source",
    "path",
    "store",
    "size_bytes",
    "format",
    "quantization",
    "estimated_vram_gb",
    "max_context",
    "parameter_count",
    "notes",
    "default_profile_id",
    "tags",
    "metadata",
];

/// One model on this host.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogModel {
    /// Stable id: the registry row's id when imported, else a new uuid.
    pub id: String,
    pub name: String,
    /// The engine kind (Python `runtime` spelling).
    pub engine: String,
    /// What the engine calls it: an Ollama tag, an LM Studio key, a HF
    /// repo, or a path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// The declared store its weights are in; `None` when no store holds
    /// them (listed by `catalog import` so a store can be declared).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    pub enabled: bool,
    /// Registry fields carried verbatim for the reverse export.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra: BTreeMap<String, Value>,
    /// `import`, `add` or `scan`.
    pub origin: String,
}

impl CatalogModel {
    /// The key Python's `import-registry` dedupes on: (runtime, source or name).
    pub fn registry_key(&self) -> (String, String) {
        (
            self.engine.clone(),
            self.source.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| self.name.clone()),
        )
    }
}

/// What happened to one bundle row on import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RowOutcome {
    /// Becomes a record, its store found.
    New,
    /// Becomes a record, but no declared store holds its weights.
    NoStore,
    /// Already a record (same id, or same registry key).
    Present,
    /// Not imported; `reason` says why. Kept verbatim for the export.
    Unmapped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RowReport {
    pub id: Option<String>,
    pub name: String,
    pub runtime: String,
    pub host: Option<String>,
    pub outcome: RowOutcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

fn s(row: &Value, k: &str) -> Option<String> {
    row.get(k).and_then(|v| v.as_str()).map(str::to_string)
}

/// A HF repo id (`org/name`), not a path.
pub fn looks_like_repo(text: &str) -> bool {
    !text.starts_with('/')
        && !text.starts_with('.')
        && !text.starts_with('~')
        && text.split('/').count() == 2
        && text.split('/').all(|p| !p.is_empty())
}

/// The store a record's weights are in, by engine and path (pure: the
/// caller checks existence when it cares).
pub fn store_for(engine: &str, path: Option<&str>, source: Option<&str>, stores: &[ModelStore]) -> Option<String> {
    let by_kind = |k: StoreKind| stores.iter().find(|s| s.kind == k).map(|s| s.name.clone());
    match engine {
        "ollama" => return by_kind(StoreKind::Ollama),
        "lmstudio" => return by_kind(StoreKind::Lmstudio),
        _ => {}
    }
    if let Some(p) = path.filter(|p| p.starts_with('/')) {
        let best = stores
            .iter()
            .filter(|s| {
                let root = s.path.trim_end_matches('/');
                p == root || p.starts_with(&format!("{root}/"))
            })
            .max_by_key(|s| s.path.len());
        if let Some(st) = best {
            return Some(st.name.clone());
        }
    }
    // A HF repo id lives in the HF cache, for the engines that load one.
    let loads_hf = matches!(engine, "vllm" | "sglang");
    let repo = path
        .filter(|p| looks_like_repo(p))
        .or_else(|| source.filter(|s| looks_like_repo(s)));
    if loads_hf && repo.is_some() {
        return by_kind(StoreKind::HfCache);
    }
    None
}

/// Map one bundle row for the agent whose host answers to `host_names`.
/// `Err(reason)` = unmapped.
pub fn map_row(row: &Value, host_names: &[String], stores: &[ModelStore]) -> Result<CatalogModel, String> {
    let name = s(row, "name").ok_or("row has no name")?;
    let runtime = s(row, "runtime").ok_or("row has no runtime")?;
    if runtime == "openai" {
        return Err("an OpenAI-compatible endpoint is a route, not weights on a host (llm-router/LiteLLM keep it)".into());
    }
    if !ENGINES.contains(&runtime.as_str()) {
        return Err(format!("runtime {runtime:?} has no engine kind here"));
    }
    if let Some(host) = s(row, "host") {
        if !host_names.iter().any(|h| h == &host) {
            return Err(format!(
                "host {host:?} is another host's (this agent answers to {}; import it there, or pass --as {host})",
                host_names.join(", ")
            ));
        }
    }
    let path = s(row, "path").filter(|p| !p.is_empty());
    let source = s(row, "source").filter(|p| !p.is_empty());
    let store = store_for(&runtime, path.as_deref(), source.as_deref(), stores);
    let mut extra = BTreeMap::new();
    for k in EXTRA_FIELDS {
        if let Some(v) = row.get(*k) {
            extra.insert((*k).to_string(), v.clone());
        }
    }
    Ok(CatalogModel {
        id: s(row, "id").unwrap_or_default(),
        name,
        engine: runtime,
        source,
        path,
        store,
        size_bytes: row.get("size_bytes").and_then(|v| v.as_u64()),
        enabled: row.get("active").and_then(|v| v.as_bool()).unwrap_or(true),
        extra,
        origin: "import".into(),
    })
}

/// A record as a registry bundle row (Python `schemas.Model`).
pub fn export_row(m: &CatalogModel, default_host: Option<&str>) -> Value {
    let get = |k: &str| m.extra.get(k).cloned().unwrap_or(Value::Null);
    let host = match m.extra.get("host") {
        Some(v) => v.clone(),
        None => default_host.map(|h| Value::String(h.into())).unwrap_or(Value::Null),
    };
    let or = |k: &str, dflt: Value| m.extra.get(k).cloned().unwrap_or(dflt);
    serde_json::json!({
        "id": if m.id.is_empty() { Value::Null } else { Value::String(m.id.clone()) },
        "name": m.name,
        "runtime": m.engine,
        "source": m.source,
        "path": m.path,
        "host": host,
        "format": get("format"),
        "quantization": get("quantization"),
        "size_bytes": m.size_bytes,
        "estimated_vram_gb": get("estimated_vram_gb"),
        "max_context": get("max_context"),
        "parameter_count": get("parameter_count"),
        "notes": get("notes"),
        "default_profile_id": get("default_profile_id"),
        "active": m.enabled,
        "tags": or("tags", Value::Array(vec![])),
        "metadata": or("metadata", Value::Object(Default::default())),
        "status": or("status", Value::String("registered".into())),
        "created_at": get("created_at"),
        "updated_at": get("updated_at"),
    })
}

/// Check a value for a settable field (`model add/edit --set`).
pub fn check_field(key: &str, v: &Value) -> Result<(), String> {
    let ok = match key {
        "name" => v.as_str().is_some_and(|s| !s.trim().is_empty()),
        "source" | "path" | "store" | "format" | "quantization" | "notes" | "default_profile_id" => {
            v.is_null() || v.is_string()
        }
        "size_bytes" | "max_context" | "parameter_count" => v.is_null() || v.as_u64().is_some(),
        "estimated_vram_gb" => v.is_null() || v.as_f64().is_some(),
        "tags" => v.as_array().is_some_and(|a| a.iter().all(|t| t.is_string())),
        "metadata" => v.is_object(),
        other => return Err(format!("{other} is not a settable catalog field ({})", EDITABLE.join(", "))),
    };
    if ok {
        Ok(())
    } else {
        Err(format!("{key}: {v} is not a valid value"))
    }
}

/// Apply one field to a record.
pub fn set_field(m: &mut CatalogModel, key: &str, v: Value) -> Result<(), String> {
    check_field(key, &v)?;
    let text = |v: &Value| v.as_str().map(str::to_string);
    match key {
        "name" => m.name = text(&v).unwrap_or_default(),
        "source" => m.source = text(&v),
        "path" => {
            if let Some(p) = text(&v) {
                if !p.starts_with('/') && !looks_like_repo(&p) {
                    return Err(format!("path {p:?} must be absolute (or a HF repo id)"));
                }
            }
            m.path = text(&v)
        }
        "store" => m.store = text(&v),
        "size_bytes" => m.size_bytes = v.as_u64(),
        _ => {
            if v.is_null() {
                m.extra.remove(key);
            } else {
                m.extra.insert(key.to_string(), v);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stores() -> Vec<ModelStore> {
        let st = |name: &str, kind, path: &str| ModelStore {
            name: name.into(),
            kind,
            path: path.into(),
            reserve_gib: 20,
            workload: None,
        };
        vec![
            st("hf-cache", StoreKind::HfCache, "/h/cache/hub"),
            st("hf2local", StoreKind::Directory, "/h/models/hf2local"),
            st("models", StoreKind::Directory, "/h/models"),
            st("ollama", StoreKind::Ollama, "/h/.ollama/models"),
        ]
    }

    #[test]
    fn rows_map_by_host_engine_and_store() {
        let hosts = vec!["workstation".to_string(), "yannik-desktop".to_string()];
        let row = |rt: &str, host: Value, path: Value, source: Value| {
            json!({"id": "i1", "name": "n", "runtime": rt, "host": host, "path": path, "source": source,
                   "active": false, "tags": ["a"], "metadata": {"k": 1}, "status": "missing", "notes": ""})
        };
        let m = map_row(&row("vllm", json!("yannik-desktop"), json!("/h/models/hf2local/Q"), json!("q")), &hosts, &stores()).unwrap();
        assert_eq!(m.store.as_deref(), Some("hf2local"), "the longest store root wins");
        assert!(!m.enabled);
        assert_eq!(m.extra["status"], json!("missing"));
        let m = map_row(&row("vllm", Value::Null, json!("org/Name"), json!("x")), &hosts, &stores()).unwrap();
        assert_eq!(m.store.as_deref(), Some("hf-cache"), "a repo id lives in the HF cache");
        let m = map_row(&row("ollama", json!("workstation"), Value::Null, json!("qwen3:8b")), &hosts, &stores()).unwrap();
        assert_eq!(m.store.as_deref(), Some("ollama"));
        let m = map_row(&row("llama_cpp", Value::Null, json!("/elsewhere/x.gguf"), Value::Null), &hosts, &stores()).unwrap();
        assert_eq!(m.store, None);
        let m = map_row(&row("lmstudio", Value::Null, Value::Null, json!("k")), &hosts, &stores()).unwrap();
        assert_eq!(m.store, None, "no lmstudio store declared");

        let err = map_row(&row("lmstudio", json!("macbookpro.lan"), Value::Null, json!("k")), &hosts, &stores()).unwrap_err();
        assert!(err.contains("another host"), "{err}");
        let err = map_row(&row("openai", Value::Null, Value::Null, json!("k")), &hosts, &stores()).unwrap_err();
        assert!(err.contains("route"), "{err}");
        let err = map_row(&row("tensorrt", Value::Null, Value::Null, json!("k")), &hosts, &stores()).unwrap_err();
        assert!(err.contains("no engine kind"), "{err}");
    }

    #[test]
    fn export_restores_every_registry_field() {
        let hosts = vec!["yannik-desktop".to_string()];
        let row = json!({"id": "i1", "name": "n", "runtime": "vllm", "source": "q", "path": "/h/models/x",
            "host": "yannik-desktop", "format": null, "quantization": "fp8", "size_bytes": 12,
            "estimated_vram_gb": 30.5, "max_context": 131072, "parameter_count": null, "notes": "hi",
            "default_profile_id": null, "active": true, "tags": ["t"], "metadata": {"port": 8003},
            "status": "registered", "created_at": "2026-08-14T18:28:32.054436", "updated_at": null});
        let m = map_row(&row, &hosts, &stores()).unwrap();
        assert_eq!(export_row(&m, None), row);
        // A record made here gets the import host name and Python defaults.
        let mut fresh = m.clone();
        fresh.extra.clear();
        let out = export_row(&fresh, Some("yannik-desktop"));
        assert_eq!(out["host"], "yannik-desktop");
        assert_eq!(out["status"], "registered");
        assert_eq!(out["tags"], json!([]));
    }

    #[test]
    fn fields_are_checked() {
        let mut m = map_row(&json!({"name": "n", "runtime": "vllm"}), &[], &[]).unwrap();
        set_field(&mut m, "max_context", json!(4096)).unwrap();
        set_field(&mut m, "tags", json!(["a", "b"])).unwrap();
        assert_eq!(m.extra["max_context"], json!(4096));
        assert!(set_field(&mut m, "max_context", json!("big")).is_err());
        assert!(set_field(&mut m, "path", json!("relative/dir/x")).is_err());
        assert!(set_field(&mut m, "bogus", json!(1)).is_err());
        set_field(&mut m, "max_context", Value::Null).unwrap();
        assert!(!m.extra.contains_key("max_context"));
        assert_eq!(m.registry_key(), ("vllm".into(), "n".into()));
    }
}
