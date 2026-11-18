//! Presets on the agent (Rust M5c item 1, plan decision 3): the same
//! `presets/*.yaml`, `settings.yaml` (`vllm.defaults` only) and router
//! profile envs the Python llmctl uses, read and written in place with
//! comments kept. No second copy, so reversal needs no export.
//!
//! A preset resolves to a workload (the host's adopted vLLM workload with
//! one env file), a profile env next to that env file, and a served name.
//! `run <preset>` switches to that profile through the ordinary profile
//! swap (M4c), so the planner, the VRAM fit, request safety and rollback
//! all apply. A profile whose values drifted from the preset is refused
//! unless the caller asks to sync it first.
//!
//! Paths (read at call time, so tests can point them elsewhere):
//! `LLMCTL_NEXT_PRESET_DIRS` (colon-separated, lowest precedence first;
//! default `$XDG_CONFIG_HOME/llmctl/presets` then `…/llm-models`, as the
//! Python loader), `LLMCTL_NEXT_SETTINGS` (default
//! `$XDG_CONFIG_HOME/llmctl/settings.yaml`), `LLMCTL_ROUTER_CONFIG`
//! (default `~/AI/llm-router/config/router.yaml`, as llmctl-stack).

use std::path::{Path, PathBuf};

use serde_yaml::Value;

use llmctl_client::catalog::{
    EnvDrift, PresetEditRequest, PresetInvalid, PresetLink, PresetListResponse, PresetNewRequest,
    PresetRunRequest, PresetRunResponse, PresetSettleRequest, PresetSettleResponse,
    PresetShowResponse, PresetSummary, PresetWriteResponse, SettleFile,
};
use llmctl_core::preset_launch::{
    self as pl, EnginePin, LauncherEnv, Preset, VllmDefaults,
};
use llmctl_core::presets::edit_preset_text;
use llmctl_core::{EngineKind, HostInventory, InstallRecipe, Ownership, WorkloadDefinition};

use crate::app::AppState;

/// Kept next to a file a write replaced (never a `.yaml`, so no loader
/// reads it as a preset; never a profile, see `profile::resolve`).
pub const BACKUP_SUFFIX: &str = ".llmctl-previous";

#[derive(Debug, thiserror::Error)]
pub enum PresetsError {
    #[error("{0}")]
    UnknownPreset(String),
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Refused(String),
    #[error("{message}")]
    Drift { message: String, drift: Vec<EnvDrift> },
    #[error("{0}")]
    Io(String),
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

fn xdg_config() -> PathBuf {
    match std::env::var_os("XDG_CONFIG_HOME") {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => home().join(".config"),
    }
}

/// Preset directories, lowest precedence first (a later dir's alias wins).
pub fn preset_dirs() -> Vec<PathBuf> {
    if let Some(v) = std::env::var_os("LLMCTL_NEXT_PRESET_DIRS") {
        let s = v.to_string_lossy().into_owned();
        return s.split(':').filter(|p| !p.is_empty()).map(PathBuf::from).collect();
    }
    vec![xdg_config().join("llmctl").join("presets"), xdg_config().join("llm-models")]
}

pub fn settings_path() -> PathBuf {
    match std::env::var_os("LLMCTL_NEXT_SETTINGS") {
        Some(v) if !v.is_empty() => PathBuf::from(v),
        _ => xdg_config().join("llmctl").join("settings.yaml"),
    }
}

pub fn router_config_path() -> PathBuf {
    match std::env::var_os("LLMCTL_ROUTER_CONFIG") {
        Some(v) if !v.is_empty() => {
            let s = v.to_string_lossy().into_owned();
            match s.strip_prefix("~/") {
                Some(rest) => home().join(rest),
                None => PathBuf::from(s),
            }
        }
        _ => home().join("AI/llm-router/config/router.yaml"),
    }
}

/// `vllm.defaults` from settings.yaml; a missing file is the defaults.
/// Only that subtree is kept: the file holds a token elsewhere, and it
/// never leaves this function.
pub fn vllm_defaults() -> Result<VllmDefaults, PresetsError> {
    let path = settings_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(VllmDefaults::default()),
        Err(e) => return Err(PresetsError::Io(format!("{}: {e}", path.display()))),
    };
    let v: Value = serde_yaml::from_str(&text)
        .map_err(|e| PresetsError::Invalid(format!("{}: not YAML: {e}", path.display())))?;
    VllmDefaults::from_settings(&v).map_err(PresetsError::Invalid)
}

/// The host facts of the env render. `python_root` falls back to the
/// prefix of the workload's pinned interpreter (the pin is the truth).
pub fn launcher_env(def: Option<&WorkloadDefinition>) -> LauncherEnv {
    let cuda_root = std::env::var("LLMCTL_CUDA_ROOT")
        .or_else(|_| std::env::var("GPU_MODELS_CUDA_ROOT"))
        .unwrap_or_else(|_| "/usr/local/cuda".into());
    let raw_hf = std::env::var("HF_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| home().join(".cache/huggingface").display().to_string());
    let expanded = match raw_hf.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None if raw_hf == "~" => home(),
        None => PathBuf::from(&raw_hf),
    };
    // Python resolves the path itself when it is a symlink (not its parents).
    let hf_home = match std::fs::symlink_metadata(&expanded) {
        Ok(m) if m.file_type().is_symlink() => std::fs::canonicalize(&expanded)
            .unwrap_or_else(|_| expanded.clone())
            .display()
            .to_string(),
        _ => expanded.display().to_string(),
    };
    LauncherEnv {
        python_root: def.and_then(pinned_interpreter).and_then(|i| EnginePin::python_root(&i)),
        cuda_root,
        hf_home,
    }
}

/// The interpreter a workload's pin names, else its recipe's.
fn pinned_interpreter(def: &WorkloadDefinition) -> Option<String> {
    crate::pin::pinned_version(def)
        .and_then(|v| v.interpreter.clone())
        .or_else(|| match &def.recipe {
            Some(InstallRecipe::PythonEnv(r)) => r.interpreter.clone(),
            _ => None,
        })
}

/// One loaded preset file.
#[derive(Debug, Clone)]
pub struct Record {
    pub preset: Preset,
    pub value: Value,
    pub text: String,
    pub path: PathBuf,
}

/// Python `load_all_records`: `*.yaml` not starting with `_`, sorted;
/// later directories win; malformed files are reported, not hidden.
pub fn load_all() -> (Vec<Record>, Vec<PresetInvalid>) {
    let mut by_alias: std::collections::BTreeMap<String, Record> = Default::default();
    let mut invalid = Vec::new();
    for dir in preset_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        let mut files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                name.ends_with(".yaml") && !name.starts_with('_')
            })
            .collect();
        files.sort();
        for path in files {
            match load_file(&path) {
                Ok(rec) => {
                    by_alias.insert(rec.preset.alias.clone(), rec);
                }
                Err(error) => invalid.push(PresetInvalid {
                    path: path.display().to_string(),
                    error,
                }),
            }
        }
    }
    (by_alias.into_values().collect(), invalid)
}

pub fn load_file(path: &Path) -> Result<Record, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let value: Value = serde_yaml::from_str(&text).map_err(|e| format!("not YAML: {e}"))?;
    let preset = Preset::from_value(&value)?;
    Ok(Record { preset, value, text, path: path.to_path_buf() })
}

fn find(alias: &str) -> Result<Record, PresetsError> {
    let (records, _) = load_all();
    records
        .into_iter()
        .find(|r| r.preset.alias == alias)
        .ok_or_else(|| {
            let (records, _) = load_all();
            let known: Vec<String> = records.iter().map(|r| r.preset.alias.clone()).collect();
            PresetsError::UnknownPreset(format!(
                "unknown preset {alias:?}; known: {}",
                if known.is_empty() { "(none)".into() } else { known.join(", ") }
            ))
        })
}

/// The adopted vLLM workload(s) a preset can run on: engine vLLM, managed,
/// exactly one EnvironmentFile.
fn vllm_workloads(inv: &HostInventory) -> Vec<&WorkloadDefinition> {
    inv.workloads
        .iter()
        .filter(|w| w.engine_kind == EngineKind::Vllm)
        .filter(|w| matches!(&w.ownership, Ownership::Managed { definition, .. } if definition.environment_files.len() == 1))
        .collect()
}

fn pick_workload<'a>(
    inv: &'a HostInventory,
    requested: Option<&str>,
) -> Result<&'a WorkloadDefinition, PresetsError> {
    let candidates = vllm_workloads(inv);
    if let Some(name) = requested {
        return candidates
            .into_iter()
            .find(|w| w.name.as_str() == name)
            .ok_or_else(|| {
                PresetsError::Refused(format!(
                    "{name} is not an adopted vLLM workload with one env file; presets run on those only"
                ))
            });
    }
    match candidates.as_slice() {
        [one] => Ok(one),
        [] => Err(PresetsError::Refused(
            "no adopted vLLM workload with one env file on this host: presets have nowhere to run".into(),
        )),
        many => Err(PresetsError::Refused(format!(
            "several vLLM workloads ({}); name one with --workload",
            many.iter().map(|w| w.name.as_str()).collect::<Vec<_>>().join(", ")
        ))),
    }
}

fn env_path(def: &WorkloadDefinition) -> PathBuf {
    match &def.ownership {
        Ownership::Managed { definition, .. } => PathBuf::from(&definition.environment_files[0].path),
        Ownership::External => PathBuf::new(),
    }
}

/// router.yaml backends: `(served_model, profile file)` where a backend
/// names a `profile_env` under `switch.services_dir`.
fn router_profiles() -> Vec<(String, PathBuf)> {
    let path = router_config_path();
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    let Ok(raw) = serde_yaml::from_str::<Value>(&text) else { return Vec::new() };
    let services = raw
        .get("switch")
        .and_then(|s| s.get("services_dir"))
        .and_then(|v| v.as_str())
        .map(|s| match s.strip_prefix("~/") {
            Some(rest) => home().join(rest),
            None => PathBuf::from(s),
        })
        .unwrap_or_else(|| path.parent().map(Path::to_path_buf).unwrap_or_default());
    let Some(Value::Sequence(backends)) = raw.get("backends") else { return Vec::new() };
    backends
        .iter()
        .filter_map(|b| {
            let served = b.get("served_model")?.as_str()?;
            let profile = b.get("profile_env")?.as_str()?;
            b.get("name")?;
            Some((served.to_string(), services.join(profile)))
        })
        .collect()
}

/// Is `name` a profile of the env file `env_name` (`<env>.<profile>`,
/// not one of llmctl's working files)?
fn is_profile_name(env_name: &str, name: &str) -> bool {
    name.strip_prefix(env_name)
        .and_then(|r| r.strip_prefix('.'))
        .is_some_and(|suffix| {
            !suffix.is_empty()
                && !suffix.starts_with("llmctl-tmp-")
                && !name.ends_with(BACKUP_SUFFIX)
                && !name.ends_with(crate::profile::BACKUP_SUFFIX)
        })
}

/// Where a preset's render lives for `def`: router.yaml's backend for its
/// served name, else a profile whose header names it, else `<env>.<alias>`.
fn locate_profile(def: &WorkloadDefinition, p: &Preset) -> (PathBuf, &'static str, bool) {
    let env = env_path(def);
    let dir = env.parent().map(Path::to_path_buf).unwrap_or_default();
    let env_name = env.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
    let in_dir = |f: &Path| f.parent() == Some(dir.as_path());
    let named = |f: &Path| f.file_name().and_then(|n| n.to_str()).is_some_and(|n| is_profile_name(&env_name, n));
    let router: Vec<PathBuf> = router_profiles()
        .into_iter()
        .filter(|(served, f)| served == &p.served_name && in_dir(f) && named(f))
        .map(|(_, f)| f)
        .collect();
    if let [one] = router.as_slice() {
        return (one.clone(), "router", one.exists());
    }
    let mut by_header: Vec<PathBuf> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|f| named(f))
        .filter(|f| {
            std::fs::read_to_string(f)
                .ok()
                .is_some_and(|t| pl::header_preset(&t) == Some(p.alias.as_str()))
        })
        .collect();
    by_header.sort();
    if let [one] = by_header.as_slice() {
        return (one.clone(), "header", true);
    }
    let fresh = dir.join(format!("{env_name}.{}", p.alias));
    let exists = fresh.exists();
    (fresh, "new", exists)
}

fn drift_of(current: &str, rendered: &str) -> Vec<EnvDrift> {
    pl::diff_env(current, rendered)
        .into_iter()
        .map(|(key, file, render)| EnvDrift { key, file, render })
        .collect()
}

/// The render of `p` for `def` (port = the workload's port).
pub fn render_for(p: &Preset, def: &WorkloadDefinition) -> Result<String, PresetsError> {
    let defaults = vllm_defaults()?;
    let spec = pl::launch_spec(p, &defaults, Some(i64::from(def.port)));
    pl::render_vllm_env(&spec, &launcher_env(Some(def))).map_err(PresetsError::Refused)
}

struct Linked {
    link: PresetLink,
    rendered: String,
    profile_path: PathBuf,
    profile_text: String,
}

fn link(p: &Preset, def: &WorkloadDefinition) -> Result<Linked, PresetsError> {
    let rendered = render_for(p, def)?;
    let (profile_path, found_by, exists) = locate_profile(def, p);
    let profile_text = if exists {
        std::fs::read_to_string(&profile_path).unwrap_or_default()
    } else {
        String::new()
    };
    let drift = if exists {
        drift_of(&profile_text, &rendered)
    } else {
        drift_of("", &rendered)
    };
    let active_text = std::fs::read_to_string(env_path(def)).unwrap_or_default();
    let active = def.model == p.served_name && pl::parse_env(&active_text) == pl::parse_env(&rendered);
    Ok(Linked {
        link: PresetLink {
            workload: def.name.to_string(),
            profile_env: exists.then(|| profile_path.display().to_string()),
            found_by: found_by.into(),
            drift,
            active,
        },
        rendered,
        profile_path,
        profile_text,
    })
}

fn summary(rec: &Record, inv: &HostInventory) -> (PresetSummary, Option<Linked>) {
    let p = &rec.preset;
    let (link, link_error, linked) = match pick_workload(inv, None).and_then(|def| link(p, def)) {
        Ok(l) => (Some(l.link.clone()), None, Some(l)),
        Err(e) => (None, Some(e.to_string()), None),
    };
    (
        PresetSummary {
            alias: p.alias.clone(),
            served_name: p.served_name.clone(),
            model_id: p.model_id.clone(),
            quantization: p.quantization.clone(),
            tensor_parallel_size: p.tensor_parallel_size,
            max_model_len: p.max_model_len,
            family: p.family.clone(),
            param_count_b: p.param_count_b,
            source_path: rec.path.display().to_string(),
            link,
            link_error,
        },
        linked,
    )
}

pub async fn list(state: &AppState) -> PresetListResponse {
    let inv = state.ops.inventory().await;
    let (records, invalid) = load_all();
    PresetListResponse {
        dirs: preset_dirs().iter().map(|d| d.display().to_string()).collect(),
        presets: records.iter().map(|r| summary(r, &inv).0).collect(),
        invalid,
    }
}

pub async fn show(state: &AppState, alias: &str) -> Result<PresetShowResponse, PresetsError> {
    let inv = state.ops.inventory().await;
    let rec = find(alias)?;
    let (summary, linked) = summary(&rec, &inv);
    let (rendered_env, render_error) = match (&linked, &summary.link_error) {
        (Some(l), _) => (Some(l.rendered.clone()), None),
        (None, Some(e)) => (None, Some(e.clone())),
        (None, None) => (None, None),
    };
    Ok(PresetShowResponse {
        summary,
        fields: serde_json::to_value(&rec.value).unwrap_or(serde_json::Value::Null),
        text: rec.text,
        rendered_env,
        render_error,
    })
}

pub fn digest(bytes: &[u8]) -> String {
    crate::profile::digest(bytes)
}

/// Write `bytes` to `path` through a private (0600, `create_new`) temp file
/// in the same directory, set `mode`, fsync, rename. A reader never sees a
/// half-written file, and the bytes are never readable beyond `mode`, not
/// even for a moment (env profiles may hold credentials).
fn atomic_write(path: &Path, bytes: &[u8], mode: u32) -> Result<(), PresetsError> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let io = |what: &str, e: std::io::Error| PresetsError::Io(format!("{what} {}: {e}", path.display()));
    let tmp = PathBuf::from(format!("{}.llmctl-tmp-{}", path.display(), std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let result = (|| {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
            .map_err(|e| io("creating a temp file for", e))?;
        f.write_all(bytes).map_err(|e| io("writing", e))?;
        f.set_permissions(std::fs::Permissions::from_mode(mode))
            .map_err(|e| io("setting the mode of", e))?;
        f.sync_all().map_err(|e| io("syncing", e))?;
        std::fs::rename(&tmp, path).map_err(|e| io("replacing", e))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Write through [`atomic_write`] in the target's own directory (a
/// symlinked preset stays a symlink: the target is written). The previous
/// bytes are kept at `<target>.llmctl-previous`, written the same way and
/// with the previous file's mode.
fn write_keeping(path: &Path, bytes: &[u8], new_mode: u32) -> Result<(PathBuf, Option<PathBuf>), PresetsError> {
    use std::os::unix::fs::PermissionsExt;
    let target = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let (mode, backup) = match std::fs::read(&target) {
        Ok(old) => {
            let mode = std::fs::metadata(&target)
                .map(|m| m.permissions().mode() & 0o777)
                .map_err(|e| PresetsError::Io(format!("reading the mode of {}: {e}", target.display())))?;
            let backup = PathBuf::from(format!("{}{BACKUP_SUFFIX}", target.display()));
            atomic_write(&backup, &old, mode)?;
            (mode, Some(backup))
        }
        Err(_) => (new_mode, None),
    };
    atomic_write(&target, bytes, mode)?;
    Ok((target, backup))
}

fn yaml_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_yaml::to_string(other).unwrap_or_default().trim_end().to_string(),
    }
}

/// Apply `set`/`unset` to a mapping; values are YAML text.
fn apply_edits(
    base: &serde_yaml::Mapping,
    set: &[(String, String)],
    unset: &[String],
) -> Result<serde_yaml::Mapping, PresetsError> {
    let mut m = base.clone();
    for (k, raw) in set {
        if !pl::PRESET_FIELDS.contains(&k.as_str()) {
            return Err(PresetsError::Invalid(format!("unknown preset field {k:?}")));
        }
        let v: Value = serde_yaml::from_str(raw)
            .map_err(|e| PresetsError::Invalid(format!("{k}: {raw:?} is not YAML: {e}")))?;
        m.insert(Value::String(k.clone()), v);
    }
    for k in unset {
        if m.remove(Value::String(k.clone())).is_none() {
            return Err(PresetsError::Invalid(format!("{k} is not set in this preset")));
        }
    }
    Ok(m)
}

fn changes_between(a: &serde_yaml::Mapping, b: &serde_yaml::Mapping) -> Vec<(String, Option<String>, Option<String>)> {
    let mut keys: Vec<String> = Vec::new();
    for (k, _) in a.iter().chain(b.iter()) {
        if let Some(k) = k.as_str() {
            if !keys.iter().any(|x| x == k) {
                keys.push(k.to_string());
            }
        }
    }
    keys.into_iter()
        .filter_map(|k| {
            let key = Value::String(k.clone());
            let (x, y) = (a.get(&key), b.get(&key));
            (x != y).then(|| (k, x.map(yaml_text), y.map(yaml_text)))
        })
        .collect()
}

/// The line edit, or a full dump when it cannot keep the comments.
fn edited_text(text: &str, desired: &serde_yaml::Mapping) -> Result<(String, bool), PresetsError> {
    let desired_v = Value::Mapping(desired.clone());
    match edit_preset_text(text, &desired_v) {
        Ok(t) => Ok((t, true)),
        Err(_) => serde_yaml::to_string(&desired_v)
            .map(|t| (t, false))
            .map_err(|e| PresetsError::Io(e.to_string())),
    }
}

fn write_response(
    alias: &str,
    path: &Path,
    before: &str,
    exists: bool,
    old: &serde_yaml::Mapping,
    desired: &serde_yaml::Mapping,
    dry_run: bool,
) -> Result<PresetWriteResponse, PresetsError> {
    let (text, comments_kept) = edited_text(before, desired)?;
    // Same guard as Python `_write_yaml`: the text must reload to a valid
    // preset equal to what was asked for.
    let reloaded: Value = serde_yaml::from_str(&text).map_err(|e| PresetsError::Io(e.to_string()))?;
    let preset = Preset::from_value(&reloaded)
        .map_err(|e| PresetsError::Invalid(format!("{}: not saved, the preset would be invalid: {e}", path.display())))?;
    if reloaded != Value::Mapping(desired.clone()) {
        return Err(PresetsError::Invalid(format!(
            "{}: not saved, the text does not reload to the requested preset",
            path.display()
        )));
    }
    if preset.alias != alias {
        return Err(PresetsError::Invalid(format!(
            "the alias is the file's identity: {alias} would become {}; use `preset new --from`",
            preset.alias
        )));
    }
    let mut out = PresetWriteResponse {
        alias: alias.to_string(),
        path: path.display().to_string(),
        before_digest: if exists { digest(before.as_bytes()) } else { "missing".into() },
        after_digest: digest(text.as_bytes()),
        changes: changes_between(old, desired),
        text,
        comments_kept,
        written: false,
        backup: None,
    };
    if !dry_run && out.before_digest != out.after_digest {
        let (_, backup) = write_keeping(path, out.text.as_bytes(), 0o664)?;
        out.written = true;
        out.backup = backup.map(|b| b.display().to_string());
    }
    Ok(out)
}

pub async fn edit(state: &AppState, req: &PresetEditRequest) -> Result<PresetWriteResponse, PresetsError> {
    let _one = state.adopt_lock.lock().await;
    let rec = find(&req.alias)?;
    if let Some(expect) = &req.expect_digest {
        let now = digest(rec.text.as_bytes());
        if &now != expect {
            return Err(PresetsError::Refused(format!(
                "{} changed since the preview ({expect} → {now}); preview again",
                rec.path.display()
            )));
        }
    }
    let Value::Mapping(old) = &rec.value else {
        return Err(PresetsError::Invalid("preset is not a mapping".into()));
    };
    let desired = apply_edits(old, &req.set, &req.unset)?;
    write_response(&req.alias, &rec.path, &rec.text, true, old, &desired, req.dry_run)
}

pub async fn new(state: &AppState, req: &PresetNewRequest) -> Result<PresetWriteResponse, PresetsError> {
    let _one = state.adopt_lock.lock().await;
    if !pl::valid_alias(&req.alias) {
        return Err(PresetsError::Invalid(format!(
            "alias must match '^[a-z0-9][a-z0-9_.-]*$'; got {:?}",
            req.alias
        )));
    }
    let (records, _) = load_all();
    if let Some(r) = records.iter().find(|r| r.preset.alias == req.alias) {
        return Err(PresetsError::Refused(format!(
            "preset {} already exists at {}; use `preset edit`",
            req.alias,
            r.path.display()
        )));
    }
    let dirs = preset_dirs();
    let dir = dirs.first().cloned().ok_or_else(|| PresetsError::Refused("no preset directory configured".into()))?;
    let path = dir.join(format!("{}.yaml", req.alias));
    if path.exists() || path.is_symlink() {
        return Err(PresetsError::Refused(format!(
            "{} exists but does not load as a preset; fix or move it first",
            path.display()
        )));
    }
    let (base_text, base_map) = match &req.from {
        Some(src) => {
            let rec = find(src)?;
            let Value::Mapping(m) = rec.value else {
                return Err(PresetsError::Invalid("source preset is not a mapping".into()));
            };
            (rec.text, m)
        }
        None => (String::new(), serde_yaml::Mapping::new()),
    };
    let mut set = vec![("alias".to_string(), req.alias.clone())];
    set.extend(req.set.iter().cloned());
    let desired = apply_edits(&base_map, &set, &[])?;
    if !req.dry_run {
        std::fs::create_dir_all(&dir).map_err(|e| PresetsError::Io(format!("{}: {e}", dir.display())))?;
    }
    let mut out = write_response(&req.alias, &path, &base_text, false, &serde_yaml::Mapping::new(), &desired, req.dry_run)?;
    out.before_digest = "missing".into();
    Ok(out)
}

/// Resolve (and with `sync`, first bring in line) the profile a preset runs
/// from. The run itself is the caller's ordinary `run --profile-env`.
pub async fn run(state: &AppState, req: &PresetRunRequest) -> Result<PresetRunResponse, PresetsError> {
    let _one = state.adopt_lock.lock().await;
    let inv = state.ops.inventory().await;
    let rec = find(&req.alias)?;
    let def = pick_workload(&inv, req.workload.as_deref())?;
    let linked = link(&rec.preset, def)?;
    let mut out = PresetRunResponse {
        alias: req.alias.clone(),
        workload: def.name.to_string(),
        model: rec.preset.served_name.clone(),
        profile_env: Some(linked.profile_path.display().to_string()),
        synced: false,
        drift: linked.link.drift.clone(),
    };
    if !out.drift.is_empty() {
        if !req.sync {
            let keys: Vec<&str> = out.drift.iter().map(|d| d.key.as_str()).collect();
            return Err(PresetsError::Drift {
                message: format!(
                    "{} does not render preset {} ({}); `--sync` writes the preset's values into it (comments kept)",
                    linked.profile_path.display(),
                    req.alias,
                    keys.join(", ")
                ),
                drift: out.drift,
            });
        }
        out.synced = true;
        if !req.dry_run {
            let text = pl::synced_profile(&linked.profile_text, &linked.rendered, &req.alias);
            // A new profile gets the active env file's mode (env files may
            // hold credentials).
            let mode = {
                use std::os::unix::fs::PermissionsExt;
                std::fs::metadata(env_path(def)).map(|m| m.permissions().mode() & 0o777).unwrap_or(0o600)
            };
            write_keeping(&linked.profile_path, text.as_bytes(), mode)?;
        }
    } else if linked.link.active {
        // Same values, same served name: no switch, no restart for a
        // comment-only difference.
        out.profile_env = None;
    }
    Ok(out)
}

/// What the router comment, the env header and the profiles say about the
/// engine's version, rewritten to the pin.
pub async fn settle(state: &AppState, req: &PresetSettleRequest) -> Result<PresetSettleResponse, PresetsError> {
    let _one = state.adopt_lock.lock().await;
    let inv = state.ops.inventory().await;
    let def = pick_workload(&inv, req.workload.as_deref())?;
    let version = crate::pin::pinned_version(def)
        .ok_or_else(|| PresetsError::Refused(format!("{} has no version pin; `version {} --repin` first", def.name, def.name)))?;
    let interpreter = version
        .interpreter
        .clone()
        .ok_or_else(|| PresetsError::Refused(format!("{}'s pin names no interpreter", def.name)))?;
    let pin = EnginePin::from_pin(&interpreter, &version.packages, "vllm")
        .ok_or_else(|| PresetsError::Refused(format!("cannot read an env name from {interpreter}")))?;

    // Which preset each profile comes from (for the "keep in sync" line).
    let (records, _) = load_all();
    let env = env_path(def);
    let dir = env.parent().map(Path::to_path_buf).unwrap_or_default();
    let env_name = env.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
    let mut files: Vec<(PathBuf, Option<String>)> = Vec::new();
    let mut profiles: Vec<PathBuf> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|f| f.file_name().and_then(|n| n.to_str()).is_some_and(|n| is_profile_name(&env_name, n)))
        .filter(|f| !f.file_name().and_then(|n| n.to_str()).unwrap_or("").contains(".bak"))
        .collect();
    profiles.sort();
    let tilde = |p: &Path| -> String {
        let s = p.display().to_string();
        let h = home().display().to_string();
        match s.strip_prefix(&format!("{h}/")) {
            Some(rest) => format!("~/{rest}"),
            None => s,
        }
    };
    for f in profiles {
        let text = std::fs::read_to_string(&f).unwrap_or_default();
        let source = pl::header_preset(&text)
            .and_then(|a| records.iter().find(|r| r.preset.alias == a))
            .map(|r| tilde(&r.path));
        files.push((f, source));
    }
    files.push((router_config_path(), None));

    let mut out = PresetSettleResponse {
        workload: def.name.to_string(),
        pin: pin.clone(),
        files: Vec::new(),
        untouched: Vec::new(),
    };
    for (path, source) in files {
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let (new, changes) = pl::settle_version_comments(&text, &pin, source.as_deref());
        if changes.is_empty() {
            out.untouched.push(path.display().to_string());
            continue;
        }
        let mut written = false;
        if req.write {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).map(|m| m.permissions().mode() & 0o777).unwrap_or(0o600);
            write_keeping(&path, new.as_bytes(), mode)?;
            written = true;
        }
        out.files.push(SettleFile { path: path.display().to_string(), changes, written });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_names_skip_working_files() {
        assert!(is_profile_name("vllm-engine.env", "vllm-engine.env.qwen38"));
        assert!(!is_profile_name("vllm-engine.env", "vllm-engine.env"));
        assert!(!is_profile_name("vllm-engine.env", "vllm-engine.envx"));
        assert!(!is_profile_name("vllm-engine.env", "vllm-engine.env.llmctl-previous"));
        assert!(!is_profile_name("vllm-engine.env", "vllm-engine.env.qwen38.llmctl-previous"));
        assert!(!is_profile_name("vllm-engine.env", "vllm-engine.env.llmctl-tmp-12"));
        assert!(!is_profile_name("vllm-engine.env", "vllm-engine2.env.x"));
    }

    #[test]
    fn writes_keep_the_previous_bytes_with_the_previous_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("e.env.p");
        std::fs::write(&f, "SECRET=1\n").unwrap();
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600)).unwrap();
        let (target, backup) = write_keeping(&f, b"SECRET=2\n", 0o644).unwrap();
        let backup = backup.unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "SECRET=2\n");
        assert_eq!(std::fs::read_to_string(&backup).unwrap(), "SECRET=1\n");
        assert_eq!((mode(&target), mode(&backup)), (0o600, 0o600));
        // No temp file left; a new file gets the mode asked for.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
        let (fresh, none) = write_keeping(&dir.path().join("new"), b"x", 0o640).unwrap();
        assert!(none.is_none());
        assert_eq!(mode(&fresh), 0o640);
    }

    #[test]
    fn edits_are_typed_by_yaml_and_checked() {
        let base: Value = serde_yaml::from_str("alias: x\nmax_num_seqs: 4\nextra_args: --a\n").unwrap();
        let Value::Mapping(m) = base else { panic!() };
        let out = apply_edits(
            &m,
            &[("max_num_seqs".into(), "8".into()), ("dtype".into(), "null".into())],
            &["extra_args".into()],
        )
        .unwrap();
        assert_eq!(out.get("max_num_seqs"), Some(&Value::from(8)));
        assert_eq!(out.get("dtype"), Some(&Value::Null));
        assert!(out.get("extra_args").is_none());
        assert!(apply_edits(&m, &[("bogus".into(), "1".into())], &[]).is_err());
        assert!(apply_edits(&m, &[], &["dtype".into()]).is_err());
        let ch = changes_between(&m, &out);
        assert_eq!(ch.len(), 3, "{ch:?}");
    }
}
