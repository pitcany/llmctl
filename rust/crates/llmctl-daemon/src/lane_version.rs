//! Version facts per llama-swap lane.
//!
//! A lane has no unit of its own: llama-swap starts it from a shell line.
//! So a lane's install is what that line runs: every executable file the
//! command names (the engine binary or console script, the helper scripts
//! around it) and, when the engine lives in a Python env, that env's
//! packages. `lanes version` compares those facts with the pin; `--pin`
//! records them. A macro change written through `lanes macro` re-pins the
//! lanes it moved, because that write is the version swap.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::time::Duration;

use sha2::{Digest, Sha256};
use tokio::process::Command;

use llmctl_client::wire::{LaneVersion, LaneVersionRequest, LaneVersionResponse};
use llmctl_core::swap::{Lane, SwapConfig};
use llmctl_core::{EngineKind, VersionRef};

use crate::app::AppState;
use crate::lanes::{refuse, LaneStore, LanesError};

const PYTHON_TIMEOUT: Duration = Duration::from_secs(60);
/// A file this large is weights, not a program; it is named, not hashed.
const MAX_HASHED_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// What the env reports: the named packages' versions, and one digest
/// over every installed distribution plus the content of the engine
/// package's own files. The content matters: mtplx on m5 carries a local
/// patch that a reinstall of the same version silently reverts.
const ENV_FACTS_PY: &str = r#"
import hashlib, importlib.metadata as m, json, sys
names = sys.argv[1:]
packages = {}
for n in names:
    try:
        packages[n] = m.version(n)
    except m.PackageNotFoundError:
        pass
lines = sorted("%s==%s" % (d.metadata["Name"], d.version) for d in m.distributions() if d.metadata["Name"])
h = hashlib.sha256("\n".join(lines).encode())
if names:
    try:
        files = sorted(m.files(names[0]) or [], key=str)
    except m.PackageNotFoundError:
        files = []
    for f in files:
        if str(f).endswith((".pyc", ".pyo")):
            continue
        try:
            data = open(f.locate(), "rb").read()
        except OSError:
            continue
        h.update(str(f).encode())
        h.update(hashlib.sha256(data).digest())
print(json.dumps({"packages": packages, "lock": "sha256:" + h.hexdigest()}))
"#;

/// The packages that name an engine's version, most specific first (the
/// first one's files are content-hashed).
fn engine_packages(kind: &EngineKind) -> &'static [&'static str] {
    match kind {
        EngineKind::Mtplx => &["mtplx", "mlx", "mlx-lm"],
        EngineKind::MlxVlm => &["mlx-vlm", "mlx", "mlx-lm"],
        EngineKind::Vllm => &["vllm", "torch"],
        _ => &[],
    }
}

fn is_system_path(path: &str) -> bool {
    ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt", "/System", "/dev", "/proc"]
        .iter()
        .any(|p| path == *p || path.starts_with(&format!("{p}/")))
}

/// Absolute paths a shell line names, in order, without repeats.
pub fn paths_in(cmd: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in cmd.split(|c: char| {
        c.is_whitespace() || matches!(c, ';' | '\'' | '"' | '&' | '|' | '(' | ')' | '<' | '>' | '=' | '`')
    }) {
        if word.starts_with('/') && word.len() > 1 && !out.iter().any(|p| p == word) {
            out.push(word.to_string());
        }
    }
    out
}

/// What an env reports: its named packages' versions and its lock digest.
type EnvFacts = (Vec<(String, String)>, String);

/// One probe run: files and envs shared by several lanes are read once.
#[derive(Default)]
pub struct ProbeCache {
    files: HashMap<String, Option<String>>,
    /// Keyed by interpreter and the package names asked for.
    envs: HashMap<(String, Vec<String>), EnvFacts>,
}

/// `Some(digest)` when `path` is a program: a regular file with an
/// execute bit, outside the system prefixes.
async fn program_digest(path: &str, cache: &mut ProbeCache) -> Option<String> {
    if let Some(hit) = cache.files.get(path) {
        return hit.clone();
    }
    let found = async {
        if is_system_path(path) {
            return None;
        }
        let meta = tokio::fs::metadata(path).await.ok()?;
        if !meta.is_file() {
            return None;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o111 == 0 {
                return None;
            }
        }
        if meta.len() > MAX_HASHED_BYTES {
            return Some(format!("not-hashed:{}-bytes", meta.len()));
        }
        match tokio::fs::read(path).await {
            Ok(bytes) => Some(format!("sha256:{:x}", Sha256::digest(bytes))),
            Err(_) => Some("unreadable".to_string()),
        }
    }
    .await;
    cache.files.insert(path.to_string(), found.clone());
    found
}

/// The interpreter of the env a console script lives in: `<env>/bin/python`
/// next to it.
fn env_interpreter(program: &str) -> Option<String> {
    let dir = Path::new(program).parent()?;
    if dir.file_name()?.to_str()? != "bin" {
        return None;
    }
    let python = dir.join("python");
    python.exists().then(|| python.display().to_string())
}

async fn env_facts(
    interpreter: &str,
    names: &[&str],
    cache: &mut ProbeCache,
) -> EnvFacts {
    let key = (
        interpreter.to_string(),
        names.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
    );
    if let Some(hit) = cache.envs.get(&key) {
        return hit.clone();
    }
    let run = async {
        let mut cmd = Command::new(interpreter);
        cmd.arg("-I")
            .arg("-c")
            .arg(ENV_FACTS_PY)
            .args(names)
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true);
        let out = tokio::time::timeout(PYTHON_TIMEOUT, cmd.output()).await.ok()?.ok()?;
        if !out.status.success() {
            return None;
        }
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
        let mut packages: Vec<(String, String)> = v
            .get("packages")?
            .as_object()?
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
            .collect();
        packages.sort();
        Some((packages, v.get("lock")?.as_str()?.to_string()))
    };
    let facts = run.await.unwrap_or_else(|| (Vec::new(), "unreadable".to_string()));
    cache.envs.insert(key, facts.clone());
    facts
}

/// What one lane's command runs, right now.
pub async fn probe_lane(cfg: &SwapConfig, lane: &Lane, cache: &mut ProbeCache) -> VersionRef {
    let cmd = cfg.expanded_cmd(lane).unwrap_or_default();
    let mut exec_digests: Vec<(String, String)> = Vec::new();
    for path in paths_in(&cmd) {
        if let Some(digest) = program_digest(&path, cache).await {
            exec_digests.push((path, digest));
        }
    }
    let interpreter = exec_digests.iter().find_map(|(p, _)| env_interpreter(p));
    let (packages, lock_digest) = match &interpreter {
        Some(python) => env_facts(python, engine_packages(&lane.engine_kind), cache).await,
        None => (Vec::new(), String::new()),
    };
    VersionRef {
        interpreter,
        packages,
        cuda: None,
        lock_digest,
        exec_digests,
        fact: None,
    }
}

/// A one-line name for a pin: the engine's own package (`mtplx 2.11.0`),
/// else the engine program and the start of its digest.
pub fn label(pin: &VersionRef, kind: &EngineKind) -> String {
    let engine_package = engine_packages(kind)
        .first()
        .and_then(|main| pin.packages.iter().find(|(name, _)| name == main))
        .or_else(|| pin.packages.first());
    if let Some((name, version)) = engine_package {
        return format!("{name} {version}");
    }
    match pin.exec_digests.last() {
        Some((path, digest)) => {
            let base = path.rsplit('/').next().unwrap_or(path);
            let short = digest.strip_prefix("sha256:").unwrap_or(digest);
            format!("{base} {}", &short[..short.len().min(12)])
        }
        None => "nothing to pin".to_string(),
    }
}

/// `lanes version`: pin, live facts and drift per lane; with `pin`, the
/// live facts become the pin first.
pub async fn versions(
    state: &AppState,
    req: LaneVersionRequest,
) -> Result<LaneVersionResponse, LanesError> {
    let path = crate::lanes::store_path(state, &req.workload);
    let mut store = crate::lanes::read_store(&path).await?.ok_or_else(|| {
        refuse(format!(
            "{} has no lane records; run `lanes import {}`",
            req.workload, req.workload
        ))
    })?;
    if let Some(id) = &req.lane {
        if store.records.lane(id).is_none() {
            return Err(refuse(format!("{} has no lane {id}", req.workload)));
        }
    }
    let selected = |id: &str| req.lane.as_deref().is_none_or(|want| want == id);

    let mut cache = ProbeCache::default();
    let mut live: BTreeMap<String, VersionRef> = BTreeMap::new();
    for lane in store.records.lanes.iter().filter(|l| selected(&l.id)) {
        live.insert(lane.id.clone(), probe_lane(&store.records, lane, &mut cache).await);
    }

    let mut pinned_now = Vec::new();
    if req.pin {
        let before = store.versions.clone();
        // In file order, like every other lane listing.
        let ids: Vec<String> = store.records.lanes.iter().map(|l| l.id.clone()).collect();
        for id in ids {
            let Some(facts) = live.get(&id) else { continue };
            if store.versions.get(&id) != Some(facts) {
                pinned_now.push(id.clone());
            }
            store.versions.insert(id, facts.clone());
        }
        // A pin for a lane that no longer exists says nothing.
        let known: BTreeSet<&str> = store.records.lanes.iter().map(|l| l.id.as_str()).collect();
        store.versions.retain(|id, _| known.contains(id.as_str()));
        if store.versions != before {
            crate::lanes::save_store(&path, &store).await.map_err(LanesError::Io)?;
            tracing::info!(workload = %req.workload, lanes = ?pinned_now, "lane version pins recorded");
        }
    }

    let lanes = store
        .records
        .lanes
        .iter()
        .filter(|l| selected(&l.id))
        .map(|l| {
            let live = live.get(&l.id).cloned();
            let pinned = store.versions.get(&l.id).cloned();
            let drift = match (&pinned, &live) {
                (Some(p), Some(now)) => crate::version::version_drift(p, now),
                _ => Vec::new(),
            };
            LaneVersion {
                id: l.id.clone(),
                engine_kind: l.engine_kind.to_string(),
                label: live.as_ref().map(|v| label(v, &l.engine_kind)),
                pinned,
                live,
                drift,
            }
        })
        .collect();
    Ok(LaneVersionResponse {
        workload: store.workload.clone(),
        lanes,
        pinned_now,
    })
}

/// After a confirmed config write: re-pin the pinned lanes whose command
/// changed (a macro path change is the version swap, and the write is the
/// owner accepting it), and drop the pins of lanes that are gone.
pub async fn follow_write(store: &mut LaneStore, previous: &SwapConfig) {
    if store.versions.is_empty() {
        return;
    }
    let known: BTreeSet<String> = store.records.lanes.iter().map(|l| l.id.clone()).collect();
    store.versions.retain(|id, _| known.contains(id));
    let mut cache = ProbeCache::default();
    let records = store.records.clone();
    for lane in &records.lanes {
        if !store.versions.contains_key(&lane.id) {
            continue;
        }
        let before = previous.lane(&lane.id).and_then(|l| previous.expanded_cmd(l));
        if before != records.expanded_cmd(lane) {
            let facts = probe_lane(&records, lane, &mut cache).await;
            store.versions.insert(lane.id.clone(), facts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shell_line_names_its_programs_once_each() {
        let cmd = "bash -lc '/a/free.sh ; /a/wait.sh ${PORT} 30 ; exec /w/.venv/bin/mtplx serve \
                   --model /m/Qwen --kv-dir=/w/kv --host 127.0.0.1 ; /a/free.sh'";
        assert_eq!(
            paths_in(cmd),
            vec!["/a/free.sh", "/a/wait.sh", "/w/.venv/bin/mtplx", "/m/Qwen", "/w/kv"]
        );
    }

    #[test]
    fn a_pin_is_named_after_the_engine_package_not_the_first_one() {
        let pin = VersionRef {
            interpreter: Some("/w/.venv/bin/python".into()),
            // Sorted by name, as the probe stores them.
            packages: vec![
                ("mlx".into(), "0.32.2".into()),
                ("mlx-lm".into(), "0.31.3".into()),
                ("mtplx".into(), "2.12.0".into()),
            ],
            cuda: None,
            lock_digest: "sha256:00".into(),
            exec_digests: vec![("/w/.venv/bin/mtplx".into(), "sha256:abcdef0123456789".into())],
            fact: None,
        };
        assert_eq!(label(&pin, &EngineKind::Mtplx), "mtplx 2.12.0");
        // An engine with no known package: the first one listed.
        assert_eq!(label(&pin, &EngineKind::Ds4), "mlx 0.32.2");
        let binary = VersionRef { packages: vec![], ..pin };
        assert_eq!(label(&binary, &EngineKind::MlxServe), "mtplx abcdef012345");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_lane_is_pinned_by_the_programs_its_command_runs() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let exe = |name: &str, body: &[u8]| {
            let p = dir.path().join(name);
            std::fs::write(&p, body).unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            p.display().to_string()
        };
        let engine = exe("engine-1.0", b"one");
        let helper = exe("free.sh", b"#!/bin/sh\n");
        // Weights: named by the command, never a program.
        let weights = dir.path().join("model.gguf");
        std::fs::write(&weights, b"weights").unwrap();

        let yaml = format!(
            "macros:\n  eng: {engine}\nmodels:\n  fast:\n    cmd: bash -lc '{helper} ; exec ${{eng}} --model {} --port ${{PORT}}'\n",
            weights.display()
        );
        let cfg = llmctl_core::swap::import(&yaml).unwrap();
        let lane = cfg.lane("fast").unwrap();
        let mut cache = ProbeCache::default();
        let pinned = probe_lane(&cfg, lane, &mut cache).await;
        let paths: Vec<&str> = pinned.exec_digests.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, vec![helper.as_str(), engine.as_str()]);
        let named = label(&pinned, &lane.engine_kind);
        assert!(named.starts_with("engine-1.0 "), "{named}");

        // The engine is replaced in place: same path, new content.
        std::fs::write(&engine, b"two").unwrap();
        let live = probe_lane(&cfg, lane, &mut ProbeCache::default()).await;
        let diffs = crate::version::version_drift(&pinned, &live);
        assert_eq!(diffs.len(), 1, "{diffs:?}");
        assert_eq!(diffs[0].field, "version.exec_digests");
    }
}
