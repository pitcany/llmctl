//! Version facts and offline validation for staged installs (plan D,
//! generalized by M4a to every recipe kind).
//!
//! The drift guard pins *what the install reports*: package versions,
//! CUDA and a `pip freeze` digest for python-env recipes; the versioned
//! directory, archive checksum and the binary's own version output for
//! release-archive and source-build recipes; the running server's
//! reported version for vendor apps. Plus, for every kind, content
//! digests of non-system `Exec*` files (the launcher script), so an
//! in-place change is drift even though the unit file never changed.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::process::Command;

use llmctl_core::{
    InstallRecipe, LaunchDefinition, PythonEnvRecipe, VendorProbe, VersionFact, VersionRef,
    WorkloadDefinition,
};

use crate::engine::EngineError;

const PROBE_TIMEOUT: Duration = Duration::from_secs(60);
/// `pip freeze` on a big env is slow; give it room.
const FREEZE_TIMEOUT: Duration = Duration::from_secs(180);

/// The executable a workload's recipe resolves: the interpreter for
/// python-env recipes, the engine binary inside the versioned directory
/// for release-archive and source-build recipes, nothing for vendor apps
/// and external recipes.
pub fn primary_executable_for(recipe: &InstallRecipe, def: &WorkloadDefinition) -> Option<String> {
    match recipe {
        InstallRecipe::PythonEnv(r) => python_interpreter_for(r, def),
        InstallRecipe::ReleaseArchive(r) => exec_under_root(&r.root, def),
        InstallRecipe::SourceBuild(r) => exec_under_root(&r.root, def),
        InstallRecipe::VendorApp(_) | InstallRecipe::External(_) => None,
    }
}

/// The interpreter this workload runs, per its python-env recipe (explicit
/// override, else the first `Exec*` path that is a python).
fn python_interpreter_for(recipe: &PythonEnvRecipe, def: &WorkloadDefinition) -> Option<String> {
    if let Some(explicit) = &recipe.interpreter {
        return Some(explicit.clone());
    }
    let pinned = match &def.ownership {
        llmctl_core::Ownership::Managed { definition, .. } => Some(&**definition),
        _ => None,
    };
    let execs: Vec<&llmctl_core::ExecCommand> = pinned
        .map(|d| {
            d.exec_start
                .iter()
                .chain(d.exec_start_pre.iter())
                .chain(d.exec_start_post.iter())
                .collect()
        })
        .unwrap_or_default();
    execs
        .iter()
        .map(|c| c.path.as_str())
        .chain(
            def.unit
                .name
                .is_empty()
                .then_some("")
                .into_iter()
                .map(|_| ""),
        )
        .find(|p| {
            let base = p.rsplit('/').next().unwrap_or(p);
            base == "python" || base.starts_with("python3") || base.starts_with("python.")
        })
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
}

/// The path of the managed definition that lives under `root` — the
/// engine binary a versioned-directory recipe runs. `Exec*` paths first;
/// then inline `Environment=` values, because the llama.cpp lanes exec a
/// guard wrapper while the versioned binary rides an env var
/// (`LLAMA_GUARD_REAL_BIN`).
fn exec_under_root(root: &str, def: &WorkloadDefinition) -> Option<String> {
    let root = Path::new(root);
    let llmctl_core::Ownership::Managed { definition, .. } = &def.ownership else {
        return None;
    };
    definition
        .exec_start_pre
        .iter()
        .chain(definition.exec_start.iter())
        .chain(definition.exec_start_post.iter())
        .map(|c| c.path.as_str())
        .chain(definition.environment.iter().map(|(_, v)| v.as_str()))
        .find(|p| Path::new(p).starts_with(root))
        .map(str::to_owned)
}

/// Probe the install's version facts, per recipe kind.
pub async fn probe(
    recipe: &InstallRecipe,
    def: &WorkloadDefinition,
) -> Result<VersionRef, EngineError> {
    let mut exec_digests = exec_digests_of(def).await;
    // Declared extra files (an engine binary a launcher execs, a binary
    // under a system prefix) are pinned with the same rule.
    for path in &def.pin_files {
        if !exec_digests.iter().any(|(p, _)| p == path) {
            exec_digests.push((path.clone(), file_digest(path).await));
        }
    }
    let managed = matches!(def.ownership, llmctl_core::Ownership::Managed { .. });
    match recipe {
        InstallRecipe::PythonEnv(py) => {
            probe_python(py, def, exec_digests).await
        }
        // No launch definition to find the binary in: the recipe's own
        // root and binary are the install (ds4 on m5).
        InstallRecipe::ReleaseArchive(r) if !managed => {
            let (exe, fact) =
                unmanaged_dir_fact(&r.root, &r.binary, r.version_arg.as_deref(), false).await;
            Ok(VersionRef {
                interpreter: Some(exe),
                packages: Vec::new(),
                cuda: None,
                lock_digest: String::new(),
                exec_digests,
                fact: Some(fact),
            })
        }
        InstallRecipe::SourceBuild(r) if !managed => {
            let (exe, fact) =
                unmanaged_dir_fact(&r.root, &r.binary, r.version_arg.as_deref(), true).await;
            Ok(VersionRef {
                interpreter: Some(exe),
                packages: Vec::new(),
                cuda: None,
                lock_digest: String::new(),
                exec_digests,
                fact: Some(fact),
            })
        }
        InstallRecipe::ReleaseArchive(r) => Ok(VersionRef {
            interpreter: exec_under_root(&r.root, def),
            packages: Vec::new(),
            cuda: None,
            lock_digest: String::new(),
            exec_digests,
            fact: Some(dir_fact(&r.root, r.version_arg.as_deref(), def).await),
        }),
        InstallRecipe::SourceBuild(r) => Ok(VersionRef {
            interpreter: exec_under_root(&r.root, def),
            packages: Vec::new(),
            cuda: None,
            lock_digest: String::new(),
            exec_digests,
            fact: Some(dir_fact(&r.root, r.version_arg.as_deref(), def).await),
        }),
        InstallRecipe::VendorApp(r) => {
            let (running, probe_desc) = probe_vendor(&r.probe, def).await;
            Ok(VersionRef {
                interpreter: None,
                packages: Vec::new(),
                cuda: None,
                lock_digest: String::new(),
                exec_digests,
                fact: Some(VersionFact::VendorApp {
                    running,
                    probe: probe_desc,
                }),
            })
        }
        InstallRecipe::External(_) => Ok(VersionRef {
            interpreter: None,
            packages: Vec::new(),
            cuda: None,
            lock_digest: String::new(),
            exec_digests,
            fact: None,
        }),
    }
}

/// The python-env probe: package versions, CUDA, `pip freeze` digest.
async fn probe_python(
    recipe: &PythonEnvRecipe,
    def: &WorkloadDefinition,
    exec_digests: Vec<(String, String)>,
) -> Result<VersionRef, EngineError> {
    let interpreter = python_interpreter_for(recipe, def)
        .ok_or_else(|| EngineError::Io("recipe has no interpreter to probe".into()))?;

    // Package versions as the env reports them.
    let mut packages: Vec<(String, String)> = Vec::new();
    if !recipe.packages.is_empty() {
        let names = format!("{:?}", recipe.packages); // a python list literal
        let code = format!(
            "import importlib.metadata as m, json\nprint(json.dumps({{n: m.version(n) for n in {names}}}))"
        );
        let out = run_python(&interpreter, &["-c", &code], PROBE_TIMEOUT).await?;
        let map: serde_json::Map<String, Value> = serde_json::from_str(out.trim())
            .map_err(|e| EngineError::Io(format!("version probe output: {e}")))?;
        for (k, v) in map {
            if let Some(v) = v.as_str() {
                packages.push((k, v.to_string()));
            }
        }
        packages.sort();
    }

    // CUDA as the env reports it (only meaningful with torch).
    let cuda = if recipe.packages.iter().any(|p| p == "torch") {
        run_python(
            &interpreter,
            &["-c", "import torch\nprint(torch.version.cuda)"],
            PROBE_TIMEOUT,
        )
        .await
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "None")
    } else {
        None
    };

    // Lock digest: sorted `pip freeze`.
    let freeze = run_python(&interpreter, &["-m", "pip", "freeze"], FREEZE_TIMEOUT)
        .await
        .unwrap_or_default();
    let mut lines: Vec<&str> = freeze
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines.sort();
    let lock_digest = sha256_hex_string(lines.join("\n").as_bytes());

    Ok(VersionRef {
        interpreter: Some(interpreter),
        packages,
        cuda,
        lock_digest,
        exec_digests,
        fact: None,
    })
}

/// The versioned-directory fact for release-archive and source-build
/// recipes: the dir the definition's Exec* runs from, its binary digest
/// and the binary's own version output when the recipe declares one.
/// The commit/archive checksum live in the dir's staged manifest; the
/// probe reads them when present.
async fn dir_fact(root: &str, version_arg: Option<&str>, def: &WorkloadDefinition) -> VersionFact {
    let rootp = Path::new(root);
    let none = || VersionFact::SourceBuild {
        dir: String::new(),
        commit: None,
        binary_digest: "missing".into(),
        reported: None,
    };
    let Some(exe) = exec_under_root(root, def) else {
        return none();
    };
    let rel = match Path::new(&exe).strip_prefix(rootp) {
        Ok(r) => r.to_path_buf(),
        Err(_) => return none(),
    };
    let dir_rel: PathBuf = rel.iter().take(1).collect();
    let dir_abs = rootp.join(&dir_rel);
    let binary_digest = match tokio::fs::read(&exe).await {
        Ok(bytes) => format!("sha256:{:x}", Sha256::digest(bytes)),
        Err(_) => "missing".into(),
    };
    let reported = match version_arg {
        Some(arg) => run_binary_version(&exe, arg).await.ok(),
        None => None,
    };
    // The staged manifest carries provenance: which commit was built or
    // which archive was unpacked into this dir.
    let manifest: Option<serde_json::Value> = tokio::fs::read(dir_abs.join("staged.json"))
        .await
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let commit = manifest
        .as_ref()
        .and_then(|m| m.get("commit"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let archive_sha256 = manifest
        .as_ref()
        .and_then(|m| m.get("archive_sha256"))
        .and_then(|v| v.as_str())
        .map(String::from);
    // A build dir (staged from git) names a commit in its manifest; an
    // unpacked archive does not. Both carry the same dir/digest identity.
    match commit {
        Some(commit) => VersionFact::SourceBuild {
            dir: dir_abs.display().to_string(),
            commit: Some(commit),
            binary_digest,
            reported,
        },
        None => VersionFact::ReleaseArchive {
            dir: dir_abs.display().to_string(),
            archive_sha256,
            binary_digest,
            reported,
        },
    }
}

/// `sha256:<hex>` of a file's content, or `missing`.
pub(crate) async fn file_digest(path: &str) -> String {
    match tokio::fs::read(path).await {
        Ok(bytes) => format!("sha256:{:x}", Sha256::digest(bytes)),
        Err(_) => "missing".into(),
    }
}

/// The versioned-directory fact for an install llmctl-next does not
/// manage: the recipe names the root and the binary. A source build also
/// records the commit the work tree is at (`git rev-parse HEAD`), so a
/// pull or a rebuild shows as drift even when the path never changes.
async fn unmanaged_dir_fact(
    root: &str,
    binary: &str,
    version_arg: Option<&str>,
    source: bool,
) -> (String, VersionFact) {
    let exe = Path::new(root).join(binary).display().to_string();
    let binary_digest = file_digest(&exe).await;
    let reported = match version_arg {
        Some(arg) if binary_digest != "missing" => run_binary_version(&exe, arg).await.ok(),
        _ => None,
    };
    let fact = if source {
        VersionFact::SourceBuild {
            dir: root.to_string(),
            commit: git_head(root).await,
            binary_digest,
            reported,
        }
    } else {
        VersionFact::ReleaseArchive {
            dir: root.to_string(),
            archive_sha256: None,
            binary_digest,
            reported,
        }
    };
    (exe, fact)
}

/// The commit a work tree is at; `None` when `root` is not one (or git
/// is missing).
async fn git_head(root: &str) -> Option<String> {
    let mut cmd = Command::new("git");
    cmd.args(["-C", root, "rev-parse", "HEAD"])
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    let out = tokio::time::timeout(Duration::from_secs(10), cmd.output())
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let head = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (head.len() >= 7 && head.chars().all(|c| c.is_ascii_hexdigit())).then_some(head)
}

/// The version in the first line a vendor command printed: the first
/// word that starts like a version (`0.33.1`, `v230`), else the first
/// word. `ollama version is 0.33.1` and `version: 230 (32bc781), built
/// at ...` both name their program before their version.
pub(crate) fn version_token(stdout: &str) -> Option<String> {
    let line = stdout.lines().find(|l| !l.trim().is_empty())?;
    let versionish = |w: &&str| {
        let w = w.strip_prefix('v').unwrap_or(w);
        w.chars().next().is_some_and(|c| c.is_ascii_digit())
    };
    let clean = |w: &str| w.trim_matches(|c: char| matches!(c, ',' | ';' | '(' | ')')).to_string();
    line.split_whitespace()
        .find(versionish)
        .or_else(|| line.split_whitespace().next())
        .map(clean)
        .filter(|w| !w.is_empty())
}

/// True for a vendor app's HTTP probe that got no answer: the service is
/// not serving, so there is no running version to compare or to pin.
pub fn unanswered_http_probe(fact: &Option<VersionFact>) -> bool {
    matches!(
        fact,
        Some(VersionFact::VendorApp { running: None, probe }) if probe.starts_with("http_json")
    )
}

/// What a probe result may be stored as. An unanswered HTTP probe is
/// left out of the pin (a pinned "no version" would read as drift on the
/// first restart); the second value says whether the pin is complete.
pub fn storable(mut live: VersionRef) -> (VersionRef, bool) {
    if unanswered_http_probe(&live.fact) {
        live.fact = None;
        (live, false)
    } else {
        (live, true)
    }
}

/// The running version of a vendor app, per its declared probe. An app
/// that cannot be observed reports `None`: a stopped server for an HTTP
/// probe (not drift, see [`version_drift`]), a missing app for the others.
async fn probe_vendor(probe: &VendorProbe, def: &WorkloadDefinition) -> (Option<String>, String) {
    match probe {
        VendorProbe::HttpJson { path, field } => {
            let desc = format!("http_json {path}");
            match crate::probe_http::fetch_json_field(def.port, path, field, PROBE_TIMEOUT).await {
                Some(v) => (Some(v), desc),
                None => (None, desc),
            }
        }
        VendorProbe::Command { argv } => {
            let desc = argv.join(" ");
            if argv.is_empty() {
                return (None, desc);
            }
            let mut cmd = Command::new(&argv[0]);
            cmd.args(&argv[1..]).kill_on_drop(true);
            match tokio::time::timeout(PROBE_TIMEOUT, cmd.output()).await {
                Ok(Ok(out)) if out.status.success() => {
                    (version_token(&String::from_utf8_lossy(&out.stdout)), desc)
                }
                _ => (None, desc),
            }
        }
        VendorProbe::DirName { path } => {
            let desc = format!("dir_name {path}");
            let mut best: Option<String> = None;
            if let Ok(entries) = std::fs::read_dir(path) {
                for e in entries.flatten() {
                    if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if best.as_deref().map(|b| name.as_str() > b).unwrap_or(true) {
                            best = Some(name);
                        }
                    }
                }
            }
            (best, desc)
        }
    }
}

fn sha256_hex_string(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

/// Content digests of every `Exec*` path outside the system prefixes.
pub async fn exec_digests_of(def: &WorkloadDefinition) -> Vec<(String, String)> {
    let llmctl_core::Ownership::Managed { definition, .. } = &def.ownership else {
        return Vec::new();
    };
    let mut paths: Vec<String> = Vec::new();
    for cmd in definition
        .exec_start_pre
        .iter()
        .chain(definition.exec_start.iter())
        .chain(definition.exec_start_post.iter())
        .chain(definition.exec_stop.iter())
    {
        if is_system_path(&cmd.path) {
            continue;
        }
        if !paths.contains(&cmd.path) {
            paths.push(cmd.path.clone());
        }
    }
    let mut out = Vec::new();
    for path in paths {
        let digest = match tokio::fs::read(&path).await {
            Ok(bytes) => format!("sha256:{:x}", Sha256::digest(bytes)),
            Err(_) => "missing".into(),
        };
        out.push((path, digest));
    }
    out
}

fn is_system_path(path: &str) -> bool {
    ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt"]
        .iter()
        .any(|p| path == *p || path.starts_with(&format!("{p}/")))
}

/// Offline validation of a staged install (plan D.4, M4a-generalized):
/// python-env recipes import, report the expected version and load the
/// served model's config; release-archive and source-build recipes prove
/// the binary exists, is executable and — when the recipe declares a
/// version argument — prints the expected version. Vendor apps and
/// external recipes have nothing to validate offline. Never touches the
/// serving install.
pub async fn validate_offline(
    recipe: &InstallRecipe,
    executable: &str,
    version: &str,
    model: &str,
) -> Result<Value, String> {
    match recipe {
        InstallRecipe::PythonEnv(py) => {
            let code = match py.name.as_str() {
                "vllm" => format!(
            "import json, torch, vllm\nfrom transformers import AutoConfig\nc = AutoConfig.from_pretrained({model:?})\nprint(json.dumps({{'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'model_type': c.model_type}}))"
        ),
                other => format!(
                    "import json\nimport {other} as pkg\nprint(json.dumps(pkg.validate({model:?})))"
                ),
            };
            let out = run_python(executable, &["-c", &code], PROBE_TIMEOUT)
                .await
                .map_err(|e| format!("validation command failed: {e}"))?;
            serde_json::from_str(out.trim()).map_err(|e| format!("validation output not JSON: {e}"))
        }
        InstallRecipe::ReleaseArchive(r) => {
            validate_dir_binary(executable, version, r.version_arg.as_deref(), r.version_pattern.as_deref())
                .await
        }
        InstallRecipe::SourceBuild(r) => {
            validate_dir_binary(executable, version, r.version_arg.as_deref(), r.version_pattern.as_deref())
                .await
        }
        InstallRecipe::VendorApp(_) | InstallRecipe::External(_) => Ok(serde_json::json!({
            "note": "vendor-app/external recipes observe; nothing to validate offline"
        })),
    }
}

/// Binary checks for versioned-directory recipes: exists, executable, and
/// its version output matches when the recipe declares an argument.
async fn validate_dir_binary(
    executable: &str,
    version: &str,
    version_arg: Option<&str>,
    version_pattern: Option<&str>,
) -> Result<Value, String> {
    let meta = tokio::fs::metadata(executable)
        .await
        .map_err(|e| format!("binary {executable}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("{executable} is not a regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o111 == 0 {
            return Err(format!("{executable} is not executable"));
        }
    }
    let expect = version_pattern.unwrap_or(version);
    let reported = match version_arg {
        Some(arg) => {
            let out = run_binary_version(executable, arg)
                .await
                .map_err(|e| format!("binary version check: {e}"))?;
            if !out.contains(expect) {
                return Err(format!(
                    "binary reports {out:?}, expected it to contain {expect:?}"
                ));
            }
            Some(out)
        }
        None => None,
    };
    Ok(serde_json::json!({
        "binary": executable,
        "reported": reported,
        "expected": expect,
    }))
}

/// Run `<binary> <arg>` and return its version-ish output: the first
/// non-empty line, stdout preferred (server binaries — llama-server —
/// print `--version` to stderr).
pub async fn run_binary_version(binary: &str, arg: &str) -> Result<String, String> {
    let body = async {
        use tokio::io::AsyncReadExt;
        let mut cmd = Command::new(binary);
        cmd.arg(arg)
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("spawning {binary}: {e}"))?;
        let mut out_buf = Vec::new();
        let mut err_buf = Vec::new();
        if let Some(mut s) = child.stdout.take() {
            let _ = s.read_to_end(&mut out_buf).await;
        }
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_end(&mut err_buf).await;
        }
        let status = child.wait().await.map_err(|e| format!("{binary}: {e}"))?;
        if !status.success() {
            return Err(format!(
                "{binary} {arg} exited {}: {}",
                status,
                String::from_utf8_lossy(&err_buf).trim()
            ));
        }
        let from = |buf: &[u8]| {
            String::from_utf8_lossy(buf)
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("")
                .to_string()
        };
        let stdout_line = from(&out_buf);
        Ok(if stdout_line.is_empty() {
            from(&err_buf)
        } else {
            stdout_line
        })
    };
    match tokio::time::timeout(PROBE_TIMEOUT, body).await {
        Ok(r) => r,
        Err(_) => Err(format!("{binary} {arg} timed out")),
    }
}

/// File name of the env file `apply: "env"` adds, inside the staged env.
pub const OVERRIDE_ENV_FILE: &str = "llmctl-next-override.env";

/// Where the override env file for `interpreter` lives and what it holds.
/// The staged env is `<env>/bin/python`, so the file sits in `<env>/`.
pub fn override_env(interpreter: &str, prefix: &str) -> (std::path::PathBuf, String) {
    let env_dir = Path::new(interpreter)
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new("/"));
    (
        env_dir.join(OVERRIDE_ENV_FILE),
        format!("{prefix}_PYTHON={interpreter}\n"),
    )
}

/// The definition for the staged executable (M4a-generalized):
///
/// - python-env `exec` recipes run the interpreter directly (rewrite
///   `ExecStart`); `env` recipes keep a launcher and point its env var at
///   the new interpreter.
/// - release-archive and source-build recipes swap every `Exec*` path that
///   lives under the recipe root to the new version directory — one path
///   swap in the unit, the M2 staged-upgrade flow generalized (owner
///   decision 3).
///
/// `env` adds an `EnvironmentFile=` *last* rather than an inline
/// `Environment=`: systemd lets `EnvironmentFile=` override `Environment=`,
/// so an inline override loses to a unit env file that sets the same
/// variable (the M2 pilot's `vllm-engine2.env` sets `VLLM_PYTHON`). A later
/// env file overrides earlier ones. The caller writes the file
/// (`override_env`); its content digest is pinned here.
///
/// Fails when an `env`-apply recipe names no prefix — the `VLLM` default
/// applies to the vllm recipe only (M4a: no `VLLM` defaults outside the
/// vLLM recipe).
pub fn rewrite_definition(
    old: &LaunchDefinition,
    recipe: &InstallRecipe,
    executable: &str,
) -> Result<LaunchDefinition, String> {
    let mut next = old.clone();
    match recipe {
        InstallRecipe::PythonEnv(py) => match py.apply.as_str() {
            "exec" => {
                for cmd in &mut next.exec_start {
                    if !is_system_path(&cmd.path) || cmd.path.contains("python") {
                        cmd.path = executable.to_string();
                        if !cmd.argv.is_empty() {
                            cmd.argv[0] = executable.to_string();
                        }
                    }
                }
            }
            _ => {
                let prefix = match py.env_prefix.clone() {
                    Some(p) => p,
                    None if py.name == "vllm" => "VLLM".into(),
                    None => {
                        return Err(format!(
                            "recipe {} applies env but names no env_prefix",
                            py.name
                        ))
                    }
                };
                let (path, content) = override_env(executable, &prefix);
                next.environment_files
                    .retain(|f| !f.path.ends_with(&format!("/{OVERRIDE_ENV_FILE}")));
                next.environment_files.push(llmctl_core::EnvFileRef {
                    path: path.display().to_string(),
                    optional: false,
                    content_digest: format!("sha256:{:x}", Sha256::digest(content.as_bytes())),
                });
            }
        },
        InstallRecipe::ReleaseArchive(r) => rewrite_root_paths(&mut next, &r.root, executable),
        InstallRecipe::SourceBuild(r) => rewrite_root_paths(&mut next, &r.root, executable),
        InstallRecipe::VendorApp(_) | InstallRecipe::External(_) => {
            return Err("vendor-app and external recipes promote nothing".into());
        }
    }
    Ok(next)
}

/// Swap every `Exec*` path — and every inline `Environment=` value that is
/// a path under `root` — to the new version directory. The primary
/// executable maps exactly (old and new installs need not share an
/// internal layout: the live llama.cpp build dir puts the binary at
/// `build/bin/llama-server`, a staged checkout at `…/<id>/build/bin/…`);
/// every other root path swaps its version directory (first component).
/// The env case is the llama.cpp lanes' shape: `ExecStart` runs the guard
/// wrapper while `LLAMA_GUARD_REAL_BIN` names the versioned binary.
fn rewrite_root_paths(next: &mut LaunchDefinition, root: &str, new_executable: &str) {
    let rootp = Path::new(root);
    let new_rel = Path::new(new_executable)
        .strip_prefix(rootp)
        .map(Path::to_path_buf)
        .unwrap_or_default();
    // The version directory is the first component under root.
    let new_dir: PathBuf = new_rel.iter().take(1).collect();
    // The old primary executable, as this definition resolves it.
    let old_exec: Option<String> = next
        .exec_start_pre
        .iter()
        .chain(next.exec_start.iter())
        .chain(next.exec_start_post.iter())
        .map(|c| c.path.as_str())
        .chain(next.environment.iter().map(|(_, v)| v.as_str()))
        .find(|p| Path::new(p).starts_with(rootp))
        .map(str::to_owned);
    let map = |p: &str| -> Option<String> {
        if old_exec.as_deref() == Some(p) {
            return Some(new_executable.to_string());
        }
        let rel = Path::new(p).strip_prefix(rootp).ok()?;
        let tail: PathBuf = rel.iter().skip(1).collect();
        Some(rootp.join(&new_dir).join(tail).display().to_string())
    };
    for cmd in next
        .exec_start_pre
        .iter_mut()
        .chain(next.exec_start.iter_mut())
        .chain(next.exec_start_post.iter_mut())
        .chain(next.exec_stop.iter_mut())
    {
        if let Some(mapped) = map(&cmd.path) {
            cmd.path = mapped;
        }
        if !cmd.argv.is_empty() {
            if let Some(mapped) = map(&cmd.argv[0]) {
                cmd.argv[0] = mapped;
            }
        }
    }
    for (_name, value) in next.environment.iter_mut() {
        if let Some(mapped) = map(value) {
            *value = mapped;
        }
    }
}

/// The unit's processes must actually run `interpreter`: some process in
/// the unit's cgroup was exec'd as `interpreter` (argv[0]) or has files
/// from its env mapped. Asking the staged interpreter for its own package
/// versions proves nothing about what the unit runs (M2 pilot finding).
pub async fn verify_running_interpreter(
    unit: &llmctl_core::UnitName,
    interpreter: &str,
) -> Result<(), String> {
    let scope = match unit.scope {
        llmctl_core::UnitScopeMarker::User => "--user",
        llmctl_core::UnitScopeMarker::System => "--system",
        llmctl_core::UnitScopeMarker::Launchd => return Err("launchd labels have no cgroup".into()),
    };
    let out = tokio::time::timeout(
        PROBE_TIMEOUT,
        Command::new("systemctl")
            .args([scope, "show", "-p", "ControlGroup", "--value", &unit.name])
            .output(),
    )
    .await
    .map_err(|_| "systemctl show timed out".to_string())?
    .map_err(|e| format!("systemctl show: {e}"))?;
    let cgroup = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if cgroup.is_empty() {
        return Err(format!("{unit} has no control group (not running?)"));
    }
    let pids = cgroup_pids(&Path::new("/sys/fs/cgroup").join(cgroup.trim_start_matches('/')));
    if pids.is_empty() {
        return Err(format!("{unit} has no processes"));
    }
    let env_prefix = Path::new(interpreter)
        .parent()
        .and_then(Path::parent)
        .map(|p| format!("{}/", p.display()))
        .unwrap_or_default();
    let mut seen = Vec::new();
    for pid in &pids {
        let argv: Vec<String> = std::fs::read(format!("/proc/{pid}/cmdline"))
            .ok()
            .map(|b| {
                b.split(|c| *c == 0)
                    .filter(|a| !a.is_empty())
                    .map(|a| String::from_utf8_lossy(a).into_owned())
                    .collect()
            })
            .unwrap_or_default();
        // argv[0] for an ELF the unit execs directly; a later element for
        // a shebang script (the interpreter runs `python3 <script>`), and
        // for `<venv>/bin/python -m pkg` styles.
        if argv.iter().any(|a| a == interpreter) {
            return Ok(());
        }
        let maps = std::fs::read_to_string(format!("/proc/{pid}/maps")).unwrap_or_default();
        if !env_prefix.is_empty() && maps.contains(&env_prefix) {
            return Ok(());
        }
        if let Some(argv0) = argv.first() {
            seen.push(argv0.clone());
        }
    }
    seen.sort();
    seen.dedup();
    Err(format!(
        "no process of {unit} runs {interpreter} (running: {})",
        seen.join(", ")
    ))
}

/// Every pid in a cgroup and its children.
fn cgroup_pids(dir: &Path) -> Vec<u32> {
    let mut pids: Vec<u32> = std::fs::read_to_string(dir.join("cgroup.procs"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                pids.extend(cgroup_pids(&e.path()));
            }
        }
    }
    pids
}

/// `X.Y` of an interpreter, so a staged env uses the same Python minor
/// version as the env it replaces.
pub async fn python_minor(interpreter: &str) -> Result<String, EngineError> {
    let out = run_python(
        interpreter,
        &["-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        PROBE_TIMEOUT,
    )
    .await?;
    Ok(out.trim().to_string())
}

/// Verify the environment reports exactly `packages` through
/// `importlib.metadata` (the `VerifyVersion` step).
pub async fn verify_packages(
    interpreter: &str,
    packages: &[(String, String)],
) -> Result<(), String> {
    if packages.is_empty() {
        return Ok(());
    }
    let names: Vec<&str> = packages.iter().map(|(n, _)| n.as_str()).collect();
    let code = format!(
        "import importlib.metadata as m, json\nprint(json.dumps({{n: m.version(n) for n in {names:?}}}))",
        names = names
    );
    let out = run_python(interpreter, &["-c", &code], PROBE_TIMEOUT)
        .await
        .map_err(|e| e.to_string())?;
    let got: std::collections::BTreeMap<String, String> =
        serde_json::from_str(out.trim()).map_err(|e| format!("version probe output: {e}"))?;
    let mut wrong = Vec::new();
    for (name, want) in packages {
        match got.get(name) {
            Some(have) if have == want => {}
            Some(have) => wrong.push(format!("{name}: running {have}, expected {want}")),
            None => wrong.push(format!("{name}: not installed")),
        }
    }
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(wrong.join("; "))
    }
}

async fn run_python(
    interpreter: &str,
    args: &[&str],
    timeout: Duration,
) -> Result<String, EngineError> {
    let mut cmd = Command::new(interpreter);
    cmd.args(args).kill_on_drop(true);
    let out = match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(EngineError::Io(format!("spawning {interpreter}: {e}"))),
        Err(_) => {
            return Err(EngineError::Timeout(format!(
                "{interpreter} {} timed out",
                args.first().unwrap_or(&"")
            )))
        }
    };
    if !out.status.success() {
        return Err(EngineError::Io(format!(
            "{interpreter} {} exited {}: {}",
            args.first().unwrap_or(&""),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// True when the recomputed version facts differ from the pin. The
/// per-kind fact is compared only when the pin carries one (a pre-M4a pin
/// of a new-kind recipe declares nothing to compare — the daemon re-pins
/// it at startup) and only when the live probe could observe one (a
/// vendor server that is stopped reports no version).
pub fn version_drift(pinned: &VersionRef, live: &VersionRef) -> Vec<llmctl_core::FieldDiff> {
    let mut diffs = Vec::new();
    let mut push = |field: &str, expected: Value, actual: Value| {
        if expected != actual {
            diffs.push(llmctl_core::FieldDiff {
                field: field.to_string(),
                expected,
                actual,
            });
        }
    };
    push(
        "version.interpreter",
        serde_json::to_value(&pinned.interpreter).unwrap_or(Value::Null),
        serde_json::to_value(&live.interpreter).unwrap_or(Value::Null),
    );
    push(
        "version.packages",
        serde_json::to_value(&pinned.packages).unwrap_or(Value::Null),
        serde_json::to_value(&live.packages).unwrap_or(Value::Null),
    );
    push(
        "version.cuda",
        serde_json::to_value(&pinned.cuda).unwrap_or(Value::Null),
        serde_json::to_value(&live.cuda).unwrap_or(Value::Null),
    );
    push(
        "version.lock_digest",
        Value::String(pinned.lock_digest.clone()),
        Value::String(live.lock_digest.clone()),
    );
    push(
        "version.exec_digests",
        serde_json::to_value(&pinned.exec_digests).unwrap_or(Value::Null),
        serde_json::to_value(&live.exec_digests).unwrap_or(Value::Null),
    );
    // A server that is not answering has no running version: a stopped
    // Ollama is not drift, and the start guard must let it start.
    if pinned.fact.is_some() && !unanswered_http_probe(&live.fact) {
        push(
            "version.fact",
            serde_json::to_value(&pinned.fact).unwrap_or(Value::Null),
            serde_json::to_value(&live.fact).unwrap_or(Value::Null),
        );
    }
    diffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmctl_core::{EnvFileRef, ReleaseArchiveRecipe};

    fn recipe(apply: &str) -> InstallRecipe {
        InstallRecipe::PythonEnv(PythonEnvRecipe {
            name: "vllm".into(),
            interpreter: None,
            packages: vec!["vllm".into()],
            apply: apply.into(),
            env_prefix: Some("VLLM".into()),
            find_links: None,
            model_config: None,
            index_url: None,
        })
    }

    #[test]
    fn env_apply_adds_a_last_env_file_instead_of_an_inline_override() {
        let unit_env = EnvFileRef {
            path: "/home/u/AI/services/vllm-engine2.env".into(),
            optional: false,
            content_digest: "sha256:aa".into(),
        };
        let old = LaunchDefinition {
            environment_files: vec![unit_env.clone()],
            ..Default::default()
        };
        let py = "/d/envs/engine2-0.32.0-abc/bin/python";
        let next = rewrite_definition(&old, &recipe("env"), py).unwrap();
        assert!(next.environment.is_empty(), "no inline override");
        assert_eq!(next.environment_files.len(), 2);
        assert_eq!(next.environment_files[0], unit_env);
        let (path, content) = override_env(py, "VLLM");
        assert_eq!(
            path,
            Path::new("/d/envs/engine2-0.32.0-abc/llmctl-next-override.env")
        );
        assert_eq!(content, format!("VLLM_PYTHON={py}\n"));
        let last = &next.environment_files[1];
        assert_eq!(last.path, path.display().to_string());
        assert_eq!(
            last.content_digest,
            format!("sha256:{:x}", Sha256::digest(content.as_bytes()))
        );

        // Re-applying (another promote) replaces the override, never stacks.
        let again = rewrite_definition(&next, &recipe("env"), "/d/envs/x/bin/python").unwrap();
        assert_eq!(again.environment_files.len(), 2);
        assert_eq!(
            again.environment_files[1].path,
            "/d/envs/x/llmctl-next-override.env"
        );
    }

    #[test]
    fn env_apply_without_a_prefix_is_refused_outside_the_vllm_recipe() {
        let mut py = match recipe("env") {
            InstallRecipe::PythonEnv(r) => r,
            _ => unreachable!(),
        };
        py.env_prefix = None;
        py.name = "mtplx".into();
        let err = rewrite_definition(
            &LaunchDefinition::default(),
            &InstallRecipe::PythonEnv(py.clone()),
            "/d/envs/x/bin/python",
        )
        .unwrap_err();
        assert!(err.contains("env_prefix"), "{err}");
        // The vllm recipe's historical default still applies.
        py.name = "vllm".into();
        assert!(rewrite_definition(
            &LaunchDefinition::default(),
            &InstallRecipe::PythonEnv(py),
            "/d/envs/x/bin/python"
        )
        .is_ok());
    }

    /// A definition whose Exec* live under a recipe root swaps to the new
    /// version directory — one path swap, argv[0] included (M4a).
    #[test]
    fn dir_recipes_swap_every_exec_path_under_the_root() {
        let recipe = InstallRecipe::ReleaseArchive(ReleaseArchiveRecipe {
            name: "llama.cpp".into(),
            root: "/d/llama.cpp".into(),
            url: "https://x/llama-{version}.tgz".into(),
            checksum: None,
            checksums: Default::default(),
            strip_components: 1,
            binary: "build/bin/llama-server".into(),
            version_arg: Some("--version".into()),
            version_pattern: None,
        });
        let old = LaunchDefinition {
            exec_start: vec![llmctl_core::ExecCommand {
                path: "/usr/bin/llama-server-guarded".into(),
                argv: vec!["/usr/bin/llama-server-guarded".into(), "--host".into()],
                ignore_errors: false,
            }],
            exec_start_pre: vec![llmctl_core::ExecCommand {
                path: "/usr/bin/id".into(),
                argv: vec!["/usr/bin/id".into()],
                ignore_errors: false,
            }],
            // The llama.cpp lanes' shape: the versioned binary rides an
            // env var (LLAMA_GUARD_REAL_BIN), the wrapper is exec'd. The
            // live dir is the bare build output (`build/bin/…`).
            environment: vec![
                ("LLAMA_GUARD_REAL_BIN".into(), "/d/llama.cpp/build/bin/llama-server".into()),
                ("LLAMA_GUARD_MIN_FREE_MIB".into(), "29500".into()),
            ],
            ..Default::default()
        };
        // The staged install is a full checkout: the binary lives one
        // level deeper. The exact mapping must bridge the layouts.
        let next = rewrite_definition(
            &old,
            &recipe,
            "/d/llama.cpp/gpt-oss-120b-a9d43ecc3-7bda46f3/build/bin/llama-server",
        )
        .unwrap();
        // The guard wrapper and system paths stay untouched...
        assert_eq!(next.exec_start[0].path, "/usr/bin/llama-server-guarded");
        assert_eq!(next.exec_start_pre[0].path, "/usr/bin/id");
        // ...the env value under the root maps exactly to the new
        // executable, across the differing internal layout.
        assert_eq!(
            next.environment[0],
            (
                "LLAMA_GUARD_REAL_BIN".to_string(),
                "/d/llama.cpp/gpt-oss-120b-a9d43ecc3-7bda46f3/build/bin/llama-server"
                    .to_string()
            )
        );
        assert_eq!(next.environment[1].1, "29500");
    }

    /// Drift: a changed per-kind fact is a finding; a pin without a fact
    /// (pre-M4a) never reads as false drift.
    #[test]
    fn version_drift_compares_the_per_kind_fact_when_the_pin_carries_one() {
        let base = || VersionRef {
            interpreter: None,
            packages: vec![],
            cuda: None,
            lock_digest: String::new(),
            exec_digests: vec![],
            fact: Some(VersionFact::VendorApp {
                running: Some("0.34.3".into()),
                probe: "http_json /api/version".into(),
            }),
        };
        let pinned = base();
        let same = base();
        assert!(version_drift(&pinned, &same).is_empty());

        let mut drifted = base();
        if let Some(VersionFact::VendorApp { running, .. }) = &mut drifted.fact {
            *running = Some("0.40.16".into());
        }
        let diffs = version_drift(&pinned, &drifted);
        assert_eq!(diffs.len(), 1, "{diffs:?}");
        assert_eq!(diffs[0].field, "version.fact");

        let mut legacy_pin = base();
        legacy_pin.fact = None;
        assert!(version_drift(&legacy_pin, &drifted).is_empty());
    }

    /// The start guard runs `version_drift`: a pinned Ollama that is
    /// stopped answers nothing, and that must not refuse its start.
    #[test]
    fn a_stopped_http_vendor_app_is_not_drift_but_a_missing_app_is() {
        let vendor = |running: Option<&str>, probe: &str| VersionRef {
            interpreter: None,
            packages: vec![],
            cuda: None,
            lock_digest: String::new(),
            exec_digests: vec![],
            fact: Some(VersionFact::VendorApp {
                running: running.map(String::from),
                probe: probe.into(),
            }),
        };
        let pinned = vendor(Some("0.33.1"), "http_json /api/version");
        let stopped = vendor(None, "http_json /api/version");
        assert!(version_drift(&pinned, &stopped).is_empty());
        assert!(unanswered_http_probe(&stopped.fact));
        let (stored, complete) = storable(stopped);
        assert!(!complete && stored.fact.is_none());

        // The other probes do not depend on a running server: nothing
        // found there means the app is gone.
        let dir_pinned = vendor(Some("0.0.18-1"), "dir_name /home/u/.lmstudio/llmster");
        let dir_gone = vendor(None, "dir_name /home/u/.lmstudio/llmster");
        let diffs = version_drift(&dir_pinned, &dir_gone);
        assert_eq!(diffs.len(), 1, "{diffs:?}");
        let (stored, complete) = storable(dir_gone);
        assert!(complete && stored.fact.is_some());
    }

    #[test]
    fn the_version_is_the_first_word_that_looks_like_one() {
        assert_eq!(version_token("0.34.3\n").as_deref(), Some("0.34.3"));
        assert_eq!(version_token("ollama version is 0.33.1\n").as_deref(), Some("0.33.1"));
        assert_eq!(
            version_token("version: 230 (32bc7813), built at 2026-06-25T04:03:31Z").as_deref(),
            Some("230")
        );
        assert_eq!(version_token("\nv1.2.3-rc1, built today").as_deref(), Some("v1.2.3-rc1"));
        // Nothing version-like: the first word, as before.
        assert_eq!(version_token("stable channel").as_deref(), Some("stable"));
        assert_eq!(version_token("  \n"), None);
    }

    fn external_workload() -> WorkloadDefinition {
        let host = llmctl_core::HostId::new("workstation").unwrap();
        crate::app::sample_inventory(&host).workloads.remove(0)
    }

    #[tokio::test]
    async fn declared_pin_files_are_digested_with_the_exec_files() {
        let dir = tempfile::tempdir().unwrap();
        let engine = dir.path().join("engine-binary");
        std::fs::write(&engine, b"v1").unwrap();
        let mut def = external_workload();
        def.pin_files = vec![
            engine.display().to_string(),
            dir.path().join("gone").display().to_string(),
        ];
        let recipe = InstallRecipe::External(llmctl_core::ExternalRecipe {
            name: "strata".into(),
            note: None,
        });
        let pinned = probe(&recipe, &def).await.unwrap();
        assert_eq!(pinned.exec_digests.len(), 2);
        assert!(pinned.exec_digests[0].1.starts_with("sha256:"));
        assert_eq!(pinned.exec_digests[1].1, "missing");

        std::fs::write(&engine, b"v2").unwrap();
        let live = probe(&recipe, &def).await.unwrap();
        let diffs = version_drift(&pinned, &live);
        assert_eq!(diffs.len(), 1, "{diffs:?}");
        assert_eq!(diffs[0].field, "version.exec_digests");
    }

    #[tokio::test]
    async fn an_unmanaged_source_build_is_identified_by_commit_and_binary() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().display().to_string();
        let git = |args: &[&str]| {
            let ok = std::process::Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
                .args(args)
                .output()
                .unwrap()
                .status
                .success();
            assert!(ok, "git {args:?}");
        };
        std::fs::write(dir.path().join("engine"), b"built from A").unwrap();
        git(&["init", "-q"]);
        git(&["add", "engine"]);
        git(&["commit", "-q", "-m", "A"]);

        let recipe = InstallRecipe::SourceBuild(llmctl_core::SourceBuildRecipe {
            name: "ds4".into(),
            root: root.clone(),
            git: "https://example.com/ds4.git".into(),
            git_ref: None,
            build: "true".into(),
            binary: "engine".into(),
            version_arg: None,
            version_pattern: None,
        });
        let def = external_workload();
        assert!(matches!(def.ownership, llmctl_core::Ownership::External));
        let pinned = probe(&recipe, &def).await.unwrap();
        assert_eq!(pinned.interpreter.as_deref(), Some(dir.path().join("engine").to_str().unwrap()));
        let Some(VersionFact::SourceBuild { dir: d, commit, binary_digest, .. }) = &pinned.fact else {
            panic!("expected a source-build fact: {:?}", pinned.fact);
        };
        assert_eq!(d, &root);
        assert_eq!(commit.as_ref().map(String::len), Some(40), "{commit:?}");
        assert!(binary_digest.starts_with("sha256:"));

        // A rebuild from a new commit moves both facts.
        std::fs::write(dir.path().join("engine"), b"built from B").unwrap();
        git(&["commit", "-q", "-am", "B"]);
        let live = probe(&recipe, &def).await.unwrap();
        let diffs = version_drift(&pinned, &live);
        assert_eq!(diffs.len(), 1, "{diffs:?}");
        assert_eq!(diffs[0].field, "version.fact");
    }
}
