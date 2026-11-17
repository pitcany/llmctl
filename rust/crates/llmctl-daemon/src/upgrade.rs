//! Staged installs and upgrades with rollback (plan D, generalized by
//! M4a to every recipe kind): version ownership without auto-update.
//! Stage builds a *new* versioned directory side by side (`uv venv` for
//! python-env recipes; fetch+unpack for release archives; git+build for
//! source builds — the serving install is never touched), validate checks
//! it offline, and promote is an ordinary admitted operation — stop, apply
//! the new definition, start, verify readiness, identity and version —
//! whose failure rolls back to the pinned N by the normal executor path.
//!
//! Old versions stay until an explicit `gc`; shared weights and HF caches
//! are never touched. Vendor apps are never installed by llmctl:
//! `vendor_plan` reports the observed/pinned version and the owner-run
//! backup/reinstall path instead.

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::Utc;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use tokio::process::Command;

use llmctl_client::wire::PROTOCOL_VERSION;
use llmctl_core::{
    InstallRecipe, LaunchDefinition, OpKind, Ownership, PythonEnvRecipe, ReleaseArchiveRecipe,
    SourceBuildRecipe, WorkloadName,
};

use crate::app::AppState;
use crate::db::UpgradeRow;
use crate::ops::{AdmitOutcome, AdmitParams, OpsError};

#[derive(Debug, thiserror::Error)]
pub enum UpgradeError {
    #[error("{0}")]
    Refused(String),
    #[error(transparent)]
    Ops(#[from] OpsError),
    #[error("database error: {0}")]
    Db(#[from] crate::db::DbError),
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for UpgradeError {
    fn from(e: std::io::Error) -> Self {
        UpgradeError::Io(e.to_string())
    }
}

fn refuse(msg: impl Into<String>) -> UpgradeError {
    UpgradeError::Refused(msg.into())
}

/// Rough floor for a staged python env (vLLM wheels unpack to ~10 GB).
const MIN_FREE_BYTES: u64 = 10 * 1024 * 1024 * 1024;
/// Rough floor for a staged versioned directory (a CUDA llama.cpp build
/// is a few GB; the archive adds its compressed size once).
const MIN_FREE_DIR_BYTES: u64 = 1024 * 1024 * 1024;

pub struct StageOutcome {
    pub row: UpgradeRow,
    pub validation: serde_json::Value,
}

fn env_id(workload: &str, version: &str) -> String {
    let hash = Sha256::digest(format!("{workload}:{version}").as_bytes());
    let hex = format!("{:x}", hash);
    format!("{workload}-{version}-{}", &hex[..8])
}

/// Each promote/rollback is a new operation. Keys built from the version
/// or the history row alone replayed an earlier promote's success when the
/// same version was promoted again after a rollback (M2 pilot window 2).
fn unique_suffix() -> String {
    Utc::now().timestamp_millis().to_string()
}

fn envs_root(state: &AppState) -> PathBuf {
    state.paths.data_dir.join("envs")
}

/// `sha256:<hex>` of a file, streamed in 1 MiB chunks.
async fn hash_file(path: &Path) -> Result<String, UpgradeError> {
    use tokio::io::AsyncReadExt;
    let mut f = tokio::fs::File::open(path)
        .await
        .map_err(|e| UpgradeError::Io(format!("opening {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .await
            .map_err(|e| UpgradeError::Io(format!("reading {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

/// The directory versioned installs of this recipe live under (M4a: the
/// recipe's declared root; python-env keeps the daemon's envs dir).
fn installs_root(state: &AppState, recipe: &InstallRecipe) -> PathBuf {
    recipe.install_root(&envs_root(state))
}

/// The package spec a recipe installs for `version` (vLLM ships as one
/// wheel; fakeengine likewise).
fn install_spec(recipe_name: &str, version: &str) -> String {
    let pkg = match recipe_name {
        "vllm" => "vllm",
        other => other,
    };
    format!("{pkg}=={version}")
}

async fn run_logged(cmd: &mut Command, log: &Path, what: &str) -> Result<String, UpgradeError> {
    run_logged_with(cmd, log, what, Duration::from_secs(600)).await
}

/// Run a command, append its transcript to `log`, and fail with the log's
/// path on a non-zero exit (the log is the evidence the owner reads).
async fn run_logged_with(
    cmd: &mut Command,
    log: &Path,
    what: &str,
    timeout: Duration,
) -> Result<String, UpgradeError> {
    cmd.kill_on_drop(true);
    let out = tokio::time::timeout(timeout, cmd.output())
        .await
        .map_err(|_| UpgradeError::Io(format!("{what} timed out")))?
        .map_err(|e| UpgradeError::Io(format!("spawning {what}: {e}")))?;
    let text = format!(
        "$ {what}\nstdout:\n{}\nstderr:\n{}\n",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    use tokio::io::AsyncWriteExt;
    let mut f = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log)
        .await
        .map_err(|e| UpgradeError::Io(format!("opening {}: {e}", log.display())))?;
    let _ = f.write_all(text.as_bytes()).await;
    if !out.status.success() {
        return Err(refuse(format!(
            "{what} exited {} (see {})",
            out.status,
            log.display()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn free_bytes(path: &Path) -> Result<u64, UpgradeError> {
    let out = std::process::Command::new("df")
        .arg("-B1")
        .arg("--output=avail")
        .arg(path)
        .output()
        .map_err(|e| UpgradeError::Io(format!("spawning df: {e}")))?;
    if !out.status.success() {
        return Err(UpgradeError::Io("df failed".into()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .nth(1)
        .and_then(|l| l.trim().parse::<u64>().ok())
        .ok_or_else(|| UpgradeError::Io("df output not understood".into()))
}

async fn managed_parts(
    state: &AppState,
    name: &str,
) -> Result<(WorkloadName, LaunchDefinition, InstallRecipe), UpgradeError> {
    let inv = state.ops.inventory().await;
    let wl = WorkloadName::new(name).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let Some(def) = inv.workload(&wl) else {
        return Err(OpsError::UnknownWorkload(name.to_string()).into());
    };
    let Ownership::Managed { definition, .. } = &def.ownership else {
        return Err(refuse(format!("{name} is not a managed workload")));
    };
    let Some(recipe) = &def.recipe else {
        return Err(refuse(format!(
            "{name} has no recipe configured; version pinning and staging are off"
        )));
    };
    Ok((wl, (**definition).clone(), recipe.clone()))
}

/// Refuse a restage that would rebuild the dir the workload runs from, or
/// the one the last promote would roll back to (the dir is removed and
/// rebuilt in place by a restage).
async fn refuse_restage_of_live_or_rollback_dir(
    state: &AppState,
    name: &str,
    wl: &WorkloadName,
    recipe: &InstallRecipe,
    dir: &Path,
) -> Result<(), UpgradeError> {
    let live =
        crate::version::primary_executable_for(recipe, &inv_def_for(state, wl).await);
    if live
        .as_deref()
        .is_some_and(|p| Path::new(p).starts_with(dir))
    {
        return Err(refuse(format!(
            "{name} is running from {}; restaging would delete the live install (roll back or promote another version first)",
            dir.display()
        )));
    }
    let rows = settle(state, name).await?;
    let rollback_target = rows
        .iter()
        .rev()
        .find(|r| r.status == "promoted")
        .and_then(|r| {
            r.detail.as_ref().and_then(|d| {
                d.get("old_executable")
                    .or_else(|| d.get("old_interpreter"))?
                    .as_str()
                    .map(String::from)
            })
        });
    if rollback_target
        .as_deref()
        .is_some_and(|p| Path::new(p).starts_with(dir))
    {
        return Err(refuse(format!(
            "{} is the rollback target of {name}'s last promote; restaging would delete it",
            dir.display()
        )));
    }
    Ok(())
}

/// Stage a new version for `workload` per its recipe kind, then validate
/// it offline. An interrupted stage leaves no manifest and is never
/// promotable.
pub async fn stage(
    state: &AppState,
    name: &str,
    version: &str,
) -> Result<StageOutcome, UpgradeError> {
    let (wl, _definition, recipe) = managed_parts(state, name).await?;
    match &recipe {
        InstallRecipe::PythonEnv(py) => stage_python_env(state, name, &wl, py, version).await,
        InstallRecipe::ReleaseArchive(r) => {
            stage_release_archive(state, name, &wl, &recipe, r, version).await
        }
        InstallRecipe::SourceBuild(r) => {
            stage_source_build(state, name, &wl, &recipe, r, version).await
        }
        InstallRecipe::VendorApp(_) => Err(refuse(format!(
            "{name} is a vendor app: llmctl never installs or upgrades it. \
             `llmctl-next install plan {name}` prints the observed/pinned \
             version and the owner-run backup/reinstall path"
        ))),
        InstallRecipe::External(_) => Err(refuse(format!(
            "{name} has an external recipe: nothing to install"
        ))),
    }
}

/// Stage a new python environment for `workload` at `version`.
async fn stage_python_env(
    state: &AppState,
    name: &str,
    wl: &WorkloadName,
    recipe: &PythonEnvRecipe,
    version: &str,
) -> Result<StageOutcome, UpgradeError> {
    let full_recipe = InstallRecipe::PythonEnv(recipe.clone());
    let root = envs_root(state);
    let id = env_id(name, version);
    let env = root.join(&id);
    let log = root.join(format!("{id}.log"));
    let interpreter = env.join("bin").join("python");

    // Free disk first: an env that cannot fit is refused before anything
    // is written.
    tokio::fs::create_dir_all(&root).await?;
    let free = free_bytes(&root)?;
    if free < MIN_FREE_BYTES {
        return Err(refuse(format!(
            "only {} GiB free under {}; a staged env needs ~{} GiB",
            free / (1 << 30),
            root.display(),
            MIN_FREE_BYTES / (1 << 30)
        )));
    }

    refuse_restage_of_live_or_rollback_dir(state, name, wl, &full_recipe, &env).await?;

    let _ = tokio::fs::remove_file(env.join("staged.json")).await;
    let _ = tokio::fs::remove_dir_all(&env).await;
    let _ = tokio::fs::remove_file(&log).await;

    // A stage marker so an interrupted attempt is identifiable garbage.
    // `with_extension` would eat the version suffix (`engine2-0.32.0-<h>`
    // → `engine2-0.32.staging`), colliding across patch versions.
    let staging = root.join(format!("{id}.staging"));
    tokio::fs::write(&staging, b"interrupted stage; not promotable\n").await?;

    // Same Python minor version as the env being replaced: uv would
    // otherwise pick whatever interpreter it finds first.
    let mut venv = Command::new("uv");
    venv.arg("venv");
    if let Some(old) =
        crate::version::primary_executable_for(&full_recipe, &inv_def_for(state, wl).await)
    {
        if let Ok(minor) = crate::version::python_minor(&old).await {
            venv.arg("--python").arg(minor);
        }
    }
    run_logged(venv.arg(&env), &log, "uv venv").await?;
    let mut install = Command::new("uv");
    install
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&interpreter)
        .arg(install_spec(&recipe.name, version));
    if let Some(links) = &recipe.find_links {
        install.arg("--find-links").arg(links);
    }
    if let Some(index) = &recipe.index_url {
        install.arg("--index-url").arg(index);
    }
    run_logged(&mut install, &log, "uv pip install").await?;

    // Offline validation before anything is promotable (plan D.4).
    let inv = state.ops.inventory().await;
    let wl_def = inv.workload(wl).cloned();
    let model = wl_def
        .as_ref()
        .and_then(|d| {
            d.recipe.as_ref().and_then(|r| match r {
                InstallRecipe::PythonEnv(py) => py.model_config.clone(),
                _ => None,
            })
        })
        .or_else(|| wl_def.as_ref().map(|d| d.model.clone()))
        .unwrap_or_default();
    let validation = match crate::version::validate_offline(
        &full_recipe,
        &interpreter.display().to_string(),
        version,
        &model,
    )
    .await
    {
        Ok(report) => report,
        Err(e) => {
            let _ = tokio::fs::remove_file(&staging).await;
            record_row(
                state,
                name,
                version,
                &env,
                &interpreter,
                "failed",
                Some(json!({"error": e})),
            )
            .await?;
            return Err(refuse(format!(
                "validation of {version} failed; never promoted: {e}"
            )));
        }
    };

    // The manifest is written LAST: its presence is what makes an env
    // promotable.
    let manifest = json!({
        "workload": name,
        "version": version,
        "kind": "python_env",
        "interpreter": interpreter.display().to_string(),
        "executable": interpreter.display().to_string(),
        "validated_at": Utc::now().to_rfc3339(),
        "validation": validation,
        "spec": install_spec(&recipe.name, version),
    });
    tokio::fs::write(
        env.join("staged.json"),
        serde_json::to_vec_pretty(&manifest)
            .map_err(|e| UpgradeError::Io(format!("manifest: {e}")))?,
    )
    .await?;
    let _ = tokio::fs::remove_file(&staging).await;

    let row = record_row(
        state,
        name,
        version,
        &env,
        &interpreter,
        "validated",
        Some(validation.clone()),
    )
    .await?;
    Ok(StageOutcome { row, validation })
}

/// Stage a release archive: fetch (or copy), verify the checksum when one
/// is declared, unpack into a staging dir, check the binary, then promote
/// = rename into `<root>/<id>/` with the manifest written last.
async fn stage_release_archive(
    state: &AppState,
    name: &str,
    wl: &WorkloadName,
    recipe: &InstallRecipe,
    r: &ReleaseArchiveRecipe,
    version: &str,
) -> Result<StageOutcome, UpgradeError> {
    let root = PathBuf::from(&r.root);
    let id = env_id(name, version);
    let dir = root.join(&id);
    let log = root.join(format!("{id}.log"));
    let staging = root.join(format!("{id}.staging"));
    let work = root.join(format!("{id}.work"));
    let archive = root.join(format!("{id}.archive"));

    tokio::fs::create_dir_all(&root).await?;
    let free = free_bytes(&root)?;
    if free < MIN_FREE_DIR_BYTES {
        return Err(refuse(format!(
            "only {} GiB free under {}; a staged archive needs ~{} GiB",
            free / (1 << 30),
            root.display(),
            MIN_FREE_DIR_BYTES / (1 << 30)
        )));
    }
    refuse_restage_of_live_or_rollback_dir(state, name, wl, recipe, &dir).await?;

    let _ = tokio::fs::remove_file(dir.join("staged.json")).await;
    let _ = tokio::fs::remove_dir_all(&dir).await;
    let _ = tokio::fs::remove_file(&log).await;
    tokio::fs::write(&staging, b"interrupted stage; not promotable\n").await?;

    // Fetch: http(s) via curl, anything else is a local path.
    let url = r.url.replace("{version}", version);
    if url.starts_with("http://") || url.starts_with("https://") {
        let mut curl = Command::new("curl");
        curl.args(["-fL", "--retry", "3", "-o"]).arg(&archive).arg(&url);
        run_logged_with(&mut curl, &log, "curl", Duration::from_secs(1800)).await?;
    } else {
        tokio::fs::copy(&url, &archive)
            .await
            .map_err(|e| UpgradeError::Io(format!("copying {url}: {e}")))?;
    }

    // Checksum: recorded always, verified when the recipe declares one.
    // Streamed — llama.cpp archives run to gigabytes.
    let digest = hash_file(&archive).await?;
    if let Some(expected) = &r.checksum {
        let expected = expected.replace("{version}", version);
        if expected != digest {
            let _ = tokio::fs::remove_file(&staging).await;
            record_row(
                state,
                name,
                version,
                &dir,
                Path::new(""),
                "failed",
                Some(json!({"error": format!("checksum mismatch: archive is {digest}, recipe expects {expected}")})),
            )
            .await?;
            return Err(refuse(format!(
                "archive checksum mismatch: got {digest}, expected {expected}"
            )));
        }
    }

    // Unpack into the work dir.
    let _ = tokio::fs::remove_dir_all(&work).await;
    tokio::fs::create_dir_all(&work).await?;
    if url.ends_with(".zip") {
        let mut unzip = Command::new("unzip");
        unzip.arg("-q").arg(&archive).arg("-d").arg(&work);
        run_logged_with(&mut unzip, &log, "unzip", Duration::from_secs(600)).await?;
    } else {
        let mut tar = Command::new("tar");
        tar.arg("-xf").arg(&archive).arg("-C").arg(&work);
        if r.strip_components > 0 {
            tar.arg("--strip-components").arg(r.strip_components.to_string());
        }
        run_logged_with(&mut tar, &log, "tar", Duration::from_secs(600)).await?;
    }

    // Binary check + offline validation.
    let exe = work.join(&r.binary);
    finish_dir_stage(
        state, name, version, recipe, &root, &dir, &work, &staging, &log, &exe, &json!({
            "kind": "release_archive",
            "url": url,
            "archive_sha256": digest,
        }),
    )
    .await
}

/// Stage a source build: clone at the recipe's ref (default: the version
/// itself), run the build command in the checkout, check the binary, then
/// promote = rename. The git SHA is recorded as the version's identity —
/// the whole point for ds4, whose binary has no `--version`.
async fn stage_source_build(
    state: &AppState,
    name: &str,
    wl: &WorkloadName,
    recipe: &InstallRecipe,
    r: &SourceBuildRecipe,
    version: &str,
) -> Result<StageOutcome, UpgradeError> {
    let root = PathBuf::from(&r.root);
    let id = env_id(name, version);
    let dir = root.join(&id);
    let log = root.join(format!("{id}.log"));
    let staging = root.join(format!("{id}.staging"));
    let work = root.join(format!("{id}.work"));

    tokio::fs::create_dir_all(&root).await?;
    let free = free_bytes(&root)?;
    if free < MIN_FREE_DIR_BYTES {
        return Err(refuse(format!(
            "only {} GiB free under {}; a staged build needs ~{} GiB",
            free / (1 << 30),
            root.display(),
            MIN_FREE_DIR_BYTES / (1 << 30)
        )));
    }
    refuse_restage_of_live_or_rollback_dir(state, name, wl, recipe, &dir).await?;

    let _ = tokio::fs::remove_file(dir.join("staged.json")).await;
    let _ = tokio::fs::remove_dir_all(&dir).await;
    let _ = tokio::fs::remove_dir_all(&work).await;
    let _ = tokio::fs::remove_file(&log).await;
    tokio::fs::write(&staging, b"interrupted stage; not promotable\n").await?;

    let git_ref = r.git_ref.clone().unwrap_or_else(|| version.to_string());
    let mut clone = Command::new("git");
    clone.arg("clone").arg(&r.git).arg(&work);
    run_logged_with(&mut clone, &log, "git clone", Duration::from_secs(1800)).await?;
    let mut checkout = Command::new("git");
    checkout
        .arg("-C")
        .arg(&work)
        .arg("checkout")
        .arg(&git_ref);
    run_logged_with(&mut checkout, &log, "git checkout", Duration::from_secs(300)).await?;
    let commit = {
        let mut rev = Command::new("git");
        rev.arg("-C").arg(&work).arg("rev-parse").arg("HEAD");
        let out = run_logged_with(&mut rev, &log, "git rev-parse", Duration::from_secs(60)).await?;
        out.trim().to_string()
    };

    // The build runs in the checkout; its output is the build log.
    let mut build = Command::new("sh");
    build.arg("-c").arg(&r.build).current_dir(&work);
    run_logged_with(&mut build, &log, "build", Duration::from_secs(3600)).await?;

    let exe = work.join(&r.binary);
    finish_dir_stage(
        state, name, version, recipe, &root, &dir, &work, &staging, &log, &exe, &json!({
            "kind": "source_build",
            "git": r.git,
            "ref": git_ref,
            "commit": commit,
        }),
    )
    .await
}

/// The shared tail of a versioned-directory stage: validate the binary,
/// write the manifest inside the work dir (LAST — its presence is what
/// makes the dir promotable), promote = rename work → dir, record the
/// row.
#[allow(clippy::too_many_arguments)]
async fn finish_dir_stage(
    state: &AppState,
    name: &str,
    version: &str,
    recipe: &InstallRecipe,
    _root: &Path,
    dir: &Path,
    work: &Path,
    staging: &Path,
    log: &Path,
    exe: &Path,
    provenance: &serde_json::Value,
) -> Result<StageOutcome, UpgradeError> {
    let fail = |e: String| refuse(format!("validation of {version} failed; never promoted: {e}"));
    let validation = match crate::version::validate_offline(recipe, &exe.display().to_string(), version, "")
        .await
    {
        Ok(report) => report,
        Err(e) => {
            let _ = tokio::fs::remove_file(staging).await;
            let _ = tokio::fs::remove_dir_all(work).await;
            record_row(
                state,
                name,
                version,
                dir,
                Path::new(""),
                "failed",
                Some(json!({"error": e})),
            )
            .await?;
            return Err(fail(e));
        }
    };
    let _ = log; // the log is appended to by the callers' run_logged calls

    let mut manifest = provenance.clone();
    if let Some(obj) = manifest.as_object_mut() {
        obj.insert("workload".into(), json!(name));
        obj.insert("version".into(), json!(version));
        obj.insert(
            "executable".into(),
            json!(dir.join(exe.strip_prefix(work).unwrap_or(exe))
                .display()
                .to_string()),
        );
        obj.insert("validated_at".into(), json!(Utc::now().to_rfc3339()));
        obj.insert("validation".into(), validation.clone());
    }
    tokio::fs::write(
        work.join("staged.json"),
        serde_json::to_vec_pretty(&manifest)
            .map_err(|e| UpgradeError::Io(format!("manifest: {e}")))?,
    )
    .await?;
    tokio::fs::rename(work, dir)
        .await
        .map_err(|e| UpgradeError::Io(format!("promoting {} → {}: {e}", work.display(), dir.display())))?;
    let _ = tokio::fs::remove_file(staging).await;

    let final_exe = manifest["executable"].as_str().unwrap_or_default();
    let row = record_row(
        state,
        name,
        version,
        dir,
        Path::new(final_exe),
        "validated",
        Some(validation.clone()),
    )
    .await?;
    Ok(StageOutcome { row, validation })
}

async fn record_row(
    state: &AppState,
    workload: &str,
    to_version: &str,
    env: &Path,
    interpreter: &Path,
    status: &str,
    detail: Option<serde_json::Value>,
) -> Result<UpgradeRow, UpgradeError> {
    let from_version = pinned_version_of(state, workload)
        .await
        .unwrap_or_else(|| "?".into());
    let now = Utc::now();
    let env_name = env
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let row = UpgradeRow {
        // One row per stage attempt: restaging a version adds a row and
        // keeps the earlier promote/rollback record (M2 pilot gap).
        id: format!("{env_name}-{}", unique_suffix()),
        workload: workload.to_string(),
        from_version,
        to_version: to_version.to_string(),
        env_path: env.display().to_string(),
        interpreter: interpreter.display().to_string(),
        status: status.to_string(),
        detail,
        operation_id: None,
        created_at: now,
        updated_at: now,
    };
    // An earlier, never-promoted stage of this env was just rebuilt over:
    // its row no longer describes what is on disk.
    for old in state.ops.db().list_upgrades(workload).await? {
        if old.env_path == row.env_path && old.status == "validated" {
            state
                .ops
                .db()
                .set_upgrade_status_if(&old.id, "validated", "superseded", None)
                .await?;
        }
    }
    state.ops.db().insert_upgrade(row.clone()).await?;
    Ok(row)
}

/// Bring rows whose operation has finished to their final status, then
/// return the history. Done on every read, so a caller that waited for
/// the operation (`promote --wait`) sees the outcome at once — a background
/// poller lagged up to 2 s behind it (M2 pilot gap).
async fn settle(state: &AppState, name: &str) -> Result<Vec<UpgradeRow>, UpgradeError> {
    let rows = state.ops.db().list_upgrades(name).await?;
    let mut changed = false;
    for r in &rows {
        let (op_id, done, failed) = match r.status.as_str() {
            "promoting" => (r.operation_id.clone(), "promoted", "failed"),
            "rolling_back" => (
                r.detail
                    .as_ref()
                    .and_then(|d| d.get("rollback_operation_id")?.as_str().map(String::from)),
                "rolled_back",
                // A rollback operation that failed *but restored the
                // world* leaves the promote standing: the row returns to
                // `promoted` with its N stash intact, so the rollback is
                // retryable instead of burning the recovery target. Only
                // a failed restoration (world left different) fails the
                // row — a human decides from the needs_attention op.
                "restored-promoted-or-failed",
            ),
            _ => continue,
        };
        let Some(op_id) = op_id else { continue };
        let Some(op) = state.ops.db().get_op(&op_id).await? else {
            continue;
        };
        if !op.status.is_terminal() {
            continue;
        }
        let to = if op.status == llmctl_client::wire::OpStatus::Succeeded {
            done.to_string()
        } else if failed == "restored-promoted-or-failed" {
            let restored = op
                .recovery
                .as_ref()
                .and_then(|rec| rec.get("outcome"))
                .and_then(|v| v.as_str())
                .is_some_and(|o| o == "restored");
            if restored {
                "promoted".to_string()
            } else {
                "failed".to_string()
            }
        } else {
            failed.to_string()
        };
        changed |= state
            .ops
            .db()
            .set_upgrade_status_if(&r.id, &r.status, &to, None)
            .await?;
    }
    if changed {
        Ok(state.ops.db().list_upgrades(name).await?)
    } else {
        Ok(rows)
    }
}

/// `detail` with one key added (detail is replaced as a whole on write).
fn detail_with(detail: &Option<serde_json::Value>, key: &str, value: &str) -> serde_json::Value {
    let mut d = detail.clone().unwrap_or_else(|| json!({}));
    if let Some(obj) = d.as_object_mut() {
        obj.insert(key.into(), json!(value));
    }
    d
}

async fn pinned_version_of(state: &AppState, workload: &str) -> Option<String> {
    let inv = state.ops.inventory().await;
    let def = inv.workload(&WorkloadName::new(workload).ok()?)?;
    let Ownership::Managed { version, .. } = &def.ownership else {
        return None;
    };
    // The recipe's first package version is "the version" for python-env
    // recipes; dir recipes carry their identity in the versioned dir, not
    // in a package pin.
    match def.recipe.as_ref()? {
        InstallRecipe::PythonEnv(_) => version
            .as_ref()
            .and_then(|v| v.packages.first().map(|(_, v)| v.clone())),
        _ => None,
    }
}

async fn inv_def_for(state: &AppState, wl: &WorkloadName) -> llmctl_core::WorkloadDefinition {
    state
        .ops
        .inventory()
        .await
        .workload(wl)
        .cloned()
        .expect("workload vanished mid-upgrade")
}

/// Promote a validated stage as an admitted operation (plan D.5).
pub async fn promote(
    state: &AppState,
    name: &str,
    version: &str,
) -> Result<llmctl_client::wire::OperationView, UpgradeError> {
    let (wl, old_def, recipe) = managed_parts(state, name).await?;
    let Ownership::Managed {
        service_file: Some(_),
        ..
    } = state
        .ops
        .inventory()
        .await
        .workload(&wl)
        .map(|d| d.ownership.clone())
        .unwrap_or(Ownership::External)
    else {
        return Err(refuse(format!(
            "{name} has no owned service file; `own` it before promoting"
        )));
    };
    if matches!(recipe, InstallRecipe::VendorApp(_) | InstallRecipe::External(_)) {
        return Err(refuse(format!(
            "{name} has a {} recipe: there is nothing llmctl staged to promote",
            recipe_kind_name(&recipe)
        )));
    }

    // The stage must be validated: manifest present, row validated.
    let root = installs_root(state, &recipe);
    let env = root.join(env_id(name, version));
    let manifest_path = env.join("staged.json");
    let manifest: serde_json::Value = match tokio::fs::read(&manifest_path).await {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|e| UpgradeError::Io(format!("staged.json: {e}")))?,
        Err(_) => {
            return Err(refuse(format!(
                "no validated stage for {name} at {version}; an interrupted or failed stage is never promotable"
            )))
        }
    };
    let rows = settle(state, name).await?;
    let Some(row) = rows.iter().rev().find(|r| r.to_version == version) else {
        return Err(refuse(format!("no upgrade record for {name} at {version}")));
    };
    if row.status != "validated" {
        return Err(refuse(format!(
            "stage for {version} is {} (not validated); restage it",
            row.status
        )));
    }
    let executable = manifest["executable"]
        .as_str()
        .or_else(|| manifest["interpreter"].as_str())
        .unwrap_or_default()
        .to_string();
    if executable.is_empty() {
        return Err(refuse("staged manifest names no executable"));
    }

    // Expected versions after promote: python packages at `version`, or
    // the binary's own version output for dir recipes that declare one.
    let (expected, binary_check) = expected_version_payload(&recipe, &executable, version);

    let next_def = crate::version::rewrite_definition(&old_def, &recipe, &executable)
        .map_err(refuse)?;
    if let InstallRecipe::PythonEnv(py) = &recipe {
        if py.apply != "exec" {
            // The override env file next_def points at must exist with
            // exactly the pinned content before the definition is applied.
            let prefix = match py.env_prefix.clone() {
                Some(p) => p,
                None if py.name == "vllm" => "VLLM".into(),
                None => {
                    return Err(refuse(format!(
                        "recipe {} applies env but names no env_prefix",
                        py.name
                    )))
                }
            };
            let (path, content) = crate::version::override_env(&executable, &prefix);
            tokio::fs::write(&path, content.as_bytes()).await?;
        }
    }
    // N (the definition and install being replaced) is the rollback
    // target: stash it in the row's detail.
    let old_executable =
        crate::version::primary_executable_for(&recipe, &inv_def_for(state, &wl).await)
            .unwrap_or_default();
    // The version being replaced is the one pinned now, not the one pinned
    // when this env was staged: another promote may have run in between.
    // Dir recipes have no package pin — their version is the dir's id.
    let from_version = pinned_version_of(state, name)
        .await
        .or_else(|| version_of_dir(name, &old_executable))
        .unwrap_or_else(|| row.from_version.clone());
    // What the old binary itself prints (dir recipes with a version
    // argument): stashed so a rollback to a pre-convention install still
    // has a fail-closed expect string (its dir name encodes no version).
    let old_version_output = match (&recipe, old_version_arg(&recipe)) {
        (InstallRecipe::ReleaseArchive(_) | InstallRecipe::SourceBuild(_), Some(arg)) => {
            crate::version::run_binary_version(&old_executable, arg)
                .await
                .ok()
        }
        _ => None,
    };
    state
        .ops
        .db()
        .update_upgrade_status(
            &row.id,
            "promoting",
            Some(json!({
                "old_definition": old_def,
                "old_executable": old_executable,
                "old_interpreter": old_executable,
                "from_version": from_version,
                "old_version_output": old_version_output,
            })),
            None,
        )
        .await?;

    // The plan is built against fresh observations; admission re-checks
    // blockers, drift and GPU occupancy under its lock before journaling.
    let inv = state.ops.inventory().await;
    let observations = state
        .ops
        .engine()
        .observe(&inv)
        .await
        .map_err(|e| OpsError::Engine(e.to_string()))?;
    let plan = llmctl_core::plan_upgrade(
        &inv,
        &observations,
        &wl,
        &old_def,
        &next_def,
        &executable,
        &expected,
        binary_check,
    )
    .map_err(|e| OpsError::Invalid(e.to_string()))?;

    let admitted = state
        .ops
        .admit(AdmitParams {
            protocol: PROTOCOL_VERSION,
            idempotency_key: format!("upgrade-{name}-{version}-{}", unique_suffix()),
            kind: OpKind::Upgrade,
            workload: name.to_string(),
            expected_revision: None,
            expected_stop_targets: None,
            plan_override: Some(plan),
        })
        .await;
    match admitted {
        Ok(AdmitOutcome::Admitted(row_op)) | Ok(AdmitOutcome::Replayed(row_op)) => {
            // The outcome is settled from this operation on the next read.
            // Detail stays: it holds the N stash for rollback.
            state
                .ops
                .db()
                .update_upgrade_status(&row.id, "promoting", None, Some(row_op.id.to_string()))
                .await?;
            Ok(crate::api::operation_view(&row_op))
        }
        Err(e) => {
            // A refused admission journaled nothing and changed no state —
            // the stage stays validated and promotable (the pilot's window-1
            // GPU refusal used to poison the row into `failed`, forcing a
            // pointless full restage). The outcome of an *admitted*
            // operation is settled by `settle` from the journal.
            Err(e.into())
        }
    }
}

/// What the `VerifyVersion` step of a promote checks: python packages at
/// the target version, and/or the staged binary printing its version.
fn expected_version_payload(
    recipe: &InstallRecipe,
    executable: &str,
    version: &str,
) -> (
    Vec<(String, String)>,
    Option<llmctl_core::BinaryCheck>,
) {
    let packages = match recipe {
        InstallRecipe::PythonEnv(py) => py
            .packages
            .iter()
            .map(|p| (p.clone(), version.to_string()))
            .collect(),
        _ => Vec::new(),
    };
    let (version_arg, version_pattern) = match recipe {
        InstallRecipe::ReleaseArchive(r) => {
            (r.version_arg.clone(), r.version_pattern.clone())
        }
        InstallRecipe::SourceBuild(r) => (r.version_arg.clone(), r.version_pattern.clone()),
        _ => (None, None),
    };
    let binary_check = version_arg.map(|arg| llmctl_core::BinaryCheck {
        path: executable.to_string(),
        arg,
        expect: version_pattern.unwrap_or_else(|| version.to_string()),
    });
    (packages, binary_check)
}

fn recipe_kind_name(recipe: &InstallRecipe) -> &'static str {
    match recipe {
        InstallRecipe::PythonEnv(_) => "python-env",
        InstallRecipe::ReleaseArchive(_) => "release-archive",
        InstallRecipe::SourceBuild(_) => "source-build",
        InstallRecipe::VendorApp(_) => "vendor-app",
        InstallRecipe::External(_) => "external",
    }
}

/// The version argument a dir recipe's binary answers, if it has one.
fn old_version_arg(recipe: &InstallRecipe) -> Option<&str> {
    match recipe {
        InstallRecipe::ReleaseArchive(r) => r.version_arg.as_deref(),
        InstallRecipe::SourceBuild(r) => r.version_arg.as_deref(),
        _ => None,
    }
}

/// The version an install dir's id names, when the executable lives in a
/// conventionally named `<root>/<workload>-<version>-<hash8>/` dir. The
/// executable may sit deeper inside (e.g. `<id>/bin/server`), so every
/// ancestor is tried.
fn version_of_dir(workload: &str, executable: &str) -> Option<String> {
    let mut dir = Path::new(executable).parent()?;
    loop {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Some(id) = owned_env_id(workload, &name) {
            return id
                .strip_prefix(&format!("{workload}-"))?
                .rsplit_once('-')
                .map(|(version, _)| version.to_string());
        }
        dir = dir.parent()?;
    }
}

/// Explicit rollback to N: the definition and env stashed when the last
/// promoted upgrade ran. An upgrade operation in its own right; the old
/// env is never touched by a promote, so it is still there.
pub async fn rollback(
    state: &AppState,
    name: &str,
) -> Result<llmctl_client::wire::OperationView, UpgradeError> {
    let (wl, current_def, recipe) = managed_parts(state, name).await?;
    let rows = settle(state, name).await?;
    let Some(last) = rows.iter().rev().find(|r| r.status == "promoted") else {
        return Err(refuse(format!(
            "no promoted upgrade to roll {name} back to"
        )));
    };
    let Some(detail) = &last.detail else {
        return Err(refuse("the promoted upgrade kept no rollback target"));
    };
    let old_def: LaunchDefinition = serde_json::from_value(
        detail
            .get("old_definition")
            .cloned()
            .ok_or_else(|| refuse("rollback target definition missing"))?,
    )
    .map_err(|e| UpgradeError::Io(format!("rollback definition: {e}")))?;
    // `old_executable` (M4a) supersedes `old_interpreter` (pre-M4a rows);
    // both name N's primary executable.
    let old_interpreter = detail
        .get("old_executable")
        .or_else(|| detail.get("old_interpreter"))
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let from_version = detail
        .get("from_version")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    // What N's binary printed when the promote stashed it (dir recipes
    // with a version argument): the fail-closed expect for rolling back
    // to a pre-convention install.
    let old_version_output = detail
        .get("old_version_output")
        .and_then(|v| v.as_str())
        .map(String::from);
    if old_interpreter.is_empty() {
        return Err(refuse(
            "the pre-upgrade executable was never pinned",
        ));
    }

    // Claim the row; a concurrent rollback of the same promote loses here.
    if !state
        .ops
        .db()
        .set_upgrade_status_if(&last.id, "promoted", "rolling_back", None)
        .await?
    {
        return Err(refuse(format!(
            "the last promote of {name} is already being rolled back"
        )));
    }
    let admitted = rollback_admit(
        state,
        name,
        &wl,
        &current_def,
        &old_def,
        &old_interpreter,
        &recipe,
        &from_version,
        old_version_output.as_deref(),
        &last.id,
    )
    .await;
    match admitted {
        Ok(row_op) => {
            // Settled from this operation on the next read.
            state
                .ops
                .db()
                .set_upgrade_status_if(
                    &last.id,
                    "rolling_back",
                    "rolling_back",
                    Some(detail_with(
                        &last.detail,
                        "rollback_operation_id",
                        &row_op.id.to_string(),
                    )),
                )
                .await?;
            Ok(crate::api::operation_view(&row_op))
        }
        Err(e) => {
            // A refused rollback changed nothing: the promote still stands.
            state
                .ops
                .db()
                .set_upgrade_status_if(&last.id, "rolling_back", "promoted", None)
                .await?;
            Err(e)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn rollback_admit(
    state: &AppState,
    name: &str,
    wl: &WorkloadName,
    current_def: &LaunchDefinition,
    old_def: &LaunchDefinition,
    old_interpreter: &str,
    recipe: &InstallRecipe,
    from_version: &str,
    old_version_output: Option<&str>,
    row_id: &str,
) -> Result<crate::db::OpRow, UpgradeError> {
    // The expect strings for the restored N. A derivable from_version
    // yields the usual payload; an underivable one (a pre-convention
    // install dir such as llama.cpp's live `build/`) falls back to the
    // old binary's stashed `--version` output — recorded by the promote —
    // so the rollback stays fail-closed. With neither (a recipe whose
    // binary has no version argument, e.g. ds4), there is no binary check
    // by design; the ApplyDefinition swap and the served-model verify are
    // the restoration proof.
    let (expected, binary_check) = if from_version != "?" {
        expected_version_payload(recipe, old_interpreter, from_version)
    } else {
        let binary_check = old_version_output.and_then(|out| {
            old_version_arg(recipe).map(|arg| llmctl_core::BinaryCheck {
                path: old_interpreter.to_string(),
                arg: arg.to_string(),
                expect: out.to_string(),
            })
        });
        (Vec::new(), binary_check)
    };
    let inv = state.ops.inventory().await;
    let observations = state
        .ops
        .engine()
        .observe(&inv)
        .await
        .map_err(|e| OpsError::Engine(e.to_string()))?;
    // Rolling back applies N over the *current* definition (N+1, whether
    // the failed promote left it or a succeeded one did).
    let plan = llmctl_core::plan_upgrade(
        &inv,
        &observations,
        wl,
        current_def,
        old_def,
        old_interpreter,
        &expected,
        binary_check,
    )
    .map_err(|e| OpsError::Invalid(e.to_string()))?;
    let admitted = state
        .ops
        .admit(AdmitParams {
            protocol: PROTOCOL_VERSION,
            idempotency_key: format!("upgrade-rollback-{name}-{row_id}-{}", unique_suffix()),
            kind: OpKind::Upgrade,
            workload: name.to_string(),
            expected_revision: None,
            expected_stop_targets: None,
            plan_override: Some(plan),
        })
        .await;
    match admitted {
        Ok(AdmitOutcome::Admitted(row_op)) | Ok(AdmitOutcome::Replayed(row_op)) => Ok(row_op),
        Err(e) => Err(e.into()),
    }
}

pub async fn list(state: &AppState, name: &str) -> Result<Vec<UpgradeRow>, UpgradeError> {
    settle(state, name).await
}

/// The vendor-app plan (M4a step 4): the observed running version, the
/// pin, the drift between them, and the owner-run backup/reinstall path
/// recorded in the recipe. llmctl never executes any of it — a vendor
/// install is an explicit, owner-run step (owner decision 1), and no
/// silent vendor auto-update counts as an upgrade.
pub async fn vendor_plan(
    state: &AppState,
    name: &str,
) -> Result<serde_json::Value, UpgradeError> {
    let inv = state.ops.inventory().await;
    let wl = WorkloadName::new(name).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let Some(def) = inv.workload(&wl) else {
        return Err(OpsError::UnknownWorkload(name.to_string()).into());
    };
    let Some(InstallRecipe::VendorApp(r)) = &def.recipe else {
        return Err(refuse(format!(
            "{name} has no vendor-app recipe (it is {})",
            def.recipe
                .as_ref()
                .map(recipe_kind_name)
                .unwrap_or("unreciped")
        )));
    };

    // The live fact: what the running app reports right now.
    let live = crate::version::probe(&InstallRecipe::VendorApp(r.clone()), def)
        .await
        .map_err(|e| UpgradeError::Io(format!("probing {name}: {e}")))?;
    let pinned = match &def.ownership {
        Ownership::Managed { version, .. } => version.clone(),
        _ => None,
    };
    let drift = match (&pinned, &live) {
        (Some(p), _) => crate::version::version_drift(p, &live),
        (None, _) => Vec::new(),
    };
    Ok(json!({
        "workload": name,
        "kind": "vendor_app",
        "probe": match &r.probe {
            llmctl_core::VendorProbe::HttpJson { path, .. } =>
                format!("http_json GET 127.0.0.1:{}{path}", def.port),
            llmctl_core::VendorProbe::Command { argv } => argv.join(" "),
            llmctl_core::VendorProbe::DirName { path } => format!("dir_name {path}"),
        },
        "running": live
            .fact
            .as_ref()
            .and_then(|f| match f {
                llmctl_core::VersionFact::VendorApp { running, .. } => running.clone(),
                _ => None,
            })
            .unwrap_or_else(|| "unobservable".into()),
        "pinned": pinned,
        "drift": drift,
        "owner_steps": {
            "backup": r.backup_path,
            "reinstall": r.reinstall_cmd,
            "note": "llmctl records and pins the version and owns the service; \
                     installing or upgrading the app itself is an owner-run step. \
                     After the vendor upgrade, re-probe (`version <name>`) and \
                     re-pin by adopting/owning again if the definition changed.",
        },
    }))
}

/// Statuses whose install may still be used: promotable, live, a rollback
/// target down the promote chain, or mid-operation.
const ENV_IN_USE: [&str; 4] = ["validated", "promoted", "promoting", "rolling_back"];

/// The install id an entry under the installs root belongs to, if it is
/// `name`'s: `{name}-{version}-{hash8}` (plus `.staging`/`.log`/`.archive`
/// for the markers), with the hash recomputed so `engine` never claims
/// `engine2`'s installs — the root is shared by every workload.
fn owned_env_id(name: &str, file_name: &str) -> Option<String> {
    let id = file_name
        .strip_suffix(".staging")
        .or_else(|| file_name.strip_suffix(".log"))
        .or_else(|| file_name.strip_suffix(".archive"))
        .unwrap_or(file_name);
    let (version, _hash) = id.strip_prefix(&format!("{name}-"))?.rsplit_once('-')?;
    (env_id(name, version) == id).then(|| id.to_string())
}

/// Remove `name`'s installs that nothing can use any more: rolled back,
/// failed, superseded, or left by an interrupted stage. Kept: the live
/// install, every install a row in [`ENV_IN_USE`] points at (its own and
/// its rollback target), and anything that is not this workload's. Rows
/// whose install is gone become `discarded`.
pub async fn gc(state: &AppState, name: &str) -> Result<serde_json::Value, UpgradeError> {
    let (wl, _definition, recipe) = managed_parts(state, name).await?;
    let def = inv_def_for(state, &wl).await;
    let rows = settle(state, name).await?;

    // Executable paths whose install dir must survive; a dir is kept when
    // one of them lies inside it.
    let mut in_use: Vec<String> = Vec::new();
    in_use.extend(crate::version::primary_executable_for(&recipe, &def));
    if let Ownership::Managed { version, .. } = &def.ownership {
        in_use.extend(version.as_ref().and_then(|v| v.interpreter.clone()));
    }
    for r in rows
        .iter()
        .filter(|r| ENV_IN_USE.contains(&r.status.as_str()))
    {
        in_use.push(r.interpreter.clone());
        in_use.push(r.env_path.clone());
        in_use.extend(
            r.detail.as_ref().and_then(|d| {
                d.get("old_executable")
                    .or_else(|| d.get("old_interpreter"))?
                    .as_str()
                    .map(String::from)
            }),
        );
    }
    let used = |env: &Path| in_use.iter().any(|p| Path::new(p).starts_with(env));

    let root = installs_root(state, &recipe);
    let mut removed = Vec::new();
    let mut kept = std::collections::BTreeSet::new();
    let mut entries = match tokio::fs::read_dir(&root).await {
        Ok(e) => Some(e),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    while let Some(entry) = match entries.as_mut() {
        Some(e) => e.next_entry().await?,
        None => None,
    } {
        let file_name = entry.file_name().to_string_lossy().into_owned();
        let Some(id) = owned_env_id(name, &file_name) else {
            continue;
        };
        let env = root.join(&id);
        let path = entry.path();
        if used(&env) {
            if path == env {
                kept.insert(path.display().to_string());
            }
            continue;
        }
        if entry.file_type().await?.is_dir() {
            tokio::fs::remove_dir_all(&path).await?;
        } else {
            tokio::fs::remove_file(&path).await?;
        }
        removed.push(path.display().to_string());
    }

    // History stays; a row whose env no longer exists says so.
    for r in rows.iter().filter(|r| r.status != "discarded") {
        if !ENV_IN_USE.contains(&r.status.as_str())
            && !Path::new(&r.env_path).exists()
            && owned_env_id(
                name,
                &Path::new(&r.env_path)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            )
            .is_some()
        {
            state
                .ops
                .db()
                .set_upgrade_status_if(&r.id, &r.status, "discarded", None)
                .await?;
        }
    }
    Ok(json!({"removed": removed, "kept": kept.into_iter().collect::<Vec<_>>()}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_env_id_claims_only_this_workloads_entries() {
        let id = env_id("engine", "0.32.0");
        assert_eq!(owned_env_id("engine", &id), Some(id.clone()));
        assert_eq!(
            owned_env_id("engine", &format!("{id}.staging")),
            Some(id.clone())
        );
        assert_eq!(
            owned_env_id("engine", &format!("{id}.log")),
            Some(id.clone())
        );
        // Another workload whose name extends ours, a forged hash, a
        // different workload's version, and unrelated names.
        let other = env_id("engine2", "0.32.0");
        assert_eq!(owned_env_id("engine", &other), None);
        assert_eq!(owned_env_id("engine", "engine-0.32.0-00000000"), None);
        assert_eq!(owned_env_id("engine2", &id), None);
        assert_eq!(owned_env_id("engine", "engine"), None);
        assert_eq!(owned_env_id("engine", "README"), None);
        // Versions with dashes still round-trip.
        let dashed = env_id("engine", "1.0-rc1");
        assert_eq!(owned_env_id("engine", &dashed), Some(dashed.clone()));
    }
}
