//! Staged upgrades with rollback (plan D): version ownership without
//! auto-update. Stage builds a *new* environment side by side with `uv`
//! (the serving env is never touched), validate checks it offline
//! (imports, reported versions, the served model's config), and promote is
//! an ordinary admitted operation — stop, apply the new definition, start,
//! verify readiness, identity and version — whose failure rolls back to
//! the pinned N by the normal executor path.
//!
//! Old environments stay until an explicit `gc`; shared weights and HF
//! caches are never touched.

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::Utc;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use tokio::process::Command;

use llmctl_client::wire::PROTOCOL_VERSION;
use llmctl_core::{LaunchDefinition, OpKind, Ownership, WorkloadName};

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

/// Rough floor for a staged engine env (vLLM wheels unpack to ~10 GB).
const MIN_FREE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

pub struct StageOutcome {
    pub row: UpgradeRow,
    pub validation: serde_json::Value,
}

fn env_id(workload: &str, version: &str) -> String {
    let hash = Sha256::digest(format!("{workload}:{version}").as_bytes());
    let hex = format!("{:x}", hash);
    format!("{workload}-{version}-{}", &hex[..8])
}

fn envs_root(state: &AppState) -> PathBuf {
    state.paths.data_dir.join("envs")
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
    cmd.kill_on_drop(true);
    let out = tokio::time::timeout(Duration::from_secs(600), cmd.output())
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
) -> Result<(WorkloadName, LaunchDefinition, llmctl_core::RecipeConfig), UpgradeError> {
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

/// Stage a new environment for `workload` at `version`, then validate it
/// offline. An interrupted stage leaves no manifest and is never
/// promotable.
pub async fn stage(
    state: &AppState,
    name: &str,
    version: &str,
) -> Result<StageOutcome, UpgradeError> {
    let (wl, _definition, recipe) = managed_parts(state, name).await?;
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
    if let Some(old) = crate::version::interpreter_for(&recipe, &inv_def_for(state, &wl).await) {
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
    let wl_def = inv.workload(&wl).cloned();
    let model = wl_def
        .as_ref()
        .and_then(|d| {
            d.recipe
                .as_ref()
                .and_then(|r| r.model_config.clone())
                .or_else(|| Some(d.model.clone()))
        })
        .unwrap_or_default();
    let validation =
        match crate::version::validate_offline(&recipe, &interpreter.display().to_string(), &model)
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
        "interpreter": interpreter.display().to_string(),
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
    let row = UpgradeRow {
        id: env
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
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
    // Idempotent re-staging replaces the row's status via a fresh id? The
    // id is the env dir name, so re-staging the same version hits the same
    // id: upsert by delete+insert.
    let _ = state.ops.db().delete_upgrade(&row.id).await;
    state.ops.db().insert_upgrade(row.clone()).await?;
    Ok(row)
}

async fn pinned_version_of(state: &AppState, workload: &str) -> Option<String> {
    let inv = state.ops.inventory().await;
    let def = inv.workload(&WorkloadName::new(workload).ok()?)?;
    let Ownership::Managed { version, .. } = &def.ownership else {
        return None;
    };
    // The recipe's first package version is "the version".
    version
        .as_ref()
        .and_then(|v| v.packages.first().map(|(_, v)| v.clone()))
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

    // The stage must be validated: manifest present, row validated.
    let env = envs_root(state).join(env_id(name, version));
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
    let rows = state.ops.db().list_upgrades(name).await?;
    let Some(row) = rows.iter().find(|r| r.to_version == version) else {
        return Err(refuse(format!("no upgrade record for {name} at {version}")));
    };
    if row.status != "validated" {
        return Err(refuse(format!(
            "stage for {version} is {} (not validated); restage it",
            row.status
        )));
    }
    let interpreter = manifest["interpreter"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    if interpreter.is_empty() {
        return Err(refuse("staged manifest has no interpreter"));
    }

    // Expected versions after promote: the recipe's packages at `version`.
    let expected: Vec<(String, String)> = recipe
        .packages
        .iter()
        .map(|p| (p.clone(), version.to_string()))
        .collect();

    let next_def = crate::version::rewrite_definition(&old_def, &recipe, &interpreter);
    if recipe.apply != "exec" {
        // The override env file next_def points at must exist with exactly
        // the pinned content before the definition is applied.
        let prefix = recipe.env_prefix.clone().unwrap_or_else(|| "VLLM".into());
        let (path, content) = crate::version::override_env(&interpreter, &prefix);
        tokio::fs::write(&path, content.as_bytes()).await?;
    }
    // N (the definition and env being replaced) is the rollback target:
    // stash it in the row's detail.
    let old_interpreter = crate::version::interpreter_for(&recipe, &inv_def_for(state, &wl).await)
        .unwrap_or_default();
    state
        .ops
        .db()
        .update_upgrade_status(
            &row.id,
            "promoting",
            Some(json!({
                "old_definition": old_def,
                "old_interpreter": old_interpreter,
                "from_version": row.from_version,
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
        &interpreter,
        &expected,
    )
    .map_err(|e| OpsError::Invalid(e.to_string()))?;

    let admitted = state
        .ops
        .admit(AdmitParams {
            protocol: PROTOCOL_VERSION,
            idempotency_key: format!("upgrade-{name}-{version}"),
            kind: OpKind::Upgrade,
            workload: name.to_string(),
            expected_revision: None,
            expected_stop_targets: None,
            plan_override: Some(plan),
        })
        .await;
    match admitted {
        Ok(AdmitOutcome::Admitted(row_op)) | Ok(AdmitOutcome::Replayed(row_op)) => {
            let row_id = row.id.clone();
            let op_id = row_op.id.to_string();
            let ops = state.ops.clone();
            tokio::spawn(async move {
                // Record the outcome once the operation terminalizes, so
                // `upgrade list` and the history stay truthful.
                let deadline = tokio::time::Instant::now() + Duration::from_secs(60 * 30);
                loop {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    let Ok(Some(op)) = ops.db().get_op(&op_id).await else {
                        continue;
                    };
                    if !op.status.is_terminal() {
                        if tokio::time::Instant::now() >= deadline {
                            break;
                        }
                        continue;
                    }
                    let status = match op.status {
                        llmctl_client::wire::OpStatus::Succeeded => "promoted",
                        _ => "failed",
                    };
                    // Detail stays (it holds the N stash for rollback); the
                    // operation id links to the full outcome.
                    let _ = ops
                        .db()
                        .update_upgrade_status(&row_id, status, None, None)
                        .await;
                    break;
                }
            });
            state
                .ops
                .db()
                .update_upgrade_status(&row.id, "promoting", None, Some(row_op.id.to_string()))
                .await?;
            Ok(crate::api::operation_view(&row_op))
        }
        Err(e) => {
            state
                .ops
                .db()
                .update_upgrade_status(
                    &row.id,
                    "failed",
                    Some(json!({"error": e.to_string()})),
                    None,
                )
                .await?;
            Err(e.into())
        }
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
    let rows = state.ops.db().list_upgrades(name).await?;
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
    let old_interpreter = detail
        .get("old_interpreter")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let from_version = detail
        .get("from_version")
        .and_then(|v| v.as_str())
        .unwrap_or("?")
        .to_string();
    if old_interpreter.is_empty() || from_version == "?" {
        return Err(refuse(
            "the pre-upgrade interpreter or version was never pinned",
        ));
    }
    let expected: Vec<(String, String)> = recipe
        .packages
        .iter()
        .map(|p| (p.clone(), from_version.clone()))
        .collect();

    state
        .ops
        .db()
        .update_upgrade_status(&last.id, "promoting", None, None)
        .await?;
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
        &wl,
        &current_def,
        &old_def,
        &old_interpreter,
        &expected,
    )
    .map_err(|e| OpsError::Invalid(e.to_string()))?;
    let admitted = state
        .ops
        .admit(AdmitParams {
            protocol: PROTOCOL_VERSION,
            idempotency_key: format!("upgrade-rollback-{name}-{}", last.id),
            kind: OpKind::Upgrade,
            workload: name.to_string(),
            expected_revision: None,
            expected_stop_targets: None,
            plan_override: Some(plan),
        })
        .await;
    match admitted {
        Ok(AdmitOutcome::Admitted(row_op)) | Ok(AdmitOutcome::Replayed(row_op)) => {
            let ops = state.ops.clone();
            let row_id = last.id.clone();
            let op_id = row_op.id.to_string();
            tokio::spawn(async move {
                let deadline = tokio::time::Instant::now() + Duration::from_secs(60 * 30);
                loop {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    let Ok(Some(op)) = ops.db().get_op(&op_id).await else {
                        continue;
                    };
                    if !op.status.is_terminal() {
                        if tokio::time::Instant::now() >= deadline {
                            break;
                        }
                        continue;
                    }
                    let status = match op.status {
                        llmctl_client::wire::OpStatus::Succeeded => "rolled_back",
                        _ => "failed",
                    };
                    let _ = ops
                        .db()
                        .update_upgrade_status(&row_id, status, None, None)
                        .await;
                    break;
                }
            });
            Ok(crate::api::operation_view(&row_op))
        }
        Err(e) => {
            state
                .ops
                .db()
                .update_upgrade_status(
                    &last.id,
                    "failed",
                    Some(json!({"error": e.to_string()})),
                    None,
                )
                .await?;
            Err(e.into())
        }
    }
}

pub async fn list(state: &AppState, name: &str) -> Result<Vec<UpgradeRow>, UpgradeError> {
    Ok(state.ops.db().list_upgrades(name).await?)
}

/// Drop envs not in use: not the current interpreter's env, and not any
/// non-discarded upgrade row's env. Their rows become `discarded`.
pub async fn gc(state: &AppState, name: &str) -> Result<serde_json::Value, UpgradeError> {
    let (wl, ..) = managed_parts(state, name).await?;
    let inv = state.ops.inventory().await;
    let current_interpreter = inv.workload(&wl).and_then(|d| match &d.ownership {
        Ownership::Managed { version, .. } => version.as_ref().and_then(|v| v.interpreter.clone()),
        _ => None,
    });
    let rows = state.ops.db().list_upgrades(name).await?;
    let keep: std::collections::BTreeSet<String> = rows
        .iter()
        .filter(|r| r.status != "discarded")
        .map(|r| r.env_path.clone())
        .chain(current_interpreter.map(|p| {
            // The env dir containing the live interpreter.
            Path::new(&p)
                .ancestors()
                .nth(2)
                .map(|a| a.display().to_string())
                .unwrap_or_default()
        }))
        .collect();
    let root = envs_root(state);
    let mut removed = Vec::new();
    let mut entries = tokio::fs::read_dir(&root).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|e| e == "staging" || e == "log")
        {
            if path.extension().is_some_and(|e| e == "staging") {
                let _ = tokio::fs::remove_file(&path).await;
                removed.push(path.display().to_string());
            }
            continue;
        }
        let ps = path.display().to_string();
        if keep.contains(&ps) {
            continue;
        }
        tokio::fs::remove_dir_all(&path).await?;
        removed.push(ps);
    }
    for r in rows.iter().filter(|r| r.status == "discarded") {
        let _ = state
            .ops
            .db()
            .update_upgrade_status(&r.id, "discarded", None, None)
            .await;
    }
    Ok(json!({"removed": removed, "kept": keep.into_iter().collect::<Vec<_>>()}))
}
