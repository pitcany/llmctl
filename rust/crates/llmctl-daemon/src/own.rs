//! Writing and owning the service definition (plan item C). Adoption pins
//! the definition and leaves the unit file as the authority; `own` goes
//! further: llmctl renders the unit file itself, proves systemd still sees
//! the identical effective definition, and records what was there before
//! so `release` can hand the exact original back (the Python-llmctl
//! hand-back path after the pilot).
//!
//! Invariants:
//!
//! * Never write through a symlink — the link (or file) itself is replaced
//!   atomically, and the repo copy a link pointed at is never touched.
//! * Prove or restore: `daemon-reload` + re-import must equal the pinned
//!   definition, or the recorded original goes back byte-for-byte and the
//!   operation refuses.
//! * The unit is never restarted by own/release (an equivalent definition
//!   leaves a running engine alone), and the drift check gets a cheap
//!   file-digest short circuit once llmctl owns the file.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use llmctl_core::{LaunchDefinition, Ownership, PreviousUnitFile, ServiceFileRecord, UnitName};

use crate::app::{AppState, InventoryFile};
use crate::engine::{write_json_atomic, EngineError, Faults};
use crate::ops::OpsError;
use crate::unitdef;

#[derive(Debug, thiserror::Error)]
pub enum OwnError {
    /// Preconditions failed or the prove step caught a bad render; the
    /// original (if any) is back in place.
    #[error("{0}")]
    Refused(String),
    #[error(transparent)]
    Ops(#[from] OpsError),
    #[error("io: {0}")]
    Io(String),
}

impl From<EngineError> for OwnError {
    fn from(e: EngineError) -> Self {
        OwnError::Io(e.to_string())
    }
}

fn refuse(msg: impl Into<String>) -> OwnError {
    OwnError::Refused(msg.into())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

/// `llmctl-next own NAME` — replace the unit file (or the link itself)
/// with the rendered definition and keep the original for `release`.
pub async fn own(state: &AppState, name: &str) -> Result<OwnOutcome, OwnError> {
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    let wl_name =
        llmctl_core::WorkloadName::new(name).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let Some(def) = inv.workload(&wl_name) else {
        return Err(OpsError::UnknownWorkload(name.to_string()).into());
    };
    let Ownership::Managed {
        definition: pinned, ..
    } = &def.ownership
    else {
        return Err(refuse(format!(
            "{name} is not an adopted (managed) workload; own applies after adopt"
        )));
    };
    if def.engine != crate::systemd::ENGINE_NAME {
        return Err(refuse(format!("{name} is not a systemd workload")));
    }

    let timeout = state.systemd.show_timeout();
    let imported = unitdef::import_unit(&def.unit, timeout).await?;
    preconditions(&imported, &pinned, name)?;

    let path = PathBuf::from(&imported.fragment_path);
    let previous = record_previous(&state.paths.data_dir, &wl_name, &path).await?;
    let header = format!(
        "llmctl-next managed workload {} (definition {}; owned {})",
        name,
        pinned.digest(),
        chrono::Utc::now().to_rfc3339()
    );
    let ctx = OwnCtx {
        paths: state.paths.clone(),
        systemd: state.systemd.clone(),
    };
    let record = write_and_prove(
        &ctx,
        &def.unit,
        &pinned,
        &header,
        &path,
        previous.clone(),
        previous,
    )
    .await?;

    // Inventory: the ownership gains the service file record. Revision
    // moves, so stale plans cannot confirm against a changed world.
    let mut next = (*inv).clone();
    for w in &mut next.workloads {
        if w.name == wl_name {
            if let Ownership::Managed { service_file, .. } = &mut w.ownership {
                *service_file = Some(record.clone());
            }
        }
    }
    bump_revision(&ctx, state.ops.as_ref(), next).await?;
    Ok(record)
}

pub type OwnOutcome = ServiceFileRecord;

/// `llmctl-next release NAME` — restore the recorded original
/// byte-for-byte (or the link) and set ownership back to external.
pub async fn release(state: &AppState, name: &str) -> Result<(), OwnError> {
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    let wl_name =
        llmctl_core::WorkloadName::new(name).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let Some(def) = inv.workload(&wl_name) else {
        return Err(OpsError::UnknownWorkload(name.to_string()).into());
    };
    let Ownership::Managed {
        definition: pinned,
        service_file: Some(record),
        ..
    } = &def.ownership
    else {
        return Err(refuse(format!(
            "{name} has no owned service file to release (adopted but not owned?)"
        )));
    };

    let timeout = state.systemd.show_timeout();
    let path = Path::new(&record.path);
    restore_previous(&record.previous, path).await?;
    unitdef::daemon_reload(timeout).await?;

    // The restored original must still be the same effective definition as
    // pinned — otherwise the world changed under us and a human decides.
    let imported = unitdef::import_unit(&def.unit, timeout).await?;
    match &imported.definition {
        Some(live) if live.diff(pinned).is_empty() => {}
        other => {
            // Put the owned file back; refuse rather than release something
            // that no longer matches what llmctl pinned.
            let header = format!(
                "llmctl-next managed workload {} (definition {}; re-owned {})",
                name,
                pinned.digest(),
                chrono::Utc::now().to_rfc3339()
            );
            let mut rendered = unitdef::render_unit(pinned, &header);
            // Keep the unit enable-able: the restored original's [Install].
            let source = tokio::fs::read_to_string(path).await.unwrap_or_default();
            if let Some(install) = unitdef::install_section(&source) {
                rendered.push('\n');
                rendered.push_str(&install);
            }
            write_file_atomic(path, rendered.as_bytes()).await?;
            unitdef::daemon_reload(timeout).await?;
            return Err(refuse(format!(
                "restored original no longer matches the pinned definition ({:?} vs pinned); kept the llmctl file",
                other.as_ref().map(|d| d.digest())
            )));
        }
    }

    let mut next = (*inv).clone();
    for w in &mut next.workloads {
        if w.name == wl_name {
            w.ownership = Ownership::External;
        }
    }
    bump_revision(
        &OwnCtx {
            paths: state.paths.clone(),
            systemd: state.systemd.clone(),
        },
        state.ops.as_ref(),
        next,
    )
    .await?;
    // The backup is no longer needed once the original is back in place.
    if let PreviousUnitFile::File { backup_path, .. } = &record.previous {
        let _ = tokio::fs::remove_file(backup_path).await;
    }
    Ok(())
}

/// Everything definition application needs, independent of `AppState`, so
/// the operation executor can swap definitions mid-operation (its own op
/// is the admission; quiesce would self-deadlock on the active op).
#[derive(Clone)]
pub struct OwnCtx {
    pub paths: crate::app::DevPaths,
    pub systemd: crate::systemd::SystemdEngine,
}

/// The staged-upgrade primitive (plan item D): write `next` with the same
/// prove-or-restore steps and re-pin the inventory to it. The caller keeps
/// the old definition as its rollback target.
pub async fn apply_definition(
    state: &AppState,
    name: &str,
    next: &LaunchDefinition,
) -> Result<(), OwnError> {
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    apply_definition_ctx(
        &OwnCtx {
            paths: state.paths.clone(),
            systemd: state.systemd.clone(),
        },
        &inv,
        name,
        next,
        state.ops.as_ref(),
    )
    .await
    .map(|_| ())
}

/// Executor entry point: same prove-or-restore steps, but the inventory
/// update goes through the given [`OpManager`] and no admission lock is
/// taken (the running operation is the admission).
pub async fn apply_definition_ctx(
    ctx: &OwnCtx,
    inv: &llmctl_core::HostInventory,
    name: &str,
    next: &LaunchDefinition,
    ops: &crate::ops::OpManager,
) -> Result<ServiceFileRecord, OwnError> {
    let wl_name =
        llmctl_core::WorkloadName::new(name).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let Some(def) = inv.workload(&wl_name) else {
        return Err(OpsError::UnknownWorkload(name.to_string()).into());
    };
    let Ownership::Managed {
        definition: pinned,
        service_file: Some(record),
        ..
    } = &def.ownership
    else {
        return Err(refuse(format!(
            "{name} has no owned service file; own it before changing its definition"
        )));
    };
    let timeout = ctx.systemd.show_timeout();
    let imported = unitdef::import_unit(&def.unit, timeout).await?;
    // Drift is measured against the *pin* (the definition being replaced),
    // not the candidate: an upgrade's whole point is that `next` differs.
    preconditions(&imported, pinned, name)?;
    let path = PathBuf::from(&record.path);
    // The *current* owned bytes are what a failed prove puts back (N). They
    // go to their own slot: the release target recorded at `own` time —
    // the original file or link — must survive every definition change, or
    // `release` hands back an llmctl render instead of the original (M2
    // pilot window 2; for a regular-file original it would also overwrite
    // the only backup of the original bytes).
    let current_bytes = tokio::fs::read(&path)
        .await
        .map_err(|e| refuse(format!("reading owned unit file {}: {e}", path.display())))?;
    let backup = undo_path(&ctx.paths.data_dir, &wl_name);
    tokio::fs::create_dir_all(backup.parent().expect("parent"))
        .await
        .map_err(|e| OwnError::Io(format!("creating backup dir: {e}")))?;
    tokio::fs::write(&backup, &current_bytes)
        .await
        .map_err(|e| OwnError::Io(format!("writing backup: {e}")))?;
    let undo = PreviousUnitFile::File {
        content_digest: sha256_hex(&current_bytes),
        backup_path: backup.display().to_string(),
    };

    let header = format!(
        "llmctl-next managed workload {} (definition {}; updated {})",
        name,
        next.digest(),
        chrono::Utc::now().to_rfc3339()
    );
    let new_record = write_and_prove(
        ctx,
        &def.unit,
        next,
        &header,
        &path,
        undo,
        record.previous.clone(),
    )
    .await?;

    let mut next_inv = inv.clone();
    for w in &mut next_inv.workloads {
        if w.name == wl_name {
            if let Ownership::Managed {
                definition,
                digest,
                service_file,
                ..
            } = &mut w.ownership
            {
                **definition = next.clone();
                *digest = next.digest();
                *service_file = Some(new_record.clone());
            }
        }
    }
    bump_revision(ctx, ops, next_inv).await?;
    Ok(new_record)
}

// ---- shared steps -----------------------------------------------------------

fn preconditions(
    imported: &unitdef::ImportedUnit,
    pinned: &LaunchDefinition,
    name: &str,
) -> Result<(), OwnError> {
    if !imported.drop_in_paths.is_empty() {
        return Err(refuse(format!(
            "{name} has drop-ins ({}); a rendered unit cannot own a definition they modify",
            imported.drop_in_paths.join(", ")
        )));
    }
    if imported.fragment_path.is_empty() {
        return Err(refuse(format!(
            "{name} has no unit file of its own (loaded from elsewhere)"
        )));
    }
    match &imported.definition {
        Some(live) if live.diff(pinned).is_empty() => Ok(()),
        Some(live) => Err(refuse(format!(
            "{name} drifted from its pinned definition ({}); own refuses a moving target",
            live.diff(pinned)
                .iter()
                .map(|d| d.field.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
        None => Err(refuse(format!("{name} is not loaded"))),
    }
}

/// What is at `path` right now: a symlink's target, or the file's bytes
/// copied to the backup slot.
async fn record_previous(
    data_dir: &Path,
    workload: &llmctl_core::WorkloadName,
    path: &Path,
) -> Result<PreviousUnitFile, OwnError> {
    let meta = tokio::fs::symlink_metadata(path)
        .await
        .map_err(|e| OwnError::Io(format!("stat {}: {e}", path.display())))?;
    if meta.file_type().is_symlink() {
        let target = tokio::fs::read_link(path)
            .await
            .map_err(|e| OwnError::Io(format!("readlink {}: {e}", path.display())))?;
        return Ok(PreviousUnitFile::Link {
            target: target.display().to_string(),
        });
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| OwnError::Io(format!("read {}: {e}", path.display())))?;
    let backup = backup_path(data_dir, workload);
    tokio::fs::create_dir_all(backup.parent().expect("parent"))
        .await
        .map_err(|e| OwnError::Io(format!("creating backup dir: {e}")))?;
    tokio::fs::write(&backup, &bytes)
        .await
        .map_err(|e| OwnError::Io(format!("writing backup: {e}")))?;
    Ok(PreviousUnitFile::File {
        content_digest: sha256_hex(&bytes),
        backup_path: backup.display().to_string(),
    })
}

async fn restore_previous(previous: &PreviousUnitFile, path: &Path) -> Result<(), OwnError> {
    match previous {
        PreviousUnitFile::Link { target } => {
            remove_path(path).await?;
            tokio::fs::symlink(target, path)
                .await
                .map_err(|e| OwnError::Io(format!("relinking {}: {e}", path.display())))?;
        }
        PreviousUnitFile::File {
            content_digest,
            backup_path,
        } => {
            let bytes = tokio::fs::read(backup_path).await.map_err(|e| {
                OwnError::Io(format!(
                    "reading backup {} (original {}): {e}",
                    backup_path, content_digest
                ))
            })?;
            write_file_atomic(path, &bytes).await?;
        }
    }
    Ok(())
}

/// Remove a file, a symlink, or nothing — never recurse.
async fn remove_path(path: &Path) -> Result<(), OwnError> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(_) => tokio::fs::remove_file(path)
            .await
            .map_err(|e| OwnError::Io(format!("removing {}: {e}", path.display()))),
        Err(_) => Ok(()),
    }
}

/// Where a definition change keeps the owned bytes it replaces, for its
/// own prove-or-restore. Never the release target's slot.
fn undo_path(data_dir: &Path, workload: &llmctl_core::WorkloadName) -> PathBuf {
    data_dir
        .join("ownership")
        .join(format!("{}.apply-undo", workload.as_str()))
}

fn backup_path(data_dir: &Path, workload: &llmctl_core::WorkloadName) -> PathBuf {
    data_dir
        .join("ownership")
        .join(format!("{}.previous", workload.as_str()))
}

/// Render, write (replacing the link/file itself, never writing through a
/// symlink), reload, re-import and compare. Any mismatch puts the recorded
/// original back and refuses.
#[allow(clippy::too_many_arguments)]
async fn write_and_prove(
    ctx: &OwnCtx,
    unit: &UnitName,
    expected: &LaunchDefinition,
    header: &str,
    path: &Path,
    // Put back if the prove fails.
    undo: PreviousUnitFile,
    // Recorded for `release` (what `own` replaced; never an llmctl render).
    release_target: PreviousUnitFile,
) -> Result<ServiceFileRecord, OwnError> {
    let timeout = ctx.systemd.show_timeout();
    // What is there now (through a link: the file it points at). Every
    // directive must be carried by the definition, or rendering would drop
    // it silently (the pilot's RestartSec/StartLimit* case): refuse.
    let source = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| OwnError::Io(format!("reading {}: {e}", path.display())))?;
    let uncovered = unitdef::uncovered_directives(&source, expected);
    if !uncovered.is_empty() {
        return Err(refuse(format!(
            "{unit} sets directives the pinned definition does not carry ({}); a rendered unit would drop them — re-adopt to pin them, or remove them from the unit",
            uncovered.join(", ")
        )));
    }
    let mut rendered = unitdef::render_unit(expected, header);
    // Test hook: a subtly wrong render must be caught by the prove step.
    let faults = Faults::load(&ctx.paths.faults).await;
    if faults.corrupt_render {
        rendered.push_str("Environment=LLMCTL_FAULT_RENDER=1\n");
    }
    if let Some(install) = unitdef::install_section(&source) {
        rendered.push('\n');
        rendered.push_str(&install);
    }

    // Replace the link/file itself, never write through a symlink.
    remove_path(path).await?;
    write_file_atomic(path, rendered.as_bytes()).await?;
    let digest = sha256_hex(rendered.as_bytes());

    unitdef::daemon_reload(timeout).await?;
    let imported = unitdef::import_unit(unit, timeout).await?;
    let equivalent = matches!(&imported.definition, Some(live) if live.diff(expected).is_empty());
    if !equivalent {
        restore_previous(&undo, path).await?;
        unitdef::daemon_reload(timeout).await?;
        return Err(refuse(format!(
            "the rendered unit for {unit} did not load to the identical effective definition; original restored"
        )));
    }
    Ok(ServiceFileRecord {
        path: path.display().to_string(),
        digest,
        previous: release_target,
    })
}

async fn write_file_atomic(path: &Path, bytes: &[u8]) -> Result<(), OwnError> {
    let tmp = path.with_extension("llmctl-next.tmp");
    tokio::fs::write(&tmp, bytes)
        .await
        .map_err(|e| OwnError::Io(format!("writing {}: {e}", tmp.display())))?;
    tokio::fs::rename(&tmp, path)
        .await
        .map_err(|e| OwnError::Io(format!("renaming to {}: {e}", path.display())))?;
    Ok(())
}

/// Persist the new inventory (file first, then swap) exactly like adoption.
async fn bump_revision(
    ctx: &OwnCtx,
    ops: &crate::ops::OpManager,
    next: llmctl_core::HostInventory,
) -> Result<(), OwnError> {
    let mut bumped = next;
    bumped.revision += 1;
    bumped
        .validate()
        .map_err(|e| OpsError::Invalid(format!("updated inventory invalid: {e}")))?;
    write_json_atomic(
        &ctx.paths.inventory,
        &InventoryFile {
            inventory: bumped.clone(),
        },
    )
    .await
    .map_err(|e| OpsError::Engine(e.to_string()))?;
    ops.update_inventory(std::sync::Arc::new(bumped)).await;
    Ok(())
}
