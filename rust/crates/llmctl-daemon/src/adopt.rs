//! Adoption: bring an existing systemd user unit under llmctl ownership.
//!
//! Preview imports the unit's effective launch definition, renders the
//! service definition llmctl would own, proves equivalence through a
//! systemd round trip, and runs the enrollment / secrets / served-identity
//! checks. Transfer repeats the preview under the admission lock, requires
//! the caller's confirmed digest to still match, and only then enrolls the
//! workload as managed with that digest pinned. Nothing about the unit
//! itself changes: it keeps running, and its file is untouched.

use std::sync::Arc;

use chrono::Utc;

use llmctl_client::wire::{
    AdoptRequest, AdoptResponse, AdoptionCheck, AdoptionPreview, CheckStatus, PROTOCOL_VERSION,
};
use llmctl_core::{
    GpuId, Ownership, ProcessState, Readiness, ServedModel, UnitName, WorkloadDefinition,
    WorkloadName,
};

use crate::app::{AppState, InventoryFile};
use crate::ops::OpsError;

#[derive(Debug)]
pub enum AdoptError {
    Ops(OpsError),
    /// The unit changed since the confirmed preview.
    Stale(Box<AdoptionPreview>),
    /// A check failed; ownership was not transferred.
    Refused(Box<AdoptionPreview>),
    /// Served identity unchecked and `allow_unverified` not given.
    Unverified(Box<AdoptionPreview>),
}

impl From<OpsError> for AdoptError {
    fn from(e: OpsError) -> Self {
        AdoptError::Ops(e)
    }
}

fn check(name: &str, status: CheckStatus, detail: impl Into<String>) -> AdoptionCheck {
    AdoptionCheck {
        name: name.to_string(),
        status,
        detail: detail.into(),
    }
}

struct Parsed {
    workload: WorkloadName,
    unit: UnitName,
}

fn parse(state: &AppState, req: &AdoptRequest) -> Result<Parsed, OpsError> {
    if req.protocol != PROTOCOL_VERSION {
        return Err(OpsError::ProtocolMismatch {
            got: req.protocol,
            supported: vec![PROTOCOL_VERSION],
        });
    }
    if req.model.trim().is_empty() {
        return Err(OpsError::Invalid("model must not be empty".into()));
    }
    let unit = state
        .supervisor
        .parse_unit(&req.unit)
        .map_err(OpsError::Invalid)?;
    Ok(Parsed {
        workload: WorkloadName::new(&req.workload).map_err(|e| OpsError::Invalid(e.to_string()))?,
        unit,
    })
}

/// Read-only: imports, renders, round-trips and checks. Never enrolls.
pub async fn preview(state: &AppState, req: &AdoptRequest) -> Result<AdoptionPreview, OpsError> {
    let parsed = parse(state, req)?;
    let inv = state.ops.inventory().await;
    let mut checks = Vec::new();

    // Enrollment: identity, port and GPUs must be free and real.
    let mut problems = Vec::new();
    if inv.workloads.iter().any(|w| w.unit == parsed.unit) {
        problems.push(format!("{} is already enrolled", parsed.unit));
    }
    if inv.workload(&parsed.workload).is_some() {
        problems.push(format!("workload name {} is taken", parsed.workload));
    }
    if let Some(w) = inv.workloads.iter().find(|w| w.port == req.port) {
        problems.push(format!("port {} is used by {}", req.port, w.name));
    }
    for gpu in &req.gpus {
        if !inv.gpus.iter().any(|g| g.index == *gpu) {
            problems.push(format!("gpu {gpu} is not in this host's inventory"));
        }
    }
    if req.gpus.is_empty() {
        problems.push("at least one gpu must be declared".into());
    }
    checks.push(if problems.is_empty() {
        check(
            "enrollment",
            CheckStatus::Pass,
            "unit, name, port and GPUs are free",
        )
    } else {
        check("enrollment", CheckStatus::Fail, problems.join("; "))
    });

    checks.push(readiness_check(req));

    let imported = match state.supervisor.import(&parsed.unit).await {
        Ok(i) => i,
        Err(e) => {
            checks.push(check("unit_loaded", CheckStatus::Fail, e.to_string()));
            return Ok(empty_preview(req, checks, inv.revision));
        }
    };
    let Some(definition) = imported.definition.clone() else {
        checks.push(check(
            "unit_loaded",
            CheckStatus::Fail,
            format!("load state is {:?}", imported.load_state),
        ));
        return Ok(AdoptionPreview {
            provenance: imported.provenance,
            ..empty_preview(req, checks, inv.revision)
        });
    };
    checks.push(check(
        "unit_loaded",
        CheckStatus::Pass,
        format!("loaded ({})", state.supervisor.engine_name()),
    ));

    checks.push(if imported.secret_env_names.is_empty() {
        check("no_inline_secrets", CheckStatus::Pass, "no credential-like Environment= names")
    } else {
        check(
            "no_inline_secrets",
            CheckStatus::Fail,
            format!(
                "inline Environment= holds {} — move it to an EnvironmentFile= (referenced and digested, never read into the inventory)",
                imported.secret_env_names.join(", ")
            ),
        )
    });

    let generated = state
        .supervisor
        .render(&definition, &parsed.workload, false)
        .map_err(OpsError::Invalid)?;
    checks.push(
        match state.supervisor.roundtrip(&definition, &generated).await {
            Ok(diffs) if diffs.is_empty() => check(
                "definition_roundtrip",
                CheckStatus::Pass,
                format!(
                    "the manager loads the generated definition to the identical effective configuration"
                ),
            ),
            Ok(diffs) => check(
                "definition_roundtrip",
                CheckStatus::Fail,
                format!(
                    "generated definition differs after the manager loads it: {}",
                    diffs
                        .iter()
                        .map(|d| d.field.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ),
            Err(e) => check("definition_roundtrip", CheckStatus::Fail, e.to_string()),
        },
    );

    // Adoption only observes; `own` later renders the file and refuses what
    // a render would lose. Say so now, not at `own` (M3 pilot finding 2:
    // llama-swap's ProcessType passed every check, then own refused it).
    checks.push(ownable_check(state, &imported, &definition));

    let probe = state
        .supervisor
        .probe_unenrolled(&parsed.unit, req.port, &req.model, &requested_kind(req))
        .await;
    checks.push(
        match (probe.process, probe.readiness, &probe.served_model) {
            (ProcessState::Running, Readiness::Ready, ServedModel::Verified { model })
                if model == &req.model =>
            {
                check(
                    "served_identity",
                    CheckStatus::Pass,
                    format!("port {} serves {model}", req.port),
                )
            }
            (ProcessState::Running, _, ServedModel::Verified { model }) => check(
                "served_identity",
                CheckStatus::Fail,
                format!("port {} serves {model}, not {}", req.port, req.model),
            ),
            (ProcessState::Running, _, _) => check(
                "served_identity",
                CheckStatus::Unverified,
                format!(
                    "unit runs but /v1/models on port {} does not answer yet",
                    req.port
                ),
            ),
            (ProcessState::Stopped, _, _) => check(
                "served_identity",
                CheckStatus::Unverified,
                "unit is not running, so the served model cannot be checked",
            ),
            (ProcessState::Unknown, _, _) => check(
                "served_identity",
                CheckStatus::Unverified,
                "systemd could not be asked for the unit's state",
            ),
        },
    );

    Ok(AdoptionPreview {
        unit: req.unit.clone(),
        workload: req.workload.clone(),
        digest: Some(definition.digest()),
        definition: Some(definition),
        generated_unit: Some(generated),
        provenance: imported.provenance,
        checks,
        inventory_revision: inv.revision,
    })
}

/// Whether `own` could take over this job's file after adoption: the same
/// preconditions own checks (no drop-ins, a backing file, nothing the
/// definition does not carry). Never blocks adoption.
fn ownable_check(
    state: &AppState,
    imported: &crate::supervisor::ImportedJob,
    definition: &llmctl_core::LaunchDefinition,
) -> AdoptionCheck {
    let mut problems = Vec::new();
    if !imported.drop_in_paths.is_empty() {
        problems.push(format!("drop-ins ({})", imported.drop_in_paths.join(", ")));
    }
    if imported.fragment_path.is_empty() {
        problems.push("no definition file of its own".to_string());
    } else {
        let uncovered = state
            .supervisor
            .uncovered(std::path::Path::new(&imported.fragment_path), definition);
        if !uncovered.is_empty() {
            problems.push(format!(
                "sets {} that the definition does not carry",
                uncovered.join(", ")
            ));
        }
    }
    if problems.is_empty() {
        check(
            "ownable",
            CheckStatus::Pass,
            "own can take over the file later",
        )
    } else {
        check(
            "ownable",
            CheckStatus::Warn,
            format!(
                "adoption works, but own would refuse: {}",
                problems.join("; ")
            ),
        )
    }
}

fn empty_preview(req: &AdoptRequest, checks: Vec<AdoptionCheck>, revision: u64) -> AdoptionPreview {
    AdoptionPreview {
        unit: req.unit.clone(),
        workload: req.workload.clone(),
        digest: None,
        definition: None,
        generated_unit: None,
        provenance: Vec::new(),
        checks,
        inventory_revision: revision,
    }
}

/// Re-preview under the admission lock and enroll as managed when the
/// confirmed digest still matches and every check allows it.
pub async fn transfer(state: &AppState, req: AdoptRequest) -> Result<AdoptResponse, AdoptError> {
    let Some(confirm) = req.confirm_digest.clone() else {
        return Err(OpsError::Invalid(
            "confirm_digest is required to transfer ownership; preview first".into(),
        )
        .into());
    };
    let _quiet = state.ops.quiesce().await?;
    let preview = preview(state, &req).await?;

    if preview.digest.as_deref() != Some(confirm.as_str()) {
        return Err(AdoptError::Stale(Box::new(preview)));
    }
    if preview.has_failures() {
        return Err(AdoptError::Refused(Box::new(preview)));
    }
    if preview.has_unverified() && !req.allow_unverified {
        return Err(AdoptError::Unverified(Box::new(preview)));
    }
    let (Some(definition), Some(digest)) = (preview.definition.clone(), preview.digest.clone())
    else {
        return Err(AdoptError::Refused(Box::new(preview)));
    };

    let parsed = parse(state, &req)?;
    let inv = state.ops.inventory().await;
    let version_pin = preview
        .definition
        .as_ref()
        .and_then(|d| state.supervisor.program_digest_pin(d));
    let mut next = (*inv).clone();
    next.workloads.push(WorkloadDefinition {
        name: parsed.workload,
        engine: state.supervisor.engine_name().into(),
        // Named at adoption (M4b) or unknown: the startup migration fills
        // a legacy kind from the recipe once the owner adds one.
        engine_kind: requested_kind(&req),
        model: req.model.clone(),
        gpus: req
            .gpus
            .iter()
            .map(|i| GpuId::new(inv.host.clone(), *i))
            .collect(),
        port: req.port,
        unit: parsed.unit.clone(),
        authority: state.supervisor.authority_for(&parsed.unit),
        ownership: Ownership::Managed {
            definition: Box::new(definition),
            digest,
            adopted_at: Utc::now(),
            provenance: preview.provenance.clone(),
            service_file: None,
            // launchd pins the program binary's digest here at adoption
            // (an in-place upgrade of the single static binary is drift);
            // systemd pins python-env facts lazily via recipes.
            version: version_pin,
            // Schema 2 (version facts) is pinned by the startup re-pin
            // once the inventory names a recipe for this workload.
            definition_schema: 1,
            pending_apply: None,
        },
        // Recovery stays off for adopted workloads until the owner turns it
        // on in the inventory (owner decision, 2026-11-16).
        recovery: None,
        recipe: None,
        readiness_timeout_secs: req.readiness_timeout_secs,
        gpu_claim: requested_claim(&req),
        hooks: Vec::new(),
        pin_files: Vec::new(),
        observed_version: None,
    });
    next.revision += 1;
    next.validate()
        .map_err(|e| OpsError::Invalid(format!("adopted inventory invalid: {e}")))?;

    // File first (fsynced, atomic rename): if persisting fails, nothing
    // changed anywhere.
    crate::engine::write_json_atomic(
        &state.paths.inventory,
        &InventoryFile {
            inventory: next.clone(),
        },
    )
    .await
    .map_err(|e| OpsError::Engine(e.to_string()))?;
    let revision = next.revision;
    state.ops.update_inventory(Arc::new(next)).await;
    tracing::info!(unit = %req.unit, workload = %req.workload, revision, "unit adopted");
    Ok(AdoptResponse {
        preview,
        inventory_revision: revision,
    })
}

/// How long a start will wait. A workload that loads at start and keeps
/// the default is called out: a Strata lane needs 11-20 minutes, and a run
/// that times out has already evicted what it replaced.
fn readiness_check(req: &AdoptRequest) -> AdoptionCheck {
    const NAME: &str = "readiness_timeout";
    let max = crate::ops::MAX_POLL_TIMEOUT_SECS;
    match req.readiness_timeout_secs {
        Some(secs) if (1..=max).contains(&secs) => {
            check(NAME, CheckStatus::Pass, format!("a start waits up to {secs} s"))
        }
        Some(secs) => check(
            NAME,
            CheckStatus::Fail,
            format!("{secs} s is outside 1..={max}"),
        ),
        None => {
            let default = llmctl_core::planner::DEFAULT_READINESS_TIMEOUT_SECS;
            let status = if requested_claim(req).is_at_start() {
                CheckStatus::Warn
            } else {
                CheckStatus::Pass
            };
            check(
                NAME,
                status,
                format!(
                    "a start waits {default} s (the default); pass --readiness-timeout for a model that loads longer"
                ),
            )
        }
    }
}

/// The GPU claim the request names, else the kind's default (vendor
/// services load on demand).
fn requested_claim(req: &AdoptRequest) -> llmctl_core::GpuClaim {
    match req.gpu_claim.as_deref() {
        Some("on_demand") | Some("on-demand") => llmctl_core::GpuClaim::OnDemand,
        Some(_) => llmctl_core::GpuClaim::AtStart,
        None => llmctl_core::GpuClaim::default_for(&requested_kind(req)),
    }
}

/// The engine kind the adoption request names, or legacy.
fn requested_kind(req: &AdoptRequest) -> llmctl_core::EngineKind {
    req.engine_kind
        .as_deref()
        .map(llmctl_core::EngineKind::infer)
        .unwrap_or_else(llmctl_core::EngineKind::legacy)
}
