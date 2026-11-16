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
    ControlAuthority, GpuId, Ownership, ProcessState, Readiness, ServedModel, UnitName, UnitScope,
    UnitScopeMarker, WorkloadDefinition, WorkloadName,
};

use crate::app::{AppState, InventoryFile};
use crate::ops::OpsError;
use crate::unitdef;

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

fn parse(req: &AdoptRequest) -> Result<Parsed, OpsError> {
    if req.protocol != PROTOCOL_VERSION {
        return Err(OpsError::ProtocolMismatch {
            got: req.protocol,
            supported: vec![PROTOCOL_VERSION],
        });
    }
    if !req.unit.ends_with(".service") || req.unit.contains('/') {
        return Err(OpsError::Invalid(format!(
            "unit must be a bare service name like foo.service, got {:?}",
            req.unit
        )));
    }
    if req.model.trim().is_empty() {
        return Err(OpsError::Invalid("model must not be empty".into()));
    }
    Ok(Parsed {
        workload: WorkloadName::new(&req.workload).map_err(|e| OpsError::Invalid(e.to_string()))?,
        unit: UnitName::systemd(UnitScopeMarker::User, &req.unit)
            .map_err(|e| OpsError::Invalid(e.to_string()))?,
    })
}

/// Read-only: imports, renders, round-trips and checks. Never enrolls.
pub async fn preview(state: &AppState, req: &AdoptRequest) -> Result<AdoptionPreview, OpsError> {
    let parsed = parse(req)?;
    let inv = state.ops.inventory().await;
    let timeout = state.systemd.show_timeout();
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

    let imported = match unitdef::import_unit(&parsed.unit, timeout).await {
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
            format!("systemd load state is {:?}", imported.load_state),
        ));
        return Ok(AdoptionPreview {
            provenance: imported.provenance,
            ..empty_preview(req, checks, inv.revision)
        });
    };
    checks.push(check(
        "unit_loaded",
        CheckStatus::Pass,
        "loaded (user scope)",
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

    let generated = unitdef::render_unit(
        &definition,
        &format!(
            "llmctl-next managed workload {} (adopted from {})",
            parsed.workload, req.unit
        ),
    );
    checks.push(
        match unitdef::roundtrip(&definition, &generated, timeout).await {
            Ok(diffs) if diffs.is_empty() => check(
                "definition_roundtrip",
                CheckStatus::Pass,
                "systemd loads the generated definition to the identical effective configuration",
            ),
            Ok(diffs) => check(
                "definition_roundtrip",
                CheckStatus::Fail,
                format!(
                    "generated definition differs after systemd loads it: {}",
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

    let probe = state
        .systemd
        .probe_unenrolled(&parsed.unit, req.port, &req.model)
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

    let parsed = parse(&req)?;
    let inv = state.ops.inventory().await;
    let mut next = (*inv).clone();
    next.workloads.push(WorkloadDefinition {
        name: parsed.workload,
        engine: crate::systemd::ENGINE_NAME.into(),
        model: req.model.clone(),
        gpus: req
            .gpus
            .iter()
            .map(|i| GpuId::new(inv.host.clone(), *i))
            .collect(),
        port: req.port,
        unit: parsed.unit,
        authority: ControlAuthority::SystemdUnit {
            scope: UnitScope::User,
            unit: req.unit.clone(),
        },
        ownership: Ownership::Managed {
            definition: Box::new(definition),
            digest,
            adopted_at: Utc::now(),
            provenance: preview.provenance.clone(),
            service_file: None,
            version: None,
            // Schema 2 (version facts) is pinned by the startup re-pin
            // once the inventory names a recipe for this workload.
            definition_schema: 1,
        },
        // Recovery stays off for adopted workloads until the owner turns it
        // on in the inventory (owner decision, 2026-11-16).
        recovery: None,
        recipe: None,
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
