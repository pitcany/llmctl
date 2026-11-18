//! Records by verb (Rust M5c item 5, M4 carry-over 1): enroll a workload
//! that cannot be adopted — an observe-only endpoint, or a vendor app with
//! its own start/stop commands — and change an external record's control
//! authority, without editing `inventory.json` by hand.
//!
//! Both are previews until the caller passes the preview's token; both
//! go through the same inventory commit as `version --repin` (quiesced,
//! validated, a new revision). An adopted (managed) workload's authority
//! is the unit adoption proved and pinned: changing it is refused —
//! `release` first.

use llmctl_core::{
    ControlAuthority, EngineKind, GpuClaim, GpuId, Ownership, Platform, UnitName, UnitScope,
    UnitScopeMarker, WorkloadDefinition, WorkloadName,
};

use llmctl_client::catalog::{AuthorityRequest, EnrollAuthority, EnrollRequest, RecordResponse};

use crate::app::AppState;
use crate::ops::OpsError;

fn token(record: &WorkloadDefinition, revision: u64) -> String {
    let mut bytes = serde_json::to_vec(record).unwrap_or_default();
    bytes.extend(revision.to_le_bytes());
    crate::profile::digest(&bytes)
}

/// The port of `http://host:port[/...]`.
fn endpoint_port(endpoint: &str) -> Result<u16, String> {
    let rest = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .ok_or_else(|| format!("endpoint {endpoint:?} is not an http(s) URL"))?;
    let authority = rest.split('/').next().unwrap_or("");
    let port = authority
        .rsplit_once(':')
        .map(|(_, p)| p)
        .ok_or_else(|| format!("endpoint {endpoint:?} names no port"))?;
    port.parse().map_err(|_| format!("endpoint {endpoint:?}: {port:?} is not a port"))
}

fn engine_kind(text: &str) -> EngineKind {
    // `infer` knows the spellings (`llama.cpp`, `lm_studio`, `lms`); serde
    // would take any unknown one as `Other` first.
    EngineKind::infer(text)
}

pub async fn enroll(state: &AppState, req: &EnrollRequest) -> Result<RecordResponse, OpsError> {
    let name = WorkloadName::new(&req.name).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    if inv.workload(&name).is_some() {
        return Err(OpsError::Invalid(format!("{name} is already enrolled; `authority {name}` changes its control")));
    }
    if req.model.trim().is_empty() {
        return Err(OpsError::Invalid("--model is required: what its endpoint serves".into()));
    }
    let mac = inv.platform == Platform::Macos;
    let (authority, port) = match &req.authority {
        EnrollAuthority::ObserveOnly { endpoint } => {
            let port = endpoint_port(endpoint).map_err(OpsError::Invalid)?;
            if req.port.is_some_and(|p| p != port) {
                return Err(OpsError::Invalid(format!("--port {} is not the endpoint's port {port}", req.port.unwrap_or(0))));
            }
            (ControlAuthority::ObserveOnly { endpoint: endpoint.clone() }, port)
        }
        EnrollAuthority::VendorCommand { start, stop, log } => {
            let port = req
                .port
                .ok_or_else(|| OpsError::Invalid("a vendor app needs --port: running means its endpoint answers".into()))?;
            (ControlAuthority::VendorCommand { start: start.clone(), stop: stop.clone(), log: log.clone() }, port)
        }
    };
    let engine = req
        .engine
        .clone()
        .unwrap_or_else(|| if mac { "launchd".into() } else { "systemd".into() });
    let unit = if engine == "launchd" {
        UnitName::launchd(format!("com.pitcany.llmctl-next.record.{name}"))
    } else {
        UnitName::systemd(UnitScopeMarker::User, format!("llmctl-next-record-{name}.service"))
    }
    .map_err(|e| OpsError::Invalid(e.to_string()))?;
    let gpus: Vec<GpuId> = if req.gpus.is_empty() {
        inv.gpus.clone()
    } else {
        req.gpus.iter().map(|i| GpuId::new(inv.host.clone(), *i)).collect()
    };
    let record = WorkloadDefinition {
        name: name.clone(),
        engine,
        engine_kind: engine_kind(&req.engine_kind),
        model: req.model.trim().to_string(),
        gpus,
        port,
        unit,
        authority,
        ownership: Ownership::External,
        recovery: None,
        recipe: None,
        readiness_timeout_secs: None,
        gpu_claim: if req.on_demand { GpuClaim::OnDemand } else { GpuClaim::AtStart },
        hooks: Vec::new(),
        pin_files: Vec::new(),
        observed_version: None,
    };
    let mut next = (*inv).clone();
    next.workloads.push(record.clone());
    // Validate now, so a preview never promises what the commit refuses.
    let mut probe = next.clone();
    probe.revision += 1;
    probe.validate().map_err(|e| OpsError::Invalid(format!("the record would be invalid: {e}")))?;
    let t = token(&record, inv.revision);
    let mut resp = RecordResponse {
        workload: name.to_string(),
        record,
        previous: None,
        confirm: t.clone(),
        applied: false,
        inventory_revision: inv.revision,
    };
    let Some(confirm) = &req.confirm else { return Ok(resp) };
    if confirm != &t {
        return Err(OpsError::Invalid("the confirm token is not this preview's (the request or the inventory changed): preview again".into()));
    }
    resp.inventory_revision = crate::pin::commit_inventory(state, next, "enroll").await?;
    resp.applied = true;
    tracing::info!(workload = %name, "record enrolled by verb");
    Ok(resp)
}

pub async fn set_authority(state: &AppState, req: &AuthorityRequest) -> Result<RecordResponse, OpsError> {
    let name = WorkloadName::new(&req.name).map_err(|e| OpsError::Invalid(e.to_string()))?;
    let _quiet = state.ops.quiesce().await?;
    let inv = state.ops.inventory().await;
    let def = inv.workload(&name).ok_or_else(|| OpsError::UnknownWorkload(req.name.clone()))?;
    if matches!(def.ownership, Ownership::Managed { .. }) {
        return Err(OpsError::Invalid(format!(
            "{name} is adopted: its authority is the unit adoption proved and pinned; `release {name}` first"
        )));
    }
    if def.authority == req.authority {
        return Err(OpsError::Invalid(format!("{name} already has that authority")));
    }
    let mut record = def.clone();
    // The unit identity follows a manager-backed authority.
    match &req.authority {
        ControlAuthority::SystemdUnit { scope, unit } => {
            let marker = match scope {
                UnitScope::User => UnitScopeMarker::User,
                UnitScope::System => UnitScopeMarker::System,
            };
            record.unit = UnitName::systemd(marker, unit.clone()).map_err(|e| OpsError::Invalid(e.to_string()))?;
            if record.engine == "launchd" {
                return Err(OpsError::Invalid("a systemd unit on a launchd host".into()));
            }
        }
        ControlAuthority::LaunchdService { label } => {
            record.unit = UnitName::launchd(label.clone()).map_err(|e| OpsError::Invalid(e.to_string()))?;
            if inv.platform != Platform::Macos && record.engine != "fake" {
                return Err(OpsError::Invalid("a launchd service on a Linux host".into()));
            }
        }
        ControlAuthority::ObserveOnly { endpoint } => {
            let port = endpoint_port(endpoint).map_err(OpsError::Invalid)?;
            if port != record.port {
                return Err(OpsError::Invalid(format!(
                    "the endpoint's port {port} is not the record's port {}",
                    record.port
                )));
            }
        }
        ControlAuthority::VendorCommand { .. } => {}
    }
    record.authority = req.authority.clone();
    let mut next = (*inv).clone();
    if let Some(slot) = next.workloads.iter_mut().find(|w| w.name == name) {
        *slot = record.clone();
    }
    let mut probe = next.clone();
    probe.revision += 1;
    probe.validate().map_err(|e| OpsError::Invalid(format!("the record would be invalid: {e}")))?;
    let t = token(&record, inv.revision);
    let mut resp = RecordResponse {
        workload: name.to_string(),
        record,
        previous: Some(def.authority.clone()),
        confirm: t.clone(),
        applied: false,
        inventory_revision: inv.revision,
    };
    let Some(confirm) = &req.confirm else { return Ok(resp) };
    if confirm != &t {
        return Err(OpsError::Invalid("the confirm token is not this preview's (the request or the inventory changed): preview again".into()));
    }
    resp.inventory_revision = crate::pin::commit_inventory(state, next, "authority").await?;
    resp.applied = true;
    tracing::info!(workload = %name, "authority changed by verb");
    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_ports() {
        assert_eq!(endpoint_port("http://127.0.0.1:1234/v1"), Ok(1234));
        assert_eq!(endpoint_port("https://h.lan:8443"), Ok(8443));
        assert!(endpoint_port("127.0.0.1:1234").is_err());
        assert!(endpoint_port("http://host/v1").is_err());
        assert!(endpoint_port("http://host:x").is_err());
        assert_eq!(engine_kind("lm_studio"), EngineKind::LmStudio);
        assert_eq!(engine_kind("llama.cpp"), EngineKind::LlamaCpp);
    }
}
