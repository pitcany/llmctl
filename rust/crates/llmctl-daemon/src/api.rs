//! The versioned local control API: HTTP/JSON over the Unix socket.
//! Mutations are admitted durably and return operation ids immediately;
//! clients poll. Errors are `{"error": {code, message, details?}}`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use llmctl_client::wire::codes;
use llmctl_client::wire::{
    AckRequest, AdmitRequest, AdmitResponse, AdoptRequest, AdoptResponse, AdoptionPreview,
    CancelRequest, GpuOccupantView, GpuView, HealthResponse, LogsRequest, LogsResponse,
    OperationListResponse, OperationView, PhaseView, PlanRequest, PlanResponse, StatusResponse,
    WorkloadView, PROTOCOL_VERSION,
};

use crate::adopt::AdoptError;
use crate::app::AppState;
use crate::db::OpRow;
use crate::ops::{AdmitOutcome, OpsError};

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/status", get(status))
        .route("/v1/inventory", get(inventory))
        .route("/v1/plans", post(create_plan))
        .route("/v1/operations", post(admit_operation))
        .route("/v1/operations", get(list_operations))
        .route("/v1/operations/{id}", get(get_operation))
        .route("/v1/operations/by-key/{key}", get(get_operation_by_key))
        .route("/v1/operations/{id}/cancel", post(cancel_operation))
        .route("/v1/operations/{id}/acknowledge", post(ack_operation))
        .route("/v1/adoptions/preview", post(adoption_preview))
        .route("/v1/adoptions", post(adoption_transfer))
        .route("/v1/ownership/own", post(ownership_own))
        .route("/v1/ownership/release", post(ownership_release))
        .route("/v1/upgrades/stage", post(upgrade_stage))
        .route("/v1/upgrades/promote", post(upgrade_promote))
        .route("/v1/upgrades/rollback", post(upgrade_rollback))
        .route("/v1/upgrades/gc", post(upgrade_gc))
        .route("/v1/upgrades", post(upgrade_list))
        .route("/v1/installs/plan", post(install_plan))
        .route("/v1/workloads/{name}/version", get(workload_version))
        .route("/v1/workloads/{name}/engine", get(workload_engine))
        .route("/v1/unloads", post(admit_unload))
        .route("/v1/lanes/import", post(lanes_import))
        .route("/v1/lanes/edit", post(lanes_edit))
        .route("/v1/lanes/{workload}", get(lanes_status))
        .route("/v1/logs", post(logs))
        .with_state(state)
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        protocol: PROTOCOL_VERSION,
        supported_protocols: vec![PROTOCOL_VERSION],
        daemon_version: env!("CARGO_PKG_VERSION").to_string(),
        host: state.host.to_string(),
        pid: std::process::id(),
        started_at: state.started_at,
    })
}

async fn status(State(state): State<Arc<AppState>>) -> Result<Json<StatusResponse>, ApiError> {
    let inv = state.ops.inventory().await;
    let observations = state
        .ops
        .engine()
        .observe(&inv)
        .await
        .map_err(|e| ApiError::from_ops(&OpsError::Engine(e.to_string())))?;
    let intents: std::collections::HashMap<String, String> = state
        .ops
        .db()
        .get_intents()
        .await
        .map_err(|e| ApiError::from_ops(&OpsError::Db(e)))?
        .into_iter()
        .collect();

    // Manager view of each unit (recovery decision input) up front.
    let mut recovery_states: std::collections::HashMap<String, String> = Default::default();
    for def in &inv.workloads {
        let s = state
            .ops
            .engine()
            .recovery_state(def)
            .await
            .map(|s| s.as_str().to_string())
            .unwrap_or_else(|_| "unknown".into());
        recovery_states.insert(def.name.to_string(), s);
    }

    let workloads = inv
        .workloads
        .iter()
        .map(|def| {
            let obs = observations.iter().find(|o| o.workload == def.name);
            let intent = intents
                .get(def.name.as_str())
                .cloned()
                .unwrap_or_else(|| "unmanaged".to_string());
            WorkloadView {
                name: def.name.to_string(),
                engine: def.engine.clone(),
                model: def.model.clone(),
                gpus: def.gpus.iter().map(|g| g.to_string()).collect(),
                port: def.port,
                unit: def.unit.to_string(),
                intent,
                process: obs
                    .map(|o| format!("{:?}", o.process))
                    .unwrap_or_else(|| "unknown".into()),
                readiness: obs
                    .map(|o| format!("{:?}", o.readiness))
                    .unwrap_or_else(|| "unknown".into()),
                served_model: obs.and_then(|o| match &o.served_model {
                    llmctl_core::ServedModel::Verified { model }
                    | llmctl_core::ServedModel::Unverified { model } => Some(model.clone()),
                    _ => None,
                }),
                served_model_verified: obs
                    .map(|o| matches!(o.served_model, llmctl_core::ServedModel::Verified { .. }))
                    .unwrap_or(false),
                observed_at: obs.map(|o| o.observed_at),
                observation_unknown: obs.is_none(),
                ownership: match &def.ownership {
                    llmctl_core::Ownership::External => "external".into(),
                    llmctl_core::Ownership::Managed { .. } => "managed".into(),
                },
                adopted_digest: match &def.ownership {
                    llmctl_core::Ownership::Managed { digest, .. } => Some(digest.clone()),
                    llmctl_core::Ownership::External => None,
                },
                recovery_state: recovery_states
                    .get(def.name.as_str())
                    .cloned()
                    .unwrap_or_else(|| "unknown".into()),
                recovery_policy: def
                    .recovery
                    .map(|p| format!("{}/{}s", p.max_attempts, p.window_secs)),
            }
        })
        .collect();

    // Per card: enrolled workloads observed running vs foreign processes.
    let enrolled_units: std::collections::BTreeSet<String> =
        inv.workloads.iter().map(|d| d.unit.to_string()).collect();
    let probe = state.ops.engine().gpu_occupancy(&inv.gpus).await;
    let gpus = inv
        .gpus
        .iter()
        .map(|g| {
            let enrolled: Vec<String> = inv
                .workloads
                .iter()
                .filter(|d| {
                    d.gpus.contains(g)
                        && observations.iter().any(|o| {
                            o.workload == d.name && o.process == llmctl_core::ProcessState::Running
                        })
                })
                .map(|d| d.name.to_string())
                .collect();
            let (foreign, error) = match &probe {
                Ok(occupants) => {
                    let foreign = occupants
                        .iter()
                        .filter(|o| {
                            o.gpu == g.index
                                && !o.unit.as_ref().is_some_and(|u| enrolled_units.contains(u))
                        })
                        .map(|o| GpuOccupantView {
                            pid: o.pid,
                            unit: o.unit.clone(),
                            memory_mib: o.memory_mib,
                        })
                        .collect();
                    (foreign, None)
                }
                Err(e) => (Vec::new(), Some(e.to_string())),
            };
            GpuView {
                index: g.index,
                enrolled,
                foreign,
                error,
            }
        })
        .collect();

    Ok(Json(StatusResponse {
        host: state.host.to_string(),
        platform: format!("{:?}", inv.platform),
        inventory_revision: inv.revision,
        generated_at: chrono::Utc::now(),
        workloads,
        gpus,
    }))
}

async fn inventory(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let inv = state.ops.inventory().await;
    Json(serde_json::to_value(&*inv).unwrap_or_else(|_| json!({})))
}

async fn create_plan(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PlanRequest>,
) -> Result<Json<PlanResponse>, ApiError> {
    let (plan, _obs) = state.ops.build_plan(req.kind, &req.workload).await?;
    let stop_targets: Vec<String> = plan
        .steps
        .iter()
        .filter_map(|s| match s {
            llmctl_core::PlanStep::StopUnit { workload, .. } => Some(workload.to_string()),
            _ => None,
        })
        .collect();
    let revision = plan.revision;
    Ok(Json(PlanResponse {
        revision,
        stop_targets,
        plan,
    }))
}

/// Read-only: no admission lock, so logs stay readable while an operation
/// runs (that is when they matter most).
async fn logs(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LogsRequest>,
) -> Result<Json<LogsResponse>, ApiError> {
    use llmctl_client::wire::{DEFAULT_LOG_LINES, MAX_LOG_LINES};
    let lines = req.lines.unwrap_or(DEFAULT_LOG_LINES);
    if !(1..=MAX_LOG_LINES).contains(&lines) {
        return Err(
            OpsError::Invalid(format!("lines must be 1..={MAX_LOG_LINES}, got {lines}")).into(),
        );
    }
    let name = llmctl_core::WorkloadName::new(&req.workload)
        .map_err(|e| OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let Some(def) = inv.workload(&name) else {
        return Err(OpsError::UnknownWorkload(req.workload).into());
    };
    let page = state
        .ops
        .engine()
        .logs(def, lines, req.after.clone())
        .await
        .map_err(|e| match e {
            crate::engine::EngineError::Unsupported(m) => OpsError::Unsupported(m),
            crate::engine::EngineError::InvalidInput(m) => OpsError::Invalid(m),
            other => OpsError::Engine(other.to_string()),
        })?;
    Ok(Json(LogsResponse {
        workload: def.name.to_string(),
        unit: def.unit.to_string(),
        entries: page.entries,
        cursor: page.cursor,
        more: page.more,
    }))
}

async fn admit_operation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AdmitRequest>,
) -> Result<Response, ApiError> {
    let outcome = state
        .ops
        .admit(crate::ops::AdmitParams {
            protocol: req.protocol,
            idempotency_key: req.idempotency_key,
            kind: req.kind,
            workload: req.workload,
            expected_revision: req.expected_revision,
            expected_stop_targets: req.expected_stop_targets,
            plan_override: None,
        })
        .await?;
    let (row, replayed, status) = match outcome {
        AdmitOutcome::Admitted(row) => (row, false, StatusCode::ACCEPTED),
        AdmitOutcome::Replayed(row) => (row, true, StatusCode::OK),
    };
    let body = AdmitResponse {
        operation: operation_view(&row),
        replayed,
    };
    Ok((status, Json(body)).into_response())
}

async fn admit_unload(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::UnloadRequest>,
) -> Result<Response, ApiError> {
    // A provisional plan carries the model; admission re-plans it against
    // observations taken under its lock (a stopped service refuses).
    let name = llmctl_core::WorkloadName::new(&req.workload)
        .map_err(|e| crate::ops::OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let def = inv
        .workload(&name)
        .ok_or_else(|| crate::ops::OpsError::UnknownWorkload(req.workload.clone()))?;
    let provisional = llmctl_core::Plan {
        id: llmctl_core::OperationId::generate(),
        kind: llmctl_core::OpKind::Unload,
        target: name.clone(),
        revision: inv.revision,
        expected_model: def.model.clone(),
        affected_gpus: Vec::new(),
        steps: vec![llmctl_core::PlanStep::UnloadModel {
            workload: name.clone(),
            unit: def.unit.clone(),
            model: req.model.clone(),
        }],
        rollback: Vec::new(),
    };
    let outcome = state
        .ops
        .admit(crate::ops::AdmitParams {
            protocol: req.protocol,
            idempotency_key: req.idempotency_key,
            kind: llmctl_core::OpKind::Unload,
            workload: req.workload,
            expected_revision: None,
            expected_stop_targets: None,
            plan_override: Some(provisional),
        })
        .await?;
    let (row, replayed, status) = match outcome {
        AdmitOutcome::Admitted(row) => (row, false, StatusCode::ACCEPTED),
        AdmitOutcome::Replayed(row) => (row, true, StatusCode::OK),
    };
    Ok((
        status,
        Json(AdmitResponse {
            operation: operation_view(&row),
            replayed,
        }),
    )
        .into_response())
}

/// Engine-level view of one workload: readiness, servable and loaded
/// models (M4b step 1). Read-only.
async fn workload_engine(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<llmctl_client::wire::EngineReport>, ApiError> {
    use crate::engine_api::EngineProtocol;
    use crate::probe_http::HttpProbe;
    let wl = llmctl_core::WorkloadName::new(&name)
        .map_err(|e| crate::ops::OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let def = inv
        .workload(&wl)
        .ok_or_else(|| crate::ops::OpsError::UnknownWorkload(name.clone()))?;
    let protocol = EngineProtocol::for_kind(&def.engine_kind);
    let ans =
        crate::engine_api::probe_local(protocol, def.port, std::time::Duration::from_secs(3)).await;
    let (state_s, servable) = match ans.probe {
        HttpProbe::Models(ids) => ("ready".to_string(), ids),
        HttpProbe::Unreachable => ("unreachable".to_string(), Vec::new()),
        HttpProbe::Status(c) => (format!("status {c}"), Vec::new()),
        HttpProbe::Garbled => ("garbled".to_string(), Vec::new()),
    };
    Ok(Json(llmctl_client::wire::EngineReport {
        workload: name,
        engine_kind: def.engine_kind.to_string(),
        protocol: protocol.name().into(),
        state: state_s,
        servable: servable
            .into_iter()
            .filter(|m| Some(m.as_str()) != protocol.service_identity())
            .collect(),
        loaded: ans.loaded,
        supports_unload: def.engine_kind.has_engine_unload(),
    }))
}

async fn lanes_import(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::LaneImportRequest>,
) -> Result<Json<llmctl_client::wire::LaneImportResponse>, ApiError> {
    check_protocol(req.protocol)?;
    let _one_at_a_time = state.adopt_lock.lock().await;
    Ok(Json(crate::lanes::import(&state, req).await.map_err(ApiError::from_lanes)?))
}

async fn lanes_status(
    State(state): State<Arc<AppState>>,
    Path(workload): Path<String>,
) -> Result<Json<llmctl_client::wire::LaneStatusResponse>, ApiError> {
    Ok(Json(crate::lanes::status(&state, &workload).await.map_err(ApiError::from_lanes)?))
}

/// Plan lane edits, or (with the plan digest) admit the write.
async fn lanes_edit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::LaneEditRequest>,
) -> Result<Response, ApiError> {
    check_protocol(req.protocol)?;
    let _one_at_a_time = state.adopt_lock.lock().await;
    let Some(confirm) = req.confirm.clone() else {
        let plan = crate::lanes::plan(&state, &req.workload, &req.edits)
            .await
            .map_err(ApiError::from_lanes)?;
        return Ok(Json(plan).into_response());
    };
    let key = req.idempotency_key.clone().unwrap_or_else(|| {
        format!("lanes-{}-{}", req.workload, &confirm[confirm.len().saturating_sub(12)..])
    });
    let outcome = crate::lanes::apply(&state, &req.workload, &req.edits, &confirm, key)
        .await
        .map_err(ApiError::from_lanes)?;
    let (row, replayed, status) = match outcome {
        AdmitOutcome::Admitted(row) => (row, false, StatusCode::ACCEPTED),
        AdmitOutcome::Replayed(row) => (row, true, StatusCode::OK),
    };
    Ok((
        status,
        Json(AdmitResponse {
            operation: operation_view(&row),
            replayed,
        }),
    )
        .into_response())
}

fn check_protocol(got: u32) -> Result<(), ApiError> {
    if got == llmctl_client::wire::PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(ApiError::from_ops(&crate::ops::OpsError::ProtocolMismatch {
            got,
            supported: vec![llmctl_client::wire::PROTOCOL_VERSION],
        }))
    }
}

async fn list_operations(
    State(state): State<Arc<AppState>>,
) -> Result<Json<OperationListResponse>, ApiError> {
    let rows = state
        .ops
        .db()
        .list_ops(100)
        .await
        .map_err(|e| ApiError::from_ops(&OpsError::Db(e)))?;
    Ok(Json(OperationListResponse {
        operations: rows.iter().map(operation_view).collect(),
    }))
}

async fn get_operation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<OperationView>, ApiError> {
    let row = state
        .ops
        .db()
        .get_op(&id)
        .await
        .map_err(|e| ApiError::from_ops(&OpsError::Db(e)))?
        .ok_or_else(|| {
            ApiError::not_found(codes::UNKNOWN_OPERATION, format!("no operation {id}"))
        })?;
    Ok(Json(operation_view(&row)))
}

/// Resolve an idempotency key to its operation — how a client closes the
/// lost-ACK window after a transport error (M3 decision 4): the key
/// answers "did my admit land?" without issuing anything new.
async fn get_operation_by_key(
    State(state): State<Arc<AppState>>,
    Path(key): Path<String>,
) -> Result<Json<OperationView>, ApiError> {
    let parsed = llmctl_core::IdempotencyKey::try_from(key.clone())
        .map_err(|e| OpsError::Invalid(e.to_string()))?;
    let row = state
        .ops
        .db()
        .find_by_key(parsed.as_str())
        .await
        .map_err(|e| ApiError::from_ops(&OpsError::Db(e)))?
        .ok_or_else(|| {
            ApiError::not_found(
                codes::UNKNOWN_OPERATION,
                format!("no operation with idempotency key {key}"),
            )
        })?;
    Ok(Json(operation_view(&row)))
}

async fn cancel_operation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(_req): Json<CancelRequest>,
) -> Result<Json<OperationView>, ApiError> {
    let row = state
        .ops
        .cancel(&id)
        .await
        .map_err(|e| ApiError::from_ops(&e))?;
    Ok(Json(operation_view(&row)))
}

async fn ack_operation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(_req): Json<AckRequest>,
) -> Result<Json<OperationView>, ApiError> {
    let row = state
        .ops
        .acknowledge(&id)
        .await
        .map_err(|e| ApiError::from_ops(&e))?;
    Ok(Json(operation_view(&row)))
}

// ---- views + errors -------------------------------------------------------

pub fn operation_view(row: &OpRow) -> OperationView {
    OperationView {
        id: row.id.to_string(),
        idempotency_key: row.idempotency_key.to_string(),
        kind: row.kind,
        workload: row.workload.to_string(),
        status: row.status,
        phases: row
            .phases
            .iter()
            .map(|p| PhaseView {
                index: p.index,
                label: p.label.clone(),
                status: p.status,
                detail: p.detail.clone(),
            })
            .collect(),
        created_at: row.created_at,
        updated_at: row.updated_at,
        result: row.result.clone(),
        recovery: row.recovery.clone(),
        acknowledged: row.acknowledged,
    }
}

async fn adoption_preview(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AdoptRequest>,
) -> Result<Json<AdoptionPreview>, ApiError> {
    let _one_at_a_time = state.adopt_lock.lock().await;
    Ok(Json(crate::adopt::preview(&state, &req).await?))
}

async fn adoption_transfer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AdoptRequest>,
) -> Result<Json<AdoptResponse>, ApiError> {
    let _one_at_a_time = state.adopt_lock.lock().await;
    crate::adopt::transfer(&state, req)
        .await
        .map(Json)
        .map_err(ApiError::from_adopt)
}

async fn ownership_own(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::OwnRequest>,
) -> Result<Json<llmctl_client::wire::OwnResponse>, ApiError> {
    let _one_at_a_time = state.adopt_lock.lock().await;
    let record = crate::own::own(&state, &req.workload)
        .await
        .map_err(ApiError::from_own)?;
    let revision = state.ops.inventory().await.revision;
    Ok(Json(llmctl_client::wire::OwnResponse {
        workload: req.workload,
        path: record.path,
        digest: record.digest,
        previous_kind: match record.previous {
            llmctl_core::PreviousUnitFile::File { .. } => "file",
            llmctl_core::PreviousUnitFile::Link { .. } => "link",
        }
        .to_string(),
        inventory_revision: revision,
    }))
}

async fn ownership_release(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::ReleaseRequest>,
) -> Result<Json<llmctl_client::wire::ReleaseResponse>, ApiError> {
    let _one_at_a_time = state.adopt_lock.lock().await;
    crate::own::release(&state, &req.workload)
        .await
        .map_err(ApiError::from_own)?;
    let revision = state.ops.inventory().await.revision;
    Ok(Json(llmctl_client::wire::ReleaseResponse {
        workload: req.workload,
        ownership: "external".into(),
        inventory_revision: revision,
    }))
}

async fn upgrade_stage(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::StageRequest>,
) -> Result<Json<llmctl_client::wire::StageResponse>, ApiError> {
    let _one_at_a_time = state.adopt_lock.lock().await;
    let out = crate::upgrade::stage(&state, &req.workload, &req.version)
        .await
        .map_err(ApiError::from_upgrade)?;
    Ok(Json(llmctl_client::wire::StageResponse {
        workload: req.workload,
        version: req.version,
        env: out.row.env_path,
        interpreter: out.row.interpreter,
        status: out.row.status,
        validation: out.validation,
    }))
}

async fn upgrade_promote(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::PromoteRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _one_at_a_time = state.adopt_lock.lock().await;
    let view = crate::upgrade::promote(&state, &req.workload, &req.version)
        .await
        .map_err(ApiError::from_upgrade)?;
    Ok(Json(serde_json::to_value(&view).unwrap_or_default()))
}

async fn upgrade_rollback(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::RollbackRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _one_at_a_time = state.adopt_lock.lock().await;
    let view = crate::upgrade::rollback(&state, &req.workload)
        .await
        .map_err(ApiError::from_upgrade)?;
    Ok(Json(serde_json::to_value(&view).unwrap_or_default()))
}

async fn upgrade_gc(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Result<Json<llmctl_client::wire::GcResponse>, ApiError> {
    let workload = req
        .get("workload")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    // Never alongside a stage (it would delete the env being built) or a
    // promote/rollback deciding which env to use.
    let _one_at_a_time = state.adopt_lock.lock().await;
    let out = crate::upgrade::gc(&state, &workload)
        .await
        .map_err(ApiError::from_upgrade)?;
    Ok(Json(llmctl_client::wire::GcResponse {
        workload,
        removed: out
            .get("removed")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default(),
        kept: out
            .get("kept")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default(),
    }))
}

async fn upgrade_list(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Result<Json<llmctl_client::wire::UpgradeListResponse>, ApiError> {
    let workload = req
        .get("workload")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let rows = crate::upgrade::list(&state, &workload)
        .await
        .map_err(ApiError::from_upgrade)?;
    Ok(Json(llmctl_client::wire::UpgradeListResponse {
        upgrades: rows
            .into_iter()
            .map(|r| llmctl_client::wire::UpgradeRowView {
                id: r.id,
                workload: r.workload,
                from_version: r.from_version,
                to_version: r.to_version,
                env_path: r.env_path,
                interpreter: r.interpreter,
                status: r.status,
                detail: r.detail,
                operation_id: r.operation_id,
                created_at: r.created_at,
                updated_at: r.updated_at,
            })
            .collect(),
    }))
}

/// The vendor-app plan (M4a step 4): observed/pinned version, drift, and
/// the owner-run backup/reinstall path. Never executes anything.
async fn install_plan(
    State(state): State<Arc<AppState>>,
    Json(req): Json<llmctl_client::wire::InstallPlanRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _one_at_a_time = state.adopt_lock.lock().await;
    let plan = crate::upgrade::vendor_plan(&state, &req.workload)
        .await
        .map_err(ApiError::from_upgrade)?;
    Ok(Json(plan))
}

/// Version facts of one workload: the pin, the live probe, and the drift
/// between them (a mismatch is a finding, never auto-corrected).
async fn workload_version(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let wl = llmctl_core::WorkloadName::new(&name)
        .map_err(|e| OpsError::Invalid(e.to_string()))?;
    let inv = state.ops.inventory().await;
    let Some(def) = inv.workload(&wl) else {
        return Err(OpsError::UnknownWorkload(name).into());
    };
    let report = match &def.recipe {
        Some(recipe) => {
            let live = crate::version::probe(recipe, def)
                .await
                .map_err(|e| OpsError::Engine(format!("probing {name}: {e}")))?;
            let pinned = match &def.ownership {
                llmctl_core::Ownership::Managed { version, .. } => version.clone(),
                _ => None,
            };
            let drift = match (&pinned, def.ownership.clone()) {
                (Some(p), _) => crate::version::version_drift(p, &live),
                (None, _) => Vec::new(),
            };
            json!({
                "workload": name,
                "engine_kind": def.engine_kind.to_string(),
                "recipe_kind": match recipe {
                    llmctl_core::InstallRecipe::PythonEnv(_) => "python_env",
                    llmctl_core::InstallRecipe::ReleaseArchive(_) => "release_archive",
                    llmctl_core::InstallRecipe::SourceBuild(_) => "source_build",
                    llmctl_core::InstallRecipe::VendorApp(_) => "vendor_app",
                    llmctl_core::InstallRecipe::External(_) => "external",
                },
                "pinned": pinned,
                "live": live,
                "drift": drift,
            })
        }
        None => json!({
            "workload": name,
            "engine_kind": def.engine_kind.to_string(),
            "recipe_kind": null,
            "pinned": null,
            "live": null,
            "drift": [],
            "note": "no recipe; version pinning is off",
        }),
    };
    Ok(Json(report))
}

pub struct ApiError {
    status: StatusCode,
    body: llmctl_client::wire::ApiError,
}

impl ApiError {
    fn not_found(code: &str, message: String) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: llmctl_client::wire::ApiError {
                error: llmctl_client::wire::ApiErrorBody {
                    code: code.to_string(),
                    message,
                    details: None,
                },
            },
        }
    }

    fn from_ops(e: &OpsError) -> Self {
        let (code, status) = e.code_and_status();
        Self {
            status: StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            body: llmctl_client::wire::ApiError {
                error: llmctl_client::wire::ApiErrorBody {
                    code: code.to_string(),
                    message: e.to_string(),
                    details: e.details(),
                },
            },
        }
    }
}

impl ApiError {
    fn from_lanes(e: crate::lanes::LanesError) -> Self {
        let (status, code, message) = match e {
            crate::lanes::LanesError::Ops(e) => return ApiError::from_ops(&e),
            crate::lanes::LanesError::Refused(m) => (StatusCode::CONFLICT, codes::LANES_REFUSED, m),
            crate::lanes::LanesError::Io(m) => (StatusCode::INTERNAL_SERVER_ERROR, codes::DAEMON_ERROR, m),
        };
        Self {
            status,
            body: llmctl_client::wire::ApiError {
                error: llmctl_client::wire::ApiErrorBody {
                    code: code.to_string(),
                    message,
                    details: None,
                },
            },
        }
    }

    fn from_upgrade(e: crate::upgrade::UpgradeError) -> Self {
        match e {
            crate::upgrade::UpgradeError::Ops(e) => ApiError::from_ops(&e),
            crate::upgrade::UpgradeError::Db(e) => ApiError::from_ops(&OpsError::Db(e)),
            crate::upgrade::UpgradeError::Refused(msg) => Self {
                status: StatusCode::CONFLICT,
                body: llmctl_client::wire::ApiError {
                    error: llmctl_client::wire::ApiErrorBody {
                        code: codes::UPGRADE_REFUSED.to_string(),
                        message: msg,
                        details: None,
                    },
                },
            },
            crate::upgrade::UpgradeError::Io(msg) => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: llmctl_client::wire::ApiError {
                    error: llmctl_client::wire::ApiErrorBody {
                        code: codes::DAEMON_ERROR.to_string(),
                        message: msg,
                        details: None,
                    },
                },
            },
        }
    }

    fn from_own(e: crate::own::OwnError) -> Self {
        match e {
            crate::own::OwnError::Ops(e) => ApiError::from_ops(&e),
            crate::own::OwnError::Refused(msg) => Self {
                status: StatusCode::CONFLICT,
                body: llmctl_client::wire::ApiError {
                    error: llmctl_client::wire::ApiErrorBody {
                        code: codes::OWNERSHIP_REFUSED.to_string(),
                        message: msg,
                        details: None,
                    },
                },
            },
            crate::own::OwnError::Io(msg) => Self {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                body: llmctl_client::wire::ApiError {
                    error: llmctl_client::wire::ApiErrorBody {
                        code: codes::DAEMON_ERROR.to_string(),
                        message: msg,
                        details: None,
                    },
                },
            },
        }
    }

    fn from_adopt(e: AdoptError) -> Self {
        let (code, message, preview) = match e {
            AdoptError::Ops(e) => return ApiError::from_ops(&e),
            AdoptError::Stale(p) => (
                codes::STALE_ADOPTION,
                "the unit's effective definition changed since the confirmed preview; review the fresh preview",
                p,
            ),
            AdoptError::Refused(p) => (
                codes::ADOPTION_REFUSED,
                "an adoption check failed; ownership was not transferred",
                p,
            ),
            AdoptError::Unverified(p) => (
                codes::ADOPTION_UNVERIFIED,
                "served identity could not be checked; start the unit or pass allow_unverified",
                p,
            ),
        };
        Self {
            status: StatusCode::CONFLICT,
            body: llmctl_client::wire::ApiError {
                error: llmctl_client::wire::ApiErrorBody {
                    code: code.to_string(),
                    message: message.to_string(),
                    details: Some(json!({ "preview": preview })),
                },
            },
        }
    }
}

impl From<OpsError> for ApiError {
    fn from(e: OpsError) -> Self {
        ApiError::from_ops(&e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}
