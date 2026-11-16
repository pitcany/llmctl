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
    CancelRequest, HealthResponse, LogsRequest, LogsResponse, OperationListResponse, OperationView,
    PhaseView, PlanRequest, PlanResponse, StatusResponse, WorkloadView, PROTOCOL_VERSION,
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
        .route("/v1/operations/{id}/cancel", post(cancel_operation))
        .route("/v1/operations/{id}/acknowledge", post(ack_operation))
        .route("/v1/adoptions/preview", post(adoption_preview))
        .route("/v1/adoptions", post(adoption_transfer))
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
            }
        })
        .collect();

    Ok(Json(StatusResponse {
        host: state.host.to_string(),
        platform: format!("{:?}", inv.platform),
        inventory_revision: inv.revision,
        generated_at: chrono::Utc::now(),
        workloads,
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
