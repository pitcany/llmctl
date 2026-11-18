//! M5c routes: presets and the model catalog. Merged into the main router
//! with one line (`api::router`), so parallel milestones do not collide in
//! `api.rs`. Errors use the shared `{"error": {code, message, details?}}`
//! shape.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use llmctl_client::catalog::{self as cat, codes as ccodes};
use llmctl_client::wire::{codes, ApiError as WireError, ApiErrorBody, PROTOCOL_VERSION};

use crate::app::AppState;
use crate::presets::PresetsError;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/presets", get(presets_list))
        .route("/v1/presets/edit", post(presets_edit))
        .route("/v1/presets/new", post(presets_new))
        .route("/v1/presets/run", post(presets_run))
        .route("/v1/presets/settle", post(presets_settle))
        .route("/v1/presets/{alias}", get(presets_show))
}

pub struct M5cError {
    status: StatusCode,
    code: &'static str,
    message: String,
    details: Option<serde_json::Value>,
}

impl M5cError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self { status, code, message: message.into(), details: None }
    }
}

impl IntoResponse for M5cError {
    fn into_response(self) -> Response {
        let body = WireError {
            error: ApiErrorBody {
                code: self.code.to_string(),
                message: self.message,
                details: self.details,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

impl From<PresetsError> for M5cError {
    fn from(e: PresetsError) -> Self {
        match e {
            PresetsError::UnknownPreset(m) => M5cError::new(StatusCode::NOT_FOUND, ccodes::UNKNOWN_PRESET, m),
            PresetsError::Invalid(m) => M5cError::new(StatusCode::BAD_REQUEST, codes::INVALID_REQUEST, m),
            PresetsError::Refused(m) => M5cError::new(StatusCode::CONFLICT, ccodes::PRESET_REFUSED, m),
            PresetsError::Drift { message, drift } => M5cError {
                status: StatusCode::CONFLICT,
                code: ccodes::PRESET_DRIFT,
                message,
                details: Some(serde_json::json!({ "drift": drift })),
            },
            PresetsError::Io(m) => M5cError::new(StatusCode::INTERNAL_SERVER_ERROR, codes::DAEMON_ERROR, m),
        }
    }
}

pub fn check_protocol(got: u32) -> Result<(), M5cError> {
    if got == PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(M5cError {
            status: StatusCode::BAD_REQUEST,
            code: codes::PROTOCOL_MISMATCH,
            message: format!("protocol {got} is not supported (this agent speaks {PROTOCOL_VERSION})"),
            details: Some(serde_json::json!({ "got": got, "supported": [PROTOCOL_VERSION] })),
        })
    }
}

async fn presets_list(State(state): State<Arc<AppState>>) -> Json<cat::PresetListResponse> {
    Json(crate::presets::list(&state).await)
}

async fn presets_show(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
) -> Result<Json<cat::PresetShowResponse>, M5cError> {
    Ok(Json(crate::presets::show(&state, &alias).await?))
}

async fn presets_edit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::PresetEditRequest>,
) -> Result<Json<cat::PresetWriteResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::presets::edit(&state, &req).await?))
}

async fn presets_new(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::PresetNewRequest>,
) -> Result<Json<cat::PresetWriteResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::presets::new(&state, &req).await?))
}

async fn presets_run(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::PresetRunRequest>,
) -> Result<Json<cat::PresetRunResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::presets::run(&state, &req).await?))
}

async fn presets_settle(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::PresetSettleRequest>,
) -> Result<Json<cat::PresetSettleResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::presets::settle(&state, &req).await?))
}
