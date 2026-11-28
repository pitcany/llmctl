//! M5c routes: presets and the model catalog. Merged into the main router
//! with one line (`api::router`), so parallel milestones do not collide in
//! `api.rs`. Errors use the shared `{"error": {code, message, details?}}`
//! shape.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

use llmctl_client::catalog::{self as cat, codes as ccodes};
use llmctl_client::wire::{codes, ApiError as WireError, ApiErrorBody, PROTOCOL_VERSION};

use crate::app::AppState;
use crate::catalog::CatalogError;
use crate::presets::PresetsError;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/presets", get(presets_list))
        .route("/v1/presets/edit", post(presets_edit))
        .route("/v1/presets/new", post(presets_new))
        .route("/v1/presets/run", post(presets_run))
        .route("/v1/presets/settle", post(presets_settle))
        .route("/v1/presets/{alias}", get(presets_show))
        .route("/v1/catalog/models", get(catalog_list))
        .route("/v1/catalog/model", get(catalog_show))
        .route("/v1/catalog/import", post(catalog_import))
        .route("/v1/catalog/export", get(catalog_export))
        .route("/v1/catalog/models/add", post(catalog_add))
        .route("/v1/catalog/models/edit", post(catalog_edit))
        .route("/v1/catalog/models/enable", post(catalog_enable))
        .route("/v1/catalog/models/delete", post(catalog_delete))
        .route("/v1/history/import", post(history_import))
        .route("/v1/history", get(history_list))
        .route("/v1/catalog/scan", post(catalog_scan))
        .route("/v1/records/enroll", post(records_enroll))
        .route("/v1/records/authority", post(records_authority))
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

impl From<CatalogError> for M5cError {
    fn from(e: CatalogError) -> Self {
        match e {
            CatalogError::Unknown(m) => M5cError::new(StatusCode::NOT_FOUND, ccodes::UNKNOWN_MODEL, m),
            CatalogError::Invalid(m) => M5cError::new(StatusCode::BAD_REQUEST, codes::INVALID_REQUEST, m),
            CatalogError::Refused(m) => M5cError::new(StatusCode::CONFLICT, ccodes::CATALOG_REFUSED, m),
            CatalogError::Io(m) => M5cError::new(StatusCode::INTERNAL_SERVER_ERROR, codes::DAEMON_ERROR, m),
        }
    }
}

#[derive(serde::Deserialize)]
struct ListQuery {
    #[serde(default)]
    all: Option<String>,
}

#[derive(serde::Deserialize)]
struct ModelQuery {
    m: String,
}

async fn catalog_list(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ListQuery>,
) -> Result<Json<cat::CatalogListResponse>, M5cError> {
    let all = q.all.as_deref().is_some_and(|v| v != "0" && v != "false");
    Ok(Json(crate::catalog::list(&state, all).await?))
}

async fn catalog_show(
    State(state): State<Arc<AppState>>,
    Query(q): Query<ModelQuery>,
) -> Result<Json<cat::CatalogModel>, M5cError> {
    Ok(Json(crate::catalog::show(&state, &q.m).await?))
}

async fn catalog_import(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::CatalogImportRequest>,
) -> Result<Json<cat::CatalogImportResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::catalog::import(&state, &req).await?))
}

async fn catalog_export(State(state): State<Arc<AppState>>) -> Result<Json<serde_json::Value>, M5cError> {
    Ok(Json(crate::catalog::export(&state).await?))
}

async fn catalog_add(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::ModelAddRequest>,
) -> Result<Json<cat::ModelWriteResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::catalog::add(&state, &req).await?))
}

async fn catalog_edit(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::ModelEditRequest>,
) -> Result<Json<cat::ModelWriteResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::catalog::edit(&state, &req).await?))
}

async fn catalog_enable(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::ModelEnableRequest>,
) -> Result<Json<cat::ModelWriteResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::catalog::enable(&state, &req).await?))
}

async fn catalog_delete(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::ModelDeleteRequest>,
) -> Result<Json<cat::ModelDeleteResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::catalog::delete(&state, &req).await?))
}

async fn history_import(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::HistoryImportRequest>,
) -> Result<Json<cat::HistoryImportResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::history::import(&state, &req).await?))
}

async fn history_list(State(state): State<Arc<AppState>>) -> Result<Json<cat::HistoryListResponse>, M5cError> {
    Ok(Json(crate::history::list(&state).await?))
}

async fn catalog_scan(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::ScanRequest>,
) -> Result<Json<cat::ScanResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::catalog::scan(&state, &req).await?))
}

impl From<crate::ops::OpsError> for M5cError {
    fn from(e: crate::ops::OpsError) -> Self {
        let (code, status) = e.code_and_status();
        M5cError {
            status: StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            code,
            message: e.to_string(),
            details: e.details(),
        }
    }
}

async fn records_enroll(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::EnrollRequest>,
) -> Result<Json<cat::RecordResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::records::enroll(&state, &req).await?))
}

async fn records_authority(
    State(state): State<Arc<AppState>>,
    Json(req): Json<cat::AuthorityRequest>,
) -> Result<Json<cat::RecordResponse>, M5cError> {
    check_protocol(req.protocol)?;
    Ok(Json(crate::records::set_authority(&state, &req).await?))
}
