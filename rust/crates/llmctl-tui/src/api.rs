//! The agent calls the TUI makes, as a trait so tests run the screens
//! against a fake agent. The real implementation is a thin wrapper over
//! [`ControlClient`] — the same client, endpoints and request shapes the
//! CLI uses. The TUI never probes an engine or runs a command itself
//! (M5 decision 5).

use std::future::Future;
use std::pin::Pin;

use llmctl_client::catalog::{
    CatalogListResponse, ModelAddRequest, ModelEnableRequest, ModelWriteResponse,
    PresetListResponse, PresetRunRequest, PresetRunResponse, ScanRequest, ScanResponse,
};
use llmctl_client::wire::{
    AdmitRequest, AdmitResponse, AuditResponse, EngineReport, HealthResponse, LaneLogsRequest,
    LaneLogsResponse, LaneMemoryRequest, LaneMemoryResponse, LaneStatusResponse,
    LaneSwapRequest, LogsRequest, LogsResponse, OperationListResponse, OperationView,
    PlanRequest, PlanResponse, StatusResponse, VramResponse, PROTOCOL_VERSION,
};
use llmctl_client::{ClientError, ControlClient};

/// One failed call, flattened for display and for the few decisions the
/// TUI makes on an error code (`stale_plan`, `unknown_operation`).
#[derive(Debug, Clone, PartialEq)]
pub struct ApiErr {
    /// The agent's machine-matchable code, when the agent answered.
    pub code: Option<String>,
    pub message: String,
    pub details: Option<serde_json::Value>,
    /// No answer at all (unreachable, connection lost, timed out): the
    /// outcome of a write is unknown, not failed.
    pub transport: bool,
}

impl ApiErr {
    pub fn transport(message: impl Into<String>) -> Self {
        Self { code: None, message: message.into(), details: None, transport: true }
    }

    pub fn api(code: &str, message: impl Into<String>) -> Self {
        Self { code: Some(code.into()), message: message.into(), details: None, transport: false }
    }

    pub fn is(&self, code: &str) -> bool {
        self.code.as_deref() == Some(code)
    }
}

impl std::fmt::Display for ApiErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.code {
            Some(c) => write!(f, "{c}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl From<ClientError> for ApiErr {
    fn from(e: ClientError) -> Self {
        match e {
            ClientError::Api { code, message, details, .. } => {
                Self { code: Some(code), message, details, transport: false }
            }
            ClientError::Unreachable { .. } | ClientError::Disconnected(_) => {
                Self::transport(e.to_string())
            }
            ClientError::BadResponse(m) => {
                Self { code: None, message: format!("malformed response: {m}"), details: None, transport: false }
            }
        }
    }
}

pub type ApiResult<T> = Result<T, ApiErr>;
pub type BoxFut<'a, T> = Pin<Box<dyn Future<Output = ApiResult<T>> + Send + 'a>>;

/// The agent's code for "this preset's profile env does not render what
/// the preset says" — the Run screen offers the sync on exactly this.
pub const DRIFT_CODE: &str = llmctl_client::catalog::codes::PRESET_DRIFT;

/// Everything the screens read or write. Reads are side-effect free on the
/// agent; the three writes (`admit`, `cancel`, `acknowledge`) are the same
/// admitted operations the CLI issues.
pub trait AgentApi: Send + Sync {
    fn health(&self) -> BoxFut<'_, HealthResponse>;
    fn status(&self) -> BoxFut<'_, StatusResponse>;
    fn operations(&self) -> BoxFut<'_, OperationListResponse>;
    fn operation(&self, id: String) -> BoxFut<'_, OperationView>;
    fn operation_by_key(&self, key: String) -> BoxFut<'_, OperationView>;
    fn logs(&self, req: LogsRequest) -> BoxFut<'_, LogsResponse>;
    fn lane_logs(&self, req: LaneLogsRequest) -> BoxFut<'_, LaneLogsResponse>;
    fn vram(&self) -> BoxFut<'_, VramResponse>;
    fn audit(&self) -> BoxFut<'_, AuditResponse>;
    fn lanes_status(&self, workload: String) -> BoxFut<'_, LaneStatusResponse>;
    /// The memory budget and peaks, read only (a request with no setters
    /// changes nothing).
    fn lanes_memory(&self, workload: String) -> BoxFut<'_, LaneMemoryResponse>;
    /// Lanes llama-swap has running now. Asked as a dry-run unload of a
    /// lane the records know (`lane`): it admits nothing and answers with
    /// `running`.
    fn lanes_running(&self, workload: String, lane: String) -> BoxFut<'_, Vec<String>>;
    fn engine_report(&self, workload: String) -> BoxFut<'_, EngineReport>;
    fn plan(&self, req: PlanRequest) -> BoxFut<'_, PlanResponse>;
    fn admit(&self, req: AdmitRequest) -> BoxFut<'_, AdmitResponse>;
    fn cancel(&self, id: String) -> BoxFut<'_, OperationView>;
    fn acknowledge(&self, id: String) -> BoxFut<'_, OperationView>;
    // ---- presets and the catalog (M5c, wired into the screens in M5e) ----
    /// Every preset with its workload link and drift (read-only).
    fn presets(&self) -> BoxFut<'_, PresetListResponse>;
    /// Resolve a preset for a run. `dry_run` and no `sync` writes nothing:
    /// the read the Run screen's preview starts from (a drifted profile
    /// answers `preset_drift`, which the screen offers to sync).
    fn preset_run(&self, req: PresetRunRequest) -> BoxFut<'_, PresetRunResponse>;
    /// This host's catalog records (every one with `all`).
    fn catalog_models(&self, all: bool) -> BoxFut<'_, CatalogListResponse>;
    /// Enable or disable one record.
    fn model_enable(&self, req: ModelEnableRequest) -> BoxFut<'_, ModelWriteResponse>;
    /// Record one weight the scan found.
    fn model_add(&self, req: ModelAddRequest) -> BoxFut<'_, ModelWriteResponse>;
    /// Walk the declared stores; `add` false changes nothing.
    fn model_scan(&self, req: ScanRequest) -> BoxFut<'_, ScanResponse>;
}

/// The real agent: a local socket or the SSH bridge to an enrolled host.
pub struct ClientApi(pub ControlClient);

macro_rules! call {
    ($e:expr) => {
        Box::pin(async move { $e.await.map_err(ApiErr::from) })
    };
}

impl AgentApi for ClientApi {
    fn health(&self) -> BoxFut<'_, HealthResponse> {
        call!(self.0.health())
    }
    fn status(&self) -> BoxFut<'_, StatusResponse> {
        call!(self.0.status())
    }
    fn operations(&self) -> BoxFut<'_, OperationListResponse> {
        call!(self.0.list_operations())
    }
    fn operation(&self, id: String) -> BoxFut<'_, OperationView> {
        Box::pin(async move { self.0.operation(&id).await.map_err(ApiErr::from) })
    }
    fn operation_by_key(&self, key: String) -> BoxFut<'_, OperationView> {
        Box::pin(async move { self.0.operation_by_key(&key).await.map_err(ApiErr::from) })
    }
    fn logs(&self, req: LogsRequest) -> BoxFut<'_, LogsResponse> {
        Box::pin(async move { self.0.logs(&req).await.map_err(ApiErr::from) })
    }
    fn lane_logs(&self, req: LaneLogsRequest) -> BoxFut<'_, LaneLogsResponse> {
        Box::pin(async move { self.0.lanes_logs(&req).await.map_err(ApiErr::from) })
    }
    fn vram(&self) -> BoxFut<'_, VramResponse> {
        call!(self.0.vram())
    }
    fn audit(&self) -> BoxFut<'_, AuditResponse> {
        call!(self.0.audit(None))
    }
    fn lanes_status(&self, workload: String) -> BoxFut<'_, LaneStatusResponse> {
        Box::pin(async move { self.0.lanes_status(&workload).await.map_err(ApiErr::from) })
    }
    fn lanes_memory(&self, workload: String) -> BoxFut<'_, LaneMemoryResponse> {
        Box::pin(async move {
            let req = LaneMemoryRequest {
                protocol: PROTOCOL_VERSION,
                workload,
                budget_gib: None,
                peaks: Vec::new(),
                from_registry: None,
            };
            self.0.lanes_memory(&req).await.map_err(ApiErr::from)
        })
    }
    fn lanes_running(&self, workload: String, lane: String) -> BoxFut<'_, Vec<String>> {
        Box::pin(async move {
            let req = LaneSwapRequest {
                protocol: PROTOCOL_VERSION,
                workload,
                lane,
                dry_run: true,
                force: false,
                idempotency_key: None,
                unload: true,
            };
            self.0.lanes_swap(&req).await.map(|r| r.running).map_err(ApiErr::from)
        })
    }
    fn engine_report(&self, workload: String) -> BoxFut<'_, EngineReport> {
        Box::pin(async move { self.0.engine_report(&workload).await.map_err(ApiErr::from) })
    }
    fn plan(&self, req: PlanRequest) -> BoxFut<'_, PlanResponse> {
        call!(self.0.plan(req))
    }
    fn admit(&self, req: AdmitRequest) -> BoxFut<'_, AdmitResponse> {
        call!(self.0.admit(req))
    }
    fn cancel(&self, id: String) -> BoxFut<'_, OperationView> {
        Box::pin(async move { self.0.cancel(&id, None).await.map_err(ApiErr::from) })
    }
    fn acknowledge(&self, id: String) -> BoxFut<'_, OperationView> {
        Box::pin(async move { self.0.acknowledge(&id, None).await.map_err(ApiErr::from) })
    }
    fn presets(&self) -> BoxFut<'_, PresetListResponse> {
        call!(self.0.presets())
    }
    fn preset_run(&self, req: PresetRunRequest) -> BoxFut<'_, PresetRunResponse> {
        call!(self.0.preset_run(&req))
    }
    fn catalog_models(&self, all: bool) -> BoxFut<'_, CatalogListResponse> {
        call!(self.0.catalog_models(all))
    }
    fn model_enable(&self, req: ModelEnableRequest) -> BoxFut<'_, ModelWriteResponse> {
        call!(self.0.model_enable(&req))
    }
    fn model_add(&self, req: ModelAddRequest) -> BoxFut<'_, ModelWriteResponse> {
        call!(self.0.model_add(&req))
    }
    fn model_scan(&self, req: ScanRequest) -> BoxFut<'_, ScanResponse> {
        call!(self.0.model_scan(&req))
    }
}

/// An agent that cannot be reached at all (no local socket): every call
/// fails at once, so the host shows STALE instead of being left out.
pub struct Unreachable(pub String);

macro_rules! down {
    ($($name:ident($($arg:ident: $t:ty),*) -> $r:ty;)*) => {
        $(fn $name(&self, $($arg: $t),*) -> BoxFut<'_, $r> {
            $(let _ = $arg;)*
            let e = ApiErr::transport(self.0.clone());
            Box::pin(async move { Err(e) })
        })*
    };
}

impl AgentApi for Unreachable {
    down! {
        health() -> HealthResponse;
        status() -> StatusResponse;
        operations() -> OperationListResponse;
        operation(id: String) -> OperationView;
        operation_by_key(key: String) -> OperationView;
        logs(req: LogsRequest) -> LogsResponse;
        lane_logs(req: LaneLogsRequest) -> LaneLogsResponse;
        vram() -> VramResponse;
        audit() -> AuditResponse;
        lanes_status(workload: String) -> LaneStatusResponse;
        lanes_memory(workload: String) -> LaneMemoryResponse;
        lanes_running(workload: String, lane: String) -> Vec<String>;
        engine_report(workload: String) -> EngineReport;
        plan(req: PlanRequest) -> PlanResponse;
        admit(req: AdmitRequest) -> AdmitResponse;
        cancel(id: String) -> OperationView;
        acknowledge(id: String) -> OperationView;
        presets() -> PresetListResponse;
        preset_run(req: PresetRunRequest) -> PresetRunResponse;
        catalog_models(all: bool) -> CatalogListResponse;
        model_enable(req: ModelEnableRequest) -> ModelWriteResponse;
        model_add(req: ModelAddRequest) -> ModelWriteResponse;
        model_scan(req: ScanRequest) -> ScanResponse;
    }
}
