//! HTTP/1.1 client over a Unix-domain socket (hyper with a custom
//! connector). The socket path is the only address; no TCP fallback.

use std::error::Error as StdError;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::header::{CONTENT_TYPE, HOST};
use hyper::{Method, Request, Uri};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thiserror::Error;
use tower::Service;

use crate::wire::{
    AckRequest, AdmitRequest, AdmitResponse, AdoptRequest, AdoptResponse, AdoptionPreview,
    ApiError, CancelRequest, GcResponse, HealthResponse, LogsRequest, LogsResponse,
    OperationListResponse, OperationView, OwnRequest, OwnResponse, PlanRequest, PlanResponse,
    PromoteRequest, ReleaseRequest, ReleaseResponse, RollbackRequest, StageRequest, StageResponse,
    StatusResponse, UpgradeListResponse, PROTOCOL_VERSION,
};

#[derive(Debug, Error)]
pub enum ClientError {
    /// No daemon answering on the socket. Callers must not fall back to an
    /// independent executor; surface this distinctly.
    #[error("daemon not reachable at {path}: {source}")]
    Unreachable {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("connection lost mid-request: {0}")]
    Disconnected(String),
    #[error("api error ({status}) {code}: {message}")]
    Api {
        status: u16,
        code: String,
        message: String,
        details: Option<serde_json::Value>,
    },
    #[error("malformed response: {0}")]
    BadResponse(String),
}

impl ClientError {
    pub fn api_code(&self) -> Option<&str> {
        match self {
            ClientError::Api { code, .. } => Some(code),
            _ => None,
        }
    }
}

/// Connector that dials the fixed Unix socket for every request.
#[derive(Clone)]
struct UnixConnector {
    path: Arc<PathBuf>,
}

impl Service<Uri> for UnixConnector {
    type Response = hyper_util::rt::TokioIo<tokio::net::UnixStream>;
    type Error = std::io::Error;
    type Future = Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        hyper_util::rt::TokioIo<tokio::net::UnixStream>,
                        std::io::Error,
                    >,
                > + Send,
        >,
    >;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _uri: Uri) -> Self::Future {
        let path = self.path.clone();
        Box::pin(async move {
            let stream = tokio::net::UnixStream::connect(&*path).await?;
            Ok(hyper_util::rt::TokioIo::new(stream))
        })
    }
}

#[derive(Clone)]
pub struct ControlClient {
    http: Client<UnixConnector, Full<Bytes>>,
    bridge: Option<Client<crate::bridge::BridgeConnector, Full<Bytes>>>,
    path: Arc<PathBuf>,
    /// Sent as [`crate::lease::UNDER_LEASE_HEADER`]: this caller runs
    /// inside the lane-lease holder with that pid.
    under_lease: Option<u32>,
    /// Sent as [`crate::oplock::LOCK_HOLDER_HEADER`]: this caller holds the
    /// host operation lock with that pid.
    holding_lock: Option<u32>,
}

impl ControlClient {
    /// Tell the agent this caller runs inside the lease holder `pid`, so a
    /// job that drives llmctl-next is not refused by its own lease.
    pub fn under_lease(mut self, pid: u32) -> Self {
        self.under_lease = Some(pid);
        self
    }

    /// Tell the agent the caller `pid` holds the host operation lock and
    /// keeps it until the admitted operation is terminal (llm-router's
    /// handoff). The agent checks the claim against the real holder.
    pub fn holding_lock(mut self, pid: u32) -> Self {
        self.holding_lock = Some(pid);
        self
    }

    pub fn connect(socket_path: impl AsRef<Path>) -> Result<Self, ClientError> {
        let path = socket_path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(ClientError::Unreachable {
                path: path.clone(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "socket file does not exist",
                ),
            });
        }
        let http = Client::builder(TokioExecutor::new()).build(UnixConnector {
            path: Arc::new(path.clone()),
        });
        Ok(Self {
            http,
            bridge: None,
            path: Arc::new(path),
            under_lease: None,
            holding_lock: None,
        })
    }

    /// A client whose transport is an SSH stdio bridge to a remote host's
    /// agent (M3-E.1). No local socket is required or checked; the first
    /// request spawns `ssh`, and a dead pipe re-dials on the next request.
    pub fn connect_bridge(alias: &str) -> Self {
        let bridge = Client::builder(TokioExecutor::new())
            .build(crate::bridge::BridgeConnector {
                alias: Arc::new(alias.to_string()),
            });
        Self {
            http: Client::builder(TokioExecutor::new()).build(UnixConnector {
                path: Arc::new(PathBuf::from(format!("bridge:{alias}"))),
            }),
            bridge: Some(bridge),
            path: Arc::new(PathBuf::from(format!("bridge:{alias}"))),
            under_lease: None,
            holding_lock: None,
        }
    }

    /// True when this client talks to a remote agent over the bridge.
    pub fn is_bridge(&self) -> bool {
        self.bridge.is_some()
    }

    pub fn socket_path(&self) -> &Path {
        &self.path
    }

    async fn request(
        &self,
        request: hyper::Request<Full<Bytes>>,
    ) -> Result<hyper::Response<hyper::body::Incoming>, ClientError> {
        match &self.bridge {
            Some(bridge) => bridge
                .request(request)
                .await
                .map_err(|e| classify_transport(&self.path, &e)),
            None => self
                .http
                .request(request)
                .await
                .map_err(|e| classify_transport(&self.path, &e)),
        }
    }

    pub async fn health(&self) -> Result<HealthResponse, ClientError> {
        self.get_json("/v1/health").await
    }

    pub async fn status(&self) -> Result<StatusResponse, ClientError> {
        self.get_json("/v1/status").await
    }

    /// Declared model stores with their free space.
    pub async fn stores(&self) -> Result<crate::wire::StoresResponse, ClientError> {
        self.get_json("/v1/stores").await
    }

    /// Check, preview and (unless `dry_run`) admit a download into a store.
    pub async fn acquire(
        &self,
        req: &crate::wire::AcquireRequest,
    ) -> Result<crate::wire::AcquireResponse, ClientError> {
        self.post_json("/v1/models/acquire", req).await
    }

    /// Measured VRAM per workload and the cards' free memory.
    pub async fn vram(&self) -> Result<crate::wire::VramResponse, ClientError> {
        self.get_json("/v1/vram").await
    }

    /// Measure a serving workload's VRAM now and remember it.
    pub async fn vram_measure(&self, workload: &str) -> Result<crate::wire::VramResponse, ClientError> {
        self.post_json(
            "/v1/vram/measure",
            &crate::wire::VramMeasureRequest { workload: workload.to_string() },
        )
        .await
    }

    /// Recorded intents per workload; no engine is probed.
    pub async fn intents(&self) -> Result<crate::wire::IntentsResponse, ClientError> {
        self.get_json("/v1/intents").await
    }

    /// The raw inventory (read-only; adoption/ownership state included).
    pub async fn inventory(&self) -> Result<serde_json::Value, ClientError> {
        self.get_json("/v1/inventory").await
    }

    pub async fn plan(&self, req: PlanRequest) -> Result<PlanResponse, ClientError> {
        self.post_json("/v1/plans", &req).await
    }

    pub async fn admit(&self, req: AdmitRequest) -> Result<AdmitResponse, ClientError> {
        self.post_json("/v1/operations", &req).await
    }

    pub async fn unload(
        &self,
        req: &crate::wire::UnloadRequest,
    ) -> Result<AdmitResponse, ClientError> {
        self.post_json("/v1/unloads", req).await
    }

    pub async fn lanes_import(
        &self,
        req: &crate::wire::LaneImportRequest,
    ) -> Result<crate::wire::LaneImportResponse, ClientError> {
        self.post_json("/v1/lanes/import", req).await
    }

    pub async fn lanes_status(
        &self,
        workload: &str,
    ) -> Result<crate::wire::LaneStatusResponse, ClientError> {
        self.get_json(&format!("/v1/lanes/{workload}")).await
    }

    pub async fn lanes_version(
        &self,
        req: &crate::wire::LaneVersionRequest,
    ) -> Result<crate::wire::LaneVersionResponse, ClientError> {
        self.post_json("/v1/lanes/version", req).await
    }

    pub async fn lanes_swap(
        &self,
        req: &crate::wire::LaneSwapRequest,
    ) -> Result<crate::wire::LaneSwapResponse, ClientError> {
        self.post_json("/v1/lane-swaps", req).await
    }

    pub async fn lanes_plan(
        &self,
        req: &crate::wire::LaneEditRequest,
    ) -> Result<crate::wire::LanePlanResponse, ClientError> {
        self.post_json("/v1/lanes/edit", req).await
    }

    pub async fn lanes_apply(
        &self,
        req: &crate::wire::LaneEditRequest,
    ) -> Result<AdmitResponse, ClientError> {
        self.post_json("/v1/lanes/edit", req).await
    }

    pub async fn engine_report(
        &self,
        workload: &str,
    ) -> Result<crate::wire::EngineReport, ClientError> {
        self.get_json(&format!("/v1/workloads/{workload}/engine")).await
    }

    pub async fn adoption_preview(
        &self,
        req: &AdoptRequest,
    ) -> Result<AdoptionPreview, ClientError> {
        self.post_json("/v1/adoptions/preview", req).await
    }

    pub async fn adopt(&self, req: &AdoptRequest) -> Result<AdoptResponse, ClientError> {
        self.post_json("/v1/adoptions", req).await
    }

    pub async fn own(&self, workload: &str) -> Result<OwnResponse, ClientError> {
        self.post_json(
            "/v1/ownership/own",
            &OwnRequest {
                workload: workload.into(),
            },
        )
        .await
    }

    pub async fn release(&self, workload: &str) -> Result<ReleaseResponse, ClientError> {
        self.post_json(
            "/v1/ownership/release",
            &ReleaseRequest {
                workload: workload.into(),
            },
        )
        .await
    }

    pub async fn stage(&self, workload: &str, version: &str) -> Result<StageResponse, ClientError> {
        self.post_json(
            "/v1/upgrades/stage",
            &StageRequest {
                workload: workload.into(),
                version: version.into(),
            },
        )
        .await
    }

    /// Returns the admitted operation view (terminal when `wait`).
    pub async fn promote(
        &self,
        workload: &str,
        version: &str,
        wait: bool,
    ) -> Result<serde_json::Value, ClientError> {
        self.post_json(
            "/v1/upgrades/promote",
            &PromoteRequest {
                workload: workload.into(),
                version: version.into(),
                wait,
            },
        )
        .await
    }

    pub async fn rollback_upgrade(
        &self,
        workload: &str,
        wait: bool,
    ) -> Result<serde_json::Value, ClientError> {
        self.post_json(
            "/v1/upgrades/rollback",
            &RollbackRequest {
                workload: workload.into(),
                wait,
            },
        )
        .await
    }

    pub async fn list_upgrades(&self, workload: &str) -> Result<UpgradeListResponse, ClientError> {
        self.post_json("/v1/upgrades", &serde_json::json!({ "workload": workload }))
            .await
    }

    pub async fn gc_upgrades(&self, workload: &str) -> Result<GcResponse, ClientError> {
        self.post_json(
            "/v1/upgrades/gc",
            &serde_json::json!({ "workload": workload }),
        )
        .await
    }

    /// The vendor-app plan: observed/pinned version, drift, owner-run
    /// backup/reinstall steps (M4a). Never executes anything.
    pub async fn install_plan(&self, workload: &str) -> Result<serde_json::Value, ClientError> {
        self.post_json(
            "/v1/installs/plan",
            &crate::wire::InstallPlanRequest {
                workload: workload.into(),
            },
        )
        .await
    }

    /// Version facts of one workload: pin, live probe, drift (M4a).
    pub async fn version_report(&self, workload: &str) -> Result<serde_json::Value, ClientError> {
        self.get_json(&format!("/v1/workloads/{workload}/version"))
            .await
    }

    /// Record what the install reports now as the workload's pin.
    pub async fn version_repin(&self, workload: &str) -> Result<serde_json::Value, ClientError> {
        self.post_json(
            &format!("/v1/workloads/{workload}/version/pin"),
            &serde_json::json!({}),
        )
        .await
    }

    pub async fn logs(&self, req: &LogsRequest) -> Result<LogsResponse, ClientError> {
        self.post_json("/v1/logs", req).await
    }

    pub async fn list_operations(&self) -> Result<OperationListResponse, ClientError> {
        self.get_json("/v1/operations").await
    }

    pub async fn operation(&self, id: &str) -> Result<OperationView, ClientError> {
        self.get_json(&format!("/v1/operations/{id}")).await
    }

    /// Resolve an idempotency key to its operation — the lost-ACK check
    /// (M3 decision 4): after a transport error, this answers whether the
    /// admit landed without issuing anything.
    pub async fn operation_by_key(&self, key: &str) -> Result<OperationView, ClientError> {
        // Keys are validated to [A-Za-z0-9._-] — URI-path safe by construction.
        self.get_json(&format!("/v1/operations/by-key/{key}")).await
    }

    pub async fn cancel(
        &self,
        id: &str,
        reason: Option<String>,
    ) -> Result<OperationView, ClientError> {
        self.post_json(
            &format!("/v1/operations/{id}/cancel"),
            &CancelRequest { reason },
        )
        .await
    }

    pub async fn acknowledge(
        &self,
        id: &str,
        note: Option<String>,
    ) -> Result<OperationView, ClientError> {
        self.post_json(
            &format!("/v1/operations/{id}/acknowledge"),
            &AckRequest { note },
        )
        .await
    }

    pub async fn send_raw(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
    ) -> Result<serde_json::Value, ClientError> {
        let mut builder = Request::builder()
            .method(method)
            .uri(format!("http://local.control{path}"))
            .header(HOST, "local.control")
            .header("llmctl-protocol", PROTOCOL_VERSION.to_string());
        if let Some(pid) = self.under_lease {
            builder = builder.header(crate::lease::UNDER_LEASE_HEADER, pid.to_string());
        }
        if let Some(pid) = self.holding_lock {
            builder = builder.header(crate::oplock::LOCK_HOLDER_HEADER, pid.to_string());
        }
        let request = match body {
            Some(payload) => {
                builder = builder.header(CONTENT_TYPE, "application/json");
                builder
                    .body(Full::new(Bytes::from(payload)))
                    .map_err(|e| ClientError::BadResponse(format!("request build failed: {e}")))?
            }
            None => builder
                .body(Full::default())
                .map_err(|e| ClientError::BadResponse(format!("request build failed: {e}")))?,
        };
        let response = self.request(request).await?;
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .map_err(|e| ClientError::Disconnected(format!("body read failed: {e}")))?
            .to_bytes();
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| ClientError::BadResponse(e.to_string()))?;
        if !status.is_success() {
            let api: Option<ApiError> = serde_json::from_value(value.clone()).ok();
            return Err(match api {
                Some(ApiError { error }) => ClientError::Api {
                    status: status.as_u16(),
                    code: error.code,
                    message: error.message,
                    details: error.details,
                },
                None => ClientError::BadResponse(format!(
                    "status {status} with non-error body: {value}"
                )),
            });
        }
        Ok(value)
    }

    async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T, ClientError> {
        let value = self.send_raw(Method::GET, path, None).await?;
        serde_json::from_value(value).map_err(|e| ClientError::BadResponse(e.to_string()))
    }

    async fn post_json<T: DeserializeOwned, B: Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, ClientError> {
        let payload = serde_json::to_vec(body)
            .map_err(|e| ClientError::BadResponse(format!("serialize failed: {e}")))?;
        let value = self.send_raw(Method::POST, path, Some(payload)).await?;
        serde_json::from_value(value).map_err(|e| ClientError::BadResponse(e.to_string()))
    }
}

fn classify_transport(path: &Path, e: &hyper_util::client::legacy::Error) -> ClientError {
    if e.is_connect() {
        if let Some(source) = e.source() {
            if let Some(io) = source.downcast_ref::<std::io::Error>() {
                return ClientError::Unreachable {
                    path: path.to_path_buf(),
                    source: std::io::Error::new(io.kind(), io.to_string()),
                };
            }
        }
    }
    ClientError::Disconnected(e.to_string())
}
