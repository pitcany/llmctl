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
    AckRequest, AdmitRequest, AdmitResponse, ApiError, CancelRequest, HealthResponse,
    OperationListResponse, OperationView, PlanRequest, PlanResponse, StatusResponse,
    PROTOCOL_VERSION,
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
    path: Arc<PathBuf>,
}

impl ControlClient {
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
            path: Arc::new(path),
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.path
    }

    pub async fn health(&self) -> Result<HealthResponse, ClientError> {
        self.get_json("/v1/health").await
    }

    pub async fn status(&self) -> Result<StatusResponse, ClientError> {
        self.get_json("/v1/status").await
    }

    pub async fn plan(&self, req: PlanRequest) -> Result<PlanResponse, ClientError> {
        self.post_json("/v1/plans", &req).await
    }

    pub async fn admit(&self, req: AdmitRequest) -> Result<AdmitResponse, ClientError> {
        self.post_json("/v1/operations", &req).await
    }

    pub async fn list_operations(&self) -> Result<OperationListResponse, ClientError> {
        self.get_json("/v1/operations").await
    }

    pub async fn operation(&self, id: &str) -> Result<OperationView, ClientError> {
        self.get_json(&format!("/v1/operations/{id}")).await
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
        let response = self
            .http
            .request(request)
            .await
            .map_err(|e| classify_transport(&self.path, &e))?;
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
