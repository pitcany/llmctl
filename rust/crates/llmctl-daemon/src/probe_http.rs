//! HTTP endpoint probing shared by the systemd and launchd adapters:
//! OpenAI-style `/v1/models` identity checks plus a plain `/health`
//! readiness check (llama-swap answers `OK` there).

use std::time::Duration;

use http_body_util::{BodyExt, Empty, Limited};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;

const MAX_MODELS_BODY: usize = 1 << 20;

/// What `GET /v1/models` said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpProbe {
    /// Nothing listening (or connect timed out): the server is not up yet.
    Unreachable,
    /// Answered with a non-success status (503 while loading, ...).
    Status(u16),
    /// 2xx with an OpenAI-style model list.
    Models(Vec<String>),
    /// 2xx but the body is not a model list.
    Garbled,
}

/// What `GET /health` said: any 2xx means the workload considers itself
/// ready (M3-B.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthProbe {
    /// 2xx answer.
    Healthy,
    /// Not answering, non-2xx, or garbled.
    NotHealthy,
}

/// `GET <path>` on the workload's port and read one JSON string field —
/// the vendor-app version probe (Ollama `/api/version` → `version`).
/// `None` for any transport failure, non-2xx, missing field or non-string
/// value: an app that cannot be observed reports nothing (M4a).
pub async fn fetch_json_field(port: u16, path: &str, field: &str, timeout: Duration) -> Option<String> {
    let out = tokio::time::timeout(timeout, fetch_json_body(port, path)).await;
    match out {
        Ok(Some(v)) => v.get(field)?.as_str().map(str::to_owned),
        _ => None,
    }
}

async fn fetch_json_body(port: u16, path: &str) -> Option<serde_json::Value> {
    let stream = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.ok()?;
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .ok()?;
    let conn = tokio::spawn(conn);
    let req = hyper::Request::get(path)
        .header(hyper::header::HOST, format!("127.0.0.1:{port}"))
        .body(Empty::<Bytes>::new())
        .expect("static request");
    let resp = sender.send_request(req).await.ok()?;
    if !(200..300).contains(&resp.status().as_u16()) {
        conn.abort();
        return None;
    }
    let bytes = Limited::new(resp.into_body(), MAX_MODELS_BODY)
        .collect()
        .await
        .ok()?
        .to_bytes();
    conn.abort();
    serde_json::from_slice(&bytes).ok()
}

/// One HTTP/1.1 request to `127.0.0.1:<port>` with an optional JSON body:
/// the response status and (size-capped) body, or `Err(())` for any
/// transport failure. The engine protocols (M4b) build on this.
pub async fn request(
    port: u16,
    method: hyper::Method,
    path: &str,
    json_body: Option<&serde_json::Value>,
    timeout: Duration,
) -> Result<(u16, Bytes), ()> {
    let fut = async {
        let stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .map_err(|_| ())?;
        let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
            .await
            .map_err(|_| ())?;
        let conn = tokio::spawn(conn);
        let body = json_body
            .map(|v| Bytes::from(serde_json::to_vec(v).expect("json serializes")))
            .unwrap_or_default();
        let mut req = hyper::Request::builder()
            .method(method)
            .uri(path)
            .header(hyper::header::HOST, format!("127.0.0.1:{port}"));
        if json_body.is_some() {
            req = req.header(hyper::header::CONTENT_TYPE, "application/json");
        }
        let req = req
            .body(http_body_util::Full::new(body))
            .map_err(|_| ())?;
        let resp = sender.send_request(req).await.map_err(|_| ())?;
        let status = resp.status().as_u16();
        let bytes = Limited::new(resp.into_body(), MAX_MODELS_BODY)
            .collect()
            .await
            .map_err(|_| ())?
            .to_bytes();
        conn.abort();
        Ok((status, bytes))
    };
    tokio::time::timeout(timeout, fut).await.unwrap_or(Err(()))
}

pub async fn probe_models(port: u16, timeout: Duration) -> HttpProbe {
    match tokio::time::timeout(timeout, fetch_models(port)).await {
        Ok(p) => p,
        Err(_) => HttpProbe::Unreachable,
    }
}

pub async fn probe_health(port: u16, timeout: Duration) -> HealthProbe {
    let got = tokio::time::timeout(timeout, fetch_path(port, "/health")).await;
    match got {
        Ok(Ok(status)) if (200..300).contains(&status) => HealthProbe::Healthy,
        _ => HealthProbe::NotHealthy,
    }
}

async fn fetch_models(port: u16) -> HttpProbe {
    match fetch_path(port, "/v1/models").await {
        Err(()) => HttpProbe::Unreachable,
        Ok(status) if !(200..300).contains(&status) => HttpProbe::Status(status),
        Ok(_) => match fetch_models_body(port).await {
            Some(ids) => HttpProbe::Models(ids),
            None => HttpProbe::Garbled,
        },
    }
}

/// Issue `GET <path>` and return the response status, or `Err(())` for any
/// transport-level failure (connect refused, timeout, closed).
async fn fetch_path(port: u16, path: &str) -> Result<u16, ()> {
    let Ok(stream) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await else {
        return Err(());
    };
    let Ok((mut sender, conn)) = hyper::client::conn::http1::handshake(TokioIo::new(stream)).await
    else {
        return Err(());
    };
    let conn = tokio::spawn(conn);
    let req = hyper::Request::get(path)
        .header(hyper::header::HOST, format!("127.0.0.1:{port}"))
        .body(Empty::<Bytes>::new())
        .expect("static request");
    let result = match sender.send_request(req).await {
        Err(_) => Err(()),
        Ok(resp) => Ok(resp.status().as_u16()),
    };
    conn.abort();
    result
}

/// Second request that also reads the body (hyper needs the body drained;
/// a fresh connection keeps this simple and the probe is rare).
async fn fetch_models_body(port: u16) -> Option<Vec<String>> {
    let stream = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.ok()?;
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream)).await.ok()?;
    let conn = tokio::spawn(conn);
    let req = hyper::Request::get("/v1/models")
        .header(hyper::header::HOST, format!("127.0.0.1:{port}"))
        .body(Empty::<Bytes>::new())
        .expect("static request");
    let resp = sender.send_request(req).await.ok()?;
    let bytes = Limited::new(resp.into_body(), MAX_MODELS_BODY)
        .collect()
        .await
        .ok()?
        .to_bytes();
    conn.abort();
    parse_models(&bytes)
}

/// Model ids from an OpenAI-style `{"data":[{"id":...}]}` body.
pub fn parse_models(body: &[u8]) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let data = v.get("data")?.as_array()?;
    Some(
        data.iter()
            .filter_map(|m| m.get("id")?.as_str().map(str::to_owned))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_models_reads_openai_list() {
        let body = br#"{"object":"list","data":[{"id":"qwen/a"},{"id":"qwen/b"}]}"#;
        assert_eq!(parse_models(body).unwrap(), vec!["qwen/a", "qwen/b"]);
        assert!(parse_models(b"<html>").is_none());
        assert!(parse_models(br#"{"models":[]}"#).is_none());
    }

    #[tokio::test]
    async fn probes_hit_a_real_listener() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let req = String::from_utf8_lossy(&buf);
                if req.starts_with("GET /health") {
                    let head = "HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n";
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(b"OK").await;
                } else {
                    let body = br#"{"data":[{"id":"spike/model"}]}"#;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
                        body.len()
                    );
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(body).await;
                }
            }
        });
        assert_eq!(
            probe_models(port, Duration::from_secs(2)).await,
            HttpProbe::Models(vec!["spike/model".into()])
        );
        assert_eq!(
            probe_health(port, Duration::from_secs(2)).await,
            HealthProbe::Healthy
        );

        // Nothing listening on a freshly released port -> unreachable.
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let dead_port = free.local_addr().unwrap().port();
        drop(free);
        assert_eq!(
            probe_models(dead_port, Duration::from_millis(500)).await,
            HttpProbe::Unreachable
        );
        assert_eq!(
            probe_health(dead_port, Duration::from_millis(500)).await,
            HealthProbe::NotHealthy
        );
    }
}
