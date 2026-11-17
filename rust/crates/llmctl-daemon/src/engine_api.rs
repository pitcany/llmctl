//! Engine protocols on top of the supervisor adapters (M4b step 1).
//!
//! The supervisor (systemd, launchd) says whether the *process* runs; the
//! engine protocol says whether it is ready, which models it can serve, which
//! are loaded right now, and how to unload one. Three protocols cover the
//! day-one engines:
//!
//! - [`EngineProtocol::OpenAi`]: vLLM, llama-server, MTPLX, mlx-vlm,
//!   mlx-serve, Strata, ds4 — `GET /v1/models` lists what the server serves.
//!   No engine-level unload: the model *is* the process.
//! - [`EngineProtocol::Ollama`]: `GET /api/version` (ready), `/api/tags`
//!   (servable), `/api/ps` (loaded); unload = `POST /api/generate` with
//!   `keep_alive: 0`.
//! - [`EngineProtocol::LmStudio`]: `GET /api/v0/models` (servable, with a
//!   per-model `state`); unload = `lms unload <id>` (the REST API has no
//!   stable unload; the CLI is what LM Studio documents).
//!
//! A vendor service hosts many models, so its workload may name the service
//! itself as the model ([`EngineProtocol::service_identity`]: `ollama`,
//! `lmstudio`) — verified whenever the service answers — or one model it
//! must be able to serve.

use std::time::Duration;

use hyper::Method;
use llmctl_core::EngineKind;
use serde_json::{json, Value};

use crate::probe_http::{self, HttpProbe};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineProtocol {
    OpenAi,
    Ollama,
    LmStudio,
}

/// What the engine reports beyond the readiness probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineAnswer {
    /// Readiness + servable identities, in the shape the supervisor
    /// adapters fold into a [`crate::engine::UnitProbe`].
    pub probe: HttpProbe,
    /// Models loaded in memory right now (vendor services); `None` when the
    /// protocol has no such notion or the engine could not say.
    pub loaded: Option<Vec<String>>,
}

impl EngineProtocol {
    pub fn for_kind(kind: &EngineKind) -> Self {
        match kind {
            EngineKind::Ollama => EngineProtocol::Ollama,
            EngineKind::LmStudio => EngineProtocol::LmStudio,
            _ => EngineProtocol::OpenAi,
        }
    }

    /// The model name that means "the service itself" for a vendor
    /// service workload.
    pub fn service_identity(self) -> Option<&'static str> {
        match self {
            EngineProtocol::OpenAi => None,
            EngineProtocol::Ollama => Some("ollama"),
            EngineProtocol::LmStudio => Some("lmstudio"),
        }
    }

    pub fn supports_unload(self) -> bool {
        !matches!(self, EngineProtocol::OpenAi)
    }

    pub fn name(self) -> &'static str {
        match self {
            EngineProtocol::OpenAi => "openai",
            EngineProtocol::Ollama => "ollama",
            EngineProtocol::LmStudio => "lmstudio",
        }
    }
}

/// Probe one engine endpoint.
pub async fn probe(protocol: EngineProtocol, port: u16, timeout: Duration) -> EngineAnswer {
    match protocol {
        EngineProtocol::OpenAi => EngineAnswer {
            probe: probe_http::probe_models(port, timeout).await,
            loaded: None,
        },
        EngineProtocol::Ollama => probe_ollama(port, timeout).await,
        EngineProtocol::LmStudio => probe_lmstudio(port, timeout).await,
    }
}

async fn get_json(port: u16, path: &str, timeout: Duration) -> Result<Value, HttpProbe> {
    match probe_http::request(port, Method::GET, path, None, timeout).await {
        Err(()) => Err(HttpProbe::Unreachable),
        Ok((status, _)) if !(200..300).contains(&status) => Err(HttpProbe::Status(status)),
        Ok((_, body)) => serde_json::from_slice(&body).map_err(|_| HttpProbe::Garbled),
    }
}

async fn probe_ollama(port: u16, timeout: Duration) -> EngineAnswer {
    let fail = |probe| EngineAnswer { probe, loaded: None };
    let version = match get_json(port, "/api/version", timeout).await {
        Ok(v) => v,
        Err(p) => return fail(p),
    };
    if version.get("version").and_then(Value::as_str).is_none() {
        return fail(HttpProbe::Garbled);
    }
    // The service answers: it is ready. Servable = pulled tags.
    let mut ids = vec!["ollama".to_string()];
    if let Ok(tags) = get_json(port, "/api/tags", timeout).await {
        ids.extend(ollama_names(&tags));
    }
    let loaded = get_json(port, "/api/ps", timeout)
        .await
        .ok()
        .map(|ps| ollama_names(&ps));
    EngineAnswer {
        probe: HttpProbe::Models(ids),
        loaded,
    }
}

/// `name`s from Ollama's `{"models":[{"name":...}]}` (`/api/tags`, `/api/ps`).
pub fn ollama_names(v: &Value) -> Vec<String> {
    v.get("models")
        .and_then(Value::as_array)
        .map(|ms| {
            ms.iter()
                .filter_map(|m| m.get("name").or_else(|| m.get("model")))
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

async fn probe_lmstudio(port: u16, timeout: Duration) -> EngineAnswer {
    match get_json(port, "/api/v0/models", timeout).await {
        Ok(v) => match lmstudio_models(&v) {
            Some((servable, loaded)) => {
                let mut ids = vec!["lmstudio".to_string()];
                ids.extend(servable);
                EngineAnswer {
                    probe: HttpProbe::Models(ids),
                    loaded: Some(loaded),
                }
            }
            None => EngineAnswer {
                probe: HttpProbe::Garbled,
                loaded: None,
            },
        },
        Err(p) => EngineAnswer { probe: p, loaded: None },
    }
}

/// (servable ids, loaded ids) from LM Studio's `/api/v0/models`.
pub fn lmstudio_models(v: &Value) -> Option<(Vec<String>, Vec<String>)> {
    let data = v.get("data")?.as_array()?;
    let mut servable = Vec::new();
    let mut loaded = Vec::new();
    for m in data {
        let Some(id) = m.get("id").and_then(Value::as_str) else {
            continue;
        };
        servable.push(id.to_string());
        if m.get("state").and_then(Value::as_str) == Some("loaded") {
            loaded.push(id.to_string());
        }
    }
    Some((servable, loaded))
}

/// Why an unload did not happen.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnloadError {
    #[error("{0} has no engine-level unload (the model is the process; stop the workload)")]
    Unsupported(&'static str),
    #[error("refusing to unload {0:?}: name one model, not the service")]
    ServiceIdentity(String),
    #[error("unload request failed: {0}")]
    Failed(String),
    #[error("{model} is still loaded {waited:?} after the unload request")]
    StillLoaded { model: String, waited: Duration },
}

/// Ask the engine to unload `model`, then confirm through the engine's own
/// loaded list. Unloading a model that is not loaded succeeds (the end state
/// is what was asked for).
pub async fn unload(
    protocol: EngineProtocol,
    port: u16,
    model: &str,
    timeout: Duration,
    confirm_within: Duration,
) -> Result<(), UnloadError> {
    if !protocol.supports_unload() {
        return Err(UnloadError::Unsupported(protocol.name()));
    }
    if model.is_empty() || Some(model) == protocol.service_identity() {
        return Err(UnloadError::ServiceIdentity(model.to_string()));
    }
    match protocol {
        EngineProtocol::Ollama => {
            let body = json!({"model": model, "keep_alive": 0});
            match probe_http::request(port, Method::POST, "/api/generate", Some(&body), timeout)
                .await
            {
                Ok((status, _)) if (200..300).contains(&status) => {}
                Ok((status, body)) => {
                    return Err(UnloadError::Failed(format!(
                        "ollama answered {status}: {}",
                        String::from_utf8_lossy(&body).chars().take(200).collect::<String>()
                    )))
                }
                Err(()) => return Err(UnloadError::Failed("ollama unreachable".into())),
            }
        }
        EngineProtocol::LmStudio => lms_unload(model, timeout).await?,
        EngineProtocol::OpenAi => unreachable!("checked by supports_unload"),
    }
    let deadline = tokio::time::Instant::now() + confirm_within;
    loop {
        let ans = probe(protocol, port, timeout).await;
        match ans.loaded {
            Some(l) if !l.iter().any(|m| m == model) => return Ok(()),
            _ if tokio::time::Instant::now() >= deadline => {
                return Err(UnloadError::StillLoaded {
                    model: model.to_string(),
                    waited: confirm_within,
                })
            }
            _ => tokio::time::sleep(Duration::from_millis(250)).await,
        }
    }
}

/// The `lms` CLI: LM Studio's own install path first, then `PATH`.
fn lms_binary() -> std::path::PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        let p = std::path::Path::new(&home).join(".lmstudio/bin/lms");
        if p.exists() {
            return p;
        }
    }
    "lms".into()
}

async fn lms_unload(model: &str, timeout: Duration) -> Result<(), UnloadError> {
    // Always an explicit identifier: without one `lms unload` prompts.
    let mut cmd = tokio::process::Command::new(lms_binary());
    cmd.args(["unload", model])
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) if out.status.success() => Ok(()),
        Ok(Ok(out)) => {
            let text = String::from_utf8_lossy(&out.stderr);
            // Not loaded is the asked-for end state.
            if text.to_ascii_lowercase().contains("not loaded")
                || text.to_ascii_lowercase().contains("no model")
            {
                Ok(())
            } else {
                Err(UnloadError::Failed(format!(
                    "lms unload exited {}: {}",
                    out.status,
                    text.trim().chars().take(200).collect::<String>()
                )))
            }
        }
        Ok(Err(e)) => Err(UnloadError::Failed(format!("spawning lms: {e}"))),
        Err(_) => Err(UnloadError::Failed(format!("lms unload timed out after {timeout:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A tiny HTTP server: `routes` maps "METHOD /path" to a JSON body; a
    /// POST to /api/generate with keep_alive 0 clears the `ps` list.
    async fn fake(routes: Vec<(&'static str, Value)>) -> (u16, Arc<Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let loaded = Arc::new(Mutex::new(vec!["qwen3:8b".to_string()]));
        let l2 = loaded.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = vec![0u8; 8192];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let line = req.lines().next().unwrap_or("").to_string();
                let key: String = line.split_whitespace().take(2).collect::<Vec<_>>().join(" ");
                let body = if key == "POST /api/generate" {
                    if req.contains("\"keep_alive\":0") {
                        l2.lock().unwrap().clear();
                    }
                    Some(json!({"done": true}))
                } else if key == "GET /api/ps" {
                    let names: Vec<Value> = l2
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|n| json!({"name": n}))
                        .collect();
                    Some(json!({ "models": names }))
                } else {
                    routes.iter().find(|(k, _)| *k == key).map(|(_, v)| v.clone())
                };
                let (status, text) = match body {
                    Some(v) => ("200 OK", v.to_string()),
                    None => ("404 Not Found", "{}".to_string()),
                };
                let head = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
                    text.len()
                );
                let _ = sock.write_all(head.as_bytes()).await;
                let _ = sock.write_all(text.as_bytes()).await;
            }
        });
        (port, loaded)
    }

    const T: Duration = Duration::from_secs(2);

    #[test]
    fn protocol_follows_engine_kind() {
        assert_eq!(EngineProtocol::for_kind(&EngineKind::Ollama), EngineProtocol::Ollama);
        assert_eq!(EngineProtocol::for_kind(&EngineKind::LmStudio), EngineProtocol::LmStudio);
        for k in [EngineKind::Vllm, EngineKind::LlamaCpp, EngineKind::Mtplx, EngineKind::Ds4] {
            assert_eq!(EngineProtocol::for_kind(&k), EngineProtocol::OpenAi);
        }
        assert!(!EngineProtocol::OpenAi.supports_unload());
    }

    #[tokio::test]
    async fn ollama_reports_service_tags_and_loaded_and_unloads() {
        let (port, loaded) = fake(vec![
            ("GET /api/version", json!({"version": "0.33.1"})),
            ("GET /api/tags", json!({"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:latest"}]})),
        ])
        .await;
        let ans = probe(EngineProtocol::Ollama, port, T).await;
        assert_eq!(
            ans.probe,
            HttpProbe::Models(vec!["ollama".into(), "qwen3:8b".into(), "llama3.2:latest".into()])
        );
        assert_eq!(ans.loaded, Some(vec!["qwen3:8b".to_string()]));

        // The service identity is refused; a model unloads and is confirmed.
        assert!(matches!(
            unload(EngineProtocol::Ollama, port, "ollama", T, T).await,
            Err(UnloadError::ServiceIdentity(_))
        ));
        unload(EngineProtocol::Ollama, port, "qwen3:8b", T, T).await.unwrap();
        assert!(loaded.lock().unwrap().is_empty());
        // Already unloaded: still Ok.
        unload(EngineProtocol::Ollama, port, "qwen3:8b", T, T).await.unwrap();
    }

    #[tokio::test]
    async fn ollama_without_version_is_garbled_and_dead_port_unreachable() {
        let (port, _) = fake(vec![("GET /api/version", json!({"nope": 1}))]).await;
        assert_eq!(probe(EngineProtocol::Ollama, port, T).await.probe, HttpProbe::Garbled);
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let dead = free.local_addr().unwrap().port();
        drop(free);
        assert_eq!(
            probe(EngineProtocol::Ollama, dead, Duration::from_millis(300)).await.probe,
            HttpProbe::Unreachable
        );
    }

    #[tokio::test]
    async fn lmstudio_lists_servable_and_loaded_by_state() {
        let (port, _) = fake(vec![(
            "GET /api/v0/models",
            json!({"data": [
                {"id": "qwen3-coder", "state": "not-loaded"},
                {"id": "text-embedding-nomic-embed-text-v1.5", "state": "loaded"}
            ]}),
        )])
        .await;
        let ans = probe(EngineProtocol::LmStudio, port, T).await;
        assert_eq!(
            ans.probe,
            HttpProbe::Models(vec![
                "lmstudio".into(),
                "qwen3-coder".into(),
                "text-embedding-nomic-embed-text-v1.5".into()
            ])
        );
        assert_eq!(ans.loaded, Some(vec!["text-embedding-nomic-embed-text-v1.5".to_string()]));
        // An older server without /api/v0 → a status, not a guess.
        let (old, _) = fake(vec![]).await;
        assert_eq!(probe(EngineProtocol::LmStudio, old, T).await.probe, HttpProbe::Status(404));
    }

    #[tokio::test]
    async fn openai_engines_have_no_unload() {
        assert_eq!(
            unload(EngineProtocol::OpenAi, 1, "m", T, T).await,
            Err(UnloadError::Unsupported("openai"))
        );
    }
}
