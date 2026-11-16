//! M2 systemd adapter: adopts *existing* systemd units listed in the
//! inventory with `engine: "systemd"`. It only ever touches units that the
//! inventory routes to it, so the installed stack (vllm-engine, ollama…)
//! stays out of reach unless someone enrolls it on purpose.
//!
//! - `observe`/`probe`: `systemctl show` for process state, then
//!   `GET http://127.0.0.1:<port>/v1/models` for readiness and the served
//!   model identity.
//! - `apply`: `systemctl start|stop` on user units. The unit file is the
//!   configuration authority (adoption); the effect does not rewrite it.
//!   System-scope mutation needs a privilege decision and is refused.
//!
//! A manager that cannot be asked yields `ProcessState::Unknown`, never
//! "stopped". A start/stop that times out is `EngineError::Unconfirmed`: the
//! effect may have happened and reconciliation must look.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use http_body_util::{BodyExt, Empty, Limited};
use hyper::body::Bytes;
use hyper_util::rt::TokioIo;
use tokio::process::Command;

use llmctl_core::{
    HostInventory, Observation, ProcessState, Readiness, ServedModel, UnitName, UnitScopeMarker,
};

use crate::engine::{BoxFut, Effect, EffectOutcome, Engine, EngineError, UnitProbe};

/// Inventory engine name that routes a workload to this adapter.
pub const ENGINE_NAME: &str = "systemd";

const MAX_MODELS_BODY: usize = 1 << 20;

#[derive(Debug, Clone)]
struct Endpoint {
    port: u16,
    model: String,
}

#[derive(Debug, Clone)]
pub struct SystemdEngine {
    /// Keyed by `UnitName` display (`user/foo.service`); shared by clones
    /// and replaced on inventory reload.
    endpoints: Arc<std::sync::RwLock<BTreeMap<String, Endpoint>>>,
    show_timeout: Duration,
    effect_timeout: Duration,
    http_timeout: Duration,
    gpu_probe_timeout: Duration,
}

impl SystemdEngine {
    /// Knows exactly the inventory workloads routed to `systemd`.
    pub fn from_inventory(inventory: &HostInventory) -> Self {
        Self {
            endpoints: Arc::new(std::sync::RwLock::new(Self::endpoints_for(inventory))),
            show_timeout: Duration::from_secs(5),
            // systemd's default stop timeout is 90 s; stay just above it.
            effect_timeout: systemd_effect_timeout(),
            http_timeout: Duration::from_secs(2),
            gpu_probe_timeout: Duration::from_secs(10),
        }
    }

    fn endpoints_for(inventory: &HostInventory) -> BTreeMap<String, Endpoint> {
        inventory
            .workloads
            .iter()
            .filter(|w| w.engine == ENGINE_NAME)
            .map(|w| {
                (
                    w.unit.to_string(),
                    Endpoint {
                        port: w.port,
                        model: w.model.clone(),
                    },
                )
            })
            .collect()
    }

    /// Manager/effect timeout used for imports.
    pub fn show_timeout(&self) -> Duration {
        self.show_timeout
    }

    /// Probe a unit that is not (yet) enrolled — adoption's served-identity
    /// check. Read-only: no effect is ever applied to an unenrolled unit.
    pub async fn probe_unenrolled(&self, unit: &UnitName, port: u16, model: &str) -> UnitProbe {
        let ep = Endpoint {
            port,
            model: model.to_string(),
        };
        self.probe_endpoint(unit, &ep).await
    }

    /// Shorter manager/effect timeouts (tests).
    pub fn with_timeouts(mut self, show: Duration, effect: Duration, http: Duration) -> Self {
        self.show_timeout = show;
        self.effect_timeout = effect;
        self.http_timeout = http;
        self
    }

    fn endpoint(&self, unit: &UnitName) -> Result<Endpoint, EngineError> {
        self.endpoints
            .read()
            .expect("endpoints lock")
            .get(&unit.to_string())
            .cloned()
            .ok_or_else(|| EngineError::UnknownUnit(unit.to_string()))
    }

    async fn show(&self, unit: &UnitName) -> Result<UnitStatus, EngineError> {
        let out = systemctl(
            unit.scope,
            &[
                "show",
                &unit.name,
                "--property=LoadState,ActiveState,SubState,MainPID,InvocationID,Restart",
            ],
            self.show_timeout,
        )
        .await?;
        if !out.status.success() {
            return Err(EngineError::Io(format!(
                "systemctl show {unit}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(parse_show(&String::from_utf8_lossy(&out.stdout)))
    }

    async fn probe_endpoint(&self, unit: &UnitName, ep: &Endpoint) -> UnitProbe {
        let status = match self.show(unit).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(%unit, error = %e, "systemd manager could not be asked");
                return UnitProbe {
                    process: ProcessState::Unknown,
                    readiness: Readiness::Unknown,
                    served_model: ServedModel::Unknown,
                };
            }
        };
        let http = if status.process() == ProcessState::Running {
            Some(probe_models(ep.port, self.http_timeout).await)
        } else {
            None
        };
        combine(&status, http.as_ref(), &ep.model)
    }

    async fn mutate(&self, unit: &UnitName, verb: &str) -> Result<UnitStatus, EngineError> {
        if unit.scope != UnitScopeMarker::User {
            return Err(EngineError::Unsupported(format!(
                "{verb} {unit}: only user-scope units are managed (system scope needs a privilege decision)"
            )));
        }
        let before = self.show(unit).await?;
        if before.load_state == "not-found" {
            return Err(EngineError::UnknownUnit(format!(
                "{unit} is enrolled but systemd has no such unit"
            )));
        }
        // A start after start-limit-hit is refused by the manager until the
        // failure bookkeeping is cleared. `reset-failed` only resets that
        // bookkeeping — it never starts or stops anything.
        if verb == "start" && before.active_state == "failed" {
            let _ = systemctl(unit.scope, &["reset-failed", &unit.name], self.show_timeout).await;
        }
        let out = match systemctl(unit.scope, &[verb, &unit.name], self.effect_timeout).await {
            Ok(out) => out,
            Err(EngineError::Timeout(msg)) => {
                return Err(EngineError::Unconfirmed(format!(
                    "systemctl {verb} {unit}: {msg}"
                )))
            }
            Err(e) => return Err(e),
        };
        if !out.status.success() {
            return Err(EngineError::Io(format!(
                "systemctl {verb} {unit} exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        // The effect completed; failing to read it back is the
        // effect-then-error window, not a clean failure.
        self.show(unit)
            .await
            .map_err(|e| EngineError::Unconfirmed(format!("{verb} {unit} done, readback: {e}")))
    }
}

/// `LLMCTL_NEXT_SYSTEMD_EFFECT_TIMEOUT_MS` overrides the fixed effect
/// timeout (dev/test hook for stop-timeout drills; default 100 s, just
/// above systemd's 90 s default stop timeout).
fn systemd_effect_timeout() -> Duration {
    std::env::var("LLMCTL_NEXT_SYSTEMD_EFFECT_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or_else(|| Duration::from_secs(100))
}

impl Engine for SystemdEngine {
    fn name(&self) -> &'static str {
        ENGINE_NAME
    }

    fn observe(&self, inventory: &HostInventory) -> BoxFut<Result<Vec<Observation>, EngineError>> {
        let this = self.clone();
        let inventory = inventory.clone();
        Box::pin(async move {
            let mut out = Vec::with_capacity(inventory.workloads.len());
            for def in &inventory.workloads {
                let ep = Endpoint {
                    port: def.port,
                    model: def.model.clone(),
                };
                let probe = this.probe_endpoint(&def.unit, &ep).await;
                out.push(Observation {
                    workload: def.name.clone(),
                    unit: Some(def.unit.clone()),
                    gpus: if probe.process == ProcessState::Running {
                        def.gpus.clone()
                    } else {
                        Vec::new()
                    },
                    process: probe.process,
                    readiness: probe.readiness,
                    served_model: probe.served_model,
                    observed_at: Utc::now(),
                });
            }
            Ok(out)
        })
    }

    fn apply(&self, effect: Effect) -> BoxFut<Result<EffectOutcome, EngineError>> {
        let this = self.clone();
        Box::pin(async move {
            let unit = effect.unit().clone();
            this.endpoint(&unit)?;
            match &effect {
                Effect::StartUnit { workload, .. } => {
                    let status = this.mutate(&unit, "start").await?;
                    // InvocationID names this run of the unit; starting an
                    // already-active unit is a no-op with the same id.
                    let invocation = status.invocation_id.unwrap_or_else(|| "none".into());
                    Ok(EffectOutcome {
                        identity: format!("started:{unit}:{workload}:{invocation}"),
                    })
                }
                Effect::StopUnit { workload, .. } => {
                    this.mutate(&unit, "stop").await?;
                    Ok(EffectOutcome {
                        identity: format!("stopped:{unit}:{workload}"),
                    })
                }
            }
        })
    }

    fn probe(&self, unit: &UnitName) -> BoxFut<Result<UnitProbe, EngineError>> {
        let this = self.clone();
        let unit = unit.clone();
        Box::pin(async move {
            let ep = this.endpoint(&unit)?;
            Ok(this.probe_endpoint(&unit, &ep).await)
        })
    }

    fn verify_definition(
        &self,
        workload: &llmctl_core::WorkloadDefinition,
    ) -> BoxFut<Result<(), EngineError>> {
        let llmctl_core::Ownership::Managed {
            definition: pinned,
            service_file,
            version: pinned_version,
            definition_schema,
            ..
        } = workload.ownership.clone()
        else {
            return Box::pin(async { Ok(()) });
        };
        let unit = workload.unit.clone();
        let timeout = self.show_timeout;
        let mut wl = workload.clone();
        Box::pin(async move {
            // Cheap short circuit first: once llmctl owns the file, its
            // digest alone answers "did anything touch it".
            if let Some(record) = &service_file {
                let actual = match tokio::fs::read(&record.path).await {
                    Ok(bytes) => {
                        use sha2::Digest as _;
                        format!("sha256:{:x}", sha2::Sha256::digest(bytes))
                    }
                    Err(_) => "missing".to_string(),
                };
                if actual != record.digest {
                    return Err(EngineError::Drift {
                        unit: unit.to_string(),
                        diffs: vec![llmctl_core::FieldDiff {
                            field: "service_file".into(),
                            expected: record.digest.clone().into(),
                            actual: actual.into(),
                        }],
                    });
                }
            }
            let imported = crate::unitdef::import_unit(&unit, timeout).await?;
            let mut diffs = match &imported.definition {
                Some(live) => pinned.diff(live),
                None => vec![llmctl_core::FieldDiff {
                    field: "load_state".into(),
                    expected: "loaded".into(),
                    actual: imported.load_state.clone().into(),
                }],
            };
            if !imported.secret_env_names.is_empty() {
                diffs.push(llmctl_core::FieldDiff {
                    field: "inline_secret_environment".into(),
                    expected: serde_json::Value::Array(vec![]),
                    actual: imported.secret_env_names.clone().into(),
                });
            }
            // Schema 2 pins version facts too: an in-place package change
            // or an edited launcher is drift even though the unit file is
            // byte-identical. The probe hashes the *live* files at the
            // pinned paths and asks the live env what it runs.
            if definition_schema >= 2 {
                if let (Some(pinned_v), Some(recipe)) = (&pinned_version, &wl.recipe) {
                    match crate::version::probe(recipe, &wl).await {
                        Ok(live_v) => {
                            diffs.extend(crate::version::version_drift(pinned_v, &live_v))
                        }
                        Err(e) => {
                            return Err(EngineError::Io(format!(
                                "probing live version facts of {unit}: {e}"
                            )))
                        }
                    }
                }
            }
            if diffs.is_empty() {
                Ok(())
            } else {
                Err(EngineError::Drift {
                    unit: unit.to_string(),
                    diffs,
                })
            }
        })
    }

    fn reload(&self, inventory: &HostInventory) {
        *self.endpoints.write().expect("endpoints lock") = Self::endpoints_for(inventory);
    }

    fn logs(
        &self,
        workload: &llmctl_core::WorkloadDefinition,
        lines: u32,
        after: Option<String>,
    ) -> BoxFut<Result<crate::engine::LogPage, EngineError>> {
        let unit = workload.unit.clone();
        let timeout = self.show_timeout;
        Box::pin(async move { crate::journal::read(&unit, lines, after.as_deref(), timeout).await })
    }

    /// The real cards, through nvidia-smi and `/proc/<pid>/cgroup`.
    fn gpu_occupancy(
        &self,
        gpus: &[llmctl_core::GpuId],
    ) -> BoxFut<Result<Vec<crate::gpu::GpuOccupant>, EngineError>> {
        let wanted: std::collections::BTreeSet<u32> = gpus.iter().map(|g| g.index).collect();
        let timeout = self.gpu_probe_timeout;
        Box::pin(async move {
            let all = crate::gpu::probe_nvidia(timeout).await?;
            Ok(all
                .into_iter()
                .filter(|o| wanted.contains(&o.gpu))
                .collect())
        })
    }

    /// systemd retries first: `activating/auto-restart` means the manager
    /// is on it; llmctl acts only on `failed` (start-limit-hit) or a down
    /// unit with `Restart=no`. A manager that cannot be asked is `Unknown`
    /// — never a licence to recover.
    fn recovery_state(
        &self,
        workload: &llmctl_core::WorkloadDefinition,
    ) -> BoxFut<Result<crate::engine::RecoveryState, EngineError>> {
        let unit = workload.unit.clone();
        let this = self.clone();
        Box::pin(async move {
            let status = match this.show(&unit).await {
                Ok(s) => s,
                Err(_) => return Ok(crate::engine::RecoveryState::Unknown),
            };
            Ok(recovery_state_of(&status))
        })
    }
}

/// Pure mapping from manager state to the recovery decision input.
///
/// `inactive` counts as `GaveUp`, not idle: an inactive unit with
/// `Restart=on-failure` is one the manager is *not* retrying (it exited
/// cleanly, or something stopped it — including a failed run's own
/// rollback). Whether that warrants action is the watcher's intent check,
/// not the manager's.
pub fn recovery_state_of(status: &UnitStatus) -> crate::engine::RecoveryState {
    use crate::engine::RecoveryState;
    if status.load_state == "not-found" {
        return RecoveryState::Unknown;
    }
    match (status.active_state.as_str(), status.sub_state.as_str()) {
        ("activating", "auto-restart") => RecoveryState::ManagerRetrying,
        ("active" | "activating" | "reloading", _) => RecoveryState::Alive,
        ("deactivating", _) => RecoveryState::Alive,
        ("failed", _) | ("inactive", _) => RecoveryState::GaveUp,
        _ => RecoveryState::Unknown,
    }
}

// ---- systemctl -------------------------------------------------------------

async fn systemctl(
    scope: UnitScopeMarker,
    args: &[&str],
    timeout: Duration,
) -> Result<std::process::Output, EngineError> {
    let scope_flag = match scope {
        UnitScopeMarker::User => "--user",
        UnitScopeMarker::System => "--system",
        UnitScopeMarker::Launchd => {
            return Err(EngineError::Unsupported(
                "launchd units are not systemd units".into(),
            ))
        }
    };
    let mut cmd = Command::new("systemctl");
    cmd.arg(scope_flag)
        .arg("--no-pager")
        .arg("--no-ask-password")
        .args(args)
        .kill_on_drop(true);
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(e)) => Err(EngineError::Io(format!("spawning systemctl: {e}"))),
        Err(_) => Err(EngineError::Timeout(format!(
            "systemctl {} timed out after {:?}",
            args.join(" "),
            timeout
        ))),
    }
}

/// The subset of `systemctl show` this adapter reads.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnitStatus {
    pub load_state: String,
    pub active_state: String,
    pub sub_state: String,
    pub main_pid: Option<u32>,
    pub invocation_id: Option<String>,
    /// `Restart=` policy (`no`, `on-failure`, `always`, …).
    pub restart: String,
}

pub fn parse_show(text: &str) -> UnitStatus {
    let mut s = UnitStatus::default();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "LoadState" => s.load_state = value.into(),
            "ActiveState" => s.active_state = value.into(),
            "SubState" => s.sub_state = value.into(),
            "MainPID" => s.main_pid = value.parse().ok().filter(|p| *p != 0),
            "InvocationID" if !value.is_empty() => s.invocation_id = Some(value.into()),
            "Restart" => s.restart = value.into(),
            _ => {}
        }
    }
    s
}

impl UnitStatus {
    pub fn process(&self) -> ProcessState {
        if self.load_state == "not-found" {
            return ProcessState::Stopped;
        }
        match (self.active_state.as_str(), self.sub_state.as_str()) {
            // Crash loop: the process is dead, systemd waits to restart it.
            ("activating", "auto-restart") => ProcessState::Stopped,
            ("active" | "activating" | "reloading" | "deactivating", _) => ProcessState::Running,
            ("inactive" | "failed", _) => ProcessState::Stopped,
            _ => ProcessState::Unknown,
        }
    }
}

// ---- readiness over HTTP ---------------------------------------------------

/// What `GET /v1/models` said.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpProbe {
    /// Nothing listening (or connect timed out): the server is not up yet.
    Unreachable,
    /// Answered with a non-success status (503 while loading, …).
    Status(u16),
    /// 2xx with an OpenAI-style model list.
    Models(Vec<String>),
    /// 2xx but the body is not a model list.
    Garbled,
}

async fn probe_models(port: u16, timeout: Duration) -> HttpProbe {
    match tokio::time::timeout(timeout, fetch_models(port)).await {
        Ok(p) => p,
        Err(_) => HttpProbe::Unreachable,
    }
}

async fn fetch_models(port: u16) -> HttpProbe {
    let Ok(stream) = tokio::net::TcpStream::connect(("127.0.0.1", port)).await else {
        return HttpProbe::Unreachable;
    };
    let Ok((mut sender, conn)) = hyper::client::conn::http1::handshake(TokioIo::new(stream)).await
    else {
        return HttpProbe::Unreachable;
    };
    let conn = tokio::spawn(conn);
    let req = hyper::Request::get("/v1/models")
        .header(hyper::header::HOST, format!("127.0.0.1:{port}"))
        .body(Empty::<Bytes>::new())
        .expect("static request");
    let result = match sender.send_request(req).await {
        Err(_) => HttpProbe::Unreachable,
        Ok(resp) if !resp.status().is_success() => HttpProbe::Status(resp.status().as_u16()),
        Ok(resp) => match Limited::new(resp.into_body(), MAX_MODELS_BODY)
            .collect()
            .await
        {
            Ok(body) => parse_models(&body.to_bytes())
                .map(HttpProbe::Models)
                .unwrap_or(HttpProbe::Garbled),
            Err(_) => HttpProbe::Garbled,
        },
    };
    conn.abort();
    result
}

/// Model ids from an OpenAI-style `{"data":[{"id":…}]}` body.
pub fn parse_models(body: &[u8]) -> Option<Vec<String>> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let data = v.get("data")?.as_array()?;
    Some(
        data.iter()
            .filter_map(|m| m.get("id")?.as_str().map(str::to_owned))
            .collect(),
    )
}

/// Fold manager state and the HTTP answer into one probe.
pub fn combine(status: &UnitStatus, http: Option<&HttpProbe>, expected: &str) -> UnitProbe {
    match status.process() {
        ProcessState::Stopped => UnitProbe {
            process: ProcessState::Stopped,
            readiness: Readiness::NotReady,
            served_model: ServedModel::NotServing,
        },
        ProcessState::Unknown => UnitProbe {
            process: ProcessState::Unknown,
            readiness: Readiness::Unknown,
            served_model: ServedModel::Unknown,
        },
        ProcessState::Running => {
            let (readiness, served_model) = match http {
                None | Some(HttpProbe::Unreachable) | Some(HttpProbe::Status(_)) => {
                    (Readiness::Loading, ServedModel::Unknown)
                }
                Some(HttpProbe::Garbled) => (Readiness::Unknown, ServedModel::Unknown),
                Some(HttpProbe::Models(ids)) if ids.is_empty() => {
                    (Readiness::Loading, ServedModel::Unknown)
                }
                Some(HttpProbe::Models(ids)) => {
                    let model = if ids.iter().any(|id| id == expected) {
                        expected.to_string()
                    } else {
                        ids.join(",")
                    };
                    (Readiness::Ready, ServedModel::Verified { model })
                }
            };
            UnitProbe {
                process: ProcessState::Running,
                readiness,
                served_model,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(active: &str, sub: &str) -> UnitStatus {
        parse_show(&format!(
            "LoadState=loaded\nActiveState={active}\nSubState={sub}\nMainPID=4242\nInvocationID=abc123\n"
        ))
    }

    #[test]
    fn parse_show_reads_properties() {
        let s = status("active", "running");
        assert_eq!(s.load_state, "loaded");
        assert_eq!(s.main_pid, Some(4242));
        assert_eq!(s.invocation_id.as_deref(), Some("abc123"));
        let empty = parse_show("LoadState=not-found\nMainPID=0\nInvocationID=\n");
        assert_eq!(empty.main_pid, None);
        assert_eq!(empty.invocation_id, None);
        assert_eq!(empty.process(), ProcessState::Stopped);
    }

    #[test]
    fn process_state_mapping() {
        assert_eq!(status("active", "running").process(), ProcessState::Running);
        assert_eq!(
            status("activating", "start").process(),
            ProcessState::Running
        );
        assert_eq!(
            status("activating", "auto-restart").process(),
            ProcessState::Stopped
        );
        assert_eq!(status("failed", "failed").process(), ProcessState::Stopped);
        assert_eq!(status("inactive", "dead").process(), ProcessState::Stopped);
        assert_eq!(status("weird", "x").process(), ProcessState::Unknown);
    }

    #[test]
    fn recovery_state_follows_the_manager() {
        use crate::engine::RecoveryState;
        assert_eq!(
            recovery_state_of(&status("active", "running")),
            RecoveryState::Alive
        );
        // Crash loop: the manager is retrying — llmctl only reports.
        assert_eq!(
            recovery_state_of(&status("activating", "auto-restart")),
            RecoveryState::ManagerRetrying
        );
        // start-limit-hit: the manager gave up — llmctl may act.
        assert_eq!(
            recovery_state_of(&status("failed", "failed")),
            RecoveryState::GaveUp
        );
        // Down and the manager is not retrying (clean exit or stopped by a
        // failed run's own rollback): whether that warrants action is the
        // watcher's intent check.
        assert_eq!(
            recovery_state_of(&status("inactive", "dead")),
            RecoveryState::GaveUp
        );
        assert_eq!(
            recovery_state_of(&status("weird", "x")),
            RecoveryState::Unknown
        );
    }

    #[test]
    fn parse_models_reads_openai_list() {
        let body = br#"{"object":"list","data":[{"id":"qwen/a"},{"id":"qwen/b"}]}"#;
        assert_eq!(parse_models(body).unwrap(), vec!["qwen/a", "qwen/b"]);
        assert!(parse_models(b"<html>").is_none());
        assert!(parse_models(br#"{"models":[]}"#).is_none());
    }

    #[test]
    fn combine_verifies_identity_and_separates_loading() {
        let running = status("active", "running");
        let ok = combine(
            &running,
            Some(&HttpProbe::Models(vec!["m/x".into(), "m/y".into()])),
            "m/y",
        );
        assert_eq!(ok.readiness, Readiness::Ready);
        assert_eq!(
            ok.served_model,
            ServedModel::Verified {
                model: "m/y".into()
            }
        );

        let wrong = combine(
            &running,
            Some(&HttpProbe::Models(vec!["m/z".into()])),
            "m/y",
        );
        assert_eq!(
            wrong.served_model,
            ServedModel::Verified {
                model: "m/z".into()
            }
        );

        let loading = combine(&running, Some(&HttpProbe::Status(503)), "m/y");
        assert_eq!(loading.readiness, Readiness::Loading);
        let refused = combine(&running, Some(&HttpProbe::Unreachable), "m/y");
        assert_eq!(refused.readiness, Readiness::Loading);

        let stopped = combine(&status("inactive", "dead"), None, "m/y");
        assert_eq!(stopped.process, ProcessState::Stopped);
        assert_eq!(stopped.served_model, ServedModel::NotServing);
    }

    #[tokio::test]
    async fn unenrolled_unit_is_refused_without_touching_systemd() {
        let engine = SystemdEngine {
            endpoints: Arc::new(std::sync::RwLock::new(BTreeMap::new())),
            show_timeout: Duration::from_millis(1),
            effect_timeout: Duration::from_millis(1),
            http_timeout: Duration::from_millis(1),
            gpu_probe_timeout: Duration::from_millis(1),
        };
        let unit = UnitName::systemd(UnitScopeMarker::User, "vllm-engine.service").unwrap();
        let err = engine
            .apply(Effect::StopUnit {
                workload: llmctl_core::WorkloadName::new("x").unwrap(),
                unit: unit.clone(),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, EngineError::UnknownUnit(_)), "{err}");
        assert!(matches!(
            engine.probe(&unit).await,
            Err(EngineError::UnknownUnit(_))
        ));
    }

    #[tokio::test]
    async fn probe_models_against_local_listener() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = sock.read(&mut buf).await;
            let body = br#"{"data":[{"id":"spike/model"}]}"#;
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
                body.len()
            );
            sock.write_all(head.as_bytes()).await.unwrap();
            sock.write_all(body).await.unwrap();
        });
        let got = probe_models(port, Duration::from_secs(2)).await;
        assert_eq!(got, HttpProbe::Models(vec!["spike/model".into()]));

        // Nothing listening on a freshly released port → unreachable.
        let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let dead_port = free.local_addr().unwrap().port();
        drop(free);
        assert_eq!(
            probe_models(dead_port, Duration::from_millis(500)).await,
            HttpProbe::Unreachable
        );
    }
}
