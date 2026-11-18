//! The launchd adapter (RUST-M3-PLAN B): observes and drives LaunchAgents
//! in the logged-in user's gui domain. launchd semantics the adapter must
//! honor (M0 Q5 spike on mbp2):
//!
//! * `KeepAlive` respawns SIGKILL *and* SIGTERM, so a stop is `bootout`,
//!   and a durable manual stop also `disable`s (a bare bootout would be
//!   undone by `RunAtLoad` at the next login).
//! * Effect identity = label + `runs` + pid (`kickstart`/`bootstrap`
//!   increment `runs`), the InvocationID analogue.
//! * A manager that cannot be asked yields `ProcessState::Unknown`, never
//!   "stopped"; an effect that times out is `Unconfirmed`.
//! * `served model` identity for launchd workloads is the *program path*
//!   the job runs (M3-B.4), not an HTTP model id: the enrolled workload's
//!   `model` field carries the expected program path.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::process::Command;

use llmctl_core::{
    HostInventory, Observation, Ownership, ProcessState, Readiness, ServedModel, UnitName,
    WorkloadDefinition,
};

use crate::engine::{BoxFut, Effect, EffectOutcome, Engine, EngineError, LogPage, RecoveryState, UnitProbe};
use crate::plistdef::{self, PrintState};
use crate::engine_api::{self, EngineProtocol};
use crate::probe_http::{probe_health, HealthProbe, HttpProbe};

/// Inventory engine name that routes a workload to this adapter.
pub const ENGINE_NAME: &str = "launchd";

#[derive(Debug, Clone)]
struct Target {
    port: u16,
    /// Expected program path (the enrolled `model` field, M3-B.4).
    model: String,
    /// Plist path from the pinned definition when managed; the convention
    /// otherwise.
    plist_path: String,
    /// Readiness protocol (M4b): vendor services (Ollama, LM Studio) answer
    /// their own API, not `/health`/`/v1/models`.
    protocol: EngineProtocol,
}

#[derive(Debug, Clone)]
pub struct LaunchdEngine {
    /// Keyed by `UnitName` display (`launchd/com.foo.bar`); replaced on
    /// inventory reload.
    targets: Arc<std::sync::RwLock<BTreeMap<String, Target>>>,
    show_timeout: Duration,
    effect_timeout: Duration,
    http_timeout: Duration,
}

fn uid() -> u32 {
    static UID: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *UID.get_or_init(|| {
        std::process::Command::new("id")
            .arg("-u")
            .output()
            .ok()
            .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
            .unwrap_or(501)
    })
}

impl LaunchdEngine {
    pub fn from_inventory(inventory: &HostInventory) -> Self {
        Self {
            targets: Arc::new(std::sync::RwLock::new(Self::targets_for(inventory))),
            show_timeout: Duration::from_secs(5),
            // bootout of a wedged job can take a while; stay generous.
            effect_timeout: Duration::from_secs(30),
            http_timeout: Duration::from_secs(2),
        }
    }

    fn targets_for(inventory: &HostInventory) -> BTreeMap<String, Target> {
        inventory
            .workloads
            .iter()
            .filter(|w| w.engine == ENGINE_NAME)
            .map(|w| {
                let pinned_path = match &w.ownership {
                    Ownership::Managed { definition, .. } => definition
                        .launchd
                        .as_ref()
                        .map(|x| x.plist_path.clone()),
                    Ownership::External => None,
                };
                let plist_path = pinned_path.unwrap_or_else(|| {
                    plistdef::conventional_plist_path(&w.unit.name)
                        .display()
                        .to_string()
                });
                (
                    w.unit.to_string(),
                    Target {
                        port: w.port,
                        model: w.model.clone(),
                        plist_path,
                        protocol: EngineProtocol::for_kind(&w.engine_kind),
                    },
                )
            })
            .collect()
    }

    pub fn show_timeout(&self) -> Duration {
        self.show_timeout
    }

    /// Shorter manager/effect timeouts (tests).
    pub fn with_timeouts(mut self, show: Duration, effect: Duration, http: Duration) -> Self {
        self.show_timeout = show;
        self.effect_timeout = effect;
        self.http_timeout = http;
        return self;
    }

    fn target(&self, unit: &UnitName) -> Result<Target, EngineError> {
        self.targets
            .read()
            .expect("targets lock")
            .get(&unit.to_string())
            .cloned()
            .ok_or_else(|| EngineError::UnknownUnit(unit.to_string()))
    }

    /// Probe a job that is not (yet) enrolled (adoption's identity check).
    pub async fn probe_unenrolled(
        &self,
        unit: &UnitName,
        port: u16,
        model: &str,
        kind: &llmctl_core::EngineKind,
    ) -> UnitProbe {
        let protocol = EngineProtocol::for_kind(kind);
        let ps = match print_job(&unit.name, self.show_timeout).await {
            Ok(p) => p,
            Err(_) => {
                return UnitProbe {
                    process: ProcessState::Unknown,
                    readiness: Readiness::Unknown,
                    served_model: ServedModel::Unknown,
                }
            }
        };
        let running = process_state(&ps) == ProcessState::Running;
        let (health, models) = if running {
            (
                probe_health(port, self.http_timeout).await,
                Some(engine_api::probe(protocol, port, self.http_timeout).await.probe),
            )
        } else {
            (HealthProbe::NotHealthy, None)
        };
        combine(&ps, health, models.as_ref(), model)
    }

    async fn probe_target(&self, unit: &UnitName, target: &Target) -> UnitProbe {
        let ps = match print_job(&unit.name, self.show_timeout).await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(%unit, error = %e, "launchctl could not be asked");
                return UnitProbe {
                    process: ProcessState::Unknown,
                    readiness: Readiness::Unknown,
                    served_model: ServedModel::Unknown,
                };
            }
        };
        let running = process_state(&ps) == ProcessState::Running;
        let (health, models) = if running {
            (
                probe_health(target.port, self.http_timeout).await,
                Some(
                    engine_api::probe(target.protocol, target.port, self.http_timeout)
                        .await
                        .probe,
                ),
            )
        } else {
            (HealthProbe::NotHealthy, None)
        };
        combine(&ps, health, models.as_ref(), &target.model)
    }

    /// Start: enable (clears a durable stop), then bootstrap (not loaded)
    /// or kickstart (loaded). An already-running job is a no-op whose
    /// identity is the current run.
    async fn start(&self, unit: &UnitName, workload: &str, target: &Target) -> Result<String, EngineError> {
        // enable clears a durable stop; failing to clear it is a real
        // error, not noise (a start that stays disabled is a lie).
        enable(&unit.name, self.show_timeout).await?;
        let before = print_job(&unit.name, self.show_timeout).await?;
        let mechanism = if before.loaded && process_state(&before) == ProcessState::Running {
            "noop"
        } else if before.loaded {
            kickstart(&unit.name, true, self.effect_timeout).await?;
            "kick"
        } else {
            let plist = std::path::PathBuf::from(&target.plist_path);
            if !plist.exists() {
                return Err(EngineError::Io(format!(
                    "start {unit}: plist {} does not exist",
                    target.plist_path
                )));
            }
            if bootstrap(&plist, self.effect_timeout).await.is_err() {
                // Race: someone bootstrapped between print and bootstrap.
                let now = print_job(&unit.name, self.show_timeout).await?;
                if !now.loaded {
                    return Err(EngineError::Io(format!(
                        "bootstrap {unit} from {} failed",
                        target.plist_path
                    )));
                }
                kickstart(&unit.name, true, self.effect_timeout).await?;
            }
            "boot"
        };
        let after = print_job(&unit.name, self.show_timeout)
            .await
            .map_err(|e| EngineError::Unconfirmed(format!("start {unit} done, readback: {e}")))?;
        let runs = after.runs.unwrap_or(0);
        let pid = after.pid.unwrap_or(0);
        Ok(format!(
            "started:{unit}:{workload}:{mechanism}:{runs}:{pid}"
        ))
    }

    /// Stop: bootout (KeepAlive would respawn a mere signal), then disable
    /// so RunAtLoad does not revive it at the next login. Stopping a
    /// not-loaded job only ensures the disable.
    async fn stop(&self, unit: &UnitName, workload: &str) -> Result<String, EngineError> {
        let before = print_job(&unit.name, self.show_timeout).await?;
        let runs = before.runs.unwrap_or(0);
        if before.loaded {
            bootout(&unit.name, self.effect_timeout).await?;
        }
        disable(&unit.name, self.show_timeout).await?;
        let after = print_job(&unit.name, self.show_timeout)
            .await
            .map_err(|e| EngineError::Unconfirmed(format!("stop {unit} done, readback: {e}")))?;
        if after.loaded {
            return Err(EngineError::Unconfirmed(format!(
                "stop {unit}: job still loaded after bootout"
            )));
        }
        Ok(format!("stopped:{unit}:{workload}:{runs}"))
    }
}

impl Engine for LaunchdEngine {
    fn name(&self) -> &'static str {
        ENGINE_NAME
    }

    fn observe(&self, inventory: &HostInventory) -> BoxFut<Result<Vec<Observation>, EngineError>> {
        let this = self.clone();
        let inventory = inventory.clone();
        Box::pin(async move {
            let mut out = Vec::with_capacity(inventory.workloads.len());
            for def in &inventory.workloads {
                if def.engine != ENGINE_NAME {
                    continue;
                }
                let target = this.target(&def.unit)?;
                let probe = this.probe_target(&def.unit, &target).await;
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
                    loaded_models: None,
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
            let target = this.target(&unit)?;
            match &effect {
                Effect::StartUnit { workload, .. } => {
                    let identity = this.start(&unit, workload.as_str(), &target).await?;
                    Ok(EffectOutcome { identity })
                }
                Effect::StopUnit { workload, .. } => {
                    let identity = this.stop(&unit, workload.as_str()).await?;
                    Ok(EffectOutcome { identity })
                }
                Effect::UnloadModel { workload, model, .. } => {
                    let ep = this.target(&unit)?;
                    let outcome = crate::engine_api::unload(
                        ep.protocol,
                        ep.port,
                        model,
                        this.effect_timeout,
                        Duration::from_secs(30),
                    )
                    .await
                    .map_err(|e| match e {
                        crate::engine_api::UnloadError::Unsupported(_)
                        | crate::engine_api::UnloadError::ServiceIdentity(_) => {
                            EngineError::Unsupported(e.to_string())
                        }
                        other => EngineError::Io(other.to_string()),
                    })?;
                    Ok(EffectOutcome {
                        identity: format!("{}:{unit}:{workload}:{model}", outcome.label()),
                    })
                }
            }
        })
    }

    fn probe(&self, unit: &UnitName) -> BoxFut<Result<UnitProbe, EngineError>> {
        let this = self.clone();
        let unit = unit.clone();
        Box::pin(async move {
            let target = this.target(&unit)?;
            Ok(this.probe_target(&unit, &target).await)
        })
    }

    /// The launchd drift guard (M3-C.3):
    /// 1. the owned plist's bytes digest (when llmctl owns the file),
    /// 2. the plist file re-imported vs the pinned definition,
    /// 3. the *loaded* job vs the file (edited-but-not-reloaded / someone
    ///    bootstrapped a different plist),
    /// 4. the program binary's content digest (an in-place upgrade of the
    ///    single static binary is drift).
    fn verify_definition(
        &self,
        workload: &WorkloadDefinition,
    ) -> BoxFut<Result<(), EngineError>> {
        let Ownership::Managed {
            definition: pinned,
            service_file,
            version,
            ..
        } = workload.ownership.clone()
        else {
            return Box::pin(async { Ok(()) });
        };
        let unit = workload.unit.clone();
        let timeout = self.show_timeout;
        Box::pin(async move {
            let mut diffs = Vec::new();
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
                            field: "plist_file".into(),
                            expected: record.digest.clone().into(),
                            actual: actual.into(),
                        }],
                    });
                }
            }
            let imported = plistdef::import_job(&unit, timeout).await?;
            match &imported.definition {
                Some(live) => diffs.extend(pinned.diff(live)),
                None => {
                    diffs.push(llmctl_core::FieldDiff {
                        field: "load_state".into(),
                        expected: "loaded".into(),
                        actual: imported.load_state.clone().into(),
                    });
                }
            }
            if !imported.secret_env_names.is_empty() {
                diffs.push(llmctl_core::FieldDiff {
                    field: "inline_secret_environment".into(),
                    expected: serde_json::Value::Array(vec![]),
                    actual: imported.secret_env_names.clone().into(),
                });
            }

            // Loaded job vs the file on disk (semantic: program + argv).
            if let (Some(live), Some(pinned_extras)) = (&imported.definition, pinned.launchd.as_ref()) {
                if let Ok(ps) = print_job(&unit.name, timeout).await {
                    if ps.loaded {
                        let expected_program = pinned_extras
                            .program
                            .clone()
                            .or_else(|| live.exec_start.first().map(|c| c.path.clone()));
                        let actual_program = ps.program.clone().unwrap_or_default();
                        if let Some(expected) = expected_program {
                            if !expected.is_empty() && actual_program != expected {
                                diffs.push(llmctl_core::FieldDiff {
                                    field: "loaded_job_program".into(),
                                    expected: expected.clone().into(),
                                    actual: actual_program.into(),
                                });
                            }
                        }
                        if let (Some(expected), Some(actual)) = (
                            live.exec_start.first().map(|c| c.argv.clone()),
                            ps.arguments.clone(),
                        ) {
                            if !actual.is_empty() && expected != actual {
                                diffs.push(llmctl_core::FieldDiff {
                                    field: "loaded_job_arguments".into(),
                                    expected: serde_json::json!(expected),
                                    actual: serde_json::json!(actual),
                                });
                            }
                        }
                    }
                }
            }

            // Program binary digest: an in-place upgrade is drift.
            if let (Some(pin), Some(extras)) = (
                version.as_ref(),
                pinned.launchd.as_ref(),
            ) {
                let program = extras
                    .program
                    .clone()
                    .or_else(|| pinned.exec_start.first().map(|c| c.path.clone()));
                if let Some(program) = program {
                    let want = pin.exec_digests.iter().find(|(p, _)| *p == program);
                    if let Some((_, want_digest)) = want {
                        let actual = match tokio::fs::read(&program).await {
                            Ok(bytes) => {
                                use sha2::Digest as _;
                                format!("sha256:{:x}", sha2::Sha256::digest(bytes))
                            }
                            Err(_) => "missing".to_string(),
                        };
                        if &actual != want_digest {
                            diffs.push(llmctl_core::FieldDiff {
                                field: "program_binary".into(),
                                expected: want_digest.clone().into(),
                                actual: actual.into(),
                            });
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
        *self.targets.write().expect("targets lock") = Self::targets_for(inventory);
    }

    /// The plist's `StandardOutPath` (falling back to `StandardErrorPath`),
    /// cursor = (device, inode, offset). Run boundaries come from the
    /// manager's `runs` counter.
    fn logs(
        &self,
        workload: &WorkloadDefinition,
        lines: u32,
        after: Option<String>,
    ) -> BoxFut<Result<LogPage, EngineError>> {
        let unit = workload.unit.clone();
        let timeout = self.show_timeout;
        Box::pin(async move {
            let imported = plistdef::import_job(&unit, timeout).await?;
            let extras = imported
                .definition
                .as_ref()
                .and_then(|d| d.launchd.as_ref())
                .ok_or_else(|| {
                    EngineError::Unsupported(format!(
                        "{unit}: no readable plist to take log paths from"
                    ))
                })?;
            let path = extras
                .stdout_path
                .clone()
                .or_else(|| extras.stderr_path.clone())
                .ok_or_else(|| {
                    EngineError::Unsupported(format!("{unit}: plist sets no Standard*Path"))
                })?;
            let invocation = print_job(&unit.name, timeout)
                .await
                .ok()
                .and_then(|ps| ps.runs)
                .map(|runs| format!("runs-{runs}"));
            crate::logfile::read(
                std::path::Path::new(&path),
                lines,
                after.as_deref(),
                invocation,
            )
            .await
        })
    }

    /// Models of a vendor service that hold GPU memory on this machine.
    fn loaded_models(&self, workload: &WorkloadDefinition) -> BoxFut<Option<Vec<String>>> {
        let target = self.target(&workload.unit);
        let timeout = self.http_timeout;
        Box::pin(async move {
            let target = target.ok()?;
            engine_api::loaded_on_gpu(target.protocol, target.port, timeout).await
        })
    }

    /// No CUDA cards on a Mac: nothing to answer, and occupancy unknown is
    /// reserved for probes that fail. The default (empty) applies.
    ///
    /// launchd retries forever while a job is loaded with KeepAlive: a
    /// down-but-loaded job is the manager retrying; a booted-out job gave
    /// up (that is what llmctl's own stop does).
    fn recovery_state(
        &self,
        workload: &WorkloadDefinition,
    ) -> BoxFut<Result<RecoveryState, EngineError>> {
        let this = self.clone();
        let unit = workload.unit.clone();
        let keep_alive = match &workload.ownership {
            Ownership::Managed { definition, .. } => definition
                .launchd
                .as_ref()
                .map(|x| x.keep_alive.is_always())
                .unwrap_or(false),
            Ownership::External => false,
        };
        Box::pin(async move {
            let ps = match print_job(&unit.name, this.show_timeout).await {
                Ok(p) => p,
                Err(_) => return Ok(RecoveryState::Unknown),
            };
            if !ps.loaded {
                return Ok(RecoveryState::GaveUp);
            }
            if process_state(&ps) == ProcessState::Running {
                return Ok(RecoveryState::Alive);
            }
            Ok(if keep_alive {
                RecoveryState::ManagerRetrying
            } else {
                RecoveryState::GaveUp
            })
        })
    }
}

// ---- pure state mapping ------------------------------------------------------

pub fn process_state(ps: &PrintState) -> ProcessState {
    if !ps.loaded {
        return ProcessState::Stopped;
    }
    match ps.state.as_str() {
        "running" => ProcessState::Running,
        // waiting = KeepAlive backoff before respawn; still launchd's loop.
        _ => ProcessState::Stopped,
    }
}

/// Fold the manager state, the health probe and the model list into one
/// probe. Identity is the program path (expected = enrolled `model`).
pub fn combine(
    ps: &PrintState,
    health: HealthProbe,
    models: Option<&HttpProbe>,
    expected_program: &str,
) -> UnitProbe {
    match process_state(ps) {
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
            let readiness = if health == HealthProbe::Healthy {
                Readiness::Ready
            } else {
                match models {
                    Some(HttpProbe::Models(ids)) if !ids.is_empty() => Readiness::Ready,
                    Some(HttpProbe::Garbled) => Readiness::Unknown,
                    _ => Readiness::Loading,
                }
            };
            let served_model = match &ps.program {
                Some(actual) if actual == expected_program => {
                    ServedModel::Verified { model: actual.clone() }
                }
                other => ServedModel::Unverified {
                    model: other.clone().unwrap_or_default(),
                },
            };
            UnitProbe {
                process: ProcessState::Running,
                readiness,
                served_model,
            }
        }
    }
}

// ---- launchctl ----------------------------------------------------------------

async fn launchctl(
    args: &[&str],
    timeout: Duration,
) -> Result<std::process::Output, EngineError> {
    let mut cmd = Command::new("launchctl");
    cmd.args(args).kill_on_drop(true);
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(e)) => Err(EngineError::Io(format!("spawning launchctl: {e}"))),
        Err(_) => Err(EngineError::Timeout(format!(
            "launchctl {} timed out after {timeout:?}",
            args.join(" ")
        ))),
    }
}

/// `bootstrap` with an empty environment: launchd otherwise inherits the
/// *caller's* exported variables into the job (found live on mbp2: an
/// ssh-launched bootstrap carried every login-shell secret into the
/// job's environment). The plist's own `EnvironmentVariables` still
/// apply; nothing else does.

/// `launchctl print gui/<uid>/<label>`. A clean non-zero exit (job not
/// loaded) yields `loaded: false`; only spawn/timeout failures are errors,
/// so a manager that answered is never mistaken for a stopped job.
pub async fn print_job(label: &str, timeout: Duration) -> Result<PrintState, EngineError> {
    let target = format!("gui/{}/{}", uid(), label);
    let out = launchctl(&["print", &target], timeout).await?;
    if out.status.success() {
        return Ok(plistdef::parse_print(&String::from_utf8_lossy(&out.stdout)));
    }
    // An app started from the Dock or as a login item (LM Studio) has no
    // fixed label: launchd names it `application.<bundle id>.<n>.<n>` and
    // the numbers change with every launch. A record names the stable
    // part; the running instance is looked up in the job list.
    if label.starts_with(plistdef::APP_LABEL_PREFIX) {
        let listed = launchctl(&["list"], timeout).await?;
        let text = String::from_utf8_lossy(&listed.stdout);
        if let Some(full) = plistdef::app_instance_label(&text, label) {
            let target = format!("gui/{}/{}", uid(), full);
            let out = launchctl(&["print", &target], timeout).await?;
            if out.status.success() {
                return Ok(plistdef::parse_print(&String::from_utf8_lossy(&out.stdout)));
            }
        }
    }
    Ok(PrintState {
        loaded: false,
        ..Default::default()
    })
}

pub async fn bootstrap(plist: &std::path::Path, timeout: Duration) -> Result<(), EngineError> {
    let domain = format!("gui/{}", uid());
    // `env -i`: a bootstrap inherits the *caller's* environment into the
    // job (found live on mbp2: an ssh-launched bootstrap inherited every
    // exported secret from the login shell). Jobs we load get none of it;
    // the plist's own EnvironmentVariables still apply.
    let mut cmd = Command::new("env");
    cmd.arg("-i")
        .arg("launchctl")
        .arg("bootstrap")
        .arg(&domain)
        .arg(plist.display().to_string())
        .kill_on_drop(true);
    let out = match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(EngineError::Io(format!("spawning launchctl: {e}"))),
        Err(_) => {
            return Err(EngineError::Timeout(
                "launchctl bootstrap timed out".into(),
            ))
        }
    };
    if out.status.success() {
        return Ok(());
    }
    Err(EngineError::Io(format!(
        "launchctl bootstrap {}: {}",
        plist.display(),
        String::from_utf8_lossy(&out.stderr).trim()
    )))
}

pub async fn bootout(label: &str, timeout: Duration) -> Result<(), EngineError> {
    let target = format!("gui/{}/{}", uid(), label);
    let out = launchctl(&["bootout", &target], timeout).await?;
    if out.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("No such process") || stderr.contains("not found") {
        return Ok(()); // already gone: idempotent
    }
    Err(EngineError::Io(format!(
        "launchctl bootout {label}: {}",
        stderr.trim()
    )))
}

/// `kickstart -k`: kill and restart the job now (launchd's own restart, so
/// KeepAlive cannot race the stop half).
pub async fn kickstart(label: &str, kill: bool, timeout: Duration) -> Result<(), EngineError> {
    let target = format!("gui/{}/{}", uid(), label);
    let mut args = vec!["kickstart"];
    if kill {
        args.push("-k");
    }
    args.push(&target);
    let out = launchctl(&args, timeout).await?;
    if out.status.success() {
        return Ok(());
    }
    Err(EngineError::Io(format!(
        "launchctl kickstart {label}: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    )))
}

pub async fn enable(label: &str, timeout: Duration) -> Result<(), EngineError> {
    let target = format!("gui/{}/{}", uid(), label);
    let out = launchctl(&["enable", &target], timeout).await?;
    if out.status.success() {
        return Ok(());
    }
    Err(EngineError::Io(format!(
        "launchctl enable {label}: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    )))
}

pub async fn disable(label: &str, timeout: Duration) -> Result<(), EngineError> {
    let target = format!("gui/{}/{}", uid(), label);
    let out = launchctl(&["disable", &target], timeout).await?;
    if out.status.success() {
        return Ok(());
    }
    Err(EngineError::Io(format!(
        "launchctl disable {label}: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    )))
}

// ---- scratch-job helpers used by the plist prove (plistdef::runtime) ----------

pub async fn bootstrap_scratch(
    _label: &str,
    plist: &std::path::Path,
    timeout: Duration,
) -> Result<(), EngineError> {
    bootstrap(plist, timeout).await
}

pub async fn bootout_job(label: &str, timeout: Duration) -> Result<(), EngineError> {
    bootout(label, timeout).await
}

/// Pin the program binary's digest at adoption (macOS only): the
/// `VersionRef.exec_digests` slot holds it so an in-place upgrade of the
/// single static binary reads as drift (M3-C.3).
pub fn program_digest_pin(def: &llmctl_core::LaunchDefinition) -> Option<llmctl_core::VersionRef> {
    use sha2::Digest as _;
    let extras = def.launchd.as_ref()?;
    let program = extras
        .program
        .clone()
        .or_else(|| def.exec_start.first().map(|c| c.path.clone()))?;
    let bytes = std::fs::read(&program).ok()?;
    Some(llmctl_core::VersionRef {
        interpreter: None,
        packages: Vec::new(),
        cuda: None,
        lock_digest: "none".into(),
        exec_digests: vec![(
            program,
            format!("sha256:{:x}", sha2::Sha256::digest(bytes)),
        )],
        fact: None,
    })
}
