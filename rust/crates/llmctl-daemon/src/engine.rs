//! The engine boundary. `Engine` is the port real adapters (systemd,
//! launchd, process runner) will implement; `FakeEngine` is the M1
//! file-backed simulation whose state survives daemon restarts — the
//! external world for tests.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::AsyncWriteExt;

use llmctl_core::{
    GpuId, HostInventory, Observation, ProcessState, Readiness, ServedModel, UnitName, WorkloadName,
};

use crate::gpu::GpuOccupant;

pub type BoxFut<T> = Pin<Box<dyn std::future::Future<Output = T> + Send>>;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("unit {0} is not known to this engine")]
    UnknownUnit(String),
    #[error("engine state io error: {0}")]
    Io(String),
    /// A fault was injected (test hook): the effect did NOT happen.
    #[error("fault injected before effect: {0}")]
    FaultBefore(String),
    /// A fault was injected (test hook): the effect DID happen, but the
    /// caller sees an error — the crash-between-effect-and-result window.
    #[error("fault injected after effect: {0}")]
    FaultAfter(String),
    /// The adapter cannot do this here (wrong scope, platform, …).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The caller's input was refused (malformed log cursor, …).
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// The service manager did not answer in time; nothing was decided.
    #[error("timed out: {0}")]
    Timeout(String),
    /// The effect may have happened but could not be confirmed (manager
    /// timed out mid-effect, readback failed). Same honesty rules as
    /// `FaultAfter`: observe before concluding anything.
    #[error("effect unconfirmed: {0}")]
    Unconfirmed(String),
    /// A managed unit's live definition no longer matches the pinned one.
    #[error("{unit} has drifted from its adopted definition ({} field(s) differ)", diffs.len())]
    Drift {
        unit: String,
        diffs: Vec<llmctl_core::FieldDiff>,
    },
}

/// One external effect the executor requests. Carries everything needed to
/// perform it without re-deriving configuration.
#[derive(Debug, Clone)]
pub enum Effect {
    StopUnit {
        workload: WorkloadName,
        unit: UnitName,
    },
    StartUnit {
        workload: WorkloadName,
        unit: UnitName,
        model: String,
        gpus: Vec<GpuId>,
    },
}

impl Effect {
    pub fn unit(&self) -> &UnitName {
        match self {
            Effect::StopUnit { unit, .. } | Effect::StartUnit { unit, .. } => unit,
        }
    }
}

/// Deterministic external identity of an applied effect, persisted in the
/// journal so restart reconciliation can ask "did this happen?".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectOutcome {
    pub identity: String,
}

/// What a probe sees for one unit right now.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitProbe {
    pub process: ProcessState,
    pub readiness: Readiness,
    pub served_model: ServedModel,
}

/// What the service manager says about a unit that is down — the recovery
/// decision input. Recovery decides; observation only reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryState {
    /// The process is alive; nothing to do.
    Alive,
    /// The manager is retrying on its own (`Restart=on-failure`,
    /// auto-restart). llmctl only reports this; acting here would stack a
    /// second retry loop on top of the manager's.
    ManagerRetrying,
    /// The manager gave up (`failed`, start-limit-hit) or the unit is down
    /// and the manager is not retrying it: llmctl may act (the watcher
    /// still gates on the recorded intent).
    GaveUp,
    /// The manager could not be asked: never recover on this.
    Unknown,
}

impl RecoveryState {
    pub fn as_str(self) -> &'static str {
        match self {
            RecoveryState::Alive => "alive",
            RecoveryState::ManagerRetrying => "manager_retrying",
            RecoveryState::GaveUp => "gave_up",
            RecoveryState::Unknown => "unknown",
        }
    }
}

/// The port for engine adapters. Futures are boxed so the daemon holds
/// `Arc<dyn Engine>`.
pub trait Engine: Send + Sync {
    fn name(&self) -> &'static str;

    /// Observe every workload in the inventory (fresh observations).
    fn observe(&self, inventory: &HostInventory) -> BoxFut<Result<Vec<Observation>, EngineError>>;

    /// Apply one external effect. Implementations must be idempotent by
    /// deterministic identity: applying an already-applied effect returns
    /// the same outcome rather than an error or a second side effect.
    fn apply(&self, effect: Effect) -> BoxFut<Result<EffectOutcome, EngineError>>;

    /// Probe one unit.
    fn probe(&self, unit: &UnitName) -> BoxFut<Result<UnitProbe, EngineError>>;

    /// Refuse (`EngineError::Drift`) when a managed workload's live launch
    /// definition differs from the pinned one. Adapters without definition
    /// ownership accept everything.
    fn verify_definition(
        &self,
        _workload: &llmctl_core::WorkloadDefinition,
    ) -> BoxFut<Result<(), EngineError>> {
        Box::pin(async { Ok(()) })
    }

    /// The inventory changed (adoption): refresh enrolled units.
    fn reload(&self, _inventory: &HostInventory) {}

    /// Read the workload's log: the newest `lines` entries, or the next
    /// `lines` after `after` (a cursor from an earlier page).
    fn logs(
        &self,
        _workload: &llmctl_core::WorkloadDefinition,
        _lines: u32,
        _after: Option<String>,
    ) -> BoxFut<Result<LogPage, EngineError>> {
        let name = self.name();
        Box::pin(async move {
            Err(EngineError::Unsupported(format!(
                "the {name} engine has no log source"
            )))
        })
    }

    /// Who currently occupies these cards (compute processes, with the
    /// owning unit when it maps to one). Occupancy is part of each engine's
    /// world: adapters that have no GPUs to see answer empty; adapters that
    /// cannot ask answer `Err` — unknown is never "free".
    fn gpu_occupancy(&self, _gpus: &[GpuId]) -> BoxFut<Result<Vec<GpuOccupant>, EngineError>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    /// What the manager says about a down unit (the recovery decision
    /// input). Engines without a retrying manager report `GaveUp` for a
    /// down unit; engines that cannot ask report `Unknown`.
    fn recovery_state(
        &self,
        _workload: &llmctl_core::WorkloadDefinition,
    ) -> BoxFut<Result<RecoveryState, EngineError>> {
        Box::pin(async { Ok(RecoveryState::Unknown) })
    }
}

/// One page of a workload's log.
#[derive(Debug, Clone)]
pub struct LogPage {
    pub entries: Vec<llmctl_client::wire::LogEntry>,
    /// Continue from here; the caller's cursor when nothing new arrived.
    pub cursor: Option<String>,
    /// The page is full; more may be waiting.
    pub more: bool,
}

// ---- the fake engine ------------------------------------------------------

/// Persistent fake world: `world.json` in the dev data dir. Units go active
/// with a fresh pid, readiness flips after a configurable delay, and the
/// file is the surviving external state a restarted daemon reconciles
/// against. `faults.json` injects failures around the effect boundary.
pub struct FakeEngine {
    dir: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct World {
    #[serde(default = "default_pid")]
    next_pid: u32,
    #[serde(default)]
    units: std::collections::BTreeMap<String, UnitState>,
}

fn default_pid() -> u32 {
    4100
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct UnitState {
    #[serde(default)]
    active: bool,
    pid: Option<u32>,
    ready_at: Option<DateTime<Utc>>,
    model: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Faults {
    /// Queued injections, consumed in order when an effect matches: the
    /// fault fires (and is removed) only when unit and kind match; other
    /// effects pass through untouched.
    #[serde(default)]
    pub fail_effects: Vec<FaultRule>,
    #[serde(default)]
    pub effect_delay_ms: u64,
    #[serde(default)]
    pub readiness_delay_ms: u64,
    #[serde(default)]
    pub serve_wrong_model: bool,
    /// Test hook: sleep this long *after* durably admitting (executor
    /// already spawned) but *before* acknowledging — simulating a lost
    /// acknowledgement without losing the operation.
    #[serde(default)]
    pub admit_delay_ms: u64,
    /// Test hook: `own`/`apply_definition` renders a subtly wrong unit, so
    /// the prove step must catch it, restore and refuse.
    #[serde(default)]
    pub corrupt_render: bool,
}

impl Faults {
    /// Read the fault file at `path`, if any (absent file = no faults).
    pub async fn load(path: &std::path::Path) -> Faults {
        match tokio::fs::read(path).await {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
            Err(_) => Faults::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultRule {
    /// Unit identity (`user/llmctl-next-fake-coder.service`).
    pub unit: String,
    /// `start`, `stop`, or null for any effect on that unit.
    pub kind: Option<String>,
    /// `before`: fail without applying. `after`: apply, then report
    /// failure — the crash-between-effect-and-result window.
    pub mode: String,
}

impl FakeEngine {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn world_path(&self) -> PathBuf {
        self.dir.join("world.json")
    }

    fn faults_path(&self) -> PathBuf {
        self.dir.join("faults.json")
    }

    async fn read_world(&self) -> Result<World, EngineError> {
        read_json(&self.world_path()).await
    }

    async fn write_world(&self, world: &World) -> Result<(), EngineError> {
        write_json_atomic(&self.world_path(), world).await
    }

    async fn take_faults(&self) -> Result<Faults, EngineError> {
        read_json(&self.faults_path()).await
    }

    /// Remove the queued rule at `index` (consumed by a matching effect).
    async fn consume_rule(&self, faults: &Faults, index: usize) -> Result<(), EngineError> {
        let mut next = faults.clone();
        if index < next.fail_effects.len() {
            next.fail_effects.remove(index);
        }
        write_json_atomic(&self.faults_path(), &next).await
    }

    /// Index of the first queued rule matching this effect, if any.
    fn matching_rule(faults: &Faults, effect: &Effect) -> Option<usize> {
        faults.fail_effects.iter().position(|rule| {
            rule.unit == effect.unit().to_string()
                && rule.kind.as_deref().is_none_or(|k| {
                    let actual = match effect {
                        Effect::StopUnit { .. } => "stop",
                        Effect::StartUnit { .. } => "start",
                    };
                    k == actual
                })
        })
    }

    fn probe_state(unit: &UnitState, faults: &Faults) -> UnitProbe {
        if !unit.active {
            return UnitProbe {
                process: ProcessState::Stopped,
                readiness: Readiness::NotReady,
                served_model: ServedModel::NotServing,
            };
        }
        let now = Utc::now();
        let ready = unit.ready_at.map(|t| now >= t).unwrap_or(false);
        let model = match (&unit.model, faults.serve_wrong_model) {
            (Some(m), true) => ServedModel::Verified {
                model: format!("{m}!wrong"),
            },
            (Some(m), false) => ServedModel::Verified { model: m.clone() },
            (None, _) => ServedModel::Unknown,
        };
        UnitProbe {
            process: ProcessState::Running,
            readiness: if ready {
                Readiness::Ready
            } else {
                Readiness::Loading
            },
            served_model: model,
        }
    }
}

impl Engine for FakeEngine {
    fn name(&self) -> &'static str {
        "fake"
    }

    fn observe(&self, inventory: &HostInventory) -> BoxFut<Result<Vec<Observation>, EngineError>> {
        let inventory = inventory.clone();
        let this = FakeEngine {
            dir: self.dir.clone(),
        };
        Box::pin(async move {
            let world = this.read_world().await?;
            let faults = this.take_faults().await.unwrap_or_default();
            let now = Utc::now();
            Ok(inventory
                .workloads
                .iter()
                .map(|def| {
                    let state = world
                        .units
                        .get(&def.unit.to_string())
                        .cloned()
                        .unwrap_or_default();
                    let probe = Self::probe_state(&state, &faults);
                    Observation {
                        workload: def.name.clone(),
                        unit: Some(def.unit.clone()),
                        process: probe.process,
                        readiness: probe.readiness,
                        served_model: probe.served_model,
                        gpus: if state.active {
                            def.gpus.clone()
                        } else {
                            Vec::new()
                        },
                        observed_at: now,
                    }
                })
                .collect())
        })
    }

    fn apply(&self, effect: Effect) -> BoxFut<Result<EffectOutcome, EngineError>> {
        let this = FakeEngine {
            dir: self.dir.clone(),
        };
        Box::pin(async move {
            let faults = this.take_faults().await.unwrap_or_default();
            let fail_mode = Self::matching_rule(&faults, &effect)
                .map(|idx| (idx, faults.fail_effects[idx].mode.clone()));
            if let Some((idx, mode)) = &fail_mode {
                this.consume_rule(&faults, *idx).await?;
                if mode == "before" {
                    return Err(EngineError::FaultBefore(format!(
                        "injected before {}",
                        effect.unit()
                    )));
                }
                tracing::warn!(unit = %effect.unit(), mode = mode,
                    "fault after effect requested; effect will apply then report failure");
            }
            if faults.effect_delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(faults.effect_delay_ms)).await;
            }
            let mut world = this.read_world().await?;
            let key = effect.unit().to_string();
            let entry = world.units.entry(key).or_default();
            let identity = match &effect {
                Effect::StopUnit { workload, .. } => {
                    entry.active = false;
                    entry.pid = None;
                    entry.ready_at = None;
                    entry.model = None;
                    format!("stopped:{}:{}", effect.unit(), workload)
                }
                Effect::StartUnit {
                    workload, model, ..
                } => {
                    entry.active = true;
                    world.next_pid += 1;
                    entry.pid = Some(world.next_pid);
                    entry.ready_at = Some(
                        Utc::now()
                            + chrono::Duration::milliseconds(faults.readiness_delay_ms as i64),
                    );
                    entry.model = Some(model.clone());
                    format!("started:{}:{}:{}", effect.unit(), workload, world.next_pid)
                }
            };
            this.write_world(&world).await?;
            // The effect is durable in the world now. If the fault wanted to
            // simulate a crash after the effect, report failure from here:
            // the journal's `effect_pending` marker plus this world state is
            // exactly the ambiguity reconciliation must resolve.
            if fail_mode.as_ref().map(|(_, m)| m.as_str()) == Some("after") {
                return Err(EngineError::FaultAfter(format!(
                    "injected after {}",
                    effect.unit()
                )));
            }
            Ok(EffectOutcome { identity })
        })
    }

    fn probe(&self, unit: &UnitName) -> BoxFut<Result<UnitProbe, EngineError>> {
        let this = FakeEngine {
            dir: self.dir.clone(),
        };
        let key = unit.to_string();
        Box::pin(async move {
            let world = this.read_world().await?;
            let faults = this.take_faults().await.unwrap_or_default();
            let state = world.units.get(&key).cloned().unwrap_or_default();
            Ok(Self::probe_state(&state, &faults))
        })
    }

    /// Foreign occupants are injected through `gpus.json` in the dev data
    /// dir (absent = every card free), so the guard is testable without a
    /// real GPU: `[{"gpu":0,"pid":999,"unit":null,"memory_mib":123}]`.
    /// `unit` is the full identity in display form (`user/foo.service`).
    fn gpu_occupancy(&self, gpus: &[GpuId]) -> BoxFut<Result<Vec<GpuOccupant>, EngineError>> {
        let path = self.dir.join("gpus.json");
        let wanted: std::collections::BTreeSet<u32> = gpus.iter().map(|g| g.index).collect();
        Box::pin(async move {
            let all: Vec<GpuOccupant> = read_json(&path).await?;
            Ok(all
                .into_iter()
                .filter(|o| wanted.contains(&o.gpu))
                .collect())
        })
    }

    /// The fake engine has no retrying manager: a down unit has given up.
    fn recovery_state(
        &self,
        workload: &llmctl_core::WorkloadDefinition,
    ) -> BoxFut<Result<RecoveryState, EngineError>> {
        let this = FakeEngine {
            dir: self.dir.clone(),
        };
        let key = workload.unit.to_string();
        Box::pin(async move {
            let world = this.read_world().await?;
            let state = world.units.get(&key).cloned().unwrap_or_default();
            Ok(if state.active {
                RecoveryState::Alive
            } else {
                RecoveryState::GaveUp
            })
        })
    }
}

// ---- routing by inventory engine name -------------------------------------

/// Dispatches each unit to the adapter its inventory workload names
/// (`engine: "fake" | "systemd"`). A unit no workload routes is unknown —
/// adapters never see units they were not enrolled for.
pub struct RoutedEngine {
    engines: BTreeMap<String, Arc<dyn Engine>>,
    /// unit display → engine name; replaced wholesale on reload.
    routes: std::sync::RwLock<BTreeMap<String, String>>,
}

impl RoutedEngine {
    pub fn new(
        inventory: &HostInventory,
        engines: BTreeMap<String, Arc<dyn Engine>>,
    ) -> Result<Self, String> {
        let routes = Self::routes_for(inventory, &engines)?;
        Ok(Self {
            engines,
            routes: std::sync::RwLock::new(routes),
        })
    }

    fn routes_for(
        inventory: &HostInventory,
        engines: &BTreeMap<String, Arc<dyn Engine>>,
    ) -> Result<BTreeMap<String, String>, String> {
        let mut routes = BTreeMap::new();
        for w in &inventory.workloads {
            if !engines.contains_key(&w.engine) {
                return Err(format!(
                    "workload {} names engine {:?}; known engines: {:?}",
                    w.name,
                    w.engine,
                    engines.keys().collect::<Vec<_>>()
                ));
            }
            routes.insert(w.unit.to_string(), w.engine.clone());
        }
        Ok(routes)
    }

    fn route(&self, unit: &UnitName) -> Result<Arc<dyn Engine>, EngineError> {
        self.routes
            .read()
            .expect("routes lock")
            .get(&unit.to_string())
            .and_then(|name| self.engines.get(name))
            .cloned()
            .ok_or_else(|| EngineError::UnknownUnit(unit.to_string()))
    }
}

impl Engine for RoutedEngine {
    fn name(&self) -> &'static str {
        "routed"
    }

    fn observe(&self, inventory: &HostInventory) -> BoxFut<Result<Vec<Observation>, EngineError>> {
        // One sub-inventory per engine, results put back in inventory order.
        let mut futs = Vec::new();
        for (name, engine) in &self.engines {
            let mut sub = inventory.clone();
            sub.workloads.retain(|w| &w.engine == name);
            if !sub.workloads.is_empty() {
                futs.push(engine.observe(&sub));
            }
        }
        let order: Vec<WorkloadName> = inventory.workloads.iter().map(|w| w.name.clone()).collect();
        Box::pin(async move {
            let mut by_name = BTreeMap::new();
            for fut in futs {
                for obs in fut.await? {
                    by_name.insert(obs.workload.to_string(), obs);
                }
            }
            Ok(order
                .iter()
                .filter_map(|n| by_name.remove(&n.to_string()))
                .collect())
        })
    }

    fn apply(&self, effect: Effect) -> BoxFut<Result<EffectOutcome, EngineError>> {
        match self.route(effect.unit()) {
            Ok(engine) => engine.apply(effect),
            Err(e) => Box::pin(async move { Err(e) }),
        }
    }

    fn probe(&self, unit: &UnitName) -> BoxFut<Result<UnitProbe, EngineError>> {
        match self.route(unit) {
            Ok(engine) => engine.probe(unit),
            Err(e) => Box::pin(async move { Err(e) }),
        }
    }

    fn verify_definition(
        &self,
        workload: &llmctl_core::WorkloadDefinition,
    ) -> BoxFut<Result<(), EngineError>> {
        match self.route(&workload.unit) {
            Ok(engine) => engine.verify_definition(workload),
            Err(e) => Box::pin(async move { Err(e) }),
        }
    }

    fn recovery_state(
        &self,
        workload: &llmctl_core::WorkloadDefinition,
    ) -> BoxFut<Result<RecoveryState, EngineError>> {
        match self.route(&workload.unit) {
            Ok(engine) => engine.recovery_state(workload),
            Err(e) => Box::pin(async move { Err(e) }),
        }
    }

    fn logs(
        &self,
        workload: &llmctl_core::WorkloadDefinition,
        lines: u32,
        after: Option<String>,
    ) -> BoxFut<Result<LogPage, EngineError>> {
        match self.route(&workload.unit) {
            Ok(engine) => engine.logs(workload, lines, after),
            Err(e) => Box::pin(async move { Err(e) }),
        }
    }

    fn reload(&self, inventory: &HostInventory) {
        match Self::routes_for(inventory, &self.engines) {
            Ok(routes) => *self.routes.write().expect("routes lock") = routes,
            // The caller validated engine names before swapping inventory;
            // keep the old routes rather than route to nothing.
            Err(e) => tracing::error!(error = %e, "inventory reload rejected"),
        }
        for engine in self.engines.values() {
            engine.reload(inventory);
        }
    }

    /// Ask every engine that has enrolled units about its world and merge;
    /// an engine with nothing enrolled cannot answer for this host, but one
    /// that has units and cannot answer makes the whole occupancy unknown.
    fn gpu_occupancy(&self, gpus: &[GpuId]) -> BoxFut<Result<Vec<GpuOccupant>, EngineError>> {
        let enrolled: std::collections::BTreeSet<String> = self
            .routes
            .read()
            .expect("routes lock")
            .values()
            .cloned()
            .collect();
        let mut futs = Vec::new();
        for (name, engine) in &self.engines {
            if enrolled.contains(name) {
                futs.push(engine.gpu_occupancy(gpus));
            }
        }
        Box::pin(async move {
            let mut out = Vec::new();
            for fut in futs {
                out.extend(fut.await?);
            }
            Ok(out)
        })
    }
}

// ---- tiny async json file helpers -----------------------------------------

async fn read_json<T: serde::de::DeserializeOwned + Default>(
    path: &std::path::Path,
) -> Result<T, EngineError> {
    match tokio::fs::read(path).await {
        Ok(bytes) => {
            if bytes.is_empty() {
                return Ok(T::default());
            }
            serde_json::from_slice(&bytes)
                .map_err(|e| EngineError::Io(format!("parse {}: {e}", path.display())))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(EngineError::Io(format!("read {}: {e}", path.display()))),
    }
}

pub(crate) async fn write_json_atomic<T: Serialize>(
    path: &std::path::Path,
    value: &T,
) -> Result<(), EngineError> {
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| EngineError::Io(format!("serialize {}: {e}", path.display())))?;
    let mut f = tokio::fs::File::create(&tmp)
        .await
        .map_err(|e| EngineError::Io(format!("create {}: {e}", tmp.display())))?;
    f.write_all(&bytes)
        .await
        .map_err(|e| EngineError::Io(format!("write {}: {e}", tmp.display())))?;
    f.flush()
        .await
        .map_err(|e| EngineError::Io(format!("flush {}: {e}", tmp.display())))?;
    f.sync_all()
        .await
        .map_err(|e| EngineError::Io(format!("sync {}: {e}", tmp.display())))?;
    tokio::fs::rename(&tmp, path)
        .await
        .map_err(|e| EngineError::Io(format!("rename to {}: {e}", path.display())))?;
    Ok(())
}
