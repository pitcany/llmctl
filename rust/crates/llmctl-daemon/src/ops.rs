//! Operation admission, execution and reconciliation — the mutation and
//! recovery contract of RUST-SUCCESSOR-PLAN.md, distilled:
//!
//! 1. resolve target + effective configuration,
//! 2. plan (named workloads/units/GPUs/rollback),
//! 3. admit under the coordinator and revalidate against fresh
//!    observations (a changed destructive footprint returns a revised plan),
//! 4. persist intent and journal the operation *before* external effects,
//! 5. execute bounded phases, marking the effect boundary in the journal,
//! 6. verify readiness; attempt restoration separately from the failure,
//! 7. release ownership only when the state is understood; after a crash,
//!    block conflicting admission until reconciliation resolves the
//!    interrupted operation or makes the unresolved condition explicit.

use std::sync::Arc;

use chrono::Utc;
use serde_json::json;
use tokio::sync::RwLock;

use llmctl_client::wire::{OpStatus, PhaseStatus, PROTOCOL_VERSION};
use llmctl_core::{
    HostInventory, OpKind, OperationId, Plan, PlanStep, ProcessState, Readiness, ServedModel,
    UnitName, WorkloadName,
};

use crate::db::{Db, OpMutation, OpRow, PhaseRecord};
use crate::engine::{Effect, Engine, EngineError, UnitProbe};

#[derive(Debug, thiserror::Error)]
pub enum OpsError {
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("unknown workload: {0}")]
    UnknownWorkload(String),
    #[error("unsupported protocol {got}, daemon supports {supported:?}")]
    ProtocolMismatch { got: u32, supported: Vec<u32> },
    #[error("idempotency key {key} already used by operation {existing} ({existing_kind} {existing_workload})")]
    IdempotencyConflict {
        key: String,
        existing: String,
        existing_kind: OpKind,
        existing_workload: String,
    },
    #[error(
        "operation {existing} ({existing_kind} {existing_workload}) is still {existing_status}"
    )]
    Overlapping {
        existing: String,
        existing_kind: OpKind,
        existing_workload: String,
        existing_status: OpStatus,
    },
    #[error("operation {existing} needs attention and is unacknowledged; run `llmctl-next ack {existing}` after reviewing it")]
    NeedsAttentionBlock { existing: String },
    #[error("{workload}: unit drifted from its adopted definition; differing: {}", diffs.iter().map(|d| d.field.as_str()).collect::<Vec<_>>().join(", "))]
    DefinitionDrift {
        workload: String,
        diffs: Vec<llmctl_core::FieldDiff>,
    },
    #[error("stale plan: {reason}")]
    StalePlan { reason: String, revised: Box<Plan> },
    #[error(
        "{workload}: gpu {gpu} is held by a process this plan does not manage ({desc}); refusing"
    )]
    GpuOccupied {
        workload: String,
        gpu: u32,
        desc: String,
    },
    #[error("gpu occupancy could not be determined: {0}; unknown is never free")]
    GpuProbe(String),
    #[error("another GPU operation is active ({holder}); waited {waited_ms} ms. Retry when it finishes")]
    HostLockBusy { holder: String, waited_ms: u64 },
    #[error("the host GPU operation lock could not be taken ({0}); unknown is never free")]
    HostLockUnavailable(String),
    #[error("{workload}: {holder} (pid {pid}) holds a lane lease on '{what}' and this operation would disturb it. Retry when it finishes, or remove {path} if the holder is gone")]
    LaneLeased {
        workload: String,
        holder: String,
        pid: u32,
        what: String,
        path: String,
    },
    #[error("{workload}: llm-router has requests in flight on {}; waited {waited_ms} ms. Retry when they finish, or pass --force to cut them", busy.iter().map(|(w, n)| format!("{w} ({n})")).collect::<Vec<_>>().join(", "))]
    RequestsInFlight {
        workload: String,
        busy: Vec<(String, u64)>,
        waited_ms: u64,
    },
    #[error("unknown operation: {0}")]
    UnknownOperation(String),
    #[error("invalid state: {0}")]
    InvalidState(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error(transparent)]
    Db(#[from] crate::db::DbError),
    #[error("engine error: {0}")]
    Engine(String),
}

impl OpsError {
    /// Wire code + HTTP status for the API layer.
    pub fn code_and_status(&self) -> (&'static str, u16) {
        match self {
            OpsError::Invalid(_) => (llmctl_client::wire::codes::INVALID_REQUEST, 400),
            OpsError::UnknownWorkload(_) => (llmctl_client::wire::codes::UNKNOWN_WORKLOAD, 404),
            OpsError::ProtocolMismatch { .. } => {
                (llmctl_client::wire::codes::PROTOCOL_MISMATCH, 400)
            }
            OpsError::IdempotencyConflict { .. } => {
                (llmctl_client::wire::codes::IDEMPOTENCY_CONFLICT, 409)
            }
            OpsError::Overlapping { .. } => {
                (llmctl_client::wire::codes::OVERLAPPING_OPERATION, 409)
            }
            OpsError::NeedsAttentionBlock { .. } => {
                (llmctl_client::wire::codes::BLOCKED_NEEDS_ATTENTION, 409)
            }
            OpsError::StalePlan { .. } => (llmctl_client::wire::codes::STALE_PLAN, 409),
            OpsError::GpuOccupied { .. } => (llmctl_client::wire::codes::GPU_OCCUPIED, 409),
            OpsError::GpuProbe(_) => (llmctl_client::wire::codes::GPU_PROBE_FAILED, 409),
            OpsError::HostLockBusy { .. } | OpsError::HostLockUnavailable(_) => {
                (llmctl_client::wire::codes::HOST_LOCK_BUSY, 409)
            }
            OpsError::LaneLeased { .. } => (llmctl_client::wire::codes::LANE_LEASED, 409),
            OpsError::RequestsInFlight { .. } => {
                (llmctl_client::wire::codes::REQUESTS_IN_FLIGHT, 409)
            }
            OpsError::DefinitionDrift { .. } => (llmctl_client::wire::codes::DEFINITION_DRIFT, 409),
            OpsError::UnknownOperation(_) => (llmctl_client::wire::codes::UNKNOWN_OPERATION, 404),
            OpsError::InvalidState(_) => (llmctl_client::wire::codes::INVALID_STATE, 409),
            OpsError::Unsupported(_) => (llmctl_client::wire::codes::UNSUPPORTED, 501),
            OpsError::Db(_) | OpsError::Engine(_) => {
                (llmctl_client::wire::codes::DAEMON_ERROR, 500)
            }
        }
    }

    pub fn details(&self) -> Option<serde_json::Value> {
        match self {
            OpsError::StalePlan { revised, .. } => Some(json!({ "revised_plan": revised })),
            OpsError::Overlapping {
                existing,
                existing_kind,
                existing_workload,
                existing_status,
            } => Some(json!({
                "operation_id": existing,
                "kind": existing_kind,
                "workload": existing_workload,
                "status": existing_status,
            })),
            OpsError::ProtocolMismatch { supported, .. } => Some(json!({ "supported": supported })),
            OpsError::DefinitionDrift { diffs, .. } => Some(json!({ "diffs": diffs })),
            OpsError::GpuOccupied {
                workload,
                gpu,
                desc,
                ..
            } => Some(json!({ "workload": workload, "gpu": gpu, "occupant": desc })),
            OpsError::HostLockBusy { holder, .. } => Some(json!({ "holder": holder })),
            OpsError::RequestsInFlight { busy, .. } => Some(json!({
                "in_flight": busy.iter().map(|(w, n)| json!({"workload": w, "requests": n})).collect::<Vec<_>>(),
            })),
            OpsError::LaneLeased {
                holder, pid, what, path, ..
            } => Some(json!({ "holder": holder, "pid": pid, "lease": what, "path": path })),
            _ => None,
        }
    }
}

pub enum AdmitOutcome {
    Admitted(OpRow),
    /// Same idempotency key, same request: the existing operation.
    Replayed(OpRow),
}

pub struct AdmitParams {
    pub protocol: u32,
    pub idempotency_key: String,
    pub kind: OpKind,
    pub workload: String,
    pub expected_revision: Option<u64>,
    pub expected_stop_targets: Option<Vec<String>>,
    /// Skip the in-flight request wait (M4c request safety).
    pub force: bool,
    /// Run only: switch the workload to this env-file profile (M4c).
    pub profile: Option<llmctl_client::wire::ProfileRequest>,
    /// Upgrade promotions carry their plan (built by the upgrade module
    /// against fresh observations under the admission lock); required for
    /// `OpKind::Upgrade`, ignored otherwise.
    pub plan_override: Option<Plan>,
}

pub struct OpManager {
    db: Db,
    engine: Arc<dyn Engine>,
    inventory: RwLock<Arc<HostInventory>>,
    /// Fault file (test hooks) read at admit time; `None` disables hooks.
    faults_path: Option<std::path::PathBuf>,
    /// Serializes admission (checks + insert) so two clients cannot both
    /// pass the overlap check concurrently.
    admit_lock: tokio::sync::Mutex<()>,
    /// Set when the host runs the systemd adapter: lets the executor apply
    /// owned definitions (upgrade steps) mid-operation.
    pub(crate) own_ctx: Option<Arc<crate::own::OwnCtx>>,
}

impl OpManager {
    pub fn new(
        db: Db,
        engine: Arc<dyn Engine>,
        inventory: Arc<HostInventory>,
        faults_path: Option<std::path::PathBuf>,
    ) -> Self {
        Self {
            db,
            engine,
            inventory: RwLock::new(inventory),
            faults_path,
            admit_lock: tokio::sync::Mutex::new(()),
            own_ctx: None,
        }
    }

    pub fn with_own_ctx(mut self, ctx: crate::own::OwnCtx) -> Self {
        self.own_ctx = Some(Arc::new(ctx));
        self
    }

    /// Blockers for any mutation: one active operation at a time, and
    /// explicit unresolved outcomes must be acknowledged first.
    async fn check_blockers(&self) -> Result<(), OpsError> {
        for row in self.db.list_ops(1000).await? {
            if !row.status.is_terminal() {
                return Err(OpsError::Overlapping {
                    existing: row.id.to_string(),
                    existing_kind: row.kind,
                    existing_workload: row.workload.to_string(),
                    existing_status: row.status,
                });
            }
            if matches!(row.status, OpStatus::NeedsAttention | OpStatus::Interrupted)
                && !row.acknowledged
            {
                return Err(OpsError::NeedsAttentionBlock {
                    existing: row.id.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Hold admission closed while a non-operation mutation (adoption)
    /// changes the inventory. Fails like admission would when another
    /// mutation is active or an outcome is unacknowledged.
    pub async fn quiesce(&self) -> Result<tokio::sync::MutexGuard<'_, ()>, OpsError> {
        let guard = self.admit_lock.lock().await;
        self.check_blockers().await?;
        Ok(guard)
    }

    pub async fn update_inventory(&self, inv: Arc<HostInventory>) {
        self.engine.reload(&inv);
        *self.inventory.write().await = inv;
    }

    pub async fn inventory(&self) -> Arc<HostInventory> {
        self.inventory.read().await.clone()
    }

    pub fn db(&self) -> &Db {
        &self.db
    }

    pub fn engine(&self) -> &Arc<dyn Engine> {
        &self.engine
    }

    /// Starting a managed workload whose unit drifted from its adopted
    /// definition is refused before anything is journaled; stopping stays
    /// allowed (it cannot make drift worse).
    async fn check_definition(
        &self,
        kind: OpKind,
        def: &llmctl_core::WorkloadDefinition,
        profile_switch: bool,
    ) -> Result<(), OpsError> {
        if !matches!(kind, OpKind::Run | OpKind::Restart) {
            return Ok(());
        }
        match self.engine.verify_definition(def).await {
            Ok(()) => Ok(()),
            // A profile switch overwrites the env file anyway: its contents
            // having changed since adoption (a switch made by llm-router's
            // own path) is not a reason to refuse the very operation that
            // re-pins it. Any other difference still is.
            Err(EngineError::Drift { diffs, .. })
                if profile_switch && crate::profile::only_env_content_differs(&diffs) =>
            {
                Ok(())
            }
            Err(EngineError::Drift { diffs, .. }) => Err(OpsError::DefinitionDrift {
                workload: def.name.to_string(),
                diffs,
            }),
            Err(e) => Err(OpsError::Engine(format!(
                "verifying adopted definition of {}: {e}",
                def.name
            ))),
        }
    }

    /// True when the recorded intent of `workload` is a manual stop.
    async fn manually_stopped(&self, workload: &WorkloadName) -> bool {
        self.db
            .get_intents()
            .await
            .ok()
            .and_then(|all| all.into_iter().find(|(w, _)| w == workload.as_str()))
            .and_then(|(_, j)| serde_json::from_str::<llmctl_core::DesiredIntent>(&j).ok())
            .is_some_and(|i| matches!(i, llmctl_core::DesiredIntent::Stopped { manual: true }))
    }

    /// Fresh observations for planning `kind` on `target`. A run or restart
    /// of an at_start workload also asks each running on-demand vendor
    /// service that really sits on one of the target's cards which models
    /// hold VRAM, so the plan can name those unloads (M4c swap plans). A
    /// service that holds nothing there is not asked: a model in RAM, or
    /// on another card, stays loaded. An occupancy probe or an engine that
    /// cannot answer leaves the list unknown and the guard decides.
    async fn observe_for(
        &self,
        inv: &HostInventory,
        kind: OpKind,
        target: &llmctl_core::WorkloadDefinition,
    ) -> Result<Vec<llmctl_core::Observation>, OpsError> {
        let mut observations = self
            .engine
            .observe(inv)
            .await
            .map_err(|e| OpsError::Engine(e.to_string()))?;
        if !matches!(kind, OpKind::Run | OpKind::Restart)
            || !target.gpu_claim.is_at_start()
            || target.gpus.is_empty()
        {
            return Ok(observations);
        }
        let Ok(occupants) = self.engine.gpu_occupancy(&target.gpus).await else {
            return Ok(observations);
        };
        let on_cards: std::collections::BTreeSet<String> =
            occupants.into_iter().filter_map(|o| o.unit).collect();
        for o in observations.iter_mut() {
            if o.process != ProcessState::Running {
                continue;
            }
            let Some(def) = inv.workload(&o.workload) else {
                continue;
            };
            if def.gpu_claim.is_at_start()
                || !def.engine_kind.has_engine_unload()
                || !on_cards.contains(&def.unit.to_string())
            {
                continue;
            }
            o.loaded_models = self.engine.loaded_models(def).await;
        }
        Ok(observations)
    }

    /// Coexistence with the other GPU controllers (M4c, decision 6). A GPU
    /// operation stands down under another tool's fresh lane lease — unless
    /// the lease is on this workload or the caller runs inside the holder —
    /// and then takes the host-wide operation lock that llm-router, Python
    /// llmctl and llm-heal share. The returned hold lives until the
    /// operation is terminal. `None` = not a GPU operation, or no lock
    /// applies here (Macs, the fake engine).
    async fn coordinate(
        &self,
        inv: &HostInventory,
        def: &llmctl_core::WorkloadDefinition,
        kind: OpKind,
    ) -> Result<Option<crate::coord::OpLock>, OpsError> {
        use crate::coord::{self, LockRefusal};
        use llmctl_client::lease;
        if !coord::is_gpu_operation(kind, def) {
            return Ok(None);
        }
        if let Some(path) = coord::lease_path(&inv.platform, def) {
            if let Some(l) = lease::read_active(&path, lease::MAX_AGE_S) {
                let on_this = l.names(def.name.as_str(), &def.unit.name);
                let inside_holder = coord::under_lease() == Some(l.pid);
                if !on_this && !inside_holder {
                    return Err(OpsError::LaneLeased {
                        workload: def.name.to_string(),
                        what: l.what(),
                        holder: l.holder,
                        pid: l.pid,
                        path: path.display().to_string(),
                    });
                }
            }
        }
        let Some(path) = coord::op_lock_path(&inv.platform, def) else {
            return Ok(None);
        };
        // The caller already holds it for this operation (llm-router's
        // handoff): run under that hold.
        if coord::caller_holds(&path) {
            return Ok(None);
        }
        let wait = coord::op_lock_wait();
        match coord::acquire(&path, &format!("{kind} {}", def.name), wait).await {
            Ok(lock) => Ok(Some(lock)),
            Err(LockRefusal::Busy(holder)) => Err(OpsError::HostLockBusy {
                holder,
                waited_ms: wait.as_millis() as u64,
            }),
            Err(LockRefusal::Io(e)) => Err(OpsError::HostLockUnavailable(e)),
        }
    }

    /// Request safety (M4c item 6): every workload this plan stops, or
    /// unloads a model from, must have no requests in flight through
    /// llm-router. Waits a bounded time for them, then refuses. Only a
    /// positive count blocks; a router that cannot say blocks nothing.
    async fn wait_for_requests(
        &self,
        inv: &HostInventory,
        def: &llmctl_core::WorkloadDefinition,
        plan: &Plan,
    ) -> Result<(), OpsError> {
        use crate::inflight;
        let mut stopping: Vec<(String, u16)> = Vec::new();
        for step in &plan.steps {
            let (PlanStep::StopUnit { workload, .. } | PlanStep::UnloadModel { workload, .. }) =
                step
            else {
                continue;
            };
            if stopping.iter().any(|(n, _)| n == workload.as_str()) {
                continue;
            }
            if let Some(w) = inv.workload(workload) {
                stopping.push((w.name.to_string(), w.port));
            }
        }
        if stopping.is_empty() {
            return Ok(());
        }
        let Some(router) = inflight::router_port(&inv.platform, def) else {
            return Ok(());
        };
        let budget = inflight::wait_budget();
        let deadline = tokio::time::Instant::now() + budget;
        loop {
            let Some(counts) = inflight::in_flight(router, std::time::Duration::from_secs(3)).await
            else {
                tracing::debug!(router, "llm-router gave no in-flight counts; not blocking");
                return Ok(());
            };
            let busy = inflight::busy(&counts, &stopping);
            if busy.is_empty() {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(OpsError::RequestsInFlight {
                    workload: def.name.to_string(),
                    busy,
                    waited_ms: budget.as_millis() as u64,
                });
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
    }

    /// Plan a run, restart or stop. A run that names an env-file profile
    /// (M4c) becomes a profile switch unless that profile is already in
    /// place; a profile on anything but a run is a malformed request.
    async fn plan_lifecycle(
        &self,
        inv: &HostInventory,
        observations: &[llmctl_core::Observation],
        def: &llmctl_core::WorkloadDefinition,
        kind: OpKind,
        profile: Option<&llmctl_client::wire::ProfileRequest>,
    ) -> Result<Plan, OpsError> {
        let name = &def.name;
        let swap = match (kind, profile) {
            (_, None) => None,
            (OpKind::Run, Some(p)) => crate::profile::resolve(def, &p.env_source, &p.model)
                .await
                .map_err(OpsError::Invalid)?,
            (other, Some(_)) => {
                return Err(OpsError::Invalid(format!(
                    "a profile switch is a `run`, not a {other}"
                )))
            }
        };
        match (kind, &swap) {
            (OpKind::Run, Some(swap)) => {
                llmctl_core::plan_run_profile(inv, observations, name, swap)
            }
            (OpKind::Run, None) => llmctl_core::plan_run(inv, observations, name),
            (OpKind::Stop, _) => llmctl_core::plan_stop(inv, observations, name),
            (OpKind::Restart, _) => llmctl_core::plan_restart(inv, observations, name),
            (other, _) => {
                return Err(OpsError::Invalid(format!("{other} is not a lifecycle plan")))
            }
        }
        .map_err(plan_error)
    }

    /// One `ApplyEnvFile` step, forward or restore (M4c profiles): the
    /// file, then the pin — the env digest and the model the workload
    /// serves — so the next operation sees no drift and verifies the right
    /// model. Returns a line for the journal; never the env's contents.
    async fn apply_env_file(&self, step: &PlanStep) -> Result<String, String> {
        let PlanStep::ApplyEnvFile {
            workload,
            path,
            source,
            source_digest,
            backup,
            model,
            ..
        } = step
        else {
            return Err("not an env-file step".into());
        };
        let Some(ctx) = self.own_ctx.clone() else {
            return Err("this daemon cannot re-pin definitions (no ownership context)".into());
        };
        let outcome =
            crate::profile::apply_file(path, source, source_digest, backup.as_deref()).await?;
        let inv = self.inventory().await;
        if let Some(next) = crate::profile::repinned(&inv, workload, path, source_digest, model) {
            crate::own::bump_revision(&ctx, self, next)
                .await
                .map_err(|e| format!("env file {outcome}, but re-pinning {workload} failed: {e}"))?;
        }
        Ok(format!("{path}: {outcome}; {workload} pinned to {source_digest}, serves {model}"))
    }

    /// Fresh observations + a plan for `kind`/`workload` (dry-run path).
    pub async fn build_plan(
        &self,
        kind: OpKind,
        workload: &str,
        profile: Option<&llmctl_client::wire::ProfileRequest>,
    ) -> Result<(Plan, Vec<ObservationLite>), OpsError> {
        let inv = self.inventory().await;
        let name = WorkloadName::new(workload).map_err(|e| OpsError::Invalid(e.to_string()))?;
        let Some(def) = inv.workload(&name) else {
            return Err(OpsError::UnknownWorkload(workload.to_string()));
        };
        self.check_definition(kind, def, profile.is_some()).await?;
        let observations = self.observe_for(&inv, kind, def).await?;
        let plan = match kind {
            OpKind::Run | OpKind::Stop | OpKind::Restart => {
                self.plan_lifecycle(&inv, &observations, def, kind, profile)
                    .await?
            }
            // Upgrade plans are built by the upgrade module and admitted
            // through `admit`; they cannot be dry-run planned generically.
            OpKind::Upgrade => {
                return Err(OpsError::Unsupported(
                    "upgrade plans are built by `upgrade promote`".into(),
                ))
            }
            OpKind::Unload => {
                return Err(OpsError::Unsupported(
                    "unload plans name a model; use `unload <workload> <model>`".into(),
                ))
            }
            OpKind::LaneWrite => {
                return Err(OpsError::Unsupported(
                    "lane writes are planned by `lanes set`/`lanes macro`".into(),
                ))
            }
        };
        self.guard_gpus(&inv, &plan).await?;
        Ok((
            plan,
            observations
                .into_iter()
                .map(ObservationLite::from)
                .collect(),
        ))
    }

    /// Foreign GPU occupancy guard: a plan that starts a unit asks the
    /// affected cards who is on them. Allowed on those cards: the target
    /// itself, enrolled workloads this plan already stops (its conflicts)
    /// and on-demand services whose loaded models this plan unloads first
    /// (M4c swap plans). Anything else — an unenrolled unit or an unmappable
    /// process — refuses the operation before anything is journaled, and a
    /// probe that cannot answer refuses too: unknown is never "free".
    async fn guard_gpus(&self, inv: &HostInventory, plan: &Plan) -> Result<(), OpsError> {
        let starts_units = plan
            .steps
            .iter()
            .any(|s| matches!(s, PlanStep::StartUnit { .. }));
        if !starts_units || plan.affected_gpus.is_empty() {
            return Ok(());
        }
        // An on-demand target (LM Studio, the main Ollama) holds no VRAM at
        // start, so who is on the cards does not matter to its start (M4c).
        if inv
            .workload(&plan.target)
            .is_some_and(|d| !d.gpu_claim.is_at_start())
        {
            return Ok(());
        }
        let touched: Vec<&WorkloadName> = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                PlanStep::StartUnit { workload, .. }
                | PlanStep::StopUnit { workload, .. }
                | PlanStep::UnloadModel { workload, .. } => Some(workload),
                _ => None,
            })
            .collect();
        // Full unit identities (scope included): a user unit and a system
        // unit with the same name are different units.
        let allowed: std::collections::BTreeSet<String> = touched
            .iter()
            .filter_map(|w| inv.workload(w).map(|d| d.unit.to_string()))
            .collect();

        let occupants = self
            .engine
            .gpu_occupancy(&plan.affected_gpus)
            .await
            .map_err(|e| OpsError::GpuProbe(e.to_string()))?;
        for occ in occupants {
            let allowed_here = occ.unit.as_ref().is_some_and(|u| allowed.contains(u));
            if allowed_here {
                continue;
            }
            // An enrolled on-demand service with a model loaded: real VRAM,
            // never stopped by this plan — name the unload instead.
            let on_demand = occ.unit.as_ref().and_then(|u| {
                inv.workloads
                    .iter()
                    .find(|w| &w.unit.to_string() == u && !w.gpu_claim.is_at_start())
            });
            let desc = match (&occ.unit, on_demand) {
                (Some(u), Some(w)) => format!(
                    "{} (on-demand, unit {u}) has a model loaded there, {} MiB; unload it first: llmctl-next engine {} / llmctl-next unload {} <model>",
                    w.name, occ.memory_mib, w.name, w.name
                ),
                (Some(u), None) => format!("pid {} in unit {u}, {} MiB", occ.pid, occ.memory_mib),
                (None, _) => format!("pid {} (no unit), {} MiB", occ.pid, occ.memory_mib),
            };
            return Err(OpsError::GpuOccupied {
                workload: plan.target.to_string(),
                gpu: occ.gpu,
                desc,
            });
        }
        Ok(())
    }

    pub async fn admit(self: &Arc<Self>, params: AdmitParams) -> Result<AdmitOutcome, OpsError> {
        if params.protocol != PROTOCOL_VERSION {
            return Err(OpsError::ProtocolMismatch {
                got: params.protocol,
                supported: vec![PROTOCOL_VERSION],
            });
        }
        let key = llmctl_core::IdempotencyKey::new(&params.idempotency_key)
            .map_err(|e| OpsError::Invalid(e.to_string()))?;
        let name =
            WorkloadName::new(&params.workload).map_err(|e| OpsError::Invalid(e.to_string()))?;

        let _guard = self.admit_lock.lock().await;

        // Idempotency: same key + same request returns the same operation.
        if let Some(existing) = self.db.find_by_key(key.as_str()).await? {
            if existing.kind == params.kind && existing.workload == name {
                return Ok(AdmitOutcome::Replayed(existing));
            }
            return Err(OpsError::IdempotencyConflict {
                key: params.idempotency_key,
                existing: existing.id.to_string(),
                existing_kind: existing.kind,
                existing_workload: existing.workload.to_string(),
            });
        }

        self.check_blockers().await?;

        // Fresh plan against fresh observations.
        let inv = self.inventory().await;
        let Some(def) = inv.workload(&name) else {
            return Err(OpsError::UnknownWorkload(params.workload));
        };
        self.check_definition(params.kind, def, params.profile.is_some())
            .await?;
        // Taken before observing: the plan is built on a world no other
        // controller is changing. Any refusal below drops the hold.
        let hold = self.coordinate(&inv, def, params.kind).await?;
        let observations = self.observe_for(&inv, params.kind, def).await?;
        let plan = match (&params.kind, params.plan_override) {
            (OpKind::Upgrade | OpKind::LaneWrite, Some(prebuilt)) => prebuilt,
            // Re-planned against the observations taken under the admission
            // lock: a service that stopped since the caller planned refuses.
            (OpKind::Unload, Some(prebuilt)) => {
                let model = match prebuilt.steps.as_slice() {
                    [PlanStep::UnloadModel { model, .. }] => model.clone(),
                    _ => return Err(OpsError::Invalid("malformed unload plan".into())),
                };
                llmctl_core::plan_unload(&inv, &observations, &name, &model)
                    .map_err(plan_error)?
            }
            (OpKind::Upgrade | OpKind::Unload | OpKind::LaneWrite, None) => {
                return Err(OpsError::Invalid(format!(
                    "{} admission requires its prebuilt plan",
                    params.kind
                )))
            }
            (kind, _) => {
                self.plan_lifecycle(&inv, &observations, def, *kind, params.profile.as_ref())
                    .await?
            }
        };
        self.guard_gpus(&inv, &plan).await?;

        // Revalidation: refuse obsolete confirmations.
        if let Some(expected) = params.expected_revision {
            if expected != plan.revision {
                return Err(OpsError::StalePlan {
                    reason: format!(
                        "inventory revision moved from {expected} to {}",
                        plan.revision
                    ),
                    revised: Box::new(plan),
                });
            }
        }
        let stop_targets: Vec<String> = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                PlanStep::StopUnit { workload, .. } => Some(workload.to_string()),
                _ => None,
            })
            .collect();
        if let Some(expected) = &params.expected_stop_targets {
            if expected != &stop_targets {
                return Err(OpsError::StalePlan {
                    reason: format!(
                        "destructive footprint changed: confirmed stops {expected:?}, fresh plan stops {stop_targets:?}"
                    ),
                    revised: Box::new(plan),
                });
            }
        }

        // An explicit `unload` names one model on purpose; the wait covers
        // stops and the unloads a run or restart plans on its own.
        if !params.force && params.kind != OpKind::Unload {
            self.wait_for_requests(&inv, def, &plan).await?;
        }

        let now = Utc::now();
        let row = OpRow {
            id: OperationId::generate(),
            idempotency_key: key,
            kind: params.kind,
            workload: name,
            status: OpStatus::Admitted,
            plan: plan.clone(),
            phases: plan
                .steps
                .iter()
                .enumerate()
                .map(|(i, s)| PhaseRecord {
                    index: i,
                    label: s.label(),
                    status: PhaseStatus::Pending,
                    detail: None,
                })
                .collect(),
            result: None,
            recovery: None,
            acknowledged: false,
            created_at: now,
            updated_at: now,
        };
        // Durable admission before any effect and before acknowledging.
        self.db.insert_op(row.clone()).await?;

        // Execute in the background; the admission returns immediately with
        // the durable operation id (client disconnect is not cancellation).
        let manager = Arc::clone(self);
        let op_id = row.id.clone();
        tokio::spawn(async move {
            // The host lock is held until the operation is terminal
            // (rollback included), then released by the drop.
            let _hold = hold;
            if let Err(e) = manager.execute(op_id).await {
                tracing::error!(error = %e, "operation execution failed");
            }
        });

        // Test hook: simulate a lost acknowledgement — the operation is
        // durable and its executor is running, but the caller times out.
        if let Some(fp) = &self.faults_path {
            let faults = crate::engine::Faults::load(fp).await;
            if faults.admit_delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(faults.admit_delay_ms)).await;
            }
        }

        Ok(AdmitOutcome::Admitted(row))
    }

    pub async fn cancel(&self, id: &str) -> Result<OpRow, OpsError> {
        let mut row = self
            .db
            .get_op(id)
            .await?
            .ok_or_else(|| OpsError::UnknownOperation(id.to_string()))?;
        if row.status.is_terminal() {
            return Err(OpsError::InvalidState(format!(
                "operation already terminal: {:?}",
                row.status
            )));
        }
        if row.status == OpStatus::CancelRequested {
            return Ok(row);
        }
        row.status = OpStatus::CancelRequested;
        self.db
            .mutate_op(
                id,
                OpMutation {
                    status: row.status,
                    phases: row.phases.clone(),
                    result: row.result.clone(),
                    recovery: row.recovery.clone(),
                    acknowledged: row.acknowledged,
                },
            )
            .await?;
        // The executor observes the persisted status between phases and
        // records the cancellation outcome; success is claimed only then.
        Ok(row)
    }

    pub async fn acknowledge(&self, id: &str) -> Result<OpRow, OpsError> {
        let mut row = self
            .db
            .get_op(id)
            .await?
            .ok_or_else(|| OpsError::UnknownOperation(id.to_string()))?;
        if !row.status.is_terminal() {
            return Err(OpsError::InvalidState(format!(
                "operation is still {:?} — cancel it before acknowledging",
                row.status
            )));
        }
        if !matches!(row.status, OpStatus::NeedsAttention | OpStatus::Interrupted) {
            return Err(OpsError::InvalidState(format!(
                "acknowledgement applies to needs_attention/interrupted, not {:?}",
                row.status
            )));
        }
        if row.acknowledged {
            return Ok(row);
        }
        row.acknowledged = true;
        self.db
            .mutate_op(
                id,
                OpMutation {
                    status: row.status,
                    phases: row.phases.clone(),
                    result: row.result.clone(),
                    recovery: row.recovery.clone(),
                    acknowledged: true,
                },
            )
            .await?;
        Ok(row)
    }

    // ---- execution ---------------------------------------------------------

    async fn execute(&self, id: OperationId) -> Result<(), OpsError> {
        let mut row = self
            .db
            .get_op(id.as_str())
            .await?
            .ok_or_else(|| OpsError::UnknownOperation(id.to_string()))?;
        row.status = OpStatus::Executing;
        self.persist(&mut row).await?;

        let mut applied_identities: Vec<String> = Vec::new();
        let steps = row.plan.steps.clone();

        for (i, step) in steps.iter().enumerate() {
            // Cancellation check between phases: explicit, persisted.
            let fresh = self.db.get_op(id.as_str()).await?;
            if let Some(fresh) = fresh {
                if fresh.status == OpStatus::CancelRequested {
                    self.finish_canceled(&mut row, i, "cancel observed between phases")
                        .await?;
                    return Ok(());
                }
            }

            set_phase(&mut row.phases, i, |p| p.status = PhaseStatus::Running);
            self.persist(&mut row).await?;

            let outcome = self
                .run_step(&mut row, step, i, &mut applied_identities)
                .await;

            match outcome {
                StepOutcome::Done => {
                    set_phase(&mut row.phases, i, |p| p.status = PhaseStatus::Done);
                    self.persist(&mut row).await?;
                }
                StepOutcome::Canceled(why) => {
                    self.finish_canceled(&mut row, i, &why).await?;
                    return Ok(());
                }
                StepOutcome::Failed(why) => {
                    set_phase(&mut row.phases, i, |p| {
                        p.status = PhaseStatus::Failed;
                        p.detail = Some(why.clone());
                    });
                    self.persist(&mut row).await?;
                    return self.run_recovery(&mut row, i, &applied_identities).await;
                }
            }
        }

        row.status = OpStatus::Succeeded;
        row.result = Some(json!({
            "outcome": "succeeded",
            "effects": applied_identities,
            "completed_at": Utc::now().to_rfc3339(),
        }));
        self.persist(&mut row).await?;
        Ok(())
    }

    async fn run_step(
        &self,
        row: &mut OpRow,
        step: &PlanStep,
        index: usize,
        applied: &mut Vec<String>,
    ) -> StepOutcome {
        match step {
            PlanStep::PersistIntent { workload, intent } => {
                // An eviction (`Stopped { manual: false }`) of a workload the
                // owner stopped by hand — and something else started again —
                // must not turn the manual stop into an ordinary one: the
                // healers and this operation's own rollback rely on it.
                if matches!(intent, llmctl_core::DesiredIntent::Stopped { manual: false })
                    && self.manually_stopped(workload).await
                {
                    set_phase(&mut row.phases, index, |p| {
                        p.detail = Some("kept: manually stopped".into())
                    });
                    return StepOutcome::Done;
                }
                if let Err(e) = self
                    .db
                    .set_intent(
                        workload.as_str(),
                        &serde_json::to_string(intent).expect("intent"),
                    )
                    .await
                {
                    return StepOutcome::Failed(format!("persisting intent: {e}"));
                }
                StepOutcome::Done
            }
            PlanStep::StopUnit { workload, unit } => {
                self.apply_effect(
                    row,
                    index,
                    Effect::StopUnit {
                        workload: workload.clone(),
                        unit: unit.clone(),
                    },
                    applied,
                )
                .await
            }
            PlanStep::StartUnit { workload, unit, .. } => {
                let model = self
                    .model_for(row, workload)
                    .unwrap_or_else(|| "unknown".into());
                let gpus = self.gpus_for(row, workload).unwrap_or_default();
                self.apply_effect(
                    row,
                    index,
                    Effect::StartUnit {
                        workload: workload.clone(),
                        unit: unit.clone(),
                        model,
                        gpus,
                    },
                    applied,
                )
                .await
            }
            PlanStep::UnloadModel {
                workload,
                unit,
                model,
            } => {
                self.apply_effect(
                    row,
                    index,
                    Effect::UnloadModel {
                        workload: workload.clone(),
                        unit: unit.clone(),
                        model: model.clone(),
                    },
                    applied,
                )
                .await
            }
            PlanStep::WriteSwapConfig { .. } => {
                // Effect boundary first: a crash mid-write is resolved by
                // reconciliation, never assumed either way.
                set_phase(&mut row.phases, index, |p| {
                    p.status = PhaseStatus::EffectPending
                });
                if self.persist(row).await.is_err() {
                    return StepOutcome::Failed("journaling effect boundary failed".into());
                }
                match run_swap_write(step).await {
                    Ok(report) => {
                        applied.push(format!("lane-write:{}", step.workload()));
                        set_phase(&mut row.phases, index, |p| {
                            p.detail = Some(report.clone())
                        });
                        StepOutcome::Done
                    }
                    Err(e) => StepOutcome::Failed(e),
                }
            }
            PlanStep::AwaitReadiness {
                workload: _,
                unit,
                timeout_secs,
            } => self.await_readiness(row, index, unit, *timeout_secs).await,
            PlanStep::VerifyServedModel { unit, model, .. } => {
                self.verify_served_model(unit, model).await
            }
            PlanStep::ApplyDefinition {
                workload,
                definition,
                ..
            } => {
                let Some(ctx) = self.own_ctx.clone() else {
                    return StepOutcome::Failed(
                        "this daemon cannot apply definitions (no systemd ownership context)"
                            .into(),
                    );
                };
                let inv = self.inventory().await;
                match crate::own::apply_definition_ctx(
                    &ctx,
                    &inv,
                    workload.as_str(),
                    definition,
                    self,
                )
                .await
                {
                    Ok(_) => StepOutcome::Done,
                    Err(e) => StepOutcome::Failed(format!("applying definition: {e}")),
                }
            }
            PlanStep::ApplyEnvFile { workload, .. } => {
                // Effect boundary first, as for every external step: a
                // crash mid-swap is resolved by looking, never assumed.
                set_phase(&mut row.phases, index, |p| {
                    p.status = PhaseStatus::EffectPending
                });
                if self.persist(row).await.is_err() {
                    return StepOutcome::Failed("journaling effect boundary failed".into());
                }
                match self.apply_env_file(step).await {
                    Ok(report) => {
                        applied.push(format!("env-file:{workload}"));
                        set_phase(&mut row.phases, index, |p| p.detail = Some(report.clone()));
                        StepOutcome::Done
                    }
                    Err(e) => StepOutcome::Failed(format!("applying the profile: {e}")),
                }
            }
            PlanStep::VerifyVersion {
                workload,
                unit,
                interpreter,
                packages,
                binary_check,
            } => {
                if let Err(e) = crate::version::verify_packages(interpreter, packages).await {
                    return StepOutcome::Failed(format!("version verification: {e}"));
                }
                if let Some(bc) = binary_check {
                    let checked = crate::version::run_binary_version(&bc.path, &bc.arg)
                        .await
                        .and_then(|out| {
                            if out.contains(&bc.expect) {
                                Ok(out)
                            } else {
                                Err(format!("binary reports {out:?}, expected {}", bc.expect))
                            }
                        });
                    if let Err(e) = checked {
                        return StepOutcome::Failed(format!("version verification: {e}"));
                    }
                }
                match crate::version::verify_running_interpreter(unit, interpreter).await {
                    Ok(()) => {
                        // The verified env is now the pinned one, or every
                        // later operation reads the upgrade as drift.
                        if let Some(ctx) = self.own_ctx.clone() {
                            if let Err(e) = crate::own::repin_version(&ctx, self, workload).await {
                                tracing::warn!(error = %e, "re-pinning version facts failed");
                            }
                        }
                        StepOutcome::Done
                    }
                    Err(e) => StepOutcome::Failed(format!("version verification: {e}")),
                }
            }
        }
    }

    async fn verify_served_model(&self, unit: &UnitName, model: &str) -> StepOutcome {
        match self.engine.probe(unit).await {
            Ok(probe) => match &probe.served_model {
                ServedModel::Verified { model: m } if m == model => StepOutcome::Done,
                ServedModel::Verified { model: m } => {
                    StepOutcome::Failed(format!("endpoint serves {m}, expected {model}"))
                }
                other => {
                    StepOutcome::Failed(format!("served model not verified (observed {other:?})"))
                }
            },
            Err(e) => StepOutcome::Failed(format!("model verification probe: {e}")),
        }
    }

    async fn apply_effect(
        &self,
        row: &mut OpRow,
        index: usize,
        effect: Effect,
        applied: &mut Vec<String>,
    ) -> StepOutcome {
        // Mark the boundary BEFORE the effect — on the caller's row, so the
        // marker survives later progress writes: a crash here is an
        // effect-pending window reconciliation must resolve by observation.
        set_phase(&mut row.phases, index, |p| {
            p.status = PhaseStatus::EffectPending
        });
        if self.persist(row).await.is_err() {
            return StepOutcome::Failed("journaling effect boundary failed".into());
        }
        let is_start = matches!(effect, Effect::StartUnit { .. });
        match self.engine.apply(effect).await {
            Ok(outcome) => {
                applied.push(outcome.identity.clone());
                // launchd deferred apply: a start that bootstrapped from
                // the plist (the ":boot:" mechanism in the identity) just
                // made an owned/released file live — clear the marker.
                if is_start && outcome.identity.contains(":boot:") {
                    if let Some(ctx) = self.own_ctx.clone() {
                        if let Err(e) = crate::own::clear_pending(&ctx, self, &row.workload).await
                        {
                            tracing::warn!(error = %e, "clearing pending_apply failed");
                        }
                    }
                }
                StepOutcome::Done
            }
            Err(EngineError::FaultAfter(msg) | EngineError::Unconfirmed(msg)) => {
                // The effect very likely happened; the journal's marker plus
                // the observed world is the ambiguity the recovery path and
                // reconciliation must handle honestly. Treat as failure and
                // attempt restoration, which will observe reality.
                applied.push(format!("unconfirmed:{msg}"));
                StepOutcome::Failed(format!("engine reported failure after effect: {msg}"))
            }
            Err(e) => StepOutcome::Failed(format!("applying effect: {e}")),
        }
    }

    async fn await_readiness(
        &self,
        row: &OpRow,
        _index: usize,
        unit: &UnitName,
        timeout_secs: u64,
    ) -> StepOutcome {
        let budget = timeout_secs.min(MAX_POLL_TIMEOUT_SECS);
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(budget);
        loop {
            if let Some(canceled) = self.check_cancel(row).await {
                return StepOutcome::Canceled(canceled);
            }
            match self.engine.probe(unit).await {
                Ok(UnitProbe {
                    process: ProcessState::Stopped,
                    ..
                }) => {
                    return StepOutcome::Failed(
                        "process exited before readiness — different from a timeout".into(),
                    )
                }
                Ok(UnitProbe {
                    readiness: Readiness::Ready,
                    ..
                }) => return StepOutcome::Done,
                Ok(_) => {}
                Err(e) => return StepOutcome::Failed(format!("readiness probe: {e}")),
            }
            if tokio::time::Instant::now() >= deadline {
                return StepOutcome::Failed(format!(
                    "readiness timeout after {budget}s{}; workload may still be loading",
                    if budget < timeout_secs {
                        format!(" (configured {timeout_secs}s, capped)")
                    } else {
                        String::new()
                    }
                ));
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn check_cancel(&self, row: &OpRow) -> Option<String> {
        match self.db.get_op(row.id.as_str()).await {
            Ok(Some(fresh)) if fresh.status == OpStatus::CancelRequested => {
                Some("cancel requested during execution".into())
            }
            _ => None,
        }
    }

    /// Restoration, recorded separately from the original failure. A failed
    /// rollback becomes needs_attention — never a relabeled success.
    async fn run_recovery(
        &self,
        row: &mut OpRow,
        failed_at: usize,
        applied: &[String],
    ) -> Result<(), OpsError> {
        let rollback = row.plan.rollback.clone();
        let rollback_steps = rollback.clone();
        if rollback.is_empty() {
            row.status = OpStatus::Failed;
            row.result = Some(json!({
                "outcome": "failed",
                "failed_phase": failed_at,
                "error": row.phases.get(failed_at).and_then(|p| p.detail.clone()),
                "rollback": "none",
                "effects": applied,
            }));
            self.persist(row).await?;
            return Ok(());
        }

        let mut recovery_steps: Vec<serde_json::Value> = Vec::new();
        let mut restoration_failed = false;
        let start_index = row.phases.len();

        for (offset, step) in rollback.into_iter().enumerate() {
            let label = format!("rollback: {}", step.label());
            row.phases.push(PhaseRecord {
                index: start_index + offset,
                label,
                status: PhaseStatus::Running,
                detail: None,
            });
            self.persist(row).await?;

            if step.is_external() {
                let idx = row.phases.len() - 1;
                set_phase(&mut row.phases, idx, |p| {
                    p.status = PhaseStatus::EffectPending
                });
                self.persist(row).await?;
            }

            // A swap's rollback never restarts a manual stop (plan item 6):
            // the workload was running against the owner's recorded wish,
            // and "restoring" it would repeat that.
            let restores_other = matches!(
                &step,
                PlanStep::StartUnit { workload, .. }
                | PlanStep::PersistIntent { workload, intent: llmctl_core::DesiredIntent::Running }
                    if *workload != row.plan.target
            );
            if restores_other && self.manually_stopped(step.workload()).await {
                recovery_steps.push(json!({
                    "step": step.label(),
                    "outcome": "skipped",
                    "reason": "manually stopped; a rollback does not start it again",
                }));
                let last = row.phases.len() - 1;
                set_phase(&mut row.phases, last, |p| {
                    p.status = PhaseStatus::Done;
                    p.detail = Some("skipped: manually stopped".into());
                });
                self.persist(row).await?;
                continue;
            }

            let outcome: Result<Option<String>, String> = match &step {
                PlanStep::StopUnit { workload, unit } => self
                    .engine
                    .apply(Effect::StopUnit {
                        workload: workload.clone(),
                        unit: unit.clone(),
                    })
                    .await
                    .map(|o| Some(o.identity))
                    .map_err(|e| e.to_string()),
                PlanStep::StartUnit { workload, unit, .. } => {
                    // What this restoration expects the unit to serve: a
                    // profile rollback starts the target on its *previous*
                    // model, not the one the failed plan was after.
                    let model = rollback_steps[offset + 1..]
                        .iter()
                        .find_map(|s| match s {
                            PlanStep::VerifyServedModel {
                                workload: w, model, ..
                            } if w == workload => Some(model.clone()),
                            _ => None,
                        })
                        .or_else(|| self.model_for(row, workload))
                        .unwrap_or_else(|| "unknown".into());
                    let gpus = self.gpus_for(row, workload).unwrap_or_default();
                    self.engine
                        .apply(Effect::StartUnit {
                            workload: workload.clone(),
                            unit: unit.clone(),
                            model,
                            gpus,
                        })
                        .await
                        .map(|o| Some(o.identity))
                        .map_err(|e| e.to_string())
                }
                PlanStep::PersistIntent { workload, intent } => {
                    let _ = self
                        .db
                        .set_intent(
                            workload.as_str(),
                            &serde_json::to_string(intent).expect("intent"),
                        )
                        .await;
                    Ok(None)
                }
                PlanStep::ApplyDefinition {
                    workload,
                    definition,
                    ..
                } => match self.own_ctx.clone() {
                    Some(ctx) => {
                        let inv = self.inventory().await;
                        crate::own::apply_definition_ctx(
                            &ctx,
                            &inv,
                            workload.as_str(),
                            definition,
                            self,
                        )
                        .await
                        .map(|_| None)
                        .map_err(|e| e.to_string())
                    }
                    None => Err("no systemd ownership context".to_string()),
                },
                PlanStep::AwaitReadiness {
                    unit, timeout_secs, ..
                } => match self
                    .await_readiness(row, start_index + offset, unit, *timeout_secs)
                    .await
                {
                    StepOutcome::Done => Ok(None),
                    StepOutcome::Canceled(why) | StepOutcome::Failed(why) => Err(why),
                },
                PlanStep::VerifyServedModel { unit, model, .. } => {
                    match self.verify_served_model(unit, model).await {
                        StepOutcome::Done => Ok(None),
                        StepOutcome::Canceled(why) | StepOutcome::Failed(why) => Err(why),
                    }
                }
                // Restoring the *version* is what ApplyDefinition in this
                // rollback already did; no separate action.
                PlanStep::VerifyVersion { .. } => Ok(None),
                // Never planned as a restoration step (an unload has no
                // rollback); refuse rather than guess.
                PlanStep::UnloadModel { .. } => {
                    Err("an unload is not a restoration step".to_string())
                }
                PlanStep::ApplyEnvFile { .. } => self.apply_env_file(&step).await.map(Some),
                PlanStep::WriteSwapConfig { .. } => run_swap_write(&step).await.map(Some),
            };
            match outcome {
                Ok(identity) => recovery_steps.push(json!({
                    "step": step.label(),
                    "outcome": "applied",
                    "identity": identity,
                })),
                Err(e) => {
                    restoration_failed = true;
                    recovery_steps.push(json!({
                        "step": step.label(),
                        "outcome": "failed",
                        "error": e,
                    }));
                }
            }
            let last = row.phases.len() - 1;
            set_phase(&mut row.phases, last, |p| {
                p.status = if restoration_failed {
                    PhaseStatus::Failed
                } else {
                    PhaseStatus::Done
                };
            });
            self.persist(row).await?;
            if restoration_failed {
                break;
            }
        }

        if restoration_failed {
            row.status = OpStatus::NeedsAttention;
            row.recovery = Some(json!({
                "outcome": "restoration_failed",
                "steps": recovery_steps,
                "note": "the world differs from where the operation started; a human decides",
            }));
        } else {
            row.status = OpStatus::Failed;
            row.recovery = Some(json!({
                "outcome": "restored",
                "steps": recovery_steps,
            }));
        }
        row.result = Some(json!({
            "outcome": "failed",
            "failed_phase": failed_at,
            "error": row.phases.get(failed_at).and_then(|p| p.detail.clone()),
            "effects": applied,
        }));
        self.persist(row).await?;
        Ok(())
    }

    async fn finish_canceled(
        &self,
        row: &mut OpRow,
        from_phase: usize,
        why: &str,
    ) -> Result<(), OpsError> {
        for p in row.phases.iter_mut() {
            if p.status == PhaseStatus::Pending || p.status == PhaseStatus::Running {
                p.status = PhaseStatus::Skipped;
            }
        }
        row.status = OpStatus::Canceled;
        row.result = Some(json!({
            "outcome": "canceled",
            "why": why,
            "at_phase": from_phase,
            "canceled_at": Utc::now().to_rfc3339(),
        }));
        self.persist(row).await?;
        Ok(())
    }

    /// Persist executor progress. The executor owns phases/result, but a
    /// concurrent `CancelRequested` from the API wins over the executor's
    /// stale `Executing` — never let a progress write erase a cancellation.
    async fn persist(&self, row: &mut OpRow) -> Result<(), OpsError> {
        if let Ok(Some(fresh)) = self.db.get_op(row.id.as_str()).await {
            if fresh.status == OpStatus::CancelRequested && row.status == OpStatus::Executing {
                row.status = OpStatus::CancelRequested;
            }
        }
        self.db
            .mutate_op(
                row.id.as_str(),
                OpMutation {
                    status: row.status,
                    phases: row.phases.clone(),
                    result: row.result.clone(),
                    recovery: row.recovery.clone(),
                    acknowledged: row.acknowledged,
                },
            )
            .await?;
        Ok(())
    }

    fn model_for(&self, row: &OpRow, workload: &WorkloadName) -> Option<String> {
        if row.plan.target == *workload {
            Some(row.plan.expected_model.clone())
        } else {
            None // conflicts' models come from the inventory at observe time;
                 // the fake engine keeps the model on the unit record already.
        }
    }

    fn gpus_for(&self, row: &OpRow, workload: &WorkloadName) -> Option<Vec<llmctl_core::GpuId>> {
        if row.plan.target == *workload {
            Some(row.plan.affected_gpus.clone())
        } else {
            None
        }
    }

    // ---- startup reconciliation --------------------------------------------

    /// Called once at daemon start, before serving. Nonterminal operations
    /// left by a crash (or a killed daemon) are finalized honestly:
    ///
    /// * no external effect had begun → `interrupted`, auto-resolved
    ///   (nothing observable could have happened);
    /// * external effects began but everything still undone is
    ///   non-destructive (observation only) → `interrupted` with the
    ///   observed world attached, auto-resolved;
    /// * anything else → `needs_attention` with an observation report,
    ///   blocking mutations until acknowledged.
    pub async fn reconcile_interrupted(&self) -> Result<Vec<String>, OpsError> {
        let mut resolved = Vec::new();
        for mut row in self.db.list_ops(1000).await? {
            if row.status.is_terminal() {
                continue;
            }
            let external_started = row.phases.iter().any(|p| {
                (p.status == PhaseStatus::EffectPending || p.status == PhaseStatus::Done)
                    && p.index < row.plan.steps.len()
                    && row.plan.steps[p.index].is_external()
            });
            let remaining_destructive = row.phases.iter().any(|p| {
                p.status == PhaseStatus::Pending
                    && p.index < row.plan.steps.len()
                    && row.plan.steps[p.index].is_external()
                    && matches!(
                        row.plan.steps[p.index],
                        PlanStep::StopUnit { .. }
                            | PlanStep::StartUnit { .. }
                            | PlanStep::ApplyEnvFile { .. }
                            | PlanStep::WriteSwapConfig { .. }
                    )
            });

            let affected_units: Vec<UnitName> = row
                .plan
                .steps
                .iter()
                .filter_map(|s| match s {
                    PlanStep::StopUnit { unit, .. } | PlanStep::StartUnit { unit, .. } => {
                        Some(unit.clone())
                    }
                    _ => None,
                })
                .collect();
            let mut observed = serde_json::Map::new();
            for unit in &affected_units {
                let probe = self.engine.probe(unit).await.unwrap_or(UnitProbe {
                    process: ProcessState::Unknown,
                    readiness: Readiness::Unknown,
                    served_model: ServedModel::Unknown,
                });
                observed.insert(
                    unit.to_string(),
                    json!({
                        "process": probe.process,
                        "readiness": probe.readiness,
                        "served_model": probe.served_model,
                    }),
                );
            }

            if !external_started {
                row.status = OpStatus::Interrupted;
                row.result = Some(json!({
                    "outcome": "interrupted",
                    "reconciled": "no external effects had begun; nothing observable happened",
                    "observed": observed,
                }));
                row.acknowledged = true; // provably safe
            } else if !remaining_destructive {
                row.status = OpStatus::Interrupted;
                row.result = Some(json!({
                    "outcome": "interrupted",
                    "reconciled": "external effects complete; only observation steps remained",
                    "observed": observed,
                }));
                row.acknowledged = true;
            } else {
                row.status = OpStatus::NeedsAttention;
                row.result = Some(json!({
                    "outcome": "needs_attention",
                    "reconciled": "interrupted with destructive steps pending; observed state attached",
                    "observed": observed,
                }));
                row.acknowledged = false;
            }
            row.updated_at = Utc::now();
            self.persist(&mut row).await?;
            resolved.push(row.id.to_string());
        }
        Ok(resolved)
    }
}

enum StepOutcome {
    Done,
    Canceled(String),
    Failed(String),
}

const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);
/// Readiness polls are bounded even for absurd configured timeouts so a
/// stuck loop cannot pin a phase forever in dev.
/// Upper bound on one readiness wait. 600 s (the M1 value) silently cut
/// real cold starts short: a Strata lane needs ~615 s from NVMe, the
/// DeepSeek lane's own unit allows 3600 s (M4b window, 2026-11-17).
const MAX_POLL_TIMEOUT_SECS: u64 = 3600;

fn set_phase(phases: &mut [PhaseRecord], index: usize, f: impl FnOnce(&mut PhaseRecord)) {
    if let Some(p) = phases.get_mut(index) {
        f(p);
    }
}

/// Trimmed observation used in API responses and reports.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ObservationLite {
    pub workload: String,
    pub process: ProcessState,
    pub readiness: Readiness,
    pub served_model: ServedModel,
    pub gpus: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loaded_models: Option<Vec<String>>,
    pub observed_at: chrono::DateTime<Utc>,
}

impl From<llmctl_core::Observation> for ObservationLite {
    fn from(o: llmctl_core::Observation) -> Self {
        Self {
            workload: o.workload.to_string(),
            process: o.process,
            readiness: o.readiness,
            served_model: o.served_model,
            gpus: o.gpus.iter().map(|g| g.to_string()).collect(),
            loaded_models: o.loaded_models,
            observed_at: o.observed_at,
        }
    }
}
/// A planner refusal about the workload's current state (restart of a
/// stopped unit) is a 409 state conflict, not a malformed request.
fn plan_error(e: llmctl_core::CoreError) -> OpsError {
    match e {
        llmctl_core::CoreError::NotRunning { .. } => OpsError::InvalidState(e.to_string()),
        other => OpsError::Invalid(other.to_string()),
    }
}

/// Run one `WriteSwapConfig` step (forward or rollback).
async fn run_swap_write(step: &PlanStep) -> Result<String, String> {
    let PlanStep::WriteSwapConfig {
        workload,
        path,
        content,
        expect_digest,
        log_path,
        port,
        lanes,
        timeout_secs,
        store_path,
        summary,
    } = step
    else {
        return Err("not a lane write".into());
    };
    crate::lanes::write_and_confirm(
        workload.as_str(),
        path,
        content,
        expect_digest,
        log_path.as_deref(),
        *port,
        lanes,
        std::time::Duration::from_secs(*timeout_secs),
        store_path,
        summary,
    )
    .await
}
