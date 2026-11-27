//! The pure planner. Given desired state, an inventory and fresh
//! observations, produce a [`Plan`] — or a domain error for invalid input.
//! GPU conflicts are plan content (stop steps + rollback), not errors:
//! the owner previews them and confirms.

use crate::error::{CoreError, CoreResult};
use crate::identity::{GpuId, OperationId, WorkloadName};
use crate::intent::DesiredIntent;
use crate::observation::{Observation, ProcessState};
use crate::plan::{OpKind, Plan, PlanStep};

/// Default readiness budget for `AwaitReadiness` steps (M1 fake engine;
/// becomes an adapter setting when real engines land).
pub const DEFAULT_READINESS_TIMEOUT_SECS: u64 = 120;

/// Conflicting workloads for a run of `target`: running (per observation),
/// not the target itself, and holding at least one of the target's GPUs.
/// A two-GPU plan accounts for both cards and other consumers.
pub fn gpu_conflicts<'a>(
    inventory: &'a crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
) -> Vec<(&'a crate::inventory::WorkloadDefinition, Vec<GpuId>)> {
    let Some(def) = inventory.workload(target) else {
        return Vec::new();
    };
    // An on-demand target claims nothing at start; an on-demand neighbour
    // is never stopped for someone else's start (M4c).
    if !def.gpu_claim.is_at_start() {
        return Vec::new();
    }
    inventory
        .workloads
        .iter()
        .filter(|w| w.name != *target)
        .filter(|w| w.gpu_claim.is_at_start())
        .filter(|w| {
            observations
                .iter()
                .any(|o| o.workload == w.name && o.process == ProcessState::Running)
        })
        .filter_map(|w| {
            let overlap: Vec<GpuId> = w
                .gpus
                .iter()
                .filter(|g| def.gpus.contains(g))
                .cloned()
                .collect();
            if overlap.is_empty() {
                None
            } else {
                Some((w, overlap))
            }
        })
        .collect()
}

/// Models a run of `target` unloads first (M4c swap plans): everything an
/// on-demand vendor service reports loaded, when that service is observed
/// running and may use one of the target's cards. Only what can be named
/// and done is planned — a service whose loaded list is unknown, whose
/// engine has no unload, or that llmctl does not control yields nothing,
/// and the daemon's occupancy guard still refuses if it holds the card.
/// The service keeps running; models are in name order.
pub fn on_demand_unloads<'a>(
    inventory: &'a crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
) -> Vec<(&'a crate::inventory::WorkloadDefinition, Vec<String>)> {
    let Some(def) = inventory.workload(target) else {
        return Vec::new();
    };
    if !def.gpu_claim.is_at_start() {
        return Vec::new();
    }
    let mut out: Vec<(&crate::inventory::WorkloadDefinition, Vec<String>)> = inventory
        .workloads
        .iter()
        .filter(|w| w.name != *target)
        .filter(|w| !w.gpu_claim.is_at_start())
        .filter(|w| w.engine_kind.has_engine_unload() && w.authority.supports_lifecycle())
        .filter(|w| w.gpus.iter().any(|g| def.gpus.contains(g)))
        .filter_map(|w| {
            let o = observations
                .iter()
                .find(|o| o.workload == w.name && o.process == ProcessState::Running)?;
            let mut models = o.loaded_models.clone()?;
            models.sort();
            models.dedup();
            (!models.is_empty()).then_some((w, models))
        })
        .collect();
    out.sort_by(|a, b| a.0.name.cmp(&b.0.name));
    out
}

/// Build a run plan: persist intent, unload on-demand models on the
/// target's cards, stop GPU conflicts, start the target, await readiness,
/// verify served-model identity. Workloads whose observed
/// process is already running for the target are restarted only via an
/// explicit `Restart`; a plain run of an already-confirmed-running target
/// still re-plans (stop then start) so the end state is deterministic.
pub fn plan_run(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
) -> CoreResult<Plan> {
    plan_run_with(inventory, observations, target, false)
}

/// [`plan_run`] with the VRAM-fit decision made by the caller (M4c item
/// 2): `keep_neighbours` = every card the target uses has room for it, so
/// nothing is unloaded and no conflict is stopped — the target starts
/// beside what runs. The decision itself needs measurements and the cards
/// ([`crate::fit::decide`]); the planner only arranges steps.
pub fn plan_run_with(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
    keep_neighbours: bool,
) -> CoreResult<Plan> {
    plan_run_raw(inventory, observations, target, keep_neighbours).map(|p| with_hooks(inventory, p))
}

/// The run plan before companion hooks are placed; builders that reshape
/// it (a profile switch) do so first and place the hooks once at the end.
fn plan_run_raw(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
    keep_neighbours: bool,
) -> CoreResult<Plan> {
    let def = inventory
        .workload(target)
        .ok_or_else(|| CoreError::UnknownWorkload(target.to_string()))?;
    if !def.authority.supports_lifecycle() {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: "control authority is observe-only; lifecycle ownership not proven".into(),
        });
    }
    let conflicts = if keep_neighbours {
        Vec::new()
    } else {
        gpu_conflicts(inventory, observations, target)
    };
    let mut steps: Vec<PlanStep> = Vec::new();
    let mut rollback: Vec<PlanStep> = Vec::new();

    steps.push(PlanStep::PersistIntent {
        workload: target.clone(),
        intent: DesiredIntent::Running,
    });

    // On-demand neighbours give their cards back first: an engine-level
    // unload is the cheapest step to fail, and nothing has been stopped
    // yet. No rollback entry — the service stays up and loads the model
    // again on its next request.
    let unloads = if keep_neighbours {
        Vec::new()
    } else {
        on_demand_unloads(inventory, observations, target)
    };
    for (svc, models) in unloads {
        for model in models {
            steps.push(PlanStep::UnloadModel {
                workload: svc.name.clone(),
                unit: svc.unit.clone(),
                model,
            });
        }
    }

    // The rollback frees the cards first: an evicted engine that starts
    // while the failed target still holds them runs out of memory.
    rollback.push(PlanStep::StopUnit {
        workload: target.clone(),
        unit: def.unit.clone(),
    });

    // Conflicts are stopped in stable (name) order; rollback restarts them
    // in reverse so the last evicted is restored first.
    let mut conflicted: Vec<&crate::inventory::WorkloadDefinition> =
        conflicts.iter().map(|(w, _)| *w).collect();
    conflicted.sort_by(|a, b| a.name.cmp(&b.name));
    for c in &conflicted {
        steps.push(PlanStep::PersistIntent {
            workload: c.name.clone(),
            intent: DesiredIntent::Stopped { manual: false },
        });
        steps.push(PlanStep::StopUnit {
            workload: c.name.clone(),
            unit: c.unit.clone(),
        });
    }
    for c in conflicted.iter().rev() {
        rollback.push(PlanStep::PersistIntent {
            workload: c.name.clone(),
            intent: DesiredIntent::Running,
        });
        rollback.push(PlanStep::StartUnit {
            workload: c.name.clone(),
            unit: c.unit.clone(),
        });
    }

    // If the target itself is running, the plan stops it first so `run` is
    // idempotent on state rather than additive.
    let target_running = observations
        .iter()
        .any(|o| &o.workload == target && o.process == ProcessState::Running);
    if target_running {
        steps.push(PlanStep::StopUnit {
            workload: target.clone(),
            unit: def.unit.clone(),
        });
    }

    steps.push(PlanStep::StartUnit {
        workload: target.clone(),
        unit: def.unit.clone(),
    });
    steps.push(PlanStep::AwaitReadiness {
        workload: target.clone(),
        unit: def.unit.clone(),
        timeout_secs: def.readiness_timeout_secs(),
    });
    steps.push(PlanStep::VerifyServedModel {
        workload: target.clone(),
        unit: def.unit.clone(),
        model: def.model.clone(),
    });
    // The target was serving when this plan was made (a restart, or a run
    // of something already up): the state to go back to is "running", so
    // the rollback starts it again and checks it, after clearing whatever
    // the failed attempt left. A target that was stopped stays stopped.
    // If this start fails too, the operation ends `needs_attention`.
    if target_running {
        rollback.push(PlanStep::StartUnit {
            workload: target.clone(),
            unit: def.unit.clone(),
        });
        rollback.push(PlanStep::AwaitReadiness {
            workload: target.clone(),
            unit: def.unit.clone(),
            timeout_secs: def.readiness_timeout_secs(),
        });
        rollback.push(PlanStep::VerifyServedModel {
            workload: target.clone(),
            unit: def.unit.clone(),
            model: def.model.clone(),
        });
    }
    // Every start comes before the first wait, so a neighbour that is slow
    // or broken does not keep another one down.
    for c in conflicted.iter().rev() {
        rollback.extend(restoration_checks(c, observations));
    }

    Ok(Plan {
        id: OperationId::generate(),
        kind: OpKind::Run,
        target: target.clone(),
        revision: inventory.revision,
        expected_model: def.model.clone(),
        affected_gpus: def.gpus.clone(),
        steps,
        rollback,
    })
}

/// What proves that an evicted workload is back where it was: ready again
/// if it was ready, and serving what it verifiably served. One that was
/// still loading or broken is only started again.
fn restoration_checks(
    evicted: &crate::inventory::WorkloadDefinition,
    observations: &[Observation],
) -> Vec<PlanStep> {
    let Some(before) = observations
        .iter()
        .find(|o| o.workload == evicted.name && o.confirmed_running())
    else {
        return Vec::new();
    };
    let mut checks = vec![PlanStep::AwaitReadiness {
        workload: evicted.name.clone(),
        unit: evicted.unit.clone(),
        timeout_secs: evicted.readiness_timeout_secs(),
    }];
    if let crate::observation::ServedModel::Verified { model } = &before.served_model {
        checks.push(PlanStep::VerifyServedModel {
            workload: evicted.name.clone(),
            unit: evicted.unit.clone(),
            model: model.clone(),
        });
    }
    checks
}

/// One env-file profile switch (M4c): which `EnvironmentFile` of the target
/// is replaced, by what, and what the workload serves afterwards. The
/// daemon fills it from the files on disk; the planner only arranges steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileSwap {
    /// The unit's env file.
    pub path: String,
    /// The profile whose bytes replace it.
    pub source: String,
    pub source_digest: String,
    /// Digest of `path` now: what the restore must put back.
    pub current_digest: String,
    /// Where the forward step keeps the replaced bytes.
    pub backup: String,
    /// Served-model identity under the new profile.
    pub model: String,
}

/// Build a run plan that switches the target to another env-file profile:
/// the run plan (unload on-demand models, stop conflicts, stop the target
/// if it runs), then swap the env file, start, and verify the *profile's*
/// model. On failure the old env goes back; a target that was serving
/// before is started again on it and verified, otherwise it stays down and
/// the evicted workloads return.
pub fn plan_run_profile(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
    profile: &ProfileSwap,
) -> CoreResult<Plan> {
    let mut plan = plan_run_raw(inventory, observations, target, false)?;
    let def = inventory
        .workload(target)
        .ok_or_else(|| CoreError::UnknownWorkload(target.to_string()))?;
    let model = profile.model.trim();
    if model.is_empty() {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: "a profile switch needs the model the profile serves".into(),
        });
    }
    let is_target_start =
        |s: &PlanStep| matches!(s, PlanStep::StartUnit { workload, .. } if workload == target);
    let start = plan
        .steps
        .iter()
        .position(is_target_start)
        .expect("a run plan starts its target");
    plan.steps.insert(
        start,
        PlanStep::ApplyEnvFile {
            workload: target.clone(),
            unit: def.unit.clone(),
            path: profile.path.clone(),
            source: profile.source.clone(),
            source_digest: profile.source_digest.clone(),
            backup: Some(profile.backup.clone()),
            model: model.to_string(),
        },
    );
    for step in &mut plan.steps {
        if let PlanStep::VerifyServedModel {
            workload, model: m, ..
        } = step
        {
            if workload == target {
                *m = model.to_string();
            }
        }
    }
    plan.expected_model = model.to_string();

    // The run plan's rollback stops the target first; the old env goes back
    // under that stopped unit, before anything starts again.
    plan.rollback.insert(
        1,
        PlanStep::ApplyEnvFile {
            workload: target.clone(),
            unit: def.unit.clone(),
            path: profile.path.clone(),
            source: profile.backup.clone(),
            source_digest: profile.current_digest.clone(),
            backup: None,
            model: def.model.clone(),
        },
    );
    Ok(with_hooks(inventory, plan))
}

/// Put every workload's companion hooks around its starts and stops (M4c
/// item 5), in both the forward steps and the rollback, then append the
/// `undo` of each forward hook to the rollback, last run first.
///
/// * `pre_stop` / `post_stop` wrap each `StopUnit` of that workload;
/// * `pre_start` goes before each `StartUnit`;
/// * `post_start` goes after the workload's `VerifyServedModel` when the
///   same list verifies it, else right after the start (a workload a
///   rollback brings back that was not serving before).
pub fn with_hooks(inventory: &crate::inventory::HostInventory, mut plan: Plan) -> Plan {
    use crate::inventory::{Hook, HookWhen};
    if inventory.workloads.iter().all(|w| w.hooks.is_empty()) {
        return plan;
    }
    let step = |w: &crate::inventory::WorkloadDefinition, h: &Hook, phase: String, argv: &[String]| {
        PlanStep::RunHook {
            workload: w.name.clone(),
            name: h.name.clone(),
            phase,
            argv: argv.to_vec(),
            timeout_secs: h.timeout_secs,
            optional: h.optional,
        }
    };
    let wrap = |steps: Vec<PlanStep>| -> Vec<PlanStep> {
        let mut out: Vec<PlanStep> = Vec::with_capacity(steps.len());
        // post_start hooks waiting for their workload's verification.
        let mut pending: Vec<(WorkloadName, Vec<PlanStep>)> = Vec::new();
        let verified_later = |from: usize, w: &WorkloadName, all: &[PlanStep]| {
            all[from..]
                .iter()
                .any(|s| matches!(s, PlanStep::VerifyServedModel { workload, .. } if workload == w))
        };
        for (i, s) in steps.iter().enumerate() {
            let def = inventory.workload(s.workload());
            match (s, def) {
                (PlanStep::StopUnit { .. }, Some(w)) => {
                    out.extend(w.hooks_at(HookWhen::PreStop).map(|h| step(w, h, "pre_stop".into(), &h.argv)));
                    out.push(s.clone());
                    out.extend(w.hooks_at(HookWhen::PostStop).map(|h| step(w, h, "post_stop".into(), &h.argv)));
                }
                (PlanStep::StartUnit { .. }, Some(w)) => {
                    out.extend(w.hooks_at(HookWhen::PreStart).map(|h| step(w, h, "pre_start".into(), &h.argv)));
                    out.push(s.clone());
                    let post: Vec<PlanStep> = w
                        .hooks_at(HookWhen::PostStart)
                        .map(|h| step(w, h, "post_start".into(), &h.argv))
                        .collect();
                    if verified_later(i + 1, &w.name, &steps) {
                        pending.push((w.name.clone(), post));
                    } else {
                        out.extend(post);
                    }
                }
                (PlanStep::VerifyServedModel { workload, .. }, _) => {
                    out.push(s.clone());
                    if let Some(pos) = pending.iter().position(|(w, _)| w == workload) {
                        out.extend(pending.remove(pos).1);
                    }
                }
                _ => out.push(s.clone()),
            }
        }
        out
    };
    plan.steps = wrap(std::mem::take(&mut plan.steps));
    plan.rollback = wrap(std::mem::take(&mut plan.rollback));
    let undos: Vec<PlanStep> = plan
        .steps
        .iter()
        .rev()
        .filter_map(|s| match s {
            PlanStep::RunHook { workload, name, phase, .. } if phase != "undo" => {
                let w = inventory.workload(workload)?;
                let h = w.hooks.iter().find(|h| &h.name == name)?;
                (!h.undo.is_empty()).then(|| step(w, h, "undo".into(), &h.undo))
            }
            _ => None,
        })
        .collect();
    plan.rollback.extend(undos);
    plan
}

/// Build a restart plan: the run plan of an observed-running target (stop,
/// start, await readiness, re-verify the served model), kept as its own
/// kind so the journal says what the owner asked for. A target that is
/// stopped — or whose state is unknown — is refused rather than started:
/// restarting something not known to run would be a silent `run`.
pub fn plan_restart(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
) -> CoreResult<Plan> {
    plan_restart_with(inventory, observations, target, false)
}

/// [`plan_restart`] with the caller's VRAM-fit decision; see
/// [`plan_run_with`].
pub fn plan_restart_with(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
    keep_neighbours: bool,
) -> CoreResult<Plan> {
    if inventory.workload(target).is_none() {
        return Err(CoreError::UnknownWorkload(target.to_string()));
    }
    let state = observations
        .iter()
        .find(|o| &o.workload == target)
        .map(|o| o.process);
    if state != Some(ProcessState::Running) {
        return Err(CoreError::NotRunning {
            workload: target.to_string(),
            state: match state {
                Some(ProcessState::Stopped) => "stopped",
                Some(ProcessState::Unknown) => "state unknown",
                Some(ProcessState::Running) | None => "not observed",
            }
            .into(),
        });
    }
    let mut plan = plan_run_with(inventory, observations, target, keep_neighbours)?;
    plan.kind = OpKind::Restart;
    Ok(plan)
}

/// Build a stop plan. Manual by default: a stop through llmctl is an owner
/// decision that survives restarts and blocks automatic recovery from
/// flipping the workload back on. Stopping a non-running target is a no-op
/// externally (intent only) — preserving the recorded intent still matters.
#[allow(clippy::too_many_arguments)]
pub fn plan_stop(
    inventory: &crate::inventory::HostInventory,
    _observations: &[Observation],
    target: &WorkloadName,
) -> CoreResult<Plan> {
    let def = inventory
        .workload(target)
        .ok_or_else(|| CoreError::UnknownWorkload(target.to_string()))?;
    if !def.authority.supports_lifecycle() {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: "control authority is observe-only; lifecycle ownership not proven".into(),
        });
    }
    let mut steps = vec![PlanStep::PersistIntent {
        workload: target.clone(),
        intent: DesiredIntent::Stopped { manual: true },
    }];
    let running = _observations
        .iter()
        .any(|o| &o.workload == target && o.process == ProcessState::Running);
    if running {
        steps.push(PlanStep::StopUnit {
            workload: target.clone(),
            unit: def.unit.clone(),
        });
    }
    let plan = Plan {
        id: OperationId::generate(),
        kind: OpKind::Stop,
        target: target.clone(),
        revision: inventory.revision,
        expected_model: def.model.clone(),
        affected_gpus: def.gpus.clone(),
        steps,
        rollback: Vec::new(),
    };
    Ok(with_hooks(inventory, plan))
}

/// Build an unload plan (M4b step 1): one engine-level unload of `model`
/// from a running vendor service. No intent change (the service stays up)
/// and no rollback (restoring would mean reloading the model, which is
/// not what anyone asked for). Refused for engines without an unload, for
/// the service identity itself, and for a target not observed running.
pub fn plan_unload(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
    model: &str,
) -> CoreResult<Plan> {
    let def = inventory
        .workload(target)
        .ok_or_else(|| CoreError::UnknownWorkload(target.to_string()))?;
    if !def.authority.supports_lifecycle() {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: "control authority is observe-only; lifecycle ownership not proven".into(),
        });
    }
    if !def.engine_kind.has_engine_unload() {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: format!(
                "engine kind {} has no engine-level unload; stop the workload instead",
                def.engine_kind
            ),
        });
    }
    let model = model.trim();
    if model.is_empty() || model == "ollama" || model == "lmstudio" {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: format!("unload needs one model name, not {model:?}"),
        });
    }
    let state = observations
        .iter()
        .find(|o| &o.workload == target)
        .map(|o| o.process);
    if state != Some(ProcessState::Running) {
        return Err(CoreError::NotRunning {
            workload: target.to_string(),
            state: match state {
                Some(ProcessState::Stopped) => "stopped",
                Some(ProcessState::Unknown) => "state unknown",
                Some(ProcessState::Running) | None => "not observed",
            }
            .into(),
        });
    }
    Ok(Plan {
        id: OperationId::generate(),
        kind: OpKind::Unload,
        target: target.clone(),
        revision: inventory.revision,
        expected_model: def.model.clone(),
        affected_gpus: Vec::new(),
        steps: vec![PlanStep::UnloadModel {
            workload: target.clone(),
            unit: def.unit.clone(),
            model: model.to_string(),
        }],
        rollback: Vec::new(),
    })
}

/// Build a load plan (M5c item 4): one engine-level load of `model` onto a
/// running vendor service, the mirror of [`plan_unload`]. No intent change
/// (the service stays up). A failed load runs one unload rollback so a
/// partial load gives its VRAM back. The catalog check and the VRAM fit are
/// the daemon's (they need the catalog store and free-VRAM reads); this
/// builder refuses the same structural cases as the unload: an engine
/// without a load, the service identity, a target not observed running.
pub fn plan_load(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
    model: &str,
    keep_alive_secs: i64,
) -> CoreResult<Plan> {
    let def = inventory
        .workload(target)
        .ok_or_else(|| CoreError::UnknownWorkload(target.to_string()))?;
    if !def.authority.supports_lifecycle() {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: "control authority is observe-only; lifecycle ownership not proven".into(),
        });
    }
    if !def.engine_kind.has_engine_load() {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: format!(
                "engine kind {} has no engine-level load; a request loads it, or use the engine's own tool",
                def.engine_kind
            ),
        });
    }
    let model = model.trim();
    if model.is_empty() || model == "ollama" || model == "lmstudio" {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: format!("load needs one model name, not {model:?}"),
        });
    }
    if keep_alive_secs == 0 {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: "keep_alive 0 unloads immediately; that is what `unload` is for".into(),
        });
    }
    let state = observations
        .iter()
        .find(|o| &o.workload == target)
        .map(|o| o.process);
    if state != Some(ProcessState::Running) {
        return Err(CoreError::NotRunning {
            workload: target.to_string(),
            state: match state {
                Some(ProcessState::Stopped) => "stopped",
                Some(ProcessState::Unknown) => "state unknown",
                Some(ProcessState::Running) | None => "not observed",
            }
            .into(),
        });
    }
    Ok(Plan {
        id: OperationId::generate(),
        kind: OpKind::Load,
        target: target.clone(),
        revision: inventory.revision,
        expected_model: def.model.clone(),
        affected_gpus: def.gpus.clone(),
        steps: vec![PlanStep::LoadModel {
            workload: target.clone(),
            unit: def.unit.clone(),
            model: model.to_string(),
            keep_alive_secs,
            timeout_secs: LOAD_TIMEOUT_SECS,
        }],
        rollback: vec![PlanStep::UnloadModel {
            workload: target.clone(),
            unit: def.unit.clone(),
            model: model.to_string(),
        }],
    })
}

/// How long the one load step may take: a cold Ollama load of a 30 B model
/// reads tens of GB from disk (the qwen3coder30b-start script waited 600 s;
/// this covers a slower disk and the confirm loop).
pub const LOAD_TIMEOUT_SECS: u64 = 900;

/// Build an upgrade (promote) plan (plan D.5): stop, swap the owned
/// definition to `next`, start, await readiness, verify the served model,
/// then verify the environment itself reports the target versions. Failure
/// rolls back to `previous` — definition, start, readiness, identity — and
/// a failed rollback is `needs_attention` (the executor's normal honesty).
pub fn plan_upgrade(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
    target: &WorkloadName,
    previous: &crate::launch::LaunchDefinition,
    next: &crate::launch::LaunchDefinition,
    interpreter: &str,
    packages: &[(String, String)],
    binary_check: Option<crate::plan::BinaryCheck>,
) -> CoreResult<Plan> {
    let def = inventory
        .workload(target)
        .ok_or_else(|| CoreError::UnknownWorkload(target.to_string()))?;
    if !def.authority.supports_lifecycle() {
        return Err(CoreError::InvalidConfig {
            workload: target.to_string(),
            reason: "control authority is observe-only; lifecycle ownership not proven".into(),
        });
    }
    let running = observations
        .iter()
        .any(|o| &o.workload == target && o.process == ProcessState::Running);
    let mut steps: Vec<PlanStep> = vec![PlanStep::PersistIntent {
        workload: target.clone(),
        intent: DesiredIntent::Running,
    }];
    if running {
        steps.push(PlanStep::StopUnit {
            workload: target.clone(),
            unit: def.unit.clone(),
        });
    }
    steps.push(PlanStep::ApplyDefinition {
        workload: target.clone(),
        unit: def.unit.clone(),
        definition: next.clone(),
    });
    steps.push(PlanStep::StartUnit {
        workload: target.clone(),
        unit: def.unit.clone(),
    });
    steps.push(PlanStep::AwaitReadiness {
        workload: target.clone(),
        unit: def.unit.clone(),
        timeout_secs: def.readiness_timeout_secs(),
    });
    steps.push(PlanStep::VerifyServedModel {
        workload: target.clone(),
        unit: def.unit.clone(),
        model: def.model.clone(),
    });
    steps.push(PlanStep::VerifyVersion {
        workload: target.clone(),
        unit: def.unit.clone(),
        interpreter: interpreter.to_string(),
        packages: packages.to_vec(),
        binary_check,
    });

    // One card means downtime; the rollback is the old definition back,
    // started, awaited and identity-verified (plan says so).
    let mut rollback = vec![
        PlanStep::ApplyDefinition {
            workload: target.clone(),
            unit: def.unit.clone(),
            definition: previous.clone(),
        },
        PlanStep::StartUnit {
            workload: target.clone(),
            unit: def.unit.clone(),
        },
        PlanStep::AwaitReadiness {
            workload: target.clone(),
            unit: def.unit.clone(),
            timeout_secs: def.readiness_timeout_secs(),
        },
        PlanStep::VerifyServedModel {
            workload: target.clone(),
            unit: def.unit.clone(),
            model: def.model.clone(),
        },
    ];
    if running {
        rollback.insert(
            0,
            PlanStep::StopUnit {
                workload: target.clone(),
                unit: def.unit.clone(),
            },
        );
    }

    let plan = Plan {
        id: OperationId::generate(),
        kind: OpKind::Upgrade,
        target: target.clone(),
        revision: inventory.revision,
        expected_model: def.model.clone(),
        affected_gpus: def.gpus.clone(),
        steps,
        rollback,
    };
    Ok(with_hooks(inventory, plan))
}
