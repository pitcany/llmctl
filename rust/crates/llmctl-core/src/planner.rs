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

/// Build a run plan: persist intent, stop GPU conflicts, start the target,
/// await readiness, verify served-model identity. Workloads whose observed
/// process is already running for the target are restarted only via an
/// explicit `Restart`; a plain run of an already-confirmed-running target
/// still re-plans (stop then start) so the end state is deterministic.
pub fn plan_run(
    inventory: &crate::inventory::HostInventory,
    observations: &[Observation],
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
    let conflicts = gpu_conflicts(inventory, observations, target);
    let mut steps: Vec<PlanStep> = Vec::new();
    let mut rollback: Vec<PlanStep> = Vec::new();

    steps.push(PlanStep::PersistIntent {
        workload: target.clone(),
        intent: DesiredIntent::Running,
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
        rollback.insert(
            0,
            PlanStep::StartUnit {
                workload: c.name.clone(),
                unit: c.unit.clone(),
            },
        );
        rollback.insert(
            0,
            PlanStep::PersistIntent {
                workload: c.name.clone(),
                intent: DesiredIntent::Running,
            },
        );
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
    rollback.push(PlanStep::StopUnit {
        workload: target.clone(),
        unit: def.unit.clone(),
    });

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
    let mut plan = plan_run(inventory, observations, target)?;
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
    Ok(Plan {
        id: OperationId::generate(),
        kind: OpKind::Stop,
        target: target.clone(),
        revision: inventory.revision,
        expected_model: def.model.clone(),
        affected_gpus: def.gpus.clone(),
        steps,
        rollback: Vec::new(),
    })
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

    Ok(Plan {
        id: OperationId::generate(),
        kind: OpKind::Upgrade,
        target: target.clone(),
        revision: inventory.revision,
        expected_model: def.model.clone(),
        affected_gpus: def.gpus.clone(),
        steps,
        rollback,
    })
}
