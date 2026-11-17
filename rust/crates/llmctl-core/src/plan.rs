//! Plans: the concrete, previewable statement of what a mutation will do.
//! Interactive confirmation refers to a plan; admission revalidates it
//! against fresh observations and returns a revised plan rather than
//! executing an obsolete confirmation.

use serde::{Deserialize, Serialize};

use crate::identity::{GpuId, OperationId, UnitName, WorkloadName};
use crate::intent::DesiredIntent;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    Run,
    Stop,
    Restart,
    /// Staged version promotion: stop, swap the definition, start, verify
    /// readiness, served identity and the version itself (plan D).
    Upgrade,
    /// Engine-level unload of one model from a vendor service (Ollama, LM
    /// Studio): the process keeps running (M4b step 1).
    Unload,
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            OpKind::Run => "run",
            OpKind::Stop => "stop",
            OpKind::Restart => "restart",
            OpKind::Upgrade => "upgrade",
            OpKind::Unload => "unload",
        })
    }
}

/// One step of a plan. Steps with external effects are marked so the
/// executor knows where the effect/result-write boundary lies: the crash
/// window the journal must survive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "step", rename_all = "snake_case")]
pub enum PlanStep {
    /// Internal: persist desired intent before any external side effect.
    PersistIntent {
        workload: WorkloadName,
        intent: DesiredIntent,
    },
    /// External: stop a unit through the service manager.
    StopUnit {
        workload: WorkloadName,
        unit: UnitName,
    },
    /// External: start a unit through the service manager.
    StartUnit {
        workload: WorkloadName,
        unit: UnitName,
    },
    /// External observation loop: wait for endpoint readiness. Readiness
    /// timeout leaves the workload `starting`, a distinct outcome from a
    /// process that exited.
    AwaitReadiness {
        workload: WorkloadName,
        unit: UnitName,
        timeout_secs: u64,
    },
    /// External observation: verify the endpoint serves the expected model
    /// identity, not merely that it answers.
    VerifyServedModel {
        workload: WorkloadName,
        unit: UnitName,
        model: String,
    },
    /// External: swap the owned unit file for `definition` and prove the
    /// effective definition matches it (restore on mismatch). Upgrade-only.
    ApplyDefinition {
        workload: WorkloadName,
        unit: UnitName,
        definition: crate::launch::LaunchDefinition,
    },
    /// External observation: the running environment must report exactly
    /// these package versions through `interpreter` (importlib metadata),
    /// and/or the staged binary must print its own version (M4a: the
    /// dir-recipe kinds). `interpreter` names the primary executable of
    /// whatever recipe kind the workload pins.
    VerifyVersion {
        workload: WorkloadName,
        unit: UnitName,
        interpreter: String,
        packages: Vec<(String, String)>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        binary_check: Option<BinaryCheck>,
    },
    /// External: ask the engine to unload one model and confirm through its
    /// loaded list (vendor services only).
    UnloadModel {
        workload: WorkloadName,
        unit: UnitName,
        model: String,
    },
}

/// A binary's own version self-check (M4a): run `path arg`, the first
/// output line must contain `expect`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryCheck {
    pub path: String,
    pub arg: String,
    pub expect: String,
}

impl PlanStep {
    pub fn workload(&self) -> &WorkloadName {
        match self {
            PlanStep::PersistIntent { workload, .. }
            | PlanStep::StopUnit { workload, .. }
            | PlanStep::StartUnit { workload, .. }
            | PlanStep::AwaitReadiness { workload, .. }
            | PlanStep::VerifyServedModel { workload, .. }
            | PlanStep::ApplyDefinition { workload, .. }
            | PlanStep::VerifyVersion { workload, .. }
            | PlanStep::UnloadModel { workload, .. } => workload,
        }
    }

    pub fn is_external(&self) -> bool {
        !matches!(self, PlanStep::PersistIntent { .. })
    }

    /// Short human label used by CLI/TUI previews.
    pub fn label(&self) -> String {
        match self {
            PlanStep::PersistIntent { workload, intent } => {
                format!("record intent: {workload} → {intent:?}")
            }
            PlanStep::StopUnit { workload, .. } => format!("stop {workload}"),
            PlanStep::StartUnit { workload, .. } => format!("start {workload}"),
            PlanStep::AwaitReadiness { workload, .. } => format!("wait for {workload} readiness"),
            PlanStep::VerifyServedModel {
                workload, model, ..
            } => {
                format!("verify {workload} serves {model}")
            }
            PlanStep::ApplyDefinition { workload, .. } => {
                format!("apply new definition to {workload}")
            }
            PlanStep::VerifyVersion {
                workload,
                packages,
                binary_check,
                ..
            } => {
                let mut what = packages
                    .iter()
                    .map(|(n, v)| format!("{n}={v}"))
                    .collect::<Vec<_>>();
                if let Some(bc) = binary_check {
                    what.push(format!("{} {}~{}", bc.path, bc.arg, bc.expect));
                }
                format!("verify {workload} runs {}", what.join(","))
            }
            PlanStep::UnloadModel {
                workload, model, ..
            } => format!("unload {model} from {workload}"),
        }
    }
}

/// A mutation plan: affected workloads/units/GPUs, expected model and the
/// rollback the executor records for restoration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub id: OperationId,
    pub kind: OpKind,
    pub target: WorkloadName,
    /// Host inventory revision this plan was built against.
    pub revision: u64,
    pub expected_model: String,
    pub affected_gpus: Vec<GpuId>,
    pub steps: Vec<PlanStep>,
    /// Restoration steps for the failure path: stop the target, put
    /// previously-running workloads back. Ordered for execution after a
    /// failed run; an empty rollback means "nothing to restore".
    pub rollback: Vec<PlanStep>,
}
