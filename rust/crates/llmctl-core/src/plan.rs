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
    /// Write a llama-swap config rendered from lane records and confirm
    /// llama-swap reloaded it (M4b step 4).
    LaneWrite,
    /// Ask llama-swap for one lane and confirm it runs; llama-swap evicts
    /// what the lane cannot share memory with (M4c).
    LaneSwap,
}

impl std::fmt::Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            OpKind::Run => "run",
            OpKind::Stop => "stop",
            OpKind::Restart => "restart",
            OpKind::Upgrade => "upgrade",
            OpKind::Unload => "unload",
            OpKind::LaneWrite => "lane_write",
            OpKind::LaneSwap => "lane_swap",
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
    /// External: make `source`'s bytes the unit's `EnvironmentFile` at
    /// `path` (a profile switch, M4c) and re-pin the workload to it: the
    /// pinned digest of that env file and the model the workload serves.
    /// `source` must still have `source_digest`. A no-op on the file when
    /// `path` already has that digest, so a restore after a failure that
    /// came before the swap changes nothing. The forward step keeps the
    /// bytes it replaces in `backup`; the restore step reads them from
    /// there (`backup: None`). Env contents never enter the journal.
    ApplyEnvFile {
        workload: WorkloadName,
        unit: UnitName,
        path: String,
        source: String,
        source_digest: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        backup: Option<String>,
        model: String,
    },
    /// External: run one companion hook of `workload` (M4c item 5): an
    /// argv without a shell, killed after `timeout_secs`, its output kept
    /// to a bounded tail. A non-zero exit fails the step unless the hook
    /// is `optional`. `phase` says where it belongs (`pre_start`, `undo`…).
    RunHook {
        workload: WorkloadName,
        name: String,
        phase: String,
        argv: Vec<String>,
        timeout_secs: u64,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        optional: bool,
    },
    /// External: request `lane` from the llama-swap on `port` and confirm
    /// it is running. llama-swap does the eviction itself; `evicts` is
    /// what the lane records predicted, kept for the journal and compared
    /// with what really left.
    LoadLane {
        workload: WorkloadName,
        port: u16,
        lane: String,
        evicts: Vec<String>,
        timeout_secs: u64,
    },
    /// External: atomically replace a llama-swap config with `content`,
    /// refusing when the file is no longer the one planned against
    /// (`expect_digest`), then confirm llama-swap reloaded it (its log, then
    /// `/v1/models` serving `lanes`). On success the agent's lane store is
    /// re-imported from `content` and the change committed to the config's
    /// git repo when it has one (history only, best effort).
    WriteSwapConfig {
        workload: WorkloadName,
        path: String,
        content: String,
        expect_digest: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        log_path: Option<String>,
        port: u16,
        lanes: Vec<String>,
        timeout_secs: u64,
        store_path: String,
        summary: String,
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
            | PlanStep::UnloadModel { workload, .. }
            | PlanStep::ApplyEnvFile { workload, .. }
            | PlanStep::LoadLane { workload, .. }
            | PlanStep::RunHook { workload, .. }
            | PlanStep::WriteSwapConfig { workload, .. } => workload,
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
            PlanStep::ApplyEnvFile {
                workload,
                source,
                backup,
                model,
                ..
            } => {
                if backup.is_some() {
                    let name = source.rsplit('/').next().unwrap_or(source);
                    format!("apply profile {name} to {workload} (serves {model})")
                } else {
                    format!("restore the previous env of {workload} (serves {model})")
                }
            }
            PlanStep::RunHook {
                workload,
                name,
                phase,
                optional,
                ..
            } => format!(
                "hook {name} ({phase} of {workload}){}",
                if *optional { ", failure ignored" } else { "" }
            ),
            PlanStep::LoadLane {
                workload,
                lane,
                evicts,
                ..
            } => format!(
                "load lane {lane} on {workload} (evicts {})",
                if evicts.is_empty() { "nothing".to_string() } else { evicts.join(", ") }
            ),
            PlanStep::WriteSwapConfig {
                workload, lanes, summary, ..
            } => format!(
                "write {workload} config ({} lanes): {summary}; confirm the reload",
                lanes.len()
            ),
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
