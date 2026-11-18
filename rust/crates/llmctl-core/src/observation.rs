//! Observations — what the world currently looks like, kept distinct from
//! desired intent. Process alive, endpoint ready, model loaded and successful
//! generation are separate facts; unknown is always representable.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::identity::{GpuId, UnitName, WorkloadName};

/// Observed process state of a workload's unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessState {
    Running,
    Stopped,
    /// The host could not be asked (agent down, manager timeout): not the
    /// same as stopped, and must never authorize lifecycle conclusions.
    Unknown,
}

/// Observed endpoint readiness, independent of process state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Readiness {
    Ready,
    /// Answering but still loading (readiness timeout leaves a workload
    /// `starting`; that is a different outcome from a dead process).
    Loading,
    NotReady,
    Unknown,
}

/// What the endpoint says it serves. An endpoint answering does not prove
/// the *expected* model is ready; adapters verify identity when available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServedModel {
    /// Endpoint reports this model and the adapter verified the identity.
    Verified {
        model: String,
    },
    /// Endpoint reports a model but identity was not verified.
    Unverified {
        model: String,
    },
    Unknown,
    NotServing,
}

/// One workload's observed state with evidence and freshness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub workload: WorkloadName,
    pub unit: Option<UnitName>,
    pub process: ProcessState,
    pub readiness: Readiness,
    pub served_model: ServedModel,
    /// Main pid when the manager names one (M5a: a pid change with no
    /// admitted operation is a foreign transition). `None` = this engine
    /// cannot name pids; never read as "no process".
    #[serde(default)]
    pub pid: Option<u32>,
    pub gpus: Vec<GpuId>,
    /// Models a vendor service (Ollama, LM Studio) holds in GPU memory on
    /// this machine right now. Filled only for planning, and only for a
    /// service that occupies a card the plan needs. `None` = not asked, not
    /// applicable, or the engine could not say — never "nothing loaded".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loaded_models: Option<Vec<String>>,
    pub observed_at: DateTime<Utc>,
}

impl Observation {
    /// True only when process and endpoint both say the workload serves.
    /// Unknown observations answer false — callers must check freshness
    /// separately rather than infer "stopped" from "not confirmed running".
    pub fn confirmed_running(&self) -> bool {
        self.process == ProcessState::Running && self.readiness == Readiness::Ready
    }

    pub fn holds_gpu(&self, gpu: &GpuId) -> bool {
        self.gpus.iter().any(|g| g == gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::HostId;

    fn obs(process: ProcessState, readiness: Readiness) -> Observation {
        let host = HostId::new("workstation").unwrap();
        Observation {
            workload: WorkloadName::new("qwen3").unwrap(),
            unit: None,
            process,
            readiness,
            served_model: ServedModel::Unknown,
            pid: None,
            gpus: vec![GpuId::new(host, 0)],
            loaded_models: None,
            observed_at: Utc::now(),
        }
    }

    #[test]
    fn unknown_is_not_running_and_not_stopped() {
        let o = obs(ProcessState::Unknown, Readiness::Unknown);
        assert!(!o.confirmed_running());
        assert_ne!(o.process, ProcessState::Stopped);
    }

    #[test]
    fn running_process_with_loading_endpoint_is_not_confirmed() {
        assert!(!obs(ProcessState::Running, Readiness::Loading).confirmed_running());
        assert!(obs(ProcessState::Running, Readiness::Ready).confirmed_running());
    }
}
