//! Host inventory: the enrolled host's platform, GPUs and workload
//! definitions, plus the revision that plans are checked against.
//!
//! M1 uses a file-backed fake inventory (see the daemon crate); the types
//! here are the contract. Validation is explicit because rows and files are
//! untrusted once they hit disk.

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};
use crate::identity::{GpuId, HostId, UnitName, WorkloadName};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    Linux,
    Macos,
    Other(String),
}

impl Platform {
    pub fn current() -> Option<Self> {
        match std::env::consts::OS {
            "linux" => Some(Platform::Linux),
            "macos" => Some(Platform::Macos),
            other => Some(Platform::Other(other.to_string())),
        }
    }
}

/// System vs user scope for systemd-backed control authority. Kept separate
/// from [`crate::identity::UnitScopeMarker`], which additionally marks
/// launchd labels and lives on the unit identity itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitScope {
    System,
    User,
}

/// What lets llmctl act on a workload. Unsupported actions must surface as
/// explicit capability errors, never silently degrade to a guess.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlAuthority {
    /// A systemd unit llmctl may start/stop through the manager.
    SystemdUnit { scope: UnitScope, unit: String },
    /// A launchd service label llmctl may manage on a Mac.
    LaunchdService { label: String },
    /// An endpoint we can observe but not move. Lifecycle ownership requires
    /// a known engine recipe on an enrolled host; until then, observe-only.
    ObserveOnly { endpoint: String },
}

impl ControlAuthority {
    pub fn supports_lifecycle(&self) -> bool {
        !matches!(self, ControlAuthority::ObserveOnly { .. })
    }
}

/// One workload definition: stable identity plus effective launch
/// configuration. GPU ids are host-qualified and must exist in the host
/// inventory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadDefinition {
    pub name: WorkloadName,
    /// Engine adapter name (`fake` for the M1 fake engine).
    pub engine: String,
    pub model: String,
    pub gpus: Vec<GpuId>,
    pub port: u16,
    /// Deterministic service identity for this workload.
    pub unit: UnitName,
    pub authority: ControlAuthority,
    /// Absent in older inventory files → external.
    #[serde(default)]
    pub ownership: crate::launch::Ownership,
}

/// The enrolled host and everything a plan needs to reason about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostInventory {
    pub host: HostId,
    pub platform: Platform,
    pub gpus: Vec<GpuId>,
    pub workloads: Vec<WorkloadDefinition>,
    /// Bumped on every change. Plans carry the revision they were built
    /// against; admission revalidates and refuses stale plans instead of
    /// executing an obsolete confirmation.
    pub revision: u64,
}

impl HostInventory {
    pub fn workload(&self, name: &WorkloadName) -> Option<&WorkloadDefinition> {
        self.workloads.iter().find(|w| &w.name == name)
    }

    pub fn has_gpu(&self, gpu: &GpuId) -> bool {
        self.gpus.iter().any(|g| g == gpu)
    }

    /// Structural validation of deserialized or imported data. Call before
    /// trusting an inventory that did not come from this process.
    pub fn validate(&self) -> CoreResult<()> {
        if self.gpus.is_empty() {
            return Err(CoreError::Validation(format!(
                "host {} has no GPUs declared",
                self.host
            )));
        }
        let mut seen = std::collections::HashSet::new();
        for gpu in &self.gpus {
            if gpu.host != self.host {
                return Err(CoreError::Validation(format!(
                    "gpu {} does not belong to host {}",
                    gpu, self.host
                )));
            }
            if !seen.insert(gpu.index) {
                return Err(CoreError::Validation(format!(
                    "duplicate gpu index {} on host {}",
                    gpu.index, self.host
                )));
            }
        }
        let mut names = std::collections::HashSet::new();
        let mut units = std::collections::HashSet::new();
        let mut ports = std::collections::HashSet::new();
        for w in &self.workloads {
            if !names.insert(w.name.clone()) {
                return Err(CoreError::Validation(format!(
                    "duplicate workload name {}",
                    w.name
                )));
            }
            if !units.insert(w.unit.clone()) {
                return Err(CoreError::Validation(format!(
                    "duplicate unit {} (workload {})",
                    w.unit, w.name
                )));
            }
            if !ports.insert(w.port) {
                return Err(CoreError::Validation(format!(
                    "duplicate port {} (workload {})",
                    w.port, w.name
                )));
            }
            if w.gpus.is_empty() {
                return Err(CoreError::InvalidConfig {
                    workload: w.name.to_string(),
                    reason: "no gpus assigned".into(),
                });
            }
            for gpu in &w.gpus {
                if !self.has_gpu(gpu) {
                    return Err(CoreError::UnknownGpu(gpu.to_string()));
                }
            }
            if w.model.trim().is_empty() {
                return Err(CoreError::InvalidConfig {
                    workload: w.name.to_string(),
                    reason: "empty model".into(),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::UnitScopeMarker;

    fn inv() -> HostInventory {
        let host = HostId::new("workstation").unwrap();
        HostInventory {
            host: host.clone(),
            platform: Platform::Linux,
            gpus: vec![GpuId::new(host.clone(), 0), GpuId::new(host, 1)],
            workloads: vec![WorkloadDefinition {
                name: WorkloadName::new("qwen3").unwrap(),
                engine: "fake".into(),
                model: "fake/qwen3".into(),
                gpus: vec![GpuId::new(HostId::new("workstation").unwrap(), 0)],
                port: 8100,
                unit: UnitName::systemd(UnitScopeMarker::User, "llmctl-next-fake-qwen3.service")
                    .unwrap(),
                authority: ControlAuthority::SystemdUnit {
                    scope: UnitScope::User,
                    unit: "llmctl-next-fake-qwen3.service".into(),
                },
                ownership: Default::default(),
            }],
            revision: 7,
        }
    }

    #[test]
    fn validate_accepts_consistent_inventory() {
        assert!(inv().validate().is_ok());
    }

    #[test]
    fn validate_rejects_foreign_gpu_and_duplicates() {
        let mut i = inv();
        let other = GpuId::new(HostId::new("mbp2").unwrap(), 0);
        i.gpus.push(other.clone());
        let mut w = i.workloads[0].clone();
        w.gpus = vec![other];
        i.workloads.push(w);
        i.workloads.push(i.workloads[0].clone());
        assert!(matches!(
            i.validate(),
            Err(CoreError::Validation(_))
                | Err(CoreError::UnknownGpu(_))
                | Err(CoreError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn validate_rejects_two_workloads_on_one_unit() {
        let mut i = inv();
        let mut w = i.workloads[0].clone();
        w.name = WorkloadName::new("qwen3-alias").unwrap();
        w.port = 8101;
        i.workloads.push(w);
        let err = i.validate().unwrap_err();
        assert!(err.to_string().contains("duplicate unit"), "{err}");
    }
}
