//! Desired intent — what the owner wants a workload to be, independent of
//! what observation currently says. Intent is durable on the target host and
//! survives daemon restarts; a manual stop is intent, not an observation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum DesiredIntent {
    /// Should be serving; recovery may pursue this.
    Running,
    /// Should be stopped. `manual: true` marks an owner stop or maintenance
    /// window: it persists across restarts and recovery must never override
    /// it, and switch plans must not "restore" a manually stopped workload.
    Stopped { manual: bool },
    /// Not ours to move (externally owned or migrated-but-not-adopted).
    Unmanaged,
}

impl DesiredIntent {
    pub fn is_running(&self) -> bool {
        matches!(self, DesiredIntent::Running)
    }

    /// Manual stops (and unmanaged targets) are protected from automatic
    /// restarts and from being silently flipped to running by a plan.
    pub fn restart_protected(&self) -> bool {
        matches!(
            self,
            DesiredIntent::Stopped { manual: true } | DesiredIntent::Unmanaged
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let v = DesiredIntent::Stopped { manual: true };
        let j = serde_json::to_string(&v).unwrap();
        assert_eq!(j, r#"{"state":"stopped","manual":true}"#);
        assert_eq!(serde_json::from_str::<DesiredIntent>(&j).unwrap(), v);
    }
}
