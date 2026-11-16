//! Host-qualified identities.
//!
//! Resource and operation identifiers include their host: a GPU is never
//! "GPU 0" in the abstract, it is GPU 0 *on a specific host*, and the same
//! holds for plans and operations once hosts other than the workstation are
//! enrolled. Newtypes enforce the distinction; construction validates, so a
//! deserialized value still has to pass [`validate`](Self::validate) (or be
//! built through `parse`) before it is trusted.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::CoreError;

/// A non-empty slug (`a-z0-9-`, at most 64 chars) naming an enrolled host.
/// The enrollment binds it stably; an address change does not mint a new one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct HostId(String);

impl HostId {
    pub fn new(raw: impl Into<String>) -> Result<Self, CoreError> {
        let raw = raw.into();
        validate_slug(&raw, "host id").map_err(CoreError::InvalidIdentity)?;
        Ok(Self(raw))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HostId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<HostId> for String {
    fn from(v: HostId) -> String {
        v.0
    }
}

impl TryFrom<String> for HostId {
    type Error = CoreError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        HostId::new(value)
    }
}

/// A workload's stable, human-chosen name (`qwen3-coder`, not a session id).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct WorkloadName(String);

impl WorkloadName {
    pub fn new(raw: impl Into<String>) -> Result<Self, CoreError> {
        let raw = raw.into();
        validate_slug(&raw, "workload name").map_err(CoreError::InvalidIdentity)?;
        Ok(Self(raw))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WorkloadName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<WorkloadName> for String {
    fn from(v: WorkloadName) -> String {
        v.0
    }
}

impl TryFrom<String> for WorkloadName {
    type Error = CoreError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        WorkloadName::new(value)
    }
}

/// One GPU on one host. Host-qualified by construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GpuId {
    pub host: HostId,
    pub index: u32,
}

impl GpuId {
    pub fn new(host: HostId, index: u32) -> Self {
        Self { host, index }
    }
}

impl fmt::Display for GpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/gpu/{}", self.host, self.index)
    }
}

/// A deterministic service-manager unit identity with its scope.
/// Scope is part of the target's identity — user and system units with the
/// same name are different units.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnitName {
    pub scope: UnitScopeMarker,
    /// Bare unit/label name, e.g. `llmctl-next-fake-qwen3.service`.
    pub name: String,
}

/// Which service manager domain a unit lives in.
///
/// Linux system vs user scope maps to [`UnitScopeMarker::System`] /
/// [`UnitScopeMarker::User`]; macOS launchd targets use their own marker so
/// Linux cgroup assumptions cannot silently carry over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitScopeMarker {
    System,
    User,
    Launchd,
}

impl UnitName {
    pub fn systemd(scope: UnitScopeMarker, name: impl Into<String>) -> Result<Self, CoreError> {
        let name = name.into();
        if name.is_empty() || name.len() > 160 {
            return Err(CoreError::InvalidIdentity("unit name out of range".into()));
        }
        match scope {
            UnitScopeMarker::Launchd => Err(CoreError::InvalidIdentity(
                "launchd labels use UnitName::launchd".into(),
            )),
            UnitScopeMarker::System | UnitScopeMarker::User => Ok(Self { scope, name }),
        }
    }

    pub fn launchd(label: impl Into<String>) -> Result<Self, CoreError> {
        let name = label.into();
        if name.is_empty() || name.len() > 160 {
            return Err(CoreError::InvalidIdentity(
                "launchd label out of range".into(),
            ));
        }
        Ok(Self {
            scope: UnitScopeMarker::Launchd,
            name,
        })
    }
}

impl fmt::Display for UnitName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.scope {
            UnitScopeMarker::System => write!(f, "system/{}", self.name),
            UnitScopeMarker::User => write!(f, "user/{}", self.name),
            UnitScopeMarker::Launchd => write!(f, "launchd/{}", self.name),
        }
    }
}

/// Durable, host-scoped operation identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct OperationId(String);

impl OperationId {
    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for OperationId {
    type Error = CoreError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if uuid::Uuid::parse_str(&value).is_err() {
            return Err(CoreError::InvalidIdentity(format!(
                "not an operation id: {value:?}"
            )));
        }
        Ok(Self(value))
    }
}

impl fmt::Display for OperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<OperationId> for String {
    fn from(v: OperationId) -> String {
        v.0
    }
}

/// Caller-chosen, host-scoped idempotency key. Repeating a key with an
/// identical request returns the same operation; reusing it for a different
/// request is refused. Charset keeps it filesystem/journal friendly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct IdempotencyKey(String);

impl IdempotencyKey {
    pub fn new(raw: impl Into<String>) -> Result<Self, CoreError> {
        let raw = raw.into();
        let ok = (1..=128).contains(&raw.len())
            && raw
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.');
        if !ok {
            return Err(CoreError::InvalidIdentity(
                "idempotency key must be 1..=128 chars of [A-Za-z0-9._-]".into(),
            ));
        }
        Ok(Self(raw))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for IdempotencyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<IdempotencyKey> for String {
    fn from(v: IdempotencyKey) -> String {
        v.0
    }
}

impl TryFrom<String> for IdempotencyKey {
    type Error = CoreError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        IdempotencyKey::new(value)
    }
}

fn validate_slug(raw: &str, what: &str) -> Result<(), String> {
    let ok = (1..=64).contains(&raw.len())
        && raw
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{what} must be 1..=64 chars of [a-z0-9-]: got {raw:?}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_reject_bad_input() {
        assert!(HostId::new("workstation").is_ok());
        assert!(HostId::new("").is_err());
        assert!(HostId::new("Has Space").is_err());
        assert!(WorkloadName::new("qwen3-coder").is_ok());
        assert!(WorkloadName::new("bad/name").is_err());
    }

    #[test]
    fn ids_reject_garbage() {
        assert!(<String as TryInto<OperationId>>::try_into("nope".to_string()).is_err());
        let gen = OperationId::generate();
        let parsed: OperationId = gen.to_string().try_into().unwrap();
        assert_eq!(parsed, gen);
        // deserializing an invalid id through serde fails closed
        let bad = serde_json::from_str::<OperationId>("\"not-a-uuid\"");
        assert!(bad.is_err());
    }

    #[test]
    fn idempotency_keys_validate_charset() {
        assert!(IdempotencyKey::new("run-qwen3-20261115-a").is_ok());
        assert!(IdempotencyKey::new("has space").is_err());
        assert!(IdempotencyKey::new("x".repeat(129)).is_err());
    }
}
