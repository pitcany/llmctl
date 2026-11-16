//! Effective launch definitions and workload ownership.
//!
//! A [`LaunchDefinition`] is what a supervised unit actually runs, as the
//! service manager reports it (not as a unit file happens to be written).
//! Adoption imports one, proves a generated service definition is
//! equivalent, and pins its digest; later lifecycle actions refuse when the
//! live definition has drifted from the pinned one.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// One `Exec*=` command line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecCommand {
    /// Binary path systemd executes.
    pub path: String,
    /// Exact argument vector from the manager's D-Bus API (`systemctl
    /// show` joins it with spaces and loses argument boundaries).
    pub argv: Vec<String>,
    /// `-` prefix: a non-zero exit is not a failure.
    pub ignore_errors: bool,
}

/// An `EnvironmentFile=` reference. Contents are never stored or shown —
/// only a digest, so an edited profile env counts as drift.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvFileRef {
    pub path: String,
    /// `-` prefix: a missing file is tolerated.
    pub optional: bool,
    /// `sha256:<hex>` of the contents, or `missing`.
    pub content_digest: String,
}

/// The effective launch configuration of one unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchDefinition {
    pub service_type: String,
    pub exec_start_pre: Vec<ExecCommand>,
    pub exec_start: Vec<ExecCommand>,
    pub exec_start_post: Vec<ExecCommand>,
    pub exec_stop: Vec<ExecCommand>,
    pub working_directory: Option<String>,
    /// Inline `Environment=` assignments, sorted by name. Adoption refuses
    /// secret-looking names, so values here are not credentials.
    pub environment: Vec<(String, String)>,
    pub environment_files: Vec<EnvFileRef>,
    pub restart: String,
    pub timeout_stop_usec: u64,
    pub kill_mode: String,
}

impl LaunchDefinition {
    /// `sha256:<hex>` over the canonical JSON form. Field order is fixed by
    /// the struct, so equal definitions always digest equally.
    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("definition serializes");
        format!("sha256:{:x}", Sha256::digest(bytes))
    }

    /// Names of the fields that differ, with both values, for drift and
    /// equivalence reports.
    pub fn diff(&self, other: &LaunchDefinition) -> Vec<FieldDiff> {
        let a = serde_json::to_value(self).expect("definition serializes");
        let b = serde_json::to_value(other).expect("definition serializes");
        let (Some(a), Some(b)) = (a.as_object(), b.as_object()) else {
            return Vec::new();
        };
        a.iter()
            .filter(|(k, v)| b.get(*k) != Some(*v))
            .map(|(k, v)| FieldDiff {
                field: k.clone(),
                expected: v.clone(),
                actual: b.get(k).cloned().unwrap_or(serde_json::Value::Null),
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDiff {
    pub field: String,
    pub expected: serde_json::Value,
    pub actual: serde_json::Value,
}

/// A unit file that contributed to the effective definition, for
/// provenance (informational; the digest pins the effective definition).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileProvenance {
    pub path: String,
    pub content_digest: String,
}

/// Who owns a workload's launch definition.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Ownership {
    /// Configured outside llmctl (fake workloads, enrolled-but-not-adopted
    /// units). Lifecycle may be allowed; the definition is not pinned.
    #[default]
    External,
    /// Adopted: llmctl holds the authoritative definition and refuses to
    /// start a unit whose live definition no longer matches it.
    Managed {
        definition: Box<LaunchDefinition>,
        digest: String,
        adopted_at: DateTime<Utc>,
        provenance: Vec<FileProvenance>,
        /// Set once llmctl has *written* the unit file itself (`own`):
        /// where the rendered unit lives and what was there before, so
        /// `release` can hand the exact original back.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        service_file: Option<ServiceFileRecord>,
        /// Version facts pinned next to the definition (plan D).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        version: Option<VersionRef>,
        /// Which checks the drift guard performs. 1: the effective
        /// definition only (legacy pins). 2: definition + service file +
        /// version facts. Re-pinned on load, so legacy pins never read as
        /// false drift.
        #[serde(default = "default_definition_schema")]
        definition_schema: u32,
    },
}

fn default_definition_schema() -> u32 {
    1
}

/// The unit file llmctl owns, plus the release target it replaced.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceFileRecord {
    /// Absolute path of the unit file (the link itself when it was one).
    pub path: String,
    /// `sha256:<hex>` of the bytes llmctl wrote.
    pub digest: String,
    pub previous: PreviousUnitFile,
}

/// Versioned-environment facts pinned next to a managed definition (plan
/// D): what the interpreter reports, plus content digests of the non-system
/// `Exec*` files (the launcher script). The drift guard recomputes these;
/// an in-place `pip install -U` or a launcher edit is `definition_drift`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionRef {
    /// The interpreter the recipe resolves (absolute path).
    pub interpreter: Option<String>,
    /// Package versions as the env reports them, sorted by name.
    pub packages: Vec<(String, String)>,
    /// CUDA runtime version as the env reports it (`torch.version.cuda`).
    pub cuda: Option<String>,
    /// `sha256:<hex>` over the env's sorted `pip freeze` output.
    pub lock_digest: String,
    /// Content digests of non-system `Exec*` paths.
    pub exec_digests: Vec<(String, String)>,
}

/// What `own` replaced — everything `release` needs to restore exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PreviousUnitFile {
    /// A regular file; its bytes are kept at `backup_path` (outside the
    /// inventory, which never stores file contents).
    File {
        content_digest: String,
        backup_path: String,
    },
    /// A symlink; re-created pointing at `target` (the repo copy is never
    /// touched in either direction).
    Link { target: String },
}

/// Inline environment names adoption refuses to import: credentials belong
/// in an `EnvironmentFile=`, which is referenced and digested, never read
/// into the inventory.
pub fn looks_secret(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    [
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "API_KEY",
        "APIKEY",
        "CREDENTIAL",
        "AUTH",
    ]
    .iter()
    .any(|needle| upper.contains(needle))
        || upper.ends_with("_KEY")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def() -> LaunchDefinition {
        LaunchDefinition {
            service_type: "simple".into(),
            exec_start_pre: vec![],
            exec_start: vec![ExecCommand {
                path: "/usr/bin/python3".into(),
                argv: vec!["/usr/bin/python3".into(), "server.py".into(), "8000".into()],
                ignore_errors: false,
            }],
            exec_start_post: vec![],
            exec_stop: vec![],
            working_directory: None,
            environment: vec![("PORT".into(), "8000".into())],
            environment_files: vec![],
            restart: "no".into(),
            timeout_stop_usec: 90_000_000,
            kill_mode: "control-group".into(),
        }
    }

    #[test]
    fn digest_is_stable_and_sensitive() {
        let a = def();
        assert_eq!(a.digest(), def().digest());
        assert!(a.digest().starts_with("sha256:"));
        let mut b = def();
        b.exec_start[0].argv.push("--x".into());
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn diff_names_changed_fields() {
        let a = def();
        let mut b = def();
        b.restart = "on-failure".into();
        b.environment.push(("EXTRA".into(), "1".into()));
        let fields: Vec<_> = a.diff(&b).into_iter().map(|d| d.field).collect();
        assert_eq!(fields, vec!["environment", "restart"]);
        assert!(a.diff(&a).is_empty());
    }

    #[test]
    fn secret_names_are_recognised() {
        for name in [
            "HF_TOKEN",
            "OPENAI_API_KEY",
            "db_password",
            "AWS_SECRET_ACCESS_KEY",
            "SSH_KEY",
        ] {
            assert!(looks_secret(name), "{name}");
        }
        for name in ["PORT", "CUDA_VISIBLE_DEVICES", "VLLM_KEYS_DIR", "MODEL"] {
            assert!(!looks_secret(name), "{name}");
        }
    }

    #[test]
    fn ownership_defaults_to_external_when_absent() {
        #[derive(Deserialize)]
        struct W {
            #[serde(default)]
            ownership: Ownership,
        }
        let w: W = serde_json::from_str("{}").unwrap();
        assert_eq!(w.ownership, Ownership::External);
    }
}
