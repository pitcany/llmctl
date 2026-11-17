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
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    // Coverage added by the 2026-11-16 pilot: the first real unit set
    // RestartSec/TimeoutStartSec/StartLimit*/SyslogIdentifier/After, which
    // the import did not read, so a rendered unit silently dropped them.
    // `None` = not covered by this pin (older pins): the diff ignores the
    // field, and `own` refuses a source that sets a directive the pin does
    // not cover.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restart_usec: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_start_usec: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_limit_burst: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_limit_interval_usec: Option<u64>,
    /// `Some("")` = covered and unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub syslog_identifier: Option<String>,
    /// Effective ordering/dependency lists as the manager reports them
    /// (sorted; implicit dependencies included, which render harmlessly).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wants: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<Vec<String>>,
    /// launchd plist facts with no systemd counterpart (macOS workloads).
    /// `None` on systemd pins; the diff/digest machinery covers it like any
    /// other field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub launchd: Option<LaunchdExtras>,
}

/// The subset of a launchd plist the launch adapter pins. launchd has no
/// drop-ins or merging, so this plus the shared fields is the whole job
/// definition (RUST-M3-PLAN, design decision 5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchdExtras {
    /// The launchd job label (`Label`).
    pub label: String,
    /// Absolute path of the job's plist file.
    pub plist_path: String,
    /// `Program` when the plist names the binary directly (no
    /// `ProgramArguments`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program: Option<String>,
    /// `RunAtLoad`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub run_at_load: bool,
    /// `KeepAlive`: bool, or the raw JSON when launchd's richer dict form
    /// is used (kept verbatim so render/import round-trips exactly).
    pub keep_alive: KeepAliveValue,
    /// `StandardOutPath` / `StandardErrorPath` — the log source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdout_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stderr_path: Option<String>,
    /// `ThrottleInterval` (seconds).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub throttle_interval_secs: Option<u64>,
    /// `ProcessType` (`Interactive`, `Standard`, `Background`, `Adaptive`):
    /// launchd's CPU/IO scheduling class. Dropping it demotes the job.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub process_type: Option<String>,
    /// `Disabled` as written in the plist (the launchd override database is
    /// separate state the adapter manages through enable/disable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disabled: Option<bool>,
}

/// `KeepAlive` as a plist value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum KeepAliveValue {
    Bool(bool),
    /// launchd's dict conditions, kept verbatim.
    Other(serde_json::Value),
}

impl KeepAliveValue {
    /// True only for `KeepAlive=true` (the respawn-always form; a dict is a
    /// conditional form this adapter does not interpret).
    pub fn is_always(&self) -> bool {
        matches!(self, KeepAliveValue::Bool(true))
    }
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
        // Only fields both sides carry: a field an older pin does not cover
        // (absent) is not drift.
        a.iter()
            .filter(|(k, v)| matches!(b.get(*k), Some(other) if other != *v))
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
        /// launchd only (RUST-M3-PLAN, design decision 6): the plist file
        /// on disk differs from the bytes launchd loaded, and the next
        /// admitted restart (bootstrap) applies it. systemd's
        /// `daemon-reload` made every owned write live immediately, so this
        /// stays `None` there.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pending_apply: Option<PendingApply>,
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

/// launchd deferred-apply marker: what still needs a bootstrap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingApply {
    /// Human-readable reason (own wrote the rendered plist; release
    /// restored the original bytes; …).
    pub reason: String,
    pub since: DateTime<Utc>,
}

/// Versioned-environment facts pinned next to a managed definition (plan
/// D): what the interpreter reports, plus content digests of the non-system
/// `Exec*` files (the launcher script). The drift guard recomputes these;
/// an in-place `pip install -U` or a launcher edit is `definition_drift`.
///
/// M4a extends the pin with one per-kind fact (owner decision 2): the
/// archive's version + checksum (release-archive), the git SHA plus the
/// binary's own `--version` where it has one (source-build; ds4 has none,
/// so its SHA is the identity), or the running server's reported version
/// (vendor-app). `fact` is absent on python-env pins and on pins made
/// before M4a; the drift guard only compares what the pin carries.
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
    /// Per-kind version fact for the non-python recipe kinds (M4a).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fact: Option<VersionFact>,
}

/// The per-kind identity of a non-python-env install (M4a, decision 2).
/// A mismatch between the pin and the live fact is a drift finding,
/// never auto-corrected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VersionFact {
    /// A released archive unpacked into a versioned directory.
    ReleaseArchive {
        /// The versioned install directory (absolute).
        dir: String,
        /// Digest of the downloaded archive, when one was used.
        archive_sha256: Option<String>,
        /// Digest of the engine binary inside the dir.
        binary_digest: String,
        /// What the binary's version argument printed, when probed.
        reported: Option<String>,
    },
    /// A source checkout built into a versioned directory.
    SourceBuild {
        /// The versioned build directory (absolute).
        dir: String,
        /// The git commit built (the identity when the binary has no
        /// `--version`, e.g. ds4).
        commit: String,
        binary_digest: String,
        reported: Option<String>,
    },
    /// A vendor-installed app, observed through its own interface.
    VendorApp {
        /// The version the running server/app reports (absent when it
        /// could not be observed — itself a finding).
        running: Option<String>,
        /// How it was probed (`http_json /api/version`, `lms version`, ...).
        probe: String,
    },
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
            ..Default::default()
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
