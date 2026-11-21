//! Per-agent settings in `<data-dir>/agent.toml` (M7a,
//! docs/RUST-M7-PLAN.md, decision 4).
//!
//! Integrations that only exist on some hosts (llm-router's config, the
//! longjob lane lease) are off unless the agent's own config names them.
//! Before M7 they had defaults under the owner's `~/AI` tree; a clean host
//! has no such tree.
//!
//! Precedence for each path: the environment override first
//! (`LLMCTL_ROUTER_CONFIG`, `LLMCTL_NEXT_LANE_LEASE`), then the key here,
//! then nothing (the integration is off). The daemon and the CLI resolve
//! through the same functions, so a longjob job's CLI call and the daemon
//! agree on which lease file binds.
//!
//! Example:
//!
//! ```toml
//! router_config = "~/AI/llm-router/config/router.yaml"
//! lane_lease = "~/AI/logs/lane-lease.json"
//! ```

use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// File name inside the data dir.
pub const FILE: &str = "agent.toml";

/// Env override for llm-router's config (`~/` expanded). Empty = unset.
pub const ROUTER_CONFIG_ENV: &str = "LLMCTL_ROUTER_CONFIG";

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentConfig {
    /// llm-router's `router.yaml` (profile_env backends for presets).
    #[serde(default)]
    pub router_config: Option<PathBuf>,
    /// The lane-lease JSON file that longjob, hf2local and llm-router use.
    #[serde(default)]
    pub lane_lease: Option<PathBuf>,
}

impl AgentConfig {
    /// Read `<data_dir>/agent.toml`. A missing file is the default (every
    /// integration off); an unreadable or invalid file is an error, so a
    /// typo never silently turns an integration off.
    pub fn load(data_dir: &Path) -> Result<Self, String> {
        let path = data_dir.join(FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(format!("reading {}: {e}", path.display())),
        };
        Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut cfg: Self = toml::from_str(text).map_err(|e| e.to_string())?;
        cfg.router_config = cfg.router_config.map(|p| expand_tilde(&p));
        cfg.lane_lease = cfg.lane_lease.map(|p| expand_tilde(&p));
        Ok(cfg)
    }
}

/// `~/x` → `$HOME/x`; anything else unchanged.
pub fn expand_tilde(p: &Path) -> PathBuf {
    match (
        p.to_str().and_then(|s| s.strip_prefix("~/")),
        std::env::var_os("HOME"),
    ) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => p.to_path_buf(),
    }
}

/// The daemon's settings for this process (set once at start by the
/// agent; the default, all off, until then).
static CURRENT: RwLock<Option<AgentConfig>> = RwLock::new(None);

/// Install the daemon's settings (called by `App::open`).
pub fn install(cfg: AgentConfig) {
    *CURRENT.write().unwrap_or_else(|e| e.into_inner()) = Some(cfg);
}

/// The installed settings, or the default when none were installed.
pub fn current() -> AgentConfig {
    CURRENT
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// llm-router's config: env override, else the key, else `None` (off).
pub fn router_config(cfg: &AgentConfig) -> Option<PathBuf> {
    match std::env::var_os(ROUTER_CONFIG_ENV) {
        Some(v) if !v.is_empty() => Some(expand_tilde(Path::new(&v))),
        _ => cfg.router_config.clone(),
    }
}

/// The lane lease: env override (`off` disables), else the key, else
/// `None` (off).
pub fn lane_lease(cfg: &AgentConfig) -> Option<PathBuf> {
    use crate::lease::{env_override, Override, LEASE_ENV};
    match env_override(LEASE_ENV) {
        Override::Off => None,
        Override::Path(p) => Some(p),
        Override::Unset => cfg.lane_lease.clone(),
    }
}

/// Where these integrations lived before M7, for the start-up warning
/// only: when a key is unset but the old file exists, the agent says the
/// file is ignored. Remove after the M6 deploy has written the keys.
pub fn pre_m7_defaults() -> Option<(PathBuf, PathBuf)> {
    let home = PathBuf::from(std::env::var_os("HOME")?);
    Some((
        home.join("AI/llm-router/config/router.yaml"),
        home.join("AI/logs/lane-lease.json"),
    ))
}

/// One line per integration whose key is unset while its pre-M7 file
/// exists (the agent logs these at start).
pub fn ignored_legacy_files(cfg: &AgentConfig) -> Vec<String> {
    let Some((router, lease)) = pre_m7_defaults() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if router_config(cfg).is_none() && router.exists() {
        out.push(format!(
            "router_config not set; {} exists and is ignored (set it in {FILE})",
            router.display()
        ));
    }
    if lane_lease(cfg).is_none() && lease.exists() {
        out.push(format!(
            "lane_lease not set; {} exists and is ignored (set it in {FILE})",
            lease.display()
        ));
    }
    out
}

/// One-line state of each integration, for `status`.
pub fn integration_lines(cfg: &AgentConfig) -> Vec<String> {
    let show = |name: &str, p: Option<PathBuf>| match p {
        Some(p) => format!("{name}: {}", p.display()),
        None => format!("{name}: off (not set in {FILE})"),
    };
    vec![
        show("router config", router_config(cfg)),
        show("lane lease", lane_lease(cfg)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_is_all_off_and_bad_files_are_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            AgentConfig::load(dir.path()).unwrap(),
            AgentConfig::default()
        );
        std::fs::write(dir.path().join(FILE), "lane_leese = \"/x\"\n").unwrap();
        let err = AgentConfig::load(dir.path()).unwrap_err();
        assert!(err.contains("lane_leese"), "{err}");
        std::fs::write(dir.path().join(FILE), "router_config = 3\n").unwrap();
        assert!(AgentConfig::load(dir.path()).is_err());
    }

    #[test]
    fn keys_are_read_and_tilde_expanded() {
        let cfg = AgentConfig::parse(
            "router_config = \"/etc/router.yaml\"\nlane_lease = \"~/lease.json\"\n",
        )
        .unwrap();
        assert_eq!(cfg.router_config, Some(PathBuf::from("/etc/router.yaml")));
        let home = PathBuf::from(std::env::var_os("HOME").unwrap());
        assert_eq!(cfg.lane_lease, Some(home.join("lease.json")));
    }
}
