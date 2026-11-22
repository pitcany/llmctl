//! Host enrollment (RUST-M3-PLAN D): the workstation's record of each
//! remote agent — `{host name, IOPlatformUUID, ssh alias, agent version,
//! protocol}` — plus the ssh-driven install/upgrade/remove flows.
//!
//! Identity rule (decision 3): the IOPlatformUUID is the host. An address
//! or alias change keeps the host; a *different* UUID behind the same
//! enrolled name is refused — never silently re-bound.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use llmctl_client::wire::HealthResponse;
use llmctl_client::ControlClient;

/// The agent's launchd label on a Mac.
pub const AGENT_LABEL: &str = "com.pitcany.llmctl-next";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostRecord {
    /// Enrolled host id (a slug; also how `--host` addresses it).
    pub name: String,
    /// `IOPlatformUUID` — the machine's stable identity.
    pub io_platform_uuid: String,
    /// ssh alias from ~/.ssh/config (route order lives there).
    pub ssh_alias: String,
    /// Where the agent binary lives on the host.
    pub remote_bin: String,
    /// Agent data dir on the host (socket: `<dir>/control.sock`).
    pub remote_data_dir: String,
    /// Arch as `uname -m` reported at enrollment (`arm64`, ...).
    pub arch: String,
    /// Agent version + protocol as the last health check saw them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<u32>,
    pub enrolled_at: DateTime<Utc>,
    /// Last successful `/v1/health` from this workstation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Hosts {
    #[serde(default)]
    pub hosts: Vec<HostRecord>,
}

/// The enrollment store: `<data-dir>/hosts.json`.
pub fn hosts_path() -> PathBuf {
    llmctl_daemon::app::default_data_dir().join("hosts.json")
}

/// Cache of the last successful remote status (freshness display when a
/// host is unreachable — observations shown stale, never "stopped").
pub fn status_cache_path(name: &str) -> PathBuf {
    llmctl_daemon::app::default_data_dir()
        .join("hosts")
        .join(format!("{name}.status.json"))
}

pub fn load_hosts() -> Hosts {
    match std::fs::read_to_string(hosts_path()) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => Hosts::default(),
    }
}

pub fn save_hosts(hosts: &Hosts) -> Result<(), String> {
    let path = hosts_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(hosts).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("writing {}: {e}", path.display()))
}

pub fn find_host<'a>(hosts: &'a Hosts, name: &str) -> Option<&'a HostRecord> {
    hosts
        .hosts
        .iter()
        .find(|h| h.name == name || h.ssh_alias == name)
}

// ---- ssh helpers (sync: the CLI process owns its lifetime) -------------------

/// Run `ssh <alias> <cmd>`, return stdout. ssh's own errors surface as
/// Err with its stderr so callers can show why.
pub fn ssh(alias: &str, cmd: &str) -> Result<String, String> {
    let out = std::process::Command::new("ssh")
        .arg(alias)
        .arg(cmd)
        .output()
        .map_err(|e| format!("spawning ssh: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "ssh {alias} {cmd:?} exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Push local bytes to `remote_path` through `ssh ... cat >`, then verify
/// the remote sha256 matches (macOS: `shasum -a 256`).
pub fn push_bytes(
    alias: &str,
    bytes: &[u8],
    remote_path: &str,
    expect_sha256: &str,
) -> Result<(), String> {
    let mut child = std::process::Command::new("ssh")
        .arg(alias)
        .arg(format!(
            "mkdir -p \"$(dirname {remote_path})\" && cat > {remote_path}"
        ))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawning ssh for push: {e}"))?;
    use std::io::Write as _;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(bytes)
        .map_err(|e| format!("writing to ssh pipe: {e}"))?;
    let out = child
        .wait_with_output()
        .map_err(|e| format!("waiting for ssh push: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "pushing {remote_path}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let sum = ssh(alias, &format!("shasum -a 256 < {remote_path}"))?;
    let got = sum.split_whitespace().next().unwrap_or_default().to_string();
    if got != expect_sha256 {
        return Err(format!(
            "digest mismatch for {remote_path}: remote {got}, expected {expect_sha256}"
        ));
    }
    Ok(())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// The agent's data dir on an enrolled host (explicit everywhere: the
/// plist and the remote bridge command both pin it, so the agent never
/// silently binds the dev default `llmctl-next-dev`). Kept as a constant
/// for documentation; both users spell it out with the remote home.
#[allow(dead_code)]
pub const REMOTE_DATA_DIR: &str = "~/.local/state/llmctl-next";

/// The agent LaunchAgent plist: KeepAlive + RunAtLoad, its own data dir,
/// logs under ~/Library/Logs (M3 design decision 2). `home` is the
/// *remote* user's home as ssh reports it — plist paths must be absolute.
/// `host_name` is the enrollment name: the agent answers to it as its host
/// id (before, every agent stayed `workstation`).
pub fn agent_plist(home: &str, host_name: &str) -> Result<String, String> {
    let dict = serde_json::json!({
        "Label": AGENT_LABEL,
        "ProgramArguments": [
            format!("{home}/.local/bin/llmctl-next"),
            "--data-dir",
            format!("{home}/.local/state/llmctl-next"),
            "daemon",
            "--host-id",
            host_name
        ],
        "RunAtLoad": true,
        "KeepAlive": true,
        "StandardOutPath": format!("{home}/Library/Logs/llmctl-next/agent.out.log"),
        "StandardErrorPath": format!("{home}/Library/Logs/llmctl-next/agent.err.log"),
    });
    llmctl_daemon::plistdef::render_xml_plist(&dict)
}

/// Read the arch and IOPlatformUUID facts from the host over ssh.
pub fn probe_host(alias: &str) -> Result<(String, String), String> {
    let arch = ssh(alias, "uname -m")?.trim().to_string();
    let ioreg = ssh(alias, "ioreg -rd1 -c IOPlatformExpertDevice")?;
    let uuid = ioreg
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once('=')?;
            k.trim()
                .contains("IOPlatformUUID")
                .then(|| v.trim().trim_matches('"').to_string())
        })
        .ok_or_else(|| "could not read IOPlatformUUID over ssh".to_string())?;
    if arch.is_empty() || uuid.is_empty() {
        return Err("host probe returned empty facts".into());
    }
    Ok((arch, uuid))
}

/// Health through the SSH bridge (also proves the whole transport).
pub async fn bridge_health(alias: &str) -> Result<HealthResponse, String> {
    let client = ControlClient::connect_bridge(alias);
    client
        .health()
        .await
        .map_err(|e| format!("agent health over bridge: {e}"))
}

/// Build the agent on the host from a local source tree (M3-D.3: built on
/// the Mac itself in M3): rsync the workspace (target/ excluded), build
/// release, install to ~/.local/bin/llmctl-next via atomic rename.
pub fn build_on_host(alias: &str, source_dir: &Path) -> Result<(), String> {
    let src = source_dir
        .to_str()
        .ok_or_else(|| "source path is not utf-8".to_string())?;
    let out = std::process::Command::new("rsync")
        .args([
            "-a",
            "--delete",
            "--exclude",
            "target/",
            &format!("{src}/"),
            &format!("{alias}:llmctl-next-src/"),
        ])
        .status()
        .map_err(|e| format!("spawning rsync: {e} (installed on both ends?)"))?;
    if !out.success() {
        return Err(format!("rsync {src} to {alias}:llmctl-next-src/ failed"));
    }
    let build = ssh(
        alias,
        "cd llmctl-next-src && cargo build --release 2>&1 | tail -5",
    )?;
    if !build.contains("Finished") {
        return Err(format!("remote cargo build did not finish:\n{build}"));
    }
    ssh(
        alias,
        "mkdir -p ~/.local/bin && mv llmctl-next-src/target/release/llmctl-next \
         ~/.local/bin/llmctl-next.new && chmod +x ~/.local/bin/llmctl-next.new \
         && mv ~/.local/bin/llmctl-next.new ~/.local/bin/llmctl-next",
    )?;
    Ok(())
}

/// launchd polls in an agent restart: 120 × 0.25 s = 30 s, for the old job
/// to go and again for a bootstrap it refuses meanwhile.
const RESTART_TRIES: u32 = 120;
const RESTART_PAUSE: &str = "0.25";

/// The remote command that restarts the agent's LaunchAgent.
///
/// `bootout` returns while the agent still shuts down, and for a moment
/// after it launchd refuses a bootstrap with error 5 (Input/output error),
/// also when `print` no longer finds the job (macOS 27.2.1, 2026-11-22).
/// Either left the agent down. So: wait until the job is gone, then retry
/// that error only.
///
/// It is one remote command, so no new ssh login is needed while the agent
/// is down, and it runs under /bin/sh whatever the login shell is (no
/// single quote may appear in the script). `print` output is discarded (it
/// shows the job's environment). `env -i`: a bootstrap inherits the
/// caller's exported variables; the agent gets none.
fn agent_restart_command(
    launchctl: &str,
    label: &str,
    plist: &str,
    tries: u32,
    pause: &str,
) -> String {
    let script = format!(
        r#"d="gui/$(id -u)"
t="$d/{label}"
{launchctl} bootout "$t" >/dev/null 2>&1
n=0
while {launchctl} print "$t" >/dev/null 2>&1; do
  n=$((n + 1))
  if [ "$n" -ge {tries} ]; then echo "$t is still loaded after bootout" >&2; exit 1; fi
  sleep {pause}
done
n=0
while :; do
  if err=$(env -i {launchctl} bootstrap "$d" "{plist}" 2>&1); then exit 0; fi
  case "$err" in *"Bootstrap failed: 5:"*) ;; *) break ;; esac
  n=$((n + 1))
  if [ "$n" -ge {tries} ]; then break; fi
  sleep {pause}
done
echo "$err" >&2
exit 1"#
    );
    format!("/bin/sh -c '{script}'")
}

/// Install the agent's LaunchAgent and wait for its control socket.
pub fn bootstrap_agent(alias: &str, host_name: &str) -> Result<(), String> {
    let remote_home = ssh(alias, "echo $HOME")?
        .trim()
        .to_string();
    let plist = agent_plist(&remote_home, host_name)?;
    let plist_rel = format!("Library/LaunchAgents/{AGENT_LABEL}.plist");
    let remote_plist = format!("~/{plist_rel}");
    push_bytes(
        alias,
        plist.as_bytes(),
        &remote_plist,
        &sha256_hex(plist.as_bytes()),
    )?;
    // A previous agent (upgrade, interrupted add) is booted out first;
    // bootout of a not-loaded job is tolerated.
    let restart = agent_restart_command(
        "launchctl",
        AGENT_LABEL,
        &format!("$HOME/{plist_rel}"),
        RESTART_TRIES,
        RESTART_PAUSE,
    );
    ssh(alias, &restart).map_err(|e| {
        // ssh's error quotes the whole script; keep what follows it.
        let why = e.rsplit_once(" exited ").map_or(e.as_str(), |(_, why)| why);
        format!(
            "restarting the agent on {alias}: {why}\n\
             the agent may be down; `host add {alias} --name {host_name}` starts it again"
        )
    })?;
    // Wait for the socket (the agent binds it on start).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let ok = ssh(
            alias,
            "test -S ~/.local/state/llmctl-next/control.sock && echo up",
        )
        .map(|o| o.trim() == "up")
        .unwrap_or(false);
        if ok {
            return Ok(());
        }
        if std::time::Instant::now() >= deadline {
            return Err(
                "agent LaunchAgent bootstrapped but its socket never appeared; check \
                 ~/Library/Logs/llmctl-next/agent.err.log on the host"
                    .into(),
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
}

/// `host add <alias>`: probe, provision the binary, bootstrap the agent,
/// verify through the bridge, record the enrollment.
pub async fn host_add(
    alias: &str,
    name: Option<String>,
    local_binary: Option<PathBuf>,
    source_dir: Option<PathBuf>,
) -> Result<HostRecord, String> {
    let (arch, uuid) = probe_host(alias)?;
    if arch != "arm64" {
        return Err(format!("M3 targets the arm64 Macs; {alias} reports {arch}"));
    }

    let mut hosts = load_hosts();
    // Identity rules (decision 3): the UUID is the host.
    if let Some(existing) = hosts.hosts.iter().find(|h| h.io_platform_uuid == uuid) {
        if existing.ssh_alias != alias {
            eprintln!(
                "note: {} was enrolled via alias {}; keeping the host and following the new address",
                existing.name, existing.ssh_alias
            );
        }
    }
    let name = name.unwrap_or_else(|| alias.to_string());
    llmctl_core::HostId::new(&name)
        .map_err(|e| format!("host name {name:?} cannot be an agent host id: {e}"))?;
    if let Some(existing) = hosts.hosts.iter().find(|h| h.name == name) {
        if existing.io_platform_uuid != uuid {
            return Err(format!(
                "host name {name} is enrolled to a different machine (uuid {}); \
                 refusing to re-bind it to {uuid}. Pick another --name.",
                existing.io_platform_uuid
            ));
        }
    }
    hosts.hosts.retain(|h| h.io_platform_uuid != uuid);

    // Binary provisioning (M3-D.1/D.3).
    let local_version = env!("CARGO_PKG_VERSION").to_string();
    match (&local_binary, &source_dir) {
        (Some(bin), _) => {
            let bytes =
                std::fs::read(bin).map_err(|e| format!("reading {}: {e}", bin.display()))?;
            push_bytes(
                alias,
                &bytes,
                "~/.local/bin/llmctl-next.new",
                &sha256_hex(&bytes),
            )?;
            ssh(
                alias,
                "mkdir -p ~/.local/bin && mv ~/.local/bin/llmctl-next.new \
                 ~/.local/bin/llmctl-next && chmod +x ~/.local/bin/llmctl-next",
            )?;
        }
        (None, Some(src)) => build_on_host(alias, src)?,
        (None, None) => {
            // Reuse what is there when it matches the local version.
            let remote_version = ssh(alias, "~/.local/bin/llmctl-next --version 2>/dev/null")
                .ok()
                .and_then(|v| v.trim().split_whitespace().last().map(str::to_string));
            let matches = remote_version.as_ref().is_some_and(|v| v == &local_version);
            if !matches {
                return Err(format!(
                    "no --binary and no --source given, and {alias} has no matching \
                     llmctl-next (want {local_version}); pass --binary <arm64 binary> or \
                     --source <rust workspace dir>"
                ));
            }
        }
    }

    bootstrap_agent(alias, &name)?;
    let health = bridge_health(alias).await?;
    let record = HostRecord {
        name: name.clone(),
        io_platform_uuid: uuid,
        ssh_alias: alias.to_string(),
        remote_bin: "~/.local/bin/llmctl-next".into(),
        remote_data_dir: "~/.local/state/llmctl-next".into(),
        arch,
        agent_version: Some(health.daemon_version.clone()),
        protocol: Some(health.protocol),
        enrolled_at: Utc::now(),
        last_seen: Some(Utc::now()),
    };
    hosts.hosts.push(record.clone());
    save_hosts(&hosts)?;
    Ok(record)
}

/// `host upgrade <name>`: protocol check before and after the swap, agent
/// restart (bootout + bootstrap, as login does), health re-verified.
pub async fn host_upgrade(
    name: &str,
    local_binary: Option<PathBuf>,
    source_dir: Option<PathBuf>,
) -> Result<(), String> {
    let mut hosts = load_hosts();
    let (alias, host_name) = find_host(&hosts, name)
        .map(|h| (h.ssh_alias.clone(), h.name.clone()))
        .ok_or_else(|| format!("no enrolled host named {name}"))?;

    // Before: the current agent must answer (protocol check one).
    let before = bridge_health(&alias).await?;

    match (&local_binary, &source_dir) {
        (Some(bin), _) => {
            let bytes =
                std::fs::read(bin).map_err(|e| format!("reading {}: {e}", bin.display()))?;
            push_bytes(
                &alias,
                &bytes,
                "~/.local/bin/llmctl-next.new",
                &sha256_hex(&bytes),
            )?;
            ssh(
                &alias,
                "mv ~/.local/bin/llmctl-next.new ~/.local/bin/llmctl-next && chmod +x ~/.local/bin/llmctl-next",
            )?;
        }
        // build_on_host installs the binary itself.
        (None, Some(src)) => build_on_host(&alias, src)?,
        (None, None) => return Err("pass --binary or --source".into()),
    }
    bootstrap_agent(&alias, &host_name)?;
    // After: health + protocol check two.
    let after = bridge_health(&alias).await?;
    if let Some(h) = hosts.hosts.iter_mut().find(|h| h.ssh_alias == alias) {
        h.agent_version = Some(after.daemon_version.clone());
        h.protocol = Some(after.protocol);
        h.last_seen = Some(Utc::now());
    }
    save_hosts(&hosts)?;
    println!(
        "agent on {name} upgraded: {} (protocol {}) -> {} (protocol {})",
        before.daemon_version, before.protocol, after.daemon_version, after.protocol
    );
    Ok(())
}

/// `host remove <name>`: boot out the agent, remove its plist and binary.
/// The data dir (journal, inventory) is kept unless `purge`.
pub fn host_remove(name: &str, purge: bool) -> Result<(), String> {
    let mut hosts = load_hosts();
    let Some(record) = hosts.hosts.iter().find(|h| &h.name == name).cloned() else {
        return Err(format!("no enrolled host named {name}"));
    };
    let alias = &record.ssh_alias;
    let _ = ssh(
        alias,
        &format!(
            "launchctl bootout gui/$(id -u)/{AGENT_LABEL} 2>/dev/null || true"
        ),
    );
    let _ = ssh(
        alias,
        &format!("rm -f ~/Library/LaunchAgents/{AGENT_LABEL}.plist ~/.local/bin/llmctl-next"),
    );
    if purge {
        let _ = ssh(alias, "rm -rf ~/.local/state/llmctl-next");
    }
    hosts.hosts.retain(|h| &h.name != name);
    save_hosts(&hosts)?;
    let _ = std::fs::remove_file(status_cache_path(name));
    println!(
        "removed agent from {name} (data dir kept{})",
        if purge {
            " and purged"
        } else {
            "; pass --purge to delete it"
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the restart command against a fake launchctl in a temp dir: no
    /// service manager. The job stays loaded for `loaded_for` prints after
    /// a bootout, then bootstrap fails `failures` times with `error`.
    /// Returns (exit ok, stdout + stderr, the calls in order).
    fn restart_with_fake(
        loaded_for: u32,
        failures: u32,
        error: &str,
    ) -> (bool, String, Vec<String>) {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let fake = d.join("launchctl.sh");
        let script = format!(
            r#"d="{dir}"
echo "$1" >> "$d/calls"
count() {{ n=$(cat "$d/n.$1" 2>/dev/null || echo 0); echo $((n + 1)) > "$d/n.$1"; }}
case "$1" in
  print)
    count print
    [ "$n" -lt {loaded_for} ] && {{ echo "environment = SECRET_FROM_PRINT"; exit 0; }}
    exit 113 ;;
  bootstrap)
    [ -z "$RESTART_TEST_LEAK" ] || {{ echo "environment leaked" >&2; exit 9; }}
    count bootstrap
    [ "$n" -lt {failures} ] && {{ echo "{error}" >&2; exit 5; }}
    exit 0 ;;
esac
exit 0
"#,
            dir = d.display()
        );
        std::fs::write(&fake, script).unwrap();
        // Started through sh: writing and then executing a file races with
        // forks in other test threads (ETXTBSY).
        let launchctl = format!("/bin/sh {}", fake.display());
        let cmd =
            agent_restart_command(&launchctl, "com.example.agent", "/x/agent.plist", 8, "0.01");
        let out = std::process::Command::new("sh")
            .args(["-c", &cmd])
            .env("RESTART_TEST_LEAK", "1")
            .output()
            .unwrap();
        let calls = std::fs::read_to_string(d.join("calls")).unwrap_or_default();
        (
            out.status.success(),
            format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ),
            calls.lines().map(str::to_string).collect(),
        )
    }

    const EIO: &str = "Bootstrap failed: 5: Input/output error";

    fn bootstraps(calls: &[String]) -> usize {
        calls.iter().filter(|c| *c == "bootstrap").count()
    }

    #[test]
    fn restart_waits_until_launchd_released_the_old_job() {
        let (ok, out, calls) = restart_with_fake(3, 0, EIO);
        assert!(ok, "{out}");
        assert_eq!(
            calls,
            ["bootout", "print", "print", "print", "print", "bootstrap"]
        );
        assert!(
            !out.contains("SECRET_FROM_PRINT"),
            "print shows the job's environment: {out}"
        );
    }

    #[test]
    fn restart_retries_launchds_transient_bootstrap_error() {
        let (ok, out, calls) = restart_with_fake(0, 2, EIO);
        assert!(ok, "{out}");
        assert_eq!(bootstraps(&calls), 3);
    }

    #[test]
    fn restart_gives_up_on_a_lasting_error_and_shows_it() {
        let (ok, out, calls) = restart_with_fake(0, 1000, EIO);
        assert!(!ok);
        assert_eq!(bootstraps(&calls), 8);
        assert!(out.contains(EIO), "{out}");
    }

    #[test]
    fn restart_does_not_retry_other_errors() {
        let other = "Bootstrap failed: 125: Domain does not support specified action";
        let (ok, out, calls) = restart_with_fake(0, 1000, other);
        assert!(!ok);
        assert_eq!(bootstraps(&calls), 1);
        assert!(out.contains(other), "{out}");
    }

    #[test]
    fn restart_never_bootstraps_over_a_job_that_stays_loaded() {
        let (ok, out, calls) = restart_with_fake(1000, 0, EIO);
        assert!(!ok);
        assert_eq!(bootstraps(&calls), 0);
        assert!(out.contains("still loaded"), "{out}");
    }

    #[test]
    fn agent_plist_renders_canonical_xml() {
        let xml = agent_plist("/Users/yannik", "mbp2").unwrap();
        assert!(
            xml.contains(&format!("<string>{AGENT_LABEL}</string>")),
            "{xml}"
        );
        assert!(xml.contains("<true/>"), "{xml}");
        assert!(xml.contains("<string>daemon</string>"), "{xml}");
        assert!(
            xml.contains("<string>--host-id</string>\n\t\t<string>mbp2</string>"),
            "the agent answers to its enrollment name: {xml}"
        );
    }

    #[test]
    fn hosts_round_trip_and_lookup_by_alias() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("LLMCTL_NEXT_DATA_DIR", dir.path());
        let mut hosts = Hosts::default();
        hosts.hosts.push(HostRecord {
            name: "mbp2".into(),
            io_platform_uuid: "UUID-A".into(),
            ssh_alias: "mbp2-lan".into(),
            remote_bin: "~/.local/bin/llmctl-next".into(),
            remote_data_dir: "~/.local/state/llmctl-next".into(),
            arch: "arm64".into(),
            agent_version: Some("0.1.0".into()),
            protocol: Some(1),
            enrolled_at: Utc::now(),
            last_seen: None,
        });
        save_hosts(&hosts).unwrap();
        let back = load_hosts();
        assert_eq!(back, hosts);
        assert!(find_host(&back, "mbp2").is_some());
        assert!(find_host(&back, "mbp2-lan").is_some());
        assert!(find_host(&back, "m5").is_none());
        std::env::remove_var("LLMCTL_NEXT_DATA_DIR");
    }
}
