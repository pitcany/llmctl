//! `agent install` / `agent uninstall` / `agent status` (M7b,
//! docs/RUST-M7-PLAN.md): put this binary on the host as a user service,
//! and take it off again, without touching anything it did not write.
//!
//! Layout (decision 3): `<prefix>/bin/llmctl-next` (default `~/.local`),
//! data dir `~/.local/state/llmctl-next`, a user systemd unit
//! (`~/.config/systemd/user/llmctl-next.service`) or a LaunchAgent
//! (`~/Library/LaunchAgents/<label>.plist`).
//!
//! Safety rules (plan rule 4 and M7b):
//! - Every service-manager call goes through [`ServiceManager`]. Tests use
//!   a fake; the real ones run only from the CLI verbs. `systemctl --user`
//!   and `launchctl gui/<uid>` reach the real user manager whatever
//!   `$HOME` is, so a test with a temp home must never use them.
//! - Install never replaces a service file or binary it did not write.
//!   A standard-layout install without a manifest (the Macs before M7) is
//!   taken over only by an explicit `--adopt`, which writes the manifest
//!   and changes nothing else.
//! - Uninstall without a manifest takes no service action. It removes
//!   exactly the files in the manifest, refuses when one was edited, and
//!   stops only the agent, never an engine.

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// The binary's name under `<prefix>/bin`.
pub const BINARY_NAME: &str = "llmctl-next";
/// Default systemd unit name (without `.service`).
pub const SYSTEMD_UNIT: &str = "llmctl-next";
/// Manifest file inside the data dir.
pub const MANIFEST: &str = "install-manifest.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Flavor {
    Systemd,
    Launchd,
}

impl Flavor {
    pub fn native() -> Self {
        if cfg!(target_os = "macos") {
            Flavor::Launchd
        } else {
            Flavor::Systemd
        }
    }
}

/// Every path an install touches. Built from the environment by
/// [`Layout::resolve`]; tests build it by hand under a temp dir.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub flavor: Flavor,
    pub home: PathBuf,
    /// `$XDG_CONFIG_HOME` or `~/.config` (systemd's user unit search path).
    pub config_home: PathBuf,
    pub prefix: PathBuf,
    pub data_dir: PathBuf,
    /// systemd: unit name without `.service`; launchd: the job label.
    pub service: String,
    pub host_id: String,
    /// GPU slots a fresh inventory declares ([`detect_gpu_slots`]).
    pub gpu_slots: Vec<u32>,
}

impl Layout {
    /// Defaults from the environment; each `Some` argument wins.
    pub fn resolve(
        prefix: Option<PathBuf>,
        data_dir: Option<PathBuf>,
        service: Option<String>,
        host_id: Option<String>,
    ) -> Result<Self, String> {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|h| h.is_absolute())
            .ok_or("HOME is not set to an absolute path")?;
        let config_home = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .unwrap_or_else(|| home.join(".config"));
        let flavor = Flavor::native();
        let host_id = match host_id {
            Some(h) => h,
            None => short_hostname()?,
        };
        llmctl_core::HostId::new(&host_id).map_err(|e| format!("host id {host_id:?}: {e}"))?;
        Ok(Self {
            flavor,
            prefix: prefix.unwrap_or_else(|| home.join(".local")),
            data_dir: data_dir.unwrap_or_else(|| home.join(".local/state/llmctl-next")),
            service: service.unwrap_or_else(|| match flavor {
                Flavor::Systemd => SYSTEMD_UNIT.to_string(),
                Flavor::Launchd => crate::enroll::AGENT_LABEL.to_string(),
            }),
            config_home,
            home,
            host_id,
            gpu_slots: detect_gpu_slots(flavor),
        })
    }

    pub fn binary(&self) -> PathBuf {
        self.prefix.join("bin").join(BINARY_NAME)
    }

    pub fn service_file(&self) -> PathBuf {
        match self.flavor {
            Flavor::Systemd => self
                .config_home
                .join("systemd/user")
                .join(format!("{}.service", self.service)),
            Flavor::Launchd => self
                .home
                .join("Library/LaunchAgents")
                .join(format!("{}.plist", self.service)),
        }
    }

    /// systemd: the directory `enable` puts its link in.
    pub fn wants_dir(&self) -> Option<PathBuf> {
        (self.flavor == Flavor::Systemd)
            .then(|| self.config_home.join("systemd/user/default.target.wants"))
    }

    /// launchd: where the agent's stdout/stderr go.
    pub fn logs_dir(&self) -> Option<PathBuf> {
        (self.flavor == Flavor::Launchd).then(|| self.home.join("Library/Logs/llmctl-next"))
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.data_dir.join(MANIFEST)
    }

    pub fn socket(&self) -> PathBuf {
        self.data_dir.join("control.sock")
    }

    /// The service file's content for this layout.
    pub fn render_service(&self) -> Result<String, String> {
        let bin = self.binary();
        match self.flavor {
            Flavor::Systemd => Ok(systemd_unit(
                &bin,
                &self.prefix,
                &self.data_dir,
                &self.host_id,
            )),
            Flavor::Launchd => {
                let logs = self.logs_dir().expect("launchd has a logs dir");
                launchd_plist(&bin, &self.data_dir, &logs, &self.host_id, &self.service)
            }
        }
    }
}

/// GPU slots for a fresh inventory: on Linux the NVIDIA indices
/// `nvidia-smi` lists; otherwise (no NVIDIA card, macOS) one slot `0` for
/// the host's memory, as the Macs' unified memory is modelled. A workload
/// needs at least one slot, so an inventory always declares one.
pub fn detect_gpu_slots(flavor: Flavor) -> Vec<u32> {
    if flavor == Flavor::Systemd {
        if let Ok(out) = run(
            "nvidia-smi",
            &["--query-gpu=index", "--format=csv,noheader"],
        ) {
            let found: Vec<u32> = out.lines().filter_map(|l| l.trim().parse().ok()).collect();
            if !found.is_empty() {
                return found;
            }
        }
    }
    vec![0]
}

/// The machine's short host name as a host id: lowercase, up to the first
/// dot, other characters folded to `-` (decision 8).
pub fn short_hostname() -> Result<String, String> {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is valid for its length; gethostname NUL-terminates
    // within it on success (and we cap at the first NUL either way).
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    if rc != 0 {
        return Err("gethostname failed; pass --host-id".into());
    }
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    let raw = String::from_utf8_lossy(&buf[..end]).to_string();
    host_id_from(&raw)
        .ok_or_else(|| format!("host name {raw:?} gives no usable host id; pass --host-id"))
}

/// `Yanniks-MacBook-Pro.local` → `yanniks-macbook-pro`.
pub fn host_id_from(raw: &str) -> Option<String> {
    let first = raw.split('.').next().unwrap_or("");
    let mut out: String = first
        .chars()
        .map(|c| {
            let c = c.to_ascii_lowercase();
            if c.is_ascii_lowercase() || c.is_ascii_digit() {
                c
            } else {
                '-'
            }
        })
        .collect();
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    let out: String = out.trim_matches('-').chars().take(64).collect();
    let out = out.trim_end_matches('-').to_string();
    (!out.is_empty()).then_some(out)
}

/// The value after `--host-id` in a unit's ExecStart or a plist's
/// ProgramArguments.
pub fn host_id_in(service_text: &str) -> Option<String> {
    let flat = service_text
        .replace("<string>", " ")
        .replace("</string>", " ");
    let mut words = flat.split_whitespace();
    while let Some(w) = words.next() {
        if w == "--host-id" {
            return words.next().map(|v| v.trim_matches('"').to_string());
        }
    }
    None
}

/// Quote one ExecStart argument the way systemd parses it.
fn systemd_arg(a: &str) -> String {
    if !a.is_empty()
        && !a
            .chars()
            .any(|c| c.is_whitespace() || c == '"' || c == '\\' || c == '\'')
    {
        return a.to_string();
    }
    format!("\"{}\"", a.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The generated user unit. `PATH` starts with `<prefix>/bin` so engine
/// installers that call `uv` bare find it there.
pub fn systemd_unit(bin: &Path, prefix: &Path, data_dir: &Path, host_id: &str) -> String {
    let exec = [
        bin.display().to_string(),
        "--data-dir".into(),
        data_dir.display().to_string(),
        "daemon".into(),
        "--host-id".into(),
        host_id.to_string(),
    ]
    .iter()
    .map(|a| systemd_arg(a))
    .collect::<Vec<_>>()
    .join(" ");
    let path = format!(
        "{}:/usr/local/bin:/usr/bin:/bin",
        prefix.join("bin").display()
    );
    format!(
        "# Generated by `llmctl-next agent install`. Recorded with its sha256 in\n\
         # {manifest}; `agent uninstall` refuses to remove it once edited.\n\
         # Stopping this unit stops the agent only, never an engine.\n\
         \n\
         [Unit]\n\
         Description=llmctl-next agent (engines under admitted, journaled operations)\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exec}\n\
         Environment={env}\n\
         Restart=on-failure\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n",
        manifest = data_dir.join(MANIFEST).display(),
        env = systemd_arg(&format!("PATH={path}")),
    )
}

/// The agent's LaunchAgent: KeepAlive + RunAtLoad, its own data dir, logs
/// under `logs_dir` (M3 design decision 2).
pub fn launchd_plist(
    bin: &Path,
    data_dir: &Path,
    logs_dir: &Path,
    host_id: &str,
    label: &str,
) -> Result<String, String> {
    let dict = serde_json::json!({
        "Label": label,
        "ProgramArguments": [
            bin.display().to_string(),
            "--data-dir",
            data_dir.display().to_string(),
            "daemon",
            "--host-id",
            host_id
        ],
        "RunAtLoad": true,
        "KeepAlive": true,
        "StandardOutPath": logs_dir.join("agent.out.log").display().to_string(),
        "StandardErrorPath": logs_dir.join("agent.err.log").display().to_string(),
    });
    llmctl_daemon::plistdef::render_xml_plist(&dict)
}

// ---- manifest ---------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: PathBuf,
    pub sha256: String,
}

/// What an install wrote. Uninstall removes exactly this.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub flavor: Flavor,
    pub service: String,
    pub service_file: PathBuf,
    pub binary: PathBuf,
    pub data_dir: PathBuf,
    pub host_id: String,
    pub files: Vec<FileEntry>,
    /// Directories the install created, in creation order (removed in
    /// reverse when empty).
    pub dirs: Vec<PathBuf>,
    /// True when `--adopt` wrote this for an install made before M7.
    pub adopted: bool,
    pub installed_at: DateTime<Utc>,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Option<Self>, String> {
        match std::fs::read(path) {
            Ok(b) => serde_json::from_slice(&b)
                .map(Some)
                .map_err(|e| format!("reading {}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("reading {}: {e}", path.display())),
        }
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        write_atomic(path, text.as_bytes(), 0o644)
    }

    /// The layout the manifest describes (for the service-manager calls).
    fn layout(&self, like: &Layout) -> Layout {
        Layout {
            flavor: self.flavor,
            service: self.service.clone(),
            data_dir: self.data_dir.clone(),
            host_id: self.host_id.clone(),
            ..like.clone()
        }
    }

    /// Recorded files whose content changed since the install, and those
    /// that are gone.
    fn drift(&self) -> (Vec<PathBuf>, Vec<PathBuf>) {
        let (mut edited, mut missing) = (Vec::new(), Vec::new());
        for f in &self.files {
            match std::fs::read(&f.path) {
                Ok(b) if sha256_hex(&b) == f.sha256 => {}
                Ok(_) => edited.push(f.path.clone()),
                Err(_) => missing.push(f.path.clone()),
            }
        }
        (edited, missing)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    crate::enroll::sha256_hex(bytes)
}

fn file_sha(path: &Path) -> Result<String, String> {
    std::fs::read(path)
        .map(|b| sha256_hex(&b))
        .map_err(|e| format!("reading {}: {e}", path.display()))
}

/// Write via a temp file in the same directory, then rename.
fn write_atomic(path: &Path, bytes: &[u8], mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt as _;
    let dir = path
        .parent()
        .ok_or_else(|| format!("{} has no parent", path.display()))?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = dir.join(format!(".{name}.new"));
    std::fs::write(&tmp, bytes).map_err(|e| format!("writing {}: {e}", tmp.display()))?;
    std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(mode))
        .map_err(|e| format!("chmod {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("renaming into {}: {e}", path.display()))
}

/// `create_dir_all`, recording each directory it actually created.
fn create_dir_tracked(path: &Path, created: &mut Vec<PathBuf>) -> Result<(), String> {
    let mut missing = Vec::new();
    let mut cur = Some(path);
    while let Some(p) = cur {
        if p.exists() {
            break;
        }
        missing.push(p.to_path_buf());
        cur = p.parent();
    }
    for p in missing.into_iter().rev() {
        std::fs::create_dir(&p).map_err(|e| format!("creating {}: {e}", p.display()))?;
        created.push(p);
    }
    Ok(())
}

/// Seed `<data-dir>/inventory.json` with an empty inventory (no
/// workloads; the layout's GPU slots) when the data dir has none, so a fresh agent does
/// not bootstrap the dev sample (three fake workloads, two GPUs). An
/// existing inventory is never touched. Workloads are enrolled with
/// `llmctl-next adopt`. Returns whether it wrote the file.
pub fn seed_empty_inventory(l: &Layout) -> Result<bool, String> {
    let path = l.data_dir.join("inventory.json");
    if path.exists() {
        return Ok(false);
    }
    let host = llmctl_core::HostId::new(&l.host_id)
        .map_err(|e| format!("host id {:?}: {e}", l.host_id))?;
    let inventory = llmctl_core::HostInventory {
        host: host.clone(),
        platform: match l.flavor {
            Flavor::Systemd => llmctl_core::Platform::Linux,
            Flavor::Launchd => llmctl_core::Platform::Macos,
        },
        gpus: l
            .gpu_slots
            .iter()
            .map(|i| llmctl_core::GpuId::new(host.clone(), *i))
            .collect(),
        workloads: Vec::new(),
        stores: Vec::new(),
        gpu_tolerated: Vec::new(),
        revision: 1,
    };
    inventory
        .validate()
        .map_err(|e| format!("empty inventory invalid: {e}"))?;
    let file = llmctl_daemon::app::InventoryFile { inventory };
    let text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    write_atomic(&path, text.as_bytes(), 0o644)?;
    Ok(true)
}

// ---- service manager --------------------------------------------------------

/// The service-manager calls an install makes. Real implementations shell
/// out to `systemctl --user` / `launchctl`; tests use a fake (rule 4).
pub trait ServiceManager {
    /// The service file is written: load it, enable it at login, and
    /// (re)start it.
    fn start(&self, l: &Layout) -> Result<(), String>;
    /// Stop and unload; files stay. Not loaded is not an error.
    fn stop(&self, l: &Layout) -> Result<(), String>;
    /// The service file is gone (systemd: `daemon-reload`).
    fn forget(&self, l: &Layout) -> Result<(), String>;
    /// One-word-ish state for `agent status`.
    fn state(&self, l: &Layout) -> String;
    /// Hints after an install (systemd: lingering off).
    fn notes(&self, l: &Layout) -> Vec<String>;
    /// Wait until the agent answers on its socket.
    fn wait_ready(&self, l: &Layout, timeout: Duration) -> Result<(), String> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if std::os::unix::net::UnixStream::connect(l.socket()).is_ok() {
                return Ok(());
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "the agent did not answer on {} within {} s (state: {})",
                    l.socket().display(),
                    timeout.as_secs(),
                    self.state(l)
                ));
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

fn run(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = std::process::Command::new(cmd)
        .args(args)
        .output()
        .map_err(|e| format!("spawning {cmd}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{cmd} {} exited {}: {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `systemctl --user` on the real user manager.
pub struct Systemd;

impl ServiceManager for Systemd {
    fn start(&self, l: &Layout) -> Result<(), String> {
        let unit = format!("{}.service", l.service);
        run("systemctl", &["--user", "daemon-reload"])?;
        run("systemctl", &["--user", "enable", &unit])?;
        run("systemctl", &["--user", "restart", &unit]).map(|_| ())
    }
    fn stop(&self, l: &Layout) -> Result<(), String> {
        let unit = format!("{}.service", l.service);
        match run("systemctl", &["--user", "disable", "--now", &unit]) {
            Ok(_) => Ok(()),
            // Already gone from the manager: nothing to stop.
            Err(e) if e.contains("not loaded") || e.contains("does not exist") => Ok(()),
            Err(e) => Err(e),
        }
    }
    fn forget(&self, _l: &Layout) -> Result<(), String> {
        run("systemctl", &["--user", "daemon-reload"]).map(|_| ())
    }
    fn state(&self, l: &Layout) -> String {
        let unit = format!("{}.service", l.service);
        // is-active exits non-zero for inactive units; read stdout either way.
        std::process::Command::new("systemctl")
            .args(["--user", "is-active", &unit])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|e| format!("unknown ({e})"))
    }
    fn notes(&self, _l: &Layout) -> Vec<String> {
        let user = run("id", &["-un"]).unwrap_or_default();
        let linger = run(
            "loginctl",
            &["show-user", user.trim(), "-p", "Linger", "--value"],
        );
        match linger.as_deref().map(str::trim) {
            Ok("yes") => Vec::new(),
            _ => vec![format!(
                "lingering is off: the agent stops at logout. To keep it running: loginctl enable-linger {}",
                user.trim()
            )],
        }
    }
}

/// `launchctl` in the user's GUI domain.
pub struct Launchd;

impl Launchd {
    fn domain() -> String {
        // SAFETY: getuid has no preconditions.
        format!("gui/{}", unsafe { libc::getuid() })
    }
    /// The job's row in `launchctl list` (`PID Status Label`). The bare
    /// list shows no environment, unlike `launchctl print`.
    fn row(label: &str) -> Option<(String, String)> {
        let out = run("launchctl", &["list"]).ok()?;
        out.lines().find_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f.len() == 3 && f[2] == label).then(|| (f[0].to_string(), f[1].to_string()))
        })
    }
}

impl ServiceManager for Launchd {
    fn start(&self, l: &Layout) -> Result<(), String> {
        let _ = self.stop(l);
        let plist = l.service_file().display().to_string();
        // env -i: a bootstrap inherits the caller's exported variables.
        run(
            "env",
            &["-i", "launchctl", "bootstrap", &Self::domain(), &plist],
        )
        .map(|_| ())
    }
    fn stop(&self, l: &Layout) -> Result<(), String> {
        if Self::row(&l.service).is_none() {
            return Ok(());
        }
        run(
            "launchctl",
            &["bootout", &format!("{}/{}", Self::domain(), l.service)],
        )
        .map(|_| ())
    }
    fn forget(&self, _l: &Layout) -> Result<(), String> {
        Ok(())
    }
    fn state(&self, l: &Layout) -> String {
        match Self::row(&l.service) {
            Some((pid, _)) if pid != "-" => format!("running (pid {pid})"),
            Some((_, status)) => format!("loaded, not running (last exit {status})"),
            None => "not loaded".into(),
        }
    }
    fn notes(&self, _l: &Layout) -> Vec<String> {
        Vec::new()
    }
}

pub fn native_manager() -> Box<dyn ServiceManager> {
    match Flavor::native() {
        Flavor::Systemd => Box::new(Systemd),
        Flavor::Launchd => Box::new(Launchd),
    }
}

// ---- flows ------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallMode {
    Fresh,
    /// Over its own earlier install (binary or service file replaced).
    Upgrade,
    /// `--adopt`: manifest written for an existing install, nothing else.
    Adopted,
}

#[derive(Debug, Clone)]
pub struct InstallReport {
    pub mode: InstallMode,
    pub manifest: Manifest,
    pub notes: Vec<String>,
    /// True when this install wrote an empty inventory.json.
    pub seeded_inventory: bool,
}

/// Why something already on disk is not ours to replace.
fn foreign_description(path: &Path) -> Option<String> {
    let meta = std::fs::symlink_metadata(path).ok()?;
    if meta.file_type().is_symlink() {
        let target = std::fs::read_link(path)
            .map(|t| t.display().to_string())
            .unwrap_or_default();
        Some(format!("{} (a symlink to {target})", path.display()))
    } else {
        Some(path.display().to_string())
    }
}

pub fn install(
    l: &Layout,
    src_binary: &Path,
    sm: &dyn ServiceManager,
    adopt: bool,
    host_explicit: bool,
    ready_timeout: Duration,
) -> Result<InstallReport, String> {
    let existing = Manifest::load(&l.manifest_path())?;
    // An upgrade keeps the id the agent answers to unless --host-id is
    // given: a new id would rename the agent's bound host on its start.
    let owned_layout;
    let l = match &existing {
        Some(m) if !host_explicit => {
            owned_layout = Layout {
                host_id: m.host_id.clone(),
                ..l.clone()
            };
            &owned_layout
        }
        _ => l,
    };
    let service_file = l.service_file();
    let binary = l.binary();

    if let Some(m) = &existing {
        if adopt {
            return Err(format!(
                "{} already has a manifest; run `agent install` without --adopt to upgrade",
                l.data_dir.display()
            ));
        }
        if m.service_file != service_file || m.binary != binary || m.flavor != l.flavor {
            return Err(format!(
                "the manifest in {} records a different install (service file {}, binary {}); \
                 uninstall it first or pass the same --prefix/--data-dir",
                l.data_dir.display(),
                m.service_file.display(),
                m.binary.display()
            ));
        }
        let (edited, _) = m.drift();
        if !edited.is_empty() {
            return Err(format!(
                "refusing to overwrite files edited since the install: {}",
                edited
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    } else {
        // Running `<prefix>/bin/llmctl-next agent install` from where the
        // archive was unpacked installs the binary over itself: not foreign.
        let in_place = matches!(
            (std::fs::canonicalize(src_binary), std::fs::canonicalize(&binary)),
            (Ok(a), Ok(b)) if a == b
        );
        let found: Vec<String> = [&service_file, &binary]
            .iter()
            .filter(|p| !(in_place && **p == &binary))
            .filter_map(|p| foreign_description(p))
            .collect();
        if adopt {
            let found_all: Vec<String> = [&service_file, &binary]
                .iter()
                .filter_map(|p| foreign_description(p))
                .collect();
            return adopt_existing(l, &found_all, host_explicit);
        }
        if !found.is_empty() {
            return Err(format!(
                "refusing to replace an agent this command did not install: found {}. \
                 Nothing was changed. For a standard-layout install made before M7, \
                 `agent install --adopt` records it without changing it.",
                found.join(", ")
            ));
        }
    }

    let mode = if existing.is_some() {
        InstallMode::Upgrade
    } else {
        InstallMode::Fresh
    };
    let mut created = existing
        .as_ref()
        .map(|m| m.dirs.clone())
        .unwrap_or_default();
    create_dir_tracked(&l.data_dir, &mut created)?;
    create_dir_tracked(binary.parent().expect("bin dir"), &mut created)?;
    create_dir_tracked(service_file.parent().expect("service dir"), &mut created)?;
    if let Some(logs) = l.logs_dir() {
        create_dir_tracked(&logs, &mut created)?;
    }
    let wants_before = l.wants_dir().map(|w| w.exists());
    let seeded_inventory = seed_empty_inventory(l)?;

    let bin_bytes =
        std::fs::read(src_binary).map_err(|e| format!("reading {}: {e}", src_binary.display()))?;
    write_atomic(&binary, &bin_bytes, 0o755)?;
    let service_text = l.render_service()?;
    write_atomic(&service_file, service_text.as_bytes(), 0o644)?;

    let mut manifest = Manifest {
        version: 1,
        flavor: l.flavor,
        service: l.service.clone(),
        service_file: service_file.clone(),
        binary: binary.clone(),
        data_dir: l.data_dir.clone(),
        host_id: l.host_id.clone(),
        files: vec![
            FileEntry {
                path: binary.clone(),
                sha256: sha256_hex(&bin_bytes),
            },
            FileEntry {
                path: service_file.clone(),
                sha256: sha256_hex(service_text.as_bytes()),
            },
        ],
        dirs: dedup(created),
        adopted: existing.as_ref().is_some_and(|m| m.adopted),
        installed_at: Utc::now(),
    };
    // Written before the service starts, so a failed start can still be
    // uninstalled.
    manifest.save(&l.manifest_path())?;

    sm.start(l)?;
    if let (Some(wants), Some(false)) = (l.wants_dir(), wants_before) {
        if wants.exists() && !manifest.dirs.contains(&wants) {
            manifest.dirs.push(wants);
            manifest.save(&l.manifest_path())?;
        }
    }
    sm.wait_ready(l, ready_timeout)?;
    Ok(InstallReport {
        mode,
        manifest,
        notes: sm.notes(l),
        seeded_inventory,
    })
}

fn dedup(dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for d in dirs {
        if !out.contains(&d) {
            out.push(d);
        }
    }
    out
}

/// `--adopt`: record a standard-layout install that has no manifest. The
/// service file must be a regular file that runs this layout's binary on
/// this layout's data dir. Nothing else changes; no service call.
fn adopt_existing(
    l: &Layout,
    found: &[String],
    host_explicit: bool,
) -> Result<InstallReport, String> {
    let service_file = l.service_file();
    let binary = l.binary();
    if found.is_empty() {
        return Err("--adopt: nothing to adopt (no service file, no binary in this layout)".into());
    }
    let meta = std::fs::symlink_metadata(&service_file)
        .map_err(|_| format!("--adopt: no service file at {}", service_file.display()))?;
    if meta.file_type().is_symlink() {
        return Err(format!(
            "--adopt: {} is a symlink; only a plain service file in the standard layout is adopted",
            service_file.display()
        ));
    }
    if !binary.is_file() {
        return Err(format!("--adopt: no binary at {}", binary.display()));
    }
    let text = std::fs::read_to_string(&service_file)
        .map_err(|e| format!("reading {}: {e}", service_file.display()))?;
    let (bin_s, data_s) = (
        binary.display().to_string(),
        l.data_dir.display().to_string(),
    );
    if !text.contains(&bin_s) || !text.contains(&data_s) {
        return Err(format!(
            "--adopt: {} does not run {bin_s} on {data_s}; it is not this layout's install",
            service_file.display()
        ));
    }
    // Keep the id the agent already answers to (mbp2's plist says
    // `--host-id mbp2`); recording the hostname instead would rename it at
    // the next upgrade.
    let host_id = match (host_id_in(&text), host_explicit) {
        (Some(found), true) if found != l.host_id => {
            return Err(format!(
                "--adopt: {} runs the agent as host {found:?}, not {:?}; drop --host-id or pass {found}",
                service_file.display(),
                l.host_id
            ));
        }
        (Some(found), _) => found,
        (None, true) => l.host_id.clone(),
        (None, false) => {
            return Err(format!(
                "--adopt: {} passes no --host-id; pass --host-id with the id this agent answers to (`llmctl-next status` shows it)",
                service_file.display()
            ));
        }
    };
    let manifest = Manifest {
        version: 1,
        flavor: l.flavor,
        service: l.service.clone(),
        service_file: service_file.clone(),
        binary: binary.clone(),
        data_dir: l.data_dir.clone(),
        host_id,
        files: vec![
            FileEntry {
                path: binary.clone(),
                sha256: file_sha(&binary)?,
            },
            FileEntry {
                path: service_file.clone(),
                sha256: file_sha(&service_file)?,
            },
        ],
        dirs: Vec::new(),
        adopted: true,
        installed_at: Utc::now(),
    };
    std::fs::create_dir_all(&l.data_dir)
        .map_err(|e| format!("creating {}: {e}", l.data_dir.display()))?;
    manifest.save(&l.manifest_path())?;
    Ok(InstallReport {
        mode: InstallMode::Adopted,
        manifest,
        notes: Vec::new(),
        seeded_inventory: false,
    })
}

#[derive(Debug, Clone)]
pub struct UninstallReport {
    pub removed_files: Vec<PathBuf>,
    pub removed_dirs: Vec<PathBuf>,
    /// Recorded files that were already gone.
    pub missing: Vec<PathBuf>,
    /// Created directories left because something else is in them.
    pub kept_dirs: Vec<PathBuf>,
    pub purged: Vec<PathBuf>,
}

pub fn uninstall(
    l: &Layout,
    sm: &dyn ServiceManager,
    purge: bool,
) -> Result<UninstallReport, String> {
    let Some(m) = Manifest::load(&l.manifest_path())? else {
        let found: Vec<String> = [l.service_file(), l.binary()]
            .iter()
            .filter_map(|p| foreign_description(p))
            .collect();
        return Err(format!(
            "no install manifest at {}: nothing removed, no service stopped{}",
            l.manifest_path().display(),
            if found.is_empty() {
                String::new()
            } else {
                format!(
                    " (found {}, not written by `agent install`)",
                    found.join(", ")
                )
            }
        ));
    };
    let (edited, missing) = m.drift();
    if !edited.is_empty() {
        return Err(format!(
            "refusing to remove files edited since the install: {}. Nothing was changed.",
            edited
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let ml = m.layout(l);
    sm.stop(&ml)?;
    let mut report = UninstallReport {
        removed_files: Vec::new(),
        removed_dirs: Vec::new(),
        missing,
        kept_dirs: Vec::new(),
        purged: Vec::new(),
    };
    for f in &m.files {
        if f.path.exists() {
            std::fs::remove_file(&f.path)
                .map_err(|e| format!("removing {}: {e}", f.path.display()))?;
            report.removed_files.push(f.path.clone());
        }
    }
    sm.forget(&ml)?;
    let _ = std::fs::remove_file(l.manifest_path());
    if purge {
        for d in std::iter::once(m.data_dir.clone()).chain(ml.logs_dir()) {
            if d.exists() {
                std::fs::remove_dir_all(&d)
                    .map_err(|e| format!("removing {}: {e}", d.display()))?;
                report.purged.push(d);
            }
        }
    }
    for d in m.dirs.iter().rev() {
        if !d.exists() {
            continue;
        }
        match std::fs::remove_dir(d) {
            Ok(()) => report.removed_dirs.push(d.clone()),
            Err(_) => report.kept_dirs.push(d.clone()),
        }
    }
    Ok(report)
}

#[derive(Debug, Clone, Serialize)]
pub struct StatusReport {
    pub installed: bool,
    pub adopted: bool,
    pub binary: PathBuf,
    pub binary_sha256: Option<String>,
    pub service_file: PathBuf,
    /// Recorded files edited since the install.
    pub edited: Vec<PathBuf>,
    pub missing: Vec<PathBuf>,
    pub service_state: String,
    pub socket_answers: bool,
}

pub fn status(l: &Layout, sm: &dyn ServiceManager) -> Result<StatusReport, String> {
    let m = Manifest::load(&l.manifest_path())?;
    let ml = m.as_ref().map(|m| m.layout(l)).unwrap_or_else(|| l.clone());
    let (edited, missing) = m.as_ref().map(Manifest::drift).unwrap_or_default();
    let binary = m
        .as_ref()
        .map(|m| m.binary.clone())
        .unwrap_or_else(|| l.binary());
    Ok(StatusReport {
        installed: m.is_some(),
        adopted: m.as_ref().is_some_and(|m| m.adopted),
        binary_sha256: file_sha(&binary).ok(),
        binary,
        service_file: m
            .as_ref()
            .map(|m| m.service_file.clone())
            .unwrap_or_else(|| l.service_file()),
        edited,
        missing,
        service_state: sm.state(&ml),
        socket_answers: std::os::unix::net::UnixStream::connect(ml.socket()).is_ok(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Records calls; never touches a real service manager (rule 4).
    #[derive(Default)]
    struct Fake {
        calls: RefCell<Vec<String>>,
        /// systemd's `enable` creates the wants dir and link.
        enable_creates_wants: bool,
        fail_start: bool,
    }

    impl ServiceManager for Fake {
        fn start(&self, l: &Layout) -> Result<(), String> {
            self.calls.borrow_mut().push(format!("start {}", l.service));
            if self.enable_creates_wants {
                if let Some(w) = l.wants_dir() {
                    std::fs::create_dir_all(&w).unwrap();
                }
            }
            if self.fail_start {
                Err("start failed (fake)".into())
            } else {
                Ok(())
            }
        }
        fn stop(&self, l: &Layout) -> Result<(), String> {
            self.calls.borrow_mut().push(format!("stop {}", l.service));
            Ok(())
        }
        fn forget(&self, l: &Layout) -> Result<(), String> {
            self.calls
                .borrow_mut()
                .push(format!("forget {}", l.service));
            Ok(())
        }
        fn state(&self, _l: &Layout) -> String {
            "fake".into()
        }
        fn notes(&self, _l: &Layout) -> Vec<String> {
            Vec::new()
        }
        fn wait_ready(&self, _l: &Layout, _t: Duration) -> Result<(), String> {
            Ok(())
        }
    }

    fn layout(root: &Path, flavor: Flavor) -> Layout {
        let home = root.join("home");
        std::fs::create_dir_all(&home).unwrap();
        Layout {
            flavor,
            config_home: home.join(".config"),
            prefix: home.join(".local"),
            data_dir: home.join(".local/state/llmctl-next"),
            service: "llmctl-next-test".into(),
            host_id: "box-1".into(),
            gpu_slots: vec![0],
            home,
        }
    }

    fn src_binary(root: &Path, bytes: &[u8]) -> PathBuf {
        let p = root.join("src-bin");
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// Every path under `root`, files and directories.
    fn tree(root: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                let p = e.path();
                out.push(p.clone());
                if p.is_dir() && !p.is_symlink() {
                    stack.push(p);
                }
            }
        }
        out.sort();
        out
    }

    const T: Duration = Duration::from_secs(1);

    #[test]
    fn host_ids_come_from_the_short_hostname() {
        assert_eq!(
            host_id_from("Yanniks-MacBook-Pro.local").as_deref(),
            Some("yanniks-macbook-pro")
        );
        assert_eq!(
            host_id_from("yannik-desktop").as_deref(),
            Some("yannik-desktop")
        );
        assert_eq!(host_id_from("A__B c.lan").as_deref(), Some("a-b-c"));
        assert_eq!(host_id_from("...").as_deref(), None);
        assert!(llmctl_core::HostId::new(host_id_from(&"x".repeat(80)).unwrap()).is_ok());
    }

    #[test]
    fn the_unit_runs_this_layout_with_its_host_id_and_path() {
        let unit = systemd_unit(
            Path::new("/h/.local/bin/llmctl-next"),
            Path::new("/h/.local"),
            Path::new("/h/my data"),
            "box-1",
        );
        assert!(
            unit.contains("ExecStart=/h/.local/bin/llmctl-next --data-dir \"/h/my data\" daemon --host-id box-1\n"),
            "{unit}"
        );
        assert!(
            unit.contains("Environment=PATH=/h/.local/bin:/usr/local/bin:/usr/bin:/bin\n"),
            "{unit}"
        );
        assert!(unit.contains("WantedBy=default.target"));
    }

    #[test]
    fn the_plist_matches_what_host_add_writes() {
        let ours = launchd_plist(
            Path::new("/Users/u/.local/bin/llmctl-next"),
            Path::new("/Users/u/.local/state/llmctl-next"),
            Path::new("/Users/u/Library/Logs/llmctl-next"),
            "mbp2",
            crate::enroll::AGENT_LABEL,
        )
        .unwrap();
        assert_eq!(
            ours,
            crate::enroll::agent_plist("/Users/u", "mbp2").unwrap()
        );
    }

    #[test]
    fn install_then_purge_leaves_the_tree_as_before() {
        let root = tempfile::tempdir().unwrap();
        for flavor in [Flavor::Systemd, Flavor::Launchd] {
            let l = layout(&root.path().join(format!("{flavor:?}")), flavor);
            let before = tree(&l.home);
            let fake = Fake {
                enable_creates_wants: true,
                ..Fake::default()
            };
            let r = install(
                &l,
                &src_binary(root.path(), b"bin-v1"),
                &fake,
                false,
                false,
                T,
            )
            .unwrap();
            assert_eq!(r.mode, InstallMode::Fresh);
            assert_eq!(std::fs::read(l.binary()).unwrap(), b"bin-v1");
            assert!(std::fs::read_to_string(l.service_file())
                .unwrap()
                .contains("--host-id"));
            assert_eq!(fake.calls.borrow().as_slice(), ["start llmctl-next-test"]);
            if flavor == Flavor::Systemd {
                assert!(
                    r.manifest.dirs.contains(&l.wants_dir().unwrap()),
                    "{:?}",
                    r.manifest.dirs
                );
            }

            let u = uninstall(&l, &fake, true).unwrap();
            assert!(u.kept_dirs.is_empty(), "{u:?}");
            assert_eq!(tree(&l.home), before, "{flavor:?}: {u:?}");
            assert_eq!(
                fake.calls.borrow().as_slice(),
                [
                    "start llmctl-next-test",
                    "stop llmctl-next-test",
                    "forget llmctl-next-test"
                ]
            );
        }
    }

    #[test]
    fn uninstall_without_purge_keeps_the_data_dir_and_dirs_holding_other_files() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        let fake = Fake::default();
        install(&l, &src_binary(root.path(), b"bin"), &fake, false, false, T).unwrap();
        std::fs::write(l.data_dir.join("ops.sqlite3"), b"journal").unwrap();
        std::fs::write(l.prefix.join("bin/other-tool"), b"x").unwrap();
        let u = uninstall(&l, &fake, false).unwrap();
        assert!(!l.binary().exists() && !l.service_file().exists());
        assert!(l.data_dir.join("ops.sqlite3").exists());
        assert!(!l.manifest_path().exists());
        assert!(u.kept_dirs.contains(&l.prefix.join("bin")), "{u:?}");
        assert!(u.kept_dirs.contains(&l.data_dir), "{u:?}");
    }

    #[test]
    fn install_twice_is_an_upgrade_and_replaces_the_binary() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        let fake = Fake::default();
        install(&l, &src_binary(root.path(), b"v1"), &fake, false, false, T).unwrap();
        let r = install(&l, &src_binary(root.path(), b"v2"), &fake, false, false, T).unwrap();
        assert_eq!(r.mode, InstallMode::Upgrade);
        assert_eq!(std::fs::read(l.binary()).unwrap(), b"v2");
        let m = Manifest::load(&l.manifest_path()).unwrap().unwrap();
        assert_eq!(m.files[0].sha256, sha256_hex(b"v2"));
        assert_eq!(
            fake.calls.borrow().len(),
            2,
            "two starts (the second restarts)"
        );
    }

    /// The workstation case: a hand-written unit (a symlink into ~/AI) for
    /// the same name. Install refuses and changes nothing, no service call.
    #[test]
    fn install_refuses_a_service_file_it_did_not_write() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        let real = root.path().join("AI/services/user/llmctl-next.service");
        std::fs::create_dir_all(real.parent().unwrap()).unwrap();
        std::fs::write(&real, "[Service]\nExecStart=/elsewhere/llmctl-next.bin\n").unwrap();
        std::fs::create_dir_all(l.service_file().parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(&real, l.service_file()).unwrap();
        let before = tree(root.path());
        let fake = Fake::default();
        let err =
            install(&l, &src_binary(root.path(), b"bin"), &fake, false, false, T).unwrap_err();
        assert!(
            err.contains("did not install") && err.contains("symlink"),
            "{err}"
        );
        let err = install(&l, &src_binary(root.path(), b"bin"), &fake, true, false, T).unwrap_err();
        assert!(err.contains("symlink"), "{err}");
        assert!(fake.calls.borrow().is_empty());
        let mut after = tree(root.path());
        after.retain(|p| !p.ends_with("src-bin"));
        let mut before = before;
        before.retain(|p| !p.ends_with("src-bin"));
        assert_eq!(after, before);
    }

    #[test]
    fn uninstall_without_a_manifest_makes_no_service_call() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        std::fs::create_dir_all(l.service_file().parent().unwrap()).unwrap();
        std::fs::write(l.service_file(), "[Service]\n").unwrap();
        let fake = Fake::default();
        let err = uninstall(&l, &fake, true).unwrap_err();
        assert!(
            err.contains("no service stopped") && err.contains("not written by"),
            "{err}"
        );
        assert!(fake.calls.borrow().is_empty());
        assert!(l.service_file().exists());
    }

    #[test]
    fn an_edited_file_blocks_uninstall_and_upgrade() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        let fake = Fake::default();
        install(&l, &src_binary(root.path(), b"bin"), &fake, false, false, T).unwrap();
        let mut unit = std::fs::read_to_string(l.service_file()).unwrap();
        unit.push_str("# owner note\n");
        std::fs::write(l.service_file(), &unit).unwrap();
        let calls = fake.calls.borrow().len();
        let err = uninstall(&l, &fake, true).unwrap_err();
        assert!(
            err.contains("edited") && err.contains("llmctl-next-test.service"),
            "{err}"
        );
        let err = install(
            &l,
            &src_binary(root.path(), b"bin2"),
            &fake,
            false,
            false,
            T,
        )
        .unwrap_err();
        assert!(err.contains("edited"), "{err}");
        assert_eq!(
            fake.calls.borrow().len(),
            calls,
            "no service call after the refusals"
        );
        assert!(l.binary().exists() && l.manifest_path().exists());
    }

    /// The Macs before M7: standard layout, no manifest. `--adopt` writes
    /// the manifest from disk and changes nothing else.
    #[test]
    fn adopt_records_a_pre_manifest_install_without_changing_it() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Launchd);
        std::fs::create_dir_all(l.binary().parent().unwrap()).unwrap();
        std::fs::write(l.binary(), b"old-bin").unwrap();
        std::fs::create_dir_all(l.service_file().parent().unwrap()).unwrap();
        std::fs::write(l.service_file(), l.render_service().unwrap()).unwrap();
        let fake = Fake::default();
        let err =
            install(&l, &src_binary(root.path(), b"new"), &fake, false, false, T).unwrap_err();
        assert!(err.contains("--adopt"), "{err}");

        let r = install(&l, &src_binary(root.path(), b"new"), &fake, true, false, T).unwrap();
        assert_eq!(r.mode, InstallMode::Adopted);
        assert!(r.manifest.adopted && r.manifest.dirs.is_empty());
        assert_eq!(
            std::fs::read(l.binary()).unwrap(),
            b"old-bin",
            "binary untouched"
        );
        assert!(fake.calls.borrow().is_empty(), "no service call");

        // Adopted: a later install upgrades it, and stays marked adopted.
        let r = install(&l, &src_binary(root.path(), b"new"), &fake, false, false, T).unwrap();
        assert_eq!(r.mode, InstallMode::Upgrade);
        assert!(r.manifest.adopted);
        assert_eq!(std::fs::read(l.binary()).unwrap(), b"new");
    }

    #[test]
    fn adopt_refuses_a_service_file_for_another_layout() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        std::fs::create_dir_all(l.binary().parent().unwrap()).unwrap();
        std::fs::write(l.binary(), b"bin").unwrap();
        std::fs::create_dir_all(l.service_file().parent().unwrap()).unwrap();
        std::fs::write(
            l.service_file(),
            "ExecStart=/other/bin --data-dir /other/data daemon\n",
        )
        .unwrap();
        let err = install(
            &l,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            true,
            true,
            T,
        )
        .unwrap_err();
        assert!(err.contains("not this layout's install"), "{err}");
        assert!(!l.manifest_path().exists());
        let empty = layout(&root.path().join("empty"), Flavor::Systemd);
        let err = install(
            &empty,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            true,
            true,
            T,
        )
        .unwrap_err();
        assert!(err.contains("nothing to adopt"), "{err}");
    }

    #[test]
    fn a_failed_start_still_leaves_a_manifest_to_uninstall_with() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        let before = tree(&l.home);
        let bad = Fake {
            fail_start: true,
            ..Fake::default()
        };
        assert!(install(&l, &src_binary(root.path(), b"bin"), &bad, false, false, T).is_err());
        assert!(l.manifest_path().exists());
        uninstall(&l, &Fake::default(), true).unwrap();
        assert_eq!(tree(&l.home), before);
    }

    #[test]
    fn status_reports_install_state_and_edits() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        let fake = Fake::default();
        let s = status(&l, &fake).unwrap();
        assert!(!s.installed && s.binary_sha256.is_none() && !s.socket_answers);
        install(&l, &src_binary(root.path(), b"bin"), &fake, false, false, T).unwrap();
        let s = status(&l, &fake).unwrap();
        assert!(s.installed && !s.adopted && s.edited.is_empty());
        assert_eq!(
            s.binary_sha256.as_deref(),
            Some(sha256_hex(b"bin").as_str())
        );
        std::fs::remove_file(l.binary()).unwrap();
        assert_eq!(status(&l, &fake).unwrap().missing, vec![l.binary()]);
    }

    #[test]
    fn installing_the_binary_over_itself_is_a_fresh_install() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        std::fs::create_dir_all(l.binary().parent().unwrap()).unwrap();
        std::fs::write(l.binary(), b"unpacked").unwrap();
        let r = install(&l, &l.binary(), &Fake::default(), false, false, T).unwrap();
        assert_eq!(r.mode, InstallMode::Fresh);
        assert_eq!(std::fs::read(l.binary()).unwrap(), b"unpacked");
        // A different file already at that path is still refused.
        let other = layout(&root.path().join("o"), Flavor::Systemd);
        std::fs::create_dir_all(other.binary().parent().unwrap()).unwrap();
        std::fs::write(other.binary(), b"someone else's").unwrap();
        let err = install(
            &other,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            false,
            false,
            T,
        )
        .unwrap_err();
        assert!(err.contains("did not install"), "{err}");
    }

    #[test]
    fn an_upgrade_keeps_the_host_id_unless_one_is_given() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        let fake = Fake::default();
        install(&l, &src_binary(root.path(), b"v1"), &fake, false, true, T).unwrap();
        let renamed = Layout {
            host_id: "from-hostname".into(),
            ..l.clone()
        };
        let r = install(
            &renamed,
            &src_binary(root.path(), b"v2"),
            &fake,
            false,
            false,
            T,
        )
        .unwrap();
        assert_eq!(r.manifest.host_id, "box-1");
        assert_eq!(
            host_id_in(&std::fs::read_to_string(l.service_file()).unwrap()).as_deref(),
            Some("box-1")
        );
        let r = install(
            &renamed,
            &src_binary(root.path(), b"v3"),
            &fake,
            false,
            true,
            T,
        )
        .unwrap();
        assert_eq!(r.manifest.host_id, "from-hostname");
    }

    #[test]
    fn adopt_keeps_the_host_id_the_service_file_passes() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Launchd);
        let mac = Layout {
            host_id: "mbp2".into(),
            ..l.clone()
        };
        std::fs::create_dir_all(l.binary().parent().unwrap()).unwrap();
        std::fs::write(l.binary(), b"old").unwrap();
        std::fs::create_dir_all(l.service_file().parent().unwrap()).unwrap();
        std::fs::write(l.service_file(), mac.render_service().unwrap()).unwrap();
        // `l` carries the hostname-derived id "box-1"; the plist says mbp2.
        let err = install(
            &l,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            true,
            true,
            T,
        )
        .unwrap_err();
        assert!(err.contains("\"mbp2\""), "{err}");
        let r = install(
            &l,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            true,
            false,
            T,
        )
        .unwrap();
        assert_eq!(r.manifest.host_id, "mbp2");
        // And the next upgrade renders mbp2, not the hostname.
        install(
            &l,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            false,
            false,
            T,
        )
        .unwrap();
        assert_eq!(
            host_id_in(&std::fs::read_to_string(l.service_file()).unwrap()).as_deref(),
            Some("mbp2")
        );
    }

    #[test]
    fn adopt_without_a_host_id_in_the_file_needs_one_given() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        std::fs::create_dir_all(l.binary().parent().unwrap()).unwrap();
        std::fs::write(l.binary(), b"old").unwrap();
        std::fs::create_dir_all(l.service_file().parent().unwrap()).unwrap();
        let text = format!(
            "ExecStart={} --data-dir {} daemon\n",
            l.binary().display(),
            l.data_dir.display()
        );
        std::fs::write(l.service_file(), text).unwrap();
        let err = install(
            &l,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            true,
            false,
            T,
        )
        .unwrap_err();
        assert!(err.contains("passes no --host-id"), "{err}");
        let r = install(
            &l,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            true,
            true,
            T,
        )
        .unwrap();
        assert_eq!(r.manifest.host_id, "box-1");
    }

    #[test]
    fn a_fresh_install_seeds_an_empty_inventory_and_never_replaces_one() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Systemd);
        let fake = Fake::default();
        let r = install(&l, &src_binary(root.path(), b"v1"), &fake, false, true, T).unwrap();
        assert!(r.seeded_inventory);
        let inv_path = l.data_dir.join("inventory.json");
        let file: llmctl_daemon::app::InventoryFile =
            serde_json::from_slice(&std::fs::read(&inv_path).unwrap()).unwrap();
        assert!(file.inventory.workloads.is_empty());
        assert_eq!(
            file.inventory.gpus.len(),
            1,
            "one slot: {:?}",
            file.inventory.gpus
        );
        assert_eq!(file.inventory.host.as_str(), "box-1");
        assert_eq!(file.inventory.platform, llmctl_core::Platform::Linux);

        // The owner's inventory survives an upgrade byte for byte.
        std::fs::write(&inv_path, b"{\"owner\": \"edited\"}").unwrap();
        let r = install(&l, &src_binary(root.path(), b"v2"), &fake, false, false, T).unwrap();
        assert!(!r.seeded_inventory);
        assert_eq!(
            std::fs::read(&inv_path).unwrap(),
            b"{\"owner\": \"edited\"}"
        );
    }

    #[test]
    fn adopt_seeds_no_inventory() {
        let root = tempfile::tempdir().unwrap();
        let l = layout(root.path(), Flavor::Launchd);
        std::fs::create_dir_all(l.binary().parent().unwrap()).unwrap();
        std::fs::write(l.binary(), b"old").unwrap();
        std::fs::create_dir_all(l.service_file().parent().unwrap()).unwrap();
        std::fs::write(l.service_file(), l.render_service().unwrap()).unwrap();
        let r = install(
            &l,
            &src_binary(root.path(), b"x"),
            &Fake::default(),
            true,
            false,
            T,
        )
        .unwrap();
        assert!(!r.seeded_inventory);
        assert!(!l.data_dir.join("inventory.json").exists());
    }
}
