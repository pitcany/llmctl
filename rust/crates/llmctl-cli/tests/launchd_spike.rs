//! M3 spike (macOS only): the daemon driving a *disposable* launchd label
//! through the `launchd` adapter, and the plist adoption/ownership flows.
//! Opt-in only — these touch the user's launchd:
//!
//!     cargo test -p llmctl-cli --test launchd_spike -- --ignored --test-threads=1
//!
//! Each test writes its own `com.pitcany.llmctl-next-spike-<id>` plist (a
//! tiny python server answering `/health` and `/v1/models`), enrolls only
//! that label, and removes it on drop. Nothing else in launchd is touched.
//!
//! `llmctl_macos` (see llmctl-daemon's Cargo.toml) lets this type-check on
//! a Linux host: RUSTFLAGS='--cfg=llmctl_macos' cargo test -p llmctl-cli
//! --test launchd_spike --no-run
#![cfg(any(target_os = "macos", llmctl_macos))]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::Duration;

use llmctl_core::{
    ControlAuthority, GpuId, HostId, HostInventory, Platform, UnitName, WorkloadDefinition,
    WorkloadName,
};

const SERVER_PY: &str = r#"
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[1])
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b"OK"
            self.send_response(200)
        elif self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "spike/model"}]}).encode()
            self.send_response(200)
        else:
            self.send_response(404); self.end_headers(); return
        self.send_header("content-type", "text/plain" if self.path == "/health" else "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
server = HTTPServer(("127.0.0.1", port), H)
print("serving on %d" % port, flush=True)
server.serve_forever()
"#;

fn launchctl(args: &[&str]) -> Output {
    Command::new("launchctl").args(args).output().expect("launchctl")
}

fn uid() -> String {
    String::from_utf8_lossy(&Command::new("id").arg("-u").output().expect("id -u").stdout)
        .trim()
        .to_string()
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

/// A disposable launchd label that exists only for the test's lifetime.
struct SpikeJob {
    label: String,
    plist: PathBuf,
    out_log: PathBuf,
}

impl SpikeJob {
    fn create(dir: &Path, port: u16) -> SpikeJob {
        let label = format!(
            "com.pitcany.llmctl-next-spike-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let plist = home().join("Library/LaunchAgents").join(format!("{label}.plist"));
        let out_log = dir.join("spike.out.log");
        let python = "/usr/bin/python3";
        let script = dir.join("server.py");
        std::fs::write(&script, SERVER_PY).unwrap();
        let body = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<plist version=\"1.0\">\n<dict>\n\
             \t<key>Label</key>\n\t<string>{label}</string>\n\
             \t<key>ProgramArguments</key>\n\t<array>\n\
             \t\t<string>{python}</string>\n\t\t<string>{}</string>\n\t\t<string>{port}</string>\n\
             \t</array>\n\
             \t<key>KeepAlive</key>\n\t<true/>\n\
             \t<key>RunAtLoad</key>\n\t<true/>\n\
             \t<key>StandardOutPath</key>\n\t<string>{}</string>\n\
             \t<key>StandardErrorPath</key>\n\t<string>{}</string>\n\
             </dict>\n</plist>\n",
            script.display(),
            out_log.display(),
            out_log.display(),
        );
        std::fs::write(&plist, body).unwrap();
        SpikeJob { label, plist, out_log }
    }

    /// The original plist bytes (release must restore these exactly).
    fn original_bytes(&self) -> Vec<u8> {
        std::fs::read(&self.plist).unwrap()
    }

    fn pid(&self) -> Option<u32> {
        let out = launchctl(&["print", &format!("gui/{}/{}", uid(), self.label)]);
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find_map(|l| l.trim().strip_prefix("pid = ")?.trim().parse().ok())
    }

    fn runs(&self) -> Option<u64> {
        let out = launchctl(&["print", &format!("gui/{}/{}", uid(), self.label)]);
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .find_map(|l| l.trim().strip_prefix("runs = ")?.trim().parse().ok())
    }
}

impl Drop for SpikeJob {
    fn drop(&mut self) {
        let _ = launchctl(&["bootout", &format!("gui/{}/{}", uid(), self.label)]);
        let _ = launchctl(&["disable", &format!("gui/{}/{}", uid(), self.label)]);
        let _ = std::fs::remove_file(&self.plist);
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Notional card no real occupant can collide with (see systemd spike).
const SPIKE_GPU: u32 = 9;

fn write_inventory(dir: &Path, label: &str, port: u16, program_identity: &str) {
    let host = HostId::new("mac-spike").unwrap();
    let inv = HostInventory {
        host: host.clone(),
        platform: Platform::Macos,
        gpus: vec![GpuId::new(host.clone(), SPIKE_GPU)],
        workloads: vec![WorkloadDefinition {
            name: WorkloadName::new("spike").unwrap(),
            engine: "launchd".into(),
            // For launchd workloads the model field carries the expected
            // program path (M3-B.4 identity).
            model: program_identity.into(),
            gpus: vec![GpuId::new(host, SPIKE_GPU)],
            port,
            unit: UnitName::launchd(label).unwrap(),
            authority: ControlAuthority::LaunchdService { label: label.into() },
            ownership: Default::default(),
            recovery: None,
            recipe: None,
            readiness_timeout_secs: Some(30),
        }],
        revision: 1,
    };
    std::fs::write(
        dir.join("inventory.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "inventory": inv })).unwrap(),
    )
    .unwrap();
}

fn write_empty_inventory(dir: &Path) {
    let host = HostId::new("mac-spike").unwrap();
    let inv = serde_json::json!({
        "inventory": {
            "host": host.to_string(),
            "platform": "macos",
            "gpus": [{ "host": host.to_string(), "index": SPIKE_GPU }],
            "workloads": [],
            "revision": 1
        }
    });
    std::fs::write(dir.join("inventory.json"), inv.to_string()).unwrap();
}

struct Daemon {
    child: Child,
    socket: PathBuf,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_daemon(dir: &Path) -> Daemon {
    let socket = dir.join("control.sock");
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir)
        .arg("daemon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn daemon");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket).is_err() {
        assert!(std::time::Instant::now() < deadline, "daemon never bound");
        std::thread::sleep(Duration::from_millis(50));
    }
    Daemon { child, socket }
}

fn cli(socket: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(socket)
        .args(args)
        .output()
        .expect("run llmctl-next");
    (
        out.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

fn cli_code(socket: &Path, args: &[&str]) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(socket)
        .args(args)
        .output()
        .expect("run llmctl-next");
    (
        out.status.code(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

fn spike_status(socket: &Path) -> serde_json::Value {
    let (ok, text) = cli(socket, &["status", "--json"]);
    assert!(ok, "{text}");
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["workloads"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["name"] == "spike")
        .cloned()
        .unwrap()
}

fn inventory_json(socket: &Path) -> serde_json::Value {
    let client = llmctl_client::ControlClient::connect(socket).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move { client.inventory().await.expect("inventory readable") })
}

#[test]
#[ignore = "touches the user's launchd: run with --ignored"]
fn run_and_stop_a_disposable_label() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let job = SpikeJob::create(dir.path(), port);
    write_inventory(dir.path(), &job.label, port, "/usr/bin/python3");
    let d = spawn_daemon(dir.path());

    let before = spike_status(&d.socket);
    assert_eq!(before["process"], "Stopped", "{before}");

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait", "--key", "spike-run"]);
    assert!(ok, "{text}");
    assert!(text.contains("succeeded"), "{text}");
    let up = spike_status(&d.socket);
    assert_eq!(up["process"], "Running", "{up}");
    assert_eq!(up["readiness"], "Ready", "{up}");
    assert_eq!(up["served_model"], "/usr/bin/python3", "{up}");
    assert!(job.pid().is_some(), "running pid observed");

    // Durable stop: bootout + disable. A plain bootstrap of the plist (as
    // RunAtLoad does at login) must NOT revive it.
    let (ok, text) = cli(&d.socket, &["stop", "spike", "--wait", "--key", "spike-stop"]);
    assert!(ok, "{text}");
    assert_eq!(job.pid(), None);
    let revived = launchctl(&[
        "bootstrap",
        &format!("gui/{}", uid()),
        &job.plist.display().to_string(),
    ]);
    assert!(
        !revived.status.success() || job.pid().is_none(),
        "a disabled label must not run: {}",
        String::from_utf8_lossy(&revived.stderr)
    );
    assert_eq!(job.pid(), None, "disabled job must stay down");
}

#[test]
#[ignore = "touches the user's launchd: run with --ignored"]
fn keepalive_respawn_survives_sigkill_and_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let job = SpikeJob::create(dir.path(), port);
    write_inventory(dir.path(), &job.label, port, "/usr/bin/python3");
    let d = spawn_daemon(dir.path());

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait", "--key", "spike-run"]);
    assert!(ok, "{text}");
    let pid = job.pid().expect("pid");
    let runs = job.runs().unwrap_or(0);

    // SIGKILL: KeepAlive respawns (launchd's own retry loop).
    let _ = Command::new("kill")
        .args(["-9", &pid.to_string()])
        .output()
        .expect("kill");
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let new_runs = loop {
        if job.pid().is_some_and(|p| p != pid) {
            break job.runs().unwrap_or(0);
        }
        assert!(
            std::time::Instant::now() < deadline,
            "KeepAlive did not respawn"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(
        new_runs > runs,
        "respawn must increment runs: {runs} -> {new_runs}"
    );
    let up = spike_status(&d.socket);
    assert_eq!(up["process"], "Running", "respawn observed as running: {up}");
}

#[test]
#[ignore = "touches the user's launchd: run with --ignored"]
fn adopt_own_release_byte_identical_and_drift_detected() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let job = SpikeJob::create(dir.path(), port);
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());

    // Start it outside llmctl (as llama-swap runs today).
    assert!(launchctl(&[
        "bootstrap",
        &format!("gui/{}", uid()),
        &job.plist.display().to_string()
    ])
    .status
    .success());
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while job.pid().is_none() {
        assert!(std::time::Instant::now() < deadline, "spike never ran");
        std::thread::sleep(Duration::from_millis(200));
    }
    let original = job.original_bytes();

    // Adopt: preview, then confirm with the digest.
    let (ok, text) = cli(
        &d.socket,
        &[
            "adopt",
            &job.label,
            "--name",
            "spike",
            "--model",
            "/usr/bin/python3",
            "--port",
            &port.to_string(),
            "--gpus",
            "9",
        ],
    );
    assert!(ok, "{text}");
    let digest = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("definition digest: "))
        .expect("digest in preview")
        .to_string();
    let (ok, text) = cli(
        &d.socket,
        &[
            "adopt",
            &job.label,
            "--name",
            "spike",
            "--model",
            "/usr/bin/python3",
            "--port",
            &port.to_string(),
            "--gpus",
            "9",
            "--confirm",
            &digest,
        ],
    );
    assert!(ok, "{text}");

    // own: deferred apply — the plist is rewritten, NOT bootstrapped; the
    // running process must survive untouched.
    let pid_before = job.pid().expect("pid before own");
    let (ok, text) = cli(&d.socket, &["own", "spike"]);
    assert!(ok, "{text}");
    assert_eq!(job.pid(), Some(pid_before), "own must not restart");
    let inv = inventory_json(&d.socket);
    let pending = inv["workloads"][0]["ownership"]["pending_apply"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    assert!(
        pending.contains_key("reason"),
        "pending_apply recorded: {pending:?}"
    );

    // An edited (hand-written) owned plist is drift before any effect.
    let owned_now = std::fs::read(&job.plist).unwrap();
    let with_env = String::from_utf8_lossy(&owned_now).replace(
        "<dict>\n\t<key>Label</key>",
        "<dict>\n\t<key>EnvironmentVariables</key>\n\t<dict>\n\t\t<key>SPIKE_EXTRA</key>\n\t\t<string>1</string>\n\t</dict>\n\t<key>Label</key>",
    );
    std::fs::write(&job.plist, with_env).unwrap();
    let (code, text) = cli_code(&d.socket, &["run", "spike", "--wait", "--key", "spike-drift"]);
    assert_eq!(code, Some(4), "drift must refuse mutations: {text}");
    assert!(text.contains("drift"), "{text}");
    // Put the owned bytes back before release.
    std::fs::write(&job.plist, owned_now).unwrap();

    // release: byte-identical restore, still no restart.
    let (ok, text) = cli(&d.socket, &["release", "spike"]);
    assert!(ok, "{text}");
    assert_eq!(
        std::fs::read(&job.plist).unwrap(),
        original,
        "release restores the original bytes exactly"
    );
    assert_eq!(job.pid(), Some(pid_before), "release must not restart");
}

#[test]
#[ignore = "touches the user's launchd: run with --ignored"]
fn secret_looking_env_is_refused_at_adoption() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let job = SpikeJob::create(dir.path(), port);
    let plist = std::fs::read_to_string(&job.plist).unwrap();
    let with_secret = plist.replace(
        "<dict>\n\t<key>Label</key>",
        "<dict>\n\t<key>EnvironmentVariables</key>\n\t<dict>\n\t\t<key>HF_TOKEN</key>\n\t\t<string>x</string>\n\t</dict>\n\t<key>Label</key>",
    );
    std::fs::write(&job.plist, with_secret).unwrap();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());

    let (ok, text) = cli(
        &d.socket,
        &[
            "adopt",
            &job.label,
            "--name",
            "spike",
            "--model",
            "/usr/bin/python3",
            "--port",
            &port.to_string(),
            "--gpus",
            "9",
        ],
    );
    assert!(!ok, "secret env must fail the preview: {text}");
    assert!(text.contains("no_inline_secrets"), "{text}");
}

#[test]
#[ignore = "touches the user's launchd: run with --ignored"]
fn logs_follow_a_restart_and_a_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let job = SpikeJob::create(dir.path(), port);
    write_inventory(dir.path(), &job.label, port, "/usr/bin/python3");
    let d = spawn_daemon(dir.path());

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait", "--key", "spike-run"]);
    assert!(ok, "{text}");
    let first_run = job.runs().unwrap_or(0);
    let (ok, text) = cli(&d.socket, &["restart", "spike", "--wait", "--key", "spike-restart"]);
    assert!(ok, "{text}");
    assert!(job.runs().unwrap_or(0) > first_run, "restart must rerun");
    let up = spike_status(&d.socket);
    assert_eq!(up["readiness"], "Ready", "{up}");

    let (ok, text) = cli(&d.socket, &["logs", "spike", "-n", "50"]);
    assert!(ok, "{text}");
    assert!(text.contains("serving on"), "{text}");

    // Truncation: the cursor's offset is beyond EOF; the reader restarts
    // from zero instead of hanging or lying.
    std::fs::write(&job.out_log, b"truncated\n").unwrap();
    let (ok, text) = cli(&d.socket, &["logs", "spike", "-n", "50"]);
    assert!(ok, "{text}");
    assert!(text.contains("truncated"), "{text}");
}
