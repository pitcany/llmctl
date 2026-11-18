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

/// Prepended to the server: it takes three seconds to exit on SIGTERM,
/// like a real engine that finishes requests and frees memory first.
const SLOW_EXIT_PY: &str = r#"
import os, signal, time
def slow_exit(*_):
    time.sleep(3)
    os._exit(0)
signal.signal(signal.SIGTERM, slow_exit)
"#;

impl SpikeJob {
    fn create(dir: &Path, port: u16) -> SpikeJob {
        Self::create_with_server(dir, port, SERVER_PY)
    }

    fn create_with_server(dir: &Path, port: u16, server: &str) -> SpikeJob {
        let label = format!(
            "com.pitcany.llmctl-next-spike-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let plist = home().join("Library/LaunchAgents").join(format!("{label}.plist"));
        let out_log = dir.join("spike.out.log");
        let python = "/usr/bin/python3";
        let script = dir.join("server.py");
        std::fs::write(&script, server).unwrap();
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

/// A pid existing is not the port listening (python binds after exec):
/// adoption's served-identity check probes HTTP, so wait for it.
fn wait_port(port: u16) {
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    loop {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "spike server never listened on {port}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
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
            // A plain HTTP test server: no engine-specific protocol.
            engine_kind: llmctl_core::EngineKind::FakeEngine,
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
            gpu_claim: Default::default(),
            hooks: Vec::new(),
            pin_files: Vec::new(),
            observed_version: None,
        }],
        stores: Vec::new(),
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
    wait_port(port);
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
    // The render sorts keys (KeepAlive first), so anchor on the dict open.
    let owned_now = std::fs::read(&job.plist).unwrap();
    assert!(
        String::from_utf8_lossy(&owned_now).starts_with("<?xml"),
        "owned plist is our render"
    );
    let with_env = String::from_utf8_lossy(&owned_now).replacen(
        "<dict>\n",
        "<dict>\n\t<key>EnvironmentVariables</key>\n\t<dict>\n\t\t<key>SPIKE_EXTRA</key>\n\t\t<string>1</string>\n\t</dict>\n",
        1,
    );
    assert_ne!(
        with_env,
        String::from_utf8_lossy(&owned_now),
        "the hand edit must change the file"
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
    let pid_before = job.pid().expect("pid after run");
    let (ok, text) = cli(&d.socket, &["restart", "spike", "--wait", "--key", "spike-restart"]);
    assert!(ok, "{text}");
    // restart = bootout + bootstrap, and bootout RESETS launchd's `runs`
    // counter, so the observable rerun is the new pid (not the counter).
    wait_port(port);
    let pid_after = job.pid().expect("pid after restart");
    assert_ne!(pid_before, pid_after, "restart must rerun the process");
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

/// A job that needs seconds to exit: `bootout` returns before it is gone.
/// The stop waits for launchd to say so, and a restart goes through (seen
/// live on m5: Ollama's restart failed on "still loaded" and left it down).
#[test]
#[ignore = "touches the user's launchd: run with --ignored"]
fn a_job_that_exits_slowly_is_stopped_and_restarted() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let slow = format!("{SLOW_EXIT_PY}{SERVER_PY}");
    let job = SpikeJob::create_with_server(dir.path(), port, &slow);
    write_inventory(dir.path(), &job.label, port, "/usr/bin/python3");
    let d = spawn_daemon(dir.path());

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait", "--key", "slow-run"]);
    assert!(ok, "{text}");
    let first = job.pid().expect("running pid");

    let (ok, text) = cli(&d.socket, &["restart", "spike", "--wait", "--key", "slow-restart"]);
    assert!(ok, "a slow exit is not a failed stop: {text}");
    let up = spike_status(&d.socket);
    assert_eq!(up["process"], "Running", "{up}");
    assert_eq!(up["readiness"], "Ready", "{up}");
    let second = job.pid().expect("running pid after the restart");
    assert_ne!(first, second, "the process was replaced");

    let (ok, text) = cli(&d.socket, &["stop", "spike", "--wait", "--key", "slow-stop"]);
    assert!(ok, "{text}");
    assert_eq!(job.pid(), None);
}

/// A vendor app has no launchd job of its own to ask: it is started and
/// stopped with its own commands, and the endpoint says whether that
/// took. Here the "app" is the test server behind a start and a stop
/// script; launchd is never asked about it.
#[test]
#[ignore = "starts a local test server: run with --ignored"]
fn a_vendor_app_is_started_stopped_and_read_through_its_own_controls() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let script = dir.path().join("server.py");
    std::fs::write(&script, SERVER_PY).unwrap();
    let log = dir.path().join("app.log");
    let pidfile = dir.path().join("app.pid");
    let start = format!(
        "nohup /usr/bin/python3 {} {port} >> {} 2>&1 & echo $! > {}",
        script.display(),
        log.display(),
        pidfile.display()
    );
    let stop = format!("kill $(cat {})", pidfile.display());
    let host = "mac-spike";
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "macos",
        "gpus": [{"host": host, "index": SPIKE_GPU}],
        "workloads": [{
            "name": "spike", "engine": "launchd", "engine_kind": "fake_engine",
            // What its endpoint must serve, not a program path: there is
            // no job to read a program from.
            "model": "spike/model",
            "gpus": [{"host": host, "index": SPIKE_GPU}], "port": port,
            "unit": {"scope": "launchd", "name": "com.pitcany.llmctl-next-spike-vendor"},
            "authority": {
                "kind": "vendor_command",
                "start": ["/bin/sh", "-c", start],
                "stop": ["/bin/sh", "-c", stop],
                "log": log
            },
            "ownership": {"kind": "external"},
            "gpu_claim": "on_demand",
            "readiness_timeout_secs": 30
        }],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let d = spawn_daemon(dir.path());
    let pid = || std::fs::read_to_string(&pidfile).ok().and_then(|p| p.trim().parse::<u32>().ok());
    struct Kill<'a>(&'a Path);
    impl Drop for Kill<'_> {
        fn drop(&mut self) {
            if let Ok(p) = std::fs::read_to_string(self.0) {
                let _ = Command::new("kill").arg(p.trim()).output();
            }
        }
    }
    let _cleanup = Kill(&pidfile);

    let before = spike_status(&d.socket);
    assert_eq!(before["process"], "Stopped", "nothing answers yet: {before}");

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait", "--key", "vendor-run"]);
    assert!(ok, "{text}");
    let up = spike_status(&d.socket);
    assert_eq!(up["process"], "Running", "{up}");
    assert_eq!(up["readiness"], "Ready", "{up}");
    assert_eq!(up["served_model"], "spike/model", "{up}");
    let first = pid().expect("the start command ran");

    // Its own log, through llmctl-next.
    let (ok, text) = cli(&d.socket, &["logs", "spike", "-n", "5"]);
    assert!(ok, "{text}");
    assert!(text.contains(&format!("serving on {port}")), "{text}");

    // A restart goes through both commands and gets a new process.
    let (ok, text) = cli(&d.socket, &["restart", "spike", "--wait", "--key", "vendor-restart"]);
    assert!(ok, "{text}");
    let second = pid().expect("started again");
    assert_ne!(first, second, "the process was replaced");

    // Stop: the endpoint goes quiet and the stop is a manual one.
    let (ok, text) = cli(&d.socket, &["stop", "spike", "--wait", "--key", "vendor-stop"]);
    assert!(ok, "{text}");
    let down = spike_status(&d.socket);
    assert_eq!(down["process"], "Stopped", "{down}");
    let (ok, text) = cli(&d.socket, &["intent", "spike", "--json"]);
    assert!(ok && text.contains("\"manual_stop\": true"), "{text}");
}
