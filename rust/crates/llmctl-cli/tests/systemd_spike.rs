//! M2 spike: the daemon driving a *disposable* systemd user unit through
//! the `systemd` adapter. Opt-in only — these touch the user's service
//! manager:
//!
//!     cargo test -p llmctl-cli --test systemd_spike -- --ignored --test-threads=1
//!
//! Each test writes its own `llmctl-next-spike-<id>.service` (a tiny python
//! server answering `/v1/models`), enrolls only that unit, and removes it
//! on drop. Nothing else in the user's systemd is enrolled or touched.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use llmctl_core::{
    ControlAuthority, GpuId, HostId, HostInventory, Platform, UnitName, UnitScope, UnitScopeMarker,
    WorkloadDefinition, WorkloadName,
};

const SERVER_PY: &str = r#"
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, model = int(sys.argv[1]), sys.argv[2]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/v1/models":
            self.send_response(404); self.end_headers(); return
        body = json.dumps({"object": "list", "data": [{"id": model}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
server = HTTPServer(("127.0.0.1", port), H)
print(f"serving {model} on {port}", flush=True)
server.serve_forever()
"#;

fn systemctl(args: &[&str]) -> std::process::Output {
    Command::new("systemctl")
        .arg("--user")
        .args(args)
        .output()
        .expect("systemctl")
}

/// A user unit file that exists only for the test's lifetime.
struct SpikeUnit {
    name: String,
    file: PathBuf,
}

impl SpikeUnit {
    fn create(dir: &Path, exec: &str) -> SpikeUnit {
        Self::create_with(dir, exec, "")
    }

    /// `extra` = additional `[Service]` lines.
    fn create_with(dir: &Path, exec: &str, extra: &str) -> SpikeUnit {
        let name = format!(
            "llmctl-next-spike-{}.service",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let unit_dir = PathBuf::from(std::env::var("HOME").unwrap()).join(".config/systemd/user");
        std::fs::create_dir_all(&unit_dir).unwrap();
        let unit = SpikeUnit {
            file: unit_dir.join(&name),
            name,
        };
        unit.rewrite(dir, exec, extra);
        unit
    }

    /// Replace the unit file and reload systemd (simulates someone editing it).
    fn rewrite(&self, dir: &Path, exec: &str, extra: &str) {
        std::fs::write(
            &self.file,
            format!(
                "[Unit]\nDescription=llmctl-next disposable spike ({})\n\n[Service]\nType=simple\nExecStart={exec}\n{extra}",
                dir.display()
            ),
        )
        .unwrap();
        assert!(systemctl(&["daemon-reload"]).status.success());
    }

    fn start(&self) {
        let out = systemctl(&["start", &self.name]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn main_pid(&self) -> u32 {
        let out = systemctl(&["show", &self.name, "--property=MainPID", "--value"]);
        String::from_utf8_lossy(&out.stdout).trim().parse().unwrap()
    }

    fn active_state(&self) -> String {
        let out = systemctl(&["show", &self.name, "--property=ActiveState", "--value"]);
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }
}

impl Drop for SpikeUnit {
    fn drop(&mut self) {
        // SIGKILL first: a test unit may trap TERM, and with
        // TimeoutStopSec=infinity a plain stop would hang the suite.
        let _ = systemctl(&["kill", "-s", "KILL", &self.name]);
        let _ = systemctl(&["stop", &self.name]);
        let _ = systemctl(&["reset-failed", &self.name]);
        let _ = std::fs::remove_file(&self.file);
        let _ = systemctl(&["daemon-reload"]);
    }
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

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Inventory enrolling exactly one systemd workload: the spike unit. The
/// unit holds no card (it is a plain HTTP server), so it is enrolled on a
/// notional index (9) no real GPU occupant can collide with — the foreign
/// occupancy guard then stays quiet for these drills while still active.
const SPIKE_GPU: u32 = 9;

/// Inventory with no workloads and the notional card available, for tests
/// that adopt a unit themselves (the bootstrapped sample inventory only
/// knows the real cards, which real occupants hold).
fn write_empty_inventory(dir: &Path) {
    let host = HostId::new("workstation").unwrap();
    let inv = serde_json::json!({
        "inventory": {
            "host": host.to_string(),
            "platform": "linux",
            "gpus": [
                {"host": host.to_string(), "index": 9}
            ],
            "workloads": [],
            "revision": 1
        }
    });
    std::fs::write(dir.join("inventory.json"), inv.to_string()).unwrap();
}

/// Inventory enrolling exactly one systemd workload: the spike unit.
fn write_inventory(dir: &Path, unit: &str, port: u16, model: &str) {
    let host = HostId::new("workstation").unwrap();
    let inv = HostInventory {
        host: host.clone(),
        platform: Platform::Linux,
        gpus: vec![GpuId::new(host.clone(), SPIKE_GPU)],
        workloads: vec![WorkloadDefinition {
            name: WorkloadName::new("spike").unwrap(),
            engine: "systemd".into(),
            engine_kind: llmctl_core::EngineKind::FakeEngine,
            model: model.into(),
            gpus: vec![GpuId::new(host, SPIKE_GPU)],
            port,
            unit: UnitName::systemd(UnitScopeMarker::User, unit).unwrap(),
            authority: ControlAuthority::SystemdUnit {
                scope: UnitScope::User,
                unit: unit.into(),
            },
            ownership: Default::default(),
            recovery: None,
            recipe: None,
            readiness_timeout_secs: None,
            gpu_claim: Default::default(),
            hooks: Vec::new(),
            pin_files: Vec::new(),
            observed_version: None,
        }],
        stores: Vec::new(),
        gpu_tolerated: Vec::new(),
        revision: 1,
    };
    std::fs::write(
        dir.join("inventory.json"),
        serde_json::to_vec_pretty(&serde_json::json!({ "inventory": inv })).unwrap(),
    )
    .unwrap();
}

fn spawn_daemon(dir: &Path) -> Daemon {
    let socket = dir.join("control.sock");
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir)
        .arg("daemon")
        // Spike units are transient and hold no real card: stay out of the
        // host's GPU operation lock and lane lease (M4c).
        .env("LLMCTL_NEXT_OP_LOCK", "off")
        .env("LLMCTL_NEXT_LANE_LEASE", "off")
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

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn run_and_stop_a_disposable_user_unit() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("server.py");
    std::fs::write(&script, SERVER_PY).unwrap();
    let port = free_port();
    let model = "spike/model";
    let unit = SpikeUnit::create(
        dir.path(),
        &format!("/usr/bin/python3 {} {port} {model}", script.display()),
    );
    write_inventory(dir.path(), &unit.name, port, model);
    let d = spawn_daemon(dir.path());

    let before = spike_status(&d.socket);
    assert_eq!(before["process"], "Stopped", "{before}");

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait", "--key", "spike-run"]);
    assert!(ok, "{text}");
    assert!(text.contains("succeeded"), "{text}");
    assert_eq!(unit.active_state(), "active");
    let up = spike_status(&d.socket);
    assert_eq!(up["process"], "Running", "{up}");
    assert_eq!(up["readiness"], "Ready", "{up}");

    let (ok, text) = cli(
        &d.socket,
        &["stop", "spike", "--wait", "--key", "spike-stop"],
    );
    assert!(ok, "{text}");
    assert_eq!(unit.active_state(), "inactive");
    let down = spike_status(&d.socket);
    assert_eq!(down["process"], "Stopped", "{down}");
}

/// Found by the M4c router pilot (2026-11-17): `vllm-engine` allows 3 starts
/// per 30 minutes, the pilot's fourth start was the *rollback's*, and
/// systemd refused it with start-limit-hit — the engine stayed down. An
/// admitted start is an operator decision, not a crash loop: the start-rate
/// counter is cleared before every start, as llm-router does.
#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn admitted_starts_are_not_refused_by_the_units_start_rate_limit() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("server.py");
    std::fs::write(&script, SERVER_PY).unwrap();
    let port = free_port();
    let model = "spike/model";
    let unit = SpikeUnit::create_with(
        dir.path(),
        &format!("/usr/bin/python3 {} {port} {model}", script.display()),
        // The limit lives in [Unit]; a second [Unit] section is merged.
        "\n[Unit]\nStartLimitIntervalSec=600\nStartLimitBurst=2\n\n[Install]\nWantedBy=default.target\n",
    );
    let shown = systemctl(&["show", &unit.name, "--property=StartLimitBurst", "--value"]);
    assert_eq!(String::from_utf8_lossy(&shown.stdout).trim(), "2", "the limit is in force");
    // systemd forgets the counter when it unloads an inactive unit nothing
    // refers to, which is what a bare spike unit is. A real engine unit is
    // enabled, so `default.target` keeps it loaded across a stop: same here.
    struct Enabled(String);
    impl Drop for Enabled {
        fn drop(&mut self) {
            let _ = systemctl(&["disable", &self.0]);
        }
    }
    assert!(systemctl(&["enable", &unit.name]).status.success());
    let _enabled = Enabled(unit.name.clone());
    write_inventory(dir.path(), &unit.name, port, model);
    let d = spawn_daemon(dir.path());

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait", "--key", "limit-run"]);
    assert!(ok, "{text}");
    // Three more starts inside the interval: well past the burst of 2.
    for i in 0..3 {
        let key = format!("limit-restart-{i}");
        let (ok, text) = cli(&d.socket, &["restart", "spike", "--wait", "--key", &key]);
        assert!(ok, "start {} was refused: {text}", i + 2);
        assert_eq!(unit.active_state(), "active");
    }
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn restart_replaces_the_process_and_reverifies() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("server.py");
    std::fs::write(&script, SERVER_PY).unwrap();
    let port = free_port();
    let model = "spike/model";
    let unit = SpikeUnit::create(
        dir.path(),
        &format!("/usr/bin/python3 {} {port} {model}", script.display()),
    );
    write_inventory(dir.path(), &unit.name, port, model);
    let d = spawn_daemon(dir.path());

    // Not running yet: refused, and nothing is started behind our back.
    let (code, text) = cli_code(&d.socket, &["restart", "spike", "--wait"]);
    assert_eq!(code, Some(4), "{text}");
    assert_eq!(unit.active_state(), "inactive");

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait"]);
    assert!(ok, "{text}");
    let before = unit.main_pid();
    assert_ne!(before, 0);

    let (ok, text) = cli(&d.socket, &["restart", "spike", "--wait"]);
    assert!(ok, "{text}");
    assert!(text.contains("succeeded"), "{text}");
    let after = unit.main_pid();
    assert_ne!(after, 0);
    assert_ne!(before, after, "restart must replace the main process");
    let up = spike_status(&d.socket);
    assert_eq!(up["readiness"], "Ready", "{up}");
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn logs_show_output_and_run_boundaries_and_follow_a_restart() {
    use std::io::BufRead;
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("server.py");
    std::fs::write(&script, SERVER_PY).unwrap();
    let port = free_port();
    let model = "spike/model";
    let unit = SpikeUnit::create(
        dir.path(),
        &format!("/usr/bin/python3 {} {port} {model}", script.display()),
    );
    write_inventory(dir.path(), &unit.name, port, model);
    let d = spawn_daemon(dir.path());

    let (ok, text) = cli(&d.socket, &["run", "spike", "--wait"]);
    assert!(ok, "{text}");
    let (ok, text) = cli(&d.socket, &["logs", "spike"]);
    assert!(ok, "{text}");
    assert!(
        text.contains(&format!("serving {model} on {port}")),
        "{text}"
    );
    assert!(
        text.contains("systemd: Started llmctl-next disposable spike"),
        "{text}"
    );
    assert!(text.contains("-- run "), "{text}");

    let (ok, text) = cli(&d.socket, &["logs", "spike", "--json", "-n", "3"]);
    assert!(ok, "{text}");
    let entries: Vec<serde_json::Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(!entries.is_empty() && entries.len() <= 3, "{text}");
    assert!(
        entries.iter().all(|e| e["invocation"].is_string()),
        "{text}"
    );

    // Follow: a restart shows up as the old run stopping and a new run
    // serving, under a second run marker.
    let mut follower = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&d.socket)
        .args(["logs", "spike", "-f", "-n", "1"])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = follower.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
        {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let (ok, text) = cli(&d.socket, &["restart", "spike", "--wait"]);
    assert!(ok, "{text}");
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut seen = Vec::new();
    let mut stopped = false;
    let mut served_again = false;
    while !served_again {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let Ok(line) = rx.recv_timeout(left) else {
            break;
        };
        stopped |= line.contains("systemd: Stopped");
        served_again |= stopped && line.contains(&format!("serving {model}"));
        seen.push(line);
    }
    let _ = follower.kill();
    let _ = follower.wait();
    let runs = seen.iter().filter(|l| l.starts_with("-- run ")).count();
    assert!(
        served_again,
        "follow never showed the restarted run: {seen:#?}"
    );
    assert!(runs >= 2, "expected a new run marker: {seen:#?}");

    // A cursor the journal does not know is a request error, not a 500.
    let out = Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--unix-socket",
        ])
        .arg(&d.socket)
        .args([
            "-H",
            "content-type: application/json",
            "-d",
            r#"{"workload":"spike","after":"s=deadbeef;i=1"}"#,
            "http://localhost/v1/logs",
        ])
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout), "400");
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn unit_that_exits_before_readiness_fails_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    // Exits immediately: never answers /v1/models.
    let unit = SpikeUnit::create(dir.path(), "/usr/bin/python3 -c 'import sys; sys.exit(3)'");
    write_inventory(dir.path(), &unit.name, port, "spike/model");
    let d = spawn_daemon(dir.path());

    let (ok, text) = cli(
        &d.socket,
        &["run", "spike", "--wait", "--key", "spike-crash"],
    );
    assert!(!ok, "run of a crashing unit must not succeed: {text}");
    assert!(
        text.contains("exited before readiness"),
        "failure should name the early exit, not a timeout: {text}"
    );
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn wrong_served_model_fails_verification() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("server.py");
    std::fs::write(&script, SERVER_PY).unwrap();
    let port = free_port();
    let unit = SpikeUnit::create(
        dir.path(),
        &format!("/usr/bin/python3 {} {port} other/model", script.display()),
    );
    write_inventory(dir.path(), &unit.name, port, "spike/model");
    let d = spawn_daemon(dir.path());

    let (ok, text) = cli(
        &d.socket,
        &["run", "spike", "--wait", "--key", "spike-wrong"],
    );
    assert!(!ok, "{text}");
    assert!(text.contains("other/model"), "{text}");
}

// ---- adoption ---------------------------------------------------------------

/// Serving spike unit whose argv carries an argument with a space, so the
/// round trip has to keep exact argument boundaries.
fn serving_exec(dir: &Path, port: u16, model: &str) -> String {
    let script = dir.join("server.py");
    std::fs::write(&script, SERVER_PY).unwrap();
    format!(
        "/usr/bin/python3 {} {port} {model} \"--note=two words\"",
        script.display()
    )
}

fn adopt_args<'a>(unit: &'a str, port: &'a str) -> Vec<&'a str> {
    vec![
        "adopt",
        unit,
        "--name",
        "adopted",
        "--model",
        "spike/model",
        "--port",
        port,
        "--gpus",
        "9",
    ]
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

fn preview_digest(socket: &Path, args: &[&str]) -> (serde_json::Value, String) {
    let mut a = args.to_vec();
    a.push("--json");
    let (_, text) = cli_code(socket, &a);
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|_| panic!("{text}"));
    let digest = v["digest"].as_str().unwrap_or("").to_string();
    (v, digest)
}

fn check_status(preview: &serde_json::Value, name: &str) -> String {
    preview["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .map(|c| c["status"].as_str().unwrap().to_string())
        .unwrap_or_else(|| panic!("no check {name}: {preview}"))
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn adopt_running_unit_then_stop_and_run_it_as_managed() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let unit = SpikeUnit::create(dir.path(), &serving_exec(dir.path(), port, "spike/model"));
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path()); // sample fake inventory; spike not enrolled
    let port_s = port.to_string();
    let args = adopt_args(&unit.name, &port_s);

    // Wait for the server to answer, then preview.
    let mut preview = serde_json::Value::Null;
    let mut digest = String::new();
    for _ in 0..50 {
        (preview, digest) = preview_digest(&d.socket, &args);
        if check_status(&preview, "served_identity") == "pass" {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    for check in [
        "enrollment",
        "unit_loaded",
        "no_inline_secrets",
        "definition_roundtrip",
        "served_identity",
        "ownable",
    ] {
        assert_eq!(check_status(&preview, check), "pass", "{check}: {preview}");
    }
    let argv = &preview["definition"]["exec_start"][0]["argv"];
    assert_eq!(argv[4], "--note=two words", "exact argv: {argv}");
    assert!(digest.starts_with("sha256:"));

    let mut confirm = args.clone();
    confirm.extend(["--confirm", &digest]);
    let (code, text) = cli_code(&d.socket, &confirm);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("adopted"), "{text}");

    let status = spike_status_named(&d.socket, "adopted");
    assert_eq!(status["ownership"], "managed", "{status}");
    assert_eq!(status["adopted_digest"], digest.as_str());
    assert_eq!(status["readiness"], "Ready", "{status}");

    let (ok, text) = cli(&d.socket, &["stop", "adopted", "--wait", "--key", "a-stop"]);
    assert!(ok, "{text}");
    assert_eq!(unit.active_state(), "inactive");
    let (ok, text) = cli(&d.socket, &["run", "adopted", "--wait", "--key", "a-run"]);
    assert!(ok, "{text}");
    assert_eq!(unit.active_state(), "active");

    // Survives a daemon restart: the inventory file holds the adoption.
    drop(d);
    let d = spawn_daemon(dir.path());
    let status = spike_status_named(&d.socket, "adopted");
    assert_eq!(status["ownership"], "managed", "{status}");
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn confirm_with_a_stale_digest_is_refused_with_a_fresh_preview() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let exec = serving_exec(dir.path(), port, "spike/model");
    let unit = SpikeUnit::create(dir.path(), &exec);
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    let port_s = port.to_string();
    let args = adopt_args(&unit.name, &port_s);
    let (_, old_digest) = preview_digest(&d.socket, &args);

    unit.rewrite(dir.path(), &exec, "Environment=EXTRA=1\n");
    let mut confirm = args.clone();
    confirm.extend(["--confirm", &old_digest, "--allow-unverified"]);
    let (code, text) = cli_code(&d.socket, &confirm);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("stale_adoption"), "{text}");
    assert!(
        text.contains("definition digest: sha256:"),
        "fresh preview shown: {text}"
    );
    assert!(!text.contains(&old_digest), "fresh digest differs: {text}");
    let (ok, text) = cli(&d.socket, &["status", "--json"]);
    assert!(ok && !text.contains("\"adopted\""), "not enrolled: {text}");
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn drift_after_adoption_blocks_run_without_touching_the_unit() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let exec = serving_exec(dir.path(), port, "spike/model");
    let unit = SpikeUnit::create(dir.path(), &exec);
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    let port_s = port.to_string();
    let args = adopt_args(&unit.name, &port_s);
    let (preview, digest) = preview_digest(&d.socket, &args);
    assert_eq!(check_status(&preview, "served_identity"), "unverified");

    // Stopped unit: unverified identity needs the explicit flag.
    let mut confirm = args.clone();
    confirm.extend(["--confirm", &digest]);
    let (code, text) = cli_code(&d.socket, &confirm);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("adoption_unverified"), "{text}");
    confirm.push("--allow-unverified");
    let (code, text) = cli_code(&d.socket, &confirm);
    assert_eq!(code, Some(0), "{text}");

    unit.rewrite(dir.path(), &exec, "Restart=on-failure\n");
    let (code, text) = cli_code(
        &d.socket,
        &["run", "adopted", "--wait", "--key", "drift-run"],
    );
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("drifted"), "{text}");
    assert!(
        text.contains("restart"),
        "names the differing field: {text}"
    );
    assert_eq!(
        unit.active_state(),
        "inactive",
        "no effect on a drifted unit"
    );
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn inline_secret_blocks_adoption_and_is_never_echoed() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let unit = SpikeUnit::create_with(
        dir.path(),
        &serving_exec(dir.path(), port, "spike/model"),
        "Environment=HF_TOKEN=dummy-not-a-real-token-xyz\n",
    );
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    let port_s = port.to_string();
    let args = adopt_args(&unit.name, &port_s);
    let (code, text) = cli_code(&d.socket, &args);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("FAIL       no_inline_secrets"), "{text}");
    assert!(text.contains("HF_TOKEN"), "{text}");
    assert!(
        !text.contains("dummy-not-a-real-token-xyz"),
        "secret value leaked: {text}"
    );
    let (preview, digest) = preview_digest(&d.socket, &args);
    assert!(!preview.to_string().contains("dummy-not-a-real-token-xyz"));

    let mut confirm = args.clone();
    confirm.extend(["--confirm", &digest, "--allow-unverified"]);
    let (code, text) = cli_code(&d.socket, &confirm);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("adoption_refused"), "{text}");
}

fn spike_status_named(socket: &Path, name: &str) -> serde_json::Value {
    let (ok, text) = cli(socket, &["status", "--json"]);
    assert!(ok, "{text}");
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["workloads"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["name"] == name)
        .cloned()
        .unwrap_or_else(|| panic!("no workload {name}: {json}"))
}

// ---- B: bounded recovery + stop timeout -------------------------------------

/// Daemon with a fast watcher and an environment override.
fn spawn_daemon_env(dir: &Path, envs: &[(&str, &str)]) -> Daemon {
    let socket = dir.join("control.sock");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_llmctl-next"));
    cmd.arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir)
        .arg("daemon")
        .env("LLMCTL_NEXT_OP_LOCK", "off")
        .env("LLMCTL_NEXT_LANE_LEASE", "off");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    // Daemon log goes next to the data dir so failures can be read after
    // the fact.
    let log = std::fs::File::create(dir.join("daemon.log")).expect("daemon log");
    let child = cmd
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .expect("spawn daemon");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket).is_err() {
        assert!(std::time::Instant::now() < deadline, "daemon never bound");
        std::thread::sleep(Duration::from_millis(50));
    }
    Daemon { child, socket }
}

fn write_inventory_recovery(
    dir: &Path,
    unit: &str,
    port: u16,
    model: &str,
    policy: serde_json::Value,
) {
    let host = HostId::new("workstation").unwrap();
    let inv = serde_json::json!({
        "inventory": {
            "host": host.to_string(),
            "platform": "linux",
            "gpus": [{"host": host.to_string(), "index": 9}],
            "workloads": [{
                "name": "spike",
                "engine": "systemd",
                "model": model,
                "gpus": [{"host": host.to_string(), "index": 9}],
                "port": port,
                "unit": {"scope": "user", "name": unit},
                "authority": {"kind": "systemd_unit", "scope": "user", "unit": unit},
                "ownership": {"kind": "external"},
                "recovery": policy
            }],
            "revision": 1
        }
    });
    std::fs::write(dir.join("inventory.json"), inv.to_string()).unwrap();
}

/// A server that exits non-zero until `gate` exists, then serves — so
/// systemd's own retry loop burns out before llmctl recovers it.
const GATED_PY: &str = r#"
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, model, gate = int(sys.argv[1]), sys.argv[2], sys.argv[3]
if not os.path.exists(gate):
    print(f"gate {gate} absent; failing", flush=True)
    sys.exit(1)
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/v1/models":
            self.send_response(404); self.end_headers(); return
        body = json.dumps({"object": "list", "data": [{"id": model}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
server = HTTPServer(("127.0.0.1", port), H)
print(f"serving {model} on {port}", flush=True)
server.serve_forever()
"#;

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn recovery_waits_out_systemd_retries_then_recovers_after_start_limit() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("gated.py");
    std::fs::write(&script, GATED_PY).unwrap();
    let port = free_port();
    let model = "spike/model";
    let gate = dir.path().join("gate");
    let unit = SpikeUnit::create_with(
        dir.path(),
        &format!(
            "/usr/bin/python3 {} {port} {model} {}",
            script.display(),
            gate.display()
        ),
        "Restart=on-failure\nRestartSec=1\nStartLimitIntervalSec=20\nStartLimitBurst=3\n",
    );
    write_inventory_recovery(
        dir.path(),
        &unit.name,
        port,
        model,
        serde_json::json!({"max_attempts": 3, "window_secs": 3600, "min_interval_secs": 0}),
    );
    // The watcher is slower than the poll below, so the test reliably sees
    // the manager's start-limit state before llmctl acts on it.
    let d = spawn_daemon_env(dir.path(), &[("LLMCTL_NEXT_WATCH_INTERVAL_MS", "1500")]);

    // Start healthy: intent Running, unit serving.
    std::fs::write(&gate, b"ok").unwrap();
    let (ok, text) = cli(
        &d.socket,
        &["run", "spike", "--wait", "--key", "sr-healthy"],
    );
    assert!(ok, "{text}");
    let run_at = chrono::Utc::now();

    // The engine breaks while the owner is not looking: remove the gate,
    // kill the main pid. systemd burns its own retries (the watcher only
    // reports), hits start-limit, gives up.
    std::fs::remove_file(&gate).unwrap();
    let _ = systemctl(&["kill", "-s", "KILL", &unit.name]);
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        if unit.active_state() == "failed" {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "never hit start limit (state {})",
            unit.active_state()
        );
        std::thread::sleep(Duration::from_millis(200));
    }

    // The fault clears; the watcher's recovery must bring it back.
    std::fs::write(&gate, b"ok").unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let (ok, text) = cli(&d.socket, &["ops", "--json"]);
        assert!(ok, "{text}");
        let ops: serde_json::Value = serde_json::from_str(&text).unwrap();
        let recoveries: Vec<_> = ops["operations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|o| {
                o["idempotency_key"]
                    .as_str()
                    .unwrap_or("")
                    .starts_with("recover-spike-")
            })
            .collect();
        let last = recoveries.last().cloned();
        if let Some(one) = last {
            if one["status"] == "succeeded" {
                assert!(!recoveries.is_empty(), "{recoveries:?}");
                assert_eq!(unit.active_state(), "active");
                let up = spike_status(&d.socket);
                assert_eq!(up["readiness"], "Ready", "{up}");
                // Both loops in order: the healthy run predates the recovery.
                assert!(
                    one["created_at"].as_str().unwrap() > run_at.to_rfc3339().as_str(),
                    "recovery must follow the original run"
                );
                return;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "recovery never succeeded: {}",
            cli(&d.socket, &["ops"]).1
        );
        std::thread::sleep(Duration::from_millis(300));
    }
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn a_stop_that_outlives_the_effect_timeout_is_unconfirmed() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    // Ignores SIGTERM forever; the manager never finishes stopping it.
    let unit = SpikeUnit::create_with(
        dir.path(),
        "/bin/bash -c 'trap \"\" TERM; while :; do sleep 1; done'",
        "TimeoutStopSec=infinity\n",
    );
    unit.start();
    write_inventory(dir.path(), &unit.name, port, "spike/model");
    let d = spawn_daemon_env(
        dir.path(),
        &[("LLMCTL_NEXT_SYSTEMD_EFFECT_TIMEOUT_MS", "6000")],
    );

    let (ok, text) = cli(
        &d.socket,
        &["stop", "spike", "--wait", "--key", "stuck-stop"],
    );
    assert!(!ok, "a stop that cannot confirm must not succeed: {text}");
    assert!(
        text.to_lowercase().contains("unconfirmed"),
        "failure must name the unconfirmed effect: {text}"
    );
    // Observed honestly: the process is still there (stuck deactivating —
    // the manager is waiting on TimeoutStopSec=infinity), and the op says so.
    let state = unit.active_state();
    assert!(
        state == "deactivating" || state == "active",
        "process must still exist: {state}"
    );
    assert!(unit.main_pid() != 0);

    // The manager can still be persuaded by hand (SIGKILL); llmctl never
    // pretends it happened.
    let _ = systemctl(&["kill", "-s", "KILL", &unit.name]);
}

// ---- C: own / release -------------------------------------------------------

/// Adopt the (running, serving) unit as workload `adopted`; returns the CLI
/// exit ok-ness. Port shared by preview/transfer.
fn adopt_running(socket: &Path, unit: &str, port: u16) {
    let port_s = port.to_string();
    let args = adopt_args(unit, &port_s);
    let (mut preview, mut digest) = preview_digest(socket, &args);
    for _ in 0..50 {
        if check_status(&preview, "served_identity") == "pass" {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
        (preview, digest) = preview_digest(socket, &args);
    }
    let mut confirm = args.clone();
    confirm.extend(["--confirm", &digest]);
    let (code, text) = cli_code(socket, &confirm);
    assert_eq!(code, Some(0), "{text}");
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn own_replaces_the_file_keeps_the_pid_and_guards_the_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let unit = SpikeUnit::create(dir.path(), &serving_exec(dir.path(), port, "spike/model"));
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &unit.name, port);

    let pid_before = unit.main_pid();
    assert_ne!(pid_before, 0);
    let (code, text) = cli_code(&d.socket, &["own", "adopted"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("previous: file"), "{text}");
    assert_eq!(unit.main_pid(), pid_before, "own must not restart the unit");
    assert_eq!(unit.active_state(), "active");

    // The file is now llmctl's, and the drift guard watches its bytes.
    let text_now = std::fs::read_to_string(&unit.file).unwrap();
    assert!(
        text_now.contains("llmctl-next managed workload"),
        "{text_now}"
    );
    let (ok, text) = cli(&d.socket, &["run", "adopted", "--wait", "--key", "own-run"]);
    assert!(ok, "equivalent definition must not block run: {text}");

    // Someone edits the owned file → drift names the service file.
    std::fs::write(&unit.file, format!("{text_now}Environment=TOUCHED=1\n")).unwrap();
    assert!(systemctl(&["daemon-reload"]).status.success());
    let (code, text) = cli_code(
        &d.socket,
        &["run", "adopted", "--wait", "--key", "own-drift"],
    );
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("service_file"), "{text}");
    assert_eq!(unit.active_state(), "active", "no effect on a drifted unit");

    // Release hands the original back byte-for-byte and unmanages it.
    let original = std::fs::read_to_string(&unit.file).unwrap();
    let (code, text) = cli_code(&d.socket, &["release", "adopted"]);
    assert_eq!(code, Some(0), "{text}");
    let status = spike_status_named(&d.socket, "adopted");
    assert_eq!(status["ownership"], "external", "{status}");
    let (ok, text) = cli(
        &d.socket,
        &["run", "adopted", "--wait", "--key", "post-release"],
    );
    assert!(ok, "external ownership has no pinned drift: {text}");
    assert_ne!(unit.main_pid(), 0);
    let _ = original; // byte identity is asserted in the symlink test below
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn own_a_symlinked_unit_replaces_the_link_and_release_restores_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let script = dir.path().join("server.py");
    std::fs::write(&script, SERVER_PY).unwrap();
    let repo_unit = dir.path().join("repo-unit.service");
    let original_bytes = format!(
        "[Unit]\nDescription=repo copy\n\n[Service]\nType=simple\nExecStart=/usr/bin/python3 {} {port} spike/model\n",
        script.display()
    );
    std::fs::write(&repo_unit, &original_bytes).unwrap();
    let name = format!(
        "llmctl-next-spike-{}.service",
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    );
    let link = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".config/systemd/user")
        .join(&name);
    std::os::unix::fs::symlink(&repo_unit, &link).unwrap();
    struct LinkGuard {
        name: String,
        link: PathBuf,
    }
    impl Drop for LinkGuard {
        fn drop(&mut self) {
            let _ = systemctl(&["kill", "-s", "KILL", &self.name]);
            let _ = systemctl(&["stop", &self.name]);
            let _ = systemctl(&["reset-failed", &self.name]);
            let _ = std::fs::remove_file(&self.link);
            let _ = systemctl(&["daemon-reload"]);
        }
    }
    let guard = LinkGuard {
        name: name.clone(),
        link: link.clone(),
    };
    assert!(systemctl(&["daemon-reload"]).status.success());
    assert!(systemctl(&["start", &name]).status.success());
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &name, port);
    let repo_before = std::fs::read(&repo_unit).unwrap();

    let (code, text) = cli_code(&d.socket, &["own", "adopted"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("previous: link"), "{text}");
    assert!(
        !std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink(),
        "the link itself must be replaced, never written through"
    );
    assert_eq!(
        std::fs::read(&repo_unit).unwrap(),
        repo_before,
        "the repo copy the link pointed at is untouched"
    );
    assert_eq!(
        String::from_utf8_lossy(
            &systemctl(&["show", &name, "--property=ActiveState", "--value"]).stdout
        )
        .trim(),
        "active"
    );

    // Release: the link comes back, pointing at the untouched repo copy.
    let (code, text) = cli_code(&d.socket, &["release", "adopted"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(std::fs::symlink_metadata(&link)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        std::fs::read_link(&link).unwrap(),
        repo_unit,
        "link target restored"
    );
    assert_eq!(std::fs::read(&repo_unit).unwrap(), repo_before);
    let _ = guard;
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn a_bad_render_is_caught_restored_and_refused() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let unit = SpikeUnit::create(dir.path(), &serving_exec(dir.path(), port, "spike/model"));
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &unit.name, port);
    let before = std::fs::read(&unit.file).unwrap();

    // Test hook: own renders a subtly wrong unit; the prove step must
    // catch it and put the original back.
    std::fs::write(
        dir.path().join("faults.json"),
        serde_json::json!({"corrupt_render": true}).to_string(),
    )
    .unwrap();
    let (code, text) = cli_code(&d.socket, &["own", "adopted"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("ownership_refused"), "{text}");
    assert_eq!(
        std::fs::read(&unit.file).unwrap(),
        before,
        "the original must be back byte-for-byte"
    );
    // And the unit was never left drifted from its pin.
    let (ok, text) = cli(
        &d.socket,
        &["run", "adopted", "--wait", "--key", "bad-render-ok"],
    );
    assert!(ok, "{text}");
}

/// A directive the definition does not carry: the preview warns that own
/// would refuse (M3 pilot: ProcessType passed every check, then own
/// refused), adoption still goes through, and own then refuses by name.
#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn an_uncovered_directive_warns_at_preview_and_own_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let unit = SpikeUnit::create_with(
        dir.path(),
        &serving_exec(dir.path(), port, "spike/model"),
        "Nice=5\n",
    );
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    let port_s = port.to_string();
    let args = adopt_args(&unit.name, &port_s);
    let (mut preview, mut digest) = preview_digest(&d.socket, &args);
    for _ in 0..50 {
        if check_status(&preview, "served_identity") == "pass" {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
        (preview, digest) = preview_digest(&d.socket, &args);
    }
    assert_eq!(check_status(&preview, "ownable"), "warn", "{preview}");
    let detail = preview["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "ownable")
        .unwrap()["detail"]
        .to_string();
    assert!(detail.contains("[Service] Nice"), "{detail}");

    let mut confirm = args.clone();
    confirm.extend(["--confirm", &digest]);
    let (code, text) = cli_code(&d.socket, &confirm);
    assert_eq!(code, Some(0), "a warning does not block adoption: {text}");
    let (code, text) = cli_code(&d.socket, &["own", "adopted"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("Nice"), "{text}");
}

#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn a_unit_with_a_dropin_is_refused_for_own() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let unit = SpikeUnit::create(dir.path(), &serving_exec(dir.path(), port, "spike/model"));
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &unit.name, port);

    // A drop-in appears after adoption: own must refuse (a rendered unit
    // cannot own a definition the drop-in modifies).
    let drop_dir = unit.file.with_extension("service.d");
    std::fs::create_dir_all(&drop_dir).unwrap();
    std::fs::write(
        drop_dir.join("override.conf"),
        "[Service]\nEnvironment=EXTRA=1\n",
    )
    .unwrap();
    assert!(systemctl(&["daemon-reload"]).status.success());
    let (code, text) = cli_code(&d.socket, &["own", "adopted"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("drop-in"), "{text}");
    assert!(
        text.contains("override.conf"),
        "the drop-in is listed: {text}"
    );
    let _ = std::fs::remove_dir_all(&drop_dir);
    assert!(systemctl(&["daemon-reload"]).status.success());
}

// ---- D: staged upgrade with rollback ----------------------------------------

const FAKEENGINE_PKG: &str = r#"
import json as _json
__version__ = "{VERSION}"
BROKEN = {BROKEN}
def validate(model_config):
    if not model_config:
        raise ValueError("no model config")
    if BROKEN:
        raise RuntimeError("fakeengine {VERSION} is broken on purpose")
    cfg = _json.load(open(model_config))
    return {"version": __version__, "model_type": cfg.get("model_type")}
"#;

const FAKEENGINE_MAIN: &str = r#"
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from fakeengine import __version__
port, model = int(sys.argv[1]), sys.argv[2]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/version":
            body = json.dumps({"version": __version__}).encode()
        elif self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": model}]}).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
"#;

/// Build fakeengine wheels (1.0, 2.0, 2.0-broken) into a wheelhouse.
fn build_fakeengine_wheels(root: &Path) -> PathBuf {
    let wheelhouse = root.join("wheelhouse");
    std::fs::create_dir_all(&wheelhouse).unwrap();
    for (version, broken) in [("1.0", false), ("2.0", false), ("2.0.1", true)] {
        let src = root.join(format!("pkg-{version}"));
        let pkg = src.join("src").join("fakeengine");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::write(
            src.join("pyproject.toml"),
            format!(
                "[project]\nname = \"fakeengine\"\nversion = \"{version}\"\n\n[build-system]\nrequires = [\"setuptools>=61\"]\nbuild-backend = \"setuptools.build_meta\"\n\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\n"
            ),
        )
        .unwrap();
        std::fs::write(
            pkg.join("__init__.py"),
            FAKEENGINE_PKG
                .replace("{VERSION}", version)
                .replace("{BROKEN}", if broken { "True" } else { "False" }),
        )
        .unwrap();
        std::fs::write(pkg.join("__main__.py"), FAKEENGINE_MAIN).unwrap();
        let out = Command::new("uv")
            .args(["build", "--wheel", "-o"])
            .arg(&wheelhouse)
            .arg(&src)
            .output()
            .expect("uv build");
        assert!(
            out.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    wheelhouse
}

fn make_env(root: &Path, wheelhouse: &Path, version: &str) -> PathBuf {
    let env = root.join(format!("env-{version}"));
    let out = Command::new("uv")
        .args(["venv"])
        .arg(&env)
        .output()
        .expect("uv venv");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let out = Command::new("uv")
        .args(["pip", "install", "--python"])
        .arg(env.join("bin/python"))
        .arg(format!("fakeengine=={version}"))
        .arg("--find-links")
        .arg(wheelhouse)
        .output()
        .expect("uv pip install");
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    env
}

fn http_get_json(port: u16, path: &str) -> serde_json::Value {
    let out = Command::new("curl")
        .args(["-sf", &format!("http://127.0.0.1:{port}{path}")])
        .output()
        .expect("curl");
    serde_json::from_slice(&out.stdout).unwrap_or(serde_json::Value::Null)
}

#[test]
#[ignore = "touches the user's systemd and builds wheels: run with --ignored"]
fn staged_upgrade_promotes_verifies_and_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let wheelhouse = build_fakeengine_wheels(dir.path());
    let env1 = make_env(dir.path(), &wheelhouse, "1.0");
    let env2 = make_env(dir.path(), &wheelhouse, "2.0");
    let port = free_port();
    let model = "spike/model";

    // The served model's config: what offline validation must load.
    std::fs::write(
        dir.path().join("model-config.json"),
        r#"{"model_type": "fake"}"#,
    )
    .unwrap();

    let unit = SpikeUnit::create_with(
        dir.path(),
        &format!(
            "{} -m fakeengine {port} {model}",
            env1.join("bin/python").display()
        ),
        &format!("WorkingDirectory={}\n", dir.path().display()),
    );
    unit.start();
    let original_unit = std::fs::read(&unit.file).unwrap();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &unit.name, port);
    let (ok, text) = cli(&d.socket, &["own", "adopted"]);
    assert!(ok, "{text}");

    // The owner adds a recipe to the inventory and restarts the daemon:
    // the startup re-pin probes the env (schema 2, fakeengine 1.0).
    drop(d);
    patch_inventory_recipe(dir.path(), &wheelhouse);
    let d = spawn_daemon(dir.path());
    let inv = std::fs::read_to_string(dir.path().join("inventory.json")).unwrap();
    assert!(
        inv.contains("\"definition_schema\": 2") && inv.contains("fakeengine"),
        "re-pin must have recorded version facts: {inv}"
    );

    // Stage + offline-validate 2.0 into the daemon's env root.
    let (ok, text) = cli(&d.socket, &["upgrade", "stage", "adopted", "2.0"]);
    assert!(ok, "{text}");
    assert!(text.contains("validated"), "{text}");

    // Promote: stop, apply definition, start, verify served identity and
    // the version itself. The unit must now serve 2.0.
    let (ok, text) = cli(
        &d.socket,
        &["upgrade", "promote", "adopted", "2.0", "--wait"],
    );
    assert!(ok, "{text}");
    assert!(
        text.contains("verify adopted runs fakeengine=2.0"),
        "{text}"
    );
    assert_eq!(
        http_get_json(port, "/v1/version")["version"],
        "2.0",
        "the promoted env must be serving"
    );
    // History is settled on read: no wait between `promote --wait` and
    // `list` or `rollback` (M2 carry-over: a 2 s poller lagged behind).
    let rows = cli(&d.socket, &["upgrade", "list", "adopted"]).1;
    assert_eq!(upgrade_rows(&rows, "promoted", "2.0"), 1, "{rows}");

    // A promoted workload stays operable: its version pin moved with the
    // promote, so a restart is not refused as version drift.
    let (code, text) = cli_code(
        &d.socket,
        &[
            "restart",
            "adopted",
            "--wait",
            "--key",
            "restart-after-promote",
        ],
    );
    assert_eq!(code, Some(0), "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "2.0");

    // Explicit rollback to N, straight away: definition and env back to 1.0.
    let (ok, text) = cli(&d.socket, &["upgrade", "rollback", "adopted", "--wait"]);
    assert!(ok, "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "1.0");

    // The same version again (pilot window 2): restage, promote, roll back.
    // Each is a new operation — a key built from the version alone
    // replayed the first promote's success and served nothing new.
    let (ok, text) = cli(&d.socket, &["upgrade", "stage", "adopted", "2.0"]);
    assert!(ok, "{text}");
    // Restaging adds a row; the first promote/rollback record stays (M2
    // carry-over: one row per env id was overwritten).
    let rows = cli(&d.socket, &["upgrade", "list", "adopted"]).1;
    assert_eq!(upgrade_rows(&rows, "rolled_back", "2.0"), 1, "{rows}");
    assert_eq!(upgrade_rows(&rows, "validated", "2.0"), 1, "{rows}");
    let (ok, text) = cli(
        &d.socket,
        &["upgrade", "promote", "adopted", "2.0", "--wait"],
    );
    assert!(ok, "{text}");
    assert!(!text.contains("replaying"), "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "2.0");
    let (ok, text) = cli(&d.socket, &["upgrade", "rollback", "adopted", "--wait"]);
    assert!(ok, "{text}");
    assert!(!text.contains("replaying"), "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "1.0");
    let rows = cli(&d.socket, &["upgrade", "list", "adopted"]).1;
    assert_eq!(upgrade_rows(&rows, "rolled_back", "2.0"), 2, "{rows}");

    // Restaging rebuilds the env dir: refused for the env that is serving
    // and for the env the last promote would roll back to.
    let (ok, text) = cli(&d.socket, &["upgrade", "stage", "adopted", "2.0"]);
    assert!(ok, "{text}");
    let (ok, text) = cli(
        &d.socket,
        &["upgrade", "promote", "adopted", "2.0", "--wait"],
    );
    assert!(ok, "{text}");
    let (code, text) = cli_code(&d.socket, &["upgrade", "stage", "adopted", "2.0"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("live install"), "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "2.0");
    let (ok, text) = cli(&d.socket, &["upgrade", "stage", "adopted", "1.0"]);
    assert!(ok, "{text}");
    let (ok, text) = cli(
        &d.socket,
        &["upgrade", "promote", "adopted", "1.0", "--wait"],
    );
    assert!(ok, "{text}");
    let (code, text) = cli_code(&d.socket, &["upgrade", "stage", "adopted", "2.0"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("rollback target"), "{text}");

    // gc keeps every env the promote chain can still use (live 1.0, its
    // rollback target 2.0) and removes an interrupted stage; entries that
    // are not this workload's — another workload, a name that only looks
    // like ours — are never touched (the envs dir is shared).
    let envs = dir.path().join("envs");
    let orphan = envs.join(env_id("adopted", "9.9"));
    std::fs::create_dir_all(orphan.join("bin")).unwrap();
    std::fs::write(
        envs.join(format!("{}.staging", env_id("adopted", "9.9"))),
        "x",
    )
    .unwrap();
    std::fs::write(envs.join(format!("{}.log", env_id("adopted", "9.9"))), "x").unwrap();
    let foreign = [
        envs.join(env_id("adopted2", "1.0")),
        envs.join("adopted-1.0-00000000"),
    ];
    for f in &foreign {
        std::fs::create_dir_all(f).unwrap();
    }
    let (ok, text) = cli(&d.socket, &["upgrade", "gc", "adopted"]);
    assert!(ok, "{text}");
    let removed: Vec<&str> = text.lines().filter(|l| l.starts_with("removed ")).collect();
    assert_eq!(removed.len(), 3, "only the interrupted stage goes: {text}");
    assert!(!orphan.exists(), "{text}");
    assert!(
        envs.join(env_id("adopted", "1.0")).exists(),
        "live env kept: {text}"
    );
    assert!(
        envs.join(env_id("adopted", "2.0")).exists(),
        "rollback target kept: {text}"
    );
    for f in &foreign {
        assert!(f.exists(), "{} is not adopted's: {text}", f.display());
    }

    // Unwind both promotes: staged 1.0 → staged 2.0 → the original env.
    let (ok, text) = cli(&d.socket, &["upgrade", "rollback", "adopted", "--wait"]);
    assert!(ok, "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "2.0");
    let (ok, text) = cli(&d.socket, &["upgrade", "rollback", "adopted", "--wait"]);
    assert!(ok, "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "1.0");

    // A broken version fails offline validation and is never promotable.
    let (code, text) = cli_code(&d.socket, &["upgrade", "stage", "adopted", "2.0.1"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("broken on purpose"), "{text}");
    let (code, text) = cli_code(&d.socket, &["upgrade", "promote", "adopted", "2.0.1"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("never promotable"), "{text}");

    // An interrupted stage (no manifest) is not promotable either.
    let staged = dir.path().join("envs");
    let manifest = staged.join(env_id("adopted", "2.0")).join("staged.json");
    std::fs::remove_file(&manifest).unwrap();
    let (code, text) = cli_code(&d.socket, &["upgrade", "promote", "adopted", "2.0"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("never promotable"), "{text}");

    // An in-place package change in the serving env is drift.
    let out = Command::new("uv")
        .args(["pip", "install", "--python"])
        .arg(env1.join("bin/python"))
        .arg("fakeengine==2.0")
        .arg("--find-links")
        .arg(&wheelhouse)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (code, text) = cli_code(
        &d.socket,
        &["run", "adopted", "--wait", "--key", "up-drift"],
    );
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("version.packages"), "{text}");
    assert!(
        text.contains("drifted"),
        "an in-place env change is definition drift: {text}"
    );

    // Everything is rolled back or failed now: gc removes those envs, keeps
    // their history rows as `discarded`, and leaves foreign entries alone.
    let (ok, text) = cli(&d.socket, &["upgrade", "gc", "adopted"]);
    assert!(ok, "{text}");
    for v in ["1.0", "2.0", "2.0.1"] {
        let gone = !std::fs::read_dir(&envs)
            .unwrap()
            .any(|e| e.unwrap().file_name().to_string_lossy() == env_id("adopted", v));
        assert!(gone, "{v} env must be removed: {text}");
    }
    for f in &foreign {
        assert!(f.exists(), "{} is not adopted's: {text}", f.display());
    }
    let (ok, rows) = cli(&d.socket, &["upgrade", "list", "adopted"]);
    assert!(ok, "{rows}");
    let discarded = rows.lines().filter(|l| l.contains(" discarded ")).count();
    let total = rows.lines().filter(|l| l.contains(" → ")).count();
    assert!(discarded > 0 && discarded == total, "{rows}");
    // A second gc has nothing left to do.
    let (ok, text) = cli(&d.socket, &["upgrade", "gc", "adopted"]);
    assert!(ok, "{text}");
    assert!(!text.contains("removed "), "{text}");

    // Hand back after several definition changes: release restores what
    // `own` replaced, byte for byte — not an llmctl render from a promote
    // or rollback (pilot window 2).
    let (code, text) = cli_code(&d.socket, &["release", "adopted"]);
    assert_eq!(code, Some(0), "{text}");
    assert_eq!(
        std::fs::read(&unit.file).unwrap(),
        original_unit,
        "release must restore the original unit file"
    );
}

/// Rows of `upgrade list` output with this status and target version.
fn upgrade_rows(list: &str, status: &str, to_version: &str) -> usize {
    list.lines()
        .filter(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            // date time status from → to env op=
            f.len() >= 6 && f[2] == status && f[5] == to_version
        })
        .count()
}

/// Add a fakeengine recipe to the adopted workload in inventory.json.
fn patch_inventory_recipe(dir: &Path, wheelhouse: &Path) {
    let mut inv: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("inventory.json")).unwrap()).unwrap();
    inv["inventory"]["workloads"][0]["recipe"] = serde_json::json!({
        "name": "fakeengine",
        "packages": ["fakeengine"],
        "apply": "exec",
        "find_links": wheelhouse.display().to_string(),
        "model_config": dir.join("model-config.json").display().to_string(),
    });
    std::fs::write(dir.join("inventory.json"), inv.to_string()).unwrap();
}

/// The daemon's env dir name for a workload and version (upgrade.rs).
fn env_id(workload: &str, version: &str) -> String {
    use sha2::{Digest as _, Sha256};
    let hex = format!(
        "{:x}",
        Sha256::digest(format!("{workload}:{version}").as_bytes())
    );
    format!("{workload}-{version}-{}", &hex[..8])
}

// ---- M4a: installs and versions per recipe kind ------------------------------

/// A tiny python "engine binary": serves /v1/models and /v1/version, and
/// answers `--version` on the command line. The version is baked in per
/// archive, exactly like a released build.
const TINYSERVE_PY: &str = r#"#!/usr/bin/env python3
import json, sys
VERSION = "{VERSION}"
if len(sys.argv) > 1 and sys.argv[1] == "--version":
    print(f"tinyserve {VERSION}")
    sys.exit(0)
from http.server import BaseHTTPRequestHandler, HTTPServer
port, model = int(sys.argv[1]), sys.argv[2]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/version":
            body = json.dumps({"version": VERSION}).encode()
        elif self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": model}]}).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
"#;

/// Build a tiny release archive for `version` into `root`: one wrapper
/// directory (the recipes' default strip) containing `bin/tinyserve`.
/// Returns (archive_path, sha256_hex).
fn make_tiny_archive(root: &Path, version: &str) -> (PathBuf, String) {
    let staging = root.join(format!("mk-{version}"));
    let bin = staging.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let script = staging.join("bin").join("tinyserve");
    std::fs::write(&script, TINYSERVE_PY.replace("{VERSION}", version)).unwrap();
    make_executable(&script);
    let archive = root.join(format!("tinyserve-{version}.tar.gz"));
    let out = Command::new("tar")
        .args(["-czf"])
        .arg(&archive)
        .arg("-C")
        .arg(root)
        .arg(format!("mk-{version}"))
        .output()
        .expect("tar czf");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(&staging);
    use sha2::{Digest as _, Sha256};
    let digest = Sha256::digest(std::fs::read(&archive).unwrap());
    (archive, format!("{:x}", digest))
}

fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut perm = std::fs::metadata(path).unwrap().permissions();
    perm.set_mode(0o755);
    std::fs::set_permissions(path, perm).unwrap();
}

/// A tiny C "engine binary" (the plan's disposable source-build spike):
/// /v1/models, /v1/version and `--version`, version baked per commit.
const TINYSERVE_C: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#define VER "{VERSION}"
int main(int argc, char **argv) {{
    if (argc > 1 && strcmp(argv[1], "--version") == 0) {{
        printf("tinyserve-c %s\n", VER);
        return 0;
    }}
    int port = atoi(argv[1]);
    const char *model = argv[2];
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return 1;
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons((unsigned short)port);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(s, (struct sockaddr *)&a, sizeof a) < 0) return 1;
    if (listen(s, 8) < 0) return 1;
    for (;;) {{
        int c = accept(s, 0, 0);
        if (c < 0) continue;
        char buf[2048];
        memset(buf, 0, sizeof buf);
        ssize_t n = read(c, buf, sizeof buf - 1);
        (void)n;
        char body[512];
        int len;
        if (strncmp(buf, "GET /v1/version", 15) == 0) {{
            len = snprintf(body, sizeof body, "{{\"version\":\"%s\"}}", VER);
        }} else {{
            len = snprintf(body, sizeof body,
                "{{\"object\":\"list\",\"data\":[{{\"id\":\"%s\"}}]}}", model);
        }}
        dprintf(c,
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
            "content-length: %d\r\nconnection: close\r\n\r\n%s", len, body);
        close(c);
    }}
}}
"#;

/// A git repo with two commits, each a tiny C server printing its own
/// version, tagged `v1` and `v2`.
fn make_tiny_git_repo(root: &Path) -> PathBuf {
    let repo = root.join("tinysrc");
    std::fs::create_dir_all(&repo).unwrap();
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .arg("-C")
            .arg(&repo)
            .args(args)
            .env("GIT_AUTHOR_NAME", "spike")
            .env("GIT_AUTHOR_EMAIL", "spike@example.invalid")
            .env("GIT_COMMITTER_NAME", "spike")
            .env("GIT_COMMITTER_EMAIL", "spike@example.invalid")
            .output()
            .expect("git");
        assert!(
            out.status.success(),
            "git {:?}: {}{}",
            args,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    };
    git(&["init", "-q", "-b", "main"]);
    for (tag, version) in [("1.0.0", "1.0.0"), ("2.0.0", "2.0.0")] {
        // Un-double the brace escapes (the raw C template keeps them
        // literal for format!-free templating).
        std::fs::write(
            repo.join("main.c"),
            TINYSERVE_C
                .replace("{VERSION}", version)
                .replace("{{", "{")
                .replace("}}", "}"),
        )
        .unwrap();
        git(&["add", "main.c"]);
        git(&["commit", "-q", "-m", &format!("tinyserve {version}")]);
        git(&["tag", tag]);
    }
    repo
}

/// Patch the adopted workload's recipe in inventory.json (any kind).
fn patch_inventory_recipe_json(dir: &Path, recipe: serde_json::Value) {
    let mut inv: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("inventory.json")).unwrap()).unwrap();
    inv["inventory"]["workloads"][0]["recipe"] = recipe;
    std::fs::write(dir.join("inventory.json"), inv.to_string()).unwrap();
}

/// M4a, release-archive kind: stage (fetch a local archive, checksum,
/// unpack, binary check), promote through the admitted operation, verify
/// the served version, roll back to N, refuse an interrupted stage, refuse
/// a checksum mismatch, gc the unused dir.
#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn release_archive_stages_promotes_verifies_and_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let builds = dir.path().join("builds");
    std::fs::create_dir_all(&builds).unwrap();
    let (archive1, _sha1) = make_tiny_archive(dir.path(), "1.0");
    let (archive2, _sha2) = make_tiny_archive(dir.path(), "2.0");
    let port = free_port();
    let model = "spike/model";

    // A pre-existing versioned install the unit starts from (the dir the
    // recipe's root owns, id-named like a stage would name it).
    let seed = builds.join(env_id("adopted", "1.0"));
    std::fs::create_dir_all(seed.join("bin")).unwrap();
    // Unpack archive1's payload into the seed dir via the same tar rules.
    let out = Command::new("tar")
        .args(["-xzf"])
        .arg(&archive1)
        .arg("-C")
        .arg(&seed)
        .arg("--strip-components")
        .arg("1")
        .output()
        .unwrap();
    assert!(out.status.success());

    let unit = SpikeUnit::create(
        dir.path(),
        &format!(
            "{} {} {model}",
            seed.join("bin").join("tinyserve").display(),
            port
        ),
    );
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &unit.name, port);
    let (ok, text) = cli(&d.socket, &["own", "adopted"]);
    assert!(ok, "{text}");

    // The owner adds a release-archive recipe; the restart re-pins the
    // version facts (schema 3: dir, binary digest, reported version).
    drop(d);
    patch_inventory_recipe_json(
        dir.path(),
        serde_json::json!({
            "kind": "release_archive",
            "name": "tinyserve",
            "root": builds.display().to_string(),
            "url": archive2.display().to_string(),
            "binary": "bin/tinyserve",
            "version_arg": "--version"
        }),
    );
    let d = spawn_daemon(dir.path());
    let inv = std::fs::read_to_string(dir.path().join("inventory.json")).unwrap();
    assert!(
        inv.contains("\"definition_schema\": 3") && inv.contains("release_archive"),
        "re-pin must record the per-kind fact: {inv}"
    );

    // Stage 2.0: fetch (copy), unpack, binary check → validated.
    let (ok, text) = cli(&d.socket, &["install", "stage", "adopted", "2.0"]);
    assert!(ok, "{text}");
    assert!(text.contains("validated"), "{text}");
    let dir2 = builds.join(env_id("adopted", "2.0"));
    assert!(dir2.join("staged.json").exists(), "{text}");
    assert!(dir2.join("bin").join("tinyserve").exists(), "{text}");

    // Promote: stop, swap the definition's path to the new dir, start,
    // verify served identity and the binary's own version.
    let (ok, text) = cli(
        &d.socket,
        &["upgrade", "promote", "adopted", "2.0", "--wait"],
    );
    assert!(ok, "{text}");
    assert!(text.contains("verify adopted runs"), "{text}");
    assert_eq!(
        http_get_json(port, "/v1/version")["version"],
        "2.0",
        "the promoted dir must be serving"
    );
    assert!(
        std::fs::read_to_string(&unit.file)
            .unwrap()
            .contains(dir2.display().to_string().as_str()),
        "the unit file must point at the new version dir"
    );

    // Roll back to N: the seed dir serves again.
    let (ok, text) = cli(&d.socket, &["upgrade", "rollback", "adopted", "--wait"]);
    assert!(ok, "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "1.0");

    // An interrupted stage (manifest removed) is never promotable.
    std::fs::remove_file(dir2.join("staged.json")).unwrap();
    let (code, text) = cli_code(&d.socket, &["upgrade", "promote", "adopted", "2.0"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("never promotable"), "{text}");

    // A declared checksum that does not match refuses the stage.
    drop(d);
    patch_inventory_recipe_json(
        dir.path(),
        serde_json::json!({
            "kind": "release_archive",
            "name": "tinyserve",
            "root": builds.display().to_string(),
            "url": dir.path().join("tinyserve-3.0.tar.gz").display().to_string(),
            "checksum": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "binary": "bin/tinyserve",
            "version_arg": "--version"
        }),
    );
    let (archive3, _sha3) = make_tiny_archive(dir.path(), "3.0");
    let d = spawn_daemon(dir.path());
    let (code, text) = cli_code(&d.socket, &["install", "stage", "adopted", "3.0"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("checksum mismatch"), "{text}");
    assert!(
        !builds.join(env_id("adopted", "3.0")).exists(),
        "a failed stage promotes nothing: {text}"
    );
    let _ = archive3;

    // gc removes the rolled-back 2.0 dir, keeps the live seed dir.
    let (ok, text) = cli(&d.socket, &["upgrade", "gc", "adopted"]);
    assert!(ok, "{text}");
    assert!(!dir2.exists(), "rolled-back dir must go: {text}");
    assert!(
        builds.join(env_id("adopted", "1.0")).exists(),
        "the live dir stays: {text}"
    );
}

/// M4a, source-build kind: a tiny C program built from a git checkout —
/// stage two tags, promote, verify the served version, roll back. The
/// binary's own `--version` is the verify step.
#[test]
#[ignore = "touches the user's systemd and builds C: run with --ignored"]
fn source_build_stages_promotes_and_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let builds = dir.path().join("cbuilds");
    std::fs::create_dir_all(&builds).unwrap();
    let repo = make_tiny_git_repo(dir.path());
    let port = free_port();
    let model = "spike/model";

    // Bootstrap: the unit initially runs the v1 binary built by hand into
    // a versioned dir (as an adopted pre-llmctl install would be).
    let seed = builds.join(env_id("adopted", "1.0.0"));
    std::fs::create_dir_all(seed.join("bin")).unwrap();
    let out = Command::new("cc")
        .arg(dir.path().join(&repo).join("main.c"))
        .arg("-o")
        .arg(seed.join("bin").join("tinyserve"))
        .output()
        .expect("cc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // v1 is the current HEAD~1 source; rebuild from the tag explicitly.
    let out = Command::new("git")
        .args(["-C", repo.to_str().unwrap(), "show", "1.0.0:main.c"])
        .output()
        .unwrap();
    std::fs::write(dir.path().join("main-v1.c"), out.stdout).unwrap();
    let out = Command::new("cc")
        .arg(dir.path().join("main-v1.c"))
        .arg("-o")
        .arg(seed.join("bin").join("tinyserve"))
        .output()
        .expect("cc");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let unit = SpikeUnit::create(
        dir.path(),
        &format!(
            "{} {} {model}",
            seed.join("bin").join("tinyserve").display(),
            port
        ),
    );
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &unit.name, port);
    let (ok, text) = cli(&d.socket, &["own", "adopted"]);
    assert!(ok, "{text}");
    drop(d);

    patch_inventory_recipe_json(
        dir.path(),
        serde_json::json!({
            "kind": "source_build",
            "name": "tinyserve-c",
            "root": builds.display().to_string(),
            "git": repo.display().to_string(),
            "build": "mkdir -p bin && cc main.c -o bin/tinyserve",
            "binary": "bin/tinyserve",
            "version_arg": "--version"
        }),
    );
    let d = spawn_daemon(dir.path());
    let inv = std::fs::read_to_string(dir.path().join("inventory.json")).unwrap();
    assert!(
        inv.contains("\"definition_schema\": 3") && inv.contains("source_build"),
        "re-pin must record the per-kind fact: {inv}"
    );

    // Stage v2: clone, checkout, build, binary check.
    let (ok, text) = cli(&d.socket, &["install", "stage", "adopted", "2.0.0"]);
    assert!(ok, "{text}");
    assert!(text.contains("validated"), "{text}");
    let dir2 = builds.join(env_id("adopted", "2.0.0"));
    assert!(dir2.join("staged.json").exists(), "{text}");

    // The manifest records the commit — the identity for SHA-pinned builds.
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir2.join("staged.json")).unwrap(),
    )
    .unwrap();
    assert!(
        manifest["commit"].as_str().is_some_and(|c| c.len() == 40),
        "the git SHA is the version identity: {manifest}"
    );

    let (ok, text) = cli(
        &d.socket,
        &["upgrade", "promote", "adopted", "2.0.0", "--wait"],
    );
    assert!(ok, "{text}");
    assert_eq!(
        http_get_json(port, "/v1/version")["version"],
        "2.0.0",
        "the promoted build must be serving"
    );

    let (ok, text) = cli(&d.socket, &["upgrade", "rollback", "adopted", "--wait"]);
    assert!(ok, "{text}");
    assert_eq!(http_get_json(port, "/v1/version")["version"], "1.0.0");
}

/// M4a, vendor-app kind: the running version is observed through the
/// declared probe, pinned at startup, drift is a finding (never
/// auto-corrected), the plan prints owner-run steps only, and stage
/// refuses.
#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn vendor_app_version_is_observed_pinned_and_drifts() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let model = "spike/model";
    // The "vendor app": a server whose /api/version reads a file the test
    // can flip — exactly what a silent vendor auto-update looks like.
    let version_file = dir.path().join("vendor-version");
    std::fs::write(&version_file, "0.1\n").unwrap();
    let script = dir.path().join("vendor-server.py");
    std::fs::write(
        &script,
        format!(
            r#"
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, model, vfile = int(sys.argv[1]), sys.argv[2], sys.argv[3]
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/version":
            body = json.dumps({{"version": open(vfile).read().strip()}}).encode()
        elif self.path == "/v1/models":
            body = json.dumps({{"object": "list", "data": [{{"id": model}}]}}).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
"#
        ),
    )
    .unwrap();

    let unit = SpikeUnit::create(
        dir.path(),
        &format!(
            "/usr/bin/python3 {} {port} {model} {}",
            script.display(),
            version_file.display()
        ),
    );
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &unit.name, port);
    let (ok, text) = cli(&d.socket, &["own", "adopted"]);
    assert!(ok, "{text}");
    drop(d);

    patch_inventory_recipe_json(
        dir.path(),
        serde_json::json!({
            "kind": "vendor_app",
            "name": "fake-vendor",
            "probe": {"probe": "http_json", "path": "/api/version", "field": "version"},
            "backup_path": "/var/backups/fake-vendor.tar.gz",
            "reinstall_cmd": "vendor-installer --restore /var/backups/fake-vendor.tar.gz"
        }),
    );
    let d = spawn_daemon(dir.path());
    let inv = std::fs::read_to_string(dir.path().join("inventory.json")).unwrap();
    assert!(
        inv.contains("\"definition_schema\": 3") && inv.contains("vendor_app"),
        "the running version is pinned at startup: {inv}"
    );

    // No drift while the app reports the pinned version.
    let (code, text) = cli_code(&d.socket, &["version", "adopted"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(
        text.contains("\"running\": \"0.1\"") && text.contains("0.1"),
        "{text}"
    );

    // A stopped vendor server reports no version. That is not drift: the
    // start guard runs the same comparison, and it must let the app start.
    let (code, text) = cli_code(&d.socket, &["stop", "adopted", "--wait"]);
    assert_eq!(code, Some(0), "{text}");
    let (code, text) = cli_code(&d.socket, &["version", "adopted"]);
    assert_eq!(code, Some(0), "a stopped server is not drift: {text}");
    assert!(text.contains("\"running\": null"), "{text}");
    let (code, text) = cli_code(&d.socket, &["run", "adopted", "--wait"]);
    assert_eq!(code, Some(0), "the pinned app starts again: {text}");

    // The vendor "auto-updates": the running version moves off the pin.
    std::fs::write(&version_file, "0.2\n").unwrap();
    let (code, text) = cli_code(&d.socket, &["version", "adopted"]);
    assert_eq!(code, Some(4), "drift must be a finding: {text}");
    assert!(text.contains("version.fact"), "{text}");

    // The plan reports both, the drift, and the owner-run steps — and
    // nothing is ever executed.
    let (code, text) = cli_code(&d.socket, &["install", "plan", "adopted"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("\"running\": \"0.2\""), "{text}");
    assert!(text.contains("/var/backups/fake-vendor.tar.gz"), "{text}");
    assert!(text.contains("vendor-installer --restore"), "{text}");

    // Stage refuses: vendor apps are never installed by llmctl.
    let (code, text) = cli_code(&d.socket, &["install", "stage", "adopted", "0.3"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("vendor app"), "{text}");
}

/// `forget` never leaves a managed unit running unwatched: it is refused
/// until the unit is stopped. After that the record goes and the unit
/// file stays where it is.
#[test]
#[ignore = "touches the user's systemd: run with --ignored"]
fn forget_refuses_a_running_managed_unit_then_removes_the_record() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let unit = SpikeUnit::create(dir.path(), &serving_exec(dir.path(), port, "spike/model"));
    unit.start();
    write_empty_inventory(dir.path());
    let d = spawn_daemon(dir.path());
    adopt_running(&d.socket, &unit.name, port);

    let (code, text) = cli_code(&d.socket, &["forget", "adopted", "--yes"]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("running and managed") && text.contains("stop"), "{text}");

    let (code, text) = cli_code(&d.socket, &["stop", "adopted", "--wait"]);
    assert_eq!(code, Some(0), "{text}");
    let (code, text) = cli_code(&d.socket, &["forget", "adopted", "--yes"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("forgot adopted"), "{text}");
    let (_, text) = cli_code(&d.socket, &["status"]);
    assert!(!text.contains("adopted"), "{text}");
    assert!(unit.file.exists(), "the unit file is not llmctl-next's to delete");
}
