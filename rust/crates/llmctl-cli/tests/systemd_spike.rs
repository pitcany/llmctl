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

/// Inventory enrolling exactly one systemd workload: the spike unit.
fn write_inventory(dir: &Path, unit: &str, port: u16, model: &str) {
    let host = HostId::new("workstation").unwrap();
    let inv = HostInventory {
        host: host.clone(),
        platform: Platform::Linux,
        gpus: vec![GpuId::new(host.clone(), 0)],
        workloads: vec![WorkloadDefinition {
            name: WorkloadName::new("spike").unwrap(),
            engine: "systemd".into(),
            model: model.into(),
            gpus: vec![GpuId::new(host, 0)],
            port,
            unit: UnitName::systemd(UnitScopeMarker::User, unit).unwrap(),
            authority: ControlAuthority::SystemdUnit {
                scope: UnitScope::User,
                unit: unit.into(),
            },
            ownership: Default::default(),
        }],
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
        "1",
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
