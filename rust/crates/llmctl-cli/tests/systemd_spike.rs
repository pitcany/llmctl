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
HTTPServer(("127.0.0.1", port), H).serve_forever()
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
        let name = format!(
            "llmctl-next-spike-{}.service",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let unit_dir = PathBuf::from(std::env::var("HOME").unwrap()).join(".config/systemd/user");
        std::fs::create_dir_all(&unit_dir).unwrap();
        let file = unit_dir.join(&name);
        std::fs::write(
            &file,
            format!(
                "[Unit]\nDescription=llmctl-next disposable spike ({})\n\n[Service]\nType=simple\nExecStart={exec}\n",
                dir.display()
            ),
        )
        .unwrap();
        assert!(systemctl(&["daemon-reload"]).status.success());
        SpikeUnit { name, file }
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
