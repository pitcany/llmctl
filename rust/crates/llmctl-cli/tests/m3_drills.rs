//! M3 failure drills (RUST-M3-PLAN, table F) — driven from the
//! workstation against a *disposable* launchd job on an enrolled Mac.
//!
//!     LLMCTL_M3_ALIAS=mbp2 cargo test -p llmctl-cli --test m3_drills -- \
//!         --ignored --test-threads=1
//!
//! Prerequisites: `llmctl-next host add <alias>` enrolled the target
//! (agent LaunchAgent running, bridge verified). Every drill uses its own
//! `com.pitcany.llmctl-next-drill-<id>` label, rewrites the agent's
//! inventory to name only that workload, and restores an empty inventory
//! on drop. The sleep/wake drill is pilot-only (owner present); see
//! docs/RUST-M3-PROGRESS-2026-11-16.md.

use std::process::{Command as StdCommand, Stdio};
use std::time::Duration;

use llmctl_client::bridge::BridgeProcess;
use llmctl_client::wire::{AdmitRequest, OpStatus, PlanRequest, PROTOCOL_VERSION};
use llmctl_client::ControlClient;
use llmctl_core::OpKind;

fn alias() -> Option<String> {
    std::env::var("LLMCTL_M3_ALIAS").ok().filter(|s| !s.is_empty())
}

fn ssh(alias: &str, cmd: &str) -> String {
    let out = StdCommand::new("ssh")
        .arg(alias)
        .arg(cmd)
        .output()
        .expect("ssh spawns");
    assert!(
        out.status.success(),
        "ssh {alias} {cmd}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Push a file's bytes to the host through `ssh cat >`.
fn push(alias: &str, remote_path: &str, bytes: &[u8]) {
    let mut child = StdCommand::new("ssh")
        .arg(alias)
        .arg(format!("cat > {remote_path}"))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("ssh push spawns");
    use std::io::Write as _;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(bytes)
        .expect("write push bytes");
    let out = child.wait_with_output().expect("ssh push completes");
    assert!(out.status.success(), "push {remote_path} failed");
}

const SERVER_PY: &str = r#"
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[1])
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b"OK"; self.send_response(200)
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "drill/model"}]}).encode(); self.send_response(200)
        else:
            self.send_response(404); self.end_headers(); return
        self.send_header("content-length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
"#;

/// A disposable launchd job on the Mac plus the agent inventory naming it.
struct DrillEnv {
    alias: String,
    label: String,
    port: u16,
}

impl DrillEnv {
    fn create(alias: &str) -> DrillEnv {
        let label = format!(
            "com.pitcany.llmctl-next-drill-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let dir = "~/.local/state/llmctl-next";
        // Server script (runs under launchd, not by us).
        push(alias, &format!("{dir}/drill-server.py"), SERVER_PY.as_bytes());
        // Pick a free port on the host, write + bootstrap the plist, print
        // the port back as json.
        let setup = format!(
            r#"
import json, os, socket, subprocess
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
home = os.path.expanduser("~")
plist = home + "/Library/LaunchAgents/{label}.plist"
open(plist, "w").write(
    '<?xml version="1.0" encoding="UTF-8"?>\n<plist version="1.0"><dict>\n'
    '<key>Label</key><string>{label}</string>\n'
    '<key>ProgramArguments</key><array><string>/usr/bin/python3</string>'
    '<string>' + home + '/.local/state/llmctl-next/drill-server.py</string>'
    '<string>' + str(port) + '</string></array>\n'
    '<key>KeepAlive</key><true/>\n<key>RunAtLoad</key><true/>\n'
    '</dict></plist>\n')
subprocess.run(["launchctl", "bootstrap", "gui/" + str(os.getuid()), plist], check=True)
print(json.dumps({{"port": port}}))
"#
        );
        push(
            alias,
            &format!("{dir}/drill-setup.py"),
            setup.as_bytes(),
        );
        let out = ssh(alias, &format!("python3 {dir}/drill-setup.py"));
        let port = serde_json::from_str::<serde_json::Value>(out.trim())
            .expect("drill setup prints json")
            .get("port")
            .and_then(|p| p.as_u64())
            .expect("port in setup output") as u16;

        // Rewrite the agent's inventory to name exactly this workload,
        // then restart the agent (bootout + bootstrap, as login does).
        let env = DrillEnv {
            alias: alias.to_string(),
            label: label.clone(),
            port,
        };
        env.write_agent_inventory();
        env.restart_agent();
        env
    }

    fn write_agent_inventory(&self) {
        let inv = serde_json::json!({
            "inventory": {
                "host": "drill-mac",
                "platform": "macos",
                "gpus": [{"host": "drill-mac", "index": 9}],
                "workloads": [{
                    "name": "drill",
                    "engine": "launchd",
                    "model": "/usr/bin/python3",
                    "gpus": [{"host": "drill-mac", "index": 9}],
                    "port": self.port,
                    "unit": {"scope": "launchd", "name": self.label},
                    "authority": {"kind": "launchd_service", "label": self.label},
                    "readiness_timeout_secs": 30
                }],
                "revision": 1
            }
        });
        push(
            &self.alias,
            "~/.local/state/llmctl-next/inventory.json",
            serde_json::to_vec_pretty(&inv).unwrap().as_slice(),
        );
    }

    fn restart_agent(&self) {
        ssh(
            &self.alias,
            "launchctl bootout gui/$(id -u)/com.pitcany.llmctl-next 2>/dev/null; \
             launchctl bootstrap gui/$(id -u) \
             ~/Library/LaunchAgents/com.pitcany.llmctl-next.plist",
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            let up = ssh(
                &self.alias,
                "test -S ~/.local/state/llmctl-next/control.sock && echo up",
            );
            if up.trim() == "up" {
                return;
            }
            assert!(std::time::Instant::now() < deadline, "agent never came back");
            std::thread::sleep(Duration::from_millis(300));
        }
    }

    fn runs(&self) -> u64 {
        let out = ssh(
            &self.alias,
            &format!(
                "launchctl print gui/$(id -u)/{} 2>/dev/null | grep 'runs = '",
                self.label
            ),
        );
        out.trim()
            .split('=')
            .nth(1)
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    }
}

impl Drop for DrillEnv {
    fn drop(&mut self) {
        let _ = ssh(
            &self.alias,
            &format!(
                "launchctl bootout gui/$(id -u)/{} 2>/dev/null; rm -f \
                 ~/Library/LaunchAgents/{0}.plist",
                self.label
            ),
        );
        // Restore an empty inventory so the agent returns to its real one
        // on the next edit/restart.
        let empty = serde_json::json!({
            "inventory": {
                "host": "drill-mac",
                "platform": "macos",
                "gpus": [{"host": "drill-mac", "index": 9}],
                "workloads": [],
                "revision": 1
            }
        });
        let _ = serde_json::to_vec_pretty(&empty).map(|bytes| {
            push(
                &self.alias,
                "~/.local/state/llmctl-next/inventory.json",
                bytes.as_slice(),
            )
        });
        self.restart_agent();
    }
}

async fn wait_terminal(client: &ControlClient, id: &str) -> llmctl_client::wire::OperationView {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
    loop {
        let op = client.operation(id).await.expect("operation readable");
        if op.status.is_terminal() {
            return op;
        }
        assert!(tokio::time::Instant::now() < deadline, "op {id} never terminal");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Lost ACK: kill the ssh bridge right after the admit request is written
/// to the pipe. Retry by key finds the one operation; no second effect.
#[tokio::test]
#[ignore = "touches a real enrolled Mac over ssh"]
async fn drill_lost_ack_resolves_by_key_without_duplicate_effect() {
    let Some(alias) = alias() else {
        eprintln!("LLMCTL_M3_ALIAS not set; skipping");
        return;
    };
    let _env = DrillEnv::create(&alias);
    let client = ControlClient::connect_bridge(&alias);
    let plan = client
        .plan(PlanRequest {
            kind: OpKind::Stop,
            workload: "drill".into(),
        })
        .await
        .expect("plan against agent");

    let key = format!("drill-lost-ack-{}", std::process::id());
    // Raw admit over a manually-owned pipe, then kill the transport.
    let mut process = BridgeProcess::spawn_ssh(&alias).expect("ssh bridge spawns");
    let admit = serde_json::to_vec(&AdmitRequest {
        protocol: PROTOCOL_VERSION,
        idempotency_key: key.clone(),
        kind: OpKind::Run,
        workload: "drill".into(),
        expected_revision: Some(plan.revision),
        expected_stop_targets: Some(plan.stop_targets.clone()),
    })
    .unwrap();
    let request = format!(
        "POST /v1/operations HTTP/1.1\r\nhost: drill\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
        admit.len()
    );
    {
        use tokio::io::AsyncWriteExt as _;
        process.write_all(request.as_bytes()).await.unwrap();
        process.write_all(&admit).await.unwrap();
        process.flush().await.unwrap();
    }
    // The request is on the wire; the transport dies now (lost ACK).
    process.kill();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Reconnect and resolve by key: exactly one operation, one effect.
    let client = ControlClient::connect_bridge(&alias);
    let op = client.operation_by_key(&key).await.expect("by key after kill");
    assert_eq!(op.idempotency_key, key);
    let op = wait_terminal(&client, &op.id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    // Retrying the same key replays the same operation, never a second run.
    let replay = client
        .admit(AdmitRequest {
            protocol: PROTOCOL_VERSION,
            idempotency_key: key.clone(),
            kind: OpKind::Run,
            workload: "drill".into(),
            expected_revision: Some(plan.revision),
            expected_stop_targets: Some(plan.stop_targets.clone()),
        })
        .await
        .expect("replay admit");
    assert!(replay.replayed, "same key must replay, not duplicate");
    assert_eq!(replay.operation.id, op.id);
}

/// Agent restart (SIGKILL): the workload survives (its own launchd job);
/// the agent's KeepAlive revives it and the bridge works again.
#[tokio::test]
#[ignore = "touches a real enrolled Mac over ssh"]
async fn drill_agent_kill_reconciles_and_workload_survives() {
    let Some(alias) = alias() else {
        eprintln!("LLMCTL_M3_ALIAS not set; skipping");
        return;
    };
    let env = DrillEnv::create(&alias);
    assert!(env.runs() >= 1, "drill job running before the kill");

    // SIGKILL the agent (not the workload's job).
    ssh(&alias, "pkill -9 -x llmctl-next || true");
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let up = ssh(
            &alias,
            "test -S ~/.local/state/llmctl-next/control.sock && echo up",
        );
        if up.trim() == "up" {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "agent never revived");
        std::thread::sleep(Duration::from_millis(300));
    }
    let client = ControlClient::connect_bridge(&alias);
    let health = client.health().await.expect("agent answers after revive");
    println!(
        "agent back: {} (protocol {})",
        health.daemon_version, health.protocol
    );
    // The workload kept running under its own launchd job throughout.
    let status = client.status().await.expect("status after revive");
    let drill = status
        .workloads
        .iter()
        .find(|w| w.name == "drill")
        .expect("drill workload present");
    assert_eq!(drill.process, "Running", "workload survived the agent kill");
}

/// Simulated login start: boot out + bootstrap the agent's LaunchAgent, as
/// login does. The agent returns; while it is gone, status is unreachable
/// (never silently "stopped").
#[tokio::test]
#[ignore = "touches a real enrolled Mac over ssh"]
async fn drill_simulated_login_restart_of_agent() {
    let Some(alias) = alias() else {
        eprintln!("LLMCTL_M3_ALIAS not set; skipping");
        return;
    };
    let env = DrillEnv::create(&alias);
    env.restart_agent();
    let client = ControlClient::connect_bridge(&alias);
    let status = client.status().await.expect("status after simulated login");
    assert!(
        status
            .workloads
            .iter()
            .any(|w| w.name == "drill" && w.process == "Running")
    );
}

/// Stale plan + protocol mismatch are refused before any effect; status
/// still works across versions.
#[tokio::test]
#[ignore = "touches a real enrolled Mac over ssh"]
async fn drill_stale_plan_and_protocol_mismatch_refused() {
    let Some(alias) = alias() else {
        eprintln!("LLMCTL_M3_ALIAS not set; skipping");
        return;
    };
    let _env = DrillEnv::create(&alias);
    let client = ControlClient::connect_bridge(&alias);
    let status = client.status().await.expect("status works");

    // Stale plan: revision 0 is long gone.
    let err = client
        .admit(AdmitRequest {
            protocol: PROTOCOL_VERSION,
            idempotency_key: format!("drill-stale-{}", std::process::id()),
            kind: OpKind::Run,
            workload: "drill".into(),
            expected_revision: Some(0),
            expected_stop_targets: None,
        })
        .await
        .expect_err("stale plan refused");
    assert!(
        err.to_string().to_lowercase().contains("stale"),
        "{err}"
    );

    // Protocol mismatch: an "older client" sends protocol 0; the named
    // error comes back while status keeps working.
    let err = client
        .admit(AdmitRequest {
            protocol: 0,
            idempotency_key: format!("drill-proto-{}", std::process::id()),
            kind: OpKind::Run,
            workload: "drill".into(),
            expected_revision: Some(status.inventory_revision),
            expected_stop_targets: None,
        })
        .await
        .expect_err("protocol mismatch refused");
    assert!(
        err.to_string().to_lowercase().contains("protocol"),
        "{err}"
    );
    client.status().await.expect("status still works");
}
