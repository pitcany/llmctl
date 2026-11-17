//! M3 pilot drills (RUST-M3-PLAN G.3) — the F drills against a REAL
//! adopted launchd workload on an enrolled Mac, owner present.
//!
//!     LLMCTL_M3_ALIAS=mbp2 LLMCTL_M3_PILOT_WORKLOAD=llama-swap \
//!     LLMCTL_M3_PILOT_LABEL=com.pitcany.llama-swap \
//!         cargo test -p llmctl-cli --test m3_pilot_drills -- --ignored --test-threads=1
//!
//! Unlike `m3_drills.rs`, nothing here writes the agent's inventory or
//! creates a launchd job: the workload must already be adopted (and owned)
//! through the CLI. The lost-ACK drill restarts the workload once (a brief
//! outage of every lane it serves); the others must leave its process
//! untouched, which each drill checks by launchd pid.

use std::process::Command as StdCommand;
use std::time::Duration;

use llmctl_client::bridge::BridgeProcess;
use llmctl_client::wire::{AdmitRequest, OpStatus, PlanRequest, PROTOCOL_VERSION};
use llmctl_client::ControlClient;
use llmctl_core::OpKind;

struct Pilot {
    alias: String,
    workload: String,
    label: String,
}

fn pilot() -> Option<Pilot> {
    let var = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
    Some(Pilot {
        alias: var("LLMCTL_M3_ALIAS")?,
        workload: var("LLMCTL_M3_PILOT_WORKLOAD")?,
        label: var("LLMCTL_M3_PILOT_LABEL")?,
    })
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

impl Pilot {
    /// The workload's launchd pid (None when not running).
    fn pid(&self) -> Option<u64> {
        let out = ssh(
            &self.alias,
            &format!(
                "launchctl print gui/$(id -u)/{} 2>/dev/null | grep -E '^\tpid = ' || true",
                self.label
            ),
        );
        out.trim().split('=').nth(1)?.trim().parse().ok()
    }

    async fn assert_running(&self, client: &ControlClient) {
        let status = client.status().await.expect("status");
        let w = status
            .workloads
            .iter()
            .find(|w| w.name == self.workload)
            .expect("pilot workload in the agent's inventory");
        assert_eq!(w.process, "Running", "{w:?}");
    }
}

async fn wait_agent_alive(alias: &str, timeout: Duration) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if ControlClient::connect_bridge(alias).health().await.is_ok() {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "agent on {alias} never answered again"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

async fn wait_terminal(client: &ControlClient, id: &str) -> llmctl_client::wire::OperationView {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    loop {
        let op = client.operation(id).await.expect("operation readable");
        if op.status.is_terminal() {
            return op;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "op {id} never terminal"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Lost ACK on a real restart: exactly one operation under the key, one
/// restart, and a replay of the key restarts nothing.
#[tokio::test]
#[ignore = "restarts a real workload on an enrolled Mac"]
async fn pilot_lost_ack_restart_resolves_by_key_once() {
    let Some(p) = pilot() else {
        eprintln!("LLMCTL_M3_ALIAS / _PILOT_WORKLOAD / _PILOT_LABEL not set; skipping");
        return;
    };
    let before = p.pid().expect("pilot running before the drill");
    let client = ControlClient::connect_bridge(&p.alias);
    let plan = client
        .plan(PlanRequest {
            kind: OpKind::Restart,
            workload: p.workload.clone(),
        })
        .await
        .expect("plan");
    let key = format!("pilot-lost-ack-{}", std::process::id());
    let admit = AdmitRequest {
        protocol: PROTOCOL_VERSION,
        idempotency_key: key.clone(),
        kind: OpKind::Restart,
        workload: p.workload.clone(),
        expected_revision: Some(plan.revision),
        expected_stop_targets: Some(plan.stop_targets.clone()),
    };
    let body = serde_json::to_vec(&admit).unwrap();
    let mut process = BridgeProcess::spawn_ssh(&p.alias).expect("ssh bridge spawns");
    {
        use tokio::io::AsyncWriteExt as _;
        let head = format!(
            "POST /v1/operations HTTP/1.1\r\nhost: pilot\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        process.write_all(head.as_bytes()).await.unwrap();
        process.write_all(&body).await.unwrap();
        process.flush().await.unwrap();
    }
    // LLMCTL_M3_KILL_AFTER_MS gives the request time to land first, so the
    // drill can hit the "admitted, reply lost" branch (ssh startup alone
    // often outlasts an immediate kill).
    let delay = std::env::var("LLMCTL_M3_KILL_AFTER_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    tokio::time::sleep(Duration::from_millis(delay)).await;
    process.kill(); // the ACK is lost
    tokio::time::sleep(Duration::from_millis(500)).await;

    let client = ControlClient::connect_bridge(&p.alias);
    let op = match client.operation_by_key(&key).await {
        Ok(op) => {
            println!("admit had landed: operation {}", op.id);
            wait_terminal(&client, &op.id).await
        }
        Err(err) => {
            assert!(err.to_string().contains("unknown_operation"), "{err}");
            println!("admit never reached the agent; admitting for real");
            let resp = client.admit(admit.clone()).await.expect("real admit");
            assert!(!resp.replayed);
            wait_terminal(&client, &resp.operation.id).await
        }
    };
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    let after = p.pid().expect("pilot running after the restart");
    assert_ne!(after, before, "the restart replaced the process");
    println!("pid {before} -> {after}");

    let replay = client.admit(admit).await.expect("replay admit");
    assert!(replay.replayed, "same key must replay, not duplicate");
    assert_eq!(replay.operation.id, op.id);
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(p.pid(), Some(after), "a replay must not restart again");
    p.assert_running(&client).await;
}

/// SIGKILL the agent: the workload's own launchd job keeps running
/// (same pid), the agent revives and reports it Running.
#[tokio::test]
#[ignore = "kills the agent on an enrolled Mac"]
async fn pilot_agent_kill_leaves_the_workload_alone() {
    let Some(p) = pilot() else {
        eprintln!("pilot env not set; skipping");
        return;
    };
    let before = p.pid().expect("pilot running");
    ssh(&p.alias, "pkill -9 -x llmctl-next || true");
    wait_agent_alive(&p.alias, Duration::from_secs(45)).await;
    assert_eq!(
        p.pid(),
        Some(before),
        "workload untouched by the agent kill"
    );
    p.assert_running(&ControlClient::connect_bridge(&p.alias))
        .await;
}

/// Simulated login start: boot out and bootstrap the agent's LaunchAgent.
/// The workload keeps its pid; the agent comes back and sees it Running.
#[tokio::test]
#[ignore = "restarts the agent on an enrolled Mac"]
async fn pilot_simulated_login_leaves_the_workload_alone() {
    let Some(p) = pilot() else {
        eprintln!("pilot env not set; skipping");
        return;
    };
    let before = p.pid().expect("pilot running");
    ssh(
        &p.alias,
        "launchctl bootout gui/$(id -u)/com.pitcany.llmctl-next 2>/dev/null || true",
    );
    // While the agent is gone the bridge fails honestly, never "stopped".
    assert!(
        ControlClient::connect_bridge(&p.alias)
            .status()
            .await
            .is_err(),
        "status must be unreachable while the agent is booted out"
    );
    // Wait out the old process (a racing bootstrap is refused).
    let gone = std::time::Instant::now() + Duration::from_secs(20);
    while ssh(
        &p.alias,
        "pgrep -x llmctl-next >/dev/null && echo busy || echo dead",
    )
    .trim()
        == "busy"
    {
        assert!(std::time::Instant::now() < gone, "old agent never exited");
        std::thread::sleep(Duration::from_millis(300));
    }
    ssh(
        &p.alias,
        "launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pitcany.llmctl-next.plist",
    );
    wait_agent_alive(&p.alias, Duration::from_secs(45)).await;
    assert_eq!(
        p.pid(),
        Some(before),
        "workload untouched by the agent restart"
    );
    p.assert_running(&ControlClient::connect_bridge(&p.alias))
        .await;
}

/// A stale plan and an older protocol are refused before any effect on the
/// real workload; status keeps working.
#[tokio::test]
#[ignore = "talks to a real workload on an enrolled Mac"]
async fn pilot_stale_plan_and_protocol_mismatch_refused() {
    let Some(p) = pilot() else {
        eprintln!("pilot env not set; skipping");
        return;
    };
    let before = p.pid().expect("pilot running");
    let client = ControlClient::connect_bridge(&p.alias);
    let status = client.status().await.expect("status works");
    let err = client
        .admit(AdmitRequest {
            protocol: PROTOCOL_VERSION,
            idempotency_key: format!("pilot-stale-{}", std::process::id()),
            kind: OpKind::Restart,
            workload: p.workload.clone(),
            expected_revision: Some(0),
            expected_stop_targets: None,
        })
        .await
        .expect_err("stale plan refused");
    assert!(err.to_string().to_lowercase().contains("stale"), "{err}");
    let err = client
        .admit(AdmitRequest {
            protocol: 0,
            idempotency_key: format!("pilot-proto-{}", std::process::id()),
            kind: OpKind::Restart,
            workload: p.workload.clone(),
            expected_revision: Some(status.inventory_revision),
            expected_stop_targets: None,
        })
        .await
        .expect_err("protocol mismatch refused");
    assert!(err.to_string().to_lowercase().contains("protocol"), "{err}");
    client.status().await.expect("status still works");
    assert_eq!(p.pid(), Some(before), "no effect from refused admits");
}
