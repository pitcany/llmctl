//! End-to-end tests for the first implementation batch
//! (docs/RUST-SUCCESSOR-PLAN.md, "First implementation batch"):
//!
//! - a second client sees the same operation (idempotency),
//! - an interrupted client leaves the operation observable,
//! - daemon restart reconciles it honestly,
//! - every test runs against a fake engine in a temp dir — nothing touches
//!   the installed inference stack.
//!
//! Each test spawns its own `llmctl-next daemon` subprocess on an isolated
//! data directory, so the suite exercises the real socket, the real
//! journal, and real process death — not in-process mocks.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use llmctl_client::wire::{codes, AdmitRequest, OpStatus, PlanRequest, PROTOCOL_VERSION};
use llmctl_client::{ClientError, ControlClient};
use llmctl_core::OpKind;

struct DaemonGuard {
    child: Child,
    #[allow(dead_code)]
    dir: tempfile::TempDir,
    socket: PathBuf,
}

impl Drop for DaemonGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// Kills a daemon child if a test panics before reaping it, so a failed
/// restart test cannot leave an orphaned daemon running.
struct ChildGuard(Option<Child>);

impl ChildGuard {
    /// SIGKILL and reap now (the simulated crash).
    fn kill_now(&mut self) {
        if let Some(mut child) = self.0.take() {
            child.kill().unwrap();
            child.wait().unwrap();
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl DaemonGuard {
    fn data_dir(&self) -> &Path {
        self.dir.path()
    }

    fn client(&self) -> ControlClient {
        ControlClient::connect(&self.socket).expect("connect to daemon")
    }

    fn write_faults(&self, faults: serde_json::Value) {
        std::fs::write(
            self.data_dir().join("faults.json"),
            serde_json::to_vec_pretty(&faults).unwrap(),
        )
        .expect("write faults");
    }

    async fn wait_terminal(&self, id: &str) -> llmctl_client::wire::OperationView {
        let client = self.client();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            let op = client.operation(id).await.expect("operation readable");
            if op.status.is_terminal() {
                return op;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "operation {id} never terminal"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

fn spawn_daemon() -> DaemonGuard {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let bin = env!("CARGO_BIN_EXE_llmctl-next");
    let mut child = Command::new(bin)
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir.path())
        .arg("daemon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn daemon");
    // Wait for the socket to accept connections.
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        if std::os::unix::net::UnixStream::connect(&socket).is_ok() {
            return DaemonGuard { child, dir, socket };
        }
        if let Ok(Some(_)) = child.try_wait() {
            panic!("daemon exited before accepting connections");
        }
        assert!(
            std::time::Instant::now() < deadline,
            "daemon did not come up within 20s"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Wait until the socket accepts connections (daemon finished binding and
/// startup reconciliation).
async fn wait_socket(socket: &Path) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        if std::os::unix::net::UnixStream::connect(socket).is_ok() {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "daemon never bound {}",
            socket.display()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn admit_req(kind: OpKind, workload: &str, key: &str) -> AdmitRequest {
    AdmitRequest {
        protocol: PROTOCOL_VERSION,
        idempotency_key: key.into(),
        kind,
        workload: workload.into(),
        expected_revision: None,
        expected_stop_targets: None,
    }
}

async fn run_to_success(d: &DaemonGuard, workload: &str, key: &str) {
    let resp = d
        .client()
        .admit(admit_req(OpKind::Run, workload, key))
        .await
        .unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
}

// ---- batch criterion 1: a second client sees the same operation ----------

#[tokio::test]
async fn second_client_sees_the_same_operation() {
    let d = spawn_daemon();
    // Slow the effect so the operation is still live when the second client
    // asks.
    d.write_faults(serde_json::json!({"effect_delay_ms": 1200}));
    let client_a = d.client();
    let client_b = d.client();

    let first = client_a
        .admit(admit_req(OpKind::Run, "coder", "k-shared-1"))
        .await
        .expect("first admit");
    assert!(!first.replayed);

    let second = client_b
        .admit(admit_req(OpKind::Run, "coder", "k-shared-1"))
        .await
        .expect("idempotent replay");
    assert!(second.replayed, "second client must get the existing op");
    assert_eq!(first.operation.id, second.operation.id);

    // Same key, different request → refused, not silently reused.
    let conflict = client_b
        .admit(admit_req(OpKind::Run, "helper", "k-shared-1"))
        .await;
    match conflict {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::IDEMPOTENCY_CONFLICT),
        other => panic!("expected idempotency conflict, got {other:?}"),
    }

    let op = d.wait_terminal(&first.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded);
}

// ---- batch criterion 2: an interrupted client leaves the op observable ----

#[tokio::test]
async fn interrupted_client_leaves_operation_observable() {
    let d = spawn_daemon();
    // The admission is durable and the executor is running, but the
    // acknowledgement is delayed past the client's patience.
    d.write_faults(serde_json::json!({"admit_delay_ms": 2000, "effect_delay_ms": 800}));
    let client = d.client();

    // Admit with a timeout far shorter than the effect: the client gives up
    // (a "lost acknowledgement"), but the operation must live on.
    let interrupted = tokio::time::timeout(
        Duration::from_millis(300),
        client.admit(admit_req(OpKind::Run, "coder", "k-lost-ack")),
    )
    .await;
    assert!(interrupted.is_err(), "client should time out");

    // A second client finds it, watches it finish.
    let ops = d.client().list_operations().await.unwrap();
    assert_eq!(ops.operations.len(), 1, "operation must be observable");
    let id = ops.operations[0].id.clone();
    let op = d.wait_terminal(&id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");

    // Retrying the same key now returns the same, completed operation.
    let retry = d
        .client()
        .admit(admit_req(OpKind::Run, "coder", "k-lost-ack"))
        .await
        .unwrap();
    assert!(retry.replayed);
    assert_eq!(retry.operation.id, id);
}

// ---- batch criterion 3: daemon restart reconciles honestly ----------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_restart_reconciles_honestly() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let bin = env!("CARGO_BIN_EXE_llmctl-next");

    let spawn_daemon_on = |socket: &Path| {
        Command::new(bin)
            .arg("--socket")
            .arg(socket)
            .arg("--data-dir")
            .arg(dir.path())
            .arg("daemon")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn daemon")
    };

    // Round 1: slow effect, admit, SIGKILL while the effect is pending.
    std::fs::write(
        dir.path().join("faults.json"),
        serde_json::to_vec(&serde_json::json!({"effect_delay_ms": 2500})).unwrap(),
    )
    .unwrap();
    let mut child = ChildGuard(Some(spawn_daemon_on(&socket)));
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket).is_err() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(50));
    }
    let client = ControlClient::connect(&socket).unwrap();
    let admitted = client
        .admit(admit_req(OpKind::Run, "coder", "k-crash-1"))
        .await
        .unwrap();
    let id = admitted.operation.id.clone();
    // Let the executor journal the effect-pending marker, then kill hard.
    tokio::time::sleep(Duration::from_millis(600)).await;
    child.kill_now();
    std::fs::remove_file(&socket).unwrap_or(());

    // Round 2: restart on the same data dir.
    std::fs::write(dir.path().join("faults.json"), b"{}").unwrap();
    let child2 = spawn_daemon_on(&socket);
    let guard = DaemonGuard {
        child: child2,
        dir,
        socket: socket.clone(),
    };
    wait_socket(&socket).await;

    let client2 = guard.client();
    let op = client2
        .operation(&id)
        .await
        .expect("interrupted op survives restart");
    // Honest outcome: terminal, never silently succeeded, with the observed
    // world attached.
    assert!(op.status.is_terminal());
    assert_ne!(op.status, OpStatus::Succeeded);
    assert!(
        matches!(op.status, OpStatus::NeedsAttention | OpStatus::Interrupted),
        "unexpected status {:?}",
        op.status
    );
    let result = op.result.expect("reconciliation report");
    assert!(
        result["observed"].is_object(),
        "report must attach observations: {result}"
    );

    if op.status == OpStatus::NeedsAttention {
        // Unacknowledged needs_attention blocks new mutations…
        match client2
            .admit(admit_req(OpKind::Run, "helper", "k-crash-2"))
            .await
        {
            Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::BLOCKED_NEEDS_ATTENTION),
            other => panic!("expected blocked, got {other:?}"),
        }
        // …until a human acknowledges it.
        let acked = client2.acknowledge(&id, None).await.unwrap();
        assert!(acked.acknowledged);
        run_to_success(&guard, "helper", "k-crash-3").await;
    } else {
        // Auto-resolved (nothing observable happened): mutations allowed.
        run_to_success(&guard, "helper", "k-crash-2").await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restart_after_effect_completes_auto_resolves_when_only_observation_remains() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let bin = env!("CARGO_BIN_EXE_llmctl-next");

    let spawn_daemon_on = |socket: &Path| {
        Command::new(bin)
            .arg("--socket")
            .arg(socket)
            .arg("--data-dir")
            .arg(dir.path())
            .arg("daemon")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn daemon")
    };

    // StartUnit applies instantly, but readiness flips only after 8s: the
    // crash lands in the AwaitReadiness phase (non-destructive remainder).
    std::fs::write(
        dir.path().join("faults.json"),
        serde_json::to_vec(&serde_json::json!({"readiness_delay_ms": 8000})).unwrap(),
    )
    .unwrap();
    let mut child = ChildGuard(Some(spawn_daemon_on(&socket)));
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket).is_err() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(50));
    }
    let client = ControlClient::connect(&socket).unwrap();
    let admitted = client
        .admit(admit_req(OpKind::Run, "coder", "k-crash-b"))
        .await
        .unwrap();
    let id = admitted.operation.id.clone();
    // Wait until the start effect is durably applied (world.json says
    // active), then kill while readiness is still pending.
    let world_path = dir.path().join("world.json");
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let world: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&world_path).unwrap_or_default())
                .unwrap_or(serde_json::json!({}));
        let active = world["units"]["user/llmctl-next-fake-coder.service"]["active"]
            .as_bool()
            .unwrap_or(false);
        if active {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "start effect never applied"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    child.kill_now();
    std::fs::remove_file(&socket).unwrap_or(());

    std::fs::write(dir.path().join("faults.json"), b"{}").unwrap();
    let child2 = spawn_daemon_on(&socket);
    let guard = DaemonGuard {
        child: child2,
        dir,
        socket: socket.clone(),
    };
    wait_socket(&socket).await;
    let client2 = guard.client();
    let op = client2.operation(&id).await.unwrap();
    assert_eq!(
        op.status,
        OpStatus::Interrupted,
        "destructive work was done; only observation remained — interrupted, not failed"
    );
    assert!(op.acknowledged, "provably safe → auto-resolved");
    // And the world really is running coder.
    let status = client2.status().await.unwrap();
    let coder = status.workloads.iter().find(|w| w.name == "coder").unwrap();
    assert_eq!(coder.process, "Running");
}

// ---- the rest of the mutation contract ------------------------------------

#[tokio::test]
async fn protocol_mismatch_is_rejected_before_mutation() {
    let d = spawn_daemon();
    let mut req = admit_req(OpKind::Run, "coder", "k-proto");
    req.protocol = 99;
    match d.client().admit(req).await {
        Err(ClientError::Api { code, status, .. }) => {
            assert_eq!(code, codes::PROTOCOL_MISMATCH);
            assert_eq!(status, 400);
        }
        other => panic!("expected protocol mismatch, got {other:?}"),
    }
    // nothing was journaled
    let ops = d.client().list_operations().await.unwrap();
    assert!(ops.operations.is_empty());
}

#[tokio::test]
async fn stale_destructive_footprint_returns_revised_plan() {
    let d = spawn_daemon();
    let client = d.client();
    // helper (gpu1) is running; a big run would stop it.
    run_to_success(&d, "helper", "k-setup").await;
    let plan = client
        .plan(PlanRequest {
            kind: OpKind::Run,
            workload: "big".into(),
        })
        .await
        .unwrap();
    assert_eq!(plan.stop_targets, vec!["helper".to_string()]);

    // World moves before admission: helper is stopped, so the fresh plan
    // would stop nothing. The stale confirmation must be refused.
    run_to_success(&d, "coder", "k-midflight").await;
    // (coder takes gpu0; big still conflicts with nothing on the stop list
    // because helper stopped.)
    let stale = client
        .admit(AdmitRequest {
            protocol: PROTOCOL_VERSION,
            idempotency_key: "k-stale".into(),
            kind: OpKind::Run,
            workload: "big".into(),
            expected_revision: Some(plan.revision),
            expected_stop_targets: Some(plan.stop_targets.clone()),
        })
        .await;
    match stale {
        Err(ClientError::Api {
            code,
            details,
            status,
            ..
        }) => {
            assert_eq!(code, codes::STALE_PLAN);
            assert_eq!(status, 409);
            assert!(
                details.unwrap()["revised_plan"].is_object(),
                "revised plan attached"
            );
        }
        other => panic!("expected stale plan, got {other:?}"),
    }
}

#[tokio::test]
async fn overlapping_mutation_is_rejected() {
    let d = spawn_daemon();
    d.write_faults(serde_json::json!({"effect_delay_ms": 1500}));
    let client = d.client();
    let first = client
        .admit(admit_req(OpKind::Run, "coder", "k-over-1"))
        .await
        .unwrap();
    match client
        .admit(admit_req(OpKind::Run, "helper", "k-over-2"))
        .await
    {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::OVERLAPPING_OPERATION),
        other => panic!("expected overlap rejection, got {other:?}"),
    }
    let op = d.wait_terminal(&first.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded);
}

#[tokio::test]
async fn failed_run_rolls_back_and_reports_restoration_separately() {
    let d = spawn_daemon();
    run_to_success(&d, "helper", "k-rr-setup").await;

    // The first effect (stop helper) fails before applying.
    d.write_faults(serde_json::json!({
        "fail_effects": [
            {"unit": "user/llmctl-next-fake-helper.service", "kind": "stop", "mode": "before"}
        ]
    }));
    let resp = d
        .client()
        .admit(admit_req(OpKind::Run, "big", "k-rollback"))
        .await
        .unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Failed);
    let recovery = op.recovery.expect("restoration recorded");
    assert_eq!(recovery["outcome"], "restored", "{recovery}");
    assert!(op.phases.iter().any(|p| p.label.starts_with("rollback:")));

    // helper is serving again: restoration is separate from the failure.
    let status = d.client().status().await.unwrap();
    let helper = status
        .workloads
        .iter()
        .find(|w| w.name == "helper")
        .unwrap();
    assert_eq!(helper.process, "Running");
}

#[tokio::test]
async fn failed_rollback_becomes_needs_attention_and_blocks() {
    let d = spawn_daemon();
    run_to_success(&d, "helper", "k-na-setup").await;

    // Deterministic two-fault scenario: "start big" applies but reports
    // failure (the crash-after-effect window), then the rollback's restart
    // of helper fails before applying — restoration fails, needs_attention.
    d.write_faults(serde_json::json!({
        "fail_effects": [
            {"unit": "user/llmctl-next-fake-big.service", "kind": "start", "mode": "after"},
            {"unit": "user/llmctl-next-fake-helper.service", "kind": "start", "mode": "before"}
        ]
    }));
    let resp = d
        .client()
        .admit(admit_req(OpKind::Run, "big", "k-na"))
        .await
        .unwrap();
    let id = resp.operation.id.clone();

    let op = d.wait_terminal(&id).await;
    assert_eq!(op.status, OpStatus::NeedsAttention, "{op:?}");
    let recovery = op.recovery.unwrap();
    assert_eq!(recovery["outcome"], "restoration_failed");

    // Blocked until acknowledged.
    match d
        .client()
        .admit(admit_req(OpKind::Run, "coder", "k-na-2"))
        .await
    {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::BLOCKED_NEEDS_ATTENTION),
        other => panic!("expected block, got {other:?}"),
    }
    d.client().acknowledge(&id, None).await.unwrap();
    run_to_success(&d, "coder", "k-na-3").await;
}

#[tokio::test]
async fn cancellation_between_phases_is_recorded() {
    let d = spawn_daemon();
    d.write_faults(serde_json::json!({"effect_delay_ms": 900, "readiness_delay_ms": 3000}));
    let client = d.client();
    let resp = client
        .admit(admit_req(OpKind::Run, "coder", "k-cancel"))
        .await
        .unwrap();
    let id = resp.operation.id.clone();
    tokio::time::sleep(Duration::from_millis(150)).await;
    let canceled = client.cancel(&id, Some("test".into())).await.unwrap();
    assert_eq!(canceled.status, OpStatus::CancelRequested);
    let op = d.wait_terminal(&id).await;
    assert_eq!(op.status, OpStatus::Canceled, "{op:?}");
    assert!(op
        .phases
        .iter()
        .any(|p| p.status == llmctl_client::wire::PhaseStatus::Skipped));
}

#[tokio::test]
async fn stop_records_manual_intent_and_status_shows_it() {
    let d = spawn_daemon();
    run_to_success(&d, "coder", "k-stop-setup").await;
    let resp = d
        .client()
        .admit(admit_req(OpKind::Stop, "coder", "k-stop"))
        .await
        .unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded);
    let status = d.client().status().await.unwrap();
    let coder = status.workloads.iter().find(|w| w.name == "coder").unwrap();
    assert!(coder.intent.contains("stopped"), "intent: {}", coder.intent);
    assert!(
        coder.intent.contains("manual"),
        "manual stop persists: {}",
        coder.intent
    );
    assert_eq!(coder.process, "Stopped");
}

// ---- CLI surface -----------------------------------------------------------

fn cli(socket: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(socket)
        .args(args)
        .output()
        .expect("run llmctl-next")
}

#[test]
fn cli_status_and_dry_run_and_exit_codes() {
    let d = spawn_daemon();
    let out = cli(&d.socket, &["status"]);
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("coder"), "{text}");
    assert!(text.contains("fake/coder"), "{text}");

    let out = cli(&d.socket, &["run", "big", "--dry-run"]);
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("plan: run big"), "{text}");
    assert!(text.contains("rollback"), "{text}");

    // Unknown workload → distinct exit code.
    let out = cli(&d.socket, &["run", "nope", "--dry-run"]);
    assert_eq!(out.status.code(), Some(5));

    // No daemon on an absent socket → exit 3, and the message says how to
    // start one (no silent fallback executor).
    let absent = tempfile::tempdir().unwrap();
    let out = cli(&absent.path().join("control.sock"), &["status"]);
    assert_eq!(out.status.code(), Some(3));
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(err.contains("llmctl-next daemon"), "{err}");
}

#[test]
fn cli_run_wait_reports_success_and_json_status() {
    let d = spawn_daemon();
    let out = cli(&d.socket, &["run", "coder", "--wait", "--key", "k-cli-run"]);
    assert!(
        out.status.success(),
        "stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("succeeded"), "{text}");

    let out = cli(&d.socket, &["status", "--json"]);
    assert!(out.status.success());
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let coder = json["workloads"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["name"] == "coder")
        .unwrap();
    assert_eq!(coder["process"], "Running");
    assert_eq!(coder["readiness"], "Ready");
}
