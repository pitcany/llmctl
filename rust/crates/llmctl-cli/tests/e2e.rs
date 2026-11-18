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

// ---- M3: remote transport primitives ----------------------------------------

/// The idempotency-key lookup closes the lost-ACK window: a client that
/// lost the admit response finds the operation by key, and an unknown key
/// is a clean 404 (the admit never landed — safe to retry).
#[tokio::test]
async fn operations_resolve_by_idempotency_key() {
    let d = spawn_daemon();
    let client = d.client();
    run_to_success(&d, "coder", "lost-ack-1").await;

    let found = client.operation_by_key("lost-ack-1").await.expect("by key");
    assert_eq!(found.idempotency_key, "lost-ack-1");
    assert_eq!(found.status, OpStatus::Succeeded);

    let err = client.operation_by_key("never-admitted").await.unwrap_err();
    match &err {
        ClientError::Api { code, .. } => assert_eq!(code, codes::UNKNOWN_OPERATION),
        other => panic!("expected UNKNOWN_OPERATION, got {other:?}"),
    }
}

/// The agent-side `bridge` subcommand relays its stdin/stdout to the
/// control socket verbatim: a raw HTTP request through the pipe gets the
/// daemon's real answer (the ssh half of the transport is exercised on
/// the real Mac in the M3 drills).
#[tokio::test]
async fn bridge_subcommand_relays_http_to_the_socket() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let request = "GET /v1/health HTTP/1.1\r\nhost: bridge-test\r\nconnection: close\r\n\r\n";

    // Under the full parallel suite a daemon can hit its inherited 256-fd
    // soft limit (macOS default) and then closes new connections without
    // answering — sticky for that daemon, so each retry gets a fresh one.
    let mut last = String::new();
    for attempt in 0..3 {
        let d = spawn_daemon();
        // Warm the accept loop with one full roundtrip.
        let health = d.client().health().await.expect("daemon answers http");
        let started = std::time::Instant::now();
        let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
            .arg("--socket")
            .arg(&d.socket)
            .arg("bridge")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn bridge");
        let mut stdin = child.stdin.take().unwrap();
        let mut stdout = child.stdout.take().unwrap();
        stdin.write_all(request.as_bytes()).await.unwrap();
        stdin.flush().await.unwrap();
        drop(stdin); // EOF: the relay half-closes and drains the answer

        let mut response = String::new();
        let read = tokio::time::timeout(
            Duration::from_secs(10),
            stdout.read_to_string(&mut response),
        )
        .await
        .expect("relay drain must not hang");
        read.unwrap();
        let status = child.wait().await.unwrap();
        if response.starts_with("HTTP/1.1 200") {
            assert!(response.contains("\"host\":\"workstation\""), "{response}");
            return;
        }
        eprintln!(
            "bridge attempt {attempt} on fresh daemon in {:?} (was healthy: {}): \
             status={status} response={response:?}",
            started.elapsed(),
            health.daemon_version
        );
        // Split child vs daemon: same request over a raw socket, no child.
        let raw = raw_http_roundtrip(&d.socket, request).await;
        eprintln!("raw socket roundtrip on the same daemon: {raw:?}");
        last = response;
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    panic!("bridge relay never answered on three fresh daemons: {last:?}");
}

/// One raw HTTP/1.1 exchange over a Unix socket, no client library, no
/// child process — returns the daemon's raw bytes (diagnostics for the
/// bridge test).
async fn raw_http_roundtrip(socket: &Path, request: &str) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = match tokio::net::UnixStream::connect(socket).await {
        Ok(s) => s,
        Err(e) => return format!("connect failed: {e}"),
    };
    if let Err(e) = stream.write_all(request.as_bytes()).await {
        return format!("write failed: {e}");
    }
    let _ = stream.shutdown().await;
    let mut out = String::new();
    match stream.read_to_string(&mut out).await {
        Ok(_) => out,
        Err(e) => format!("read failed: {e}"),
    }
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

#[test]
fn cli_restart_refuses_stopped_then_restarts_running_each_time() {
    let d = spawn_daemon();
    // Stopped workload: restart is refused (409 invalid_state → exit 4),
    // nothing is admitted.
    let out = cli(&d.socket, &["restart", "coder", "--wait"]);
    assert_eq!(out.status.code(), Some(4));
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(err.contains("not running"), "{err}");

    let out = cli(&d.socket, &["run", "coder", "--wait"]);
    assert!(out.status.success());

    let out = cli(&d.socket, &["restart", "coder", "--dry-run"]);
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("plan: restart coder"), "{text}");

    // Two restarts at the same revision are two operations, not a replay.
    let mut ids = Vec::new();
    for _ in 0..2 {
        let out = cli(&d.socket, &["restart", "coder", "--wait"]);
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(out.status.success(), "{text}");
        assert!(!text.contains("replaying"), "{text}");
        assert!(text.contains("succeeded"), "{text}");
        ids.push(
            text.lines()
                .find_map(|l| l.strip_prefix("admitted operation "))
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .to_string(),
        );
    }
    assert_ne!(ids[0], ids[1]);

    let out = cli(&d.socket, &["status", "--json"]);
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

#[test]
fn cli_logs_is_an_explicit_capability_error_on_the_fake_engine() {
    let d = spawn_daemon();
    let out = cli(&d.socket, &["logs", "coder"]);
    assert_eq!(out.status.code(), Some(4));
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(err.contains("no log source"), "{err}");

    let out = cli(&d.socket, &["logs", "nope"]);
    assert_eq!(out.status.code(), Some(5));

    let out = cli(&d.socket, &["logs", "coder", "-n", "0"]);
    assert_eq!(out.status.code(), Some(6));
}

#[test]
fn unload_on_a_single_model_engine_is_refused_before_anything_is_journaled() {
    let d = spawn_daemon();
    let before = cli(&d.socket, &["status", "--json"]);
    assert!(before.status.success());
    let out = cli(&d.socket, &["unload", "coder", "some-model", "--wait"]);
    assert!(!out.status.success());
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(err.contains("no engine-level unload"), "{err}");
    // Nothing admitted: the operation list is still empty.
    let ops = cli(&d.socket, &["ops", "--json"]);
    let text = String::from_utf8(ops.stdout).unwrap();
    assert!(!text.contains("unload"), "{text}");

    // The engine report names the protocol and the missing unload.
    let out = cli(&d.socket, &["engine", "coder", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["protocol"], "openai");
    assert_eq!(v["supports_unload"], false);
}

// ---- A: foreign GPU occupancy guard ----------------------------------------

/// `gpus.json` (fake-engine world): occupants per card.
fn write_gpus(d: &DaemonGuard, occupants: serde_json::Value) {
    std::fs::write(
        d.data_dir().join("gpus.json"),
        serde_json::to_vec_pretty(&occupants).unwrap(),
    )
    .expect("write gpus.json");
}

#[tokio::test]
async fn foreign_gpu_occupant_refuses_run_before_anything_is_journaled() {
    let d = spawn_daemon();
    write_gpus(
        &d,
        serde_json::json!([{"gpu": 0, "pid": 99123, "unit": null, "memory_mib": 4096}]),
    );
    let client = d.client();

    // Dry run refuses too — the preview tells the owner why.
    match client
        .plan(PlanRequest {
            kind: OpKind::Run,
            workload: "coder".into(),
        })
        .await
    {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, codes::GPU_OCCUPIED, "{message}");
            assert!(message.contains("99123"), "{message}");
        }
        other => panic!("expected gpu_occupied on dry run, got {other:?}"),
    }

    match client
        .admit(admit_req(OpKind::Run, "coder", "gpu-foreign-1"))
        .await
    {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, codes::GPU_OCCUPIED, "{message}");
        }
        other => panic!("expected gpu_occupied, got {other:?}"),
    }

    // Nothing journaled, nothing started.
    let ops = client.list_operations().await.unwrap();
    assert!(ops.operations.is_empty(), "{ops:?}");
    let status = client.status().await.unwrap();
    let gpu0 = status.gpus.iter().find(|g| g.index == 0).unwrap();
    assert_eq!(gpu0.foreign.len(), 1);
    assert_eq!(gpu0.foreign[0].pid, 99123);
    assert_eq!(gpu0.foreign[0].unit, None);
    assert_eq!(gpu0.enrolled.len(), 0);

    // The card freed → the same run succeeds.
    write_gpus(&d, serde_json::json!([]));
    run_to_success(&d, "coder", "gpu-foreign-2").await;
}

#[tokio::test]
async fn enrolled_workload_the_plan_stops_is_not_foreign_but_a_stale_one_is() {
    let d = spawn_daemon();

    // helper runs on gpu 1; big (gpus 0+1) plans to stop it → allowed.
    run_to_success(&d, "helper", "gpu-enr-1").await;
    write_gpus(
        &d,
        serde_json::json!([{"gpu": 1, "pid": 4242, "unit": "user/llmctl-next-fake-helper.service", "memory_mib": 512}]),
    );
    run_to_success(&d, "big", "gpu-enr-2").await;

    // A stopped enrolled workload's leftover process is foreign: the plan
    // does not manage it.
    run_to_success(&d, "coder", "gpu-enr-3").await; // coder now on gpu 0
    write_gpus(
        &d,
        serde_json::json!([{"gpu": 0, "pid": 777, "unit": "user/llmctl-next-fake-helper.service", "memory_mib": 8}]),
    );
    match d
        .client()
        .admit(admit_req(OpKind::Run, "coder", "gpu-enr-4"))
        .await
    {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::GPU_OCCUPIED),
        other => panic!("expected gpu_occupied, got {other:?}"),
    }
}

/// M4c: an on-demand vendor service claims nothing at start, so a foreign
/// process on its card does not block it; an at_start workload on a card
/// where an on-demand service has a model loaded is refused with the unload.
#[tokio::test]
async fn on_demand_claims_pass_the_guard_and_name_the_unload() {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let host = "workstation";
    let wl = |name: &str, claim: Option<&str>| {
        let mut w = serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": 8100,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        });
        if let Some(c) = claim {
            w["gpu_claim"] = c.into();
        }
        w
    };
    let mut svc = wl("svc", Some("on_demand"));
    svc["port"] = 8101.into();
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [wl("coder", None), svc],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();

    // A foreign process on the card: the on-demand service still starts.
    write_gpus(&d, serde_json::json!([{"gpu": 0, "pid": 4242, "unit": null, "memory_mib": 9000}]));
    run_to_success(&d, "svc", "ondemand-1").await;

    // The service now has a model loaded on the card (its unit holds VRAM):
    // an at_start workload is refused, and told what to unload.
    write_gpus(&d, serde_json::json!([{"gpu": 0, "pid": 5151, "unit": "user/llmctl-next-fake-svc.service", "memory_mib": 17000}]));
    match client.admit(admit_req(OpKind::Run, "coder", "ondemand-2")).await {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, codes::GPU_OCCUPIED, "{message}");
            assert!(message.contains("llmctl-next unload svc"), "{message}");
        }
        other => panic!("expected gpu_occupied with the unload hint, got {other:?}"),
    }

    // Once unloaded (card empty) coder starts and the service keeps running.
    write_gpus(&d, serde_json::json!([]));
    run_to_success(&d, "coder", "ondemand-3").await;
    let st = client.status().await.unwrap();
    let svc = st.workloads.iter().find(|w| w.name == "svc").unwrap();
    assert!(
        format!("{:?}", svc.process).contains("Running"),
        "the on-demand service was evicted: {:?}",
        svc.process
    );
}

// ---- B: bounded recovery ----------------------------------------------------

/// A daemon whose inventory gives `coder` a recovery budget and whose
/// watcher runs fast (200 ms), in its own temp dir. Attempt spacing is 0 so
/// the budget math is observable in seconds.
fn spawn_recovery_daemon(mut policy: serde_json::Value) -> DaemonGuard {
    policy["min_interval_secs"] = serde_json::json!(0);
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let host = llmctl_core::HostId::new("workstation").unwrap();
    let inv = serde_json::json!({
        "inventory": {
            "host": host.to_string(),
            "platform": "linux",
            "gpus": [
                {"host": host.to_string(), "index": 0},
                {"host": host.to_string(), "index": 1}
            ],
            "workloads": [{
                "name": "coder",
                "engine": "fake",
                "model": "fake/coder",
                "gpus": [{"host": host.to_string(), "index": 0}],
                "port": 8100,
                "unit": {"scope": "user", "name": "llmctl-next-fake-coder.service"},
                "authority": {"kind": "systemd_unit", "scope": "user", "unit": "llmctl-next-fake-coder.service"},
                "ownership": {"kind": "external"},
                "recovery": policy
            }],
            "revision": 1
        }
    });
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir.path())
        .env("LLMCTL_NEXT_WATCH_INTERVAL_MS", "200")
        .arg("daemon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn daemon");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        if std::os::unix::net::UnixStream::connect(&socket).is_ok() {
            return DaemonGuard { child, dir, socket };
        }
        if let Ok(Some(_)) = child.try_wait() {
            panic!("daemon exited before accepting connections");
        }
        assert!(std::time::Instant::now() < deadline, "daemon never bound");
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Simulate the workload's process dying: the fake world no longer has it
/// active, while the durable intent still says Running.
fn kill_fake_unit(d: &DaemonGuard, name: &str) {
    let path = d.data_dir().join("world.json");
    let mut world: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    world["units"][&format!("user/llmctl-next-fake-{name}.service")]["active"] = false.into();
    world["units"][&format!("user/llmctl-next-fake-{name}.service")]["pid"] =
        serde_json::Value::Null;
    std::fs::write(&path, world.to_string()).unwrap();
}

/// Wait until a predicate over the operations list holds.
async fn wait_for_ops(
    d: &DaemonGuard,
    mut pred: impl FnMut(&[llmctl_client::wire::OperationView]) -> bool,
) -> Vec<llmctl_client::wire::OperationView> {
    let client = d.client();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let ops = client.list_operations().await.unwrap().operations;
        if pred(&ops) {
            return ops;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "condition on ops never held: {ops:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn a_dead_unit_with_running_intent_is_recovered_once() {
    let d = spawn_recovery_daemon(serde_json::json!({"max_attempts": 2, "window_secs": 3600}));
    run_to_success(&d, "coder", "rec-1").await;

    kill_fake_unit(&d, "coder");
    let ops = wait_for_ops(&d, |ops| {
        ops.iter()
            .any(|o| o.idempotency_key == "recover-coder-1" && o.status.is_terminal())
    })
    .await;
    let recover = ops
        .iter()
        .find(|o| o.idempotency_key == "recover-coder-1")
        .unwrap();
    assert_eq!(recover.status, OpStatus::Succeeded, "{recover:?}");
    assert_eq!(recover.kind, OpKind::Run);

    let status = d.client().status().await.unwrap();
    let coder = status.workloads.iter().find(|w| w.name == "coder").unwrap();
    assert_eq!(coder.process, "Running");
    assert_eq!(coder.recovery_state, "alive");
}

#[tokio::test]
async fn recovery_budget_exhausts_blocks_and_survives_the_daemon_restart() {
    let mut d = spawn_recovery_daemon(serde_json::json!({"max_attempts": 2, "window_secs": 3600}));
    run_to_success(&d, "coder", "rec-2").await;

    // Every start fails from here on: the watcher's recoveries must fail
    // and the budget must run out.
    let unit = "user/llmctl-next-fake-coder.service";
    d.write_faults(serde_json::json!({"fail_effects": [
        {"unit": unit, "kind": "start", "mode": "before"},
        {"unit": unit, "kind": "start", "mode": "before"},
        {"unit": unit, "kind": "start", "mode": "before"},
        {"unit": unit, "kind": "start", "mode": "before"},
        {"unit": unit, "kind": "start", "mode": "before"}
    ]}));
    kill_fake_unit(&d, "coder");

    // Two failed recoveries, then the exhaustion marker.
    let ops = wait_for_ops(&d, |ops| {
        ops.iter()
            .any(|o| o.idempotency_key.starts_with("recover-stalled-coder-"))
    })
    .await;
    let recoveries: Vec<_> = ops
        .iter()
        .filter(|o| o.idempotency_key.starts_with("recover-coder-"))
        .collect();
    assert_eq!(recoveries.len(), 2, "{ops:?}");
    assert!(
        recoveries.iter().all(|o| o.status == OpStatus::Failed),
        "{recoveries:?}"
    );
    let marker = ops
        .iter()
        .find(|o| o.idempotency_key.starts_with("recover-stalled-coder-"))
        .unwrap();
    assert_eq!(marker.status, OpStatus::NeedsAttention);
    assert!(!marker.acknowledged);

    // No third attempt, and mutations are blocked until the marker is acked.
    tokio::time::sleep(Duration::from_millis(700)).await;
    let ops = d.client().list_operations().await.unwrap().operations;
    assert_eq!(
        ops.iter()
            .filter(|o| o.idempotency_key.starts_with("recover-coder-"))
            .count(),
        2
    );
    match d
        .client()
        .admit(admit_req(OpKind::Run, "coder", "rec-blocked"))
        .await
    {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::BLOCKED_NEEDS_ATTENTION),
        other => panic!("expected blocked_needs_attention, got {other:?}"),
    }

    // Budget survives the daemon: after a restart nothing new is attempted.
    d.child.kill().unwrap();
    d.child.wait().unwrap();
    let socket2 = d.socket.clone();
    let dir2 = d.data_dir().to_path_buf();
    wait_socket_gone(&socket2).await;
    let mut child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&socket2)
        .arg("--data-dir")
        .arg(&dir2)
        .env("LLMCTL_NEXT_WATCH_INTERVAL_MS", "200")
        .arg("daemon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket2).is_err() {
        assert!(std::time::Instant::now() < deadline, "daemon never rebound");
        std::thread::sleep(Duration::from_millis(25));
    }
    // Several ticks pass: still exactly two attempts, still blocked.
    tokio::time::sleep(Duration::from_millis(1000)).await;
    let client = ControlClient::connect(&socket2).unwrap();
    let ops = client.list_operations().await.unwrap().operations;
    assert_eq!(
        ops.iter()
            .filter(|o| o.idempotency_key.starts_with("recover-coder-"))
            .count(),
        2,
        "restart must not reset the budget"
    );
    assert!(
        ops.iter()
            .any(|o| o.idempotency_key.starts_with("recover-stalled-coder-")),
        "the exhaustion marker survives"
    );
    child.kill().unwrap();
    child.wait().unwrap();
}

async fn wait_socket_gone(socket: &Path) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        match std::os::unix::net::UnixStream::connect(socket) {
            Err(_) => return,
            Ok(_) => {
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "socket never closed"
                );
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

#[tokio::test]
async fn a_manually_stopped_workload_is_never_recovered() {
    let d = spawn_recovery_daemon(serde_json::json!({"max_attempts": 3, "window_secs": 3600}));
    run_to_success(&d, "coder", "rec-3").await;

    // Owner stop: durable manual intent, unit down.
    let resp = d
        .client()
        .admit(admit_req(OpKind::Stop, "coder", "rec-stop"))
        .await
        .unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded);

    // The watcher has plenty of ticks to misbehave.
    tokio::time::sleep(Duration::from_millis(800)).await;
    let ops = d.client().list_operations().await.unwrap().operations;
    assert!(
        !ops.iter()
            .any(|o| o.idempotency_key.starts_with("recover-")),
        "manual stop must not be recovered: {ops:?}"
    );
}

#[test]
fn cli_run_after_a_stop_at_the_same_revision_starts_again() {
    // M2 pilot finding: the default key used to be `run-<name>-r<rev>`, so
    // a second `run` at the same revision replayed the first run's success
    // and left the workload stopped.
    let d = spawn_daemon();
    for verb in ["run", "stop", "run"] {
        let out = cli(&d.socket, &[verb, "coder", "--wait"]);
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(out.status.success(), "{verb}: {text}");
        assert!(
            !text.contains("replaying"),
            "{verb} must be a new operation: {text}"
        );
    }
    let out = cli(&d.socket, &["status", "--json"]);
    let status: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let coder = status["workloads"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["name"] == "coder")
        .unwrap();
    assert_eq!(coder["process"], "Running");
}

/// Start a daemon over an existing data dir (kept by the caller) and wait
/// until it accepts connections.
fn start_daemon_in(dir: &Path, extra: &[&str]) -> Child {
    let socket = dir.join("control.sock");
    let _ = std::fs::remove_file(&socket);
    let mut child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir)
        .arg("daemon")
        .args(extra)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn daemon");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket).is_err() {
        if let Ok(Some(st)) = child.try_wait() {
            panic!("daemon exited before accepting connections: {st}");
        }
        assert!(
            std::time::Instant::now() < deadline,
            "daemon did not come up"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    child
}

fn stop_daemon(mut child: Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// An enrolled agent answers to its enrollment name: `--host-id` renames
/// the bound id and the inventory's host and GPU ids once, and the name
/// sticks without the flag (M3 pilot: mbp2's agent said `workstation`).
#[test]
fn host_id_flag_renames_the_bound_host_and_its_inventory_once() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let host_of = |socket: &Path| -> String {
        let out = cli(socket, &["status", "--json"]);
        assert!(out.status.success());
        let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        json["host"].as_str().unwrap().to_string()
    };
    let inventory = || -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(dir.path().join("inventory.json")).unwrap()).unwrap()
    };

    let d = start_daemon_in(dir.path(), &[]);
    assert_eq!(host_of(&socket), "workstation");
    stop_daemon(d);
    let before = inventory();

    let d = start_daemon_in(dir.path(), &["--host-id", "mbp2"]);
    assert_eq!(host_of(&socket), "mbp2");
    stop_daemon(d);
    let after = inventory();
    assert_eq!(after["inventory"]["host"], "mbp2");
    let gpu_hosts: Vec<&serde_json::Value> = after["inventory"]["gpus"]
        .as_array()
        .unwrap()
        .iter()
        .chain(
            after["inventory"]["workloads"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|w| w["gpus"].as_array().unwrap().iter()),
        )
        .map(|g| &g["host"])
        .collect();
    assert!(!gpu_hosts.is_empty());
    assert!(gpu_hosts.iter().all(|h| *h == "mbp2"), "{after}");
    assert_eq!(
        after["inventory"]["revision"].as_u64().unwrap(),
        before["inventory"]["revision"].as_u64().unwrap() + 1
    );

    // Bound now: no flag, same name, inventory untouched.
    let d = start_daemon_in(dir.path(), &[]);
    assert_eq!(host_of(&socket), "mbp2");
    stop_daemon(d);
    assert_eq!(inventory(), after);
}
