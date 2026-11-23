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
        force: false,
        profile: None,
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
            profile: None,
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
            force: false,
            profile: None,
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

/// Spawn a daemon on an existing data dir (the restart tests reuse one).
fn spawn_daemon_in(dir: &Path, socket: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(socket)
        .arg("--data-dir")
        .arg(dir)
        .arg("daemon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn daemon")
}

/// Poll an operation until one of its phases matches `pred`.
async fn wait_phase(
    client: &ControlClient,
    id: &str,
    what: &str,
    pred: impl Fn(&llmctl_client::wire::PhaseView) -> bool,
) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    loop {
        let op = client.operation(id).await.expect("operation readable");
        if op.phases.iter().any(&pred) {
            return;
        }
        assert!(
            !op.status.is_terminal() && tokio::time::Instant::now() < deadline,
            "never reached {what}: {op:?}"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

fn effect_pending(label_prefix: &'static str) -> impl Fn(&llmctl_client::wire::PhaseView) -> bool {
    move |p| {
        p.label.starts_with(label_prefix)
            && p.status == llmctl_client::wire::PhaseStatus::EffectPending
    }
}

/// A daemon killed while it restores the world after a failed step must
/// not auto-resolve on restart: the neighbours it was restarting may still
/// be down. needs_attention, unacknowledged, blocking.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restart_during_rollback_needs_attention() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let mut child = ChildGuard(Some(spawn_daemon_in(dir.path(), &socket)));
    wait_socket(&socket).await;
    let client = ControlClient::connect(&socket).unwrap();
    let setup = client
        .admit(admit_req(OpKind::Run, "helper", "k-rb-setup"))
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while !client.operation(&setup.operation.id).await.unwrap().status.is_terminal() {
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // "start big" applies but reports failure; every effect is slow, so
    // the rollback's restart of helper sits in its effect window.
    std::fs::write(
        dir.path().join("faults.json"),
        serde_json::to_vec(&serde_json::json!({
            "effect_delay_ms": 1500,
            "fail_effects": [
                {"unit": "user/llmctl-next-fake-big.service", "kind": "start", "mode": "after"}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let admitted = client
        .admit(admit_req(OpKind::Run, "big", "k-rb"))
        .await
        .unwrap();
    let id = admitted.operation.id.clone();
    wait_phase(&client, &id, "a rollback effect", effect_pending("rollback:")).await;
    child.kill_now();
    std::fs::remove_file(&socket).unwrap_or(());

    std::fs::write(dir.path().join("faults.json"), b"{}").unwrap();
    let guard = DaemonGuard {
        child: spawn_daemon_in(dir.path(), &socket),
        dir,
        socket: socket.clone(),
    };
    wait_socket(&socket).await;
    let client2 = guard.client();
    let op = client2.operation(&id).await.unwrap();
    assert_eq!(op.status, OpStatus::NeedsAttention, "{op:?}");
    assert!(!op.acknowledged, "an incomplete restoration is never auto-resolved");
    match client2
        .admit(admit_req(OpKind::Run, "coder", "k-rb-2"))
        .await
    {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::BLOCKED_NEEDS_ATTENTION),
        other => panic!("expected block, got {other:?}"),
    }
}

/// A daemon killed inside a start effect (outcome unknown) must not report
/// "external effects complete" and auto-resolve.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restart_inside_start_effect_needs_attention() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    std::fs::write(
        dir.path().join("faults.json"),
        serde_json::to_vec(&serde_json::json!({"effect_delay_ms": 3000})).unwrap(),
    )
    .unwrap();
    let mut child = ChildGuard(Some(spawn_daemon_in(dir.path(), &socket)));
    wait_socket(&socket).await;
    let client = ControlClient::connect(&socket).unwrap();
    let admitted = client
        .admit(admit_req(OpKind::Run, "coder", "k-mid-start"))
        .await
        .unwrap();
    let id = admitted.operation.id.clone();
    wait_phase(&client, &id, "the start effect", effect_pending("start")).await;
    child.kill_now();
    std::fs::remove_file(&socket).unwrap_or(());

    std::fs::write(dir.path().join("faults.json"), b"{}").unwrap();
    let guard = DaemonGuard {
        child: spawn_daemon_in(dir.path(), &socket),
        dir,
        socket: socket.clone(),
    };
    wait_socket(&socket).await;
    let op = guard.client().operation(&id).await.unwrap();
    assert_eq!(op.status, OpStatus::NeedsAttention, "{op:?}");
    assert!(!op.acknowledged);
}

/// A cancel that lands after a neighbour was evicted restores it: the
/// world returns to where the operation started.
#[tokio::test]
async fn cancel_after_eviction_restores_the_neighbour() {
    let d = spawn_daemon();
    run_to_success(&d, "helper", "k-cx-setup").await;
    d.write_faults(serde_json::json!({"effect_delay_ms": 1000}));
    let client = d.client();
    let resp = client
        .admit(admit_req(OpKind::Run, "big", "k-cx"))
        .await
        .unwrap();
    let id = resp.operation.id.clone();
    wait_phase(&client, &id, "stopping helper", effect_pending("stop")).await;
    client.cancel(&id, Some("test".into())).await.unwrap();
    let op = d.wait_terminal(&id).await;
    assert_eq!(op.status, OpStatus::Canceled, "{op:?}");
    let recovery = op.recovery.expect("restoration recorded");
    assert_eq!(recovery["outcome"], "restored", "{recovery}");

    let status = d.client().status().await.unwrap();
    let process = |name: &str| {
        status
            .workloads
            .iter()
            .find(|w| w.name == name)
            .unwrap()
            .process
            .clone()
    };
    assert_eq!(process("helper"), "Running", "evicted neighbour restored");
    assert_ne!(process("big"), "Running", "canceled target not left up");
}

/// A canceled restart restores the target, and the restoration's own
/// readiness wait is not cut short by the cancel that triggered it.
#[tokio::test]
async fn cancel_during_restart_restores_the_target() {
    let d = spawn_daemon();
    run_to_success(&d, "coder", "k-cr-setup").await;
    d.write_faults(serde_json::json!({"effect_delay_ms": 1000}));
    let client = d.client();
    let resp = client
        .admit(admit_req(OpKind::Restart, "coder", "k-cr"))
        .await
        .unwrap();
    let id = resp.operation.id.clone();
    wait_phase(&client, &id, "stopping coder", effect_pending("stop")).await;
    client.cancel(&id, Some("test".into())).await.unwrap();
    let op = d.wait_terminal(&id).await;
    assert_eq!(op.status, OpStatus::Canceled, "{op:?}");
    assert_eq!(op.recovery.as_ref().unwrap()["outcome"], "restored", "{op:?}");
    let status = d.client().status().await.unwrap();
    let coder = status.workloads.iter().find(|w| w.name == "coder").unwrap();
    assert_eq!(coder.process, "Running");
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
            profile: None,
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

/// A stand-in Ollama HTTP server: `/api/ps` names what is resident (with a
/// `size_vram`), a load-only `POST /api/generate` with a non-zero
/// `keep_alive` makes the model resident, `keep_alive: 0` drops it.
async fn stand_in_ollama() -> (u16, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let loaded = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let l2 = loaded.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            let l3 = l2.clone();
            tokio::spawn(async move {
                loop {
                    let mut buf = vec![0u8; 16384];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let line = req.lines().next().unwrap_or("").to_string();
                    let body: serde_json::Value = if line.starts_with("POST /api/generate") {
                        let model = req
                            .split("\"model\":\"")
                            .nth(1)
                            .and_then(|r| r.split('"').next())
                            .unwrap_or("")
                            .to_string();
                        if req.contains("\"keep_alive\":0") {
                            l3.lock().unwrap().retain(|m| m != &model);
                        } else if !model.is_empty() {
                            let mut ps = l3.lock().unwrap();
                            if !ps.iter().any(|m| *m == model) {
                                ps.push(model);
                            }
                        }
                        serde_json::json!({"done": true})
                    } else if line.starts_with("GET /api/ps") {
                        let models: Vec<serde_json::Value> = l3
                            .lock()
                            .unwrap()
                            .iter()
                            .map(|m| serde_json::json!({"name": m, "size_vram": 19_000_000_000u64}))
                            .collect();
                        serde_json::json!({ "models": models })
                    } else if line.starts_with("GET /api/version") {
                        serde_json::json!({"version": "0.33.1"})
                    } else if line.starts_with("GET /api/tags") {
                        let models: Vec<serde_json::Value> = l3
                            .lock()
                            .unwrap()
                            .iter()
                            .map(|m| serde_json::json!({"name": m}))
                            .collect();
                        serde_json::json!({ "models": models })
                    } else {
                        serde_json::json!({})
                    };
                    let text = body.to_string();
                    let head = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
                        text.len()
                    );
                    let _ = sock.write_all(head.as_bytes()).await;
                    let _ = sock.write_all(text.as_bytes()).await;
                }
            });
        }
    });
    (port, loaded)
}

/// M5c item 4: `load` preloads one model onto a running Ollama service —
/// catalog-checked, fit-checked, confirmed through `/api/ps`, and the
/// measured need recorded for the next plan. Multi-thread runtime: the
/// stand-in Ollama must answer the daemon while `cli()` blocks a worker.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_load_verb_preloads_an_ollama_model_checked_and_measured() {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let (port, loaded) = stand_in_ollama().await;
    let host = "workstation";
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [{
            "name": "svc", "engine": "fake", "engine_kind": "ollama", "model": "ollama",
            "gpu_claim": "on_demand",
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": "llmctl-next-fake-svc.service"},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": "llmctl-next-fake-svc.service"},
            "ownership": {"kind": "external"}
        }],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();

    // The catalog knows the model, with an estimate (GiB).
    std::fs::write(
        d.data_dir().join("catalog.json"),
        serde_json::json!({
            "revision": 1,
            "models": [{
                "id": "m-1", "name": "qwen3-coder:30b", "engine": "ollama",
                "source": "qwen3-coder:30b", "enabled": true, "origin": "registry",
                "extra": {"estimated_vram_gb": 18.5}
            }]
        })
        .to_string(),
    )
    .unwrap();

    // A stopped service refuses before anything is journaled.
    let out = cli(&d.socket, &["load", "svc", "qwen3-coder:30b", "--wait"]);
    assert!(!out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    // Start it (the on-demand service claims nothing at start).
    run_to_success(&d, "svc", "load-e2e-1").await;

    // A model the catalog does not name: refused, exit 4, nothing journaled.
    let out = cli(&d.socket, &["load", "svc", "unknown:1b", "--wait"]);
    assert_eq!(out.status.code(), Some(4), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("not in this host's catalog"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Free VRAM comes from `gpu_free.json` (absent = unreadable = never
    // free). A card too small for the estimate: refused with the numbers.
    std::fs::write(d.data_dir().join("gpu_free.json"), r#"{"0": 2000}"#).unwrap();
    let out = cli(&d.socket, &["load", "svc", "qwen3-coder:30b", "--wait"]);
    assert_eq!(out.status.code(), Some(4));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("gpu 0"), "{err}");

    // The card roomy again: the load runs, confirms through /api/ps, and
    // records the engine-reported size_vram as the measured need.
    std::fs::write(d.data_dir().join("gpu_free.json"), r#"{"0": 32000}"#).unwrap();
    let out = cli(&d.socket, &["load", "svc", "qwen3-coder:30b", "--wait"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(loaded.lock().unwrap().last().unwrap(), "qwen3-coder:30b");
    let vram: serde_json::Value = serde_json::from_slice(
        &std::fs::read(d.data_dir().join("vram.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        vram["svc|qwen3-coder:30b"]["per_card"]["0"].as_u64(),
        Some(19_000_000_000u64 / 1_048_576),
        "{vram}"
    );
    // The engine report names the load.
    let report = client.engine_report("svc").await.unwrap();
    assert!(report.supports_load);

    // Already loaded: the same load succeeds again (the end state holds).
    let out = cli(&d.socket, &["load", "svc", "qwen3-coder:30b", "--wait"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

/// Set what a fake vendor service holds in memory (`world.json`).
fn set_loaded(d: &DaemonGuard, name: &str, models: &[&str]) {
    let path = d.data_dir().join("world.json");
    let mut world: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    world["units"][&format!("user/llmctl-next-fake-{name}.service")]["loaded"] =
        serde_json::json!(models);
    std::fs::write(&path, world.to_string()).unwrap();
}

fn loaded_of(d: &DaemonGuard, name: &str) -> serde_json::Value {
    let world: serde_json::Value =
        serde_json::from_slice(&std::fs::read(d.data_dir().join("world.json")).unwrap()).unwrap();
    world["units"][&format!("user/llmctl-next-fake-{name}.service")]["loaded"].clone()
}

/// M4c swap plan: one run unloads the on-demand service's models, evicts
/// the at_start conflict, starts the target; a failed start restores the
/// evicted workload and never stops the service.
#[tokio::test]
async fn swap_plan_unloads_on_demand_models_then_evicts_and_rolls_back() {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let host = "workstation";
    let wl = |name: &str, port: u16| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        })
    };
    let mut svc = wl("svc", 8102);
    svc["gpu_claim"] = "on_demand".into();
    svc["engine_kind"] = "ollama".into();
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [wl("coder", 8100), wl("old", 8101), svc],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();
    let running = |st: &llmctl_client::wire::StatusResponse, name: &str| {
        let w = st.workloads.iter().find(|w| w.name == name).unwrap();
        format!("{:?}", w.process).contains("Running")
    };

    // `old` serves on the card; the on-demand service has two models on it.
    run_to_success(&d, "old", "swap-setup-1").await;
    run_to_success(&d, "svc", "swap-setup-2").await;
    set_loaded(&d, "svc", &["qwen3:8b", "nomic-embed"]);

    // The service holds nothing on the card (its models sit in RAM): the
    // plan leaves them alone.
    write_gpus(&d, serde_json::json!([
        {"gpu": 0, "pid": 4101, "unit": "user/llmctl-next-fake-old.service", "memory_mib": 20000}
    ]));
    let plan = client
        .plan(PlanRequest { kind: OpKind::Run, workload: "coder".into(), profile: None })
        .await
        .unwrap();
    assert!(
        !plan.plan.steps.iter().any(|s| s.label().starts_with("unload")),
        "{:?}",
        plan.plan.steps
    );

    // Now it does occupy the card.
    write_gpus(&d, serde_json::json!([
        {"gpu": 0, "pid": 4101, "unit": "user/llmctl-next-fake-old.service", "memory_mib": 20000},
        {"gpu": 0, "pid": 4102, "unit": "user/llmctl-next-fake-svc.service", "memory_mib": 17000}
    ]));

    // The preview names the unloads, before the eviction; only `old` stops.
    let plan = client
        .plan(PlanRequest { kind: OpKind::Run, workload: "coder".into(), profile: None })
        .await
        .expect("the swap plan passes the occupancy guard");
    let labels: Vec<String> = plan.plan.steps.iter().map(|s| s.label()).collect();
    assert_eq!(
        labels[1..5],
        [
            "unload nomic-embed from svc",
            "unload qwen3:8b from svc",
            "record intent: old → Stopped { manual: false }",
            "stop old",
        ],
        "{labels:?}"
    );
    assert_eq!(plan.stop_targets, vec!["old"]);

    // The target's start fails: `old` is restored, the service was never
    // stopped, and its models stay unloaded (they load again on demand).
    d.write_faults(serde_json::json!({"fail_effects": [
        {"unit": "user/llmctl-next-fake-coder.service", "kind": "start", "mode": "before"}
    ]}));
    let resp = client.admit(admit_req(OpKind::Run, "coder", "swap-fail")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Failed, "{op:?}");
    assert_eq!(op.recovery.expect("restoration recorded")["outcome"], "restored");
    let st = client.status().await.unwrap();
    assert!(running(&st, "old") && running(&st, "svc") && !running(&st, "coder"));
    assert_eq!(loaded_of(&d, "svc"), serde_json::json!([]));

    // Second try, one model loaded again: the swap goes through.
    set_loaded(&d, "svc", &["qwen3:8b"]);
    let resp = client.admit(admit_req(OpKind::Run, "coder", "swap-ok")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    assert!(op.phases.iter().any(|p| p.label == "unload qwen3:8b from svc"));
    let st = client.status().await.unwrap();
    assert!(running(&st, "coder") && running(&st, "svc") && !running(&st, "old"));
    assert_eq!(loaded_of(&d, "svc"), serde_json::json!([]));
}

// ---- M4c: legacy host flock + lane lease -------------------------------------

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Non-blocking exclusive flock, as llm-router and Python llmctl take it.
fn try_flock(file: &std::fs::File) -> bool {
    use std::os::fd::AsRawFd;
    unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 }
}

/// Every GPU operation holds the host-wide operation lock that llm-router
/// and Python llmctl share, and is refused under another tool's fresh lane
/// lease unless the lease names the workload or the caller runs inside the
/// lease holder.
#[tokio::test]
async fn gpu_operations_take_the_host_lock_and_honour_the_lane_lease() {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let lock_path = dir.path().join("operations.lock");
    let lease_path = dir.path().join("lane-lease.json");
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
        .env("LLMCTL_NEXT_OP_LOCK", &lock_path)
        .env("LLMCTL_NEXT_LANE_LEASE", &lease_path)
        .env("LLMCTL_NEXT_OP_LOCK_WAIT_MS", "300")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();
    let refused = |r: Result<llmctl_client::wire::AdmitResponse, ClientError>, want: &str| match r {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, want, "{message}");
            message
        }
        other => panic!("expected {want}, got {other:?}"),
    };

    // 1. Another controller holds the lock: refused, naming the holder, and
    //    nothing is journaled (the same key is free afterwards).
    std::fs::write(
        &lock_path,
        serde_json::json!({"pid": std::process::id(), "hostname": "desk", "interface": "llm-router",
            "purpose": "switch qwen27b", "acquired_at": unix_now(), "thread": 0})
        .to_string(),
    )
    .unwrap();
    let held = std::fs::OpenOptions::new().read(true).write(true).open(&lock_path).unwrap();
    assert!(try_flock(&held));
    let msg = refused(
        client.admit(admit_req(OpKind::Run, "coder", "lock-1")).await,
        codes::HOST_LOCK_BUSY,
    );
    assert!(msg.contains("llm-router") && msg.contains("switch qwen27b"), "{msg}");

    // PID claims never replace the executor's own lock, including a true
    // claim. Cooperating callers release their hold before delegation.
    for pid in [1, std::process::id()] {
        let holder = d.client().holding_lock(pid);
        refused(holder.admit(admit_req(OpKind::Run, "helper", "lock-pass")).await, codes::HOST_LOCK_BUSY);
    }
    drop(held);

    // 2. Free: the run goes through under our own hold, recorded in the
    //    format the other controllers read; the lock is free afterwards.
    let resp = client.admit(admit_req(OpKind::Run, "coder", "lock-1")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    let holder: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&lock_path).unwrap()).unwrap();
    assert_eq!(holder["interface"], "llmctl-next");
    assert_eq!(holder["purpose"], "run coder");
    assert_eq!(holder["pid"], d.child.id());
    let probe = std::fs::OpenOptions::new().read(true).write(true).open(&lock_path).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !try_flock(&probe) {
        assert!(std::time::Instant::now() < deadline, "lock still held after the operation");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    drop(probe);

    // 3. A fresh lease of a live tool on another lane: refused.
    let lease = |lanes: &[&str], updated: f64| {
        std::fs::write(
            &lease_path,
            serde_json::json!({"holder": "longjob", "pid": std::os::unix::process::parent_id(),
                "lanes": lanes, "updated": updated})
            .to_string(),
        )
        .unwrap();
    };
    lease(&["bench-x"], unix_now());
    let msg = refused(
        client.admit(admit_req(OpKind::Stop, "coder", "lease-1")).await,
        codes::LANE_LEASED,
    );
    assert!(msg.contains("longjob") && msg.contains("bench-x"), "{msg}");

    //    A caller that runs inside the lease holder passes (a job that
    //    drives llmctl-next itself).
    let inside = d.client().under_lease(std::os::unix::process::parent_id());
    let resp = inside.admit(admit_req(OpKind::Stop, "coder", "lease-2")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);
    //    ... but not one that names the wrong holder.
    let wrong = d.client().under_lease(1);
    refused(wrong.admit(admit_req(OpKind::Run, "coder", "lease-3")).await, codes::LANE_LEASED);

    //    A lease that names the workload, a stale lease and a dead holder
    //    all let the operation through.
    lease(&["coder"], unix_now());
    run_to_success(&d, "coder", "lease-4").await;
    lease(&["bench-x"], unix_now() - 3600.0);
    run_to_success(&d, "coder", "lease-5").await;
    std::fs::write(
        &lease_path,
        serde_json::json!({"holder": "longjob", "pid": 2147483646, "lanes": ["bench-x"],
            "updated": unix_now()})
        .to_string(),
    )
    .unwrap();
    run_to_success(&d, "coder", "lease-6").await;
}

// ---- M4c: request safety (in-flight requests from llm-router) ----------------

/// A stand-in for llm-router's `GET /running`: in-flight counts per
/// upstream, set by the test. Returns the port and the shared counts.
async fn fake_router() -> (u16, std::sync::Arc<std::sync::Mutex<serde_json::Value>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let counts = std::sync::Arc::new(std::sync::Mutex::new(serde_json::json!({})));
    let shared = counts.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = listener.accept().await else { return };
            let mut buf = vec![0u8; 4096];
            let _ = sock.read(&mut buf).await;
            let body = serde_json::json!({
                "loaded": [],
                "idle_revert": {"in_flight": shared.lock().unwrap().clone(), "deferred": null}
            })
            .to_string();
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(head.as_bytes()).await;
            let _ = sock.write_all(body.as_bytes()).await;
        }
    });
    (port, counts)
}

/// A stop, restart or eviction of a workload that is answering requests
/// through the router waits for them, then refuses — unless forced.
#[tokio::test]
async fn stops_wait_for_in_flight_requests_or_refuse_without_force() {
    let (router_port, counts) = fake_router().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
        .env("LLMCTL_NEXT_ROUTER", router_port.to_string())
        .env("LLMCTL_NEXT_INFLIGHT_WAIT_MS", "700")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();
    let st = client.status().await.unwrap();
    let port_of = |name: &str| st.workloads.iter().find(|w| w.name == name).unwrap().port;
    let upstream = |name: &str| format!("http://127.0.0.1:{}", port_of(name));
    let set = |v: serde_json::Value| *counts.lock().unwrap() = v;
    let running = |st: &llmctl_client::wire::StatusResponse, name: &str| {
        let w = st.workloads.iter().find(|w| w.name == name).unwrap();
        format!("{:?}", w.process).contains("Running")
    };

    // Starting something that stops nobody never asks.
    set(serde_json::json!({upstream("coder"): 3}));
    run_to_success(&d, "coder", "rs-setup").await;

    // 1. Two requests in flight on coder: a stop waits out its bound, then
    //    refuses, naming the workload and the count. Nothing is stopped.
    set(serde_json::json!({upstream("coder"): 2}));
    let started = std::time::Instant::now();
    match client.admit(admit_req(OpKind::Stop, "coder", "rs-1")).await {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, codes::REQUESTS_IN_FLIGHT, "{message}");
            assert!(message.contains("coder") && message.contains('2'), "{message}");
            assert!(message.contains("--force"), "{message}");
        }
        other => panic!("expected requests_in_flight, got {other:?}"),
    }
    assert!(started.elapsed() >= Duration::from_millis(600), "did not wait: {:?}", started.elapsed());
    assert!(running(&client.status().await.unwrap(), "coder"));

    // 2. The requests finish inside the bound: the stop is admitted.
    let finish = counts.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(250)).await;
        *finish.lock().unwrap() = serde_json::json!({});
    });
    let resp = client.admit(admit_req(OpKind::Stop, "coder", "rs-2")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);

    // 3. An eviction counts too: `big` needs coder's card, coder is busy.
    run_to_success(&d, "coder", "rs-3-setup").await;
    set(serde_json::json!({upstream("coder"): 1}));
    match client.admit(admit_req(OpKind::Run, "big", "rs-3")).await {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, codes::REQUESTS_IN_FLIGHT, "{message}");
            assert!(message.contains("coder"), "{message}");
        }
        other => panic!("expected requests_in_flight, got {other:?}"),
    }

    // 4. Forced: admitted at once, requests or not.
    let mut forced = admit_req(OpKind::Run, "big", "rs-4");
    forced.force = true;
    let started = std::time::Instant::now();
    let resp = client.admit(forced).await.unwrap();
    assert!(started.elapsed() < Duration::from_millis(600), "forced admit waited");
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);

    // 5. Requests on workloads this plan does not stop, and on ports that
    //    are nobody's, do not matter.
    set(serde_json::json!({upstream("helper"): 4, "http://127.0.0.1:1": 9}));
    let resp = client.admit(admit_req(OpKind::Stop, "big", "rs-5")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);
}

// ---- M4c: env-file profiles ---------------------------------------------------

/// A run that names a profile swaps the workload's env file inside the
/// operation, verifies the profile's model, re-pins the workload, and puts
/// everything back — env, model, a serving target — when the start fails.
#[tokio::test]
async fn a_profile_run_swaps_the_env_repins_and_rolls_back() {
    use llmctl_client::wire::ProfileRequest;
    use llmctl_core::launch::{EnvFileRef, LaunchDefinition};
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let services = dir.path().canonicalize().unwrap().join("services");
    std::fs::create_dir_all(&services).unwrap();
    let env = services.join("coder.env");
    let file = |name: &str| services.join(name).display().to_string();
    std::fs::write(&env, "MODEL=base\n").unwrap();
    std::fs::write(file("coder.env.fast"), "MODEL=fast\n").unwrap();
    std::fs::write(file("coder.env.slow"), "MODEL=slow\n").unwrap();
    let sha = |bytes: &[u8]| {
        use sha2::Digest;
        format!("sha256:{:x}", sha2::Sha256::digest(bytes))
    };

    // `coder` is adopted with one pinned env file; `helper` is not adopted.
    let definition = LaunchDefinition {
        environment_files: vec![EnvFileRef {
            path: env.display().to_string(),
            optional: false,
            content_digest: sha(b"MODEL=base\n"),
        }],
        ..Default::default()
    };
    let host = "workstation";
    let wl = |name: &str, port: u16| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        })
    };
    let mut coder = wl("coder", 8100);
    coder["ownership"] = serde_json::json!({
        "kind": "managed", "definition": definition, "digest": definition.digest(),
        "adopted_at": "2026-11-17T00:00:00Z", "provenance": []
    });
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [coder, wl("helper", 8101)],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();
    let profile = |name: &str, model: &str| {
        Some(ProfileRequest { env_source: file(name), model: model.into() })
    };
    let run_with = |key: &str, p: Option<ProfileRequest>| {
        let mut req = admit_req(OpKind::Run, "coder", key);
        req.profile = p;
        req
    };
    let coder_view = || async {
        let st = client.status().await.unwrap();
        let w = st.workloads.into_iter().find(|w| w.name == "coder").unwrap();
        (st.inventory_revision, w.model, w.served_model, w.served_model_verified, w.process)
    };
    let backup = file("coder.env.llmctl-previous");

    run_to_success(&d, "coder", "prof-setup").await;
    let (rev0, model, ..) = coder_view().await;
    assert_eq!(model, "fake/coder");

    // The preview: stop, swap, start, verify the profile's model.
    let plan = client
        .plan(PlanRequest {
            kind: OpKind::Run,
            workload: "coder".into(),
            profile: profile("coder.env.fast", "fake/coder-fast"),
        })
        .await
        .unwrap();
    let labels: Vec<String> = plan.plan.steps.iter().map(|s| s.label()).collect();
    assert_eq!(
        labels[1..],
        [
            "stop coder",
            "apply profile coder.env.fast to coder (serves fake/coder-fast)",
            "start coder",
            "wait for coder readiness",
            "verify coder serves fake/coder-fast",
        ],
        "{labels:?}"
    );
    assert_eq!(std::fs::read_to_string(&env).unwrap(), "MODEL=base\n", "a preview changes nothing");

    // 1. The switch: env swapped, old env kept, workload re-pinned.
    let resp = client.admit(run_with("prof-1", profile("coder.env.fast", "fake/coder-fast"))).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    assert_eq!(std::fs::read_to_string(&env).unwrap(), "MODEL=fast\n");
    assert_eq!(std::fs::read_to_string(&backup).unwrap(), "MODEL=base\n");
    let (rev1, model, served, verified, _) = coder_view().await;
    assert!(rev1 > rev0, "the re-pin is an inventory revision");
    assert_eq!((model.as_str(), served.as_deref(), verified), ("fake/coder-fast", Some("fake/coder-fast"), true));
    // The journal says what happened and never what the env contains.
    let journal = serde_json::to_string(&op).unwrap();
    assert!(journal.contains("swapped") && !journal.contains("MODEL="), "{journal}");

    //    Later operations run on the new pin: a plain restart verifies the
    //    profile's model, and naming the same profile again is a plain run.
    let resp = client.admit(admit_req(OpKind::Restart, "coder", "prof-1r")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);
    let again = client
        .plan(PlanRequest {
            kind: OpKind::Run,
            workload: "coder".into(),
            profile: profile("coder.env.fast", "fake/coder-fast"),
        })
        .await
        .unwrap();
    assert!(!again.plan.steps.iter().any(|s| s.label().starts_with("apply profile")));

    // 2. A switch whose start fails: the old env is back, coder serves the
    //    previous profile again (verified), and the pin is the previous one.
    d.write_faults(serde_json::json!({"fail_effects": [
        {"unit": "user/llmctl-next-fake-coder.service", "kind": "start", "mode": "before"}
    ]}));
    let resp = client.admit(run_with("prof-2", profile("coder.env.slow", "fake/coder-slow"))).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Failed, "{op:?}");
    assert_eq!(op.recovery.expect("restoration recorded")["outcome"], "restored");
    assert_eq!(std::fs::read_to_string(&env).unwrap(), "MODEL=fast\n");
    let (_, model, served, verified, process) = coder_view().await;
    assert_eq!((model.as_str(), served.as_deref(), verified), ("fake/coder-fast", Some("fake/coder-fast"), true));
    assert!(process.contains("Running"), "{process}");

    // 3. Requests the agent refuses before anything happens.
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("x.env"), "MODEL=x\n").unwrap();
    let bad = [
        (run_with("prof-b1", Some(ProfileRequest {
            env_source: outside.path().join("x.env").display().to_string(), model: "m".into() })), "next to the env file"),
        (run_with("prof-b2", profile("missing.env", "m")), "missing.env"),
        (run_with("prof-b3", profile("coder.env.slow", " ")), "needs the model"),
        ({ let mut r = run_with("prof-b4", profile("coder.env.slow", "m")); r.workload = "helper".into(); r }, "not adopted"),
        ({ let mut r = run_with("prof-b5", profile("coder.env.slow", "m")); r.kind = OpKind::Restart; r }, "is a `run`"),
    ];
    for (req, why) in bad {
        match client.admit(req).await {
            Err(ClientError::Api { code, message, .. }) => {
                assert_eq!(code, codes::INVALID_REQUEST, "{message}");
                assert!(message.contains(why), "{message}");
            }
            other => panic!("expected a refusal mentioning {why:?}, got {other:?}"),
        }
    }
    assert_eq!(std::fs::read_to_string(&env).unwrap(), "MODEL=fast\n");
}

// ---- M4c: manual stops ---------------------------------------------------------

/// Something outside llmctl-next started the workload again (a start script,
/// a healer): the fake world has it active while the intent is untouched.
fn start_fake_unit_outside(d: &DaemonGuard, name: &str) {
    let path = d.data_dir().join("world.json");
    let mut world: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let unit = &mut world["units"][&format!("user/llmctl-next-fake-{name}.service")];
    unit["active"] = true.into();
    unit["pid"] = 9999.into();
    unit["ready_at"] = "2026-01-01T00:00:00Z".into();
    unit["model"] = format!("fake/{name}").into();
    std::fs::write(&path, world.to_string()).unwrap();
}

/// A manual stop is the owner's decision. The healers ask for it (`intent`),
/// an eviction does not turn it into an ordinary stop, and a rollback never
/// starts the workload again.
#[tokio::test]
async fn a_manual_stop_is_reported_survives_eviction_and_is_not_restored() {
    let d = spawn_daemon();
    let client = d.client();
    let manual = |intents: &llmctl_client::wire::IntentsResponse, name: &str| {
        intents.intents.iter().find(|i| i.workload == name).map(|i| i.manual_stop)
    };
    let running = |st: &llmctl_client::wire::StatusResponse, name: &str| {
        let w = st.workloads.iter().find(|w| w.name == name).unwrap();
        format!("{:?}", w.process).contains("Running")
    };

    run_to_success(&d, "coder", "ms-setup").await;
    let intents = client.intents().await.unwrap();
    assert_eq!(manual(&intents, "coder"), Some(false));
    let coder = intents.intents.iter().find(|i| i.workload == "coder").unwrap();
    assert_eq!(coder.unit, "llmctl-next-fake-coder.service");

    // The owner stops coder: reported as a manual stop.
    let resp = client.admit(admit_req(OpKind::Stop, "coder", "ms-stop")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);
    assert_eq!(manual(&client.intents().await.unwrap(), "coder"), Some(true));

    // Something else starts it anyway. `big` needs its card, so the plan
    // evicts it; big's start fails; the rollback must not bring coder back.
    start_fake_unit_outside(&d, "coder");
    assert!(running(&client.status().await.unwrap(), "coder"));
    d.write_faults(serde_json::json!({"fail_effects": [
        {"unit": "user/llmctl-next-fake-big.service", "kind": "start", "mode": "before"}
    ]}));
    let resp = client.admit(admit_req(OpKind::Run, "big", "ms-fail")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Failed, "{op:?}");
    let recovery = op.recovery.expect("restoration recorded");
    assert_eq!(recovery["outcome"], "restored", "{recovery}");
    let skipped: Vec<&str> = recovery["steps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["outcome"] == "skipped")
        .filter_map(|s| s["step"].as_str())
        .collect();
    assert_eq!(skipped, vec!["record intent: coder → Running", "start coder"], "{recovery}");
    assert!(recovery.to_string().contains("manually stopped"), "{recovery}");
    let st = client.status().await.unwrap();
    assert!(!running(&st, "coder") && !running(&st, "big"));
    assert_eq!(manual(&client.intents().await.unwrap(), "coder"), Some(true));

    // The same eviction in a run that succeeds: still a manual stop after.
    start_fake_unit_outside(&d, "coder");
    run_to_success(&d, "big", "ms-ok").await;
    assert_eq!(manual(&client.intents().await.unwrap(), "coder"), Some(true));

    // An ordinary eviction is not a manual stop, and its rollback restores.
    let resp = client.admit(admit_req(OpKind::Stop, "big", "ms-stop-big")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);
    run_to_success(&d, "helper", "ms-helper").await;
    run_to_success(&d, "big", "ms-evict-helper").await;
    assert_eq!(manual(&client.intents().await.unwrap(), "helper"), Some(false));
}

// ---- M4c: VRAM fit ---------------------------------------------------------------

fn write_gpu_free(d: &DaemonGuard, free: serde_json::Value) {
    std::fs::write(d.data_dir().join("gpu_free.json"), free.to_string()).expect("write gpu_free.json");
}

/// A start keeps its neighbours when the target was measured to fit beside
/// them, and makes room the old way on every unknown.
#[tokio::test]
async fn a_start_that_fits_runs_beside_its_neighbours() {
    let d = spawn_daemon();
    let client = d.client();
    let unit = |n: &str| format!("user/llmctl-next-fake-{n}.service");
    let plan_big = || async {
        client
            .plan(PlanRequest { kind: OpKind::Run, workload: "big".into(), profile: None })
            .await
            .unwrap()
    };
    let running = |st: &llmctl_client::wire::StatusResponse, name: &str| {
        let w = st.workloads.iter().find(|w| w.name == name).unwrap();
        format!("{:?}", w.process).contains("Running")
    };

    // coder holds gpu 0. `big` (both cards) was never measured: it evicts.
    run_to_success(&d, "coder", "fit-setup").await;
    write_gpus(&d, serde_json::json!([{"gpu": 0, "pid": 4101, "unit": unit("coder"), "memory_mib": 8000}]));
    write_gpu_free(&d, serde_json::json!({"0": 24000, "1": 32000}));
    let plan = plan_big().await;
    assert_eq!(plan.stop_targets, vec!["coder"]);
    assert!(plan.notes[0].contains("makes room (coder): never measured"), "{:?}", plan.notes);

    // It runs once (alone); what it holds then is remembered.
    write_gpus(&d, serde_json::json!([
        {"gpu": 0, "pid": 4201, "unit": unit("big"), "memory_mib": 6000},
        {"gpu": 1, "pid": 4202, "unit": unit("big"), "memory_mib": 6000}
    ]));
    run_to_success(&d, "big", "fit-measure").await;
    let vram = client.vram().await.unwrap();
    let rec = vram.records.iter().find(|r| r.workload == "big").expect("big was measured");
    assert_eq!(rec.per_card_mib, [(0u32, 6000u64), (1, 6000)].into_iter().collect());
    assert_eq!(vram.free_mib, Some([(0u32, 24000u64), (1, 32000)].into_iter().collect()));

    // Back to coder alone on gpu 0, with room for big on both cards.
    let resp = client.admit(admit_req(OpKind::Stop, "big", "fit-stop")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);
    write_gpus(&d, serde_json::json!([]));
    run_to_success(&d, "coder", "fit-coder").await;
    write_gpus(&d, serde_json::json!([{"gpu": 0, "pid": 4101, "unit": unit("coder"), "memory_mib": 8000}]));
    let plan = plan_big().await;
    assert!(plan.stop_targets.is_empty(), "{:?}", plan.stop_targets);
    assert!(plan.notes[0].starts_with("starts beside coder: fits: needs 5.9 GiB on gpu 0"), "{:?}", plan.notes);
    run_to_success(&d, "big", "fit-beside").await;
    let st = client.status().await.unwrap();
    assert!(running(&st, "coder") && running(&st, "big"));

    // One MiB short of the headroom on gpu 0: back to making room.
    let resp = client.admit(admit_req(OpKind::Stop, "big", "fit-stop-2")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);
    write_gpu_free(&d, serde_json::json!({"0": 7023, "1": 32000}));
    let plan = plan_big().await;
    assert_eq!(plan.stop_targets, vec!["coder"]);
    assert!(plan.notes[0].contains("does not fit"), "{:?}", plan.notes);

    // Cards that cannot be read are never "free".
    std::fs::remove_file(d.data_dir().join("gpu_free.json")).unwrap();
    let plan = plan_big().await;
    assert_eq!(plan.stop_targets, vec!["coder"]);
    assert!(plan.notes[0].contains("free VRAM could not be read"), "{:?}", plan.notes);

    // Measuring on request needs the workload on its cards.
    match client.vram_measure("helper").await {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, codes::INVALID_STATE, "{message}");
            assert!(message.contains("holds no VRAM"), "{message}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
    let measured = client.vram_measure("coder").await.unwrap().measured.expect("coder measured");
    assert_eq!(measured.per_card_mib, [(0u32, 8000u64)].into_iter().collect());
}

// ---- M4c: companion hooks -----------------------------------------------------

/// Hooks declared in the inventory run inside the operation, around their
/// workload's start and stop; a failed required hook fails the operation,
/// an optional one does not, and a rollback runs the undo.
#[tokio::test]
async fn companion_hooks_run_in_order_and_are_undone_on_rollback() {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    let log = dir.path().join("hooks.log");
    let say = |word: &str| serde_json::json!(["/bin/sh", "-c", format!("echo {word} >> {}", log.display())]);
    let host = "workstation";
    let wl = |name: &str, port: u16, hooks: serde_json::Value| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"},
            "hooks": hooks
        })
    };
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [
            wl("coder", 8100, serde_json::json!([
                {"name": "drain", "when": "pre_stop", "argv": say("drain")},
                {"name": "detach", "when": "post_stop", "argv": say("detach")}
            ])),
            wl("big", 8101, serde_json::json!([
                {"name": "free", "when": "pre_start", "argv": say("free"), "undo": say("undo-free")},
                {"name": "announce", "when": "post_start", "optional": true,
                 "argv": ["/bin/sh", "-c", format!("echo announce >> {}; echo boom >&2; exit 1", log.display())]}
            ])),
            wl("broken", 8102, serde_json::json!([
                {"name": "must", "when": "pre_start", "argv": ["/bin/sh", "-c", "echo cannot >&2; exit 7"]}
            ]))
        ],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();
    let lines = || std::fs::read_to_string(&log).unwrap_or_default().lines().map(str::to_owned).collect::<Vec<_>>();
    let running = |st: &llmctl_client::wire::StatusResponse, name: &str| {
        let w = st.workloads.iter().find(|w| w.name == name).unwrap();
        format!("{:?}", w.process).contains("Running")
    };

    run_to_success(&d, "coder", "hk-setup").await;
    assert!(lines().is_empty(), "coder has no start hooks");

    // big evicts coder, but its start fails: coder's stop hooks and big's
    // pre_start ran, then the rollback ran big's undo.
    d.write_faults(serde_json::json!({"fail_effects": [
        {"unit": "user/llmctl-next-fake-big.service", "kind": "start", "mode": "before"}
    ]}));
    let resp = client.admit(admit_req(OpKind::Run, "big", "hk-fail")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Failed, "{op:?}");
    assert_eq!(op.recovery.as_ref().expect("restoration")["outcome"], "restored");
    assert_eq!(lines(), ["drain", "detach", "free", "undo-free"]);
    assert!(running(&client.status().await.unwrap(), "coder"));

    // The same run succeeding: the optional post_start hook fails and is
    // only recorded, with the tail of what it printed.
    std::fs::remove_file(&log).unwrap();
    let resp = client.admit(admit_req(OpKind::Run, "big", "hk-ok")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    assert_eq!(lines(), ["drain", "detach", "free", "announce"]);
    let announce = op.phases.iter().find(|p| p.label.starts_with("hook announce")).expect("the hook is a phase");
    let detail = announce.detail.clone().unwrap_or_default();
    assert!(detail.contains("boom") && detail.contains("optional: ignored"), "{detail}");

    // A required hook that fails stops the operation before the start, and
    // the evicted workload comes back.
    let resp = client.admit(admit_req(OpKind::Run, "broken", "hk-broken")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Failed, "{op:?}");
    assert!(serde_json::to_string(&op).unwrap().contains("hook must failed"), "{op:?}");
    let st = client.status().await.unwrap();
    assert!(running(&st, "big") && !running(&st, "broken"));
}

// ---- M4c: model acquisition ----------------------------------------------------

/// Downloads go into declared stores as operations: the disk is checked
/// first, the tool does the fetching, a download runs beside a lifecycle
/// operation, and it can be cancelled.
#[tokio::test]
async fn weights_are_downloaded_into_a_store_with_a_disk_check_and_can_be_cancelled() {
    use llmctl_client::wire::AcquireRequest;
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().canonicalize().unwrap();
    let socket = root.join("control.sock");
    let ctl = root.join("fake-ctl");
    std::fs::create_dir_all(&ctl).unwrap();
    // A stand-in for `hf`: the dry run prints a size, the download writes a
    // 2 MB file where it was told to. Files in $FAKE_CTL steer it.
    let bin = root.join("fake-bin");
    std::fs::create_dir_all(&bin).unwrap();
    let hf = bin.join("hf");
    std::fs::write(
        &hf,
        r#"#!/bin/sh
dry=0; dest=""; prev=""
for a in "$@"; do
  [ "$a" = "--dry-run" ] && dry=1
  if [ "$prev" = "--cache-dir" ] || [ "$prev" = "--local-dir" ]; then dest="$a"; fi
  prev="$a"
done
if [ "$dry" = 1 ]; then
  size=2.0M; [ -f "$FAKE_CTL/size" ] && size=$(cat "$FAKE_CTL/size")
  echo "[{\"file\": \"model.safetensors\", \"size\": \"$size\"}]"; exit 0
fi
echo "Fetching 1 files"
[ -f "$FAKE_CTL/sleep" ] && sleep "$(cat "$FAKE_CTL/sleep")"
if [ -f "$FAKE_CTL/fail" ]; then echo "401 Unauthorized: token missing" >&2; exit 1; fi
mkdir -p "$dest" && head -c 2000000 /dev/zero > "$dest/model.safetensors" && echo "Downloaded to $dest"
"#,
    )
    .unwrap();
    let ollama = bin.join("ollama");
    std::fs::write(&ollama, "#!/bin/sh\necho \"$OLLAMA_HOST $*\" > \"$FAKE_CTL/ollama-call\"\necho success\n").unwrap();
    for f in [&hf, &ollama] {
        std::fs::set_permissions(f, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let host = "workstation";
    let wl = |name: &str, port: u16| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        })
    };
    let store = |name: &str, kind: &str, workload: Option<&str>| {
        let mut s = serde_json::json!({"name": name, "kind": kind, "path": root.join(name), "reserve_gib": 1});
        if let Some(w) = workload {
            s["workload"] = w.into();
        }
        s
    };
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [wl("coder", 8100), wl("svc", 8101)],
        "stores": [store("hub", "hf_cache", Some("coder")), store("models", "directory", None), store("ollama", "ollama", Some("svc"))],
        "revision": 1
    }});
    std::fs::write(root.join("inventory.json"), inv.to_string()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(&root).arg("daemon")
        .env("LLMCTL_NEXT_HF_BIN", &hf)
        .env("LLMCTL_NEXT_OLLAMA_BIN", &ollama)
        .env("FAKE_CTL", &ctl)
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();
    let req = |source: &str, store: &str| AcquireRequest {
        protocol: PROTOCOL_VERSION,
        source: source.into(),
        store: store.into(),
        for_workload: None,
        size_gib: None,
        dry_run: false,
        idempotency_key: None,
    };
    let refused = |r: Result<llmctl_client::wire::AcquireResponse, ClientError>, code_want: &str, why: &str| match r {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, code_want, "{message}");
            assert!(message.contains(why), "{message}");
        }
        other => panic!("expected {code_want} ({why}), got {other:?}"),
    };

    // The stores are listed with what is free on their filesystem.
    let stores = client.stores().await.unwrap();
    assert_eq!(stores.stores.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["hub", "models", "ollama"]);
    assert!(stores.stores.iter().all(|s| s.free_bytes.unwrap_or(0) > 0));

    // A preview checks the disk and names the command; nothing is fetched.
    let mut dry = req("hf:org/tiny@abc123", "hub");
    dry.dry_run = true;
    let p = client.acquire(&dry).await.unwrap();
    assert_eq!(p.need_bytes, Some(2_000_000));
    assert_eq!(p.command[1..], ["download", "org/tiny", "--revision", "abc123", "--cache-dir", p.path.as_str()]);
    assert!(p.operation.is_none() && !root.join("hub").exists());

    // Refusals before anything runs: not enough disk, an unknown size, a
    // source that does not belong in the store, a store with no workload.
    std::fs::write(ctl.join("size"), "9000000.0G").unwrap();
    refused(client.acquire(&req("hf:org/huge", "hub")).await, codes::INVALID_STATE, "not enough disk");
    std::fs::remove_file(ctl.join("size")).unwrap();
    refused(client.acquire(&req("ollama:qwen3:8b", "ollama")).await, codes::INVALID_STATE, "--size-gib");
    refused(client.acquire(&req("ollama:qwen3:8b", "hub")).await, codes::INVALID_REQUEST, "cannot go into store hub");
    refused(client.acquire(&req("hf:org/tiny", "models")).await, codes::INVALID_REQUEST, "--for");
    refused(client.acquire(&req("hf:--help/x", "hub")).await, codes::INVALID_REQUEST, "not a plain model name");
    refused(client.acquire(&req("hf:org/tiny", "nowhere")).await, codes::INVALID_REQUEST, "declared: hub, models, ollama");

    // The download: an operation filed under the store's workload.
    let r = client.acquire(&req("hf:org/tiny", "hub")).await.unwrap();
    let op = d.wait_terminal(&r.operation.expect("admitted").id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    assert_eq!(op.workload, "coder");
    assert_eq!(std::fs::metadata(root.join("hub/model.safetensors")).unwrap().len(), 2_000_000);
    let detail = op.phases[0].detail.clone().unwrap_or_default();
    assert!(detail.contains("hf:org/tiny is in") && detail.contains("Downloaded to"), "{detail}");

    // A directory store gets one directory per model, filed under --for.
    let mut into_models = req("hf:org/tiny", "models");
    into_models.for_workload = Some("coder".into());
    let r = client.acquire(&into_models).await.unwrap();
    assert_eq!(d.wait_terminal(&r.operation.unwrap().id).await.status, OpStatus::Succeeded);
    assert!(root.join("models/tiny/model.safetensors").exists());

    // An Ollama store pulls through its server, with a size given.
    let mut pull = req("ollama:qwen3:8b", "ollama");
    pull.size_gib = Some(0);
    let r = client.acquire(&pull).await.unwrap();
    let op = d.wait_terminal(&r.operation.unwrap().id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    assert_eq!(std::fs::read_to_string(ctl.join("ollama-call")).unwrap().trim(), "127.0.0.1:8101 pull qwen3:8b");

    // A tool that fails: the operation fails with what the tool said.
    std::fs::write(ctl.join("fail"), "1").unwrap();
    let r = client.acquire(&req("hf:org/tiny", "hub")).await.unwrap();
    let op = d.wait_terminal(&r.operation.unwrap().id).await;
    assert_eq!(op.status, OpStatus::Failed, "{op:?}");
    assert!(serde_json::to_string(&op).unwrap().contains("401 Unauthorized"), "{op:?}");
    std::fs::remove_file(ctl.join("fail")).unwrap();

    // A long download does not block a start, a second download waits its
    // turn, and a cancel stops the tool.
    std::fs::write(ctl.join("sleep"), "60").unwrap();
    let slow = client.acquire(&req("hf:org/tiny", "hub")).await.unwrap().operation.unwrap();
    run_to_success(&d, "coder", "acq-beside").await;
    refused(client.acquire(&req("hf:org/other", "hub")).await, codes::OVERLAPPING_OPERATION, "acquire");
    assert!(!client.operation(&slow.id).await.unwrap().status.is_terminal(), "still downloading");
    client.cancel(&slow.id, None).await.unwrap();
    let op = d.wait_terminal(&slow.id).await;
    assert_eq!(op.status, OpStatus::Canceled, "{op:?}");
}

/// LM Studio models come through `lms get` into an `lmstudio` store: an
/// exact catalog entry, a size given by the caller, and only while the LM
/// Studio service runs (`lms` would otherwise start one of its own).
#[tokio::test]
async fn lm_studio_models_are_fetched_through_lms_while_the_service_runs() {
    use llmctl_client::wire::AcquireRequest;
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().canonicalize().unwrap();
    let socket = root.join("control.sock");
    let ctl = root.join("fake-ctl");
    let bin = root.join("fake-bin");
    std::fs::create_dir_all(&ctl).unwrap();
    std::fs::create_dir_all(&bin).unwrap();
    // A stand-in for `lms`: records its arguments, draws spinner lines the
    // way the real one does without a terminal, then succeeds or fails.
    let lms = bin.join("lms");
    std::fs::write(
        &lms,
        "#!/bin/sh\necho \"$*\" > \"$FAKE_CTL/lms-call\"\n\
         printf '\\r   \\342\\240\\271 %s - Resolving...\\033[0K\\n\\r\\342\\240\\247 Resolving download plan...\\033[0K\\n' \"$2\"\n\
         if [ -f \"$FAKE_CTL/fail\" ]; then echo \"Error: Failed to resolve artifact \\\"$2\\\"\" >&2; exit 1; fi\n\
         echo 'Download completed.'\n",
    )
    .unwrap();
    std::fs::set_permissions(&lms, std::fs::Permissions::from_mode(0o755)).unwrap();
    let host = "workstation";
    let wl = |name: &str, port: u16| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        })
    };
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [wl("studio", 8100)],
        "stores": [
            {"name": "hub", "kind": "hf_cache", "path": root.join("hub"), "reserve_gib": 1, "workload": "studio"},
            {"name": "lmstudio", "kind": "lmstudio", "path": root.join("lmstudio"), "reserve_gib": 1, "workload": "studio"}
        ],
        "revision": 1
    }});
    std::fs::write(root.join("inventory.json"), inv.to_string()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(&root).arg("daemon")
        .env("LLMCTL_NEXT_LMS_BIN", &lms)
        .env("FAKE_CTL", &ctl)
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let client = d.client();
    let req = |source: &str, store: &str, size_gib: Option<u64>| AcquireRequest {
        protocol: PROTOCOL_VERSION,
        source: source.into(),
        store: store.into(),
        for_workload: None,
        size_gib,
        dry_run: false,
        idempotency_key: None,
    };
    let refused = |r: Result<llmctl_client::wire::AcquireResponse, ClientError>, code_want: &str, why: &str| match r {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, code_want, "{message}");
            assert!(message.contains(why), "{message}");
        }
        other => panic!("expected {code_want} ({why}), got {other:?}"),
    };
    let called = || std::fs::read_to_string(ctl.join("lms-call")).ok().map(|s| s.trim().to_string());

    assert!(client.stores().await.unwrap().stores.iter().any(|s| s.name == "lmstudio" && s.kind == "lmstudio"));

    // Each source only goes into its own kind of store, and an LM Studio
    // source is an exact catalog entry, never a search term.
    refused(client.acquire(&req("lms:pub/model", "hub", Some(0))).await, codes::INVALID_REQUEST, "cannot go into store hub");
    refused(client.acquire(&req("hf:org/tiny", "lmstudio", Some(0))).await, codes::INVALID_REQUEST, "kind lmstudio");
    refused(client.acquire(&req("lms:llama3.2", "lmstudio", Some(0))).await, codes::INVALID_REQUEST, "catalog entry");

    // The service is not running: nothing is run, so `lms` cannot start one.
    refused(client.acquire(&req("lms:pub/model", "lmstudio", Some(0))).await, codes::INVALID_STATE, "studio is not running");
    assert_eq!(called(), None);

    run_to_success(&d, "studio", "lms-studio-up").await;

    // `lms` cannot say a size in advance: the caller gives the bound.
    refused(client.acquire(&req("lms:pub/model", "lmstudio", None)).await, codes::INVALID_STATE, "--size-gib");
    refused(client.acquire(&req("lms:pub/model", "lmstudio", Some(9_000_000_000))).await, codes::INVALID_STATE, "not enough disk");

    // A preview names the command and runs nothing.
    let mut dry = req("lms:pub/model@q4_k_m", "lmstudio", Some(0));
    dry.dry_run = true;
    let p = client.acquire(&dry).await.unwrap();
    assert_eq!(p.command[1..], ["get", "pub/model@q4_k_m", "--yes"]);
    assert!(p.operation.is_none());
    assert_eq!(called(), None);

    // The download: one `lms get`, non-interactive, filed under the service.
    let r = client.acquire(&req("lms:pub/model@q4_k_m", "lmstudio", Some(0))).await.unwrap();
    let op = d.wait_terminal(&r.operation.expect("admitted").id).await;
    assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
    assert_eq!(op.workload, "studio");
    assert_eq!(called().as_deref(), Some("get pub/model@q4_k_m --yes"));
    let detail = op.phases[0].detail.clone().unwrap_or_default();
    assert!(detail.contains("lms:pub/model@q4_k_m is in") && detail.contains("Download completed."), "{detail}");
    assert!(!detail.contains("Resolving"), "spinner frames stay out of the journal: {detail}");

    // A model the catalog does not have: the operation fails with what
    // `lms` said.
    std::fs::write(ctl.join("fail"), "1").unwrap();
    let r = client.acquire(&req("lms:pub/absent", "lmstudio", Some(0))).await.unwrap();
    let op = d.wait_terminal(&r.operation.unwrap().id).await;
    assert_eq!(op.status, OpStatus::Failed, "{op:?}");
    let text = serde_json::to_string(&op).unwrap();
    assert!(text.contains("Failed to resolve artifact") && !text.contains("Resolving download plan"), "{op:?}");
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

/// `version --repin` records what an install reports as its pin and says
/// what changed. Here the install is one llmctl-next does not manage (the
/// ds4 case): the recipe names the binary, a declared extra file is pinned
/// with it, and the pin lives on the record.
#[test]
fn version_repin_records_the_pin_and_names_what_changed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("install");
    std::fs::create_dir(&root).unwrap();
    let engine = root.join("engine");
    let helper = dir.path().join("helper.sh");
    std::fs::write(&engine, "build one").unwrap();
    std::fs::write(&helper, "#!/bin/sh\n").unwrap();
    let host = "workstation";
    let wl = |name: &str, port: u16| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        })
    };
    let mut eng = wl("eng", 8100);
    eng["recipe"] = serde_json::json!({
        "kind": "source_build", "name": "eng", "root": root, "git": "https://example.invalid/eng.git",
        "build": "true", "binary": "engine"
    });
    eng["pin_files"] = serde_json::json!([helper]);
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [eng, wl("plain", 8101)],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = start_daemon_in(dir.path(), &[]);
    let socket = dir.path().join("control.sock");
    let run = |args: &[&str]| {
        let out = cli(&socket, args);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code(), text)
    };
    let json = |text: &str| -> serde_json::Value { serde_json::from_str(text).expect(text) };

    // No pin yet: nothing to compare, so no drift.
    let (code, text) = run(&["version", "eng"]);
    assert_eq!(code, Some(0), "{text}");
    let report = json(&text);
    assert!(report["pinned"].is_null(), "{text}");
    assert_eq!(report["live"]["fact"]["kind"], "source_build", "{text}");

    // The first pin.
    let (code, text) = run(&["version", "eng", "--repin"]);
    assert_eq!(code, Some(0), "{text}");
    let report = json(&text);
    assert_eq!(report["repinned"], true, "{text}");
    assert_eq!(report["complete"], true, "{text}");
    assert_eq!(report["inventory_revision"], 2, "{text}");
    assert_eq!(report["pinned"]["exec_digests"][0][0], helper.to_str().unwrap(), "{text}");
    let saved = std::fs::read_to_string(dir.path().join("inventory.json")).unwrap();
    assert!(saved.contains("observed_version"), "{saved}");
    let (code, text) = run(&["version", "eng"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(json(&text)["drift"].as_array().unwrap().is_empty(), "{text}");

    // The binary is rebuilt and the helper edited: both are findings.
    std::fs::write(&engine, "build two").unwrap();
    std::fs::write(&helper, "#!/bin/sh\n# edited\n").unwrap();
    let (code, text) = run(&["version", "eng"]);
    assert_eq!(code, Some(4), "drift is a finding: {text}");
    assert!(text.contains("version.fact") && text.contains("version.exec_digests"), "{text}");

    // Accepting them names each one; a second re-pin writes nothing.
    let (code, text) = run(&["version", "eng", "--repin"]);
    assert_eq!(code, Some(0), "{text}");
    let report = json(&text);
    let fields: Vec<&str> = report["changes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["field"].as_str().unwrap())
        .collect();
    assert_eq!(fields, vec!["version.exec_digests", "version.fact"], "{text}");
    assert_eq!(report["inventory_revision"], 3, "{text}");
    let (code, text) = run(&["version", "eng"]);
    assert_eq!(code, Some(0), "{text}");
    let (code, text) = run(&["version", "eng", "--repin"]);
    assert_eq!(code, Some(0), "{text}");
    let report = json(&text);
    assert_eq!(report["repinned"], false, "{text}");
    assert_eq!(report["inventory_revision"], 3, "{text}");

    // The pin survives a daemon restart.
    stop_daemon(child);
    let child = start_daemon_in(dir.path(), &[]);
    let (code, text) = run(&["version", "eng"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(!json(&text)["pinned"].is_null(), "{text}");

    // Without a recipe there is nothing to probe.
    let (code, text) = run(&["version", "plain", "--repin"]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("no recipe"), "{text}");
    stop_daemon(child);
}

/// A remote archive's binary runs on this host, so a version without a
/// declared sha256 is refused before anything is fetched: each release
/// has its own digest, and an upgrade must add its own.
#[test]
fn a_remote_archive_version_without_its_checksum_is_never_fetched() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("releases");
    let host = "workstation";
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [{
            "name": "eng", "engine": "fake", "model": "fake/eng",
            "gpus": [{"host": host, "index": 0}], "port": 8100,
            "unit": {"scope": "user", "name": "llmctl-next-fake-eng.service"},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": "llmctl-next-fake-eng.service"},
            // Staging is for managed workloads only.
            "ownership": {
                "kind": "managed",
                "definition": llmctl_core::LaunchDefinition::default(),
                "digest": "sha256:00",
                "adopted_at": "2026-01-01T00:00:00Z",
                "provenance": []
            },
            "recipe": {
                "kind": "release_archive", "name": "eng", "root": root,
                // Nothing listens here: a fetch attempt would fail differently.
                "url": "https://127.0.0.1:9/eng-{version}.tar.gz",
                "checksums": {"1.0": "sha256:00"},
                "binary": "eng"
            }
        }],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = start_daemon_in(dir.path(), &[]);
    let out = cli(&dir.path().join("control.sock"), &["install", "stage", "eng", "2.0"]);
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    stop_daemon(child);
    assert!(!out.status.success(), "{text}");
    assert!(text.contains("no sha256 is declared for eng 2.0"), "{text}");
    let fetched = std::fs::read_dir(&root)
        .map(|d| d.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".archive")))
        .unwrap_or(false);
    assert!(!fetched, "nothing was downloaded");
}

/// A restart whose start fails does not leave a serving workload down:
/// the rollback starts it again and checks it (seen live on m5, where a
/// failed restart of Ollama left it stopped). When that start fails too,
/// the operation says so and waits for the owner.
#[tokio::test]
async fn a_failed_restart_brings_the_workload_back_or_asks_for_attention() {
    let d = spawn_daemon();
    let client = d.client();
    let running = |st: &llmctl_client::wire::StatusResponse, name: &str| {
        let w = st.workloads.iter().find(|w| w.name == name).unwrap();
        format!("{:?}", w.process).contains("Running")
    };
    let start_fails = || serde_json::json!(
        {"unit": "user/llmctl-next-fake-coder.service", "kind": "start", "mode": "before"}
    );
    run_to_success(&d, "coder", "fr-setup").await;

    // One failing start: the restart's. The rollback's start is the next.
    d.write_faults(serde_json::json!({"fail_effects": [start_fails()]}));
    let resp = client.admit(admit_req(OpKind::Restart, "coder", "fr-once")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::Failed, "{op:?}");
    let recovery = op.recovery.expect("restoration recorded");
    assert_eq!(recovery["outcome"], "restored", "{recovery}");
    let applied: Vec<&str> = recovery["steps"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["outcome"] == "applied")
        .filter_map(|s| s["step"].as_str())
        .collect();
    assert!(applied.contains(&"start coder"), "{recovery}");
    assert!(running(&client.status().await.unwrap(), "coder"), "back up after the rollback");

    // Two failing starts: the rollback cannot restore it either.
    d.write_faults(serde_json::json!({"fail_effects": [start_fails(), start_fails()]}));
    let resp = client.admit(admit_req(OpKind::Restart, "coder", "fr-twice")).await.unwrap();
    let op = d.wait_terminal(&resp.operation.id).await;
    assert_eq!(op.status, OpStatus::NeedsAttention, "{op:?}");
    assert!(!running(&client.status().await.unwrap(), "coder"));
}

/// Recipes and records without editing the inventory file: `recipe set`
/// attaches a recipe and pins at once, `recipe clear` takes both away,
/// and `forget` removes a record after saying what stays on the host.
#[test]
fn recipes_are_set_and_cleared_and_a_record_is_forgotten_by_verbs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("install");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("engine"), "build one").unwrap();
    let helper = dir.path().join("helper.sh");
    std::fs::write(&helper, "#!/bin/sh\n").unwrap();
    let host = "workstation";
    let wl = |name: &str, port: u16| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        })
    };
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [wl("eng", 8100), wl("plain", 8101)],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = start_daemon_in(dir.path(), &[]);
    let socket = dir.path().join("control.sock");
    let run = |args: &[&str]| {
        let out = cli(&socket, args);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code(), text)
    };
    let json = |text: &str| -> serde_json::Value { serde_json::from_str(text).expect(text) };
    let recipe_file = dir.path().join("recipe.json");
    let write_recipe = |root: &str| {
        std::fs::write(
            &recipe_file,
            serde_json::json!({
                "kind": "source_build", "name": "eng", "root": root,
                "git": "https://example.invalid/eng.git", "build": "true", "binary": "engine"
            })
            .to_string(),
        )
        .unwrap();
    };
    let file = recipe_file.to_str().unwrap();

    // A recipe that fails validation changes nothing.
    write_recipe("relative/root");
    let (code, text) = run(&["recipe", "set", "eng", "--file", file]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("root must be absolute"), "{text}");
    let (_, text) = run(&["recipe", "show", "eng"]);
    assert!(json(&text)["recipe"].is_null(), "{text}");

    // Set: attached and pinned in one step, extra file included.
    write_recipe(root.to_str().unwrap());
    let (code, text) = run(&["recipe", "set", "eng", "--file", file, "--pin-file", helper.to_str().unwrap()]);
    assert_eq!(code, Some(0), "{text}");
    let report = json(&text);
    assert_eq!(report["changed"], true, "{text}");
    assert_eq!(report["complete"], true, "{text}");
    assert_eq!(report["pinned"]["fact"]["kind"], "source_build", "{text}");
    assert_eq!(report["pinned"]["exec_digests"][0][0], helper.to_str().unwrap(), "{text}");
    let (code, text) = run(&["version", "eng"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(!json(&text)["pinned"].is_null(), "{text}");
    let (_, text) = run(&["recipe", "show", "eng"]);
    let shown = json(&text);
    assert_eq!(shown["recipe"]["binary"], "engine", "{text}");
    assert_eq!(shown["pin_files"][0], helper.to_str().unwrap(), "{text}");

    // Set again without --pin-file keeps the list; --no-pin-files drops it.
    let (code, text) = run(&["recipe", "set", "eng", "--file", file]);
    assert_eq!(code, Some(0), "{text}");
    assert_eq!(json(&text)["pin_files"].as_array().unwrap().len(), 1, "{text}");
    let (code, text) = run(&["recipe", "set", "eng", "--file", file, "--no-pin-files"]);
    assert_eq!(code, Some(0), "{text}");
    assert_eq!(json(&text)["pin_files"].as_array().unwrap().len(), 0, "{text}");

    // Clear: recipe and pin are gone; clearing twice is a no-op.
    let (code, text) = run(&["recipe", "clear", "eng"]);
    assert_eq!(code, Some(0), "{text}");
    let (code, text) = run(&["version", "eng"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("no recipe"), "{text}");
    let (code, text) = run(&["recipe", "clear", "eng"]);
    assert_eq!(code, Some(0), "{text}");
    assert_eq!(json(&text)["changed"], false, "{text}");

    // Forget: a preview first, then the record goes; the file follows.
    let (code, text) = run(&["forget", "plain"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("would forget plain") && text.contains("--yes"), "{text}");
    let (_, text) = run(&["status"]);
    assert!(text.contains("plain"), "a preview removes nothing: {text}");
    let (code, text) = run(&["forget", "plain", "--yes"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("forgot plain"), "{text}");
    let (_, text) = run(&["status"]);
    assert!(!text.contains("plain") && text.contains("eng"), "{text}");
    let saved = std::fs::read_to_string(dir.path().join("inventory.json")).unwrap();
    assert!(!saved.contains("llmctl-next-fake-plain"), "{saved}");
    let (code, text) = run(&["forget", "plain", "--yes"]);
    assert_eq!(code, Some(5), "an unknown workload: {text}");
    stop_daemon(child);
}

/// Stores are declared by verb, and weights leave a store only through a
/// preview and its token, never while a workload still names them.
#[test]
fn stores_are_declared_and_weights_removed_only_when_nothing_names_them() {
    let dir = tempfile::tempdir().expect("tempdir");
    let hub = dir.path().join("hub");
    for repo in ["models--org--tiny", "models--org--used"] {
        let blobs = hub.join(repo).join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::write(blobs.join("weights.bin"), vec![7u8; 4096]).unwrap();
    }
    let host = "workstation";
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [{
            // This workload serves org/used: those weights are in use.
            "name": "eng", "engine": "fake", "model": "org/used",
            "gpus": [{"host": host, "index": 0}], "port": 8100,
            "unit": {"scope": "user", "name": "llmctl-next-fake-eng.service"},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": "llmctl-next-fake-eng.service"},
            "ownership": {"kind": "external"}
        }],
        "revision": 1
    }});
    std::fs::write(dir.path().join("inventory.json"), inv.to_string()).unwrap();
    let child = start_daemon_in(dir.path(), &[]);
    let socket = dir.path().join("control.sock");
    let run = |args: &[&str]| {
        let out = cli(&socket, args);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        (out.status.code(), text)
    };
    let hub_s = hub.to_str().unwrap();

    // Declaring: a bad kind, a service store without its service and a
    // second store of the same name are refused.
    let (code, text) = run(&["stores", "add", "hub", "--kind", "bucket", "--path", hub_s]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("hf_cache, directory, ollama or lmstudio"), "{text}");
    let (code, text) = run(&["stores", "add", "oll", "--kind", "ollama", "--path", hub_s]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("must name the Ollama workload"), "{text}");
    let (code, text) = run(&["stores", "add", "hub", "--kind", "hf_cache", "--path", hub_s, "--reserve-gib", "1"]);
    assert_eq!(code, Some(0), "{text}");
    let (code, text) = run(&["stores", "add", "hub", "--kind", "hf_cache", "--path", hub_s]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("already declared"), "{text}");
    let (_, text) = run(&["stores"]);
    assert!(text.contains("hub") && text.contains("hf_cache"), "{text}");

    // In use: named, listed, not removable even with a token.
    let (code, text) = run(&["model", "rm", "hf:org/used", "--store", "hub"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(text.contains("still referenced") && text.contains("workload eng serves it"), "{text}");
    let (code, text) = run(&["model", "rm", "hf:org/used", "--store", "hub", "--confirm", "0000000000000000"]);
    assert_eq!(code, Some(4), "{text}");
    assert!(hub.join("models--org--used").exists());

    // Not in use: a preview with a token, and nothing is gone yet.
    let (code, text) = run(&["model", "rm", "hf:org/tiny", "--store", "hub"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("nothing references it") && text.contains("--confirm "), "{text}");
    assert!(hub.join("models--org--tiny").exists(), "a preview removes nothing");
    let token = text.split("--confirm ").nth(1).unwrap().split_whitespace().next().unwrap().to_string();
    let (code, text) = run(&["model", "rm", "hf:org/tiny", "--store", "hub", "--confirm", "ffffffffffffffff"]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("stale confirmation"), "{text}");
    assert!(hub.join("models--org--tiny").exists());
    let (code, text) = run(&["model", "rm", "hf:org/tiny", "--store", "hub", "--confirm", &token]);
    assert_eq!(code, Some(0), "{text}");
    assert!(text.contains("removed"), "{text}");
    assert!(!hub.join("models--org--tiny").exists());
    assert!(hub.join("models--org--used").exists(), "only the named weights went");
    let (code, text) = run(&["model", "rm", "hf:org/tiny", "--store", "hub"]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("is not in store hub"), "{text}");

    // A path cannot leave the store.
    let (code, text) = run(&["model", "rm", "path:../inventory.json", "--store", "hub"]);
    assert_ne!(code, Some(0), "{text}");
    assert!(text.contains("not a plain relative path"), "{text}");
    assert!(dir.path().join("inventory.json").exists());

    // Forgetting the store leaves its weights.
    let (code, text) = run(&["stores", "remove", "hub"]);
    assert_eq!(code, Some(0), "{text}");
    assert!(hub.join("models--org--used").exists());
    let (_, text) = run(&["stores"]);
    assert!(text.contains("no store declared"), "{text}");
    stop_daemon(child);
}

// ---- M5a: the foreign-write record ------------------------------------------

#[tokio::test]
async fn idempotency_refuses_a_changed_profile_or_force_flag() {
    let d = spawn_daemon();
    let request = admit_req(OpKind::Run, "coder", "payload-identity");
    let first = d.client().admit(request.clone()).await.unwrap();
    assert_eq!(
        d.wait_terminal(&first.operation.id).await.status,
        OpStatus::Succeeded
    );
    let replay = d.client().admit(request.clone()).await.unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.operation.id, first.operation.id);

    let mut changed = request.clone();
    changed.profile = Some(llmctl_client::wire::ProfileRequest {
        env_source: "/tmp/different-profile".into(),
        model: "different-model".into(),
    });
    let mut forced = request;
    forced.force = true;
    for request in [changed, forced] {
        match d.client().admit(request).await {
            Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::IDEMPOTENCY_CONFLICT),
            other => panic!("a changed payload must conflict, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_claimed_caller_lock_never_substitutes_for_an_executor_lock() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let lock_path = dir.path().join("operations.lock");
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir.path())
        .arg("daemon")
        .env("LLMCTL_NEXT_OP_LOCK", &lock_path)
        .env("LLMCTL_NEXT_LANE_LEASE", "off")
        .env("LLMCTL_NEXT_OP_LOCK_WAIT_MS", "100")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    let holder = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)
        .unwrap();
    assert!(try_flock(&holder));
    let request = admit_req(OpKind::Run, "coder", "caller-lock");
    match d
        .client()
        .holding_lock(std::process::id())
        .admit(request.clone())
        .await
    {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::HOST_LOCK_BUSY),
        other => panic!("a borrowed lock must not authorize execution: {other:?}"),
    }
    drop(holder);
    d.write_faults(serde_json::json!({"effect_delay_ms": 1500}));
    let mut caller = ChildGuard(Some(
        Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
            .arg("--socket")
            .arg(&d.socket)
            .args(["run", "coder", "--key", "caller-lock", "--wait"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    ));
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let operation = loop {
        if let Ok(op) = d.client().operation_by_key("caller-lock").await {
            break op;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "caller never admitted"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(!operation.status.is_terminal());
    caller.kill_now();
    let other = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock_path)
        .unwrap();
    assert!(
        !try_flock(&other),
        "caller death must not release the executor's lock"
    );
    assert_eq!(
        d.wait_terminal(&operation.id).await.status,
        OpStatus::Succeeded
    );
}

#[tokio::test]
async fn request_identity_survives_daemon_restart() {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("control.sock");
    let mut daemon = ChildGuard(Some(start_daemon_in(dir.path(), &[])));
    let request = admit_req(OpKind::Run, "coder", "durable-identity");
    let client = ControlClient::connect(&socket).unwrap();
    let admitted = client.admit(request.clone()).await.unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        if client
            .operation(&admitted.operation.id)
            .await
            .unwrap()
            .status
            .is_terminal()
        {
            break;
        }
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    daemon.kill_now();
    let _daemon = ChildGuard(Some(start_daemon_in(dir.path(), &[])));
    let client = ControlClient::connect(&socket).unwrap();
    let replay = client.admit(request.clone()).await.unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.operation.id, admitted.operation.id);
    let mut changed = request;
    changed.force = true;
    match client.admit(changed).await {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::IDEMPOTENCY_CONFLICT),
        other => panic!("changed payload after restart: {other:?}"),
    }
}

#[tokio::test]
async fn a_completed_start_does_not_explain_a_foreign_stop_before_the_next_sample() {
    let d = spawn_managed().await;
    let client = d.client();
    client.status().await.unwrap();
    run_to_success(&d, "coder", "audit-first-start").await;
    client.status().await.unwrap();
    let restarted = client
        .admit(admit_req(OpKind::Restart, "coder", "audit-restart"))
        .await
        .unwrap();
    assert_eq!(
        d.wait_terminal(&restarted.operation.id).await.status,
        OpStatus::Succeeded
    );
    foreign_flip(
        &d,
        "coder",
        serde_json::json!({
            "active": false, "pid": null, "ready_at": null, "model": null
        }),
    );
    client.status().await.unwrap();
    assert_eq!(client.audit(None).await.unwrap().findings.len(), 1);
}

#[tokio::test]
async fn acknowledging_an_old_effect_does_not_explain_a_new_foreign_transition() {
    let d = spawn_managed().await;
    let client = d.client();
    client.status().await.unwrap();
    run_to_success(&d, "coder", "audit-ack-start").await;
    client.status().await.unwrap();
    let fault = serde_json::json!({
        "unit": "user/llmctl-next-fake-coder.service", "kind": "start", "mode": "before"
    });
    d.write_faults(serde_json::json!({"fail_effects": [fault.clone(), fault]}));
    let failed = client
        .admit(admit_req(OpKind::Restart, "coder", "audit-ack-failure"))
        .await
        .unwrap();
    assert_eq!(
        d.wait_terminal(&failed.operation.id).await.status,
        OpStatus::NeedsAttention
    );
    client.status().await.unwrap();
    assert!(client.audit(None).await.unwrap().findings.is_empty());
    foreign_flip(
        &d,
        "coder",
        serde_json::json!({
            "active": true, "pid": 49999, "ready_at": "2020-01-01T00:00:00Z", "model": "fake/coder"
        }),
    );
    client.status().await.unwrap();
    assert_eq!(client.audit(None).await.unwrap().findings.len(), 1);
    client
        .acknowledge(&failed.operation.id, None)
        .await
        .unwrap();
    foreign_flip(
        &d,
        "coder",
        serde_json::json!({
            "active": false, "pid": null, "ready_at": null, "model": null
        }),
    );
    client.status().await.unwrap();
    assert_eq!(
        client.audit(None).await.unwrap().findings.len(),
        2,
        "acknowledgement must not renew old effect evidence"
    );
}

#[tokio::test]
async fn audit_preserves_its_baseline_until_a_restart_finishes() {
    let d = spawn_managed().await;
    let client = d.client();
    client.status().await.unwrap();
    run_to_success(&d, "coder", "audit-overlap-start").await;
    client.status().await.unwrap();
    d.write_faults(serde_json::json!({"effect_delay_ms": 300}));
    let restarted = client
        .admit(admit_req(OpKind::Restart, "coder", "audit-overlap"))
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !client
        .operation(&restarted.operation.id)
        .await
        .unwrap()
        .status
        .is_terminal()
    {
        assert!(tokio::time::Instant::now() < deadline);
        client.status().await.unwrap();
        assert!(client.audit(None).await.unwrap().findings.is_empty());
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    client.status().await.unwrap();
    assert!(client.audit(None).await.unwrap().findings.is_empty());
}

#[tokio::test]
async fn a_noop_run_does_not_explain_a_foreign_stop() {
    let d = spawn_managed().await;
    let client = d.client();
    client.status().await.unwrap();
    run_to_success(&d, "coder", "audit-start").await;
    client.status().await.unwrap();
    assert!(client.audit(None).await.unwrap().findings.is_empty());
    run_to_success(&d, "coder", "audit-noop").await;
    foreign_flip(
        &d,
        "coder",
        serde_json::json!({
            "active": false, "pid": null, "ready_at": null, "model": null
        }),
    );
    client.status().await.unwrap();
    let findings = client.audit(None).await.unwrap().findings;
    assert_eq!(findings.len(), 1, "a no-op run cannot explain a stop");
    assert_eq!(findings[0].workload, "coder");
}

/// One fake workload, adopted (managed) + one external sibling — the
/// audit compares managed workloads only.
fn write_managed_inventory(dir: &Path) {
    let host = "workstation";
    let definition = llmctl_core::LaunchDefinition::default();
    let wl = |name: &str, port: u16| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        })
    };
    let mut coder = wl("coder", 8100);
    coder["ownership"] = serde_json::json!({
        "kind": "managed", "definition": definition, "digest": "sha256:0",
        "adopted_at": "2026-11-18T00:00:00Z", "provenance": []
    });
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [coder, wl("helper", 8101)],
        "revision": 1
    }});
    std::fs::write(dir.join("inventory.json"), inv.to_string()).unwrap();
}

async fn spawn_managed() -> DaemonGuard {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    write_managed_inventory(dir.path());
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let d = DaemonGuard { child, dir, socket };
    wait_socket(&d.socket).await;
    d
}

/// Flip the fake world behind the agent's back: the one thing no admitted
/// operation explains.
fn foreign_flip(d: &DaemonGuard, name: &str, patch: serde_json::Value) {
    foreign_flip_dir(d.data_dir(), name, patch);
}

fn foreign_flip_dir(dir: &Path, name: &str, patch: serde_json::Value) {
    let path = dir.join("world.json");
    let mut world: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let unit = world["units"]
        .as_object_mut().unwrap()
        .entry(format!("user/llmctl-next-fake-{name}.service"))
        .or_insert(serde_json::json!({}));
    if let Some(obj) = unit.as_object_mut() {
        for (k, v) in patch.as_object().unwrap() {
            obj.insert(k.clone(), v.clone());
        }
    }
    std::fs::write(&path, world.to_string()).unwrap();
}

/// A unit flipped behind the agent's back is a finding; the same flip
/// through an admitted operation is not (M5a tests, plan decision 1).
#[tokio::test]
async fn a_foreign_flip_is_a_finding_and_an_operated_flip_is_not() {
    let d = spawn_managed().await;
    let client = d.client();

    // Baseline: both stopped.
    let st = client.status().await.unwrap();
    assert_eq!(st.workloads.len(), 2);

    // Through the front door: an admitted run explains its own transition.
    run_to_success(&d, "coder", "audit-op-1").await;
    client.status().await.unwrap();
    let audit = client.audit(None).await.unwrap();
    assert!(audit.findings.is_empty(), "{:?}", audit.findings);

    // Behind the agent's back: the unit goes away with no operation.
    foreign_flip(&d, "coder", serde_json::json!({
        "active": false, "pid": null, "ready_at": null, "model": null
    }));
    client.status().await.unwrap();
    let audit = client.audit(None).await.unwrap();
    assert_eq!(audit.findings.len(), 1, "{:?}", audit.findings);
    let f = &audit.findings[0];
    assert_eq!(f.workload, "coder");
    assert_eq!(f.kind, "foreign_transition");
    assert_eq!(f.what, "unit");
    assert!(f.before_state.contains("\"running\""), "{}", f.before_state);
    assert!(f.after_state.contains("\"stopped\""), "{}", f.after_state);
    assert!(f.actor.is_none());

    // Restarted from outside: pid change while running is a finding too.
    foreign_flip(&d, "coder", serde_json::json!({
        "active": true, "pid": 49999,
        "ready_at": "2020-01-01T00:00:00Z", "model": "fake/coder"
    }));
    client.status().await.unwrap();
    let audit = client.audit(None).await.unwrap();
    assert_eq!(audit.findings.len(), 2, "{:?}", audit.findings);
    assert!(audit.findings[0].after_state.contains("49999"), "{:?}", audit.findings[0]);

    // An external workload's flips are not llmctl-next's findings.
    foreign_flip(&d, "helper", serde_json::json!({"active": true, "pid": 50000,
        "ready_at": "2020-01-01T00:00:00Z", "model": "fake/helper"}));
    client.status().await.unwrap();
    let audit = client.audit(None).await.unwrap();
    assert_eq!(audit.findings.len(), 2, "{:?}", audit.findings);

    // The CLI verb reads the same record, filtered and unfiltered.
    let out = cli(&d.socket, &["audit", "--json"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("audit --json parses");
    assert_eq!(parsed["findings"].as_array().unwrap().len(), 2);
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let out = cli(&d.socket, &["audit", "--since", &today]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("coder") && text.contains("2 foreign write(s)"), "{text}");
}

/// Findings live in the journal DB: an agent restart keeps them and does
/// not re-record what it already knows (M5a tests).
#[tokio::test]
async fn agent_restart_keeps_findings_without_duplicating() {
    let dir = tempfile::tempdir().expect("tempdir");
    let socket = dir.path().join("control.sock");
    write_managed_inventory(dir.path());
    let spawn_on = || {
        Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
            .arg("--socket").arg(&socket).arg("--data-dir").arg(dir.path()).arg("daemon")
            .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon")
    };
    let mut child = spawn_on();
    wait_socket(&socket).await;
    let probe = |f: fn(&std::path::Path)| {} ;
    let _ = probe;

    async fn with_client<T>(socket: &Path, f: impl std::future::Future<Output = T>) -> T {
        let client = ControlClient::connect(socket).expect("connect");
        f.await
    }
    let _ = &child;

    // Baseline, an explained run, then a foreign stop.
    let client = ControlClient::connect(&socket).unwrap();
    client.status().await.unwrap(); // baseline
    let resp = client
        .admit(admit_req(OpKind::Run, "coder", "audit-restart-1"))
        .await
        .unwrap();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        let op = client.operation(&resp.operation.id).await.unwrap();
        if op.status.is_terminal() {
            assert_eq!(op.status, OpStatus::Succeeded, "{op:?}");
            break;
        }
        assert!(tokio::time::Instant::now() < deadline, "run never terminal");
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    client.status().await.unwrap(); // explained: no finding
    foreign_flip_dir(dir.path(), "coder", serde_json::json!({
        "active": false, "pid": null, "ready_at": null, "model": null
    }));
    client.status().await.unwrap();
    let audit = client.audit(None).await.unwrap();
    assert_eq!(audit.findings.len(), 1, "{:?}", audit.findings);

    // Restart the agent on the same data dir.
    child.kill().unwrap();
    child.wait().unwrap();
    std::thread::sleep(Duration::from_millis(200));
    let mut child2 = spawn_on();
    wait_socket(&socket).await;
    let client = ControlClient::connect(&socket).unwrap();

    // The finding survived, and the first pass after restart (baseline was
    // persisted too) records nothing new.
    client.status().await.unwrap();
    let audit = client.audit(None).await.unwrap();
    assert_eq!(audit.findings.len(), 1, "{:?}", audit.findings);
    assert_eq!(audit.findings[0].workload, "coder");
    child2.kill().unwrap();
    child2.wait().unwrap();
}

// ---- M5b: tolerated GPU occupants ---------------------------------------------

/// A small known process (a display encoder, a speech server) can be
/// tolerated on a card: by unit identity, or by a command-line substring
/// when it has no unit (Docker). Its VRAM cap is the limit: above it the
/// process counts as foreign again. The list lives in the inventory and
/// changes through `tolerate add/remove`.
#[tokio::test]
async fn tolerated_gpu_occupants_let_a_run_through_until_one_outgrows_its_cap() {
    let d = spawn_daemon();
    let occupants = |speaches_mib: u64| {
        serde_json::json!([
            {"gpu": 0, "pid": 501, "unit": null, "memory_mib": speaches_mib,
             "cmdline": "/home/ubuntu/speaches/.venv/bin/python uvicorn --factory speaches.main:create_app"},
            {"gpu": 0, "pid": 502, "unit": "system/rustdesk.service", "memory_mib": 614}
        ])
    };
    write_gpus(&d, occupants(784));
    let client = d.client();
    match client.admit(admit_req(OpKind::Run, "coder", "tol-1")).await {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::GPU_OCCUPIED),
        other => panic!("expected gpu_occupied before any tolerance, got {other:?}"),
    }

    // Bad requests: neither or both matchers, a zero cap.
    for bad in [
        vec!["tolerate", "add", "x", "--max-mib", "100"],
        vec!["tolerate", "add", "x", "--unit", "system/a.service", "--cmdline", "a", "--max-mib", "100"],
        vec!["tolerate", "add", "x", "--cmdline", "a", "--max-mib", "0"],
    ] {
        let out = cli(&d.socket, &bad);
        assert!(!out.status.success(), "{bad:?} should be refused");
    }

    let out = cli(&d.socket, &["tolerate", "add", "speaches", "--cmdline", "speaches.main", "--max-mib", "2048"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = cli(&d.socket, &["tolerate", "add", "rustdesk", "--unit", "system/rustdesk.service", "--max-mib", "1024"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = cli(&d.socket, &["tolerate", "add", "rustdesk", "--unit", "system/other.service", "--max-mib", "1"]);
    assert!(!out.status.success(), "a duplicate name is refused");

    let listed = String::from_utf8(cli(&d.socket, &["tolerate"]).stdout).unwrap();
    assert!(listed.contains("speaches") && listed.contains("speaches.main"), "{listed}");
    assert!(listed.contains("rustdesk") && listed.contains("system/rustdesk.service"), "{listed}");

    // Both occupants tolerated → the run goes through.
    run_to_success(&d, "coder", "tol-2").await;

    // speaches grows past its cap → foreign again, and the refusal says why.
    let resp = client.admit(admit_req(OpKind::Stop, "coder", "tol-3")).await.unwrap();
    assert_eq!(d.wait_terminal(&resp.operation.id).await.status, OpStatus::Succeeded);
    write_gpus(&d, occupants(4096));
    match client.admit(admit_req(OpKind::Run, "coder", "tol-4")).await {
        Err(ClientError::Api { code, message, .. }) => {
            assert_eq!(code, codes::GPU_OCCUPIED, "{message}");
            assert!(message.contains("501") && message.contains("2048"), "{message}");
        }
        other => panic!("expected gpu_occupied over the cap, got {other:?}"),
    }

    // Removing a tolerance makes its process foreign again.
    write_gpus(&d, occupants(784));
    let out = cli(&d.socket, &["tolerate", "remove", "speaches"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    match client.admit(admit_req(OpKind::Run, "coder", "tol-5")).await {
        Err(ClientError::Api { code, .. }) => assert_eq!(code, codes::GPU_OCCUPIED),
        other => panic!("expected gpu_occupied after the removal, got {other:?}"),
    }
    let listed = String::from_utf8(cli(&d.socket, &["tolerate"]).stdout).unwrap();
    assert!(!listed.contains("speaches.main") && listed.contains("rustdesk"), "{listed}");
}
