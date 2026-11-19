//! State tests against a fake agent: the real worker, the real loop, a
//! test terminal. Confirm/cancel paths, a stale plan refused, a sleeping
//! host shown STALE from the status cache, logs following.

mod common;

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use llmctl_client::wire::{OpStatus, PhaseStatus};
use llmctl_core::OpKind;
use llmctl_tui::app::{Modal, Screen};
use llmctl_tui::fleet::HostTarget;
use llmctl_tui::{Options, Runner};
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;

fn key(c: char) -> Option<KeyEvent> {
    Some(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
}

fn code(c: KeyCode) -> Option<KeyEvent> {
    Some(KeyEvent::new(c, KeyModifiers::NONE))
}

fn runner(targets: Vec<HostTarget>, read_only: bool, timeout: Duration) -> Runner<TestBackend> {
    let term = Terminal::new(TestBackend::new(132, 40)).unwrap();
    Runner::new(targets, &Options { read_only, timeout }, term)
}

/// Step the loop until `pred` holds (or fail after `limit`).
async fn settle(r: &mut Runner<TestBackend>, limit: Duration, what: &str, pred: impl Fn(&Runner<TestBackend>) -> bool) {
    let start = Instant::now();
    loop {
        r.step(None).unwrap();
        if pred(r) {
            return;
        }
        assert!(start.elapsed() < limit, "timed out waiting for: {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn screen_text(r: &Runner<TestBackend>) -> String {
    let buf = r.terminal.backend().buffer();
    let mut out = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            out.push_str(buf[(x, y)].symbol());
        }
        out.push('\n');
    }
    out
}

fn ws() -> Arc<Fake> {
    Arc::new(Fake::new("workstation", workstation_status()))
}

const T: Duration = Duration::from_secs(5);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancel_asks_first_and_no_sends_nothing() {
    let fake = ws();
    fake.ops.lock().unwrap().push(op("op-1", OpKind::Restart, "ollama-gpu1", OpStatus::Executing, &[("stop", PhaseStatus::Running)]));
    let mut r = runner(vec![target(fake.clone(), true)], false, T);
    settle(&mut r, T, "hosts", |r| !r.app.hosts.is_empty()).await;
    r.step(key('o')).unwrap();
    settle(&mut r, T, "ops", |r| !r.app.ops.is_empty()).await;

    r.step(key('C')).unwrap();
    assert!(matches!(r.app.modal, Some(Modal::Confirm { .. })));
    assert!(screen_text(&r).contains("Cancel operation op-1 on workstation?"));
    r.step(key('n')).unwrap();
    assert!(r.app.modal.is_none());
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!fake.calls().iter().any(|c| c.starts_with("cancel")), "{:?}", fake.calls());

    r.step(key('C')).unwrap();
    r.step(key('y')).unwrap();
    settle(&mut r, T, "cancel answered", |r| {
        r.app.message.as_deref().is_some_and(|m| m.contains("cancellation requested"))
    })
    .await;
    assert_eq!(fake.calls().iter().filter(|c| c.as_str() == "cancel op-1").count(), 1);
    assert_eq!(r.app.ops[0].1.status, OpStatus::CancelRequested);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acknowledge_asks_first_and_only_for_terminal_unacknowledged_ops() {
    let fake = ws();
    fake.ops.lock().unwrap().push(op("op-9", OpKind::Run, "vllm-engine", OpStatus::NeedsAttention, &[("start", PhaseStatus::Failed)]));
    fake.ops.lock().unwrap().push(op("op-8", OpKind::Run, "vllm-engine", OpStatus::Executing, &[("start", PhaseStatus::Running)]));
    let mut r = runner(vec![target(fake.clone(), true)], false, T);
    settle(&mut r, T, "hosts", |r| !r.app.hosts.is_empty()).await;
    r.step(key('o')).unwrap();
    settle(&mut r, T, "ops", |r| r.app.ops.len() == 2).await;

    // Running ops sort first: op-8 is selected; it cannot be acknowledged.
    assert_eq!(r.app.ops[0].1.id, "op-8");
    r.step(key('A')).unwrap();
    assert!(r.app.modal.is_none());
    assert!(r.app.message.as_deref().unwrap().contains("nothing to acknowledge"));

    r.step(code(KeyCode::Down)).unwrap();
    r.step(key('A')).unwrap();
    assert!(matches!(r.app.modal, Some(Modal::Confirm { .. })));
    r.step(code(KeyCode::Esc)).unwrap();
    assert!(r.app.modal.is_none());
    r.step(key('A')).unwrap();
    r.step(key('y')).unwrap();
    settle(&mut r, T, "ack answered", |r| r.app.message.as_deref().is_some_and(|m| m.starts_with("acknowledged"))).await;
    assert_eq!(fake.calls().iter().filter(|c| c.starts_with("ack")).cloned().collect::<Vec<_>>(), vec!["ack op-9"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_only_never_offers_a_write() {
    let fake = ws();
    fake.ops.lock().unwrap().push(op("op-1", OpKind::Restart, "ollama-gpu1", OpStatus::Executing, &[]));
    let mut r = runner(vec![target(fake.clone(), true)], true, T);
    settle(&mut r, T, "hosts", |r| !r.app.hosts.is_empty()).await;
    r.step(key('o')).unwrap();
    settle(&mut r, T, "ops", |r| !r.app.ops.is_empty()).await;
    r.step(key('C')).unwrap();
    assert!(r.app.modal.is_none());
    r.step(key('w')).unwrap();
    r.step(code(KeyCode::Enter)).unwrap();
    assert!(r.app.modal.is_none());
    assert!(r.app.message.as_deref().unwrap().contains("read-only"));
    tokio::time::sleep(Duration::from_millis(50)).await;
    let calls = fake.calls();
    assert!(!calls.iter().any(|c| c.starts_with("cancel") || c.starts_with("plan") || c.starts_with("admit")), "{calls:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_previews_then_admits_exactly_that_plan_and_follows_it() {
    let fake = ws();
    let mut r = runner(vec![target(fake.clone(), true)], false, T);
    settle(&mut r, T, "hosts", |r| !r.app.hosts.is_empty()).await;
    r.step(key('w')).unwrap();
    assert_eq!(r.app.screen, Screen::Run);

    // Preview, then decline: nothing admitted.
    r.step(code(KeyCode::Enter)).unwrap();
    settle(&mut r, T, "plan", |r| matches!(&r.app.modal, Some(Modal::Preview { plan: Some(Ok(_)), .. }))).await;
    assert!(screen_text(&r).contains("STOP TARGETS: ollama-gpu1"));
    r.step(key('n')).unwrap();
    assert!(r.app.modal.is_none());
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!fake.calls().iter().any(|c| c.starts_with("admit")));

    // Preview, confirm: one admit with the previewed revision and stops.
    r.step(code(KeyCode::Enter)).unwrap();
    settle(&mut r, T, "plan", |r| matches!(&r.app.modal, Some(Modal::Preview { plan: Some(Ok(_)), .. }))).await;
    r.step(key('y')).unwrap();
    assert!(matches!(r.app.modal, Some(Modal::Progress { .. })));
    settle(&mut r, T, "op done", |r| {
        matches!(&r.app.modal, Some(Modal::Progress { op: Some(o), .. }) if o.status == OpStatus::Succeeded)
    })
    .await;
    let admits: Vec<String> = fake.calls().into_iter().filter(|c| c.starts_with("admit")).collect();
    assert_eq!(admits, vec![r#"admit run vllm-engine rev=Some(42) stops=Some(["ollama-gpu1"])"#.to_string()]);
    // The protocol check ran before the admit, as in the CLI.
    let calls = fake.calls();
    let admit_at = calls.iter().position(|c| c.starts_with("admit")).unwrap();
    assert_eq!(calls[admit_at - 1], "health");
    assert!(screen_text(&r).contains("succeeded"));
    r.step(code(KeyCode::Esc)).unwrap();
    assert!(r.app.modal.is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stale_plan_is_refused_and_the_fresh_one_must_be_confirmed_again() {
    let fake = ws();
    fake.stale_next_admit.store(true, Ordering::SeqCst);
    let mut r = runner(vec![target(fake.clone(), true)], false, T);
    settle(&mut r, T, "hosts", |r| !r.app.hosts.is_empty()).await;
    r.step(key('w')).unwrap();
    r.step(code(KeyCode::Enter)).unwrap();
    settle(&mut r, T, "plan", |r| matches!(&r.app.modal, Some(Modal::Preview { plan: Some(Ok(_)), .. }))).await;
    r.step(key('y')).unwrap();

    // The agent refuses the old confirmation; the TUI shows the fresh plan
    // and asks again. Nothing ran.
    settle(&mut r, T, "refused", |r| matches!(&r.app.modal, Some(Modal::Preview { refused: Some(_), plan: Some(Ok(_)), .. }))).await;
    let text = screen_text(&r);
    assert!(text.contains("REFUSED: the plan changed since the preview"), "{text}");
    assert!(text.contains("STOP TARGETS: ollama-gpu1, llama-router"), "{text}");
    assert!(fake.ops.lock().unwrap().is_empty(), "nothing may run on a stale confirmation");
    assert_eq!(fake.calls().iter().filter(|c| c.starts_with("admit")).count(), 1);
    // No retry on its own.
    for _ in 0..10 {
        r.step(None).unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(fake.calls().iter().filter(|c| c.starts_with("admit")).count(), 1);

    // Confirming the fresh plan admits that plan, with its own revision.
    r.step(key('y')).unwrap();
    settle(&mut r, T, "op done", |r| {
        matches!(&r.app.modal, Some(Modal::Progress { op: Some(o), .. }) if o.status == OpStatus::Succeeded)
    })
    .await;
    let admits: Vec<String> = fake.calls().into_iter().filter(|c| c.starts_with("admit")).collect();
    assert_eq!(admits[1], r#"admit run vllm-engine rev=Some(43) stops=Some(["ollama-gpu1", "llama-router"])"#);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sleeping_host_shows_stale_from_the_cache_within_the_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("hosts/m5.status.json");
    std::fs::create_dir_all(cache.parent().unwrap()).unwrap();
    std::fs::write(&cache, serde_json::to_string(&mac_status("m5")).unwrap()).unwrap();

    let mut sleeping = Fake::new("m5", mac_status("m5"));
    sleeping.hang = true;
    let sleeping = Arc::new(sleeping);
    let mut m5 = target(sleeping.clone(), false);
    m5.cache_path = Some(cache.clone());
    let timeout = Duration::from_millis(300);
    let mut r = runner(vec![target(ws(), true), m5], false, timeout);

    let start = Instant::now();
    settle(&mut r, Duration::from_secs(3), "both hosts", |r| r.app.hosts.len() == 2).await;
    let took = start.elapsed();
    assert!(took < timeout + Duration::from_millis(500), "status --all bound not applied: {took:?}");
    let m5 = r.app.hosts.iter().find(|h| h.name == "m5").unwrap();
    assert!(m5.is_stale());
    assert!(m5.cached.is_some(), "cached observations must be shown");
    let text = screen_text(&r);
    assert!(text.contains("STALE"), "{text}");
    assert!(text.contains("[stale]"), "{text}");
    assert!(!text.lines().any(|l| l.contains("llama-swap") && l.contains("stopped")), "a sleeping host is never 'stopped'");

    // Other screens skip it rather than wait on it.
    r.step(key('o')).unwrap();
    settle(&mut r, Duration::from_secs(2), "ops", |r| !r.app.ops_errors.iter().any(|_| true) && screen_text(r).contains("m5 STALE, skipped")).await;
    // And the next automatic status refresh does not dial it again (a dial
    // can wake it); `r` does.
    let req = r.app.tick(Instant::now() + Duration::from_secs(6));
    assert!(req.iter().any(|q| matches!(q, llmctl_tui::app::Request::RefreshHosts { skip } if skip == &vec!["m5".to_string()])), "{req:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fresh_answer_refreshes_the_status_cache() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("hosts/m5.status.json");
    let fake = Arc::new(Fake::new("m5", mac_status("m5")));
    let mut t = target(fake, false);
    t.cache_path = Some(cache.clone());
    let mut r = runner(vec![t], false, T);
    settle(&mut r, T, "hosts", |r| !r.app.hosts.is_empty()).await;
    let cached: llmctl_client::wire::StatusResponse = serde_json::from_str(&std::fs::read_to_string(&cache).unwrap()).unwrap();
    assert_eq!(cached.host, "m5");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn logs_follow_pages_by_cursor_and_lane_logs_come_from_the_lane() {
    let fake = ws();
    fake.logs.lock().unwrap().extend([entry(1, "first"), entry(2, "second")]);
    let mut r = runner(vec![target(fake.clone(), true)], false, T);
    settle(&mut r, T, "hosts", |r| !r.app.hosts.is_empty()).await;
    // Enter on Hosts opens the selected workload's log.
    r.step(code(KeyCode::Enter)).unwrap();
    assert_eq!(r.app.screen, Screen::Logs);
    settle(&mut r, T, "first page", |r| r.app.logs.lines.iter().any(|l| l.contains("second"))).await;
    fake.logs.lock().unwrap().push(entry(3, "third"));
    settle(&mut r, T, "followed", |r| r.app.logs.lines.iter().any(|l| l.contains("third"))).await;
    let calls = fake.calls();
    assert!(calls.contains(&"logs vllm-engine after=None".to_string()), "{calls:?}");
    assert!(calls.contains(&"logs vllm-engine after=Some(\"2\")".to_string()), "{calls:?}");
    assert_eq!(r.app.logs.lines.iter().filter(|l| l.contains("second")).count(), 1, "no duplicates across pages");

    // Pause: no more polls.
    r.step(key('f')).unwrap();
    let n = fake.calls().iter().filter(|c| c.starts_with("logs")).count();
    for _ in 0..30 {
        r.step(None).unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(fake.calls().iter().filter(|c| c.starts_with("logs")).count(), n);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_screen_gathers_lanes_and_engine_models_and_hosts_show_them_as_rows() {
    let mut mac = Fake::new("m5", mac_status("m5"));
    *mac.lanes.get_mut().unwrap() = Some(LaneFixture {
        workload: "llama-swap".into(),
        lanes: vec!["qwen3-30b".into(), "gemma-27b".into()],
        budget: Some(96),
        peaks: [("qwen3-30b".to_string(), 38.5)].into_iter().collect(),
        running: vec!["qwen3-30b".into()],
    });
    let mac = Arc::new(mac);
    let fake = ws();
    fake.engines.lock().unwrap().push(llmctl_client::wire::EngineReport {
        workload: "llama-router".into(),
        engine_kind: "llama_cpp".into(),
        protocol: "openai".into(),
        state: "ready".into(),
        servable: vec!["gpt-oss-120b".into()],
        loaded: Some(vec!["gpt-oss-120b".into()]),
        supports_unload: true,
    });
    let mut r = runner(vec![target(fake, true), target(mac.clone(), false)], false, T);
    settle(&mut r, T, "hosts", |r| r.app.hosts.len() == 2).await;
    r.step(key('g')).unwrap();
    settle(&mut r, T, "gpu", |r| r.app.gpu.len() == 2).await;
    let text = screen_text(&r);
    assert!(text.contains("budget 96 GiB, loaded peaks 38.5 GiB"), "{text}");
    assert!(text.contains("llama-router (llama_cpp): 1 loaded — gpt-oss-120b"), "{text}");
    // The running-lanes read is a dry run: nothing admitted.
    assert!(mac.calls().iter().any(|c| c == "lanes-running llama-swap qwen3-30b"));
    assert!(mac.ops.lock().unwrap().is_empty());
    r.step(key('d')).unwrap();
    let text = screen_text(&r);
    assert!(text.contains("└ loaded gpt-oss-120b"), "{text}");
    assert!(text.contains("└ loaded qwen3-30b"), "{text}");

    // The logs picker lists the lanes; a lane's log comes from the lane.
    r.step(key('l')).unwrap();
    let idx = r.app.log_targets().iter().position(|t| t.lane.as_deref() == Some("gemma-27b")).unwrap();
    for _ in 0..idx {
        r.step(code(KeyCode::Down)).unwrap();
    }
    r.step(code(KeyCode::Enter)).unwrap();
    settle(&mut r, T, "lane log", |r| r.app.logs.lines.iter().any(|l| l == "gemma-27b line 1")).await;
    assert!(mac.calls().iter().any(|c| c == "lane-logs llama-swap gemma-27b"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn audit_screen_lists_each_hosts_findings() {
    let fake = ws();
    fake.audit.lock().unwrap().push(llmctl_client::wire::AuditFinding {
        id: 1,
        host: "workstation".into(),
        workload: "ollama-gpu1".into(),
        kind: "foreign_transition".into(),
        what: "unit".into(),
        before_state: r#"{"process":"stopped"}"#.into(),
        after_state: r#"{"process":"running","pid":77}"#.into(),
        actor: None,
        observed_at: t0(),
    });
    let mut r = runner(vec![target(fake, true), target(Arc::new(Fake::new("m5", mac_status("m5"))), false)], false, T);
    settle(&mut r, T, "hosts", |r| r.app.hosts.len() == 2).await;
    r.step(key('a')).unwrap();
    settle(&mut r, T, "audit", |r| r.app.audit.len() == 2).await;
    let text = screen_text(&r);
    assert!(text.contains("Audit: foreign writes (1)"), "{text}");
    assert!(text.contains("stopped -> running pid=77"), "{text}");
    assert!(text.contains("no foreign writes"), "{text}");
}
