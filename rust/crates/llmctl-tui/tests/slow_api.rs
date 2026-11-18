//! The parent plan's UI budget: no stall over 100 ms, whatever the agents
//! do. Every fake call here takes 2 s and one host never answers at all;
//! the loop must keep turning keys into frames well inside the budget.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use llmctl_tui::app::Screen;
use llmctl_tui::{percentiles_ms, Options, Runner};
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;

const BUDGET: Duration = Duration::from_millis(100);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_and_silent_agents_never_stall_a_frame() {
    let mut slow_ws = Fake::new("workstation", workstation_status());
    slow_ws.delay = Duration::from_secs(2);
    let mut slow_m5 = Fake::new("m5", mac_status("m5"));
    slow_m5.delay = Duration::from_secs(2);
    let mut asleep = Fake::new("mbp2", mac_status("mbp2"));
    asleep.hang = true;
    let targets = vec![
        target(Arc::new(slow_ws), true),
        target(Arc::new(slow_m5), false),
        target(Arc::new(asleep), false),
    ];
    let term = Terminal::new(TestBackend::new(132, 40)).unwrap();
    let mut r = Runner::new(targets, &Options { read_only: false, timeout: Duration::from_secs(10) }, term);

    let keys = ['o', 'g', 'a', 'l', 'w', 'd', 'm'];
    let mut steps = Vec::new();
    let mut key_frames = Vec::new();
    let start = Instant::now();
    let mut i = 0;
    // 3 s: the first status answers land at ~2 s, the sleeping host never.
    while start.elapsed() < Duration::from_secs(3) {
        let c = keys[i % keys.len()];
        i += 1;
        let t = Instant::now();
        r.step(Some(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))).unwrap();
        let took = t.elapsed();
        key_frames.push(took);
        // The frame drawn by that step already shows the new screen.
        let want = Screen::ALL.into_iter().find(|s| s.key() == c).unwrap();
        assert_eq!(r.app.screen, want);
        for _ in 0..5 {
            let t = Instant::now();
            r.step(None).unwrap();
            steps.push(t.elapsed());
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    let (p50, p95, max) = percentiles_ms(&key_frames);
    let (_, _, step_max) = percentiles_ms(&steps);
    eprintln!("key→frame p50 {p50:.2} ms p95 {p95:.2} ms max {max:.2} ms; idle step max {step_max:.2} ms over {} keys", key_frames.len());
    assert!(Duration::from_secs_f64(max / 1000.0) < BUDGET, "a key took {max:.1} ms to reach the screen");
    assert!(Duration::from_secs_f64(step_max / 1000.0) < BUDGET, "a loop pass took {step_max:.1} ms");
    // The slow hosts did answer meanwhile; the silent one is still pending.
    assert!(r.app.hosts.iter().any(|h| h.name == "workstation" && !h.is_stale()));
    assert!(!r.app.hosts.iter().any(|h| h.name == "mbp2"), "mbp2 cannot have answered");
}
