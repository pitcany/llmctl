//! Render snapshots per screen, from fixed state (no agent involved).
//! `UPDATE_SNAPSHOTS=1 cargo test -p llmctl-tui --test screens` rewrites
//! them; review the diff before committing.

mod common;

use std::collections::BTreeMap;

use common::*;
use llmctl_client::wire::{AuditFinding, EngineReport, OpStatus, PhaseStatus, VramRecordView, VramResponse};
use llmctl_core::OpKind;
use llmctl_tui::app::{App, HostGpu, HostResult, LaneHost, LogTarget, Modal, Request, Screen};

const W: u16 = 132;
const H: u16 = 30;

/// Workstation fresh, m5 fresh (lanes), mbp2 asleep (STALE, cached).
fn base_app() -> App {
    let mut app = App::new(vec!["workstation".into(), "m5".into(), "mbp2".into()], false);
    app.now = t0();
    app.local_time = false;
    app.hosts = vec![
        snapshot("workstation", workstation_status(), true),
        snapshot("m5", mac_status("m5"), false),
        stale_snapshot("mbp2", mac_status("mbp2")),
    ];
    app.gpu = vec![
        HostGpu {
            host: "workstation".into(),
            vram: Some(Ok(VramResponse {
                free_mib: Some(BTreeMap::from([(0, 3072), (1, 10240)])),
                records: vec![VramRecordView {
                    workload: "vllm-engine".into(),
                    model: "qwen3.8-27b".into(),
                    per_card_mib: BTreeMap::from([(0, 28000), (1, 28000)]),
                    measured_at: t0(),
                }],
                measured: None,
            })),
            lanes: vec![],
            // A llama.cpp router reporting its models: extra rows, no
            // special case (M5d adds these).
            engines: vec![EngineReport {
                workload: "llama-router".into(),
                engine_kind: "llama_cpp".into(),
                protocol: "openai".into(),
                state: "ready".into(),
                servable: vec!["gpt-oss-120b".into(), "qwen3-coder".into()],
                loaded: Some(vec!["gpt-oss-120b".into(), "qwen3-coder".into()]),
                supports_unload: true,
            }],
            error: None,
        },
        HostGpu {
            host: "m5".into(),
            vram: Some(Err("unsupported: no GPUs on this host".into())),
            lanes: vec![LaneHost {
                workload: "llama-swap".into(),
                lane_ids: vec!["qwen3-30b".into(), "gemma-27b".into()],
                budget_gib: Some(96),
                peaks: BTreeMap::from([("qwen3-30b".into(), 38.5), ("gemma-27b".into(), 30.0)]),
                lines: vec![],
                over: vec![],
                running: Some(vec!["qwen3-30b".into()]),
            }],
            engines: vec![],
            error: None,
        },
    ];
    app
}

#[test]
fn hosts_screen() {
    let app = base_app();
    assert_snapshot("hosts", &render(&app, W, H));
}

#[test]
fn hosts_screen_shows_stale_never_stopped() {
    let text = render(&base_app(), W, H);
    let mbp2: Vec<&str> = text.lines().skip_while(|l| !l.contains("mbp2")).take(3).collect();
    assert!(mbp2[0].contains("STALE"), "{text}");
    assert!(mbp2.iter().any(|l| l.contains("llama-swap") && l.contains("running") && l.contains("[stale]")), "{text}");
}

#[test]
fn run_screen() {
    let mut app = base_app();
    app.screen = Screen::Run;
    app.run_sel = 0;
    assert_snapshot("run", &render(&app, W, H));
}

#[test]
fn ops_screen() {
    let mut app = base_app();
    app.screen = Screen::Ops;
    let mut failed = op(
        "0b5f2c9e-aaaa",
        OpKind::Run,
        "vllm-engine",
        OpStatus::NeedsAttention,
        &[("stop ollama-gpu1", PhaseStatus::Done), ("start vllm-engine", PhaseStatus::Failed)],
    );
    failed.result = Some(serde_json::json!({"error": "readiness timeout after 600 s"}));
    failed.recovery = Some(serde_json::json!({"restored": ["ollama-gpu1"], "serving": true}));
    app.ops = vec![
        (
            "workstation".into(),
            op("9f1e0d7c-bbbb", OpKind::Restart, "ollama-gpu1", OpStatus::Executing, &[("stop ollama-gpu1", PhaseStatus::Done), ("start ollama-gpu1", PhaseStatus::Running)]),
        ),
        ("workstation".into(), failed),
        ("m5".into(), op("77aa0000-cccc", OpKind::LaneSwap, "llama-swap", OpStatus::Succeeded, &[("load qwen3-30b", PhaseStatus::Done)])),
    ];
    app.ops_sel = 1;
    assert_snapshot("ops", &render(&app, W, H));
}

#[test]
fn logs_screen() {
    let mut app = base_app();
    app.screen = Screen::Logs;
    app.logs.target = Some(LogTarget { host: "workstation".into(), workload: "vllm-engine".into(), lane: None });
    app.logs.lines = vec![
        "-- run abcdef12 --".into(),
        "Nov 18 18:00:01 INFO Started server process".into(),
        "Nov 18 18:00:05 INFO Application startup complete.".into(),
        "Nov 18 18:00:09 manager: vllm-engine.service: Consumed 1min CPU time.".into(),
    ];
    app.logs.picker_sel = 0;
    assert_snapshot("logs", &render(&app, W, H));
}

#[test]
fn logs_picker_lists_lanes_under_their_workload() {
    let app = base_app();
    let lanes: Vec<_> = app.log_targets().into_iter().filter_map(|t| t.lane).collect();
    assert_eq!(lanes, vec!["qwen3-30b".to_string(), "gemma-27b".to_string()]);
}

#[test]
fn gpu_screen() {
    let mut app = base_app();
    app.screen = Screen::Gpu;
    assert_snapshot("gpu", &render(&app, W, H));
}

#[test]
fn audit_screen() {
    let mut app = base_app();
    app.screen = Screen::Audit;
    app.audit = vec![
        HostResult { host: "workstation".into(), result: Ok(vec![]) },
        HostResult {
            host: "m5".into(),
            result: Ok(vec![AuditFinding {
                id: 3,
                host: "m5".into(),
                workload: "llama-swap".into(),
                kind: "foreign_transition".into(),
                what: "lanes".into(),
                before_state: r#"{"process":"running","pid":812,"lanes":["qwen3-30b"]}"#.into(),
                after_state: r#"{"process":"running","pid":812,"lanes":[]}"#.into(),
                actor: Some("llama-swap (ttl: 600 s)".into()),
                observed_at: t0() - chrono::Duration::hours(2),
            }]),
        },
    ];
    assert_snapshot("audit", &render(&app, W, H));
}

#[test]
fn catalog_screen_is_marked_pending() {
    let mut app = base_app();
    app.screen = Screen::Catalog;
    let text = render(&app, W, H);
    assert!(text.contains("Catalog: pending"), "{text}");
    assert_snapshot("catalog", &text);
}

#[test]
fn preview_modal() {
    let mut app = base_app();
    app.screen = Screen::Run;
    app.modal = Some(Modal::Preview {
        host: "workstation".into(),
        kind: OpKind::Run,
        workload: "vllm-engine".into(),
        plan: Some(Ok(plan_response(OpKind::Run, "vllm-engine", 42, &["ollama-gpu1"]))),
        refused: None,
    });
    assert_snapshot("preview", &render(&app, W, H));
}

#[test]
fn stale_refused_preview_modal() {
    let mut app = base_app();
    app.screen = Screen::Run;
    app.modal = Some(Modal::Preview {
        host: "workstation".into(),
        kind: OpKind::Run,
        workload: "vllm-engine".into(),
        plan: Some(Ok(plan_response(OpKind::Run, "vllm-engine", 43, &["ollama-gpu1", "llama-router"]))),
        refused: Some("inventory revision moved 42 -> 43: the stop targets changed".into()),
    });
    assert_snapshot("preview_refused", &render(&app, W, H));
}

#[test]
fn confirm_modal() {
    let mut app = base_app();
    app.screen = Screen::Ops;
    app.modal = Some(Modal::Confirm {
        title: "Cancel".into(),
        lines: vec!["Cancel operation 9f1e0d7c-bbbb on workstation?".into(), "restart ollama-gpu1 — executing".into()],
        on_yes: Request::Cancel { host: "workstation".into(), id: "9f1e0d7c-bbbb".into() },
    });
    assert_snapshot("confirm", &render(&app, W, H));
}
