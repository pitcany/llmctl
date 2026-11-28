//! A fake agent and fixed fixtures for the TUI tests.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, TimeZone, Utc};
use llmctl_client::catalog::{
    CatalogListResponse, CatalogModel, ModelAddRequest, ModelEnableRequest, ModelWriteResponse,
    PresetLink, PresetListResponse, PresetRunRequest, PresetRunResponse, PresetSummary,
    ScanFound, ScanRequest, ScanResponse,
};
use llmctl_client::wire::{
    AdmitRequest, AdmitResponse, AuditFinding, AuditResponse, EngineReport, GpuOccupantView,
    GpuView, HealthResponse, LaneLogsRequest, LaneLogsResponse, LaneMemoryResponse,
    LaneStatusResponse, LaneSummary, LogEntry, LogSource, LogsRequest, LogsResponse, OpStatus,
    OperationListResponse, OperationView, PhaseStatus, PhaseView, PlanRequest, PlanResponse,
    StatusResponse, VramRecordView, VramResponse, WorkloadView,
};
use llmctl_core::{GpuId, HostId, OpKind, OperationId, Plan, PlanStep, UnitName, WorkloadName};
use llmctl_tui::api::{AgentApi, ApiErr, BoxFut};
use llmctl_tui::app::App;
use llmctl_tui::fleet::{HostSnapshot, HostTarget};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

pub fn t0() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 11, 18, 18, 0, 0).unwrap()
}

pub fn workload(name: &str, process: &str, readiness: &str, intent: &str, serves: Option<&str>) -> WorkloadView {
    WorkloadView {
        name: name.into(),
        engine: "systemd".into(),
        model: serves.unwrap_or("-").into(),
        gpus: vec![],
        port: 8000,
        unit: format!("{name}.service"),
        intent: match intent {
            "unmanaged" => intent.into(),
            "stopped (manual)" => r#"{"state":"stopped","manual":true}"#.into(),
            s => format!(r#"{{"state":"{s}"}}"#),
        },
        process: process.into(),
        readiness: readiness.into(),
        served_model: serves.map(String::from),
        served_model_verified: serves.is_some(),
        observed_at: Some(t0() - chrono::Duration::seconds(4)),
        observation_unknown: false,
        ownership: "managed".into(),
        adopted_digest: None,
        recovery_state: "alive".into(),
        recovery_policy: None,
        pid: None,
        engine_kind: None,
    }
}

pub fn workstation_status() -> StatusResponse {
    StatusResponse {
        host: "workstation".into(),
        platform: "linux".into(),
        inventory_revision: 42,
        generated_at: t0() - chrono::Duration::seconds(2),
        workloads: vec![
            workload("vllm-engine", "Running", "Ready", "running", Some("qwen3.8-27b")),
            workload("llama-router", "Running", "Ready", "running", Some("router")),
            workload("ollama-gpu1", "Stopped", "NotReady", "stopped (manual)", None),
        ],
        gpus: vec![
            GpuView { index: 0, enrolled: vec!["vllm-engine".into()], foreign: vec![], error: None },
            GpuView {
                index: 1,
                enrolled: vec!["vllm-engine".into()],
                foreign: vec![GpuOccupantView { pid: 4242, unit: Some("comfyui.service".into()), memory_mib: 2048 }],
                error: None,
            },
        ],
    }
}

pub fn mac_status(host: &str) -> StatusResponse {
    StatusResponse {
        host: host.into(),
        platform: "macos".into(),
        inventory_revision: 7,
        generated_at: t0() - chrono::Duration::seconds(3),
        workloads: vec![workload("llama-swap", "Running", "Ready", "running", None)],
        gpus: vec![],
    }
}

pub fn snapshot(name: &str, status: StatusResponse, local: bool) -> HostSnapshot {
    HostSnapshot {
        name: name.into(),
        local,
        fresh: Some(status),
        cached: None,
        error: None,
        agent_version: Some("0.1.0".into()),
        last_seen: Some(t0() - chrono::Duration::seconds(2)),
        fetched_at: t0(),
    }
}

pub fn stale_snapshot(name: &str, cached: StatusResponse) -> HostSnapshot {
    HostSnapshot {
        name: name.into(),
        local: false,
        fresh: None,
        cached: Some(cached),
        error: Some("no answer within 10 s: asleep or unreachable (LLMCTL_NEXT_STATUS_TIMEOUT_S sets the limit)".into()),
        agent_version: None,
        last_seen: Some(t0() - chrono::Duration::minutes(42)),
        fetched_at: t0(),
    }
}

pub fn op(id: &str, kind: OpKind, workload: &str, status: OpStatus, phases: &[(&str, PhaseStatus)]) -> OperationView {
    OperationView {
        id: id.into(),
        idempotency_key: format!("{kind}-{workload}-r42-1"),
        kind,
        workload: workload.into(),
        status,
        phases: phases
            .iter()
            .enumerate()
            .map(|(i, (l, s))| PhaseView { index: i, label: (*l).into(), status: *s, detail: None })
            .collect(),
        created_at: t0() - chrono::Duration::minutes(5),
        updated_at: t0() - chrono::Duration::seconds(30),
        result: None,
        recovery: None,
        acknowledged: false,
    }
}

pub fn plan_response(kind: OpKind, workload: &str, revision: u64, stops: &[&str]) -> PlanResponse {
    let w = WorkloadName::new(workload).unwrap();
    let unit = UnitName::systemd(llmctl_core::UnitScopeMarker::User, format!("{workload}.service")).unwrap();
    let mut steps = Vec::new();
    for s in stops {
        let sw = WorkloadName::new(*s).unwrap();
        let su = UnitName::systemd(llmctl_core::UnitScopeMarker::User, format!("{s}.service")).unwrap();
        steps.push(PlanStep::StopUnit { workload: sw, unit: su });
    }
    steps.push(PlanStep::StartUnit { workload: w.clone(), unit: unit.clone() });
    steps.push(PlanStep::AwaitReadiness { workload: w.clone(), unit: unit.clone(), timeout_secs: 600 });
    PlanResponse {
        revision,
        stop_targets: stops.iter().map(|s| s.to_string()).collect(),
        plan: Plan {
            id: OperationId::generate(),
            kind,
            target: w.clone(),
            revision,
            expected_model: "qwen3.8-27b".into(),
            affected_gpus: vec![GpuId::new(HostId::new("workstation").unwrap(), 0)],
            steps,
            rollback: vec![PlanStep::StopUnit { workload: w, unit }],
        },
        notes: vec!["fits: 2 cards free".into()],
    }
}

// ---- the fake agent -----------------------------------------------------

pub struct Fake {
    pub host: String,
    /// Every call waits this long first.
    pub delay: Duration,
    /// Never answer (a sleeping Mac behind the bridge).
    pub hang: bool,
    pub status: Mutex<StatusResponse>,
    pub ops: Mutex<Vec<OperationView>>,
    pub audit: Mutex<Vec<AuditFinding>>,
    pub logs: Mutex<Vec<LogEntry>>,
    pub lanes: Mutex<Option<LaneFixture>>,
    pub engines: Mutex<Vec<EngineReport>>,
    pub revision: AtomicU64,
    /// The next admit is refused as stale (and the revision moves).
    pub stale_next_admit: AtomicBool,
    /// Every call, in order (`admit rev=42 stops=[...]`).
    pub calls: Mutex<Vec<String>>,
    /// Presets the Run screen lists (None: the host has none).
    pub presets: Mutex<Vec<PresetSummary>>,
    /// The resolve answer's profile env; None = already active (ordinary
    /// run). Empty string forces `preset_drift` on resolve.
    pub preset_profile: Mutex<Option<String>>,
    /// Catalog records the Catalog screen lists.
    pub catalog: Mutex<Vec<CatalogModel>>,
    /// Weights the scan finds (record None = not cataloged).
    pub scan_found: Mutex<Vec<ScanFound>>,
}

#[derive(Clone)]
pub struct LaneFixture {
    pub workload: String,
    pub lanes: Vec<String>,
    pub budget: Option<u64>,
    pub peaks: BTreeMap<String, f64>,
    pub running: Vec<String>,
}

impl Fake {
    pub fn new(host: &str, status: StatusResponse) -> Self {
        Self {
            host: host.into(),
            delay: Duration::ZERO,
            hang: false,
            revision: AtomicU64::new(status.inventory_revision),
            status: Mutex::new(status),
            ops: Mutex::new(Vec::new()),
            audit: Mutex::new(Vec::new()),
            logs: Mutex::new(Vec::new()),
            lanes: Mutex::new(None),
            engines: Mutex::new(Vec::new()),
            stale_next_admit: AtomicBool::new(false),
            calls: Mutex::new(Vec::new()),
            presets: Mutex::new(Vec::new()),
            preset_profile: Mutex::new(Some(String::new())),
            catalog: Mutex::new(Vec::new()),
            scan_found: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }

    fn log(&self, c: String) {
        self.calls.lock().unwrap().push(c);
    }

    async fn pause(&self) {
        if self.hang {
            std::future::pending::<()>().await;
        }
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
    }
}

macro_rules! fake {
    ($self:ident, $name:expr, $body:expr) => {
        Box::pin(async move {
            $self.pause().await;
            $self.log($name);
            $body
        })
    };
}

impl AgentApi for Fake {
    fn health(&self) -> BoxFut<'_, HealthResponse> {
        fake!(self, "health".into(), Ok(HealthResponse {
            protocol: 1,
            supported_protocols: vec![1],
            daemon_version: "0.1.0".into(),
            host: self.host.clone(),
            pid: 1,
            started_at: t0(),
        }))
    }
    fn status(&self) -> BoxFut<'_, StatusResponse> {
        fake!(self, "status".into(), {
            let mut s = self.status.lock().unwrap().clone();
            s.inventory_revision = self.revision.load(Ordering::SeqCst);
            Ok(s)
        })
    }
    fn operations(&self) -> BoxFut<'_, OperationListResponse> {
        fake!(self, "operations".into(), Ok(OperationListResponse { operations: self.ops.lock().unwrap().clone() }))
    }
    fn operation(&self, id: String) -> BoxFut<'_, OperationView> {
        fake!(self, format!("operation {id}"), {
            let mut ops = self.ops.lock().unwrap();
            match ops.iter_mut().find(|o| o.id == id) {
                Some(o) => {
                    // Each look moves an executing operation to the end.
                    if o.status == OpStatus::Executing {
                        o.status = OpStatus::Succeeded;
                        for p in &mut o.phases {
                            p.status = PhaseStatus::Done;
                        }
                    }
                    Ok(o.clone())
                }
                None => Err(ApiErr::api("unknown_operation", format!("no operation {id}"))),
            }
        })
    }
    fn operation_by_key(&self, key: String) -> BoxFut<'_, OperationView> {
        fake!(self, format!("by-key {key}"), {
            self.ops
                .lock()
                .unwrap()
                .iter()
                .find(|o| o.idempotency_key == key)
                .cloned()
                .ok_or_else(|| ApiErr::api("unknown_operation", "no such key"))
        })
    }
    fn logs(&self, req: LogsRequest) -> BoxFut<'_, LogsResponse> {
        fake!(self, format!("logs {} after={:?}", req.workload, req.after), {
            let all = self.logs.lock().unwrap().clone();
            let start: usize = req.after.as_deref().and_then(|c| c.parse().ok()).unwrap_or(0);
            let entries: Vec<LogEntry> = all.iter().skip(start).cloned().collect();
            Ok(LogsResponse {
                workload: req.workload.clone(),
                unit: format!("{}.service", req.workload),
                cursor: Some(all.len().to_string()),
                entries,
                more: false,
            })
        })
    }
    fn lane_logs(&self, req: LaneLogsRequest) -> BoxFut<'_, LaneLogsResponse> {
        fake!(self, format!("lane-logs {} {}", req.workload, req.lane), Ok(LaneLogsResponse {
            workload: req.workload.clone(),
            lane: req.lane.clone(),
            lines: vec![format!("{} line 1", req.lane), format!("{} line 2", req.lane)],
            available: 2,
        }))
    }
    fn vram(&self) -> BoxFut<'_, VramResponse> {
        fake!(self, "vram".into(), {
            if self.host == "workstation" {
                Ok(VramResponse {
                    free_mib: Some(BTreeMap::from([(0, 3072), (1, 10240)])),
                    records: vec![VramRecordView {
                        workload: "vllm-engine".into(),
                        model: "qwen3.8-27b".into(),
                        per_card_mib: BTreeMap::from([(0, 28000), (1, 28000)]),
                        measured_at: t0() - chrono::Duration::days(1),
                    }],
                    measured: None,
                })
            } else {
                Err(ApiErr::api("unsupported", "no GPUs on this host"))
            }
        })
    }
    fn audit(&self) -> BoxFut<'_, AuditResponse> {
        fake!(self, "audit".into(), Ok(AuditResponse { host: self.host.clone(), findings: self.audit.lock().unwrap().clone() }))
    }
    fn lanes_status(&self, workload: String) -> BoxFut<'_, LaneStatusResponse> {
        fake!(self, format!("lanes {workload}"), {
            match self.lanes.lock().unwrap().clone() {
                Some(l) if l.workload == workload => Ok(LaneStatusResponse {
                    workload,
                    config_path: "/tmp/config.yaml".into(),
                    log_path: None,
                    revision: 1,
                    file_digest: "d".into(),
                    file_digest_now: "d".into(),
                    drift: false,
                    lanes: l
                        .lanes
                        .iter()
                        .map(|id| LaneSummary {
                            id: id.clone(),
                            engine_kind: "llama_cpp".into(),
                            name: None,
                            model_name: None,
                            ttl: None,
                            pinned: None,
                        })
                        .collect(),
                    macros: vec![],
                    validation: vec![],
                    updated_at: t0(),
                    memory: vec![],
                    matrix: None,
                }),
                _ => Err(ApiErr::api("lanes_refused", format!("{workload} has no lane records"))),
            }
        })
    }
    fn lanes_memory(&self, workload: String) -> BoxFut<'_, LaneMemoryResponse> {
        fake!(self, format!("lanes-memory {workload}"), {
            let l = self.lanes.lock().unwrap().clone().expect("lanes fixture");
            Ok(LaneMemoryResponse {
                workload,
                budget_gib: l.budget,
                peaks: l.peaks.clone(),
                lines: vec![],
                over: vec![],
                changed: false,
            })
        })
    }
    fn lanes_running(&self, workload: String, lane: String) -> BoxFut<'_, Vec<String>> {
        fake!(self, format!("lanes-running {workload} {lane}"), {
            Ok(self.lanes.lock().unwrap().clone().map(|l| l.running).unwrap_or_default())
        })
    }
    fn engine_report(&self, workload: String) -> BoxFut<'_, EngineReport> {
        fake!(self, format!("engine {workload}"), {
            self.engines
                .lock()
                .unwrap()
                .iter()
                .find(|e| e.workload == workload)
                .cloned()
                .ok_or_else(|| ApiErr::api("unsupported", "no engine report"))
        })
    }
    fn plan(&self, req: PlanRequest) -> BoxFut<'_, PlanResponse> {
        fake!(self, format!("plan {} {}", req.kind, req.workload), {
            let rev = self.revision.load(Ordering::SeqCst);
            let stops: &[&str] = if rev > 42 { &["ollama-gpu1", "llama-router"] } else { &["ollama-gpu1"] };
            Ok(plan_response(req.kind, &req.workload, rev, stops))
        })
    }
    fn admit(&self, req: AdmitRequest) -> BoxFut<'_, AdmitResponse> {
        let profile = req
            .profile
            .as_ref()
            .map(|p| format!(" profile={}", p.env_source.rsplit('/').next().unwrap_or(&p.env_source)))
            .unwrap_or_default();
        fake!(self, format!("admit {} {} rev={:?} stops={:?}{profile}", req.kind, req.workload, req.expected_revision, req.expected_stop_targets), {
            if self.stale_next_admit.swap(false, Ordering::SeqCst) {
                let old = self.revision.fetch_add(1, Ordering::SeqCst);
                return Err(ApiErr::api(
                    "stale_plan",
                    format!("inventory revision moved {old} -> {}: the stop targets changed", old + 1),
                ));
            }
            if req.expected_revision != Some(self.revision.load(Ordering::SeqCst)) {
                return Err(ApiErr::api("stale_plan", "revision mismatch"));
            }
            let o = op(
                &format!("op-{}", self.ops.lock().unwrap().len() + 1),
                req.kind,
                &req.workload,
                OpStatus::Executing,
                &[("stop ollama-gpu1", PhaseStatus::Done), ("start vllm-engine", PhaseStatus::Running)],
            );
            let o = OperationView { idempotency_key: req.idempotency_key.clone(), ..o };
            self.ops.lock().unwrap().push(o.clone());
            Ok(AdmitResponse { operation: o, replayed: false })
        })
    }
    fn cancel(&self, id: String) -> BoxFut<'_, OperationView> {
        fake!(self, format!("cancel {id}"), {
            let mut ops = self.ops.lock().unwrap();
            let o = ops.iter_mut().find(|o| o.id == id).ok_or_else(|| ApiErr::api("unknown_operation", "no"))?;
            o.status = OpStatus::CancelRequested;
            Ok(o.clone())
        })
    }
    fn acknowledge(&self, id: String) -> BoxFut<'_, OperationView> {
        fake!(self, format!("ack {id}"), {
            let mut ops = self.ops.lock().unwrap();
            let o = ops.iter_mut().find(|o| o.id == id).ok_or_else(|| ApiErr::api("unknown_operation", "no"))?;
            o.acknowledged = true;
            Ok(o.clone())
        })
    }
    fn presets(&self) -> BoxFut<'_, PresetListResponse> {
        fake!(self, "presets".into(), Ok(PresetListResponse {
            dirs: vec!["~/.config/llmctl/presets".into()],
            presets: self.presets.lock().unwrap().clone(),
            invalid: vec![],
        }))
    }
    fn preset_run(&self, req: PresetRunRequest) -> BoxFut<'_, PresetRunResponse> {
        fake!(self, format!("preset-run {} sync={}", req.alias, req.sync), {
            let known = self.presets.lock().unwrap().iter().any(|p| p.alias == req.alias);
            if !known {
                return Err(ApiErr::api("unknown_preset", format!("no preset {}", req.alias)));
            }
            // Bound before the match: an arm below re-locks this mutex (the
            // sync writes the profile), and a guard held across the whole
            // match would deadlock against it.
            let profile = self.preset_profile.lock().unwrap().clone();
            match profile {
                // Empty string stands for a drifted profile: refused until
                // the caller syncs — the sync write happens and the profile
                // is named from then on.
                Some(p) if p.is_empty() && !req.sync => {
                    Err(ApiErr::api("preset_drift", "profile does not render the preset; --sync writes it"))
                }
                Some(p) if p.is_empty() => {
                    let written = format!("~/.config/llmctl/vllm-engine.env.{}", req.alias);
                    *self.preset_profile.lock().unwrap() = Some(written.clone());
                    Ok(PresetRunResponse {
                        alias: req.alias.clone(),
                        workload: "vllm-engine".into(),
                        model: "qwen3.8-27b".into(),
                        profile_env: Some(written),
                        synced: true,
                        drift: vec![],
                    })
                }
                Some(p) => Ok(PresetRunResponse {
                    alias: req.alias,
                    workload: "vllm-engine".into(),
                    model: "qwen3.8-27b".into(),
                    profile_env: Some(p),
                    synced: req.sync,
                    drift: vec![],
                }),
                None => Ok(PresetRunResponse {
                    alias: req.alias,
                    workload: "vllm-engine".into(),
                    model: "qwen3.8-27b".into(),
                    profile_env: None,
                    synced: false,
                    drift: vec![],
                }),
            }
        })
    }
    fn catalog_models(&self, all: bool) -> BoxFut<'_, CatalogListResponse> {
        fake!(self, format!("catalog all={all}"), Ok(CatalogListResponse {
            host: self.host.clone(),
            host_names: vec![self.host.clone()],
            revision: 1,
            disabled_hidden: 0,
            unmapped: 0,
            profiles: 0,
            models: self.catalog.lock().unwrap().clone(),
        }))
    }
    fn model_enable(&self, req: ModelEnableRequest) -> BoxFut<'_, ModelWriteResponse> {
        fake!(self, format!("model-enable {} {}", req.model, req.enabled), {
            let mut models = self.catalog.lock().unwrap();
            let m = models.iter_mut().find(|m| m.id == req.model || m.name == req.model)
                .ok_or_else(|| ApiErr::api("unknown_model", format!("no record {}", req.model)))?;
            m.enabled = req.enabled;
            Ok(ModelWriteResponse { model: m.clone(), changes: vec![], revision: 2 })
        })
    }
    fn model_add(&self, req: ModelAddRequest) -> BoxFut<'_, ModelWriteResponse> {
        fake!(self, format!("model-add {} engine={}", req.name, req.engine), {
            let m = catalog_record(&req.name, &req.engine, req.store.as_deref());
            Ok(ModelWriteResponse { model: m, changes: vec![], revision: 3 })
        })
    }
    fn model_scan(&self, req: ScanRequest) -> BoxFut<'_, ScanResponse> {
        fake!(self, format!("scan add={}", req.add), Ok(ScanResponse {
            stores: vec!["hf-cache".into()],
            found: self.scan_found.lock().unwrap().clone(),
            missing: vec![],
            added: 0,
            revision: 1,
        }))
    }
}

/// A catalog record with the fields the screen reads.
pub fn catalog_record(name: &str, engine: &str, store: Option<&str>) -> CatalogModel {
    CatalogModel {
        id: format!("m-{name}"),
        name: name.into(),
        engine: engine.into(),
        source: Some(format!("hf:{name}")),
        path: Some(format!("~/.cache/huggingface/{name}")),
        store: store.map(String::from),
        enabled: true,
        size_bytes: Some(19_000_000_000),
        extra: [(
            "estimated_vram_gb".to_string(),
            serde_json::json!(18.5),
        )]
        .into_iter()
        .collect(),
        origin: "registry".into(),
    }
}

/// A preset summary with the given alias and link state.
pub fn preset(alias: &str, state: &str) -> PresetSummary {
    let (link, link_error) = match state {
        "serving" => (Some(PresetLink {
            workload: "vllm-engine".into(),
            profile_env: Some(format!("~/.config/llmctl/vllm-engine.env.{alias}")),
            found_by: "router".into(),
            drift: vec![],
            active: true,
        }), None),
        "ready" => (Some(PresetLink {
            workload: "vllm-engine".into(),
            profile_env: Some(format!("~/.config/llmctl/vllm-engine.env.{alias}")),
            found_by: "router".into(),
            drift: vec![],
            active: false,
        }), None),
        "drift" => (Some(PresetLink {
            workload: "vllm-engine".into(),
            profile_env: Some(format!("~/.config/llmctl/vllm-engine.env.{alias}")),
            found_by: "router".into(),
            drift: vec![llmctl_client::catalog::EnvDrift {
                key: "VLLM_MAX_MODEL_LEN".into(),
                file: Some("262144".into()),
                render: Some("131072".into()),
            }],
            active: false,
        }), None),
        other => (None, Some(format!("no vLLM workload for a preset here ({other})"))),
    };
    PresetSummary {
        alias: alias.into(),
        served_name: format!("qwen-{alias}"),
        model_id: format!("Qwen/Qwen-{alias}"),
        quantization: "fp8".into(),
        tensor_parallel_size: 2,
        max_model_len: 262144,
        family: Some("qwen".into()),
        param_count_b: Some(27.0),
        source_path: format!("~/.config/llmctl/presets/{alias}.yaml"),
        link,
        link_error,
    }
}

pub fn target(fake: Arc<Fake>, local: bool) -> HostTarget {
    HostTarget { name: fake.host.clone(), api: fake, local, cache_path: None, last_seen: None }
}

/// The screen as text, trailing blanks cut (one line per terminal row).
pub fn render(app: &App, w: u16, h: u16) -> String {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| llmctl_tui::ui::draw(f, app)).unwrap();
    let buf = term.backend().buffer().clone();
    let mut out = String::new();
    for y in 0..buf.area.height {
        let mut line = String::new();
        for x in 0..buf.area.width {
            line.push_str(buf[(x, y)].symbol());
        }
        out.push_str(line.trim_end());
        out.push('\n');
    }
    out
}

/// Compare with `tests/snapshots/<name>.txt`; `UPDATE_SNAPSHOTS=1` writes it.
pub fn assert_snapshot(name: &str, actual: &str) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots").join(format!("{name}.txt"));
    if std::env::var_os("UPDATE_SNAPSHOTS").is_some() || !path.exists() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
        if std::env::var_os("UPDATE_SNAPSHOTS").is_none() {
            panic!("snapshot {name} did not exist; wrote it — review and re-run");
        }
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap();
    if expected != actual {
        panic!("snapshot {name} differs\n--- expected\n{expected}\n--- actual\n{actual}");
    }
}

pub fn entry(secs: i64, msg: &str) -> LogEntry {
    LogEntry {
        timestamp: t0() + chrono::Duration::seconds(secs),
        priority: None,
        source: LogSource::Unit,
        invocation: Some("abcdef1234567890".into()),
        message: msg.into(),
        truncated: false,
        time_estimated: false,
    }
}
