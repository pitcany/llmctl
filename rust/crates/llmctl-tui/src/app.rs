//! TUI state: what each screen shows, what a key does, and which agent
//! calls that asks for. Pure and synchronous: keys and agent answers come
//! in, [`Request`]s go out to the [`crate::worker::Worker`], which runs them
//! off the UI thread. Nothing here waits on the network, so a slow or
//! sleeping agent can never stall a frame.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use llmctl_client::catalog::{
    CatalogListResponse, CatalogModel, PresetListResponse, PresetSummary, ScanFound, ScanResponse,
};
use llmctl_client::wire::{
    AuditFinding, EngineReport, LogEntry, LogSource, OpStatus, OperationView, PlanResponse,
    ProfileRequest, VramResponse, WorkloadView,
};
use llmctl_core::OpKind;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::api::ApiErr;
use crate::fleet::HostSnapshot;

/// How often each screen asks again while it is shown.
pub const HOSTS_EVERY: Duration = Duration::from_secs(5);
pub const OPS_EVERY: Duration = Duration::from_secs(3);
pub const OPS_ACTIVE_EVERY: Duration = Duration::from_secs(1);
pub const GPU_EVERY: Duration = Duration::from_secs(10);
pub const AUDIT_EVERY: Duration = Duration::from_secs(30);
/// Presets and the catalog change by hand and rarely.
pub const PRESETS_EVERY: Duration = Duration::from_secs(30);
pub const CATALOG_EVERY: Duration = Duration::from_secs(30);
pub const LOGS_FOLLOW_EVERY: Duration = Duration::from_secs(1);
/// An unreachable host is asked again on its own at most this often:
/// every ssh attempt can wake a sleeping Mac (DarkWake), and each costs a
/// full status timeout. `r` asks at once.
pub const STALE_RETRY: Duration = Duration::from_secs(60);
/// Log lines kept on screen.
pub const LOG_KEEP: usize = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Screen {
    Hosts,
    Run,
    Ops,
    Logs,
    Gpu,
    Catalog,
    Audit,
}

impl Screen {
    pub const ALL: [Screen; 7] = [
        Screen::Hosts,
        Screen::Run,
        Screen::Ops,
        Screen::Logs,
        Screen::Gpu,
        Screen::Catalog,
        Screen::Audit,
    ];

    pub fn key(self) -> char {
        match self {
            Screen::Hosts => 'd',
            Screen::Run => 'w',
            Screen::Ops => 'o',
            Screen::Logs => 'l',
            Screen::Gpu => 'g',
            Screen::Catalog => 'm',
            Screen::Audit => 'a',
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Screen::Hosts => "Hosts",
            Screen::Run => "Run",
            Screen::Ops => "Operations",
            Screen::Logs => "Logs",
            Screen::Gpu => "GPUs/memory",
            Screen::Catalog => "Catalog",
            Screen::Audit => "Audit",
        }
    }

    fn from_key(c: char) -> Option<Screen> {
        Screen::ALL.into_iter().find(|s| s.key() == c)
    }
}

/// What the logs pane reads: a workload's service log, or one lane's
/// output on a llama-swap workload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogTarget {
    pub host: String,
    pub workload: String,
    pub lane: Option<String>,
}

impl LogTarget {
    pub fn label(&self) -> String {
        match &self.lane {
            Some(l) => format!("{} · {} / lane {l}", self.host, self.workload),
            None => format!("{} · {}", self.host, self.workload),
        }
    }
}

/// Work for the worker. Every variant is one bounded agent conversation.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// The bounded multi-host status call; `skip` = unreachable hosts not
    /// due for another try (their snapshot stays as it is).
    RefreshHosts { skip: Vec<String> },
    RefreshOps { hosts: Vec<String> },
    RefreshAudit { hosts: Vec<String> },
    /// Per host: its workload names (lane records and engine reports are
    /// asked per workload).
    RefreshGpu { hosts: Vec<(String, Vec<String>)> },
    Logs { gen: u64, target: LogTarget, cursor: Option<String> },
    /// The plan preview (read-only on the agent). `profile`: a preset's
    /// resolved router profile env, applied by the run.
    Plan { host: String, kind: OpKind, workload: String, profile: Option<ProfileRequest> },
    /// Admit exactly the previewed plan: the agent refuses with
    /// `stale_plan` if the revision or the stop targets moved.
    Admit {
        host: String,
        kind: OpKind,
        workload: String,
        revision: u64,
        stop_targets: Vec<String>,
        profile: Option<ProfileRequest>,
    },
    Cancel { host: String, id: String },
    Ack { host: String, id: String },
    /// The Run screen's preset list (read-only).
    RefreshPresets { hosts: Vec<String> },
    /// The Catalog screen's records (read-only).
    RefreshCatalog { hosts: Vec<String> },
    /// Resolve a preset and plan its run (writes nothing; a drifted
    /// profile answers `preset_drift`).
    PresetPlan { host: String, alias: String },
    /// Write a preset's render into its profile env (the `s` key on a
    /// drift refusal), then the App plans again.
    PresetSync { host: String, alias: String },
    /// Enable or disable one catalog record.
    ModelEnable { host: String, model: String, enabled: bool },
    /// Record one weight a scan found.
    ModelAdd { host: String, found: ScanFound },
    /// Walk one host's stores; adds nothing.
    ModelScan { host: String },
}

#[derive(Debug, Clone)]
pub struct HostResult<T> {
    pub host: String,
    pub result: Result<T, String>,
}

/// One llama-swap workload's memory picture.
#[derive(Debug, Clone, Default)]
pub struct LaneHost {
    pub workload: String,
    pub lane_ids: Vec<String>,
    pub budget_gib: Option<u64>,
    pub peaks: BTreeMap<String, f64>,
    pub lines: Vec<String>,
    pub over: Vec<(String, u64)>,
    /// `None` when llama-swap could not be asked.
    pub running: Option<Vec<String>>,
}

/// One host's GPU/memory answers.
#[derive(Debug, Clone, Default)]
pub struct HostGpu {
    pub host: String,
    pub vram: Option<Result<VramResponse, String>>,
    pub lanes: Vec<LaneHost>,
    /// Engines that can say what they have loaded (Ollama, LM Studio, a
    /// llama.cpp router): one extra row per loaded model.
    pub engines: Vec<EngineReport>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub enum LogPage {
    Unit { entries: Vec<LogEntry>, cursor: Option<String>, more: bool },
    Lane { lines: Vec<String> },
}

/// Answers from the worker.
#[derive(Debug, Clone)]
pub enum Msg {
    /// Snapshots as they land: one host's answer (or timeout) at a time, so
    /// a sleeping host never holds back the others.
    Hosts(Vec<HostSnapshot>),
    /// Every host asked in this refresh has answered or timed out.
    HostsDone,
    Ops(Vec<HostResult<Vec<OperationView>>>),
    Audit(Vec<HostResult<Vec<AuditFinding>>>),
    Gpu(Vec<HostGpu>),
    Logs { gen: u64, target: LogTarget, result: Result<LogPage, ApiErr> },
    Planned { host: String, kind: OpKind, workload: String, result: Result<PlanResponse, ApiErr> },
    /// The admit landed (`replayed` = an earlier admit with the same key)
    /// or failed for a reason other than a stale plan.
    Admitted { host: String, result: Result<(OperationView, bool), ApiErr> },
    /// The agent refused the confirmation because the world moved; `fresh`
    /// is a new preview to confirm (or not).
    StaleRefused { host: String, kind: OpKind, workload: String, reason: String, fresh: Result<PlanResponse, ApiErr> },
    /// Live progress of an admitted operation.
    OpUpdate { host: String, op: OperationView },
    /// A cancel or acknowledge answered.
    WriteDone { host: String, what: &'static str, result: Result<OperationView, ApiErr> },
    /// The Run screen's preset lists.
    Presets(Vec<HostResult<PresetListResponse>>),
    /// The Catalog screen's record lists.
    Catalog(Vec<HostResult<CatalogListResponse>>),
    /// One preset resolved: the workload to run, the profile to switch to
    /// (`None` = the active env already renders it) and the plan preview.
    PresetPlanned {
        host: String,
        alias: String,
        result: Result<Box<PresetResolved>, ApiErr>,
    },
    /// A store scan answered (read-only; rows for the `a` key).
    ScanDone { host: String, result: Result<ScanResponse, ApiErr> },
    /// A preset sync or a catalog write answered; `text` is the one line
    /// to show (the App re-plans a preset after its sync).
    SimpleWrite { host: String, what: &'static str, result: Result<String, ApiErr> },
}

/// What a resolved preset runs on (M5e Run screen).
#[derive(Debug, Clone)]
pub struct PresetResolved {
    pub workload: String,
    pub profile: Option<ProfileRequest>,
    pub plan: Result<PlanResponse, ApiErr>,
}

#[derive(Debug, Clone)]
pub enum Modal {
    Help,
    /// A yes/no question in front of a write that has no plan (cancel,
    /// acknowledge).
    Confirm { title: String, lines: Vec<String>, on_yes: Request },
    /// The plan preview of a run/stop/restart; `y` admits it. `alias`
    /// names a preset run (resolved with its `profile`); a `preset_drift`
    /// refusal offers `s` to sync the profile env and re-plan.
    Preview {
        host: String,
        kind: OpKind,
        workload: String,
        alias: Option<String>,
        profile: Option<ProfileRequest>,
        plan: Option<Result<PlanResponse, ApiErr>>,
        /// Set when the agent refused the previous confirmation as stale.
        refused: Option<String>,
    },
    /// Live progress after admission.
    Progress {
        host: String,
        kind: OpKind,
        workload: String,
        op: Option<OperationView>,
        error: Option<String>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct LogsView {
    pub target: Option<LogTarget>,
    pub lines: Vec<String>,
    pub cursor: Option<String>,
    pub follow: bool,
    pub error: Option<String>,
    /// Bumped on every new target, so late pages of the old one are dropped.
    pub gen: u64,
    pub in_flight: bool,
    pub last_req: Option<Instant>,
    pub picker_sel: usize,
    /// Lines up from the bottom (0 = newest at the bottom).
    pub scroll: usize,
    last_run: Option<String>,
}

#[derive(Debug, Default)]
struct Pace {
    in_flight: bool,
    last: Option<Instant>,
}

impl Pace {
    fn due(&self, now: Instant, every: Duration) -> bool {
        !self.in_flight && self.last.is_none_or(|t| now.duration_since(t) >= every)
    }

    fn start(&mut self, now: Instant) {
        self.in_flight = true;
        self.last = Some(now);
    }
}

/// One selectable workload line (Hosts and Run screens).
#[derive(Debug, Clone)]
pub struct WorkloadRow<'a> {
    pub host: &'a str,
    pub workload: &'a WorkloadView,
    pub stale: bool,
}

/// One selectable line of the Run screen (M5e: presets above workloads).
#[derive(Debug, Clone)]
pub enum RunRow<'a> {
    Preset { host: &'a str, preset: &'a PresetSummary, stale: bool },
    Workload(WorkloadRow<'a>),
}

/// One selectable line of the Catalog screen.
#[derive(Debug, Clone)]
pub enum CatalogRow<'a> {
    Model { host: &'a str, model: &'a CatalogModel, stale: bool },
    Found { host: &'a str, found: &'a ScanFound, stale: bool },
}

impl CatalogRow<'_> {
    pub fn host(&self) -> &str {
        match self {
            CatalogRow::Model { host, .. } | CatalogRow::Found { host, .. } => host,
        }
    }
}

pub struct App {
    pub screen: Screen,
    /// Wall clock for "seen 3 s ago"; set by the loop, fixed in tests.
    pub now: DateTime<Utc>,
    /// Times on screen in local time (UTC in tests).
    pub local_time: bool,
    pub read_only: bool,
    pub quit: bool,
    /// Target names in order (local first): the Hosts screen's order.
    pub host_order: Vec<String>,
    pub hosts: Vec<HostSnapshot>,
    pub hosts_sel: usize,
    pub run_sel: usize,
    pub ops: Vec<(String, OperationView)>,
    pub ops_errors: Vec<(String, String)>,
    pub ops_sel: usize,
    pub audit: Vec<HostResult<Vec<AuditFinding>>>,
    pub audit_sel: usize,
    pub gpu: Vec<HostGpu>,
    pub logs: LogsView,
    pub modal: Option<Modal>,
    /// One line at the bottom: the last result or refusal.
    pub message: Option<String>,
    /// The Run screen's preset lists, per host (the Macs hold none).
    pub presets: Vec<HostResult<PresetListResponse>>,
    /// The Catalog screen's records, per host.
    pub catalog: Vec<HostResult<CatalogListResponse>>,
    pub catalog_sel: usize,
    /// Latest store scan per host (read-only; rows the `a` key records).
    pub scans: Vec<(String, Result<ScanResponse, String>)>,
    /// The preset whose profile sync was confirmed and is being written;
    /// its answer re-opens the preview and re-plans.
    pending_preset: Option<(String, String)>,
    /// Hosts never asked yet: the first refresh asks every host.
    first_hosts: bool,
    hosts_pace: Pace,
    ops_pace: Pace,
    audit_pace: Pace,
    gpu_pace: Pace,
    presets_pace: Pace,
    catalog_pace: Pace,
}

impl App {
    pub fn new(host_order: Vec<String>, read_only: bool) -> Self {
        Self {
            screen: Screen::Hosts,
            now: Utc::now(),
            local_time: true,
            read_only,
            quit: false,
            host_order,
            hosts: Vec::new(),
            hosts_sel: 0,
            run_sel: 0,
            ops: Vec::new(),
            ops_errors: Vec::new(),
            ops_sel: 0,
            audit: Vec::new(),
            audit_sel: 0,
            gpu: Vec::new(),
            logs: LogsView { follow: true, ..LogsView::default() },
            modal: None,
            message: None,
            presets: Vec::new(),
            catalog: Vec::new(),
            catalog_sel: 0,
            scans: Vec::new(),
            pending_preset: None,
            first_hosts: true,
            hosts_pace: Pace::default(),
            ops_pace: Pace::default(),
            audit_pace: Pace::default(),
            gpu_pace: Pace::default(),
            presets_pace: Pace::default(),
            catalog_pace: Pace::default(),
        }
    }

    // ---- derived views ------------------------------------------------------

    /// One selectable row of the Run screen: a preset (first, per host) or
    /// a workload.
    pub fn run_rows(&self) -> Vec<RunRow<'_>> {
        let mut rows = Vec::new();
        for h in &self.presets {
            if let Ok(p) = &h.result {
                for p in &p.presets {
                    rows.push(RunRow::Preset { host: &h.host, preset: p, stale: self.host_stale(&h.host) });
                }
            }
        }
        for r in self.workload_rows() {
            rows.push(RunRow::Workload(r));
        }
        rows
    }

    /// One selectable row of the Catalog screen: a record, or a weight the
    /// latest scan found uncataloged (the `a` key records it).
    pub fn catalog_rows(&self) -> Vec<CatalogRow<'_>> {
        let mut rows = Vec::new();
        for h in &self.catalog {
            if let Ok(c) = &h.result {
                for m in &c.models {
                    rows.push(CatalogRow::Model { host: &h.host, model: m, stale: self.host_stale(&h.host) });
                }
            }
        }
        for (host, scan) in &self.scans {
            if let Ok(s) = scan {
                for f in s.found.iter().filter(|f| f.record.is_none()) {
                    rows.push(CatalogRow::Found { host, found: f, stale: self.host_stale(host) });
                }
            }
        }
        rows
    }

    /// Workload lines across hosts, in Hosts-screen order.
    pub fn workload_rows(&self) -> Vec<WorkloadRow<'_>> {
        let mut rows = Vec::new();
        for h in &self.hosts {
            if let Some(s) = h.shown() {
                for w in &s.workloads {
                    rows.push(WorkloadRow { host: &h.name, workload: w, stale: h.is_stale() });
                }
            }
        }
        rows
    }

    /// Hosts whose last status answered: the other screens ask only these,
    /// so a sleeping Mac costs nothing (and is not woken) until it answers
    /// the status call again.
    pub fn reachable_hosts(&self) -> Vec<String> {
        self.hosts.iter().filter(|h| !h.is_stale()).map(|h| h.name.clone()).collect()
    }

    pub fn host_stale(&self, host: &str) -> bool {
        self.hosts.iter().find(|h| h.name == host).is_none_or(|h| h.is_stale())
    }

    /// Lane ids per (host, workload), from the last GPU/memory answer.
    pub fn lanes_of(&self, host: &str, workload: &str) -> Vec<String> {
        self.gpu
            .iter()
            .filter(|g| g.host == host)
            .flat_map(|g| g.lanes.iter())
            .filter(|l| l.workload == workload)
            .flat_map(|l| l.lane_ids.iter().cloned())
            .collect()
    }

    /// Models an engine reports loaded, per (host, workload): extra rows
    /// under the workload on the Hosts screen (llama.cpp router models,
    /// Ollama models, llama-swap lanes).
    pub fn loaded_under(&self, host: &str, workload: &str) -> Vec<String> {
        let mut out = Vec::new();
        for g in self.gpu.iter().filter(|g| g.host == host) {
            for e in g.engines.iter().filter(|e| e.workload == workload) {
                out.extend(e.loaded.clone().unwrap_or_default());
            }
            for l in g.lanes.iter().filter(|l| l.workload == workload) {
                out.extend(l.running.clone().unwrap_or_default());
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// The logs picker: every workload, then each lane of a lane workload.
    pub fn log_targets(&self) -> Vec<LogTarget> {
        let mut out = Vec::new();
        for r in self.workload_rows() {
            out.push(LogTarget { host: r.host.to_string(), workload: r.workload.name.clone(), lane: None });
            for lane in self.lanes_of(r.host, &r.workload.name) {
                out.push(LogTarget {
                    host: r.host.to_string(),
                    workload: r.workload.name.clone(),
                    lane: Some(lane),
                });
            }
        }
        out
    }

    pub fn fmt_time(&self, t: DateTime<Utc>, fmt: &str) -> String {
        if self.local_time {
            t.with_timezone(&chrono::Local).format(fmt).to_string()
        } else {
            t.format(fmt).to_string()
        }
    }

    pub fn ago(&self, t: DateTime<Utc>) -> String {
        let secs = (self.now - t).num_seconds().max(0);
        if secs < 60 {
            format!("{secs} s ago")
        } else if secs < 3600 {
            format!("{} min ago", secs / 60)
        } else if secs < 86_400 {
            format!("{} h ago", secs / 3600)
        } else {
            format!("{} d ago", secs / 86_400)
        }
    }

    // ---- time ---------------------------------------------------------------

    /// Periodic work for the shown screen. Called by the loop on every
    /// pass; returns at most one request per kind, never while the same
    /// kind is still in flight.
    pub fn tick(&mut self, now: Instant) -> Vec<Request> {
        let mut out = Vec::new();
        if self.hosts_pace.due(now, HOSTS_EVERY) {
            out.push(self.hosts_request(now, false));
        }
        if self.hosts.is_empty() {
            // Nothing else can be asked before the first status answer.
            return out;
        }
        match self.screen {
            Screen::Ops => {
                let every = if self.ops.iter().any(|(_, o)| !o.status.is_terminal()) {
                    OPS_ACTIVE_EVERY
                } else {
                    OPS_EVERY
                };
                if self.ops_pace.due(now, every) {
                    out.push(self.ops_request(now));
                }
            }
            Screen::Audit if self.audit_pace.due(now, AUDIT_EVERY) => out.push(self.audit_request(now)),
            Screen::Gpu if self.gpu_pace.due(now, GPU_EVERY) => out.push(self.gpu_request(now)),
            Screen::Run if self.presets_pace.due(now, PRESETS_EVERY) => {
                out.push(self.presets_request(now))
            }
            Screen::Catalog if self.catalog_pace.due(now, CATALOG_EVERY) => {
                out.push(self.catalog_request(now))
            }
            Screen::Logs => {
                // Lanes come with the memory answer; ask once for the picker.
                if self.gpu.is_empty() && self.gpu_pace.due(now, GPU_EVERY) {
                    out.push(self.gpu_request(now));
                }
                if let Some(r) = self.logs_poll(now) {
                    out.push(r);
                }
            }
            _ => {}
        }
        out
    }

    fn hosts_request(&mut self, now: Instant, force: bool) -> Request {
        self.hosts_pace.start(now);
        let skip = if force || self.first_hosts {
            Vec::new()
        } else {
            let wall = self.now;
            self.hosts
                .iter()
                .filter(|h| h.is_stale())
                .filter(|h| {
                    (wall - h.fetched_at).to_std().unwrap_or_default() < STALE_RETRY
                })
                .map(|h| h.name.clone())
                .collect()
        };
        self.first_hosts = false;
        Request::RefreshHosts { skip }
    }

    fn ops_request(&mut self, now: Instant) -> Request {
        self.ops_pace.start(now);
        Request::RefreshOps { hosts: self.reachable_hosts() }
    }

    fn audit_request(&mut self, now: Instant) -> Request {
        self.audit_pace.start(now);
        Request::RefreshAudit { hosts: self.reachable_hosts() }
    }

    fn gpu_request(&mut self, now: Instant) -> Request {
        self.gpu_pace.start(now);
        let hosts = self
            .hosts
            .iter()
            .filter(|h| !h.is_stale())
            .map(|h| {
                let names = h
                    .shown()
                    .map(|s| s.workloads.iter().map(|w| w.name.clone()).collect())
                    .unwrap_or_default();
                (h.name.clone(), names)
            })
            .collect();
        Request::RefreshGpu { hosts }
    }

    fn presets_request(&mut self, now: Instant) -> Request {
        self.presets_pace.start(now);
        Request::RefreshPresets { hosts: self.reachable_hosts() }
    }

    fn catalog_request(&mut self, now: Instant) -> Request {
        self.catalog_pace.start(now);
        Request::RefreshCatalog { hosts: self.reachable_hosts() }
    }

    fn logs_poll(&mut self, now: Instant) -> Option<Request> {
        let target = self.logs.target.clone()?;
        if !self.logs.follow || self.logs.in_flight {
            return None;
        }
        if self.logs.last_req.is_some_and(|t| now.duration_since(t) < LOGS_FOLLOW_EVERY) {
            return None;
        }
        Some(self.logs_request(now, target))
    }

    fn logs_request(&mut self, now: Instant, target: LogTarget) -> Request {
        self.logs.in_flight = true;
        self.logs.last_req = Some(now);
        let cursor = if target.lane.is_some() { None } else { self.logs.cursor.clone() };
        Request::Logs { gen: self.logs.gen, target, cursor }
    }

    /// `r`: ask the shown screen again now, unreachable hosts included.
    fn refresh_now(&mut self, now: Instant) -> Vec<Request> {
        let mut out = Vec::new();
        if !self.hosts_pace.in_flight {
            out.push(self.hosts_request(now, true));
        }
        match self.screen {
            Screen::Ops if !self.ops_pace.in_flight => out.push(self.ops_request(now)),
            Screen::Audit if !self.audit_pace.in_flight => out.push(self.audit_request(now)),
            Screen::Gpu if !self.gpu_pace.in_flight => out.push(self.gpu_request(now)),
            Screen::Run if !self.presets_pace.in_flight => out.push(self.presets_request(now)),
            Screen::Catalog if !self.catalog_pace.in_flight => out.push(self.catalog_request(now)),
            Screen::Logs => {
                if let Some(t) = self.logs.target.clone() {
                    if !self.logs.in_flight {
                        out.push(self.logs_request(now, t));
                    }
                }
            }
            _ => {}
        }
        out
    }

    // ---- keys ---------------------------------------------------------------

    pub fn on_key(&mut self, key: KeyEvent, now: Instant) -> Vec<Request> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.quit = true;
            return Vec::new();
        }
        if self.modal.is_some() {
            return self.modal_key(key);
        }
        match key.code {
            KeyCode::Char('q') => {
                self.quit = true;
                Vec::new()
            }
            KeyCode::Char('?') => {
                self.modal = Some(Modal::Help);
                Vec::new()
            }
            KeyCode::Char('r') => self.refresh_now(now),
            KeyCode::Esc => {
                self.message = None;
                Vec::new()
            }
            KeyCode::Char(c) if Screen::from_key(c).is_some() => {
                self.screen = Screen::from_key(c).expect("checked");
                // Show what is there at once; ask for fresh data on entry.
                self.on_enter(now)
            }
            _ => self.screen_key(key, now),
        }
    }

    fn on_enter(&mut self, now: Instant) -> Vec<Request> {
        let mut out = Vec::new();
        if self.hosts.is_empty() {
            return out;
        }
        match self.screen {
            Screen::Ops if !self.ops_pace.in_flight => out.push(self.ops_request(now)),
            Screen::Audit if !self.audit_pace.in_flight => out.push(self.audit_request(now)),
            Screen::Gpu if !self.gpu_pace.in_flight => out.push(self.gpu_request(now)),
            Screen::Run if !self.presets_pace.in_flight => out.push(self.presets_request(now)),
            Screen::Catalog if !self.catalog_pace.in_flight => out.push(self.catalog_request(now)),
            _ => {}
        }
        out
    }

    fn screen_key(&mut self, key: KeyEvent, now: Instant) -> Vec<Request> {
        let (up, down) = (
            matches!(key.code, KeyCode::Up | KeyCode::Char('k')),
            matches!(key.code, KeyCode::Down | KeyCode::Char('j')),
        );
        match self.screen {
            Screen::Hosts => {
                let n = self.workload_rows().len();
                step(&mut self.hosts_sel, n, up, down);
                if key.code == KeyCode::Enter {
                    // Enter on a workload: its log, following.
                    let picked = self
                        .workload_rows()
                        .get(self.hosts_sel)
                        .map(|r| LogTarget { host: r.host.to_string(), workload: r.workload.name.clone(), lane: None });
                    if let Some(t) = picked {
                        self.screen = Screen::Logs;
                        let idx = self.log_targets().iter().position(|x| *x == t).unwrap_or(0);
                        self.logs.picker_sel = idx;
                        return self.open_logs(t, now);
                    }
                }
                Vec::new()
            }
            Screen::Run => {
                let n = self.run_rows().len();
                step(&mut self.run_sel, n, up, down);
                // Owned before any &mut self call: which row is selected.
                enum Picked {
                    Preset { host: String, alias: String, stale: bool },
                    Workload,
                    None,
                }
                let rows = self.run_rows();
                let picked = match rows.get(self.run_sel) {
                    Some(RunRow::Preset { host, preset, stale }) => Picked::Preset {
                        host: host.to_string(),
                        alias: preset.alias.clone(),
                        stale: *stale,
                    },
                    Some(RunRow::Workload(_)) => Picked::Workload,
                    None => Picked::None,
                };
                let kind = match key.code {
                    KeyCode::Enter => Some(OpKind::Run),
                    KeyCode::Char('S') => Some(OpKind::Stop),
                    KeyCode::Char('T') => Some(OpKind::Restart),
                    _ => None,
                };
                match (picked, kind) {
                    (Picked::Preset { host, alias, stale }, Some(OpKind::Run)) => {
                        self.start_preset_preview(&host, &alias, stale)
                    }
                    (Picked::Preset { alias, .. }, Some(_)) => {
                        self.message = Some(format!(
                            "preset {alias}: enter runs it (a preset is always a run)"
                        ));
                        Vec::new()
                    }
                    (Picked::Workload, Some(kind)) => self.start_preview(kind),
                    _ => Vec::new(),
                }
            }
            Screen::Ops => {
                step(&mut self.ops_sel, self.ops.len(), up, down);
                match key.code {
                    KeyCode::Char('C') => self.confirm_op_write(true),
                    KeyCode::Char('A') => self.confirm_op_write(false),
                    _ => Vec::new(),
                }
            }
            Screen::Audit => {
                let n: usize = self
                    .audit
                    .iter()
                    .map(|h| h.result.as_ref().map(|f| f.len()).unwrap_or(0))
                    .sum();
                step(&mut self.audit_sel, n, up, down);
                Vec::new()
            }
            Screen::Logs => {
                let n = self.log_targets().len();
                step(&mut self.logs.picker_sel, n, up, down);
                match key.code {
                    KeyCode::Enter => match self.log_targets().get(self.logs.picker_sel).cloned() {
                        Some(t) => self.open_logs(t, now),
                        None => Vec::new(),
                    },
                    KeyCode::Char('f') => {
                        self.logs.follow = !self.logs.follow;
                        if self.logs.follow {
                            self.logs.scroll = 0;
                        }
                        Vec::new()
                    }
                    KeyCode::PageUp => {
                        self.logs.scroll = (self.logs.scroll + 20).min(self.logs.lines.len());
                        self.logs.follow = false;
                        Vec::new()
                    }
                    KeyCode::PageDown => {
                        self.logs.scroll = self.logs.scroll.saturating_sub(20);
                        Vec::new()
                    }
                    KeyCode::End => {
                        self.logs.scroll = 0;
                        Vec::new()
                    }
                    _ => Vec::new(),
                }
            }
            Screen::Gpu => Vec::new(),
            Screen::Catalog => {
                let n = self.catalog_rows().len();
                step(&mut self.catalog_sel, n, up, down);
                // Owned before any &mut self call: which row is selected.
                enum Picked {
                    Model { host: String },
                    Found { host: String, found: ScanFound, stale: bool },
                    None,
                }
                let rows = self.catalog_rows();
                let picked = match rows.get(self.catalog_sel) {
                    Some(CatalogRow::Model { host, .. }) => Picked::Model { host: host.to_string() },
                    Some(CatalogRow::Found { host, found, stale }) => Picked::Found {
                        host: host.to_string(),
                        found: (*found).clone(),
                        stale: *stale,
                    },
                    None => Picked::None,
                };
                match (key.code, picked) {
                    // Capitals act (the app's write-key convention): the
                    // lowercase `d`/`a` are the Hosts and Audit screen keys.
                    (KeyCode::Char('E'), Picked::Model { .. }) => self.confirm_model_write(true),
                    (KeyCode::Char('D'), Picked::Model { .. }) => self.confirm_model_write(false),
                    // A scan with `add` false is a read: allowed even
                    // read-only (`R`, which records, stays off).
                    (KeyCode::Char('s'), Picked::Model { host, .. })
                    | (KeyCode::Char('s'), Picked::Found { host, .. }) => {
                        self.message = Some(format!("{host}: scanning the stores…"));
                        vec![Request::ModelScan { host }]
                    }
                    (KeyCode::Char('R'), Picked::Found { host, found, stale }) => {
                        self.confirm_model_add(&host, &found, stale)
                    }
                    (KeyCode::Char('R'), _) => {
                        self.message = Some("R records a scanned weight; s scans first".into());
                        Vec::new()
                    }
                    _ => Vec::new(),
                }
            }
        }
    }

    fn open_logs(&mut self, target: LogTarget, now: Instant) -> Vec<Request> {
        if self.host_stale(&target.host) {
            self.message = Some(format!("{} is STALE (no answer): logs need the agent", target.host));
            return Vec::new();
        }
        self.logs.gen += 1;
        self.logs.target = Some(target.clone());
        self.logs.lines.clear();
        self.logs.cursor = None;
        self.logs.error = None;
        self.logs.scroll = 0;
        self.logs.last_run = None;
        self.logs.in_flight = false;
        vec![self.logs_request(now, target)]
    }

    fn start_preview(&mut self, kind: OpKind) -> Vec<Request> {
        let Some((host, workload, stale)) = self
            .run_rows()
            .get(self.run_sel)
            .and_then(|r| match r {
                RunRow::Workload(w) => Some((w.host.to_string(), w.workload.name.clone(), w.stale)),
                RunRow::Preset { .. } => None,
            })
        else {
            return Vec::new();
        };
        if self.read_only {
            self.message = Some("read-only (--read-only): writes are off".into());
            return Vec::new();
        }
        if stale {
            self.message = Some(format!(
                "{host} is STALE (no answer): no plan against cached observations"
            ));
            return Vec::new();
        }
        self.modal = Some(Modal::Preview {
            host: host.clone(),
            kind,
            workload: workload.clone(),
            alias: None,
            profile: None,
            plan: None,
            refused: None,
        });
        vec![Request::Plan { host, kind, workload, profile: None }]
    }

    /// Enter on a preset: resolve it on the agent (read-only), then the
    /// preview shows the plan for its workload with its profile.
    fn start_preset_preview(&mut self, host: &str, alias: &str, stale: bool) -> Vec<Request> {
        if self.read_only {
            self.message = Some("read-only (--read-only): writes are off".into());
            return Vec::new();
        }
        if stale {
            self.message =
                Some(format!("{host} is STALE (no answer): no plan against cached observations"));
            return Vec::new();
        }
        let (host, alias) = (host.to_string(), alias.to_string());
        self.modal = Some(Modal::Preview {
            host: host.clone(),
            kind: OpKind::Run,
            workload: String::new(),
            alias: Some(alias.clone()),
            profile: None,
            plan: None,
            refused: None,
        });
        vec![Request::PresetPlan { host, alias }]
    }

    /// `e`/`d` on a catalog record: enable or disable, behind y/n.
    fn confirm_model_write(&mut self, enable: bool) -> Vec<Request> {
        let rows = self.catalog_rows();
        let Some(CatalogRow::Model { host, model, stale }) = rows.get(self.catalog_sel) else {
            return Vec::new();
        };
        if self.read_only {
            self.message = Some("read-only (--read-only): writes are off".into());
            return Vec::new();
        }
        if *stale {
            self.message = Some(format!("{host} is STALE (no answer): no write against cached records"));
            return Vec::new();
        }
        let verb = if enable { "Enable" } else { "Disable" };
        let on_yes = Request::ModelEnable {
            host: host.to_string(),
            model: model.id.clone(),
            enabled: enable,
        };
        self.modal = Some(Modal::Confirm {
            title: verb.to_string(),
            lines: vec![
                format!("{verb} {} on {host}?", model.name),
                format!("engine {} · store {}", model.engine, model.store.as_deref().unwrap_or("-")),
                "A disabled record is kept, hidden from `model list`, exported inactive.".into(),
            ],
            on_yes,
        });
        Vec::new()
    }

    /// `a` on a scanned weight the catalog does not name: record it,
    /// behind y/n. Every field comes from the scan; nothing to type.
    fn confirm_model_add(&mut self, host: &str, found: &ScanFound, stale: bool) -> Vec<Request> {
        if self.read_only {
            self.message = Some("read-only (--read-only): writes are off".into());
            return Vec::new();
        }
        if stale {
            self.message = Some(format!("{host} is STALE (no answer): no write against cached records"));
            return Vec::new();
        }
        let on_yes = Request::ModelAdd { host: host.to_string(), found: found.clone() };
        self.modal = Some(Modal::Confirm {
            title: "Record model".into(),
            lines: vec![
                format!("Record {} on {host}?", found.source),
                format!("engine {} · store {} · {}", found.engine, found.store, found.path),
                "The name is what the engine calls it; edit it later with `model edit`.".into(),
            ],
            on_yes,
        });
        Vec::new()
    }

    fn confirm_op_write(&mut self, cancel: bool) -> Vec<Request> {
        let Some((host, op)) = self.ops.get(self.ops_sel).cloned() else {
            return Vec::new();
        };
        if self.read_only {
            self.message = Some("read-only (--read-only): writes are off".into());
            return Vec::new();
        }
        if cancel && op.status.is_terminal() {
            self.message = Some(format!("{} is {}: nothing to cancel", short(&op.id), op.status));
            return Vec::new();
        }
        if !cancel && (op.acknowledged || !op.status.is_terminal()) {
            self.message = Some(format!(
                "{} is {}{}: nothing to acknowledge",
                short(&op.id),
                op.status,
                if op.acknowledged { ", already acknowledged" } else { "" }
            ));
            return Vec::new();
        }
        let verb = if cancel { "Cancel" } else { "Acknowledge" };
        let mut lines = vec![
            format!("{verb} operation {} on {host}?", op.id),
            format!("{} {} — {}", op.kind, op.workload, op.status),
        ];
        if cancel {
            lines.push("The executor stops before its next external effect;".into());
            lines.push("the operation is canceled only once it records that.".into());
        } else {
            lines.push("Acknowledging says a human has seen this outcome and".into());
            lines.push("unblocks mutations it was blocking.".into());
        }
        let on_yes = if cancel {
            Request::Cancel { host, id: op.id.clone() }
        } else {
            Request::Ack { host, id: op.id.clone() }
        };
        self.modal = Some(Modal::Confirm { title: verb.to_string(), lines, on_yes });
        Vec::new()
    }

    fn modal_key(&mut self, key: KeyEvent) -> Vec<Request> {
        let yes = matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y'));
        let no = matches!(key.code, KeyCode::Char('n') | KeyCode::Esc | KeyCode::Char('q'));
        match self.modal.take() {
            None => Vec::new(),
            Some(Modal::Help) => Vec::new(),
            Some(Modal::Confirm { title, lines, on_yes }) => {
                if yes {
                    self.message = Some(format!("{title}: sent, waiting for the agent"));
                    vec![on_yes]
                } else if no {
                    self.message = Some(format!("{title}: not sent"));
                    Vec::new()
                } else {
                    self.modal = Some(Modal::Confirm { title, lines, on_yes });
                    Vec::new()
                }
            }
            Some(Modal::Preview { host, kind, workload, alias, profile, plan, refused }) => {
                // `s` on a preset whose profile drifted: offer the sync (a
                // write: comments kept, previous bytes beside the file).
                let drift = matches!(&plan, Some(Err(e)) if e.is(crate::api::DRIFT_CODE));
                if key.code == KeyCode::Char('s') && drift && alias.is_some() {
                    if self.read_only {
                        self.message = Some("read-only (--read-only): writes are off".into());
                        self.modal = Some(Modal::Preview { host, kind, workload, alias, profile, plan, refused });
                        return Vec::new();
                    }
                    let a = alias.clone().expect("checked");
                    self.pending_preset = Some((host.clone(), a.clone()));
                    self.modal = Some(Modal::Confirm {
                        title: "Sync preset".into(),
                        lines: vec![
                            format!("Write preset {a}'s values into its router profile env on {host}?"),
                            "Values merged, comments kept, previous bytes kept beside it.".into(),
                            "The plan then previews again with the synced profile.".into(),
                        ],
                        on_yes: Request::PresetSync { host, alias: a },
                    });
                    return Vec::new();
                }
                match (&plan, yes, no) {
                    (Some(Ok(p)), true, _) => {
                        let req = Request::Admit {
                            host: host.clone(),
                            kind,
                            workload: workload.clone(),
                            revision: p.revision,
                            stop_targets: p.stop_targets.clone(),
                            profile: profile.clone(),
                        };
                        self.modal =
                            Some(Modal::Progress { host, kind, workload, op: None, error: None });
                        vec![req]
                    }
                    (_, _, true) => {
                        self.message = Some(match &alias {
                            Some(a) => format!("preset {a}: not admitted"),
                            None => format!("{kind} {workload}: not admitted"),
                        });
                        Vec::new()
                    }
                    _ => {
                        self.modal = Some(Modal::Preview {
                            host,
                            kind,
                            workload,
                            alias,
                            profile,
                            plan,
                            refused,
                        });
                        Vec::new()
                    }
                }
            }
            Some(m @ Modal::Progress { .. }) => {
                if !(no || key.code == KeyCode::Enter) {
                    self.modal = Some(m);
                }
                Vec::new()
            }
        }
    }

    // ---- answers ------------------------------------------------------------

    pub fn on_msg(&mut self, msg: Msg, now: Instant) -> Vec<Request> {
        match msg {
            Msg::HostsDone => {
                self.hosts_pace.in_flight = false;
                Vec::new()
            }
            Msg::Hosts(snaps) => {
                for s in snaps {
                    match self.hosts.iter_mut().find(|h| h.name == s.name) {
                        Some(h) => *h = s,
                        None => self.hosts.push(s),
                    }
                }
                let order = &self.host_order;
                self.hosts.sort_by_key(|h| order.iter().position(|n| *n == h.name).unwrap_or(usize::MAX));
                let n = self.workload_rows().len();
                clamp(&mut self.hosts_sel, n);
                let run_n = self.run_rows().len();
                clamp(&mut self.run_sel, run_n);
                // The first answer unblocks the shown screen's own request.
                self.on_enter_if_idle(now)
            }
            Msg::Ops(per_host) => {
                self.ops_pace.in_flight = false;
                let mut ops = Vec::new();
                let mut errors = Vec::new();
                for h in per_host {
                    match h.result {
                        Ok(list) => ops.extend(list.into_iter().map(|o| (h.host.clone(), o))),
                        Err(e) => errors.push((h.host, e)),
                    }
                }
                // Running first, then newest first.
                ops.sort_by(|(_, a), (_, b)| {
                    a.status.is_terminal().cmp(&b.status.is_terminal()).then(b.created_at.cmp(&a.created_at))
                });
                let keep = self.ops.get(self.ops_sel).map(|(h, o)| (h.clone(), o.id.clone()));
                self.ops = ops;
                self.ops_errors = errors;
                if let Some((h, id)) = keep {
                    if let Some(i) = self.ops.iter().position(|(hh, o)| *hh == h && o.id == id) {
                        self.ops_sel = i;
                    }
                }
                clamp(&mut self.ops_sel, self.ops.len());
                Vec::new()
            }
            Msg::Audit(per_host) => {
                self.audit_pace.in_flight = false;
                self.audit = per_host;
                Vec::new()
            }
            Msg::Gpu(per_host) => {
                self.gpu_pace.in_flight = false;
                self.gpu = per_host;
                Vec::new()
            }
            Msg::Logs { gen, target, result } => {
                if gen != self.logs.gen {
                    return Vec::new();
                }
                self.logs.in_flight = false;
                match result {
                    Ok(LogPage::Unit { entries, cursor, more }) => {
                        self.logs.error = None;
                        for e in &entries {
                            self.push_entry(e);
                        }
                        trim_front(&mut self.logs.lines, LOG_KEEP);
                        if cursor.is_some() {
                            self.logs.cursor = cursor;
                        }
                        if more && self.logs.follow {
                            // A full page: the next one may already wait.
                            return vec![self.logs_request(now, target)];
                        }
                    }
                    Ok(LogPage::Lane { lines }) => {
                        self.logs.error = None;
                        self.logs.lines = lines.iter().map(|l| crate::ui::clean(l)).collect();
                        trim_front(&mut self.logs.lines, LOG_KEEP);
                    }
                    Err(e) => self.logs.error = Some(e.to_string()),
                }
                Vec::new()
            }
            Msg::Planned { host, kind, workload, result } => {
                if let Some(Modal::Preview { host: h, kind: k, workload: w, plan, .. }) = &mut self.modal {
                    if *h == host && *k == kind && *w == workload {
                        *plan = Some(result);
                    }
                }
                Vec::new()
            }
            Msg::StaleRefused { host, kind, workload, reason, fresh } => {
                // The old confirmation is void: show the fresh plan and ask
                // again. Nothing ran. A preset preview keeps its alias and
                // profile (the fresh plan may change them: re-resolving is
                // the `s`/retry path, not this one).
                let (alias, profile) = match &self.modal {
                    Some(Modal::Preview { host: h, kind: k, workload: w, alias, profile, .. })
                        if *h == host && *k == kind && *w == workload =>
                    {
                        (alias.clone(), profile.clone())
                    }
                    _ => (None, None),
                };
                self.modal = Some(Modal::Preview {
                    host,
                    kind,
                    workload,
                    alias,
                    profile,
                    plan: Some(fresh),
                    refused: Some(reason),
                });
                Vec::new()
            }
            Msg::Admitted { host, result } => {
                match result {
                    Ok((op, replayed)) => {
                        self.message = Some(format!(
                            "{} operation {} on {host}",
                            if replayed { "replaying existing" } else { "admitted" },
                            short(&op.id)
                        ));
                        self.upsert_op(&host, &op);
                        if let Some(Modal::Progress { host: h, op: slot, .. }) = &mut self.modal {
                            if *h == host {
                                *slot = Some(op);
                            }
                        }
                    }
                    Err(e) => {
                        self.message = Some(format!("not admitted: {e}"));
                        if let Some(Modal::Progress { error, .. }) = &mut self.modal {
                            *error = Some(e.to_string());
                        }
                    }
                }
                Vec::new()
            }
            Msg::OpUpdate { host, op } => {
                self.upsert_op(&host, &op);
                if let Some(Modal::Progress { host: h, op: slot, .. }) = &mut self.modal {
                    if *h == host && slot.as_ref().is_none_or(|o| o.id == op.id) {
                        *slot = Some(op.clone());
                    }
                }
                if op.status.is_terminal() {
                    self.message = Some(format!("{} {} on {host}: {}", op.kind, op.workload, op.status));
                }
                Vec::new()
            }
            Msg::WriteDone { host, what, result } => {
                match result {
                    Ok(op) => {
                        self.message = Some(match what {
                            "cancel" => format!(
                                "cancellation requested for {} on {host} (status {}) — canceled only once the executor records it",
                                short(&op.id),
                                op.status
                            ),
                            _ => format!("acknowledged {} on {host} ({})", short(&op.id), op.status),
                        });
                        self.upsert_op(&host, &op);
                    }
                    Err(e) => self.message = Some(format!("{what} refused: {e}")),
                }
                Vec::new()
            }
            Msg::Presets(per_host) => {
                self.presets_pace.in_flight = false;
                self.presets = per_host;
                let n = self.run_rows().len();
                clamp(&mut self.run_sel, n);
                Vec::new()
            }
            Msg::Catalog(per_host) => {
                self.catalog_pace.in_flight = false;
                // Drop scan rows of hosts whose catalog answer failed; a
                // write against them would race a stale view.
                self.catalog = per_host;
                let live: Vec<String> =
                    self.catalog.iter().filter(|h| h.result.is_ok()).map(|h| h.host.clone()).collect();
                self.scans.retain(|(h, _)| live.contains(h));
                let n = self.catalog_rows().len();
                clamp(&mut self.catalog_sel, n);
                Vec::new()
            }
            Msg::PresetPlanned { host, alias, result } => {
                if let Some(Modal::Preview { host: h, alias: a, workload, profile, plan, .. }) =
                    &mut self.modal
                {
                    if *h == host && a.as_deref() == Some(alias.as_str()) {
                        match result {
                            Ok(boxed) => {
                                let resolved = *boxed;
                                *workload = resolved.workload;
                                *profile = resolved.profile;
                                *plan = Some(resolved.plan);
                            }
                            Err(e) => *plan = Some(Err(e)),
                        }
                    }
                }
                Vec::new()
            }
            Msg::ScanDone { host, result } => {
                match result {
                    Ok(s) => {
                        let new: Vec<&str> = s.found.iter().filter(|f| f.record.is_none()).map(|f| f.source.as_str()).collect();
                        self.message = Some(format!(
                            "{host}: {} weights in the stores, {} not cataloged{}",
                            s.found.len(),
                            new.len(),
                            if s.missing.is_empty() {
                                String::new()
                            } else {
                                format!(", {} records point at weights that are gone", s.missing.len())
                            }
                        ));
                        match self.scans.iter_mut().find(|(h, _)| h == &host) {
                            Some(slot) => *slot = (host, Ok(s)),
                            None => self.scans.push((host, Ok(s))),
                        }
                        let n = self.catalog_rows().len();
                        clamp(&mut self.catalog_sel, n);
                    }
                    Err(e) => {
                        self.message = Some(format!("{host}: scan failed: {e}"));
                        match self.scans.iter_mut().find(|(h, _)| h == &host) {
                            Some(slot) => *slot = (host.clone(), Err(e.to_string())),
                            None => self.scans.push((host, Err(e.to_string()))),
                        }
                    }
                }
                Vec::new()
            }
            Msg::SimpleWrite { host, what, result } => {
                match result {
                    Ok(text) => {
                        self.message = Some(format!("{host}: {text}"));
                        if what == "sync" {
                            // The profile now renders the preset: reopen the
                            // preview (the sync's confirm closed it) and
                            // plan again.
                            if let Some((p_host, alias)) = self.pending_preset.take() {
                                if p_host == host {
                                    self.modal = Some(Modal::Preview {
                                        host: p_host.clone(),
                                        kind: OpKind::Run,
                                        workload: String::new(),
                                        alias: Some(alias.clone()),
                                        profile: None,
                                        plan: None,
                                        refused: None,
                                    });
                                    return vec![Request::PresetPlan { host: p_host, alias }];
                                }
                            }
                        }
                        if matches!(what, "enable" | "disable" | "add") {
                            self.catalog_pace.last = None;
                        }
                    }
                    Err(e) => self.message = Some(format!("{host}: {what} refused: {e}")),
                }
                Vec::new()
            }
        }
    }

    fn on_enter_if_idle(&mut self, now: Instant) -> Vec<Request> {
        let idle = match self.screen {
            Screen::Ops => self.ops_pace.last.is_none(),
            Screen::Audit => self.audit_pace.last.is_none(),
            Screen::Gpu => self.gpu_pace.last.is_none(),
            Screen::Run => self.presets_pace.last.is_none(),
            Screen::Catalog => self.catalog_pace.last.is_none(),
            _ => false,
        };
        if idle {
            self.on_enter(now)
        } else {
            Vec::new()
        }
    }

    fn upsert_op(&mut self, host: &str, op: &OperationView) {
        match self.ops.iter_mut().find(|(h, o)| h == host && o.id == op.id) {
            Some((_, o)) => *o = op.clone(),
            None => self.ops.insert(0, (host.to_string(), op.clone())),
        }
    }

    fn push_entry(&mut self, e: &LogEntry) {
        if e.invocation.is_some() && e.invocation != self.logs.last_run {
            let id = e.invocation.as_deref().unwrap_or_default();
            self.logs.lines.push(format!("-- run {} --", &id[..id.len().min(8)]));
            self.logs.last_run = e.invocation.clone();
        }
        let when = self.fmt_time(e.timestamp, "%b %d %H:%M:%S");
        let est = if e.time_estimated { "~" } else { "" };
        let who = match e.source {
            LogSource::Manager => "manager: ",
            LogSource::Unit => "",
        };
        let cut = if e.truncated { " […truncated]" } else { "" };
        let msg = crate::ui::clean(&e.message);
        self.logs.lines.push(format!("{est}{when} {who}{msg}{cut}"));
    }
}

/// An operation still able to move (for the progress modal's footer).
pub fn op_active(op: &OperationView) -> bool {
    !op.status.is_terminal() || matches!(op.status, OpStatus::CancelRequested)
}

pub fn short(id: &str) -> &str {
    &id[..id.len().min(8)]
}

fn step(sel: &mut usize, n: usize, up: bool, down: bool) {
    if n == 0 {
        *sel = 0;
        return;
    }
    if up {
        *sel = sel.saturating_sub(1);
    }
    if down {
        *sel = (*sel + 1).min(n - 1);
    }
}

fn clamp(sel: &mut usize, n: usize) {
    if n == 0 {
        *sel = 0;
    } else if *sel >= n {
        *sel = n - 1;
    }
}

fn trim_front(lines: &mut Vec<String>, keep: usize) {
    if lines.len() > keep {
        let cut = lines.len() - keep;
        lines.drain(..cut);
    }
}
