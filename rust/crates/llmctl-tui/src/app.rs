//! TUI state: what each screen shows, what a key does, and which agent
//! calls that asks for. Pure and synchronous: keys and agent answers come
//! in, [`Request`]s go out to the [`crate::worker::Worker`], which runs them
//! off the UI thread. Nothing here waits on the network, so a slow or
//! sleeping agent can never stall a frame.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use llmctl_client::wire::{
    AuditFinding, EngineReport, LogEntry, LogSource, OpStatus, OperationView, PlanResponse,
    VramResponse, WorkloadView,
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
    /// The plan preview (read-only on the agent).
    Plan { host: String, kind: OpKind, workload: String },
    /// Admit exactly the previewed plan: the agent refuses with
    /// `stale_plan` if the revision or the stop targets moved.
    Admit { host: String, kind: OpKind, workload: String, revision: u64, stop_targets: Vec<String> },
    Cancel { host: String, id: String },
    Ack { host: String, id: String },
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
}

#[derive(Debug, Clone)]
pub enum Modal {
    Help,
    /// A yes/no question in front of a write that has no plan (cancel,
    /// acknowledge).
    Confirm { title: String, lines: Vec<String>, on_yes: Request },
    /// The plan preview of a run/stop/restart; `y` admits it.
    Preview {
        host: String,
        kind: OpKind,
        workload: String,
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
    /// Hosts never asked yet: the first refresh asks every host.
    first_hosts: bool,
    hosts_pace: Pace,
    ops_pace: Pace,
    audit_pace: Pace,
    gpu_pace: Pace,
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
            first_hosts: true,
            hosts_pace: Pace::default(),
            ops_pace: Pace::default(),
            audit_pace: Pace::default(),
            gpu_pace: Pace::default(),
        }
    }

    // ---- derived views ------------------------------------------------------

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

    fn host_stale(&self, host: &str) -> bool {
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
                let n = self.workload_rows().len();
                step(&mut self.run_sel, n, up, down);
                let kind = match key.code {
                    KeyCode::Enter => Some(OpKind::Run),
                    KeyCode::Char('S') => Some(OpKind::Stop),
                    KeyCode::Char('T') => Some(OpKind::Restart),
                    _ => None,
                };
                match kind {
                    Some(kind) => self.start_preview(kind),
                    None => Vec::new(),
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
            Screen::Gpu | Screen::Catalog => Vec::new(),
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
            .workload_rows()
            .get(self.run_sel)
            .map(|r| (r.host.to_string(), r.workload.name.clone(), r.stale))
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
            plan: None,
            refused: None,
        });
        vec![Request::Plan { host, kind, workload }]
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
            Some(Modal::Preview { host, kind, workload, plan, refused }) => {
                match (&plan, yes, no) {
                    (Some(Ok(p)), true, _) => {
                        let req = Request::Admit {
                            host: host.clone(),
                            kind,
                            workload: workload.clone(),
                            revision: p.revision,
                            stop_targets: p.stop_targets.clone(),
                        };
                        self.modal = Some(Modal::Progress { host, kind, workload, op: None, error: None });
                        vec![req]
                    }
                    (_, _, true) => {
                        self.message = Some(format!("{kind} {workload}: not admitted"));
                        Vec::new()
                    }
                    _ => {
                        self.modal = Some(Modal::Preview { host, kind, workload, plan, refused });
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
                clamp(&mut self.run_sel, n);
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
                // again. Nothing ran.
                self.modal = Some(Modal::Preview {
                    host,
                    kind,
                    workload,
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
        }
    }

    fn on_enter_if_idle(&mut self, now: Instant) -> Vec<Request> {
        let idle = match self.screen {
            Screen::Ops => self.ops_pace.last.is_none(),
            Screen::Audit => self.audit_pace.last.is_none(),
            Screen::Gpu => self.gpu_pace.last.is_none(),
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
