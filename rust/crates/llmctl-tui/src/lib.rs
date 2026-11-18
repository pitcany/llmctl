//! `llmctl-next tui` (M5e, M5 decision 5): a ratatui client of the agent
//! API and nothing more. It reads the local agent over its socket and the
//! Mac agents over the SSH bridge, through the same [`ControlClient`] the
//! CLI uses; every write is a plan preview or a confirmation followed by
//! the same admitted operation the CLI issues.
//!
//! Structure: [`app::App`] holds state and turns keys and answers into
//! [`app::Request`]s; [`worker::Worker`] runs those as tokio tasks, each
//! agent call bounded; [`ui`] draws. The loop never awaits an agent, so a
//! slow or sleeping host cannot stall the screen.
//!
//! [`ControlClient`]: llmctl_client::ControlClient

pub mod api;
pub mod app;
pub mod fleet;
pub mod ui;
pub mod worker;

use std::collections::HashMap;
use std::io;
use std::time::{Duration, Instant};

use ratatui::backend::Backend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Terminal;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};

use crate::app::{App, Msg, Request};
use crate::fleet::HostTarget;
use crate::worker::Worker;

pub struct Options {
    pub read_only: bool,
    /// Per-host bound on every read (`LLMCTL_NEXT_STATUS_TIMEOUT_S`).
    pub timeout: Duration,
}

/// The loop's state: the app, the worker, the answer queue, a terminal.
pub struct Runner<B: Backend> {
    pub app: App,
    pub worker: Worker,
    rx: UnboundedReceiver<Msg>,
    pub terminal: Terminal<B>,
    /// When each kind of request was last sent, for answer latency.
    sent: HashMap<&'static str, Instant>,
    /// (kind, latency) of answers received; key handling + frame times.
    pub answer_times: Vec<(&'static str, Duration)>,
}

fn kind_of_req(r: &Request) -> &'static str {
    match r {
        Request::RefreshHosts { .. } => "hosts",
        Request::RefreshOps { .. } => "ops",
        Request::RefreshAudit { .. } => "audit",
        Request::RefreshGpu { .. } => "gpu",
        Request::Logs { .. } => "logs",
        Request::Plan { .. } => "plan",
        Request::Admit { .. } => "admit",
        Request::Cancel { .. } => "cancel",
        Request::Ack { .. } => "ack",
    }
}

fn kind_of_msg(m: &Msg) -> &'static str {
    match m {
        Msg::Hosts(_) => "host",
        Msg::HostsDone => "hosts",
        Msg::Ops(_) => "ops",
        Msg::Audit(_) => "audit",
        Msg::Gpu(_) => "gpu",
        Msg::Logs { .. } => "logs",
        Msg::Planned { .. } => "plan",
        Msg::Admitted { .. } | Msg::StaleRefused { .. } => "admit",
        Msg::OpUpdate { .. } => "op",
        Msg::WriteDone { what: "cancel", .. } => "cancel",
        Msg::WriteDone { .. } => "ack",
    }
}

impl<B: Backend> Runner<B> {
    /// Build a runner on the current tokio runtime.
    pub fn new(targets: Vec<HostTarget>, opts: &Options, terminal: Terminal<B>) -> Self {
        let (tx, rx) = unbounded_channel();
        let worker = Worker::new(targets, opts.timeout, tx, tokio::runtime::Handle::current());
        let app = App::new(worker.host_names(), opts.read_only);
        Self { app, worker, rx, terminal, sent: HashMap::new(), answer_times: Vec::new() }
    }

    fn submit(&mut self, reqs: Vec<Request>) {
        for r in reqs {
            self.sent.insert(kind_of_req(&r), Instant::now());
            self.worker.submit(r);
        }
    }

    /// One pass: the key (if any), every answer that has arrived, the
    /// periodic requests, one frame. Never waits on an agent.
    pub fn step(&mut self, key: Option<KeyEvent>) -> io::Result<()> {
        let now = Instant::now();
        self.app.now = chrono::Utc::now();
        if let Some(k) = key {
            let reqs = self.app.on_key(k, now);
            self.submit(reqs);
        }
        while let Ok(msg) = self.rx.try_recv() {
            let kind = kind_of_msg(&msg);
            if let Some(t) = self.sent.get(kind) {
                self.answer_times.push((kind, t.elapsed()));
            }
            let reqs = self.app.on_msg(msg, now);
            self.submit(reqs);
        }
        let reqs = self.app.tick(now);
        self.submit(reqs);
        self.terminal.draw(|f| ui::draw(f, &self.app))?;
        Ok(())
    }
}

/// Run the TUI on the real terminal until `q`.
pub async fn run(targets: Vec<HostTarget>, opts: Options) -> io::Result<()> {
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let _guard = handle.enter();
        let terminal = ratatui::init();
        let mut runner = Runner::new(targets, &opts, terminal);
        let res = (|| -> io::Result<()> {
            let mut key = None;
            loop {
                runner.step(key.take())?;
                if runner.app.quit {
                    return Ok(());
                }
                if event::poll(Duration::from_millis(50))? {
                    if let Event::Key(k) = event::read()? {
                        if k.kind == KeyEventKind::Press {
                            key = Some(k);
                        }
                    }
                }
            }
        })();
        ratatui::restore();
        res
    })
    .await
    .map_err(io::Error::other)?
}

/// Percentiles of a sample (nearest rank), in milliseconds.
pub fn percentiles_ms(samples: &[Duration]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut v: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let rank = |p: f64| v[((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
    (rank(0.50), rank(0.95), v[v.len() - 1])
}

/// What `tui --bench` measured.
#[derive(Debug)]
pub struct BenchReport {
    /// Key press to the frame showing the new screen.
    pub key_to_frame: Vec<Duration>,
    /// Hosts-screen refresh: request sent to answer on screen.
    pub hosts_refresh: Vec<Duration>,
    /// Longest single loop pass while requests were outstanding.
    pub max_step: Duration,
}

/// Headless measurement against the live agents (read-only: screen
/// switches and refreshes, no writes): `rounds` cycles through the five
/// built screens on a 160x50 test terminal.
pub async fn bench(targets: Vec<HostTarget>, opts: Options, rounds: usize) -> io::Result<BenchReport> {
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || {
        let _guard = handle.enter();
        let terminal = Terminal::new(ratatui::backend::TestBackend::new(160, 50))?;
        let mut r = Runner::new(targets, &Options { read_only: true, ..opts }, terminal);
        // Wait for the first status answer (bounded by the status timeout).
        let start = Instant::now();
        while r.app.hosts.is_empty() && start.elapsed() < opts.timeout + Duration::from_secs(2) {
            r.step(None)?;
            std::thread::sleep(Duration::from_millis(5));
        }
        let mut key_to_frame = Vec::new();
        let mut max_step = Duration::ZERO;
        for _ in 0..rounds {
            for c in ['d', 'o', 'l', 'g', 'a'] {
                let t = Instant::now();
                r.step(Some(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)))?;
                key_to_frame.push(t.elapsed());
                // Let answers arrive between keys, as a person would.
                for _ in 0..4 {
                    let t = Instant::now();
                    r.step(None)?;
                    max_step = max_step.max(t.elapsed());
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        }
        // Refresh latency: `r` on Hosts, until the answer is drawn.
        r.step(Some(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::NONE)))?;
        r.answer_times.clear();
        let mut hosts_refresh = Vec::new();
        for _ in 0..rounds.clamp(1, 20) {
            let before = r.answer_times.iter().filter(|(k, _)| *k == "hosts").count();
            let t = Instant::now();
            r.step(Some(KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE)))?;
            while r.answer_times.iter().filter(|(k, _)| *k == "hosts").count() == before
                && t.elapsed() < opts.timeout + Duration::from_secs(2)
            {
                std::thread::sleep(Duration::from_millis(2));
                let s = Instant::now();
                r.step(None)?;
                max_step = max_step.max(s.elapsed());
            }
            hosts_refresh.push(t.elapsed());
        }
        Ok(BenchReport { key_to_frame, hosts_refresh, max_step })
    })
    .await
    .map_err(io::Error::other)?
}
