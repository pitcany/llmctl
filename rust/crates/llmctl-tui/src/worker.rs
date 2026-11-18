//! Runs [`Request`]s as tokio tasks and posts [`Msg`]s back. Every agent
//! call is bounded: reads by the status timeout, writes by a longer one
//! whose expiry counts as "outcome unknown" and goes through the CLI's
//! lost-ACK check, never as "failed".

use std::sync::Arc;
use std::time::Duration;

use llmctl_client::wire::{
    codes, AdmitRequest, LaneLogsRequest, LogsRequest, OperationView, PlanRequest,
    MAX_LOG_LINES, PROTOCOL_VERSION,
};
use llmctl_core::OpKind;
use tokio::runtime::Handle;
use tokio::sync::mpsc::UnboundedSender;

use crate::api::{AgentApi, ApiErr, ApiResult};
use crate::app::{HostGpu, HostResult, LaneHost, LogPage, LogTarget, Msg, Request};
use crate::fleet::{fetch_one, HostTarget};

/// Bound on one write (admit, cancel, acknowledge).
pub const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the progress view follows an admitted operation.
pub const WATCH_BUDGET: Duration = Duration::from_secs(30 * 60);
pub const WATCH_EVERY: Duration = Duration::from_millis(500);
/// Lines asked for when a log is opened, and per lane-log poll.
pub const LOG_FIRST_PAGE: u32 = 200;

#[derive(Clone)]
pub struct Worker {
    targets: Arc<Vec<HostTarget>>,
    timeout: Duration,
    tx: UnboundedSender<Msg>,
    handle: Handle,
}

async fn bounded<T>(limit: Duration, f: impl std::future::Future<Output = ApiResult<T>>) -> ApiResult<T> {
    match tokio::time::timeout(limit, f).await {
        Ok(r) => r,
        Err(_) => Err(ApiErr::transport(format!("no answer within {} s", limit.as_secs_f64()))),
    }
}

impl Worker {
    pub fn new(targets: Vec<HostTarget>, timeout: Duration, tx: UnboundedSender<Msg>, handle: Handle) -> Self {
        Self { targets: Arc::new(targets), timeout, tx, handle }
    }

    pub fn host_names(&self) -> Vec<String> {
        self.targets.iter().map(|t| t.name.clone()).collect()
    }

    /// Start `req` in the background; never blocks.
    pub fn submit(&self, req: Request) {
        let me = self.clone();
        self.handle.spawn(async move { me.run(req).await });
    }

    fn api(&self, host: &str) -> Option<Arc<dyn AgentApi>> {
        self.targets.iter().find(|t| t.name == host).map(|t| t.api.clone())
    }

    fn send(&self, m: Msg) {
        let _ = self.tx.send(m);
    }

    async fn run(self, req: Request) {
        match req {
            Request::RefreshHosts { skip } => {
                // Concurrently, each bounded; each snapshot is posted as
                // it lands.
                let mut tasks = Vec::new();
                for t in self.targets.iter() {
                    if skip.contains(&t.name) {
                        continue;
                    }
                    let (t, me) = (t.clone(), self.clone());
                    tasks.push(tokio::spawn(async move {
                        let snap = fetch_one(&t, me.timeout).await;
                        me.send(Msg::Hosts(vec![snap]));
                    }));
                }
                for t in tasks {
                    let _ = t.await;
                }
                self.send(Msg::HostsDone);
            }
            Request::RefreshOps { hosts } => {
                let res = self
                    .per_host(hosts, |api, limit| async move {
                        bounded(limit, api.operations()).await.map(|r| r.operations)
                    })
                    .await;
                self.send(Msg::Ops(res));
            }
            Request::RefreshAudit { hosts } => {
                let res = self
                    .per_host(hosts, |api, limit| async move {
                        bounded(limit, api.audit()).await.map(|r| r.findings)
                    })
                    .await;
                self.send(Msg::Audit(res));
            }
            Request::RefreshGpu { hosts } => {
                let mut tasks = Vec::new();
                for (host, workloads) in hosts {
                    let Some(api) = self.api(&host) else { continue };
                    let limit = self.timeout;
                    tasks.push(tokio::spawn(async move {
                        match tokio::time::timeout(limit, gpu_of(api, host.clone(), workloads, limit)).await {
                            Ok(g) => g,
                            Err(_) => HostGpu {
                                host,
                                error: Some(format!("no answer within {} s", limit.as_secs_f64())),
                                ..HostGpu::default()
                            },
                        }
                    }));
                }
                let mut out = Vec::new();
                for t in tasks {
                    if let Ok(g) = t.await {
                        out.push(g);
                    }
                }
                self.send(Msg::Gpu(out));
            }
            Request::Logs { gen, target, cursor } => {
                let result = match self.api(&target.host) {
                    None => Err(ApiErr::transport(format!("unknown host {}", target.host))),
                    Some(api) => logs_page(api, &target, cursor, self.timeout).await,
                };
                self.send(Msg::Logs { gen, target, result });
            }
            Request::Plan { host, kind, workload } => {
                let result = match self.api(&host) {
                    None => Err(ApiErr::transport(format!("unknown host {host}"))),
                    Some(api) => bounded(self.timeout, api.plan(plan_req(kind, &workload))).await,
                };
                self.send(Msg::Planned { host, kind, workload, result });
            }
            Request::Admit { host, kind, workload, revision, stop_targets } => {
                self.admit(host, kind, workload, revision, stop_targets).await;
            }
            Request::Cancel { host, id } => {
                let result = match self.api(&host) {
                    None => Err(ApiErr::transport(format!("unknown host {host}"))),
                    Some(api) => bounded(WRITE_TIMEOUT, api.cancel(id)).await,
                };
                self.send(Msg::WriteDone { host, what: "cancel", result });
            }
            Request::Ack { host, id } => {
                let result = match self.api(&host) {
                    None => Err(ApiErr::transport(format!("unknown host {host}"))),
                    Some(api) => bounded(WRITE_TIMEOUT, api.acknowledge(id)).await,
                };
                self.send(Msg::WriteDone { host, what: "acknowledge", result });
            }
        }
    }

    /// Ask each host concurrently; a host the worker does not know is an
    /// error row, not a panic.
    async fn per_host<T, F, Fut>(&self, hosts: Vec<String>, f: F) -> Vec<HostResult<T>>
    where
        T: Send + 'static,
        F: Fn(Arc<dyn AgentApi>, Duration) -> Fut,
        Fut: std::future::Future<Output = ApiResult<T>> + Send + 'static,
    {
        let mut tasks = Vec::new();
        for host in hosts {
            match self.api(&host) {
                Some(api) => {
                    let fut = f(api, self.timeout);
                    tasks.push((host, Some(tokio::spawn(fut))));
                }
                None => tasks.push((host, None)),
            }
        }
        let mut out = Vec::new();
        for (host, task) in tasks {
            let result = match task {
                None => Err(format!("unknown host {host}")),
                Some(t) => match t.await {
                    Ok(r) => r.map_err(|e| describe_read_error(&e)),
                    Err(e) => Err(format!("task failed: {e}")),
                },
            };
            out.push(HostResult { host, result });
        }
        out
    }

    /// The CLI's admit path (`cmd_mutate`): protocol check, admit the
    /// previewed revision and stop targets under a fresh idempotency key,
    /// resolve a lost acknowledgement by key, then follow the operation.
    async fn admit(&self, host: String, kind: OpKind, workload: String, revision: u64, stop_targets: Vec<String>) {
        let Some(api) = self.api(&host) else {
            self.send(Msg::Admitted { host: host.clone(), result: Err(ApiErr::transport(format!("unknown host {host}"))) });
            return;
        };
        match bounded(self.timeout, api.health()).await {
            Ok(h) if !h.supported_protocols.contains(&PROTOCOL_VERSION) => {
                let e = ApiErr::api(
                    codes::PROTOCOL_MISMATCH,
                    format!(
                        "agent {} speaks protocol {} (supported: {:?}); this client speaks {PROTOCOL_VERSION}. \
                         Mutations refused — `llmctl-next host upgrade` the agent.",
                        h.daemon_version, h.protocol, h.supported_protocols
                    ),
                );
                self.send(Msg::Admitted { host, result: Err(e) });
                return;
            }
            Ok(_) => {}
            Err(e) => {
                self.send(Msg::Admitted { host, result: Err(e) });
                return;
            }
        }
        let key = format!(
            "{kind}-{workload}-r{revision}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or_default()
        );
        let req = AdmitRequest {
            protocol: PROTOCOL_VERSION,
            idempotency_key: key.clone(),
            kind,
            workload: workload.clone(),
            expected_revision: Some(revision),
            expected_stop_targets: Some(stop_targets),
            force: false,
            profile: None,
        };
        let mut retried = false;
        let landed: Result<(OperationView, bool), ApiErr> = loop {
            match bounded(WRITE_TIMEOUT, api.admit(req.clone())).await {
                Ok(resp) => break Ok((resp.operation, resp.replayed)),
                Err(e) if e.is(codes::STALE_PLAN) => {
                    let fresh = bounded(self.timeout, api.plan(plan_req(kind, &workload))).await;
                    self.send(Msg::StaleRefused { host, kind, workload, reason: e.message, fresh });
                    return;
                }
                Err(e) if e.transport && !retried => {
                    // Lost-ACK rule (M3 decision 4): ask by key before
                    // concluding anything.
                    match bounded(self.timeout, api.operation_by_key(key.clone())).await {
                        Ok(op) => break Ok((op, true)),
                        Err(e2) if e2.is(codes::UNKNOWN_OPERATION) => {
                            retried = true;
                            continue;
                        }
                        Err(e2) => {
                            break Err(ApiErr::transport(format!(
                                "cannot tell whether the operation landed (key {key}): first {e}; then {e2}. \
                                 Check `llmctl-next ops` once the host answers."
                            )))
                        }
                    }
                }
                Err(e) => break Err(e),
            }
        };
        let op_id = landed.as_ref().ok().map(|(op, _)| op.id.clone());
        self.send(Msg::Admitted { host: host.clone(), result: landed });
        if let Some(id) = op_id {
            self.watch(api, host, id).await;
        }
    }

    /// Follow an operation until it is terminal (or the budget runs out).
    async fn watch(&self, api: Arc<dyn AgentApi>, host: String, id: String) {
        let deadline = tokio::time::Instant::now() + WATCH_BUDGET;
        while tokio::time::Instant::now() < deadline {
            if let Ok(op) = bounded(self.timeout, api.operation(id.clone())).await {
                let done = op.status.is_terminal();
                self.send(Msg::OpUpdate { host: host.clone(), op });
                if done {
                    return;
                }
            }
            tokio::time::sleep(WATCH_EVERY).await;
        }
    }
}

fn plan_req(kind: OpKind, workload: &str) -> PlanRequest {
    PlanRequest { kind, workload: workload.to_string(), profile: None }
}

fn describe_read_error(e: &ApiErr) -> String {
    if e.message.contains("404") {
        "not supported by this agent — upgrade it".to_string()
    } else {
        e.to_string()
    }
}

async fn logs_page(
    api: Arc<dyn AgentApi>,
    target: &LogTarget,
    cursor: Option<String>,
    limit: Duration,
) -> ApiResult<LogPage> {
    match &target.lane {
        Some(lane) => {
            let req = LaneLogsRequest {
                protocol: PROTOCOL_VERSION,
                workload: target.workload.clone(),
                lane: lane.clone(),
                lines: LOG_FIRST_PAGE,
            };
            bounded(limit, api.lane_logs(req)).await.map(|r| LogPage::Lane { lines: r.lines })
        }
        None => {
            let lines = if cursor.is_none() { LOG_FIRST_PAGE } else { MAX_LOG_LINES };
            let req = LogsRequest { workload: target.workload.clone(), lines: Some(lines), after: cursor };
            bounded(limit, api.logs(req))
                .await
                .map(|r| LogPage::Unit { entries: r.entries, cursor: r.cursor, more: r.more })
        }
    }
}

/// One host's GPU/memory picture: cards and VRAM records, the lane
/// workloads' budgets and running lanes, and every engine that can say
/// what it has loaded. A workload without lane records is not a lane
/// workload (the error is the answer, not a fault).
async fn gpu_of(api: Arc<dyn AgentApi>, host: String, workloads: Vec<String>, limit: Duration) -> HostGpu {
    let vram = Some(bounded(limit, api.vram()).await.map_err(|e| e.to_string()));
    let mut lanes = Vec::new();
    let mut engines = Vec::new();
    for w in workloads {
        let (ls, er) = tokio::join!(api.lanes_status(w.clone()), api.engine_report(w.clone()));
        if let Ok(ls) = ls {
            let lane_ids: Vec<String> = ls.lanes.iter().map(|l| l.id.clone()).collect();
            let mem = api.lanes_memory(w.clone()).await.ok();
            let running = match lane_ids.first() {
                Some(first) => api.lanes_running(w.clone(), first.clone()).await.ok(),
                None => Some(Vec::new()),
            };
            lanes.push(LaneHost {
                workload: w.clone(),
                lane_ids,
                budget_gib: mem.as_ref().and_then(|m| m.budget_gib),
                peaks: mem.as_ref().map(|m| m.peaks.clone()).unwrap_or_default(),
                lines: mem.as_ref().map(|m| m.lines.clone()).unwrap_or_default(),
                over: mem.map(|m| m.over).unwrap_or_default(),
                running,
            });
        }
        if let Ok(er) = er {
            if er.loaded.is_some() {
                engines.push(er);
            }
        }
    }
    HostGpu { host, vram, lanes, engines, error: None }
}
