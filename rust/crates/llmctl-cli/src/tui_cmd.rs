//! `llmctl-next tui` (M5e): builds the host list — the local agent over
//! its socket, every enrolled host over the SSH bridge — and hands it to
//! the `llmctl-tui` crate. Kept out of main.rs so the TUI touches the CLI
//! in three lines.

use std::sync::Arc;
use std::time::Duration;

use llmctl_client::ControlClient;
use llmctl_tui::api::{AgentApi, ClientApi, Unreachable};
use llmctl_tui::fleet::HostTarget;
use llmctl_tui::{percentiles_ms, Options};

use crate::{enroll, Cli};

fn targets(cli: &Cli, local_name: Option<String>) -> Vec<HostTarget> {
    let mut out = Vec::new();
    let local: Arc<dyn AgentApi> = match ControlClient::connect(crate::socket_path(cli)) {
        // Same headers as the CLI's own writes (lease holder, op lock).
        Ok(c) => {
            let c = match llmctl_client::lease::holder_ancestor(&crate::data_dir(cli)) {
                Some(pid) => c.under_lease(pid),
                None => c,
            };
            let c = match llmctl_client::oplock::holder_ancestor() {
                Some(pid) => c.holding_lock(pid),
                None => c,
            };
            Arc::new(ClientApi(c))
        }
        Err(e) => Arc::new(Unreachable(e.to_string())),
    };
    out.push(HostTarget {
        name: local_name.unwrap_or_else(|| "local".into()),
        api: local,
        local: true,
        cache_path: None,
        last_seen: None,
    });
    for h in enroll::load_hosts().hosts {
        out.push(HostTarget {
            api: Arc::new(ClientApi(ControlClient::connect_bridge(&h.ssh_alias))),
            cache_path: Some(enroll::status_cache_path(&h.name)),
            last_seen: h.last_seen,
            name: h.name,
            local: false,
        });
    }
    out
}

/// The local agent's host id, asked once (bounded) for the Hosts header.
async fn local_name(cli: &Cli) -> Option<String> {
    let c = ControlClient::connect(crate::socket_path(cli)).ok()?;
    tokio::time::timeout(Duration::from_secs(2), c.health()).await.ok()?.ok().map(|h| h.host)
}

pub async fn cmd_tui(cli: &Cli, read_only: bool, bench: Option<usize>) -> Result<u8, String> {
    let name = local_name(cli).await;
    let targets = targets(cli, name);
    let opts = Options { read_only, timeout: crate::remote_status_timeout() };
    match bench {
        None => {
            llmctl_tui::run(targets, opts).await.map_err(|e| format!("tui: {e}"))?;
            Ok(0)
        }
        Some(rounds) => {
            let r = llmctl_tui::bench(targets, opts, rounds).await.map_err(|e| format!("tui bench: {e}"))?;
            let (k50, k95, kmax) = percentiles_ms(&r.key_to_frame);
            let (h50, h95, hmax) = percentiles_ms(&r.hosts_refresh);
            println!("key to frame   n={:<4} p50 {k50:.2} ms  p95 {k95:.2} ms  max {kmax:.2} ms", r.key_to_frame.len());
            println!("hosts refresh  n={:<4} p50 {h50:.2} ms  p95 {h95:.2} ms  max {hmax:.2} ms", r.hosts_refresh.len());
            println!("longest loop pass while answers were pending: {:.2} ms", r.max_step.as_secs_f64() * 1000.0);
            Ok(0)
        }
    }
}
