//! Request safety (M4c item 6): before an operation stops a serving
//! workload — or unloads a model from one — ask llm-router how many
//! requests it has in flight there, wait a bounded time for them, and
//! refuse rather than cut them unless the caller forces it.
//!
//! The router is the only place that counts: it reports
//! `idle_revert.in_flight` (`{upstream URL: requests}`) on `GET /running`.
//! Only a positive count blocks. A router that is down, or one that does
//! not report counts, blocks nothing: with no router there are no routed
//! requests, and traffic that bypasses the router was never visible here.
//!
//! Defaults apply to real engines on Linux, like the lock and the lease.

use std::collections::BTreeMap;
use std::time::Duration;

use llmctl_core::{Platform, WorkloadDefinition};
use serde_json::Value;

/// `off` disables the check; a port number names the router (any engine).
pub const ROUTER_ENV: &str = "LLMCTL_NEXT_ROUTER";
/// How long admission waits for in-flight requests before refusing.
pub const WAIT_ENV: &str = "LLMCTL_NEXT_INFLIGHT_WAIT_MS";
/// llm-router's port on the workstation.
pub const DEFAULT_ROUTER_PORT: u16 = 4100;

/// The router to ask for this workload's host, if any.
pub fn router_port(platform: &Platform, def: &WorkloadDefinition) -> Option<u16> {
    match std::env::var(ROUTER_ENV) {
        Ok(v) if v.is_empty() || v == "off" => None,
        Ok(v) => v.parse().ok(),
        Err(_) => (*platform == Platform::Linux && def.engine != "fake").then_some(DEFAULT_ROUTER_PORT),
    }
}

pub fn wait_budget() -> Duration {
    std::env::var(WAIT_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(20))
}

/// In-flight requests per local port from a `/running` body. `None` when
/// the body carries no counts (a router without the idle accounting).
/// Upstreams on other machines are ignored: only this host's ports are
/// this agent's workloads.
pub fn parse_in_flight(body: &Value) -> Option<BTreeMap<u16, u64>> {
    let counts = body.get("idle_revert")?.get("in_flight")?.as_object()?;
    let mut out = BTreeMap::new();
    for (upstream, n) in counts {
        let (Some(port), Some(n)) = (local_port(upstream), n.as_u64()) else {
            continue;
        };
        *out.entry(port).or_insert(0) += n;
    }
    Some(out)
}

/// The port of a loopback URL such as `http://127.0.0.1:8003` (a path or
/// trailing slash is fine); `None` for other hosts and for garbage.
fn local_port(upstream: &str) -> Option<u16> {
    let rest = upstream.split_once("://").map_or(upstream, |(_, r)| r);
    let authority = rest.split('/').next()?;
    let (host, port) = authority.rsplit_once(':')?;
    matches!(host, "127.0.0.1" | "localhost" | "[::1]" | "0.0.0.0")
        .then(|| port.parse().ok())
        .flatten()
}

/// Ask the router. `None` = it cannot say (down, not a router, no counts).
pub async fn in_flight(router_port: u16, timeout: Duration) -> Option<BTreeMap<u16, u64>> {
    let (status, body) = tokio::time::timeout(
        timeout,
        crate::probe_http::request(router_port, hyper::Method::GET, "/running", None, timeout),
    )
    .await
    .ok()?
    .ok()?;
    if !(200..300).contains(&status) {
        return None;
    }
    parse_in_flight(&serde_json::from_slice(&body).ok()?)
}

/// The workloads among `stopping` (name, port) that have requests in
/// flight, with their counts, in the given order.
pub fn busy(counts: &BTreeMap<u16, u64>, stopping: &[(String, u16)]) -> Vec<(String, u64)> {
    stopping
        .iter()
        .filter_map(|(name, port)| {
            counts
                .get(port)
                .copied()
                .filter(|n| *n > 0)
                .map(|n| (name.clone(), n))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn counts_come_from_the_idle_accounting_keyed_by_local_port() {
        let body = json!({"loaded": [], "idle_revert": {"deferred": null, "in_flight": {
            "http://127.0.0.1:8003": 2,
            "http://localhost:8003/": 1,
            "http://127.0.0.1:8009/v1": 4,
            "http://10.0.0.144:8003": 7,
            "nonsense": 3,
            "http://127.0.0.1:8005": "x"
        }}});
        let counts = parse_in_flight(&body).unwrap();
        assert_eq!(counts, BTreeMap::from([(8003, 3), (8009, 4)]));

        // A router without the accounting cannot say; that is not "zero".
        assert_eq!(parse_in_flight(&json!({"loaded": []})), None);
        assert_eq!(parse_in_flight(&json!({"idle_revert": {"deferred": null}})), None);
        // The live shape when nothing is in flight.
        assert_eq!(
            parse_in_flight(&json!({"idle_revert": {"in_flight": {}, "deferred": null}})),
            Some(BTreeMap::new())
        );
    }

    #[test]
    fn only_workloads_this_plan_stops_and_that_are_busy_count() {
        let counts = BTreeMap::from([(8003, 3), (8009, 0), (11434, 1)]);
        let stopping = vec![
            ("vllm-engine".to_string(), 8003),
            ("strata-fast".to_string(), 8009),
            ("gpt-oss-120b".to_string(), 8005),
        ];
        assert_eq!(busy(&counts, &stopping), vec![("vllm-engine".to_string(), 3)]);
        assert!(busy(&counts, &[]).is_empty());
    }

    #[tokio::test]
    async fn a_router_that_is_down_cannot_say() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        assert_eq!(in_flight(port, Duration::from_millis(300)).await, None);
    }
}
