//! `health`: one pass or fail per host, from the agent's status (M5
//! close-out gap R2). A workload passes when what the agent observes
//! equals its recorded intent. Exit 0 = every host passes, 1 = not.

use llmctl_client::wire::{StatusResponse, WorkloadView};
use llmctl_client::ControlClient;
use serde_json::json;

use crate::{ask_enrolled, connect, enroll, socket_path, Cli, CliError};

#[derive(Debug, PartialEq)]
pub enum Verdict {
    Ok,
    /// Intent `unmanaged`: the agent has no opinion on what it should do.
    Skipped,
    Fail(String),
}

pub fn verdict(w: &WorkloadView) -> Verdict {
    if w.intent == "unmanaged" {
        return Verdict::Skipped;
    }
    if w.observation_unknown {
        return Verdict::Fail("its state could not be observed".into());
    }
    let state = serde_json::from_str::<serde_json::Value>(&w.intent)
        .ok()
        .and_then(|v| v.get("state")?.as_str().map(str::to_owned));
    let recovery = match w.recovery_state.as_str() {
        "" | "alive" => String::new(),
        other => format!(" ({other})"),
    };
    match state.as_deref() {
        Some("running") if w.process == "Running" && w.readiness == "Ready" => Verdict::Ok,
        Some("running") => Verdict::Fail(format!(
            "intent running, observed {} {}{recovery}",
            w.process, w.readiness
        )),
        Some("stopped") if w.process == "Stopped" => Verdict::Ok,
        Some("stopped") => Verdict::Fail(format!("intent stopped, observed {}", w.process)),
        _ => Verdict::Fail(format!("intent {:?} is not understood", w.intent)),
    }
}

/// `health`: this host (or `--host`). `health --all`: this host, then
/// every enrolled host; a host that does not answer fails.
pub async fn cmd_health(cli: &Cli, all: bool, json: bool) -> Result<u8, CliError> {
    let mut hosts: Vec<(String, Result<StatusResponse, String>)> = Vec::new();
    if all {
        let local = match ControlClient::connect(socket_path(cli)) {
            Ok(c) => c.status().await.map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        let label = local.as_ref().map(|s| s.host.clone()).unwrap_or_else(|_| "local".into());
        hosts.push((label, local));
        let enrolled = enroll::load_hosts();
        let answers = ask_enrolled(&enrolled).await;
        hosts.extend(enrolled.hosts.iter().map(|h| h.name.clone()).zip(answers));
    } else {
        let status = connect(cli)?.status().await?;
        hosts.push((status.host.clone(), Ok(status)));
    }

    let mut healthy = true;
    let mut out = Vec::new();
    for (host, answer) in &hosts {
        let status = match answer {
            Ok(s) => s,
            Err(e) => {
                healthy = false;
                if !json {
                    println!("host {host}: UNHEALTHY (no answer: {e})");
                }
                out.push(json!({"host": host, "healthy": false, "error": e}));
                continue;
            }
        };
        let rows: Vec<(&str, Verdict)> =
            status.workloads.iter().map(|w| (w.name.as_str(), verdict(w))).collect();
        let fails = rows.iter().filter(|(_, v)| matches!(v, Verdict::Fail(_))).count();
        let skipped = rows.iter().filter(|(_, v)| *v == Verdict::Skipped).count();
        healthy &= fails == 0;
        if !json {
            let detail = format!("{} ok, {skipped} unmanaged", rows.len() - fails - skipped);
            match fails {
                0 => println!("host {host}: healthy ({detail})"),
                n => println!("host {host}: UNHEALTHY ({n} of {} fail; {detail})", rows.len()),
            }
            for (name, v) in &rows {
                if let Verdict::Fail(why) = v {
                    println!("  FAIL {name:<16} {why}");
                }
            }
        }
        let workloads: Vec<_> = rows
            .iter()
            .map(|(name, v)| match v {
                Verdict::Ok => json!({"name": name, "verdict": "ok"}),
                Verdict::Skipped => json!({"name": name, "verdict": "skipped"}),
                Verdict::Fail(why) => json!({"name": name, "verdict": "fail", "reason": why}),
            })
            .collect();
        out.push(json!({"host": host, "healthy": fails == 0, "workloads": workloads}));
    }
    if json {
        let doc = json!({"healthy": healthy, "hosts": out});
        println!("{}", serde_json::to_string_pretty(&doc).unwrap());
    }
    Ok(if healthy { 0 } else { 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workload(intent: &str, process: &str, readiness: &str) -> WorkloadView {
        serde_json::from_value(json!({
            "name": "w", "engine": "systemd", "model": "m", "gpus": [], "port": 1,
            "unit": "user/w.service", "intent": intent, "process": process,
            "readiness": readiness, "served_model": null, "served_model_verified": false,
            "observed_at": null, "observation_unknown": false, "recovery_state": "gave_up",
        }))
        .unwrap()
    }

    #[test]
    fn a_workload_passes_when_the_observation_equals_the_intent() {
        let running = r#"{"state":"running"}"#;
        let stopped = r#"{"state":"stopped","manual":true}"#;
        assert_eq!(verdict(&workload(running, "Running", "Ready")), Verdict::Ok);
        assert_eq!(verdict(&workload(stopped, "Stopped", "NotReady")), Verdict::Ok);
        assert_eq!(verdict(&workload("unmanaged", "Running", "Ready")), Verdict::Skipped);
        assert_eq!(
            verdict(&workload(running, "Stopped", "NotReady")),
            Verdict::Fail("intent running, observed Stopped NotReady (gave_up)".into())
        );
        // Running and not ready is not healthy.
        assert!(matches!(verdict(&workload(running, "Running", "NotReady")), Verdict::Fail(_)));
        assert_eq!(
            verdict(&workload(stopped, "Running", "Ready")),
            Verdict::Fail("intent stopped, observed Running".into())
        );
        assert!(matches!(verdict(&workload("???", "Running", "Ready")), Verdict::Fail(_)));

        let mut unknown = workload(running, "Running", "Ready");
        unknown.observation_unknown = true;
        assert!(matches!(verdict(&unknown), Verdict::Fail(_)));
    }
}
