//! GPU occupancy: who is actually on the cards. llmctl's conflict planner
//! only knows enrolled workloads, so before starting a unit the cards are
//! asked directly. Unknown is never "free": a probe that fails refuses the
//! operation.

use std::collections::BTreeMap;
use std::time::Duration;

use tokio::process::Command;

use crate::engine::EngineError;

/// One compute process on one card, as the driver reports it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GpuOccupant {
    /// Card index on this host.
    pub gpu: u32,
    pub pid: u32,
    /// Bare unit name (`vllm-engine.service`) when the pid maps to a service
    /// manager cgroup; `None` when it does not map — which always counts as
    /// foreign to llmctl.
    pub unit: Option<String>,
    pub memory_mib: u64,
}

/// `nvidia-smi --query-gpu=index,uuid --format=csv,noheader` rows:
/// `0, GPU-6d44f7...`.
pub fn parse_uuid_index(csv: &str) -> BTreeMap<String, u32> {
    let mut map = BTreeMap::new();
    for line in csv.lines() {
        let mut parts = line.split(',');
        let (Some(idx), Some(uuid)) = (parts.next(), parts.next()) else {
            continue;
        };
        let (Ok(idx), uuid) = (idx.trim().parse::<u32>(), uuid.trim().to_string()) else {
            continue;
        };
        if uuid.starts_with("GPU-") || uuid.starts_with("MIG-") {
            map.insert(uuid, idx);
        }
    }
    map
}

/// `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory
/// --format=csv,noheader,nounits` rows: `1234, GPU-6d44..., 2048`.
pub fn parse_compute_apps(csv: &str, uuid_index: &BTreeMap<String, u32>) -> Vec<GpuOccupant> {
    csv.lines()
        .filter_map(|line| {
            let mut parts = line.split(',');
            let pid = parts.next()?.trim().parse::<u32>().ok()?;
            let uuid = parts.next()?.trim().to_string();
            let mem = parts
                .next()?
                .trim()
                .parse::<u64>()
                .ok()
                .filter(|m| *m > 0)
                .unwrap_or(0);
            let gpu = *uuid_index.get(&uuid)?;
            Some(GpuOccupant {
                gpu,
                pid,
                unit: None,
                memory_mib: mem,
            })
        })
        .collect()
}

/// Extract the owning unit name from a cgroup file's contents, e.g.
/// `0::/user.slice/user-1000.slice/user@1000.service/app.slice/foo.service`
/// → `foo.service`, or `/system.slice/vllm-engine.service` →
/// `vllm-engine.service`. `user@1000.service` is the user manager, not the
/// unit; it is skipped so an unmapped process stays foreign.
pub fn unit_from_cgroup(cgroup: &str) -> Option<String> {
    cgroup
        .lines()
        .flat_map(|line| line.rsplit("::").next().unwrap_or(line).split('/'))
        .filter(|seg| seg.ends_with(".service") && !seg.contains('@'))
        .next_back()
        .map(str::to_owned)
}

/// Probe the cards through nvidia-smi and map pids to their units through
/// `/proc/<pid>/cgroup`. Any probe failure is an error: callers must refuse
/// rather than treat an unobservable card as free.
pub async fn probe_nvidia(timeout: Duration) -> Result<Vec<GpuOccupant>, EngineError> {
    let uuids = run_nvidia(&["--query-gpu=index,uuid", "--format=csv,noheader"], timeout).await?;
    let apps = run_nvidia(
        &[
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout,
    )
    .await?;
    let uuid_index = parse_uuid_index(&uuids);
    let mut occupants = parse_compute_apps(&apps, &uuid_index);
    for occ in &mut occupants {
        occ.unit = cgroup_unit_of(occ.pid).await;
    }
    Ok(occupants)
}

async fn run_nvidia(args: &[&str], timeout: Duration) -> Result<String, EngineError> {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args(args).kill_on_drop(true);
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) if out.status.success() => {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        }
        Ok(Ok(out)) => Err(EngineError::Io(format!(
            "nvidia-smi {}: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        ))),
        Ok(Err(e)) => Err(EngineError::Io(format!("spawning nvidia-smi: {e}"))),
        Err(_) => Err(EngineError::Timeout("nvidia-smi timed out".into())),
    }
}

async fn cgroup_unit_of(pid: u32) -> Option<String> {
    let text = tokio::fs::read_to_string(format!("/proc/{pid}/cgroup"))
        .await
        .ok()?;
    unit_from_cgroup(&text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_index_maps_uuids_to_indexes() {
        let csv = "0, GPU-6d44f7aa-9b3e\n1, GPU-91cb0c33-27f5\n";
        let m = parse_uuid_index(csv);
        assert_eq!(m.len(), 2);
        assert_eq!(m.get("GPU-91cb0c33-27f5"), Some(&1));
        assert!(parse_uuid_index("garbage\nx,y\n").is_empty());
    }

    #[test]
    fn compute_apps_rows_become_occupants() {
        let mut idx = BTreeMap::new();
        idx.insert("GPU-a".into(), 0u32);
        idx.insert("GPU-b".into(), 1u32);
        let csv = "1234, GPU-a, 2048\n77, GPU-b, 512\n999, GPU-zz, 10\nnotapid, GPU-a, 5\n";
        let occ = parse_compute_apps(csv, &idx);
        assert_eq!(occ.len(), 2, "{occ:?}");
        assert_eq!(occ[0].pid, 1234);
        assert_eq!(occ[0].gpu, 0);
        assert_eq!(occ[0].memory_mib, 2048);
        assert_eq!(occ[1].gpu, 1);
    }

    #[test]
    fn cgroup_yields_the_owning_unit_not_the_user_manager() {
        assert_eq!(
            unit_from_cgroup(
                "0::/user.slice/user-1000.slice/user@1000.service/app.slice/vllm-engine2.service"
            ),
            Some("vllm-engine2.service".into())
        );
        assert_eq!(
            unit_from_cgroup("0::/system.slice/vllm-engine.service/some/scope"),
            Some("vllm-engine.service".into())
        );
        // Only the user manager: no unit → foreign.
        assert_eq!(
            unit_from_cgroup("0::/user.slice/user-1000.slice/user@1000.service/app.slice"),
            None
        );
        assert_eq!(unit_from_cgroup(""), None);
        // cgroup v1 lines (multiple controllers) also handled: last .service wins.
        assert_eq!(
            unit_from_cgroup("10:memory:/system.slice/foo.service\n0::/"),
            Some("foo.service".into())
        );
    }
}
