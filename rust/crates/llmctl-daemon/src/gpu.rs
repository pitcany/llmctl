//! GPU occupancy: who is actually on the cards. llmctl's conflict planner
//! only knows enrolled workloads, so before starting a unit the cards are
//! asked directly. Unknown is never "free": a probe that fails refuses the
//! operation.

use std::collections::BTreeMap;

/// One compute process on one card, as the driver reports it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GpuOccupant {
    /// Card index on this host.
    pub gpu: u32,
    pub pid: u32,
    /// Full unit identity in display form (`system/vllm-engine.service`,
    /// `user/foo.service`) when the pid maps to a service-manager cgroup;
    /// `None` when it does not map — which always counts as foreign to
    /// llmctl. Scope is part of the identity: a system unit and a user
    /// unit with the same name are different units.
    pub unit: Option<String>,
    pub memory_mib: u64,
    /// `/proc/<pid>/cmdline` with NULs as spaces (at most 512 bytes), for
    /// tolerances that match a process with no unit. `None` when unread.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cmdline: Option<String>,
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
///
/// Fail-closed: a row whose pid does not parse or whose `gpu_uuid` is not
/// in `uuid_index` (e.g. a MIG device this host does not index) is a
/// process we cannot place — `Err`, never a silently dropped row that
/// would read as "free". Memory is informational: `[N/A]` becomes 0.
pub fn parse_compute_apps(
    csv: &str,
    uuid_index: &BTreeMap<String, u32>,
) -> Result<Vec<GpuOccupant>, String> {
    let mut out = Vec::new();
    for line in csv.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split(',');
        let pid = parts
            .next()
            .and_then(|p| p.trim().parse::<u32>().ok())
            .ok_or_else(|| format!("compute-apps row with unparseable pid: {line:?}"))?;
        let uuid = parts
            .next()
            .map(|u| u.trim().to_string())
            .ok_or_else(|| format!("compute-apps row without gpu_uuid: {line:?}"))?;
        let gpu = *uuid_index.get(&uuid).ok_or_else(|| {
            format!("compute app {pid} on unknown device {uuid} (not an indexed GPU)")
        })?;
        let mem = parts
            .next()
            .and_then(|m| m.trim().parse::<u64>().ok())
            .unwrap_or(0);
        out.push(GpuOccupant {
            gpu,
            pid,
            unit: None,
            memory_mib: mem,
            cmdline: None,
        });
    }
    Ok(out)
}

/// Extract the owning unit identity (display form, scope included) from a
/// cgroup file's contents, e.g.
/// `0::/user.slice/user-1000.slice/user@1000.service/app.slice/foo.service`
/// → `user/foo.service`, or `/system.slice/vllm-engine.service` →
/// `system/vllm-engine.service`. Scope comes from the path (a
/// `user-<uid>.slice` / `user@<uid>.service` ancestor ⇒ user scope);
/// `user@1000.service` itself is the user manager, not the unit, and is
/// skipped so an unmapped process stays foreign. Docker-style `.scope`
/// cgroups carry no unit name and map to `None` (foreign).
pub fn unit_from_cgroup(cgroup: &str) -> Option<String> {
    let user_scope = cgroup.lines().any(|line| {
        line.rsplit("::")
            .next()
            .unwrap_or(line)
            .split('/')
            .any(|seg| {
                (seg.starts_with("user-") && seg.ends_with(".slice"))
                    || (seg.starts_with("user@") && seg.ends_with(".service"))
            })
    });
    let scope = if user_scope { "user" } else { "system" };
    cgroup
        .lines()
        .flat_map(|line| line.rsplit("::").next().unwrap_or(line).split('/'))
        .filter(|seg| seg.ends_with(".service") && !seg.contains('@'))
        .next_back()
        .map(|name| format!("{scope}/{name}"))
}

/// `nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits`
/// rows: `0, 5123`. A row that does not parse makes the whole answer
/// `None`: a card that could not be read is never reported as free.
pub fn parse_free_mib(csv: &str) -> Option<BTreeMap<u32, u64>> {
    let mut out = BTreeMap::new();
    for line in csv.lines().filter(|l| !l.trim().is_empty()) {
        let (idx, free) = line.split_once(',')?;
        out.insert(idx.trim().parse().ok()?, free.trim().parse().ok()?);
    }
    (!out.is_empty()).then_some(out)
}

/// Free MiB per card right now, or `None` when the cards cannot be read.
#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
pub async fn free_mib_nvidia(timeout: std::time::Duration) -> Option<BTreeMap<u32, u64>> {
    let csv = run_nvidia(
        &["--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        timeout,
    )
    .await
    .ok()?;
    parse_free_mib(&csv)
}

/// Probe the cards through nvidia-smi and map pids to their units through
/// `/proc/<pid>/cgroup` (Linux only). Any probe failure is an error:
/// callers must refuse rather than treat an unobservable card as free.
#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
pub async fn probe_nvidia(timeout: std::time::Duration) -> Result<Vec<GpuOccupant>, EngineError> {
    let uuids = run_nvidia(
        &["--query-gpu=index,uuid", "--format=csv,noheader"],
        timeout,
    )
    .await?;
    let apps = run_nvidia(
        &[
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout,
    )
    .await?;
    let uuid_index = parse_uuid_index(&uuids);
    let mut occupants = parse_compute_apps(&apps, &uuid_index).map_err(EngineError::Io)?;
    for occ in &mut occupants {
        occ.unit = cgroup_unit_of(occ.pid).await;
        occ.cmdline = cmdline_of(occ.pid).await;
    }
    Ok(occupants)
}

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
async fn run_nvidia(args: &[&str], timeout: std::time::Duration) -> Result<String, EngineError> {
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

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
async fn cmdline_of(pid: u32) -> Option<String> {
    let bytes = tokio::fs::read(format!("/proc/{pid}/cmdline")).await.ok()?;
    Some(cmdline_text(&bytes))
}

/// `/proc/<pid>/cmdline` bytes as one line: NULs become spaces, at most
/// 512 bytes (cut at a char boundary), trailing space trimmed.
pub fn cmdline_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes).replace('\0', " ");
    let mut end = text.len().min(512);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].trim_end().to_string()
}

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
async fn cgroup_unit_of(pid: u32) -> Option<String> {
    let text = tokio::fs::read_to_string(format!("/proc/{pid}/cgroup"))
        .await
        .ok()?;
    unit_from_cgroup(&text)
}

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
use tokio::process::Command;

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
use crate::engine::EngineError;

#[cfg(test)]
mod tests {
    #[test]
    fn cmdline_text_joins_args_and_caps_the_length() {
        assert_eq!(
            super::cmdline_text(b"/usr/bin/python\0uvicorn\0speaches.main:create_app\0"),
            "/usr/bin/python uvicorn speaches.main:create_app"
        );
        let long = "\u{e9}".repeat(400); // 800 bytes, 2 per char
        let cut = super::cmdline_text(long.as_bytes());
        assert!(cut.len() <= 512 && cut.chars().all(|c| c == '\u{e9}'));
    }

    #[test]
    fn free_mib_parses_or_says_unknown() {
        assert_eq!(
            super::parse_free_mib("0, 5123\n1, 31000\n"),
            Some([(0u32, 5123u64), (1, 31000)].into_iter().collect())
        );
        assert_eq!(super::parse_free_mib("0, [N/A]\n1, 31000\n"), None);
        assert_eq!(super::parse_free_mib(""), None);
    }

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
        let csv = "1234, GPU-a, 2048\n77, GPU-b, 512\n";
        let occ = parse_compute_apps(csv, &idx).unwrap();
        assert_eq!(occ.len(), 2, "{occ:?}");
        assert_eq!(occ[0].pid, 1234);
        assert_eq!(occ[0].gpu, 0);
        assert_eq!(occ[0].memory_mib, 2048);
        assert_eq!(occ[1].gpu, 1);
        // Memory is informational: [N/A] becomes 0, not an error.
        let occ = parse_compute_apps("5, GPU-a, [N/A]\n", &idx).unwrap();
        assert_eq!(occ[0].memory_mib, 0);
    }

    #[test]
    fn compute_apps_fail_closed_on_unknown_rows() {
        let mut idx = BTreeMap::new();
        idx.insert("GPU-a".into(), 0u32);
        // A process on a device that is not an indexed GPU (e.g. MIG):
        // dropping the row would read the card as free.
        let err = parse_compute_apps("999, GPU-zz, 10\n", &idx).unwrap_err();
        assert!(err.contains("unknown device"), "{err}");
        let err = parse_compute_apps("notapid, GPU-a, 5\n", &idx).unwrap_err();
        assert!(err.contains("unparseable pid"), "{err}");
    }

    #[test]
    fn cgroup_yields_the_scoped_owning_unit() {
        assert_eq!(
            unit_from_cgroup(
                "0::/user.slice/user-1000.slice/user@1000.service/app.slice/vllm-engine2.service"
            ),
            Some("user/vllm-engine2.service".into())
        );
        assert_eq!(
            unit_from_cgroup("0::/system.slice/vllm-engine.service/some/scope"),
            Some("system/vllm-engine.service".into())
        );
        // A user-scope workload must never match a same-named system unit.
        assert_ne!(
            unit_from_cgroup("0::/system.slice/foo.service"),
            unit_from_cgroup(
                "0::/user.slice/user-1000.slice/user@1000.service/app.slice/foo.service"
            )
        );
        // Only the user manager: no unit → foreign.
        assert_eq!(
            unit_from_cgroup("0::/user.slice/user-1000.slice/user@1000.service/app.slice"),
            None
        );
        assert_eq!(unit_from_cgroup(""), None);
        // Docker-style scope cgroups carry no unit name → foreign.
        assert_eq!(
            unit_from_cgroup("0::/system.slice/docker-abc123.scope"),
            None
        );
        // cgroup v1 lines (multiple controllers) also handled: last .service wins.
        assert_eq!(
            unit_from_cgroup("10:memory:/system.slice/foo.service\n0::/"),
            Some("system/foo.service".into())
        );
    }
}
