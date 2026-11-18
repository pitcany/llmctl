//! The lane lease (M4c, design decision 6): a small JSON file a tool writes
//! while it needs lanes left alone — longjob GPU jobs, hf2local windows,
//! modellab, mathlab. llm-router refuses switches and llm-heal refuses
//! restarts under a fresh lease; llmctl-next refuses GPU operations.
//!
//! Same reading rules as llm-router's `_read_lease`: a lease binds only
//! while it is fresh and its holder is alive, so a crashed holder never
//! blocks anyone (fails open).
//!
//! This lives in the client crate because both sides need it: the daemon
//! to refuse, the CLI to tell the daemon "I run inside the lease holder".

use std::path::{Path, PathBuf};

/// `off` disables the lease check, a path replaces the default file.
pub const LEASE_ENV: &str = "LLMCTL_NEXT_LANE_LEASE";

/// Request header: the caller runs inside this lease holder (a pid).
pub const UNDER_LEASE_HEADER: &str = "llmctl-under-lease";

/// llm-router's `lease_max_age_s` default; holders refresh well inside it.
pub const MAX_AGE_S: f64 = 300.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Lease {
    pub holder: String,
    pub pid: u32,
    pub swap: String,
    pub unit: String,
    pub lanes: Vec<String>,
    pub age_s: f64,
}

impl Lease {
    /// What the lease is on, for messages.
    pub fn what(&self) -> String {
        if !self.swap.is_empty() {
            self.swap.clone()
        } else if !self.unit.is_empty() {
            self.unit.clone()
        } else if !self.lanes.is_empty() {
            self.lanes.join(",")
        } else {
            "?".into()
        }
    }

    /// True when the lease is on this workload itself: by name (`swap`,
    /// `lanes`) or by unit (`.service` optional), case-insensitive.
    pub fn names(&self, workload: &str, unit: &str) -> bool {
        let bare = |s: &str| s.trim_end_matches(".service").to_ascii_lowercase();
        let workload = workload.to_ascii_lowercase();
        (!self.swap.is_empty() && self.swap.to_ascii_lowercase() == workload)
            || self.lanes.iter().any(|l| l.to_ascii_lowercase() == workload)
            || (!self.unit.is_empty() && !unit.is_empty() && bare(&self.unit) == bare(unit))
    }
}

/// Parse a lease file's text. `None` for anything that is not a lease
/// object with a positive pid; freshness and liveness are judged by
/// [`read_active`].
pub fn parse(text: &str, now: f64) -> Option<Lease> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let obj = v.as_object()?;
    let text_of = |k: &str| {
        obj.get(k)
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let pid = obj.get("pid").and_then(serde_json::Value::as_u64)?;
    let pid = u32::try_from(pid).ok().filter(|p| *p > 0)?;
    let updated = obj.get("updated").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
    let holder = text_of("holder");
    Some(Lease {
        holder: if holder.is_empty() { "a tool".into() } else { holder },
        pid,
        swap: text_of("swap"),
        unit: text_of("unit"),
        lanes: obj
            .get("lanes")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        age_s: now - updated,
    })
}

/// The lease at `path` if it binds right now: readable, fresh within
/// `max_age_s`, holder alive.
pub fn read_active(path: &Path, max_age_s: f64) -> Option<Lease> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    let lease = parse(&std::fs::read_to_string(path).ok()?, now)?;
    (lease.age_s <= max_age_s && pid_alive(lease.pid)).then_some(lease)
}

/// Linux: the pid has a `/proc` entry. Elsewhere there is no cheap answer;
/// a fresh lease is taken at its word.
pub fn pid_alive(pid: u32) -> bool {
    if cfg!(target_os = "linux") {
        Path::new(&format!("/proc/{pid}")).exists()
    } else {
        true
    }
}

/// Where the lease lives unless [`LEASE_ENV`] says otherwise: the file
/// longjob, hf2local and llm-router use on the workstation.
pub fn default_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join("AI/logs/lane-lease.json"))
}

/// An environment override shared by the lease and the operation lock.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Override {
    /// Unset: the caller's default applies.
    Unset,
    /// `off` (or empty): disabled.
    Off,
    Path(PathBuf),
}

pub fn env_override(var: &str) -> Override {
    match std::env::var_os(var) {
        None => Override::Unset,
        Some(v) if v.is_empty() || v == "off" => Override::Off,
        Some(v) => Override::Path(PathBuf::from(v)),
    }
}

/// The pid of the active lease holder when this process runs inside it
/// (a longjob job driving `llmctl-next`): the CLI sends it so the daemon
/// lets the holder's own commands through.
pub fn holder_ancestor() -> Option<u32> {
    let path = match env_override(LEASE_ENV) {
        Override::Off => return None,
        Override::Path(p) => p,
        Override::Unset => default_path()?,
    };
    let lease = read_active(&path, MAX_AGE_S)?;
    is_ancestor(lease.pid, std::process::id(), parent_of).then_some(lease.pid)
}

/// True when `ancestor` is `pid` or one of its parents (bounded walk).
pub fn is_ancestor(ancestor: u32, pid: u32, parent: impl Fn(u32) -> Option<u32>) -> bool {
    let mut cur = pid;
    for _ in 0..64 {
        if cur == ancestor {
            return true;
        }
        match parent(cur) {
            Some(p) if p > 0 && p != cur => cur = p,
            _ => return false,
        }
    }
    false
}

/// Parent pid from `/proc/<pid>/stat` (field 4; the command name before
/// it may contain spaces and parentheses, so split after the last `)`).
pub fn parent_of(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_three_lease_shapes_and_rejects_non_leases() {
        let l = parse(
            r#"{"holder":"longjob","pid":4242,"lanes":["bench-x","p4"],"updated":100.0}"#,
            130.0,
        )
        .unwrap();
        assert_eq!((l.holder.as_str(), l.pid, l.age_s), ("longjob", 4242, 30.0));
        assert_eq!(l.what(), "bench-x,p4");
        assert!(l.names("P4", "") && !l.names("coder", "coder.service"));

        let l = parse(r#"{"pid":7,"swap":"qwen27b","updated":1}"#, 2.0).unwrap();
        assert_eq!((l.holder.as_str(), l.what().as_str()), ("a tool", "qwen27b"));
        assert!(l.names("qwen27b", "x.service"));

        let l = parse(r#"{"pid":7,"unit":"vllm-engine.service","updated":1}"#, 2.0).unwrap();
        assert!(l.names("anything", "vllm-engine") && l.names("x", "VLLM-engine.service"));
        assert!(!l.names("vllm-engine", "ollama.service"));

        for bad in ["", "[]", "{}", r#"{"pid":0}"#, r#"{"pid":"x"}"#, "not json"] {
            assert_eq!(parse(bad, 1.0), None, "{bad}");
        }
    }

    #[test]
    fn a_lease_binds_only_while_fresh_and_its_holder_lives() {
        let dir = std::env::temp_dir().join(format!("llmctl-lease-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lane-lease.json");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let write = |pid: u32, updated: f64| {
            std::fs::write(&path, format!(r#"{{"pid":{pid},"lanes":["x"],"updated":{updated}}}"#))
                .unwrap()
        };
        assert_eq!(read_active(&path, MAX_AGE_S), None, "missing file");
        write(std::process::id(), now);
        assert!(read_active(&path, MAX_AGE_S).is_some());
        write(std::process::id(), now - 1000.0);
        assert_eq!(read_active(&path, MAX_AGE_S), None, "stale");
        if cfg!(target_os = "linux") {
            write(2_147_483_646, now);
            assert_eq!(read_active(&path, MAX_AGE_S), None, "dead holder");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ancestry_walks_parents_and_stops() {
        let parent = |p: u32| match p {
            30 => Some(20),
            20 => Some(10),
            10 => Some(1),
            _ => None,
        };
        assert!(is_ancestor(10, 30, parent));
        assert!(is_ancestor(30, 30, parent));
        assert!(!is_ancestor(99, 30, parent));
        // A parent loop ends instead of spinning.
        assert!(!is_ancestor(5, 7, |_| Some(8)));
        if cfg!(target_os = "linux") {
            let me = std::process::id();
            let ppid = parent_of(me).expect("own stat readable");
            assert!(is_ancestor(ppid, me, parent_of));
        }
    }
}
