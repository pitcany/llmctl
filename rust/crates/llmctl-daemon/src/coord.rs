//! Coexistence with the other GPU controllers on the workstation (M4c,
//! design decision 6). llm-router, Python llmctl and llm-heal serialize
//! their changes with an `flock` on one host-wide file and stand down under
//! a lane lease; llmctl-next does the same for every GPU operation so the
//! controllers cannot overlap while they coexist (M4–M5).
//!
//! * [`OpLock`]: the flock on `~/.local/share/llmctl/operations.lock`,
//!   held from admission until the operation is terminal (rollback
//!   included). The kernel drops it if the daemon dies, so a crash never
//!   wedges the host. The file names the holder in the format the other
//!   controllers print (`pid`, `hostname`, `interface`, `purpose`,
//!   `acquired_at`).
//! * the lane lease (`llmctl_client::lease`): read, never written — an
//!   operation is short and already exclusive through the flock.
//!
//! Defaults apply to real engines on Linux only. The fake engine (tests,
//! dev daemons) touches neither file unless the environment names one.

use std::fs::File;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use llmctl_client::lease::{self, Override};
use llmctl_core::{OpKind, Platform, WorkloadDefinition};

pub use llmctl_client::oplock::{flock_owner_pid, OP_LOCK_ENV};
/// How long admission waits for the lock before refusing (default 5 s,
/// Python llmctl's `DEFAULT_TIMEOUT_S`).
pub const OP_LOCK_WAIT_ENV: &str = "LLMCTL_NEXT_OP_LOCK_WAIT_MS";

tokio::task_local! {
    /// The lease-holder pid the current API request claims to run inside
    /// (set by the API layer from `llmctl-under-lease`).
    pub static UNDER_LEASE: Option<u32>;
}

/// The caller's claimed lease holder, if this task serves an API request
/// that sent one. Internal callers (the recovery watcher) have none.
pub fn under_lease() -> Option<u32> {
    UNDER_LEASE.try_with(|v| *v).ok().flatten()
}

/// Operations that change what holds the cards: every operation kind on a
/// workload that has cards. That includes the two lane operations — a
/// lane swap makes llama-swap evict and load models, and a lane write
/// reloads llama-swap, which restarts its running lanes.
pub fn is_gpu_operation(kind: OpKind, def: &WorkloadDefinition) -> bool {
    !def.gpus.is_empty()
        && matches!(
            kind,
            OpKind::Run
                | OpKind::Stop
                | OpKind::Restart
                | OpKind::Upgrade
                | OpKind::Unload
                | OpKind::LaneSwap
                | OpKind::LaneWrite
                | OpKind::RouterModel
                | OpKind::RouterWrite
        )
}

fn real_engine_on_linux(platform: &Platform, def: &WorkloadDefinition) -> bool {
    *platform == Platform::Linux && def.engine != "fake"
}

/// The operation-lock file for this workload, if one applies.
pub fn op_lock_path(platform: &Platform, def: &WorkloadDefinition) -> Option<PathBuf> {
    match lease::env_override(OP_LOCK_ENV) {
        Override::Off => None,
        Override::Path(p) => Some(p),
        Override::Unset => real_engine_on_linux(platform, def)
            .then(llmctl_client::oplock::legacy_path)
            .flatten(),
    }
}

/// The lane-lease file for this workload, if one applies.
pub fn lease_path(platform: &Platform, def: &WorkloadDefinition) -> Option<PathBuf> {
    match lease::env_override(lease::LEASE_ENV) {
        Override::Off => None,
        Override::Path(p) => Some(p),
        Override::Unset => real_engine_on_linux(platform, def)
            .then(lease::default_path)
            .flatten(),
    }
}

pub fn op_lock_wait() -> Duration {
    std::env::var(OP_LOCK_WAIT_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(5))
}

/// A held operation lock. Dropping it explicitly unlocks, then closes
/// the descriptor; the holder record stays in the file and the next
/// holder overwrites it, as the other controllers do.
#[derive(Debug)]
pub struct OpLock {
    _file: File,
}

impl Drop for OpLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        // Explicit unlock also releases copies briefly inherited across fork.
        // No other process is allowed to borrow this executor's ownership.
        unsafe {
            libc::flock(self._file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

/// Why the lock could not be taken.
#[derive(Debug)]
pub enum LockRefusal {
    /// Someone else holds it; the text describes the holder.
    Busy(String),
    /// The file could not be opened or written: unknown is never free.
    Io(String),
}

#[cfg(unix)]
fn flock_exclusive_nonblocking(file: &File) -> std::io::Result<bool> {
    use std::os::fd::AsRawFd;
    // SAFETY: `flock` on a descriptor this function borrows for the call.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc == 0 {
        return Ok(true);
    }
    let err = std::io::Error::last_os_error();
    match err.raw_os_error() {
        Some(code) if code == libc::EWOULDBLOCK || code == libc::EINTR => Ok(false),
        _ => Err(err),
    }
}

/// One non-blocking attempt. On success the file names this daemon as the
/// holder with `purpose` (e.g. `run vllm-engine`).
pub fn try_lock(path: &Path, purpose: &str) -> Result<OpLock, LockRefusal> {
    use std::os::unix::fs::OpenOptionsExt;
    let io = |e: std::io::Error| LockRefusal::Io(format!("{}: {e}", path.display()));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io)?;
    }
    // No truncate at open: the record belongs to whoever holds the lock.
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)
        .map_err(io)?;
    if !flock_exclusive_nonblocking(&file).map_err(io)? {
        return Err(LockRefusal::Busy(describe_holder(path)));
    }
    let record = serde_json::json!({
        "pid": std::process::id(),
        "hostname": hostname(),
        "interface": "llmctl-next",
        "purpose": purpose,
        "acquired_at": unix_now(),
        "thread": 0,
    });
    file.set_len(0).map_err(io)?;
    file.rewind().map_err(io)?;
    file.write_all(record.to_string().as_bytes()).map_err(io)?;
    file.flush().map_err(io)?;
    Ok(OpLock { _file: file })
}

/// Take the lock, retrying for up to `wait` (a fast conflicting change is
/// allowed to finish; an operator is not left hanging).
pub async fn acquire(path: &Path, purpose: &str, wait: Duration) -> Result<OpLock, LockRefusal> {
    let deadline = tokio::time::Instant::now() + wait;
    loop {
        match try_lock(path, purpose) {
            Err(LockRefusal::Busy(_)) if tokio::time::Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            other => return other,
        }
    }
}

/// One line about who holds the lock (same shape as Python llmctl's
/// `describe_holder`). The file keeps whatever the last cooperating holder
/// wrote, so the record is checked against the process that really holds
/// the flock (`/proc/locks`): a plain `flock(1)` or a holder that died
/// after another took over must not be reported under the old name.
pub fn describe_holder(path: &Path) -> String {
    let recorded = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .filter(|v| v.as_object().is_some_and(|o| !o.is_empty()));
    let recorded_pid = recorded.as_ref().and_then(|h| h.get("pid")).and_then(|v| v.as_u64());
    if let Some(real) = flock_owner_pid(path, Path::new("/proc/locks")) {
        if recorded_pid != Some(u64::from(real)) {
            let comm = std::fs::read_to_string(format!("/proc/{real}/comm"))
                .map(|c| c.trim().to_string())
                .ok()
                .filter(|c| !c.is_empty())
                .unwrap_or_else(|| "?".into());
            let stale = recorded_pid
                .map(|p| format!("; lock file names pid {p}, which does not hold it"))
                .unwrap_or_default();
            return format!("pid {real} on {} via {comm} (not recorded{stale})", hostname());
        }
    }
    let Some(h) = recorded else {
        return "unknown holder".into();
    };
    let text = |k: &str| h.get(k).and_then(|v| v.as_str()).unwrap_or("?").to_string();
    let pid = recorded_pid
        .map(|p| p.to_string())
        .unwrap_or_else(|| "?".into());
    let age = h
        .get("acquired_at")
        .and_then(|v| v.as_f64())
        .filter(|a| *a > 0.0)
        .map(|a| format!(", {}s ago", (unix_now() - a).max(0.0) as u64))
        .unwrap_or_default();
    format!(
        "pid {pid} on {} via {} ({}{age})",
        text("hostname"),
        text("interface"),
        text("purpose")
    )
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .or_else(|_| std::fs::read_to_string("/etc/hostname"))
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "?".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lock_excludes_a_second_holder_and_names_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/operations.lock");
        let first = try_lock(&path, "run vllm-engine").expect("free lock");
        // The record is what Python llmctl, llm-router and llm-heal read.
        let rec: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(rec["pid"], std::process::id());
        assert_eq!(rec["interface"], "llmctl-next");
        assert_eq!(rec["purpose"], "run vllm-engine");
        assert!(rec["acquired_at"].as_f64().unwrap() > 1.0e9);

        // A second open file description conflicts, even in this process.
        match try_lock(&path, "stop ollama") {
            Err(LockRefusal::Busy(who)) => {
                assert!(who.contains("via llmctl-next (run vllm-engine"), "{who}");
            }
            other => panic!("expected busy, got {other:?}"),
        }
        // The refused attempt did not clobber the holder's record.
        assert!(std::fs::read_to_string(&path).unwrap().contains("run vllm-engine"));

        drop(first);
        let second = try_lock(&path, "stop ollama").expect("released on drop");
        assert!(std::fs::read_to_string(&path).unwrap().contains("stop ollama"));
        drop(second);
    }

    #[tokio::test]
    async fn acquire_waits_for_a_short_holder_and_gives_up_on_a_long_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("operations.lock");
        let held = try_lock(&path, "switch qwen27b").unwrap();
        let waited = acquire(&path, "run coder", Duration::from_millis(150)).await;
        assert!(matches!(waited, Err(LockRefusal::Busy(_))), "{waited:?}");

        let release = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            drop(held);
        });
        acquire(&path, "run coder", Duration::from_secs(5))
            .await
            .expect("taken once the holder let go");
        release.await.unwrap();
    }

    #[test]
    fn the_real_flock_holder_wins_over_a_stale_record() {
        use std::os::unix::fs::MetadataExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("operations.lock");
        std::fs::write(&path, r#"{"pid": 4242, "interface": "cli", "purpose": "run old"}"#).unwrap();
        let meta = std::fs::metadata(&path).unwrap();
        let dev = meta.dev();
        let key = format!(
            "{:02x}:{:02x}:{}",
            ((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0xfff),
            ((dev >> 12) & 0xffff_ff00) | (dev & 0xff),
            meta.ino()
        );
        let locks = dir.path().join("locks");
        std::fs::write(
            &locks,
            format!(
                "1: POSIX  ADVISORY  WRITE 11 {key} 0 EOF\n\
                 2: -> FLOCK  ADVISORY  WRITE 22 {key} 0 EOF\n\
                 3: FLOCK  ADVISORY  WRITE 33 00:00:1 0 EOF\n\
                 4: FLOCK  ADVISORY  WRITE 777 {key} 0 EOF\n"
            ),
        )
        .unwrap();
        // Not the POSIX lock, not the blocked waiter, not another file.
        assert_eq!(flock_owner_pid(&path, &locks), Some(777));
        assert_eq!(flock_owner_pid(&path, &dir.path().join("absent")), None);

        // Live (Linux): this process flocks the file without recording
        // itself, as flock(1) would; the description names it, not pid 4242.
        if cfg!(target_os = "linux") {
            let file = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
            assert!(flock_exclusive_nonblocking(&file).unwrap());
            if flock_owner_pid(&path, Path::new("/proc/locks")).is_some() {
                let who = describe_holder(&path);
                assert!(
                    who.starts_with(&format!("pid {} on ", std::process::id()))
                        && who.contains("not recorded; lock file names pid 4242"),
                    "{who}"
                );
            }
        }
    }

    #[test]
    fn holder_description_survives_a_missing_or_foreign_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("operations.lock");
        assert_eq!(describe_holder(&path), "unknown holder");
        std::fs::write(&path, "{}").unwrap();
        assert_eq!(describe_holder(&path), "unknown holder");
        std::fs::write(
            &path,
            r#"{"pid": 2004553, "hostname": "yannik-desktop", "interface": "cli", "purpose": "run qwen3.8-27b", "acquired_at": 1794700155.99, "thread": 136446215866176}"#,
        )
        .unwrap();
        let who = describe_holder(&path);
        assert!(
            who.starts_with("pid 2004553 on yannik-desktop via cli (run qwen3.8-27b, "),
            "{who}"
        );
    }
}
