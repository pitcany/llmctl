//! Host lock location and kernel ownership inspection. The daemon owns
//! its flock throughout execution; cooperating callers release their hold
//! before delegation and reacquire it before continuing.

use std::path::{Path, PathBuf};

use crate::lease::{env_override, is_ancestor, Override};

/// `off` disables the lock, a path replaces the legacy file (any engine).
pub const OP_LOCK_ENV: &str = "LLMCTL_NEXT_OP_LOCK";
/// The override Python llmctl itself honours for the legacy file.
pub const LEGACY_OP_LOCK_ENV: &str = "LLMCTL_OP_LOCK";
/// Legacy request header, ignored by current agents.
pub const LOCK_HOLDER_HEADER: &str = "llmctl-lock-holder";

/// Where Python llmctl and llm-router keep the lock: `LLMCTL_OP_LOCK`, else
/// the XDG data dir.
pub fn legacy_path() -> Option<PathBuf> {
    if let Override::Path(p) = env_override(LEGACY_OP_LOCK_ENV) {
        return Some(p);
    }
    let data = std::env::var_os("XDG_DATA_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))?;
    Some(data.join("llmctl/operations.lock"))
}

/// Pid holding the exclusive flock on `path` per a `/proc/locks` listing
/// (`None` when unknown: not Linux, unreadable, or nobody holds it). Lines
/// look like `12: FLOCK  ADVISORY  WRITE 4242 103:02:21664095 0 EOF`;
/// blocked waiters carry `->` in the second column and are skipped.
#[cfg(unix)]
pub fn flock_owner_pid(path: &Path, proc_locks: &Path) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(path).ok()?;
    let dev = meta.dev();
    // glibc's dev_t layout.
    let major = ((dev >> 32) & 0xffff_f000) | ((dev >> 8) & 0xfff);
    let minor = ((dev >> 12) & 0xffff_ff00) | (dev & 0xff);
    let key = format!("{major:02x}:{minor:02x}:{}", meta.ino());
    std::fs::read_to_string(proc_locks).ok()?.lines().find_map(|line| {
        let parts: Vec<&str> = line.split_whitespace().collect();
        (parts.len() >= 6 && parts[1] == "FLOCK" && parts[3] == "WRITE" && parts[5] == key)
            .then(|| parts[4].parse().ok())
            .flatten()
    })
}

/// The lock owner when this process runs inside its ancestry. Diagnostic
/// only: a PID does not transfer lock ownership to an executor.
pub fn holder_ancestor() -> Option<u32> {
    let path = match env_override(OP_LOCK_ENV) {
        Override::Off => return None,
        Override::Path(p) => p,
        Override::Unset => legacy_path()?,
    };
    let owner = flock_owner_pid(&path, Path::new("/proc/locks"))?;
    is_ancestor(owner, std::process::id(), crate::lease::parent_of).then_some(owner)
}
