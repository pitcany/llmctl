//! Reading a workload's log from the plist's `StandardOutPath` /
//! `StandardErrorPath` files (RUST-M3-PLAN B.5). Same wire shape as the
//! journald reader: newest `lines` entries, or the next `lines` after a
//! cursor. The cursor is `(device, inode, offset)` so rotation and
//! truncation are detected instead of silently re-read or lost.

use std::path::Path;

use chrono::{DateTime, Utc};

use llmctl_client::wire::{LogEntry, LogSource};

use crate::engine::{EngineError, LogPage};

/// Longest message kept per entry (bytes, cut on a char boundary).
pub const MESSAGE_CAP: usize = 16 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogCursor {
    pub dev: u64,
    pub inode: u64,
    pub offset: u64,
}

impl LogCursor {
    pub fn encode(self) -> String {
        format!("file:{}.{}.{}", self.dev, self.inode, self.offset)
    }
}

/// `file:<dev>.<inode>.<offset>` — anything else is refused before it
/// reaches the file system.
pub fn parse_cursor(raw: &str) -> Option<LogCursor> {
    let rest = raw.strip_prefix("file:")?;
    let mut parts = rest.split('.');
    let dev = parts.next()?.parse().ok()?;
    let inode = parts.next()?.parse().ok()?;
    let offset = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(LogCursor {
        dev,
        inode,
        offset,
    })
}

pub fn valid_cursor(raw: &str) -> bool {
    raw.len() <= 128 && parse_cursor(raw).is_some()
}

#[cfg(unix)]
fn file_ids(meta: &std::fs::Metadata) -> std::io::Result<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    Ok((meta.dev(), meta.ino()))
}

#[cfg(not(unix))]
fn file_ids(_meta: &std::fs::Metadata) -> std::io::Result<(u64, u64)> {
    Ok((0, 0))
}

/// Read one page from one log file. `invocation` (run marker) and the
/// entry timestamps come from the caller: file lines carry no reliable
/// timestamps of their own, so the engine passes the manager's run
/// identity and the file's mtime as the best-effort time.
pub async fn read(
    path: &Path,
    lines: u32,
    after: Option<&str>,
    invocation: Option<String>,
) -> Result<LogPage, EngineError> {
    let meta = tokio::fs::metadata(path).await.map_err(|e| {
        EngineError::Io(format!("stat log file {}: {e}", path.display()))
    })?;
    let (dev, inode) = file_ids(&meta).map_err(|e| {
        EngineError::Io(format!("stat log file {}: {e}", path.display()))
    })?;
    let mtime: DateTime<Utc> = meta
        .modified()
        .ok()
        .and_then(|t| DateTime::from_timestamp(t.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() as i64, 0))
        .unwrap_or_else(Utc::now);

    let cursor = match after {
        None => None,
        Some(raw) => match parse_cursor(raw) {
            Some(c) => Some(c),
            None => {
                return Err(EngineError::InvalidInput(
                    "cursor is not a file cursor (file:<dev>.<inode>.<offset>)".into(),
                ))
            }
        },
    };

    // Rotation or truncation: the file we are watching is not the file the
    // cursor points at (new inode), or it shrank (offset beyond EOF).
    // Restart from the beginning and say so via the run marker semantics
    // of the caller; never hand back a gap as if it were "no new lines".
    let start = match cursor {
        Some(c) if c.dev == dev && c.inode == inode && c.offset <= meta.len() => c.offset as usize,
        Some(_) => 0,
        None => 0,
    };

    let bytes = tokio::fs::read(path)
        .await
        .map_err(|e| EngineError::Io(format!("read {}: {e}", path.display())))?;
    let window = &bytes[start.min(bytes.len())..];

    // Complete lines only; a trailing partial line stays unread (its bytes
    // are not part of the returned offset).
    let mut end = window.len();
    if !window.is_empty() && window[window.len() - 1] != b'\n' {
        end = window
            .iter()
            .rposition(|b| *b == b'\n')
            .map(|p| p + 1)
            .unwrap_or(0);
    }
    let window = &window[..end];

    let all_lines: Vec<&[u8]> = if window.is_empty() {
        Vec::new()
    } else {
        window.split(|b| *b == b'\n').collect::<Vec<_>>()
    };
    // split() on a trailing newline yields one empty tail; drop it.
    let all_lines = if all_lines.last().is_some_and(|l| l.is_empty()) {
        &all_lines[..all_lines.len() - 1]
    } else {
        &all_lines[..]
    };

    let skip = all_lines.len().saturating_sub(lines as usize);
    let page: Vec<&[u8]> = all_lines[skip..].to_vec();
    let more = skip > 0;

    let entries = page
        .iter()
        .map(|raw| {
            let (text, truncated) = cap_utf8(raw, MESSAGE_CAP);
            LogEntry {
                timestamp: mtime,
                priority: None,
                source: LogSource::Unit,
                invocation: invocation.clone(),
                message: text,
                truncated,
            }
        })
        .collect();
    Ok(LogPage {
        entries,
        cursor: Some(LogCursor {
            dev,
            inode,
            offset: (start + end) as u64,
        }
        .encode()),
        more,
    })
}

/// Lossy-decode to UTF-8, capped at `cap` bytes on a char boundary.
fn cap_utf8(raw: &[u8], cap: usize) -> (String, bool) {
    if raw.len() <= cap {
        return (String::from_utf8_lossy(raw).into_owned(), false);
    }
    let mut cut = cap;
    while cut > 0 && (raw[cut] & 0xC0) == 0x80 {
        cut -= 1;
    }
    (
        String::from_utf8_lossy(&raw[..cut]).into_owned(),
        true,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursors_round_trip_and_reject_garbage() {
        let c = LogCursor {
            dev: 123,
            inode: 456,
            offset: 789,
        };
        let raw = c.encode();
        assert_eq!(raw, "file:123.456.789");
        assert_eq!(parse_cursor(&raw), Some(c));
        assert!(valid_cursor(&raw));
        assert!(!valid_cursor("sometthingelse"));
        assert!(!valid_cursor("file:1.2"));
        assert!(!valid_cursor("file:1.2.3.4"));
        assert!(!valid_cursor("file:a.b.c"));
    }

    #[tokio::test]
    async fn reads_tail_then_follows_by_offset() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("out.log");
        std::fs::write(&log, "one\ntwo\nthree\n").unwrap();

        let page = read(&log, 2, None, Some("run-1".into())).await.unwrap();
        assert_eq!(
            page.entries.iter().map(|e| e.message.as_str()).collect::<Vec<_>>(),
            vec!["two", "three"]
        );
        assert!(page.more);
        assert!(page.entries[0].invocation.as_deref() == Some("run-1"));

        // New content after the cursor only.
        let cursor = page.cursor.clone().unwrap();
        std::fs::write(&log, "one\ntwo\nthree\nfour\nfive\n").unwrap();
        let page = read(&log, 100, Some(&cursor), None).await.unwrap();
        assert_eq!(
            page.entries.iter().map(|e| e.message.as_str()).collect::<Vec<_>>(),
            vec!["four", "five"]
        );
        assert!(!page.more);

        // Nothing new: same cursor, empty page.
        let cursor = page.cursor.clone().unwrap();
        let page = read(&log, 100, Some(&cursor), None).await.unwrap();
        assert!(page.entries.is_empty());

        // Truncation is detected: cursor offset beyond EOF restarts at 0.
        std::fs::write(&log, "fresh\n").unwrap();
        let page = read(&log, 100, Some(&cursor), None).await.unwrap();
        assert_eq!(
            page.entries.iter().map(|e| e.message.as_str()).collect::<Vec<_>>(),
            vec!["fresh"]
        );
    }

    #[tokio::test]
    async fn partial_trailing_line_is_not_returned() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("out.log");
        std::fs::write(&log, "complete\npartial-without-newline").unwrap();
        let page = read(&log, 10, None, None).await.unwrap();
        assert_eq!(
            page.entries.iter().map(|e| e.message.as_str()).collect::<Vec<_>>(),
            vec!["complete"]
        );
        // The partial line's bytes are not consumed...
        let cursor = page.cursor.unwrap();
        // ...until the newline arrives.
        std::fs::write(&log, "complete\npartial-without-newline\n").unwrap();
        let page = read(&log, 10, Some(&cursor), None).await.unwrap();
        assert_eq!(
            page.entries.iter().map(|e| e.message.as_str()).collect::<Vec<_>>(),
            vec!["partial-without-newline"]
        );
    }

    #[test]
    fn cap_cuts_on_char_boundary() {
        let bytes = "ä".repeat(100).into_bytes(); // 200 bytes, 100 chars
        let (text, truncated) = cap_utf8(&bytes, 101);
        assert!(truncated);
        assert_eq!(text.chars().count(), 50);
        let (text, truncated) = cap_utf8(&bytes, 200);
        assert!(!truncated);
        assert_eq!(text.chars().count(), 100);
    }
}
