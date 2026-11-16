//! A unit's log, read from the journal (`journalctl --output=json`).
//!
//! The workload's own output carries `_SYSTEMD_INVOCATION_ID`; the
//! manager's messages about the unit (started, exited, failed) carry
//! `USER_INVOCATION_ID` (user scope) or `INVOCATION_ID` (system scope). Both
//! are kept so a reader can see run boundaries and why a run ended.
//!
//! Without a cursor `--lines=N` yields the newest N entries; with
//! `--after-cursor` it yields the *first* N after the cursor (systemd 249),
//! so a follower pages forward without gaps.

use std::time::Duration;

use chrono::DateTime;
use serde_json::Value;
use tokio::process::Command;

use llmctl_client::wire::{LogEntry, LogSource};
use llmctl_core::{UnitName, UnitScopeMarker};

use crate::engine::{EngineError, LogPage};

/// Longest message kept per entry (bytes, cut on a char boundary).
pub const MESSAGE_CAP: usize = 16 * 1024;

/// journald cursors are `key=value;…` with hex/alnum values. Anything else
/// is refused before it reaches journalctl.
pub fn valid_cursor(cursor: &str) -> bool {
    !cursor.is_empty()
        && cursor.len() <= 512
        && cursor
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '=' | ';' | '-' | '_'))
}

pub async fn read(
    unit: &UnitName,
    lines: u32,
    after: Option<&str>,
    timeout: Duration,
) -> Result<LogPage, EngineError> {
    let scope = match unit.scope {
        UnitScopeMarker::User => "--user",
        UnitScopeMarker::System => "--system",
        UnitScopeMarker::Launchd => {
            return Err(EngineError::Unsupported(
                "launchd labels have no journal".into(),
            ))
        }
    };
    // `--unit` treats glob characters as a pattern; an enrolled unit is
    // one exact name.
    if unit.name.contains(['*', '?', '[']) {
        return Err(EngineError::InvalidInput(format!(
            "unit name {:?} is a pattern, not a unit",
            unit.name
        )));
    }
    if let Some(cursor) = after {
        if !valid_cursor(cursor) {
            return Err(EngineError::InvalidInput(
                "cursor is not a journal cursor".into(),
            ));
        }
    }
    let mut cmd = Command::new("journalctl");
    cmd.arg(scope)
        .args(["--output=json", "--all", "--quiet", "--no-pager"])
        .arg(format!("--unit={}", unit.name))
        .arg(format!("--lines={lines}"));
    if let Some(cursor) = after {
        cmd.arg(format!("--after-cursor={cursor}"));
    }
    cmd.kill_on_drop(true);
    let out = match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(EngineError::Io(format!("spawning journalctl: {e}"))),
        Err(_) => {
            return Err(EngineError::Timeout(format!(
                "journalctl timed out after {}s",
                timeout.as_secs()
            )))
        }
    };
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        if stderr.contains("cursor") {
            return Err(EngineError::InvalidInput(format!(
                "journal rejected the cursor: {}",
                stderr.trim()
            )));
        }
        return Err(EngineError::Io(format!("journalctl: {}", stderr.trim())));
    }
    parse_page(&out.stdout, lines, after)
}

/// Parse journalctl's JSON lines into a page.
pub fn parse_page(stdout: &[u8], lines: u32, after: Option<&str>) -> Result<LogPage, EngineError> {
    let mut entries = Vec::new();
    let mut cursor = None;
    let mut seen = 0u32;
    for line in stdout.split(|b| *b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let value: Value = serde_json::from_slice(line)
            .map_err(|e| EngineError::Io(format!("journalctl json: {e}")))?;
        seen += 1;
        if let Some(c) = value.get("__CURSOR").and_then(Value::as_str) {
            cursor = Some(c.to_string());
        }
        if let Some(entry) = parse_entry(&value) {
            entries.push(entry);
        }
    }
    Ok(LogPage {
        entries,
        cursor: cursor.or_else(|| after.map(str::to_string)),
        more: seen >= lines,
    })
}

fn field<'a>(v: &'a Value, name: &str) -> Option<&'a str> {
    v.get(name)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// One journal record → entry. `None` when it lacks a timestamp.
pub fn parse_entry(v: &Value) -> Option<LogEntry> {
    let usec: i64 = field(v, "__REALTIME_TIMESTAMP")?.parse().ok()?;
    let timestamp = DateTime::from_timestamp_micros(usec)?;
    let own = field(v, "_SYSTEMD_INVOCATION_ID");
    let manager = field(v, "USER_INVOCATION_ID").or_else(|| field(v, "INVOCATION_ID"));
    let about_unit = field(v, "USER_UNIT").is_some() || field(v, "UNIT").is_some();
    let source = if own.is_none() && about_unit {
        LogSource::Manager
    } else {
        LogSource::Unit
    };
    let (message, truncated) = message_text(v.get("MESSAGE"));
    Some(LogEntry {
        timestamp,
        priority: field(v, "PRIORITY")
            .and_then(|p| p.parse().ok())
            .filter(|p| *p <= 7),
        source,
        invocation: own.or(manager).map(str::to_string),
        message,
        truncated,
    })
}

/// MESSAGE is a string, or an array of bytes when it is not valid UTF-8.
fn message_text(v: Option<&Value>) -> (String, bool) {
    let text = match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(bytes)) => {
            let raw: Vec<u8> = bytes
                .iter()
                .filter_map(|b| b.as_u64().and_then(|b| u8::try_from(b).ok()))
                .collect();
            String::from_utf8_lossy(&raw).into_owned()
        }
        _ => String::new(),
    };
    if text.len() <= MESSAGE_CAP {
        return (text, false);
    }
    let mut end = MESSAGE_CAP;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const INV: &str = "b91af58103f944afbd628dd612c99689";

    #[test]
    fn process_output_and_manager_messages_are_told_apart() {
        let own = parse_entry(&json!({
            "__REALTIME_TIMESTAMP": "1794846826598166",
            "MESSAGE": "line-one",
            "PRIORITY": "6",
            "_SYSTEMD_USER_UNIT": "x.service",
            "_SYSTEMD_INVOCATION_ID": INV,
        }))
        .unwrap();
        assert_eq!(own.source, LogSource::Unit);
        assert_eq!(own.invocation.as_deref(), Some(INV));
        assert_eq!(own.priority, Some(6));
        assert_eq!(own.timestamp.timestamp_micros(), 1794846826598166);

        let mgr = parse_entry(&json!({
            "__REALTIME_TIMESTAMP": "1794846826598541",
            "MESSAGE": "x.service: Main process exited, code=exited, status=2/INVALIDARGUMENT",
            "PRIORITY": "5",
            "_SYSTEMD_USER_UNIT": "init.scope",
            "USER_UNIT": "x.service",
            "USER_INVOCATION_ID": INV,
        }))
        .unwrap();
        assert_eq!(mgr.source, LogSource::Manager);
        assert_eq!(mgr.invocation.as_deref(), Some(INV));
    }

    #[test]
    fn byte_array_messages_decode_lossily_and_long_ones_are_capped() {
        let e = parse_entry(&json!({
            "__REALTIME_TIMESTAMP": "1",
            "MESSAGE": [98, 97, 100, 255, 98, 121, 116, 101],
        }))
        .unwrap();
        assert_eq!(e.message, "bad\u{fffd}byte");
        assert!(!e.truncated);

        let long = "é".repeat(MESSAGE_CAP); // 2 bytes each
        let e = parse_entry(&json!({"__REALTIME_TIMESTAMP": "1", "MESSAGE": long})).unwrap();
        assert!(e.truncated);
        assert!(e.message.len() <= MESSAGE_CAP);
        assert!(e.message.chars().all(|c| c == 'é'));
    }

    #[test]
    fn records_without_a_timestamp_are_skipped_but_still_advance_the_cursor() {
        let out = br#"{"__CURSOR":"s=a;i=1","MESSAGE":"no time"}
{"__CURSOR":"s=a;i=2","__REALTIME_TIMESTAMP":"5","MESSAGE":"kept"}

"#;
        let page = parse_page(out, 2, None).unwrap();
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.entries[0].message, "kept");
        assert_eq!(page.cursor.as_deref(), Some("s=a;i=2"));
        assert!(page.more, "two records for lines=2 is a full page");
    }

    #[test]
    fn empty_page_keeps_the_callers_cursor() {
        let page = parse_page(b"", 50, Some("s=a;i=9")).unwrap();
        assert!(page.entries.is_empty());
        assert_eq!(page.cursor.as_deref(), Some("s=a;i=9"));
        assert!(!page.more);
    }

    #[test]
    fn cursor_validation_refuses_malformed_input() {
        assert!(valid_cursor(
            "s=150509f73d7147b68d127cc577e714df;i=2dd8b9;b=134f;m=c512feed39;t=66067578691a4;x=f228"
        ));
        // (An option-looking cursor is harmless anyway: it is passed as part of
        // the single `--after-cursor=` argument.)
        for bad in ["", "--output=cat x", "s=a i=b", "s=a\n", &"a".repeat(513)] {
            assert!(!valid_cursor(bad), "{bad:?}");
        }
    }
}
