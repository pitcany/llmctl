//! M7a gate (docs/RUST-M7-PLAN.md): no owner- or workstation-specific
//! string in the workspace's non-test code, unless it is allowlisted here
//! with a reason. A clean host must not depend on `~/AI`, `~/local-llm`,
//! this owner's user name or the workstation's host name.
//!
//! Scope: `crates/*/src/**/*.rs`. Skipped: comment lines, and the item that
//! follows each `#[cfg(test)]` (a test module or test helper).

use std::path::{Path, PathBuf};

const PATTERNS: &[&str] = &[
    "/home/",
    "/Users/",
    "AI/",
    "local-llm",
    "workstation",
    "pitcany",
];

/// (file path suffix, text the line must contain, reason).
const ALLOWED: &[(&str, &str, &str)] = &[
    (
        "llmctl-cli/src/enroll.rs",
        "\"com.pitcany.llmctl-next\"",
        "launchd label of the agents already on the Macs; renaming it breaks them",
    ),
    (
        "llmctl-daemon/src/plistdef.rs",
        "\"com.pitcany.llmctl-next-probe-{}\"",
        "label prefix of the agent's own launchd probe jobs (same vendor id)",
    ),
    (
        "llmctl-daemon/src/records.rs",
        "\"com.pitcany.llmctl-next.record.{name}\"",
        "label prefix of the agent's record jobs (same vendor id)",
    ),
    (
        "llmctl-client/src/agent_config.rs",
        "home.join(\"AI/llm-router/config/router.yaml\")",
        "pre-M7 default, only for the start-up 'is ignored' warning; remove after the M6 deploy",
    ),
    (
        "llmctl-client/src/agent_config.rs",
        "home.join(\"AI/logs/lane-lease.json\")",
        "pre-M7 default, only for the start-up 'is ignored' warning; remove after the M6 deploy",
    ),
    (
        "llmctl-daemon/src/app.rs",
        ".unwrap_or_else(|| \"workstation\".to_string())",
        "host id of a fresh data dir without --host-id; bound once in the journal, so live agents keep theirs",
    ),
];

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let p = entry.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

/// (line number, pattern, line) for every hit outside comments and
/// `#[cfg(test)]` items.
fn hits(text: &str) -> Vec<(usize, &'static str, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let s = lines[i].trim();
        if s == "#[cfg(test)]" {
            // Skip the next item: up to its closing brace, or one `;` line.
            let (mut j, mut depth, mut seen) = (i + 1, 0i64, false);
            while j < lines.len() {
                let l = lines[j];
                depth += l.matches('{').count() as i64 - l.matches('}').count() as i64;
                seen |= l.contains('{');
                if (seen && depth <= 0) || (!seen && l.trim_end().ends_with(';')) {
                    break;
                }
                j += 1;
            }
            i = j + 1;
            continue;
        }
        if !s.starts_with("//") {
            for p in PATTERNS {
                if lines[i].contains(p) {
                    out.push((i + 1, *p, s.to_string()));
                }
            }
        }
        i += 1;
    }
    out
}

#[test]
fn no_owner_or_workstation_strings_outside_tests() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut files = Vec::new();
    for krate in std::fs::read_dir(crates).unwrap().flatten() {
        let src = krate.path().join("src");
        if src.is_dir() {
            rust_files(&src, &mut files);
        }
    }
    assert!(files.len() > 20, "scanned too few files: {}", files.len());

    let mut used = vec![false; ALLOWED.len()];
    let mut bad = Vec::new();
    for f in &files {
        let rel = f
            .strip_prefix(crates)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        for (n, pat, line) in hits(&std::fs::read_to_string(f).unwrap()) {
            match ALLOWED
                .iter()
                .position(|(file, text, _)| rel.ends_with(file) && line.contains(text))
            {
                Some(k) => used[k] = true,
                None => bad.push(format!("{rel}:{n}: [{pat}] {line}")),
            }
        }
    }
    assert!(
        bad.is_empty(),
        "owner/workstation strings in non-test code (move to config, or allowlist with a reason in {}):\n{}",
        file!(),
        bad.join("\n")
    );
    let stale: Vec<_> = ALLOWED
        .iter()
        .zip(&used)
        .filter(|(_, u)| !**u)
        .map(|((f, t, _), _)| format!("{f}: {t}"))
        .collect();
    assert!(
        stale.is_empty(),
        "allowlist entries that match nothing (remove them):\n{}",
        stale.join("\n")
    );
}

#[test]
fn the_scanner_skips_comments_and_test_items_but_not_code() {
    let text = "// ~/AI/x in a comment\n\
                let a = \"/home/u\";\n\
                #[cfg(test)]\n\
                mod tests {\n    fn f() { let _ = \"local-llm\"; }\n}\n\
                #[cfg(test)]\n\
                use x::workstation;\n\
                let b = \"pitcany\";\n";
    let got: Vec<_> = hits(text).into_iter().map(|(n, p, _)| (n, p)).collect();
    assert_eq!(got, vec![(2, "/home/"), (9, "pitcany")]);
}
