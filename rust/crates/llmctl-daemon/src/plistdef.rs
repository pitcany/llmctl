//! launchd plist definitions: the pure plist <-> [`LaunchDefinition`]
//! mapping, a deterministic XML-plist writer, and the `launchctl print`
//! text parser (RUST-M3-PLAN B/C).
//!
//! The *canonical* form of a launchd job is the plist dictionary (decision
//! 5) -- launchd has no drop-ins or merging. Equivalence between a source
//! plist and a render is equality of those dictionaries (via `plutil
//! -convert json`), never byte equality of the XML.
//!
//! The mapping is pure and compiled on every platform so its round-trip
//! tests run everywhere; the subprocess layer (`plutil`, `launchctl`) is
//! macOS-only.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use llmctl_core::{
    looks_secret, ExecCommand, FieldDiff, LaunchDefinition, LaunchdExtras, KeepAliveValue,
};

/// Top-level plist keys this mapping carries. Anything else in a source
/// plist is reported as uncovered (an own/render would drop it silently).
pub const CARRIED_KEYS: &[&str] = &[
    "Label",
    "Program",
    "ProgramArguments",
    "EnvironmentVariables",
    "WorkingDirectory",
    "KeepAlive",
    "RunAtLoad",
    "StandardOutPath",
    "StandardErrorPath",
    "ThrottleInterval",
    "ProcessType",
    "Disabled",
];

/// Where a label's plist lives by convention.
pub fn conventional_plist_path(label: &str) -> PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_default();
    Path::new(&home)
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{label}.plist"))
}

/// The subset of `launchctl print gui/<uid>/<label>` the adapter reads.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PrintState {
    /// The job exists in the target domain.
    pub loaded: bool,
    /// `state =` (`running`, `waiting`, ...); empty when unknown.
    pub state: String,
    pub pid: Option<u32>,
    /// `runs =` -- launchd's run counter (part of effect identity).
    pub runs: Option<u64>,
    /// `program =` -- the executable launchd would run.
    pub program: Option<String>,
    /// `arguments =` when print lists them (scratch-job proof).
    pub arguments: Option<Vec<String>>,
    /// `last exit code =` / `last exit status =` / `last terminating signal =`.
    pub last_exit_status: Option<String>,
    /// The disabled override applies to this label.
    pub disabled: bool,
}

/// Parse `launchctl print` text output. Only TOP-LEVEL job keys count:
/// the print also lists per-service sections whose `state = active` lines
/// would otherwise overwrite the job's own `state = running` (found on the
/// first real run on mbp2 — a healthy job read as stopped). The
/// `arguments` block appears either brace- or paren-delimited, entries
/// unquoted or quoted with a trailing comma.
pub fn parse_print(text: &str) -> PrintState {
    let mut out = PrintState::default();
    for line in text.lines() {
        // Exactly one tab of indentation = a job-level property.
        let Some(rest) = line.strip_prefix('\t') else {
            continue;
        };
        if rest.starts_with('\t') || rest.starts_with(' ') {
            continue; // nested section (services, endpoints, ...)
        }
        let Some((key, value)) = rest.split_once(" = ") else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "state" => {
                out.loaded = true;
                out.state = value.to_string();
            }
            "pid" => out.pid = value.parse().ok(),
            "runs" => out.runs = value.parse().ok(),
            "program" => out.program = Some(value.to_string()),
            "last exit code" | "last exit status" | "last terminating signal" => {
                out.last_exit_status = Some(format!("{key} = {value}"))
            }
            "disabled" => out.disabled = value == "true",
            _ => {}
        }
    }
    out.arguments = parse_arguments_block(text);
    out
}

/// Collect the `arguments` block (`arguments = {` … `}` or `( … )`), one
/// word per line, unquoted (modern launchd) or quoted-with-comma (older).
/// The block sits at job level (one tab); two tabs kept for tolerance.
fn parse_arguments_block(text: &str) -> Option<Vec<String>> {
    for indent in ["\t", "\t\t"] {
        for (brace, paren) in [("{", "}"), ("(", ")")] {
            let open = format!("{indent}arguments = {brace}\n");
            let close = format!("{indent}{paren}\n");
            if let Some(start) = text.find(&open) {
                let rest = &text[start + open.len()..];
                if let Some(end) = rest.find(&close) {
                    return Some(
                        rest[..end]
                            .lines()
                            .filter_map(|l| {
                                let t = l.trim().trim_end_matches(',');
                                let t = t
                                    .strip_prefix('"')
                                    .and_then(|s| s.strip_suffix('"'))
                                    .unwrap_or(t);
                                (!t.is_empty()).then(|| t.to_owned())
                            })
                            .collect(),
                    );
                }
            }
        }
    }
    None
}

/// Import a plist JSON dictionary (plutil-converted) into a definition.
/// Returns the definition and the inline environment names that look like
/// credentials (their values are never kept).
pub fn definition_from_plist(
    plist: &Value,
    plist_path: &str,
) -> Result<(LaunchDefinition, Vec<String>), String> {
    let obj = plist
        .as_object()
        .ok_or_else(|| "plist root is not a dictionary".to_string())?;
    let label = obj
        .get("Label")
        .and_then(Value::as_str)
        .ok_or_else(|| "plist has no Label".to_string())?
        .to_string();

    let args = obj.get("ProgramArguments").and_then(Value::as_array);
    let program = obj.get("Program").and_then(Value::as_str);
    let exec_start = match (args, program) {
        (Some(argv), _) if !argv.is_empty() => {
            let mut words = Vec::with_capacity(argv.len());
            for v in argv {
                let s = v
                    .as_str()
                    .ok_or_else(|| "ProgramArguments entry is not a string".to_string())?;
                words.push(s.to_string());
            }
            let path = words[0].clone();
            vec![ExecCommand {
                path,
                argv: words,
                ignore_errors: false,
            }]
        }
        (Some(_), _) => return Err("ProgramArguments is empty".into()),
        (None, Some(p)) => vec![ExecCommand {
            path: p.to_string(),
            argv: vec![p.to_string()],
            ignore_errors: false,
        }],
        (None, None) => {
            return Err("plist has neither Program nor ProgramArguments".into());
        }
    };

    let mut environment = Vec::new();
    let mut secret_env_names = Vec::new();
    if let Some(env) = obj.get("EnvironmentVariables") {
        let env = env
            .as_object()
            .ok_or_else(|| "EnvironmentVariables is not a dictionary".to_string())?;
        for (k, v) in env {
            let val = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if looks_secret(k) {
                secret_env_names.push(k.clone());
            }
            environment.push((k.clone(), val));
        }
        environment.sort();
    }

    let keep_alive = match obj.get("KeepAlive") {
        None => KeepAliveValue::Bool(false),
        Some(Value::Bool(b)) => KeepAliveValue::Bool(*b),
        Some(other) => KeepAliveValue::Other(other.clone()),
    };
    let opt_str = |k: &str| obj.get(k).and_then(Value::as_str).map(str::to_owned);

    let definition = LaunchDefinition {
        service_type: "launchd".into(),
        exec_start_pre: Vec::new(),
        exec_start,
        exec_start_post: Vec::new(),
        exec_stop: Vec::new(),
        working_directory: opt_str("WorkingDirectory"),
        environment,
        environment_files: Vec::new(),
        restart: if keep_alive.is_always() {
            "always".into()
        } else {
            "no".into()
        },
        timeout_stop_usec: 0,
        kill_mode: "process".into(),
        restart_usec: None,
        timeout_start_usec: None,
        start_limit_burst: None,
        start_limit_interval_usec: None,
        syslog_identifier: None,
        after: None,
        wants: None,
        requires: None,
        documentation: None,
        exec_stop_post: Vec::new(),
        kill_signal: None,
        standard_output: None,
        standard_error: None,
        state_directory: None,
        cache_directory: None,
        no_new_privileges: None,
        private_tmp: None,
        protect_home: None,
        protect_system: None,
        read_write_paths: None,
        launchd: Some(LaunchdExtras {
            label,
            plist_path: plist_path.to_string(),
            program: program.map(str::to_owned),
            run_at_load: obj
                .get("RunAtLoad")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            keep_alive,
            stdout_path: opt_str("StandardOutPath"),
            stderr_path: opt_str("StandardErrorPath"),
            throttle_interval_secs: obj
                .get("ThrottleInterval")
                .and_then(Value::as_u64)
                .filter(|s| *s > 0),
            process_type: opt_str("ProcessType"),
            disabled: obj.get("Disabled").and_then(Value::as_bool),
        }),
    };
    Ok((definition, secret_env_names))
}

/// Render a definition back to the canonical plist JSON dictionary.
/// Inverse of [`definition_from_plist`] for every carried key.
pub fn plist_from_definition(def: &LaunchDefinition) -> Result<Value, String> {
    let extras = def
        .launchd
        .as_ref()
        .ok_or_else(|| "definition carries no launchd facts".to_string())?;
    let mut obj = Map::new();
    obj.insert("Label".into(), Value::String(extras.label.clone()));

    if let [one] = def.exec_start.as_slice() {
        // A single-word exec that mirrors `Program` renders as `Program`,
        // so the dictionary round-trips to exactly the source form.
        let from_program =
            extras.program.as_deref() == Some(one.path.as_str()) && one.argv.len() == 1;
        if from_program {
            obj.insert("Program".into(), Value::String(one.path.clone()));
        } else {
            obj.insert(
                "ProgramArguments".into(),
                Value::Array(one.argv.iter().cloned().map(Value::String).collect()),
            );
        }
    } else if let Some(p) = &extras.program {
        obj.insert("Program".into(), Value::String(p.clone()));
    } else {
        return Err("definition has no exec_start to render".into());
    }

    if !def.environment.is_empty() {
        let mut env = Map::new();
        for (k, v) in &def.environment {
            env.insert(k.clone(), Value::String(v.clone()));
        }
        obj.insert("EnvironmentVariables".into(), Value::Object(env));
    }
    if let Some(wd) = &def.working_directory {
        obj.insert("WorkingDirectory".into(), Value::String(wd.clone()));
    }
    let keep_alive = match &extras.keep_alive {
        KeepAliveValue::Bool(false) => None,
        KeepAliveValue::Bool(true) => Some(Value::Bool(true)),
        KeepAliveValue::Other(v) => Some(v.clone()),
    };
    if let Some(v) = keep_alive {
        obj.insert("KeepAlive".into(), v);
    }
    if extras.run_at_load {
        obj.insert("RunAtLoad".into(), Value::Bool(true));
    }
    if let Some(p) = &extras.stdout_path {
        obj.insert("StandardOutPath".into(), Value::String(p.clone()));
    }
    if let Some(p) = &extras.stderr_path {
        obj.insert("StandardErrorPath".into(), Value::String(p.clone()));
    }
    if let Some(secs) = extras.throttle_interval_secs {
        obj.insert("ThrottleInterval".into(), Value::from(secs));
    }
    if let Some(t) = &extras.process_type {
        obj.insert("ProcessType".into(), Value::String(t.clone()));
    }
    if let Some(d) = extras.disabled {
        obj.insert("Disabled".into(), Value::Bool(d));
    }
    Ok(Value::Object(obj))
}

/// Top-level plist keys a render of `def` would silently drop.
pub fn uncovered_keys_in(plist: &Value) -> Vec<String> {
    plist
        .as_object()
        .map(|obj| {
            obj.keys()
                .filter(|k| !CARRIED_KEYS.contains(&k.as_str()))
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

// ---- deterministic XML plist writer -----------------------------------------

/// Serialize a JSON dict as an XML plist with keys in sorted order, so a
/// definition always renders to identical bytes. Only the plist value
/// subset is accepted (string, bool, integer, array, dict); anything else
/// is an error, never a guess.
pub fn render_xml_plist(dict: &Value) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n");
    out.push_str("<plist version=\"1.0\">\n");
    write_xml_value(&mut out, dict, 0)?;
    out.push_str("</plist>\n");
    Ok(out)
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push('\t');
    }
}

fn write_xml_value(out: &mut String, v: &Value, depth: usize) -> Result<(), String> {
    match v {
        Value::String(s) => {
            indent(out, depth);
            out.push_str("<string>");
            push_xml_escaped(out, s);
            out.push_str("</string>\n");
        }
        Value::Bool(b) => {
            indent(out, depth);
            out.push_str(if *b { "<true/>\n" } else { "<false/>\n" });
        }
        Value::Number(n) => {
            let Some(i) = n.as_i64() else {
                return Err(format!("plist numbers must be integers: {n}"));
            };
            indent(out, depth);
            out.push_str(&format!("<integer>{i}</integer>\n"));
        }
        Value::Array(items) => {
            indent(out, depth);
            out.push_str("<array>\n");
            for item in items {
                write_xml_value(out, item, depth + 1)?;
            }
            indent(out, depth);
            out.push_str("</array>\n");
        }
        Value::Object(obj) => {
            indent(out, depth);
            out.push_str("<dict>\n");
            let mut keys: Vec<&String> = obj.keys().collect();
            keys.sort();
            for key in keys {
                indent(out, depth + 1);
                out.push_str("<key>");
                push_xml_escaped(out, key);
                out.push_str("</key>\n");
                write_xml_value(out, &obj[key], depth + 1)?;
            }
            indent(out, depth);
            out.push_str("</dict>\n");
        }
        Value::Null => return Err("null is not a plist value".into()),
    }
    Ok(())
}

fn push_xml_escaped(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            // Escaped as character references so a log line or argument can
            // never break out of the string element.
            c if (c as u32) < 0x20 => out.push_str(&format!("&#x{:x};", c as u32)),
            c => out.push(c),
        }
    }
}

/// Render the definition as an XML plist file body. `fault_corrupt_render`
/// injects a subtly wrong extra key so the prove step must catch it.
pub fn render_plist(def: &LaunchDefinition, fault_corrupt_render: bool) -> Result<String, String> {
    let mut dict = plist_from_definition(def)?;
    if fault_corrupt_render {
        let obj = dict
            .as_object_mut()
            .ok_or_else(|| "render: plist root is not a dictionary".to_string())?;
        let env = obj
            .entry("EnvironmentVariables")
            .or_insert_with(|| serde_json::json!({}));
        if let Some(e) = env.as_object_mut() {
            e.insert("LLMCTL_FAULT_RENDER".into(), serde_json::json!("1"));
        }
    }
    render_xml_plist(&dict)
}

/// Field diffs between two definitions, ignoring the fields that differ by
/// construction between a file import and a scratch-label round trip
/// (`launchd.label`, `launchd.plist_path`).
pub fn semantic_diff(a: &LaunchDefinition, b: &LaunchDefinition) -> Vec<FieldDiff> {
    let strip = |mut d: LaunchDefinition| {
        if let Some(extras) = &mut d.launchd {
            extras.label = String::new();
            extras.plist_path = String::new();
        }
        d
    };
    strip(a.clone()).diff(&strip(b.clone()))
}

// ---- macOS subprocess layer --------------------------------------------------

#[cfg(any(target_os = "macos", llmctl_macos))]
mod runtime {
    use super::*;
    use crate::supervisor::ImportedJob;
    use llmctl_core::UnitName;
    use std::time::Duration;

    use sha2::Digest as _;
    use tokio::process::Command;

    use crate::engine::EngineError;

    /// `plutil -convert json -o - -- <path>`: the canonical dictionary of a
    /// plist file, whatever its original serialization.
    pub async fn plist_json(path: &Path, timeout: Duration) -> Result<Value, EngineError> {
        let out = run(&["-convert", "json", "-o", "-", "--"], Some(path), timeout).await?;
        serde_json::from_str(&out).map_err(|e| {
            EngineError::Io(format!("parsing plutil json for {}: {e}", path.display()))
        })
    }

    /// `plutil -convert xml1` canonical bytes (equivalence checks).
    pub async fn plist_canonical_xml(
        path: &Path,
        timeout: Duration,
    ) -> Result<String, EngineError> {
        run(&["-convert", "xml1", "-o", "-", "--"], Some(path), timeout).await
    }

    async fn run(
        args: &[&str],
        path: Option<&Path>,
        timeout: Duration,
    ) -> Result<String, EngineError> {
        let mut cmd = Command::new("plutil");
        cmd.args(args).kill_on_drop(true);
        if let Some(p) = path {
            cmd.arg(p);
        }
        match tokio::time::timeout(timeout, cmd.output()).await {
            Ok(Ok(out)) if out.status.success() => {
                Ok(String::from_utf8_lossy(&out.stdout).into_owned())
            }
            Ok(Ok(out)) => Err(EngineError::Io(format!(
                "plutil {}: {}",
                args.first().unwrap_or(&""),
                String::from_utf8_lossy(&out.stderr).trim()
            ))),
            Ok(Err(e)) => Err(EngineError::Io(format!("spawning plutil: {e}"))),
            Err(_) => Err(EngineError::Timeout("plutil timed out".into())),
        }
    }

    /// Import a label's job: the plist file is the canonical definition
    /// (decision 5); the loaded job is checked separately for drift.
    pub async fn import_job(unit: &UnitName, timeout: Duration) -> Result<ImportedJob, EngineError> {
        let path = conventional_plist_path(&unit.name);
        if !path.exists() {
            return Ok(ImportedJob {
                load_state: "not-found".into(),
                definition: None,
                provenance: Vec::new(),
                secret_env_names: Vec::new(),
                fragment_path: String::new(),
                drop_in_paths: Vec::new(),
            });
        }
        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| EngineError::Io(format!("reading {}: {e}", path.display())))?;
        let digest = format!("sha256:{:x}", sha2::Sha256::digest(&bytes));
        let json = plist_json(&path, timeout).await?;
        let (definition, secret_env_names) = definition_from_plist(
            &json,
            &path.display().to_string(),
        )
        .map_err(EngineError::InvalidInput)?;

        // Is the job loaded in the user's gui domain?
        let loaded = crate::launchd::print_job(&unit.name, timeout)
            .await
            .map(|p| p.loaded)
            .unwrap_or(false);
        Ok(ImportedJob {
            load_state: if loaded { "loaded" } else { "not-loaded" }.into(),
            definition: Some(definition),
            provenance: vec![llmctl_core::FileProvenance {
                path: path.display().to_string(),
                content_digest: digest,
            }],
            secret_env_names,
            fragment_path: path.display().to_string(),
            drop_in_paths: Vec::new(),
        })
    }

    /// Prove a rendered plist loads in the real manager under a scratch
    /// label (bootstrap, read back, boot out). Never touches the live
    /// label. Diffs compare everything except the label/plist path, which
    /// differ by construction.
    pub async fn roundtrip(
        def: &LaunchDefinition,
        rendered: &str,
        timeout: Duration,
    ) -> Result<Vec<FieldDiff>, EngineError> {
        let extras = def
            .launchd
            .as_ref()
            .ok_or_else(|| EngineError::InvalidInput("definition has no launchd facts".into()))?;
        let scratch_label = format!(
            "com.pitcany.llmctl-next-probe-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let dir = std::env::temp_dir().join("llmctl-next-probe");
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| EngineError::Io(format!("creating probe dir: {e}")))?;
        let file = dir.join(format!("{scratch_label}.plist"));
        let rendered = rendered.replace(
            &format!("<string>{}</string>", extras.label),
            &format!("<string>{scratch_label}</string>"),
        );
        tokio::fs::write(&file, &rendered)
            .await
            .map_err(|e| EngineError::Io(format!("writing probe plist: {e}")))?;

        // Equivalence part 1 (pure): the file as written re-imports to the
        // same definition (dictionary equality, decision 5).
        let json = plist_json(&file, timeout).await?;
        let (reimported, _) = definition_from_plist(&json, &file.display().to_string())
            .map_err(EngineError::InvalidInput)?;
        let mut diffs = semantic_diff(def, &reimported);

        // Equivalence part 2: launchd accepts it (bootstrap under the
        // scratch label, read the loaded job back, boot out).
        if let Err(e) = crate::launchd::bootstrap_scratch(&scratch_label, &file, timeout).await {
            let _ = tokio::fs::remove_file(&file).await;
            return Err(e);
        }
        // Read the loaded scratch job back and compare semantics.
        let print = crate::launchd::print_job(&scratch_label, timeout).await?;
        let expected_program = def
            .launchd
            .as_ref()
            .and_then(|x| x.program.clone())
            .or_else(|| def.exec_start.first().map(|c| c.path.clone()));
        let expected_argv: Option<Vec<String>> = def.exec_start.first().map(|c| c.argv.clone());
        if let Some(expected) = &expected_program {
            let actual = print.program.clone().unwrap_or_default();
            if !expected.is_empty() && &actual != expected {
                diffs.push(FieldDiff {
                    field: "launchd.program".into(),
                    expected: serde_json::json!(expected),
                    actual: serde_json::json!(actual),
                });
            }
        }
        if let (Some(expected), Some(actual)) = (&expected_argv, &print.arguments) {
            if expected != actual && !actual.is_empty() {
                diffs.push(FieldDiff {
                    field: "exec_start".into(),
                    expected: serde_json::json!(expected),
                    actual: serde_json::json!(actual),
                });
            }
        }
        let _ = crate::launchd::bootout_job(&scratch_label, timeout).await;
        let _ = tokio::fs::remove_file(&file).await;
        Ok(diffs)
    }

    /// Uncovered top-level keys in a plist *file*, by canonical dictionary.
    /// Synchronous (short plutil call) because the supervisor port is sync
    /// here; the caller runs it once per `own`.
    pub fn uncovered_keys(source_path: &Path, def: &LaunchDefinition) -> Vec<String> {
        // Only keys the definition actually carries count as covered; a
        // definition without launchd extras covers nothing.
        if def.launchd.is_none() {
            return Vec::new();
        }
        let out = std::process::Command::new("plutil")
            .args(["-convert", "json", "-o", "-", "--"])
            .arg(source_path)
            .output();
        match out {
            Ok(out) if out.status.success() => {
                let text = String::from_utf8_lossy(&out.stdout).into_owned();
                serde_json::from_str::<Value>(&text)
                    .map(|v| uncovered_keys_in(&v))
                    .unwrap_or_default()
            }
            _ => Vec::new(),
        }
    }

    /// Canonical-XML bytes of a plist *file* (adopt equivalence check),
    /// synchronous for the same reason as `uncovered_keys`.
    pub fn canonical_xml_sync(source_path: &Path) -> Result<String, String> {
        let out = std::process::Command::new("plutil")
            .args(["-convert", "xml1", "-o", "-", "--"])
            .arg(source_path)
            .output()
            .map_err(|e| format!("spawning plutil: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "plutil: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

#[cfg(any(target_os = "macos", llmctl_macos))]
pub use runtime::*;

/// The log to read for a path a record names: the file itself, or the
/// newest regular file below a directory (two levels: LM Studio keeps
/// `server-logs/<month>/<day>.<n>.log`).
pub fn newest_log(path: &std::path::Path) -> Option<std::path::PathBuf> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.is_file() {
        return Some(path.to_path_buf());
    }
    fn walk(dir: &std::path::Path, depth: u32, best: &mut Option<(std::time::SystemTime, std::path::PathBuf)>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for e in entries.flatten() {
            let Ok(meta) = e.metadata() else { continue };
            if meta.is_dir() {
                if depth > 0 {
                    walk(&e.path(), depth - 1, best);
                }
            } else if meta.is_file() {
                let Ok(modified) = meta.modified() else { continue };
                if best.as_ref().is_none_or(|(t, _)| modified > *t) {
                    *best = Some((modified, e.path()));
                }
            }
        }
    }
    let mut best = None;
    walk(path, 2, &mut best);
    best.map(|(_, p)| p)
}

/// Labels launchd gives to apps it starts for LaunchServices (the Dock,
/// a login item, `open -a`).
pub const APP_LABEL_PREFIX: &str = "application.";

/// The running instance of an app in `launchctl list` output, given the
/// stable part of its label (`application.ai.elementlabs.lmstudio`). The
/// full label ends in two numbers that change with every launch. A line
/// with a pid wins over one without (an instance that already exited).
pub fn app_instance_label(list_output: &str, stable: &str) -> Option<String> {
    let mut exited: Option<String> = None;
    for line in list_output.lines() {
        let mut cols = line.split_whitespace();
        let (Some(pid), Some(_status), Some(label)) = (cols.next(), cols.next(), cols.next())
        else {
            continue;
        };
        let Some(rest) = label.strip_prefix(stable).and_then(|r| r.strip_prefix('.')) else {
            continue;
        };
        let numbered = !rest.is_empty()
            && rest
                .split('.')
                .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()));
        if !numbered {
            // `application.com.foo` must not match `application.com.foo.helper.1.2`.
            continue;
        }
        if pid.chars().all(|c| c.is_ascii_digit()) {
            return Some(label.to_string());
        }
        exited.get_or_insert_with(|| label.to_string());
    }
    exited
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_swap_plist() -> Value {
        serde_json::json!({
            "Label": "com.pitcany.llama-swap",
            "ProgramArguments": [
                "/Users/yannik/local-llm/bin/llama-swap",
                "--config",
                "/Users/yannik/local-llm/llama-swap/config.yaml"
            ],
            "EnvironmentVariables": {
                "HOME": "/Users/yannik",
                "PATH": "/usr/local/bin:/usr/bin:/bin"
            },
            "KeepAlive": true,
            "RunAtLoad": true,
            "StandardOutPath": "/Users/yannik/Library/Logs/llama-swap/llama-swap.out.log",
            "StandardErrorPath": "/Users/yannik/Library/Logs/llama-swap/llama-swap.err.log",
            "ThrottleInterval": 10,
            "ProcessType": "Interactive"
        })
    }

    #[test]
    fn import_maps_every_carried_key() {
        let (def, secrets) = definition_from_plist(&llama_swap_plist(), "/x/com.pitcany.llama-swap.plist")
            .expect("import");
        assert!(secrets.is_empty());
        let extras = def.launchd.as_ref().expect("extras");
        assert_eq!(extras.label, "com.pitcany.llama-swap");
        assert_eq!(def.exec_start.len(), 1);
        assert_eq!(
            def.exec_start[0].path,
            "/Users/yannik/local-llm/bin/llama-swap"
        );
        assert_eq!(def.exec_start[0].argv.len(), 3);
        assert_eq!(
            def.working_directory.as_deref(),
            None
        );
        assert_eq!(def.environment.len(), 2);
        assert_eq!(def.restart, "always");
        assert!(extras.run_at_load);
        assert!(extras.keep_alive.is_always());
        assert_eq!(extras.throttle_interval_secs, Some(10));
        assert_eq!(extras.process_type.as_deref(), Some("Interactive"));
        assert_eq!(
            extras.stdout_path.as_deref(),
            Some("/Users/yannik/Library/Logs/llama-swap/llama-swap.out.log")
        );
    }

    #[test]
    fn import_refuses_secret_env_and_missing_parts() {
        let mut bad = llama_swap_plist();
        bad["EnvironmentVariables"]["HF_TOKEN"] = Value::String("x".into());
        let (_, secrets) = definition_from_plist(&bad, "/x").expect("import");
        assert_eq!(secrets, vec!["HF_TOKEN"]);

        let mut no_args = llama_swap_plist();
        no_args.as_object_mut().unwrap().remove("ProgramArguments");
        assert!(definition_from_plist(&no_args, "/x").is_err());

        let mut no_label = llama_swap_plist();
        no_label.as_object_mut().unwrap().remove("Label");
        assert!(definition_from_plist(&no_label, "/x").is_err());
    }

    #[test]
    fn render_round_trips_to_the_same_dictionary() {
        let source = llama_swap_plist();
        let (def, _) = definition_from_plist(&source, "/x/y.plist").expect("import");
        let rendered = plist_from_definition(&def).expect("render");
        // Dictionary equality is the equivalence test (decision 5).
        let (reimported, _) =
            definition_from_plist(&rendered, "/x/y.plist").expect("re-import");
        assert!(
            def.diff(&reimported).is_empty(),
            "diff: {:?}",
            def.diff(&reimported)
        );
        // The pure render/re-import pair preserves the dictionary exactly
        // for value types that plutil keeps stable (bool/string/int/array).
        let render_obj = rendered.as_object().unwrap();
        let source_obj = source.as_object().unwrap();
        for (k, v) in source_obj {
            assert_eq!(render_obj.get(k), Some(v), "key {k} changed");
        }
    }

    #[test]
    fn xml_rendering_is_deterministic_and_escaped() {
        let mut source = llama_swap_plist();
        source["ProgramArguments"] = serde_json::json!([
            "/bin/echo", "<tag & \"quoted\">", "line\nbreak"
        ]);
        let (def, _) = definition_from_plist(&source, "/x").expect("import");
        let xml = render_plist(&def, false).expect("xml");
        assert!(xml.contains("&lt;tag &amp; &quot;quoted&quot;&gt;"), "{xml}");
        assert_eq!(render_plist(&def, false).unwrap(), xml);
        // Control characters must never appear raw.
        assert!(xml.contains("line&#xa;break"), "{xml}");
    }

    #[test]
    fn keepalive_dict_is_kept_verbatim() {
        let mut source = llama_swap_plist();
        source["KeepAlive"] = serde_json::json!({"SuccessfulExit": false});
        let (def, _) = definition_from_plist(&source, "/x").expect("import");
        assert!(!def.launchd.as_ref().unwrap().keep_alive.is_always());
        assert_eq!(def.restart, "no");
        let rendered = plist_from_definition(&def).unwrap();
        assert_eq!(rendered.get("KeepAlive"), source.get("KeepAlive"));
    }

    #[test]
    fn uncovered_keys_reports_unknown_directives() {
        let mut source = llama_swap_plist();
        source["Nice"] = Value::from(5);
        source["MachServices"] = serde_json::json!({"a": true});
        let uncovered = uncovered_keys_in(&source);
        assert_eq!(uncovered, vec!["MachServices", "Nice"]);
    }

    const PRINT_SAMPLE: &str = r#"gui/501/com.pitcany.llama-swap = {
	active count = 1
	path = /Users/yannik/Library/LaunchAgents/com.pitcany.llama-swap.plist
	state = running
	program = /Users/yannik/local-llm/bin/llama-swap
	arguments = {
		/Users/yannik/local-llm/bin/llama-swap
		--config
		/Users/yannik/local-llm/llama-swap/config.yaml
	}

	stdout path = /Users/yannik/Library/Logs/llama-swap/llama-swap.out.log
	stderr path = /Users/yannik/Library/Logs/llama-swap/llama-swap.err.log

	inactive count = 2
	run count = 4
	pid = 48121
	runs = 4
	last exit code = 78
	disabled = false

	services = {
			state = active
			state = active
	}
};
"#;

    #[test]
    fn print_parser_reads_state_pid_runs_program() {
        let p = parse_print(PRINT_SAMPLE);
        assert!(p.loaded);
        // The nested per-service `state = active` lines must not win.
        assert_eq!(p.state, "running");
        assert_eq!(p.pid, Some(48121));
        assert_eq!(p.runs, Some(4));
        assert_eq!(
            p.program.as_deref(),
            Some("/Users/yannik/local-llm/bin/llama-swap")
        );
        assert!(!p.disabled);
        assert_eq!(
            p.arguments.as_deref(),
            Some(
                &[
                    "/Users/yannik/local-llm/bin/llama-swap".to_string(),
                    "--config".to_string(),
                    "/Users/yannik/local-llm/llama-swap/config.yaml".to_string()
                ][..]
            )
        );
        // "run count" must not shadow "runs".
        assert_eq!(parse_print("\truns = 7\n").runs, Some(7));
        // A manager that cannot be asked yields nothing (not loaded).
        let empty = parse_print("Could not find service \"x\" in domain for system");
        assert!(!empty.loaded);
    }

    #[test]
    fn print_parser_also_accepts_the_quoted_paren_form() {
        let legacy = "\t\targuments = (\n\t\t\t\"/bin/echo\",\n\t\t\t\"two words\"\n\t\t)\n";
        assert_eq!(
            parse_arguments_block(legacy).as_deref(),
            Some(&["/bin/echo".to_string(), "two words".to_string()][..])
        );
    }
    #[test]
    fn an_app_instance_is_found_by_the_stable_part_of_its_label() {
        let list = "PID\tStatus\tLabel\n\
            -\t0\tapplication.ai.elementlabs.lmstudio.111.222\n\
            37069\t0\tapplication.ai.elementlabs.lmstudio.13994546.64849245\n\
            500\t0\tapplication.ai.elementlabs.lmstudio.helper.1.2\n\
            3387\t0\tcom.pitcany.ollama\n";
        assert_eq!(
            app_instance_label(list, "application.ai.elementlabs.lmstudio").as_deref(),
            Some("application.ai.elementlabs.lmstudio.13994546.64849245")
        );
        // Only an exited instance is listed: still the app's label.
        let exited = "-\t0\tapplication.com.example.app.7.9\n";
        assert_eq!(
            app_instance_label(exited, "application.com.example.app").as_deref(),
            Some("application.com.example.app.7.9")
        );
        // A different app whose bundle id merely starts the same way.
        assert_eq!(app_instance_label(list, "application.ai.elementlabs"), None);
        assert_eq!(app_instance_label(list, "application.com.absent"), None);
    }

    #[test]
    fn the_newest_file_below_a_log_directory_is_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let month = dir.path().join("2026-11");
        std::fs::create_dir(&month).unwrap();
        let old = month.join("2026-11-17.1.log");
        let new = month.join("2026-11-18.1.log");
        std::fs::write(&old, "old").unwrap();
        std::fs::write(&new, "new").unwrap();
        let past = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        std::fs::File::options().write(true).open(&old).unwrap().set_modified(past).unwrap();
        assert_eq!(newest_log(dir.path()), Some(new.clone()));
        // A file is its own answer; nothing there is no answer.
        assert_eq!(newest_log(&old), Some(old));
        assert_eq!(newest_log(&dir.path().join("absent")), None);
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(newest_log(empty.path()), None);
    }

}
