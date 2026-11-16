//! Import, render and round-trip-verify a systemd unit's effective launch
//! definition.
//!
//! Import reads the manager's D-Bus view through `busctl --json`, which
//! keeps exact argv arrays and microsecond timeouts (`systemctl show` joins
//! argv with spaces and loses argument boundaries). Equivalence of a
//! generated service definition is proven by systemd itself: the rendered
//! unit is loaded under a scratch name (never started), re-imported, and
//! compared field by field.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::process::Command;

use llmctl_core::{
    EnvFileRef, ExecCommand, FieldDiff, FileProvenance, LaunchDefinition, UnitName, UnitScopeMarker,
};

use crate::engine::EngineError;

const SERVICE_IFACE: &str = "org.freedesktop.systemd1.Service";
const UNIT_IFACE: &str = "org.freedesktop.systemd1.Unit";
const SERVICE_PROPS: [&str; 11] = [
    "Type",
    "ExecStartPre",
    "ExecStart",
    "ExecStartPost",
    "ExecStop",
    "WorkingDirectory",
    "Environment",
    "EnvironmentFiles",
    "Restart",
    "TimeoutStopUSec",
    "KillMode",
];

/// What import learned about one unit.
#[derive(Debug, Clone)]
pub struct ImportedUnit {
    pub load_state: String,
    /// `None` unless the unit is loaded.
    pub definition: Option<LaunchDefinition>,
    pub provenance: Vec<FileProvenance>,
    /// Inline `Environment=` names that look like credentials. Their values
    /// are never kept; their presence blocks adoption.
    pub secret_env_names: Vec<String>,
}

pub async fn import_unit(unit: &UnitName, timeout: Duration) -> Result<ImportedUnit, EngineError> {
    let scope = scope_flag(unit.scope)?;
    let loaded = busctl(
        scope,
        &[
            "call",
            "org.freedesktop.systemd1",
            "/org/freedesktop/systemd1",
            "org.freedesktop.systemd1.Manager",
            "LoadUnit",
            "s",
            &unit.name,
        ],
        timeout,
    )
    .await?;
    let object = loaded
        .first()
        .and_then(|v| v["data"][0].as_str())
        .ok_or_else(|| EngineError::Io(format!("LoadUnit {unit}: no object path")))?
        .to_string();

    let unit_props = busctl(
        scope,
        &[
            "get-property",
            "org.freedesktop.systemd1",
            &object,
            UNIT_IFACE,
            "LoadState",
            "FragmentPath",
            "DropInPaths",
        ],
        timeout,
    )
    .await?;
    let [load_state, fragment, dropins] = take3(&unit_props, unit)?;
    let load_state = load_state["data"].as_str().unwrap_or("").to_string();

    let mut provenance = Vec::new();
    let mut files: Vec<String> = Vec::new();
    if let Some(f) = fragment["data"].as_str().filter(|s| !s.is_empty()) {
        files.push(f.to_string());
    }
    if let Some(list) = dropins["data"].as_array() {
        files.extend(list.iter().filter_map(|v| v.as_str().map(str::to_owned)));
    }
    for path in files {
        provenance.push(FileProvenance {
            content_digest: file_digest(Path::new(&path)).await,
            path,
        });
    }

    if load_state != "loaded" {
        return Ok(ImportedUnit {
            load_state,
            definition: None,
            provenance,
            secret_env_names: Vec::new(),
        });
    }

    let mut args = vec![
        "get-property",
        "org.freedesktop.systemd1",
        &object,
        SERVICE_IFACE,
    ];
    args.extend(SERVICE_PROPS);
    let props = busctl(scope, &args, timeout).await?;
    if props.len() != SERVICE_PROPS.len() {
        return Err(EngineError::Io(format!(
            "busctl returned {} of {} service properties for {unit}",
            props.len(),
            SERVICE_PROPS.len()
        )));
    }
    let data = |i: usize| &props[i]["data"];

    let mut environment = Vec::new();
    let mut secret_env_names = Vec::new();
    for assignment in data(6).as_array().into_iter().flatten() {
        let Some((name, value)) = assignment.as_str().and_then(|a| a.split_once('=')) else {
            continue;
        };
        if llmctl_core::launch::looks_secret(name) {
            secret_env_names.push(name.to_string());
        } else {
            environment.push((name.to_string(), value.to_string()));
        }
    }
    environment.sort();

    let mut environment_files = Vec::new();
    for entry in data(7).as_array().into_iter().flatten() {
        let path = entry[0].as_str().unwrap_or("").to_string();
        environment_files.push(EnvFileRef {
            content_digest: file_digest(Path::new(&path)).await,
            optional: entry[1].as_bool().unwrap_or(false),
            path,
        });
    }

    let definition = LaunchDefinition {
        service_type: data(0).as_str().unwrap_or("").to_string(),
        exec_start_pre: parse_exec(data(1)),
        exec_start: parse_exec(data(2)),
        exec_start_post: parse_exec(data(3)),
        exec_stop: parse_exec(data(4)),
        working_directory: data(5)
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        environment,
        environment_files,
        restart: data(8).as_str().unwrap_or("").to_string(),
        timeout_stop_usec: data(9).as_u64().unwrap_or(0),
        kill_mode: data(10).as_str().unwrap_or("").to_string(),
    };
    Ok(ImportedUnit {
        load_state,
        definition: Some(definition),
        provenance,
        secret_env_names,
    })
}

/// `a(sasbttttuii)` → commands; runtime fields (pids, timestamps) dropped.
fn parse_exec(v: &Value) -> Vec<ExecCommand> {
    v.as_array()
        .into_iter()
        .flatten()
        .map(|cmd| ExecCommand {
            path: cmd[0].as_str().unwrap_or("").to_string(),
            argv: cmd[1]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|a| a.as_str().map(str::to_owned))
                .collect(),
            ignore_errors: cmd[2].as_bool().unwrap_or(false),
        })
        .collect()
}

fn take3<'a>(props: &'a [Value], unit: &UnitName) -> Result<[&'a Value; 3], EngineError> {
    match props {
        [a, b, c] => Ok([a, b, c]),
        _ => Err(EngineError::Io(format!(
            "busctl returned {} unit properties for {unit}",
            props.len()
        ))),
    }
}

async fn file_digest(path: &Path) -> String {
    match tokio::fs::read(path).await {
        Ok(bytes) => format!("sha256:{:x}", Sha256::digest(bytes)),
        Err(_) => "missing".to_string(),
    }
}

fn scope_flag(scope: UnitScopeMarker) -> Result<&'static str, EngineError> {
    match scope {
        UnitScopeMarker::User => Ok("--user"),
        UnitScopeMarker::System => Ok("--system"),
        UnitScopeMarker::Launchd => Err(EngineError::Unsupported(
            "launchd labels are not systemd units".into(),
        )),
    }
}

/// Runs busctl with `--json=short`; one JSON value per output line.
async fn busctl(scope: &str, args: &[&str], timeout: Duration) -> Result<Vec<Value>, EngineError> {
    let mut cmd = Command::new("busctl");
    cmd.arg(scope)
        .arg("--json=short")
        .arg("--no-pager")
        .args(args)
        .kill_on_drop(true);
    let out = match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(EngineError::Io(format!("spawning busctl: {e}"))),
        Err(_) => {
            return Err(EngineError::Timeout(format!(
                "busctl {} timed out",
                args.first().unwrap_or(&"")
            )))
        }
    };
    if !out.status.success() {
        return Err(EngineError::Io(format!(
            "busctl {}: {}",
            args.first().unwrap_or(&""),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| EngineError::Io(format!("busctl json: {e}"))))
        .collect()
}

// ---- rendering --------------------------------------------------------------

/// The service definition llmctl would own for `def`.
pub fn render_unit(def: &LaunchDefinition, description: &str) -> String {
    let mut out = String::new();
    out.push_str("[Unit]\n");
    out.push_str(&format!("Description={}\n\n", escape_percent(description)));
    out.push_str("[Service]\n");
    out.push_str(&format!("Type={}\n", def.service_type));
    for (key, cmds) in [
        ("ExecStartPre", &def.exec_start_pre),
        ("ExecStart", &def.exec_start),
        ("ExecStartPost", &def.exec_start_post),
        ("ExecStop", &def.exec_stop),
    ] {
        for cmd in cmds {
            out.push_str(&format!("{key}={}\n", render_exec(cmd)));
        }
    }
    if let Some(wd) = &def.working_directory {
        // D-Bus marks "missing is ok" with `!`; unit files spell it `-`.
        let wd = wd.strip_prefix('!').map(|w| format!("-{w}"));
        let wd = wd
            .as_deref()
            .unwrap_or(def.working_directory.as_deref().unwrap());
        out.push_str(&format!("WorkingDirectory={}\n", escape_percent(wd)));
    }
    for (name, value) in &def.environment {
        out.push_str(&format!(
            "Environment={}\n",
            quote_word(&format!("{name}={value}"), true)
        ));
    }
    for file in &def.environment_files {
        let prefix = if file.optional { "-" } else { "" };
        out.push_str(&format!(
            "EnvironmentFile={prefix}{}\n",
            escape_percent(&file.path)
        ));
    }
    out.push_str(&format!("Restart={}\n", def.restart));
    out.push_str(&format!("TimeoutStopSec={}us\n", def.timeout_stop_usec));
    out.push_str(&format!("KillMode={}\n", def.kill_mode));
    out
}

fn render_exec(cmd: &ExecCommand) -> String {
    let mut s = String::new();
    if cmd.ignore_errors {
        s.push('-');
    }
    let argv0_matches = cmd.argv.first().map(|a| a == &cmd.path).unwrap_or(false);
    if !argv0_matches {
        // `@path argv0 …`: the executed binary differs from argv[0].
        s.push('@');
        s.push_str(&quote_word(&cmd.path, false));
        s.push(' ');
    }
    let words: Vec<String> = if cmd.argv.is_empty() {
        vec![quote_word(&cmd.path, false)]
    } else {
        cmd.argv.iter().map(|a| quote_word(a, false)).collect()
    };
    s.push_str(&words.join(" "));
    s
}

fn escape_percent(s: &str) -> String {
    s.replace('%', "%%")
}

/// Quote one unit-file word. `always` forces quotes (Environment=).
fn quote_word(word: &str, always: bool) -> String {
    let needs = always
        || word.is_empty()
        || word
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '"' | '\'' | '\\' | ';'));
    let escaped = escape_percent(word);
    if !needs {
        return escaped;
    }
    let mut q = String::with_capacity(escaped.len() + 2);
    q.push('"');
    for c in escaped.chars() {
        match c {
            '"' => q.push_str("\\\""),
            '\\' => q.push_str("\\\\"),
            '\n' => q.push_str("\\n"),
            '\t' => q.push_str("\\t"),
            c => q.push(c),
        }
    }
    q.push('"');
    q
}

// ---- round trip -------------------------------------------------------------

/// Load `rendered` under a scratch user unit name, re-import it, and return
/// how its effective definition differs from `expected` (empty = proven
/// equivalent). The scratch unit is never started and is always removed.
pub async fn roundtrip(
    expected: &LaunchDefinition,
    rendered: &str,
    timeout: Duration,
) -> Result<Vec<FieldDiff>, EngineError> {
    let name = format!(
        "llmctl-next-verify-{}.service",
        &uuid::Uuid::new_v4().simple().to_string()[..12]
    );
    let dir = user_unit_dir()?;
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| EngineError::Io(format!("creating {}: {e}", dir.display())))?;
    let path = dir.join(&name);
    tokio::fs::write(&path, rendered)
        .await
        .map_err(|e| EngineError::Io(format!("writing {}: {e}", path.display())))?;

    let result = async {
        daemon_reload(timeout).await?;
        let unit = UnitName::systemd(UnitScopeMarker::User, &name)
            .map_err(|e| EngineError::Io(e.to_string()))?;
        let imported = import_unit(&unit, timeout).await?;
        match imported.definition {
            Some(def) => Ok(expected.diff(&def)),
            None => Err(EngineError::Io(format!(
                "systemd refused the generated definition (load state {})",
                imported.load_state
            ))),
        }
    }
    .await;

    let _ = tokio::fs::remove_file(&path).await;
    let _ = daemon_reload(timeout).await;
    result
}

fn user_unit_dir() -> Result<PathBuf, EngineError> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Ok(PathBuf::from(xdg).join("systemd/user"));
        }
    }
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".config/systemd/user"))
        .map_err(|_| EngineError::Io("HOME is not set".into()))
}

async fn daemon_reload(timeout: Duration) -> Result<(), EngineError> {
    let mut cmd = Command::new("systemctl");
    cmd.args(["--user", "daemon-reload"]).kill_on_drop(true);
    match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) if out.status.success() => Ok(()),
        Ok(Ok(out)) => Err(EngineError::Io(format!(
            "daemon-reload: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))),
        Ok(Err(e)) => Err(EngineError::Io(format!("spawning systemctl: {e}"))),
        Err(_) => Err(EngineError::Timeout("daemon-reload timed out".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_exec_keeps_exact_argv_and_drops_runtime_fields() {
        let v = json!([[
            "/bin/echo",
            ["/bin/echo", "two words", "x"],
            true,
            1,
            2,
            3,
            4,
            99,
            0,
            0
        ]]);
        let cmds = parse_exec(&v);
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0].argv, vec!["/bin/echo", "two words", "x"]);
        assert!(cmds[0].ignore_errors);
    }

    #[test]
    fn render_quotes_boundaries_specifiers_and_prefixes() {
        let def = LaunchDefinition {
            service_type: "simple".into(),
            exec_start_pre: vec![ExecCommand {
                path: "/bin/true".into(),
                argv: vec!["/bin/true".into()],
                ignore_errors: true,
            }],
            exec_start: vec![ExecCommand {
                path: "/usr/bin/python3".into(),
                argv: vec![
                    "py".into(),
                    "-c".into(),
                    "print(\"hi there\")".into(),
                    "100%".into(),
                    "$FOO".into(),
                ],
                ignore_errors: false,
            }],
            exec_start_post: vec![],
            exec_stop: vec![],
            working_directory: Some("!/tmp".into()),
            environment: vec![("A".into(), "1 2".into())],
            environment_files: vec![EnvFileRef {
                path: "/x.env".into(),
                optional: true,
                content_digest: "missing".into(),
            }],
            restart: "no".into(),
            timeout_stop_usec: 95_000_000,
            kill_mode: "control-group".into(),
        };
        let text = render_unit(&def, "spike");
        assert!(text.contains("ExecStartPre=-/bin/true\n"), "{text}");
        assert!(
            text.contains(r#"ExecStart=@/usr/bin/python3 py -c "print(\"hi there\")" 100%% $FOO"#),
            "{text}"
        );
        assert!(text.contains("WorkingDirectory=-/tmp\n"), "{text}");
        assert!(text.contains("Environment=\"A=1 2\"\n"), "{text}");
        assert!(text.contains("EnvironmentFile=-/x.env\n"), "{text}");
        assert!(text.contains("TimeoutStopSec=95000000us\n"), "{text}");
    }
}
