//! Version facts and offline validation for staged environments (plan D).
//!
//! The drift guard pins *what the interpreter reports* (package versions,
//! CUDA, a `pip freeze` digest) plus content digests of non-system `Exec*`
//! files (the launcher script), so an in-place `pip install -U` or a
//! launcher edit is drift even though the unit file never changed.

use std::path::Path;
use std::time::Duration;

use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::process::Command;

use llmctl_core::{LaunchDefinition, RecipeConfig, VersionRef, WorkloadDefinition};

use crate::engine::EngineError;

const PROBE_TIMEOUT: Duration = Duration::from_secs(60);
/// `pip freeze` on a big env is slow; give it room.
const FREEZE_TIMEOUT: Duration = Duration::from_secs(180);

/// The interpreter this workload runs, per its recipe (explicit override,
/// else the first `Exec*` path that is a python).
pub fn interpreter_for(recipe: &RecipeConfig, def: &WorkloadDefinition) -> Option<String> {
    if let Some(explicit) = &recipe.interpreter {
        return Some(explicit.clone());
    }
    let pinned = match &def.ownership {
        llmctl_core::Ownership::Managed { definition, .. } => Some(&**definition),
        _ => None,
    };
    let execs: Vec<&llmctl_core::ExecCommand> = pinned
        .map(|d| {
            d.exec_start
                .iter()
                .chain(d.exec_start_pre.iter())
                .chain(d.exec_start_post.iter())
                .collect()
        })
        .unwrap_or_default();
    execs
        .iter()
        .map(|c| c.path.as_str())
        .chain(
            def.unit
                .name
                .is_empty()
                .then_some("")
                .into_iter()
                .map(|_| ""),
        )
        .find(|p| {
            let base = p.rsplit('/').next().unwrap_or(p);
            base == "python" || base.starts_with("python3") || base.starts_with("python.")
        })
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
}

/// Probe the environment's version facts.
pub async fn probe(
    recipe: &RecipeConfig,
    def: &WorkloadDefinition,
) -> Result<VersionRef, EngineError> {
    let interpreter = interpreter_for(recipe, def)
        .ok_or_else(|| EngineError::Io("recipe has no interpreter to probe".into()))?;

    // Package versions as the env reports them.
    let mut packages: Vec<(String, String)> = Vec::new();
    if !recipe.packages.is_empty() {
        let names = format!("{:?}", recipe.packages); // a python list literal
        let code = format!(
            "import importlib.metadata as m, json\nprint(json.dumps({{n: m.version(n) for n in {names}}}))"
        );
        let out = run_python(&interpreter, &["-c", &code], PROBE_TIMEOUT).await?;
        let map: serde_json::Map<String, Value> = serde_json::from_str(out.trim())
            .map_err(|e| EngineError::Io(format!("version probe output: {e}")))?;
        for (k, v) in map {
            if let Some(v) = v.as_str() {
                packages.push((k, v.to_string()));
            }
        }
        packages.sort();
    }

    // CUDA as the env reports it (only meaningful with torch).
    let cuda = if recipe.packages.iter().any(|p| p == "torch") {
        run_python(
            &interpreter,
            &["-c", "import torch\nprint(torch.version.cuda)"],
            PROBE_TIMEOUT,
        )
        .await
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s != "None")
    } else {
        None
    };

    // Lock digest: sorted `pip freeze`.
    let freeze = run_python(&interpreter, &["-m", "pip", "freeze"], FREEZE_TIMEOUT)
        .await
        .unwrap_or_default();
    let mut lines: Vec<&str> = freeze
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines.sort();
    let lock_digest = sha256_hex_string(lines.join("\n").as_bytes());

    // Non-system Exec* file digests (the launcher script).
    let exec_digests = exec_digests_of(def).await;

    Ok(VersionRef {
        interpreter: Some(interpreter),
        packages,
        cuda,
        lock_digest,
        exec_digests,
    })
}

fn sha256_hex_string(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

/// Content digests of every `Exec*` path outside the system prefixes.
pub async fn exec_digests_of(def: &WorkloadDefinition) -> Vec<(String, String)> {
    let llmctl_core::Ownership::Managed { definition, .. } = &def.ownership else {
        return Vec::new();
    };
    let mut paths: Vec<String> = Vec::new();
    for cmd in definition
        .exec_start_pre
        .iter()
        .chain(definition.exec_start.iter())
        .chain(definition.exec_start_post.iter())
        .chain(definition.exec_stop.iter())
    {
        if is_system_path(&cmd.path) {
            continue;
        }
        if !paths.contains(&cmd.path) {
            paths.push(cmd.path.clone());
        }
    }
    let mut out = Vec::new();
    for path in paths {
        let digest = match tokio::fs::read(&path).await {
            Ok(bytes) => format!("sha256:{:x}", Sha256::digest(bytes)),
            Err(_) => "missing".into(),
        };
        out.push((path, digest));
    }
    out
}

fn is_system_path(path: &str) -> bool {
    ["/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt"]
        .iter()
        .any(|p| path == *p || path.starts_with(&format!("{p}/")))
}

/// Offline validation of a staged environment (plan D.4): the package
/// imports, reports the expected version, and the served model's config
/// loads under the new version. Never touches the serving env.
pub async fn validate_offline(
    recipe: &RecipeConfig,
    interpreter: &str,
    model: &str,
) -> Result<Value, String> {
    let code = match recipe.name.as_str() {
        "vllm" => format!(
            "import json, torch, vllm\nfrom transformers import AutoConfig\nc = AutoConfig.from_pretrained({model:?})\nprint(json.dumps({{'vllm': vllm.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'model_type': c.model_type}}))"
        ),
        other => format!(
            "import json\nimport {other} as pkg\nprint(json.dumps(pkg.validate({model:?})))"
        ),
    };
    let out = run_python(interpreter, &["-c", &code], PROBE_TIMEOUT)
        .await
        .map_err(|e| format!("validation command failed: {e}"))?;
    serde_json::from_str(out.trim()).map_err(|e| format!("validation output not JSON: {e}"))
}

/// The definition for the staged interpreter: `exec` recipes run the
/// interpreter directly (rewrite `ExecStart`), `env` recipes keep a
/// launcher and point its env var at the new interpreter.
pub fn rewrite_definition(
    old: &LaunchDefinition,
    recipe: &RecipeConfig,
    interpreter: &str,
) -> LaunchDefinition {
    let mut next = old.clone();
    match recipe.apply.as_str() {
        "exec" => {
            for cmd in &mut next.exec_start {
                if !is_system_path(&cmd.path) || cmd.path.contains("python") {
                    cmd.path = interpreter.to_string();
                    if !cmd.argv.is_empty() {
                        cmd.argv[0] = interpreter.to_string();
                    }
                }
            }
        }
        _ => {
            let prefix = recipe.env_prefix.clone().unwrap_or_else(|| "VLLM".into());
            let name = format!("{prefix}_PYTHON");
            next.environment.retain(|(k, _)| k != &name);
            next.environment.push((name, interpreter.to_string()));
            next.environment.sort();
        }
    }
    next
}

/// Verify the environment reports exactly `packages` through
/// `importlib.metadata` (the `VerifyVersion` step).
pub async fn verify_packages(
    interpreter: &str,
    packages: &[(String, String)],
) -> Result<(), String> {
    if packages.is_empty() {
        return Ok(());
    }
    let names: Vec<&str> = packages.iter().map(|(n, _)| n.as_str()).collect();
    let code = format!(
        "import importlib.metadata as m, json\nprint(json.dumps({{n: m.version(n) for n in {names:?}}}))",
        names = names
    );
    let out = run_python(interpreter, &["-c", &code], PROBE_TIMEOUT)
        .await
        .map_err(|e| e.to_string())?;
    let got: std::collections::BTreeMap<String, String> =
        serde_json::from_str(out.trim()).map_err(|e| format!("version probe output: {e}"))?;
    let mut wrong = Vec::new();
    for (name, want) in packages {
        match got.get(name) {
            Some(have) if have == want => {}
            Some(have) => wrong.push(format!("{name}: running {have}, expected {want}")),
            None => wrong.push(format!("{name}: not installed")),
        }
    }
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(wrong.join("; "))
    }
}

async fn run_python(
    interpreter: &str,
    args: &[&str],
    timeout: Duration,
) -> Result<String, EngineError> {
    let mut cmd = Command::new(interpreter);
    cmd.args(args).kill_on_drop(true);
    let out = match tokio::time::timeout(timeout, cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(e)) => return Err(EngineError::Io(format!("spawning {interpreter}: {e}"))),
        Err(_) => {
            return Err(EngineError::Timeout(format!(
                "{interpreter} {} timed out",
                args.first().unwrap_or(&"")
            )))
        }
    };
    if !out.status.success() {
        return Err(EngineError::Io(format!(
            "{interpreter} {} exited {}: {}",
            args.first().unwrap_or(&""),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// True when the recomputed version facts differ from the pin.
pub fn version_drift(pinned: &VersionRef, live: &VersionRef) -> Vec<llmctl_core::FieldDiff> {
    let mut diffs = Vec::new();
    let mut push = |field: &str, expected: Value, actual: Value| {
        if expected != actual {
            diffs.push(llmctl_core::FieldDiff {
                field: field.to_string(),
                expected,
                actual,
            });
        }
    };
    push(
        "version.interpreter",
        serde_json::to_value(&pinned.interpreter).unwrap_or(Value::Null),
        serde_json::to_value(&live.interpreter).unwrap_or(Value::Null),
    );
    push(
        "version.packages",
        serde_json::to_value(&pinned.packages).unwrap_or(Value::Null),
        serde_json::to_value(&live.packages).unwrap_or(Value::Null),
    );
    push(
        "version.cuda",
        serde_json::to_value(&pinned.cuda).unwrap_or(Value::Null),
        serde_json::to_value(&live.cuda).unwrap_or(Value::Null),
    );
    push(
        "version.lock_digest",
        Value::String(pinned.lock_digest.clone()),
        Value::String(live.lock_digest.clone()),
    );
    push(
        "version.exec_digests",
        serde_json::to_value(&pinned.exec_digests).unwrap_or(Value::Null),
        serde_json::to_value(&live.exec_digests).unwrap_or(Value::Null),
    );
    let _ = Path::new("");
    diffs
}
