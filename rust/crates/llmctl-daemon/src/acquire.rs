//! Model acquisition (M4c item 1): download weights into a declared store
//! as an operation — disk checked first, progress in the journal,
//! cancellable, and running beside lifecycle operations so a long download
//! never blocks a swap.
//!
//! The tools do the downloading (`hf download`, `ollama pull`, `lms get`); this
//! module decides whether there is room, builds the command, and watches.
//! Credentials stay with the tools: the Hugging Face CLI reads its own
//! cached token, and nothing here reads or prints one.

use std::path::{Path, PathBuf};
use std::time::Duration;

use llmctl_client::wire::{AcquireRequest, AcquireResponse, StoreView, StoresResponse};
use llmctl_core::acquire::{check_space, hf_dry_run_bytes, GIB};
use llmctl_core::{ModelStore, OpKind, Plan, PlanStep, Source, StoreKind, WorkloadName};

use crate::app::AppState;
use crate::ops::{AdmitOutcome, AdmitParams, OpsError};

/// Test hooks / overrides for the tools (default: found on `PATH`; `lms`
/// in LM Studio's own install path first).
pub const HF_BIN_ENV: &str = "LLMCTL_NEXT_HF_BIN";
pub const OLLAMA_BIN_ENV: &str = "LLMCTL_NEXT_OLLAMA_BIN";
pub const LMS_BIN_ENV: &str = "LLMCTL_NEXT_LMS_BIN";

/// Longest one download may run.
const TIMEOUT_SECS: u64 = 12 * 3600;

fn tool(env: &str, default: &str) -> String {
    std::env::var(env).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

/// Free bytes on the filesystem that holds `path` (`df -Pk`, which GNU and
/// macOS both speak). The nearest existing parent is asked when the store
/// directory does not exist yet. `None` when it cannot be read.
pub async fn free_bytes(path: &Path) -> Option<u64> {
    let mut probe = path.to_path_buf();
    while !probe.exists() {
        probe = probe.parent()?.to_path_buf();
    }
    let out = tokio::process::Command::new("df").arg("-Pk").arg(&probe).output().await.ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let avail_kib: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
    Some(avail_kib * 1024)
}

fn kind_name(kind: StoreKind) -> &'static str {
    match kind {
        StoreKind::HfCache => "hf_cache",
        StoreKind::Directory => "directory",
        StoreKind::Ollama => "ollama",
        StoreKind::Lmstudio => "lmstudio",
    }
}

pub async fn stores(state: &AppState) -> StoresResponse {
    let inv = state.ops.inventory().await;
    let mut out = Vec::new();
    for s in &inv.stores {
        let path = Path::new(&s.path);
        out.push(StoreView {
            name: s.name.clone(),
            kind: kind_name(s.kind).into(),
            path: s.path.clone(),
            reserve_gib: s.reserve_gib,
            workload: s.workload.as_ref().map(|w| w.to_string()),
            exists: path.exists(),
            free_bytes: free_bytes(path).await,
        });
    }
    StoresResponse { host: inv.host.to_string(), stores: out }
}

/// The command that puts `source` into `store`, with the environment it
/// needs. `port` is the Ollama server's, for an Ollama store.
pub fn command(source: &Source, store: &ModelStore, port: Option<u16>) -> (Vec<String>, Vec<(String, String)>) {
    match source {
        Source::Hf { repo, revision } => {
            let mut argv = vec![tool(HF_BIN_ENV, "hf"), "download".into(), repo.clone()];
            if let Some(rev) = revision {
                argv.extend(["--revision".into(), rev.clone()]);
            }
            match store.kind {
                StoreKind::Directory => {
                    let name = repo.rsplit('/').next().unwrap_or(repo);
                    argv.extend(["--local-dir".into(), format!("{}/{name}", store.path.trim_end_matches('/'))]);
                }
                _ => argv.extend(["--cache-dir".into(), store.path.clone()]),
            }
            (argv, Vec::new())
        }
        Source::Ollama { tag } => (
            vec![tool(OLLAMA_BIN_ENV, "ollama"), "pull".into(), tag.clone()],
            port.map(|p| ("OLLAMA_HOST".to_string(), format!("127.0.0.1:{p}"))).into_iter().collect(),
        ),
        // `--yes`: no prompt may wait on a terminal that is not there. The
        // name is an exact catalog entry (see `Source::parse`), so "the
        // first match" cannot pick something else.
        Source::Lms { model, variant } => {
            let lms = tool(LMS_BIN_ENV, &crate::engine_api::lms_binary().to_string_lossy());
            let name = match variant {
                Some(v) => format!("{model}@{v}"),
                None => model.clone(),
            };
            (vec![lms, "get".into(), name, "--yes".into()], Vec::new())
        }
    }
}

/// How much is left to fetch, through the tool's own dry run of the very
/// command the operation will run (same repository, revision and
/// destination, so files already in that store count as zero).
async fn hf_size(argv: &[String]) -> Option<u64> {
    let (program, args) = argv.split_first()?;
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args)
        .args(["--dry-run", "--format", "json"])
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    let out = tokio::time::timeout(Duration::from_secs(90), cmd.output()).await.ok()?.ok()?;
    hf_dry_run_bytes(&String::from_utf8_lossy(&out.stdout))
}

/// Check and preview a download; unless `dry_run`, admit it.
pub async fn acquire(
    state: &AppState,
    req: &AcquireRequest,
) -> Result<(AcquireResponse, Option<AdmitOutcome>), OpsError> {
    let source = Source::parse(&req.source).map_err(OpsError::Invalid)?;
    let inv = state.ops.inventory().await;
    let store = inv
        .stores
        .iter()
        .find(|s| s.name == req.store)
        .ok_or_else(|| {
            let known: Vec<&str> = inv.stores.iter().map(|s| s.name.as_str()).collect();
            OpsError::Invalid(format!(
                "no store {:?} in the inventory (declared: {})",
                req.store,
                if known.is_empty() { "none".to_string() } else { known.join(", ") }
            ))
        })?;
    if !source.fits_store(store.kind) {
        return Err(OpsError::Invalid(format!(
            "{source} cannot go into store {} (kind {})",
            store.name,
            kind_name(store.kind)
        )));
    }
    let workload: WorkloadName = match (&req.for_workload, &store.workload) {
        (Some(w), _) => WorkloadName::new(w).map_err(|e| OpsError::Invalid(e.to_string()))?,
        (None, Some(w)) => w.clone(),
        (None, None) => {
            return Err(OpsError::Invalid(format!(
                "store {} names no workload; say which workload the download is for (--for)",
                store.name
            )))
        }
    };
    let def = inv
        .workload(&workload)
        .ok_or_else(|| OpsError::UnknownWorkload(workload.to_string()))?;
    let port = match store.kind {
        StoreKind::Ollama => store.workload.as_ref().and_then(|w| inv.workload(w)).map(|d| d.port),
        _ => None,
    };

    // `lms` talks to the LM Studio service and may start one of its own
    // when none answers. A download must not bring up a service this
    // daemon manages, least of all one that was stopped on purpose.
    if store.kind == StoreKind::Lmstudio {
        let Some(service) = store.workload.as_ref() else {
            return Err(OpsError::Invalid(format!(
                "store {} names no LM Studio workload to download through",
                store.name
            )));
        };
        let running = state
            .ops
            .engine()
            .observe(&inv)
            .await
            .map_err(|e| OpsError::Engine(e.to_string()))?
            .iter()
            .any(|o| &o.workload == service && o.process == llmctl_core::ProcessState::Running);
        if !running {
            return Err(OpsError::InvalidState(format!(
                "{source} into {}: {service} is not running, and `lms get` downloads through it; start it first (llmctl-next run {service})",
                store.name
            )));
        }
    }

    let (argv, env) = command(&source, store, port);
    // The disk check comes first: nothing is fetched into a full disk.
    let need = match (req.size_gib, &source) {
        (Some(g), _) => Some(g.saturating_mul(GIB)),
        (None, Source::Hf { .. }) => hf_size(&argv).await,
        // Neither tool can say a size without starting the download.
        (None, Source::Ollama { .. } | Source::Lms { .. }) => None,
    };
    let free = free_bytes(Path::new(&store.path)).await.ok_or_else(|| {
        OpsError::InvalidState(format!(
            "free space of {} could not be read; unknown is never enough",
            store.path
        ))
    })?;
    check_space(need, free, store.reserve_gib)
        .map_err(|e| OpsError::InvalidState(format!("{source} into {}: {e}", store.name)))?;

    let mut resp = AcquireResponse {
        source: source.to_string(),
        store: store.name.clone(),
        path: store.path.clone(),
        workload: workload.to_string(),
        need_bytes: need,
        free_bytes: free,
        reserve_gib: store.reserve_gib,
        command: argv.clone(),
        operation: None,
        replayed: false,
    };
    if req.dry_run {
        return Ok((resp, None));
    }
    let plan = Plan {
        id: llmctl_core::OperationId::generate(),
        kind: OpKind::Acquire,
        target: workload.clone(),
        revision: inv.revision,
        expected_model: def.model.clone(),
        affected_gpus: Vec::new(),
        steps: vec![PlanStep::Acquire {
            workload: workload.clone(),
            store: store.name.clone(),
            path: store.path.clone(),
            source: source.to_string(),
            argv,
            env,
            need_bytes: need,
            reserve_gib: store.reserve_gib,
            timeout_secs: TIMEOUT_SECS,
        }],
        // Nothing to undo: a partial download is resumed by the tool, and
        // weights are never removed by an operation.
        rollback: Vec::new(),
    };
    let key = req.idempotency_key.clone().unwrap_or_else(|| {
        format!("acquire-{}-{}", store.name, chrono::Utc::now().timestamp_millis())
    });
    let outcome = state
        .ops
        .admit(AdmitParams {
            protocol: llmctl_client::wire::PROTOCOL_VERSION,
            idempotency_key: key,
            kind: OpKind::Acquire,
            workload: workload.to_string(),
            expected_revision: None,
            expected_stop_targets: None,
            force: false,
            profile: None,
            plan_override: Some(plan),
        })
        .await?;
    resp.replayed = matches!(outcome, AdmitOutcome::Replayed(_));
    Ok((resp, Some(outcome)))
}

/// Where a download's own output goes (kept whole; the journal gets the
/// tail).
pub fn log_path(data_dir: &Path, op_id: &str) -> PathBuf {
    data_dir.join("acquire").join(format!("{op_id}.log"))
}

/// The last part of a download's log, for the journal: progress bars
/// rewrite one line with `\r`, so only the final state of each is kept.
pub async fn log_tail(path: &Path, max: usize) -> String {
    let bytes = tokio::fs::read(path).await.unwrap_or_default();
    let text = strip_ansi(&String::from_utf8_lossy(&bytes));
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.rsplit('\r').next().unwrap_or(l).trim())
        .filter(|l| !l.is_empty())
        // `lms` redraws spinner lines (braille frames) on fresh lines;
        // they say nothing once the run is over.
        .filter(|l| !l.chars().next().is_some_and(|c| ('\u{2800}'..='\u{28ff}').contains(&c)))
        .collect();
    let mut tail = lines.iter().rev().take(4).rev().cloned().collect::<Vec<_>>().join(" | ");
    if tail.len() > max {
        // Cut on a character boundary: progress bars are made of
        // multi-byte characters, and a byte offset inside one would panic.
        let mut start = tail.len() - max;
        while !tail.is_char_boundary(start) {
            start += 1;
        }
        tail = tail[start..].to_string();
    }
    tail
}

/// Drop terminal colour sequences (`ESC [ … letter`): the journal is read
/// as plain text.
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(kind: StoreKind, path: &str) -> ModelStore {
        ModelStore { name: "s".into(), kind, path: path.into(), reserve_gib: 20, workload: None }
    }

    #[test]
    fn the_command_follows_the_store_kind() {
        let hf = Source::parse("hf:Qwen/Qwen3.8-27B-FP8@abc123").unwrap();
        let (argv, env) = command(&hf, &store(StoreKind::HfCache, "/cache/hf"), None);
        assert_eq!(
            argv[1..],
            ["download", "Qwen/Qwen3.8-27B-FP8", "--revision", "abc123", "--cache-dir", "/cache/hf"]
        );
        assert!(env.is_empty());
        let (argv, _) = command(&hf, &store(StoreKind::Directory, "/models/"), None);
        assert_eq!(argv[argv.len() - 2..], ["--local-dir", "/models/Qwen3.8-27B-FP8"]);

        let ol = Source::parse("ollama:qwen3-embedding:0.6b").unwrap();
        let (argv, env) = command(&ol, &store(StoreKind::Ollama, "/ollama"), Some(11434));
        assert_eq!(argv[1..], ["pull", "qwen3-embedding:0.6b"]);
        assert_eq!(env, vec![("OLLAMA_HOST".to_string(), "127.0.0.1:11434".to_string())]);

        let lms = Source::parse("lms:qwen/qwen3.5-9b@q8_0").unwrap();
        let (argv, env) = command(&lms, &store(StoreKind::Lmstudio, "/lmstudio/models"), None);
        assert_eq!(argv[1..], ["get", "qwen/qwen3.5-9b@q8_0", "--yes"]);
        assert!(env.is_empty());
        let (argv, _) = command(&Source::parse("lms:openai/gpt-oss-20b").unwrap(), &store(StoreKind::Lmstudio, "/m"), None);
        assert_eq!(argv[1..], ["get", "openai/gpt-oss-20b", "--yes"]);
    }

    #[tokio::test]
    async fn free_space_is_read_even_before_the_store_exists() {
        let dir = tempfile::tempdir().unwrap();
        let here = free_bytes(dir.path()).await.expect("df answers");
        let below = free_bytes(&dir.path().join("not/yet/there")).await.expect("nearest parent");
        assert!(here > 0 && below > 0);
    }

    #[tokio::test]
    async fn the_log_tail_keeps_the_last_state_of_progress_lines() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("x.log");
        std::fs::write(&log, "Fetching 12 files\n  1%|\r 50%|\r100%|done\n\nDownloaded to /store\n").unwrap();
        assert_eq!(log_tail(&log, 500).await, "Fetching 12 files | 100%|done | Downloaded to /store");
        assert_eq!(log_tail(&dir.path().join("absent"), 500).await, "");
        std::fs::write(&log, "\u{1b}[32m✓ Downloaded\u{1b}[0m\n  path: /store\n").unwrap();
        assert_eq!(log_tail(&log, 500).await, "✓ Downloaded | path: /store");
        // `lms get` without a terminal: spinner frames on their own lines,
        // then the line that matters.
        std::fs::write(
            &log,
            "\u{1b}[?25l\u{1b}[0K\n\r   \u{2839} a/b - Resolving...\u{1b}[0K\n\r\u{1b}[0K\n\r\u{2827} Resolving download plan...\u{1b}[0K\n\u{1b}[?25hError: Failed to resolve artifact \"a/b\"\n",
        )
        .unwrap();
        assert_eq!(log_tail(&log, 500).await, "Error: Failed to resolve artifact \"a/b\"");
        // A long line of multi-byte progress blocks, cut at every offset.
        std::fs::write(&log, format!("{}\n", "█".repeat(300))).unwrap();
        for max in 1..40 {
            let tail = log_tail(&log, max).await;
            assert!(tail.len() <= max && tail.chars().all(|c| c == '█'), "max {max}: {tail:?}");
        }
    }
}
