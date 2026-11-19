//! `llmctl-next llamacpp …` (M5d): router models on a llama.cpp router
//! workload — import, list, load/unload by verb, preset edits, facts, logs.

use clap::Subcommand;

use llmctl_client::llamacpp::{
    RouterEdit, RouterEditRequest, RouterFactsRequest, RouterImportRequest, RouterLoadRequest,
    RouterModelsResponse,
};
use llmctl_client::wire::{LogsRequest, PROTOCOL_VERSION};
use llmctl_core::llamacpp::RouterModel;

use super::{connect, ensure_protocol, watch_and_report, Cli, CliError};

#[derive(Subcommand, Clone)]
pub enum LlamacppCmd {
    /// Import the router's preset INI into records (preview; `--confirm`
    /// with the preview's digest saves them). The INI is not touched.
    Import {
        workload: String,
        /// Preset path (default: `--models-preset` of its pinned launch).
        #[arg(long)]
        preset: Option<String>,
        #[arg(long)]
        confirm: Option<String>,
    },
    /// The router's models: record, state now, facts.
    Models {
        workload: String,
        #[arg(long)]
        json: bool,
    },
    /// Load one model, after the VRAM fit (or Mac memory budget), the
    /// models-max bound and the pre-load guard said yes.
    Load {
        workload: String,
        model: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        wait: bool,
        /// Stop the at-start workloads on the router's cards when the
        /// model does not fit beside them (started again if the load fails).
        #[arg(long)]
        evict: bool,
        /// Do not wait for requests in flight on what is stopped.
        #[arg(long)]
        force: bool,
        #[arg(long)]
        key: Option<String>,
    },
    /// Unload one model; the router keeps running.
    Unload {
        workload: String,
        model: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        wait: bool,
        /// Do not wait for requests in flight.
        #[arg(long)]
        force: bool,
        #[arg(long)]
        key: Option<String>,
    },
    /// Set one key of a model section (`*` = every model). Plans; apply
    /// with `--confirm <digest>`.
    Set {
        workload: String,
        model: String,
        key: String,
        value: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Remove one key from a model section.
    Unset {
        workload: String,
        model: String,
        key: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Add a model section.
    Add {
        workload: String,
        id: String,
        /// Absolute GGUF path (a split model: its first shard).
        #[arg(long)]
        model: String,
        #[arg(long)]
        mmproj: Option<String>,
        /// Extra key, `KEY=VALUE` (repeatable): `--arg ctx-size=32768`.
        #[arg(long = "arg")]
        args: Vec<String>,
        #[arg(long)]
        load_on_startup: bool,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Remove a model section.
    Remove {
        workload: String,
        id: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Record what a model needs (llmctl-only, not in the INI): declared
    /// VRAM per card, host RAM for CPU offload, Mac peak, VRAM floor; or
    /// the host's memory budget.
    Need {
        workload: String,
        model: Option<String>,
        /// MiB per card: `0=20000,1=20000`.
        #[arg(long)]
        mib: Option<String>,
        #[arg(long)]
        host_ram: Option<f64>,
        #[arg(long)]
        peak: Option<f64>,
        #[arg(long)]
        min_free: Option<u64>,
        #[arg(long)]
        budget: Option<u64>,
        /// Facts to drop: need, host_ram, peak, min_free, budget.
        #[arg(long, value_delimiter = ',')]
        clear: Vec<String>,
    },
    /// A llama.cpp lane on a llama-swap workload (the Macs, decision 8):
    /// `llama-server -m <gguf>` started by llama-swap, its binary named by
    /// a macro (the pinned release, `llamacpp install`). Plans a lane
    /// write; apply with `--confirm <digest>`.
    Lane {
        workload: String,
        lane: String,
        /// Absolute GGUF path (required unless --remove).
        #[arg(long)]
        model: Option<String>,
        #[arg(long = "macro", default_value = "llamacpp")]
        binary_macro: String,
        #[arg(long)]
        mmproj: Option<String>,
        /// Extra flag, `NAME=VALUE` or `NAME` (repeatable): `--arg embedding`.
        #[arg(long = "arg")]
        args: Vec<String>,
        #[arg(long)]
        ttl: Option<i64>,
        #[arg(long)]
        name: Option<String>,
        /// Remove the lane instead.
        #[arg(long)]
        remove: bool,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Stage a llama.cpp release archive into its own versioned dir for a
    /// llama-swap workload's lanes (checksum checked, binary version
    /// checked). Prints the `lanes macro` write that switches to it.
    Install {
        workload: String,
        /// Release tag (`b12527`).
        version: String,
        #[arg(long = "macro", default_value = "llamacpp")]
        binary_macro: String,
        /// Recipe JSON (a release-archive recipe); saved for the macro.
        #[arg(long)]
        recipe: Option<std::path::PathBuf>,
        /// `sha256:<hex>` of this version's archive (from the release page).
        #[arg(long)]
        checksum: Option<String>,
    },
    /// The router's log; with a model, only lines that name it.
    Logs {
        workload: String,
        model: Option<String>,
        #[arg(long, default_value_t = 200)]
        lines: u32,
    },
}

fn print_models(m: &RouterModelsResponse) {
    println!(
        "{}  preset {} (revision {}){}",
        m.workload,
        m.preset_path,
        m.revision,
        if m.drift { "  DRIFT: the file changed since the import" } else { "" }
    );
    println!(
        "  router {}; --models-max {}{}",
        if m.router_answers { "answers" } else { "DOES NOT ANSWER" },
        m.models_max,
        m.budget_gib.map(|b| format!("; memory budget {b}G")).unwrap_or_default()
    );
    for v in &m.models {
        let state = match (&v.state, v.failed) {
            (Some(s), true) => format!("{s} (last load failed, exit {})", v.exit_code.map(|c| c.to_string()).unwrap_or("?".into())),
            (Some(s), false) => s.to_string(),
            (None, _) => "-".into(),
        };
        let need = v
            .facts
            .need_mib
            .as_ref()
            .map(|n| {
                format!(
                    "{} ({})",
                    llmctl_core::fit::format_mib(n),
                    v.facts.need_source.clone().unwrap_or_default().split(' ').next().unwrap_or("")
                )
            })
            .unwrap_or_else(|| "need unknown".into());
        println!("  {:<24} {:<12} {}", v.record.id, state, need);
        if let Some(p) = &v.record.path {
            println!("      {p}");
        }
    }
    if !m.unknown_listed.is_empty() {
        println!("  listed by the router, not in the records: {}", m.unknown_listed.join(", "));
    }
    for h in &m.hooks {
        println!("  hook: {h}");
    }
    for v in &m.validation {
        println!("  INVALID: {v}");
    }
}

fn parse_mib(text: &str) -> Result<std::collections::BTreeMap<u32, u64>, String> {
    text.split(',')
        .map(|pair| {
            let (c, m) = pair
                .split_once('=')
                .ok_or_else(|| format!("--mib takes CARD=MIB pairs, not {pair:?}"))?;
            Ok((
                c.trim().parse().map_err(|_| format!("card {c:?} is not a number"))?,
                m.trim().parse().map_err(|_| format!("MiB {m:?} is not a number"))?,
            ))
        })
        .collect()
}

async fn edit(
    cli: &Cli,
    workload: String,
    edits: Vec<RouterEdit>,
    confirm: Option<String>,
    wait: bool,
) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let req = RouterEditRequest {
        protocol: PROTOCOL_VERSION,
        workload: workload.clone(),
        edits,
        confirm: None,
        idempotency_key: None,
    };
    let Some(confirm) = confirm else {
        let plan = client.llamacpp_plan(&req).await?;
        println!("{}: {}", plan.workload, plan.summary);
        for c in &plan.changes {
            println!("  {c}");
        }
        if !plan.reload_unloads.is_empty() {
            println!("  the reload unloads loaded models: {}", plan.reload_unloads.join(", "));
        }
        for v in &plan.validation {
            println!("  INVALID: {v}");
        }
        if !plan.validation.is_empty() {
            return Ok(1);
        }
        if !plan.bytes_changed {
            println!("  the file already says this; nothing to write");
            return Ok(0);
        }
        println!("apply: add --confirm {}", plan.plan_digest);
        return Ok(0);
    };
    let admitted = client
        .llamacpp_apply(&RouterEditRequest {
            confirm: Some(confirm),
            ..req
        })
        .await?;
    let op = admitted.operation;
    println!("admitted operation {} ({})", op.id, op.idempotency_key);
    if !wait {
        println!("progress: llmctl-next op {} --watch", op.id);
        return Ok(0);
    }
    watch_and_report(&client, &op.id, std::time::Duration::from_secs(5 * 60)).await
}

pub async fn cmd_llamacpp(cli: &Cli, cmd: LlamacppCmd) -> Result<u8, CliError> {
    match cmd {
        LlamacppCmd::Import {
            workload,
            preset,
            confirm,
        } => {
            let client = connect(cli)?;
            ensure_protocol(&client).await?;
            let r = client
                .llamacpp_import(&RouterImportRequest {
                    protocol: PROTOCOL_VERSION,
                    workload,
                    preset_path: preset,
                    confirm: confirm.clone(),
                })
                .await?;
            println!("{}: preset {} ({})", r.workload, r.preset_path, r.file_digest);
            println!("  round trip: {}", r.roundtrip);
            println!("  {} models, {} notes kept", r.models.len(), r.notes_kept);
            for m in &r.models {
                println!("    {:<24} {}", m.id, m.path.clone().or(m.hf_repo.clone()).unwrap_or_default());
            }
            for p in &r.argv_problems {
                println!("  COMMAND LINE: {p}");
            }
            if let Some(n) = &r.argv_note {
                println!("  note: {n}");
            }
            for v in &r.validation {
                println!("  INVALID: {v}");
            }
            if r.persisted {
                println!("records saved");
            } else if let Some(d) = &r.records_digest {
                if r.argv_problems.is_empty() && r.validation.is_empty() {
                    println!("save: add --confirm {d}");
                } else {
                    return Ok(1);
                }
            } else {
                return Ok(1);
            }
            Ok(0)
        }
        LlamacppCmd::Models { workload, json } => {
            let client = connect(cli)?;
            let m = client.llamacpp_models(&workload).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&m).unwrap_or_default());
            } else {
                print_models(&m);
            }
            Ok(0)
        }
        LlamacppCmd::Load {
            workload,
            model,
            dry_run,
            wait,
            evict,
            force,
            key,
        } => load(cli, workload, model, false, dry_run, wait, evict, force, key).await,
        LlamacppCmd::Unload {
            workload,
            model,
            dry_run,
            wait,
            force,
            key,
        } => load(cli, workload, model, true, dry_run, wait, false, force, key).await,
        LlamacppCmd::Set {
            workload,
            model,
            key,
            value,
            confirm,
            wait,
        } => edit(cli, workload, vec![RouterEdit::Set { model, key, value }], confirm, wait).await,
        LlamacppCmd::Unset {
            workload,
            model,
            key,
            confirm,
            wait,
        } => edit(cli, workload, vec![RouterEdit::Remove { model, key }], confirm, wait).await,
        LlamacppCmd::Add {
            workload,
            id,
            model,
            mmproj,
            args,
            load_on_startup,
            confirm,
            wait,
        } => {
            let mut parsed = Vec::new();
            for a in &args {
                let Some((k, v)) = a.split_once('=') else {
                    eprintln!("error: --arg takes KEY=VALUE, not {a:?}");
                    return Ok(2);
                };
                parsed.push((k.trim_start_matches('-').to_string(), v.to_string()));
            }
            let m = RouterModel {
                id,
                model: Some(model),
                hf_repo: None,
                mmproj,
                load_on_startup,
                args: parsed,
            };
            edit(cli, workload, vec![RouterEdit::AddModel { model: m }], confirm, wait).await
        }
        LlamacppCmd::Remove {
            workload,
            id,
            confirm,
            wait,
        } => edit(cli, workload, vec![RouterEdit::RemoveModel { id }], confirm, wait).await,
        LlamacppCmd::Need {
            workload,
            model,
            mib,
            host_ram,
            peak,
            min_free,
            budget,
            clear,
        } => {
            let need_mib = match mib.as_deref().map(parse_mib).transpose() {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("error: {e}");
                    return Ok(2);
                }
            };
            let client = connect(cli)?;
            ensure_protocol(&client).await?;
            let m = client
                .llamacpp_facts(&RouterFactsRequest {
                    protocol: PROTOCOL_VERSION,
                    workload,
                    model,
                    need_mib,
                    host_ram_gib: host_ram,
                    peak_gib: peak,
                    min_free_mib: min_free,
                    budget_gib: budget,
                    clear,
                })
                .await?;
            print_models(&m);
            Ok(0)
        }
        LlamacppCmd::Lane {
            workload,
            lane,
            model,
            binary_macro,
            mmproj,
            args,
            ttl,
            name,
            remove,
            confirm,
            wait,
        } => {
            let edit = if remove {
                llmctl_client::wire::LaneEdit::RemoveLane { lane }
            } else {
                let Some(model) = model else {
                    eprintln!("error: --model <gguf> is required (or --remove)");
                    return Ok(2);
                };
                let parsed = args
                    .iter()
                    .map(|a| match a.split_once('=') {
                        Some((k, v)) => (k.trim_start_matches('-').to_string(), v.to_string()),
                        None => (a.trim_start_matches('-').to_string(), "true".to_string()),
                    })
                    .collect();
                let l = llmctl_core::llamacpp::LlamaLane {
                    binary_macro,
                    model,
                    mmproj,
                    args: parsed,
                    ttl,
                    name,
                };
                let problems = l.validate();
                if !problems.is_empty() {
                    for p in problems {
                        eprintln!("error: {p}");
                    }
                    return Ok(2);
                }
                println!("cmd: {}", l.cmd());
                llmctl_client::wire::LaneEdit::AddLane {
                    lane,
                    fields: l.fields(),
                }
            };
            lane_write(cli, workload, edit, confirm, wait).await
        }
        LlamacppCmd::Install {
            workload,
            version,
            binary_macro,
            recipe,
            checksum,
        } => {
            let recipe = match recipe {
                Some(path) => match std::fs::read(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|b| serde_json::from_slice::<llmctl_core::ReleaseArchiveRecipe>(&b).map_err(|e| e.to_string()))
                {
                    Ok(r) => Some(r),
                    Err(e) => {
                        eprintln!("error: recipe {}: {e}", path.display());
                        return Ok(2);
                    }
                },
                None => None,
            };
            let client = connect(cli)?;
            ensure_protocol(&client).await?;
            let r = client
                .llamacpp_install(&llmctl_client::llamacpp::LaneEngineInstallRequest {
                    protocol: PROTOCOL_VERSION,
                    workload,
                    macro_name: binary_macro,
                    version,
                    recipe,
                    checksum,
                })
                .await?;
            println!("{} {}: staged {} into {}", r.workload, r.macro_name, r.version, r.dir);
            println!("  binary: {}", r.executable);
            if let Some(d) = &r.archive_sha256 {
                println!("  archive: {d}");
            }
            println!("  validation: {}", r.validation);
            println!("  the lanes run now: {}", r.current.clone().unwrap_or_else(|| "(macro not set)".into()));
            println!("switch (re-pins the lanes): {}", r.next);
            if let Some(b) = &r.rollback {
                println!("back, after switching: {b}");
            }
            Ok(0)
        }
        LlamacppCmd::Logs {
            workload,
            model,
            lines,
        } => {
            let client = connect(cli)?;
            let page = client
                .logs(&LogsRequest {
                    workload: workload.clone(),
                    lines: Some(lines),
                    after: None,
                })
                .await?;
            for e in &page.entries {
                if model.as_deref().is_none_or(|m| e.message.contains(m)) {
                    println!("{} {}", e.timestamp.format("%H:%M:%S"), e.message);
                }
            }
            Ok(0)
        }
    }
}

async fn lane_write(
    cli: &Cli,
    workload: String,
    edit: llmctl_client::wire::LaneEdit,
    confirm: Option<String>,
    wait: bool,
) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let req = llmctl_client::wire::LaneEditRequest {
        protocol: PROTOCOL_VERSION,
        workload,
        edits: vec![edit],
        confirm: None,
        idempotency_key: None,
    };
    let Some(confirm) = confirm else {
        let plan = client.lanes_plan(&req).await?;
        println!("{}: {}", plan.workload, plan.summary);
        for c in &plan.changes {
            println!("  {c}");
        }
        if !plan.reload_restarts.is_empty() {
            println!("  reload restarts running lanes: {}", plan.reload_restarts.join(", "));
        }
        for v in &plan.validation {
            println!("  INVALID: {v}");
        }
        if !plan.validation.is_empty() {
            return Ok(1);
        }
        println!("apply: add --confirm {}", plan.plan_digest);
        return Ok(0);
    };
    let admitted = client
        .lanes_apply(&llmctl_client::wire::LaneEditRequest {
            confirm: Some(confirm),
            ..req
        })
        .await?;
    let op = admitted.operation;
    println!("admitted operation {} ({})", op.id, op.idempotency_key);
    if !wait {
        println!("progress: llmctl-next op {} --watch", op.id);
        return Ok(0);
    }
    watch_and_report(&client, &op.id, std::time::Duration::from_secs(5 * 60)).await
}

#[allow(clippy::too_many_arguments)]
async fn load(
    cli: &Cli,
    workload: String,
    model: String,
    unload: bool,
    dry_run: bool,
    wait: bool,
    evict: bool,
    force: bool,
    key: Option<String>,
) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let r = client
        .llamacpp_load(&RouterLoadRequest {
            protocol: PROTOCOL_VERSION,
            workload,
            model,
            unload,
            dry_run,
            evict,
            force,
            idempotency_key: key,
        })
        .await?;
    let list = |v: &[String]| if v.is_empty() { "nothing".to_string() } else { v.join(", ") };
    println!("{}: {} {}", r.workload, if r.unload { "unload" } else { "load" }, r.model);
    println!("  holding memory now: {}", list(&r.loaded_now));
    println!("  decision: {} — {}", r.decision, r.reason);
    for p in &r.preload {
        println!("  PRE-LOAD CHECK: {p}");
    }
    if !r.evicts.is_empty() {
        println!("  stops first: {} (started again if the load fails)", r.evicts.join(", "));
    }
    for h in &r.hooks {
        println!("  hook: {h}");
    }
    match r.decision.as_str() {
        "refuse" => return Ok(1),
        "already_loaded" | "not_loaded" => return Ok(0),
        _ => {}
    }
    let Some(op) = r.operation else {
        return Ok(0);
    };
    println!("admitted operation {} ({})", op.id, op.idempotency_key);
    if !wait {
        println!("progress: llmctl-next op {} --watch", op.id);
        return Ok(0);
    }
    watch_and_report(&client, &op.id, std::time::Duration::from_secs(25 * 60)).await
}
