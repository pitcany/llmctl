//! `preset …` verbs and `run <preset>` (Rust M5c item 1). The agent reads
//! and writes the preset files; this module only asks and prints.

use clap::Subcommand;

use llmctl_client::catalog::{
    PresetEditRequest, PresetNewRequest, PresetRunRequest, PresetSettleRequest, PresetWriteResponse,
};
use llmctl_client::wire::{ProfileRequest, PROTOCOL_VERSION};
use llmctl_core::OpKind;

use super::{cmd_mutate, connect, ensure_protocol, Cli, CliError};

#[derive(Subcommand, Clone)]
pub enum PresetCmd {
    /// Every preset with its workload, profile env and drift.
    List {
        #[arg(long)]
        json: bool,
    },
    /// One preset: its file, its link and the env it renders.
    Show {
        alias: String,
        #[arg(long)]
        json: bool,
    },
    /// What differs between the preset's render and its profile env (and
    /// whether that profile is the one serving).
    Diff {
        alias: String,
        #[arg(long)]
        json: bool,
    },
    /// Change keys in place; comments and untouched keys stay. Values are
    /// YAML (`--set max_num_seqs=16`, `--set dtype=null`).
    Edit {
        alias: String,
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        #[arg(long = "unset", value_name = "KEY")]
        unset: Vec<String>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        json: bool,
    },
    /// A new preset: a copy of `--from` (comments too) with `--set` keys,
    /// or bare `--set` keys (the seven required ones at least).
    New {
        alias: String,
        #[arg(long)]
        from: Option<String>,
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        json: bool,
    },
    /// Write the preset's render into its router profile env (values
    /// merged, comments kept, generated-from header on line 1). Runs
    /// nothing.
    Sync {
        alias: String,
        #[arg(long)]
        workload: Option<String>,
        #[arg(long)]
        dry_run: bool,
    },
    /// Make the comments that name the engine's version (env header,
    /// profiles, router.yaml) say what the workload's version pin says.
    /// Without --write only shows the lines.
    Settle {
        #[arg(long)]
        workload: Option<String>,
        #[arg(long)]
        write: bool,
        #[arg(long)]
        json: bool,
    },
}

fn pairs(set: &[String]) -> Result<Vec<(String, String)>, CliError> {
    set.iter()
        .map(|s| {
            s.split_once('=')
                .map(|(k, v)| (k.trim().to_string(), v.to_string()))
                .ok_or_else(|| CliError::Daemon(format!("--set {s:?}: expected KEY=VALUE")))
        })
        .collect()
}

fn print_write(resp: &PresetWriteResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(resp).unwrap());
        return;
    }
    let verb = if resp.written {
        "wrote"
    } else if resp.before_digest == resp.after_digest {
        "unchanged"
    } else {
        "would write"
    };
    println!("{verb} {} ({})", resp.path, resp.alias);
    for (k, a, b) in &resp.changes {
        let show = |v: &Option<String>| v.clone().unwrap_or_else(|| "(unset)".into());
        println!("  {k}: {} -> {}", show(a), show(b));
    }
    if !resp.comments_kept {
        println!("  note: the line edit could not keep the comments; the file is re-dumped");
    }
    if let Some(b) = &resp.backup {
        println!("  previous bytes kept at {b}");
    }
}

pub async fn cmd_preset(cli: &Cli, cmd: PresetCmd) -> Result<u8, CliError> {
    let client = connect(cli)?;
    match cmd {
        PresetCmd::List { json } => {
            let resp = client.presets().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            println!("{:<22} {:<22} {:>3} {:>7}  {:<10} PROFILE", "ALIAS", "SERVED", "TP", "CTX", "STATE");
            for p in &resp.presets {
                let (state, profile) = match (&p.link, &p.link_error) {
                    (Some(l), _) => {
                        let state = if l.active {
                            "serving"
                        } else if l.profile_env.is_none() {
                            "no-profile"
                        } else if l.drift.is_empty() {
                            "ready"
                        } else {
                            "drift"
                        };
                        (state.to_string(), l.profile_env.clone().unwrap_or_else(|| "-".into()))
                    }
                    (None, Some(e)) => ("unlinked".to_string(), e.clone()),
                    (None, None) => ("-".to_string(), "-".to_string()),
                };
                println!(
                    "{:<22} {:<22} {:>3} {:>7}  {:<10} {}",
                    p.alias, p.served_name, p.tensor_parallel_size, p.max_model_len, state, profile
                );
            }
            for bad in &resp.invalid {
                println!("invalid {}: {}", bad.path, bad.error);
            }
            Ok(0)
        }
        PresetCmd::Show { alias, json } => {
            let resp = client.preset(&alias).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            let s = &resp.summary;
            println!("{}  ({})", s.alias, s.source_path);
            println!("  served {}  model {}  tp {}  ctx {}", s.served_name, s.model_id, s.tensor_parallel_size, s.max_model_len);
            if let Some(l) = &s.link {
                println!(
                    "  workload {}  profile {} ({})  {}",
                    l.workload,
                    l.profile_env.as_deref().unwrap_or("-"),
                    l.found_by,
                    if l.active { "serving" } else if l.drift.is_empty() { "in sync" } else { "drift" }
                );
            }
            if let Some(e) = &resp.render_error {
                println!("  cannot render: {e}");
            }
            if let Some(env) = &resp.rendered_env {
                println!("--- renders");
                print!("{env}");
            }
            Ok(0)
        }
        PresetCmd::Diff { alias, json } => {
            let resp = client.preset(&alias).await?;
            let Some(l) = resp.summary.link.clone() else {
                return Err(CliError::Daemon(
                    resp.summary.link_error.unwrap_or_else(|| "no workload for this preset".into()),
                ));
            };
            if json {
                println!("{}", serde_json::to_string_pretty(&l).unwrap());
                return Ok(if l.drift.is_empty() { 0 } else { 3 });
            }
            println!(
                "{} -> {} ({})",
                alias,
                l.profile_env.as_deref().unwrap_or("(no profile yet)"),
                l.found_by
            );
            if l.drift.is_empty() {
                println!("  in sync{}", if l.active { "; serving now" } else { "" });
                return Ok(0);
            }
            for d in &l.drift {
                println!(
                    "  {}: file {} | preset {}",
                    d.key,
                    d.file.as_deref().unwrap_or("(absent)"),
                    d.render.as_deref().unwrap_or("(absent)")
                );
            }
            println!("  `llmctl-next preset sync {alias}` writes the preset's values (comments kept)");
            // Exit 3 like `diff`: differences found.
            Ok(3)
        }
        PresetCmd::Edit { alias, set, unset, dry_run, json } => {
            ensure_protocol(&client).await?;
            let resp = client
                .preset_edit(&PresetEditRequest {
                    protocol: PROTOCOL_VERSION,
                    alias,
                    set: pairs(&set)?,
                    unset,
                    dry_run,
                    expect_digest: None,
                })
                .await?;
            print_write(&resp, json);
            Ok(0)
        }
        PresetCmd::New { alias, from, set, dry_run, json } => {
            ensure_protocol(&client).await?;
            let resp = client
                .preset_new(&PresetNewRequest { protocol: PROTOCOL_VERSION, alias, from, set: pairs(&set)?, dry_run })
                .await?;
            print_write(&resp, json);
            Ok(0)
        }
        PresetCmd::Sync { alias, workload, dry_run } => {
            ensure_protocol(&client).await?;
            let resp = client
                .preset_run(&PresetRunRequest { protocol: PROTOCOL_VERSION, alias, workload, sync: true, dry_run })
                .await?;
            let path = resp.profile_env.clone().unwrap_or_else(|| "(the active env already renders it)".into());
            if !resp.synced {
                println!("{path}: in sync with {}", resp.alias);
            } else {
                println!("{} {path}", if dry_run { "would write" } else { "wrote" });
                for d in &resp.drift {
                    println!("  {}: {} -> {}", d.key, d.file.as_deref().unwrap_or("(absent)"), d.render.as_deref().unwrap_or("(absent)"));
                }
            }
            Ok(0)
        }
        PresetCmd::Settle { workload, write, json } => {
            ensure_protocol(&client).await?;
            let resp = client
                .preset_settle(&PresetSettleRequest { protocol: PROTOCOL_VERSION, workload, write })
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            println!(
                "{}: pin says {} (vLLM {})",
                resp.workload,
                resp.pin.env_name,
                resp.pin.version.as_deref().unwrap_or("?")
            );
            if resp.files.is_empty() {
                println!("  every comment already says so ({} files read)", resp.untouched.len());
            }
            for f in &resp.files {
                println!("  {} {}", if f.written { "wrote" } else { "would change" }, f.path);
                for c in &f.changes {
                    println!("    {}: {}", c.line, c.before.trim());
                    println!("    {}  {}", " ".repeat(c.line.to_string().len()), c.after.trim());
                }
            }
            if !write && !resp.files.is_empty() {
                println!("  --write applies these (previous bytes kept as <file>.llmctl-previous)");
            }
            Ok(0)
        }
    }
}

/// Is `name` a workload in the agent's inventory?
async fn is_workload(cli: &Cli, name: &str) -> Result<bool, CliError> {
    let client = connect(cli)?;
    let inv = client.inventory().await?;
    let found = inv
        .get("workloads")
        .or_else(|| inv.get("inventory").and_then(|i| i.get("workloads")))
        .and_then(|w| w.as_array())
        .is_some_and(|ws| ws.iter().any(|w| w.get("name").and_then(|n| n.as_str()) == Some(name)));
    Ok(found)
}

/// `run <name>`: a workload runs as before; otherwise (or with --preset)
/// the name is a preset: resolve its profile and run the workload on it.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    cli: &Cli,
    name: &str,
    preset: bool,
    workload: Option<String>,
    sync: bool,
    dry_run: bool,
    wait: bool,
    key: Option<String>,
    force: bool,
) -> Result<Option<u8>, CliError> {
    if !preset && is_workload(cli, name).await? {
        return Ok(None);
    }
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let resp = client
        .preset_run(&PresetRunRequest {
            protocol: PROTOCOL_VERSION,
            alias: name.to_string(),
            workload,
            sync,
            dry_run,
        })
        .await?;
    if resp.synced {
        println!(
            "{} {} from preset {name}",
            if dry_run { "would sync" } else { "synced" },
            resp.profile_env.as_deref().unwrap_or("?")
        );
        if dry_run {
            for d in &resp.drift {
                println!("  {}: {} -> {}", d.key, d.file.as_deref().unwrap_or("(absent)"), d.render.as_deref().unwrap_or("(absent)"));
            }
            return Ok(Some(0));
        }
    }
    let profile = resp.profile_env.clone().map(|env_source| ProfileRequest {
        env_source,
        model: resp.model.clone(),
    });
    match &profile {
        Some(p) => println!("preset {name}: {} on {} (model {})", p.env_source, resp.workload, resp.model),
        None => println!("preset {name}: {} already renders it (model {})", resp.workload, resp.model),
    }
    let code = cmd_mutate(cli, OpKind::Run, &resp.workload, dry_run, wait, key, force, profile).await?;
    Ok(Some(code))
}
