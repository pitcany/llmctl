//! `model list/show/add/edit/enable/disable/delete/scan` and `catalog
//! import/export` (Rust M5c item 3). Per host: `--host` routes to that
//! host's agent and its catalog.

use clap::Subcommand;

use llmctl_client::catalog::{
    CatalogImportRequest, CatalogModel, ModelAddRequest, ModelDeleteRequest, ModelEditRequest,
    ModelEnableRequest, ModelWriteResponse, ScanRequest,
};
use llmctl_client::wire::PROTOCOL_VERSION;

use super::{connect, ensure_protocol, Cli, CliError};

/// The catalog verbs that live under `model` (beside `get` and `rm`).
#[derive(Subcommand, Clone)]
pub enum ModelCatalogCmd {
    /// The models this host's catalog knows (enabled ones; --all for every).
    List {
        #[arg(long)]
        all: bool,
        #[arg(long)]
        json: bool,
    },
    /// One record (by id, or a name unique on the host).
    Show {
        model: String,
        #[arg(long)]
        json: bool,
    },
    /// Add a record. --source is what the engine calls it (tag, key, repo);
    /// --path the weights; the store is found from the path or engine.
    Add {
        name: String,
        /// vllm, llama_cpp, lmstudio, ollama, python_script, sglang, freetoken, strata.
        #[arg(long)]
        engine: String,
        #[arg(long)]
        source: Option<String>,
        #[arg(long)]
        path: Option<String>,
        #[arg(long)]
        store: Option<String>,
        /// Other fields: quantization, max_context, estimated_vram_gb,
        /// tags, notes, … (YAML values).
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Change fields of a record.
    Edit {
        model: String,
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        #[arg(long = "unset", value_name = "KEY")]
        unset: Vec<String>,
        #[arg(long)]
        json: bool,
    },
    /// Enable a record.
    Enable { model: String },
    /// Disable a record (kept, hidden from `model list`, exported inactive).
    Disable { model: String },
    /// Delete a record from this host's catalog. Without --confirm it shows
    /// the record, what still names it and the token. Refused while a
    /// workload or a preset names it. The weights stay (`model rm`).
    Delete {
        model: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Walk the declared stores: weights the catalog does not name, and
    /// records whose weights are gone. --add records what is new.
    Scan {
        #[arg(long)]
        store: Option<String>,
        #[arg(long)]
        add: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Clone)]
pub enum CatalogCmd {
    /// Import a Python registry bundle (`llmctl export-registry FILE`).
    /// Without --confirm only reports what each row becomes.
    Import {
        bundle: std::path::PathBuf,
        /// A registry host name that means this host (repeatable), e.g.
        /// `--as yannik-desktop`. Rows of other hosts stay unmapped.
        #[arg(long = "as", value_name = "HOSTNAME")]
        host_names: Vec<String>,
        /// The token the report printed.
        #[arg(long)]
        confirm: Option<String>,
        /// Print every row (default: the counts and the rows that need a look).
        #[arg(long)]
        verbose: bool,
        #[arg(long)]
        json: bool,
    },
    /// Write the catalog as a registry bundle `llmctl import-registry` takes.
    Export { out: std::path::PathBuf },
}

/// `history import` / `history list` (M6 decision 9).
#[derive(Subcommand, Clone)]
pub enum HistoryCmd {
    /// Import the sessions and operations of a Python DB copy (take it with
    /// `sqlite3 <db> ".backup <copy>"`; a live DB is refused). The rows stay
    /// verbatim and read-only as `py:<host>:<id>`; none authorizes an action.
    /// Without --confirm only reports what each row becomes.
    Import {
        snapshot: std::path::PathBuf,
        /// A host name in the rows (`owner.hostname`) that means this host,
        /// e.g. `--as yannik-desktop`. Rows of other hosts are reported.
        #[arg(long = "as", value_name = "HOSTNAME")]
        host_names: Vec<String>,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// The imported rows: id, kind and a few fields (no env, no request body).
    List {
        #[arg(long)]
        json: bool,
    },
}

pub async fn cmd_history(cli: &Cli, cmd: HistoryCmd) -> Result<u8, CliError> {
    use llmctl_client::catalog::HistoryImportRequest;
    let client = connect(cli)?;
    match cmd {
        HistoryCmd::Import { snapshot, host_names, confirm, json } => {
            ensure_protocol(&client).await?;
            let (sha, rows) = llmctl_daemon::history::read_snapshot(&snapshot).map_err(CliError::Daemon)?;
            let applying = confirm.is_some();
            let resp = client
                .history_import(&HistoryImportRequest {
                    protocol: PROTOCOL_VERSION,
                    snapshot_sha256: sha.clone(),
                    host_names,
                    rows,
                    confirm,
                })
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            println!("snapshot {} (sha256 {})", snapshot.display(), &sha[..16]);
            for r in resp.rows.iter().filter(|r| r.outcome != "new" && r.outcome != "present") {
                println!("  {:<11} {:<9} {}  {}", r.outcome, r.kind, r.id, r.reason.as_deref().unwrap_or(""));
            }
            let counts: Vec<String> = resp.counts.iter().map(|(k, v)| format!("{v} {k}")).collect();
            println!("{} rows: {}", resp.rows.len(), counts.join(", "));
            if resp.applied {
                println!("imported; {} legacy rows in the journal", resp.total);
            } else if !applying {
                println!("dry run. Apply exactly this report with --confirm {}", resp.confirm);
            }
            Ok(0)
        }
        HistoryCmd::List { json } => {
            let resp = client.history_list().await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            if resp.entries.is_empty() {
                println!("no legacy history on {}", resp.host);
            }
            for e in &resp.entries {
                let s: Vec<String> = e
                    .summary
                    .iter()
                    .map(|(k, v)| format!("{k}={}", v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string())))
                    .collect();
                println!("{:<9} {}  {}", e.kind, e.id, s.join(" "));
            }
            Ok(0)
        }
    }
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

fn row(m: &CatalogModel) -> String {
    format!(
        "{:<36} {:<34} {:<10} {:<10} {}{}",
        m.id,
        m.name,
        m.engine,
        m.store.as_deref().unwrap_or("-"),
        m.source.as_deref().or(m.path.as_deref()).unwrap_or("-"),
        if m.enabled { "" } else { "  (disabled)" }
    )
}

fn print_write(resp: &ModelWriteResponse, json: bool) {
    if json {
        println!("{}", serde_json::to_string_pretty(resp).unwrap());
        return;
    }
    println!("{}", row(&resp.model));
    if resp.changes.is_empty() {
        println!("  nothing changed");
    }
    for (k, a, b) in &resp.changes {
        println!("  {k}: {} -> {}", a.as_deref().unwrap_or("(unset)"), b.as_deref().unwrap_or("(unset)"));
    }
}

pub async fn cmd_model_catalog(cli: &Cli, cmd: ModelCatalogCmd) -> Result<u8, CliError> {
    let client = connect(cli)?;
    match cmd {
        ModelCatalogCmd::List { all, json } => {
            let resp = client.catalog_models(all).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            println!("{:<36} {:<34} {:<10} {:<10} SOURCE", "ID", "NAME", "ENGINE", "STORE");
            for m in &resp.models {
                println!("{}", row(m));
            }
            let mut tail = vec![format!("{} models on {}", resp.models.len(), resp.host)];
            if resp.disabled_hidden > 0 {
                tail.push(format!("{} disabled hidden (--all)", resp.disabled_hidden));
            }
            if resp.unmapped > 0 {
                tail.push(format!("{} bundle rows kept for the export only", resp.unmapped));
            }
            println!("{}", tail.join("; "));
            Ok(0)
        }
        ModelCatalogCmd::Show { model, json } => {
            let m = client.catalog_model(&model).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&m).unwrap());
                return Ok(0);
            }
            println!("{}", row(&m));
            if let Some(p) = &m.path {
                println!("  path {p}");
            }
            for (k, v) in &m.extra {
                println!("  {k}: {v}");
            }
            println!("  origin {}", m.origin);
            Ok(0)
        }
        ModelCatalogCmd::Add { name, engine, source, path, store, set, json } => {
            ensure_protocol(&client).await?;
            let resp = client
                .model_add(&ModelAddRequest { protocol: PROTOCOL_VERSION, name, engine, source, path, store, set: pairs(&set)? })
                .await?;
            print_write(&resp, json);
            Ok(0)
        }
        ModelCatalogCmd::Edit { model, set, unset, json } => {
            ensure_protocol(&client).await?;
            let resp = client
                .model_edit(&ModelEditRequest { protocol: PROTOCOL_VERSION, model, set: pairs(&set)?, unset })
                .await?;
            print_write(&resp, json);
            Ok(0)
        }
        ModelCatalogCmd::Enable { model } => {
            ensure_protocol(&client).await?;
            let resp = client.model_enable(&ModelEnableRequest { protocol: PROTOCOL_VERSION, model, enabled: true }).await?;
            print_write(&resp, false);
            Ok(0)
        }
        ModelCatalogCmd::Disable { model } => {
            ensure_protocol(&client).await?;
            let resp = client.model_enable(&ModelEnableRequest { protocol: PROTOCOL_VERSION, model, enabled: false }).await?;
            print_write(&resp, false);
            Ok(0)
        }
        ModelCatalogCmd::Delete { model, confirm, json } => {
            ensure_protocol(&client).await?;
            let applying = confirm.is_some();
            let resp = client.model_delete(&ModelDeleteRequest { protocol: PROTOCOL_VERSION, model, confirm }).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            println!("{}", row(&resp.model));
            for w in &resp.warnings {
                println!("  note: {w}");
            }
            if resp.deleted {
                println!("  deleted (catalog revision {}); its weights are not touched", resp.revision);
            } else if !resp.references.is_empty() {
                println!("  still named by: {}; a delete is refused", resp.references.join(", "));
            } else if !applying {
                println!("  dry run. Delete exactly this with --confirm {}", resp.confirm);
            }
            Ok(0)
        }
        ModelCatalogCmd::Scan { store, add, json } => {
            ensure_protocol(&client).await?;
            let resp = client.model_scan(&ScanRequest { protocol: PROTOCOL_VERSION, store, add }).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            let new: Vec<_> = resp.found.iter().filter(|f| f.record.is_none() || add).collect();
            println!("scanned {}: {} weights found", resp.stores.join(", "), resp.found.len());
            for f in resp.found.iter().filter(|f| f.record.is_none()) {
                println!("  not cataloged  {:<10} {:<9} {}", f.store, f.engine, f.source);
            }
            for m in &resp.missing {
                println!("  gone           {:<10} {:<9} {} ({})", m.store.as_deref().unwrap_or("-"), m.engine, m.path.as_deref().unwrap_or("-"), m.name);
            }
            if add {
                println!("  added {} records", resp.added);
            } else if !new.is_empty() {
                println!("  --add records the ones not cataloged");
            }
            Ok(0)
        }
    }
}

pub async fn cmd_catalog(cli: &Cli, cmd: CatalogCmd) -> Result<u8, CliError> {
    let client = connect(cli)?;
    match cmd {
        CatalogCmd::Import { bundle, host_names, confirm, verbose, json } => {
            ensure_protocol(&client).await?;
            let text = std::fs::read_to_string(&bundle)
                .map_err(|e| CliError::Daemon(format!("{}: {e}", bundle.display())))?;
            let value: serde_json::Value = serde_json::from_str(&text)
                .map_err(|e| CliError::Daemon(format!("{}: not JSON: {e}", bundle.display())))?;
            let resp = client
                .catalog_import(&CatalogImportRequest { protocol: PROTOCOL_VERSION, bundle: value, host_names, confirm })
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
                return Ok(0);
            }
            for r in &resp.rows {
                let interesting = !matches!(r.outcome, llmctl_client::catalog::RowOutcome::New | llmctl_client::catalog::RowOutcome::Present);
                if verbose || interesting {
                    println!(
                        "  {:<9} {:<10} {:<34} {}{}",
                        serde_json::to_value(r.outcome).ok().and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default(),
                        r.runtime,
                        r.name,
                        r.store.as_deref().map(|s| format!("[{s}] ")).unwrap_or_default(),
                        r.reason.as_deref().unwrap_or("")
                    );
                }
            }
            let counts: Vec<String> = resp.counts.iter().map(|(k, v)| format!("{k} {v}")).collect();
            println!("{} rows: {}; {} new profiles", resp.rows.len(), counts.join(", "), resp.profiles);
            if resp.applied {
                println!("imported (catalog revision {})", resp.revision);
            } else {
                println!("dry run. Apply exactly this report with --confirm {}", resp.confirm);
            }
            Ok(0)
        }
        CatalogCmd::Export { out } => {
            let bundle = client.catalog_export().await?;
            let n = bundle["models"].as_array().map(Vec::len).unwrap_or(0);
            let p = bundle["profiles"].as_array().map(Vec::len).unwrap_or(0);
            std::fs::write(&out, serde_json::to_string_pretty(&bundle).unwrap() + "\n")
                .map_err(|e| CliError::Daemon(format!("{}: {e}", out.display())))?;
            println!("wrote {} ({n} models, {p} profiles); `llmctl import-registry {}` takes it", out.display(), out.display());
            Ok(0)
        }
    }
}
