//! `alias` and `stores roots`: two config files of the Python llmctl,
//! read in place on the host where the command runs (M5 close-out gaps C4
//! and C5). The file code is `llmctl_daemon::local_config`.

use std::path::{Path, PathBuf};

use clap::Subcommand;
use llmctl_client::wire::{StatusResponse, StoreView};
use llmctl_daemon::local_config;
use serde_json::json;

use crate::{connect, Cli, CliError};

#[derive(Subcommand, Clone)]
pub enum AliasCmd {
    /// Each alias with its target, and whether a serving workload answers
    /// for the target.
    List {
        #[arg(long)]
        json: bool,
    },
    /// Bind ALIAS to TARGET (a workload name or a served model name) in
    /// `router_aliases.json`. `-` as TARGET removes the binding.
    /// settings.yaml is never written.
    Set { alias: String, target: String },
}

fn local_only(cli: &Cli, file: &str) -> Result<(), CliError> {
    match &cli.host {
        Some(host) => Err(CliError::Daemon(format!(
            "{file} is a file on the host where this command runs; --host {host} cannot reach it. Run the command on that host."
        ))),
        None => Ok(()),
    }
}

/// `healthy`: a workload that is running and ready has the target as its
/// name or as its served model. `bound`: no such workload now.
fn alias_status(target: Option<&str>, status: Option<&StatusResponse>) -> &'static str {
    let Some(target) = target else {
        return "unbound";
    };
    let Some(status) = status else {
        return "unknown";
    };
    let serving = status.workloads.iter().any(|w| {
        (w.name == target || w.served_model.as_deref() == Some(target))
            && w.process == "Running"
            && w.readiness == "Ready"
    });
    if serving {
        "healthy"
    } else {
        "bound"
    }
}

pub async fn cmd_alias(cli: &Cli, cmd: AliasCmd) -> Result<u8, CliError> {
    local_only(cli, "router_aliases.json")?;
    match cmd {
        AliasCmd::Set { alias, target } => {
            let target = (target != "-").then_some(target.as_str());
            local_config::set_alias(&alias, target).map_err(CliError::Daemon)?;
            match target {
                Some(t) => println!("bound {alias} -> {t}"),
                None => println!("cleared {alias}"),
            }
            Ok(0)
        }
        AliasCmd::List { json } => {
            let aliases = local_config::aliases().map_err(CliError::Daemon)?;
            let status = match connect(cli) {
                Ok(client) => client.status().await.ok(),
                Err(_) => None,
            };
            if json {
                let rows: Vec<_> = aliases
                    .iter()
                    .map(|(name, target)| {
                        json!({
                            "name": name,
                            "public_id": format!("local-{name}"),
                            "target": target,
                            "status": alias_status(target.as_deref(), status.as_ref()),
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&rows).unwrap());
                return Ok(0);
            }
            if aliases.is_empty() {
                println!("no alias in {}", local_config::alias_overlay_path().display());
            }
            for (name, target) in &aliases {
                println!(
                    "{:<18} {:<40} {}",
                    name,
                    target.as_deref().unwrap_or("-"),
                    alias_status(target.as_deref(), status.as_ref())
                );
            }
            if status.is_none() {
                eprintln!("note: the agent did not answer, so no target was checked");
            }
            Ok(0)
        }
    }
}

/// The declared stores that hold `root`, or that `root` holds.
fn stores_for(root: &Path, stores: &[StoreView]) -> Vec<String> {
    stores
        .iter()
        .filter(|s| root.starts_with(&s.path) || Path::new(&s.path).starts_with(root))
        .map(|s| s.name.clone())
        .collect()
}

/// `stores roots`: the roots of model_dirs.yaml against the declared
/// stores, and the broken symlinks under each root. Exit 1 when a broken
/// symlink exists.
pub async fn cmd_roots(cli: &Cli, json: bool) -> Result<u8, CliError> {
    local_only(cli, "model_dirs.yaml")?;
    let dirs = local_config::model_dirs().map_err(CliError::Daemon)?;
    let broken = local_config::check_roots(&dirs);
    // The check needs no agent; only the store column does.
    let stores = match connect(cli) {
        Ok(client) => client.stores().await.ok().map(|r| r.stores),
        Err(_) => None,
    };
    let rows: Vec<(&str, Option<PathBuf>, Option<Vec<String>>, Option<&Vec<PathBuf>>)> = dirs
        .roots
        .iter()
        .map(|root| {
            let covered =
                root.path.as_ref().zip(stores.as_ref()).map(|(p, stores)| stores_for(p, stores));
            (root.name.as_str(), root.path.clone(), covered, broken.get(&root.name))
        })
        .collect();
    let any_broken = broken.values().any(|links| !links.is_empty());
    if json {
        let doc = json!({
            "file": local_config::model_dirs_path(),
            "max_depth": dirs.max_depth,
            "roots": rows.iter().map(|(name, path, stores, links)| json!({
                "name": name, "path": path, "stores": stores, "broken_symlinks": links,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&doc).unwrap());
        return Ok(u8::from(any_broken));
    }
    if rows.is_empty() {
        println!("no root in {}", local_config::model_dirs_path().display());
    }
    for (name, path, stores, links) in &rows {
        let store = match stores {
            None => "?".to_string(),
            Some(names) if names.is_empty() => "none".to_string(),
            Some(names) => names.join(","),
        };
        let check = match links {
            // No path, no directory, or the directory of an earlier root.
            None => "not swept".to_string(),
            Some(links) => format!("{} broken symlink(s)", links.len()),
        };
        println!(
            "{:<16} {:<44} store {:<24} {}",
            name,
            path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "-".into()),
            store,
            check
        );
        for link in links.iter().flat_map(|l| l.iter()) {
            println!("  broken: {}", link.display());
        }
    }
    Ok(u8::from(any_broken))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_alias_is_healthy_only_when_a_ready_workload_answers_for_it() {
        let status: StatusResponse = serde_json::from_value(json!({
            "host": "h", "platform": "Linux", "inventory_revision": 1,
            "generated_at": "2026-11-27T00:00:00Z",
            "workloads": [{
                "name": "vllm-engine", "engine": "systemd", "model": "m", "gpus": [], "port": 1,
                "unit": "user/vllm-engine.service", "intent": "unmanaged", "process": "Running",
                "readiness": "Ready", "served_model": "qwen3.8-27b",
                "served_model_verified": true, "observed_at": null, "observation_unknown": false,
            }],
        }))
        .unwrap();
        assert_eq!(alias_status(Some("qwen3.8-27b"), Some(&status)), "healthy");
        assert_eq!(alias_status(Some("vllm-engine"), Some(&status)), "healthy");
        assert_eq!(alias_status(Some("flash"), Some(&status)), "bound");
        assert_eq!(alias_status(None, Some(&status)), "unbound");
        assert_eq!(alias_status(Some("flash"), None), "unknown");
    }

    #[test]
    fn a_root_names_the_stores_above_and_below_it() {
        let store = |name: &str, path: &str| -> StoreView {
            serde_json::from_value(json!({
                "name": name, "kind": "directory", "path": path, "reserve_gib": 1,
                "workload": null, "exists": true, "free_bytes": null,
            }))
            .unwrap()
        };
        let stores = [store("inside", "/models/hf2local"), store("other", "/mnt/storage")];
        assert_eq!(stores_for(Path::new("/models"), &stores), vec!["inside"]);
        assert_eq!(stores_for(Path::new("/mnt/storage/models/x"), &stores), vec!["other"]);
        assert!(stores_for(Path::new("/modelsX"), &stores).is_empty());
    }
}
