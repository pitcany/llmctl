//! `lanes matrix <workload> …` (Rust M5c item 4): the llama-swap matrix by
//! verb, through the ordinary lane plan → `--confirm` → write operation,
//! with the memory budget as the guard on the write.

use clap::Subcommand;

use llmctl_client::wire::LaneEdit;
use llmctl_core::matrix_edit::MatrixEdit;

use super::{connect, Cli, CliError};

#[derive(Subcommand, Clone)]
pub enum MatrixOp {
    /// Add or re-point a var: a short id (1–8 letters/digits) for a lane.
    Var {
        var: String,
        lane: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Remove a var (and its eviction cost).
    RmVar {
        var: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Add or replace a set: which vars may run together (`a & b | c`).
    /// A set named `risky…` may go over the budget on purpose.
    Set {
        name: String,
        expr: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Remove a set.
    RmSet {
        name: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Set a var's eviction cost (`none` removes it).
    Cost {
        var: String,
        cost: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
}

/// The lane edit an op asks for, with its `--confirm` and `--wait`.
pub fn edit(op: MatrixOp) -> Result<(Vec<LaneEdit>, Option<String>, bool), CliError> {
    let (edit, confirm, wait) = match op {
        MatrixOp::Var { var, lane, confirm, wait } => (MatrixEdit::SetVar { var, lane }, confirm, wait),
        MatrixOp::RmVar { var, confirm, wait } => (MatrixEdit::RemoveVar { var }, confirm, wait),
        MatrixOp::Set { name, expr, confirm, wait } => (MatrixEdit::SetSet { name, expr }, confirm, wait),
        MatrixOp::RmSet { name, confirm, wait } => (MatrixEdit::RemoveSet { name }, confirm, wait),
        MatrixOp::Cost { var, cost, confirm, wait } => {
            let cost = match cost.as_str() {
                "none" | "-" => None,
                n => Some(n.parse::<i64>().map_err(|_| CliError::Daemon(format!("cost {n:?}: a whole number or `none`")))?),
            };
            (MatrixEdit::SetEvictCost { var, cost }, confirm, wait)
        }
    };
    Ok((vec![LaneEdit::Matrix { edit }], confirm, wait))
}

/// `lanes matrix <workload>` with no op: the matrix and the budget.
pub async fn show(cli: &Cli, workload: &str) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let status = client.lanes_status(workload).await?;
    match &status.matrix {
        None => println!("{workload}: no matrix (llama-swap keeps one lane at a time, or `groups:` decide)"),
        Some(m) => {
            println!("{workload} matrix");
            for (k, v) in &m.vars {
                println!("  var  {k:<8} {v}");
            }
            for (k, v) in &m.evict_costs {
                println!("  cost {k:<8} {}", serde_yaml_text(v));
            }
            for (k, v) in &m.sets {
                println!("  set  {k:<16} {v}");
            }
        }
    }
    let memory = client
        .lanes_memory(&llmctl_client::wire::LaneMemoryRequest {
            protocol: llmctl_client::wire::PROTOCOL_VERSION,
            workload: workload.to_string(),
            budget_gib: None,
            peaks: Default::default(),
            from_registry: None,
        })
        .await?;
    for line in &memory.lines {
        println!("  {line}");
    }
    Ok(if memory.over.is_empty() { 0 } else { 4 })
}

fn serde_yaml_text(v: &impl serde::Serialize) -> String {
    serde_json::to_value(v).map(|j| j.to_string()).unwrap_or_default()
}
