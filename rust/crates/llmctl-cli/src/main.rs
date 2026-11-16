//! `llmctl-next` — CLI for the Rust successor control plane (dev stage).
//! Clients never start engines directly: everything mutating goes through
//! the daemon's local API over the Unix socket; a missing daemon is a
//! distinct error, never a silent fallback.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use llmctl_client::wire::{
    codes, AdmitRequest, OpStatus, OperationView, PlanRequest, PROTOCOL_VERSION,
};
use llmctl_client::{ClientError, ControlClient};
use llmctl_core::{OpKind, Plan};

#[derive(Parser)]
#[command(
    name = "llmctl-next",
    version,
    about = "llmctl successor control plane (Rust, dev stage) — talks to the local daemon over a Unix socket"
)]
struct Cli {
    /// Control socket path (default: <data-dir>/control.sock)
    #[arg(long, global = true)]
    socket: Option<PathBuf>,
    /// Dev data directory (default: $LLMCTL_NEXT_DATA_DIR or ~/.local/state/llmctl-next-dev)
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Clone)]
enum Command {
    /// Show managed workloads with fresh observations and recorded intent.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Plan (and by default execute) running a workload. --dry-run only previews.
    Run {
        name: String,
        #[arg(long)]
        dry_run: bool,
        /// Wait for the operation to reach a terminal state.
        #[arg(long)]
        wait: bool,
        /// Override the idempotency key (default run-<name>-r<revision>).
        #[arg(long)]
        key: Option<String>,
    },
    /// Stop a workload (records a manual-stop intent that recovery respects).
    Stop {
        name: String,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        key: Option<String>,
    },
    /// List operations from the journal.
    Ops {
        #[arg(long)]
        json: bool,
    },
    /// Show one operation; --watch polls until it is terminal.
    Op {
        id: String,
        #[arg(long)]
        watch: bool,
    },
    /// Request cancellation (succeeds only when the executor records it).
    Cancel { id: String },
    /// Acknowledge a needs_attention/interrupted operation; unblocks mutations.
    Ack { id: String },
    /// Run the daemon in the foreground (binds the control socket).
    Daemon {
        #[arg(long)]
        socket: Option<PathBuf>,
    },
}

fn exit_code_for_client_error(e: &ClientError) -> u8 {
    match e {
        ClientError::Unreachable { .. } => 3,
        ClientError::Api { code, .. } => match code.as_str() {
            codes::UNKNOWN_WORKLOAD | codes::UNKNOWN_OPERATION => 5,
            codes::INVALID_REQUEST | codes::PROTOCOL_MISMATCH => 6,
            codes::STALE_PLAN
            | codes::OVERLAPPING_OPERATION
            | codes::IDEMPOTENCY_CONFLICT
            | codes::BLOCKED_NEEDS_ATTENTION
            | codes::INVALID_STATE => 4,
            _ => 1,
        },
        ClientError::Disconnected(_) | ClientError::BadResponse(_) => 1,
    }
}

fn socket_path(cli: &Cli) -> PathBuf {
    if let Some(s) = &cli.socket {
        return s.clone();
    }
    let dir = match &cli.data_dir {
        Some(d) => d.clone(),
        None => llmctl_daemon::app::default_data_dir(),
    };
    dir.join("control.sock")
}

fn connect(cli: &Cli) -> Result<ControlClient, ClientError> {
    ControlClient::connect(socket_path(cli))
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let command = cli.command.clone();
    let result = dispatch(&cli, command).await;
    match result {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("error: {e}");
            let code = match &e {
                CliError::Client(c) => exit_code_for_client_error(c),
                CliError::Daemon(_) => 1,
            };
            if let CliError::Client(ClientError::Unreachable { path, .. }) = &e {
                eprintln!(
                    "no daemon at {} — start one with `llmctl-next daemon`",
                    path.display()
                );
            }
            ExitCode::from(code)
        }
    }
}

enum CliError {
    Client(ClientError),
    Daemon(String),
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::Client(c) => write!(f, "{c}"),
            CliError::Daemon(s) => write!(f, "{s}"),
        }
    }
}

impl From<ClientError> for CliError {
    fn from(c: ClientError) -> Self {
        CliError::Client(c)
    }
}

async fn dispatch(cli: &Cli, command: Command) -> Result<u8, CliError> {
    match command {
        Command::Status { json } => cmd_status(cli, json).await,
        Command::Run {
            name,
            dry_run,
            wait,
            key,
        } => cmd_mutate(cli, OpKind::Run, &name, dry_run, wait, key).await,
        Command::Stop { name, wait, key } => {
            cmd_mutate(cli, OpKind::Stop, &name, false, wait, key).await
        }
        Command::Ops { json } => cmd_ops(cli, json).await,
        Command::Op { id, watch } => cmd_op(cli, &id, watch).await,
        Command::Cancel { id } => cmd_cancel(cli, &id).await,
        Command::Ack { id } => cmd_ack(cli, &id).await,
        Command::Daemon { socket } => {
            llmctl_daemon::run_daemon(cli.data_dir.clone(), socket)
                .await
                .map_err(CliError::Daemon)?;
            Ok(0)
        }
    }
}

async fn cmd_status(cli: &Cli, json: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let status = client.status().await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&status).unwrap());
        return Ok(0);
    }
    println!(
        "host {} ({}) · inventory revision {} · {}",
        status.host,
        status.platform,
        status.inventory_revision,
        if status.workloads.len() == 1 {
            "1 workload".to_string()
        } else {
            format!("{} workloads", status.workloads.len())
        }
    );
    for w in &status.workloads {
        let serves = match (&w.served_model, w.served_model_verified) {
            (Some(m), true) => format!("{m} ✓"),
            (Some(m), false) => format!("{m} ~"),
            (None, _) => "-".to_string(),
        };
        let observed = match w.observed_at {
            Some(t) => t.format("%H:%M:%S").to_string(),
            None => "never".to_string(),
        };
        println!(
            "{:<10} {:<14} gpus={:<28} {:<8} {:<8} serves={:<18} intent={:<18} seen {}",
            w.name,
            w.model,
            w.gpus.join(","),
            w.process,
            w.readiness,
            serves,
            w.intent,
            observed
        );
    }
    Ok(0)
}

async fn cmd_mutate(
    cli: &Cli,
    kind: OpKind,
    name: &str,
    dry_run: bool,
    wait: bool,
    key: Option<String>,
) -> Result<u8, CliError> {
    let client = connect(cli)?;
    // Build the plan against fresh observations; the printed plan is what a
    // confirmation would refer to.
    let plan_resp = client
        .plan(PlanRequest {
            kind,
            workload: name.to_string(),
        })
        .await?;
    if dry_run {
        print_plan(&plan_resp.plan, &plan_resp.stop_targets);
        return Ok(0);
    }

    let idempotency_key = key.unwrap_or_else(|| {
        format!(
            "{}-{}-r{}",
            match kind {
                OpKind::Run => "run",
                OpKind::Stop => "stop",
                OpKind::Restart => "restart",
            },
            name,
            plan_resp.revision
        )
    });
    let admit = client
        .admit(AdmitRequest {
            protocol: PROTOCOL_VERSION,
            idempotency_key,
            kind,
            workload: name.to_string(),
            expected_revision: Some(plan_resp.revision),
            expected_stop_targets: Some(plan_resp.stop_targets.clone()),
        })
        .await?;
    let op = admit.operation;
    if admit.replayed {
        println!(
            "replaying existing operation {} (same idempotency key)",
            op.id
        );
    } else {
        println!("admitted operation {} ({})", op.id, op.idempotency_key);
    }
    if !wait {
        println!("progress: llmctl-next op {} --watch", op.id);
        return Ok(0);
    }
    watch_and_report(&client, &op.id).await
}

async fn cmd_ops(cli: &Cli, json: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let ops = client.list_operations().await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&ops).unwrap());
        return Ok(0);
    }
    if ops.operations.is_empty() {
        println!("no operations");
        return Ok(0);
    }
    for op in &ops.operations {
        println!(
            "{}  {:<7} {:<6} {:<16} {} phases={}",
            op.created_at.format("%Y-%m-%d %H:%M:%S"),
            format!("{:?}", op.kind),
            op.workload,
            format!("{:?}", op.status),
            if op.acknowledged {
                ""
            } else {
                " [ack pending]"
            },
            op.phases.len()
        );
    }
    Ok(0)
}

async fn cmd_op(cli: &Cli, id: &str, watch: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    if watch {
        return watch_and_report(&client, id).await;
    }
    let op = client.operation(id).await?;
    print_operation(&op);
    Ok(exit_code_for_op(&op))
}

async fn cmd_cancel(cli: &Cli, id: &str) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let op = client.cancel(id, None).await?;
    println!(
        "cancellation requested for {} (status: {:?}) — successful only once the executor records it",
        op.id, op.status
    );
    Ok(0)
}

async fn cmd_ack(cli: &Cli, id: &str) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let op = client.acknowledge(id, None).await?;
    println!("acknowledged {} ({:?})", op.id, op.status);
    Ok(0)
}

async fn watch_and_report(client: &ControlClient, id: &str) -> Result<u8, CliError> {
    let mut last_signature = String::new();
    let final_op: Option<OperationView>;
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10 * 60);
    loop {
        let op = client.operation(id).await?;
        let signature = format!(
            "{:?}|{}",
            op.status,
            op.phases
                .iter()
                .map(|p| format!("{:?}{}", p.status, p.detail.clone().unwrap_or_default()))
                .collect::<Vec<_>>()
                .join("|")
        );
        if signature != last_signature {
            println!("== {:?} ==", op.status);
            for p in &op.phases {
                println!(
                    "  {:>2}. {:<40} {:?}{}",
                    p.index,
                    p.label,
                    p.status,
                    p.detail
                        .as_ref()
                        .map(|d| format!(" — {d}"))
                        .unwrap_or_default()
                );
            }
            last_signature = signature;
        }
        if op.status.is_terminal() {
            final_op = Some(op);
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            eprintln!("watch timeout; operation still {:?}", op.status);
            return Ok(1);
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    let op = final_op.expect("terminal op");
    if let Some(recovery) = &op.recovery {
        println!("recovery: {recovery}");
    }
    if let Some(result) = &op.result {
        println!("result: {result}");
    }
    if matches!(op.status, OpStatus::NeedsAttention | OpStatus::Interrupted) && !op.acknowledged {
        eprintln!(
            "\nthis operation blocks new mutations until acknowledged: llmctl-next op ack {}",
            op.id
        );
    }
    Ok(exit_code_for_op(&op))
}

fn exit_code_for_op(op: &OperationView) -> u8 {
    match op.status {
        OpStatus::Succeeded => 0,
        _ => 1,
    }
}

fn print_plan(plan: &Plan, stop_targets: &[String]) {
    println!(
        "plan: {} {} @ inventory revision {}",
        plan.kind, plan.target, plan.revision
    );
    println!(
        "expected model: {}  gpus: {}",
        plan.expected_model,
        plan.affected_gpus
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    if stop_targets.is_empty() {
        println!("stops nothing");
    } else {
        println!("STOP TARGETS: {}", stop_targets.join(", "));
    }
    println!("steps:");
    for (i, s) in plan.steps.iter().enumerate() {
        println!(
            "  {:>2}. {}{}",
            i + 1,
            s.label(),
            if s.is_external() { "" } else { "  (internal)" }
        );
    }
    if !plan.rollback.is_empty() {
        println!("rollback (on failure):");
        for (i, s) in plan.rollback.iter().enumerate() {
            println!("  {:>2}. {}", i + 1, s.label());
        }
    }
}

fn print_operation(op: &OperationView) {
    println!("operation {} ({})", op.id, op.idempotency_key);
    println!(
        "{:?} {} — {:?}{}",
        op.kind,
        op.workload,
        op.status,
        if op.acknowledged {
            ""
        } else {
            " (ack pending)"
        }
    );
    for p in &op.phases {
        println!(
            "  {:>2}. {:<40} {:?}{}",
            p.index,
            p.label,
            p.status,
            p.detail
                .as_ref()
                .map(|d| format!(" — {d}"))
                .unwrap_or_default()
        );
    }
    if let Some(r) = &op.result {
        println!("result: {r}");
    }
    if let Some(r) = &op.recovery {
        println!("recovery: {r}");
    }
}
