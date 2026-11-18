//! `llmctl-next` — CLI for the Rust successor control plane (dev stage).
//! Clients never start engines directly: everything mutating goes through
//! the daemon's local API over the Unix socket; a missing daemon is a
//! distinct error, never a silent fallback.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use llmctl_client::wire::{
    codes, AdmitRequest, AdoptRequest, AdoptionPreview, CheckStatus, LogEntry, LogSource,
    LogsRequest, OpStatus, OperationView, PlanRequest, ProfileRequest, DEFAULT_LOG_LINES,
    MAX_LOG_LINES, PROTOCOL_VERSION,
};
use llmctl_client::{ClientError, ControlClient};
use llmctl_core::{OpKind, Plan};

mod enroll;
mod llamacpp_cmd;

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
    /// Route every verb to this enrolled host's agent over the SSH bridge
    /// (M3-E.2). `host add` records enrollments.
    #[arg(long, global = true)]
    host: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Clone)]
enum Command {
    /// Show managed workloads with fresh observations and recorded intent.
    Status {
        #[arg(long)]
        json: bool,
        /// Every enrolled host too: local first, then each remote with
        /// freshness (`seen 12 s ago` / `unreachable since ...` —
        /// observations shown stale, never "stopped").
        #[arg(long)]
        all: bool,
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
        /// Do not wait for requests llm-router has in flight on the
        /// workloads this stops; cut them.
        #[arg(long)]
        force: bool,
        /// Switch the workload to this env-file profile first: the file
        /// (next to the unit's EnvironmentFile) becomes that env file, and
        /// the old one comes back if the start fails. Needs --model.
        #[arg(long, value_name = "PATH", requires = "model")]
        profile_env: Option<String>,
        /// The model the workload serves under --profile-env.
        #[arg(long, value_name = "NAME", requires = "profile_env")]
        model: Option<String>,
    },
    /// Stop a workload (records a manual-stop intent that recovery respects).
    Stop {
        name: String,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        key: Option<String>,
        /// Do not wait for requests llm-router has in flight on the
        /// workloads this stops; cut them.
        #[arg(long)]
        force: bool,
    },
    /// Unload one model from a running vendor service (Ollama, LM Studio)
    /// without stopping it; confirmed through the engine's loaded list.
    Unload {
        name: String,
        model: String,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        key: Option<String>,
    },
    /// What a workload's engine reports: readiness, servable and loaded
    /// models.
    Engine {
        name: String,
        #[arg(long)]
        json: bool,
    },
    /// Restart a running workload: stop, start, await readiness and
    /// re-verify the served model. Refused when it is not running.
    Restart {
        name: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        wait: bool,
        /// Override the idempotency key. The default is unique per
        /// invocation (each restart is a new request); pass the key printed
        /// by an earlier attempt to retry that attempt instead.
        #[arg(long)]
        key: Option<String>,
        /// Do not wait for requests llm-router has in flight on the
        /// workloads this stops; cut them.
        #[arg(long)]
        force: bool,
    },
    /// Recorded intent per workload, without probing anything. What a
    /// healer asks before it restarts or loads something: a `manual_stop`
    /// means the owner stopped it here, so leave it alone.
    Intent {
        /// Workload name, unit name or launchd label; omit for all.
        name: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Foreign-write findings (M5a): a managed workload's state changed
    /// and no admitted operation explains it. Read-only — the pilot day's
    /// "no unexplained foreign writes" evidence.
    Audit {
        /// Only findings at or after this instant: RFC 3339
        /// (`2026-11-18T00:00:00Z`) or a bare UTC date (`2026-11-18`).
        #[arg(long)]
        since: Option<String>,
        /// Ask every enrolled host too, local first.
        #[arg(long)]
        all: bool,
        #[arg(long)]
        json: bool,
    },
    /// Declared model stores (where weights live) with their free space.
    /// `stores add` / `stores remove` declare or forget one; neither
    /// creates or deletes anything on disk.
    Stores {
        #[command(subcommand)]
        cmd: Option<StoresCmd>,
        #[arg(long)]
        json: bool,
    },
    /// Model weights.
    Model {
        #[command(subcommand)]
        cmd: ModelCmd,
    },
    /// Measured VRAM per workload and the cards' free memory: what the
    /// fit decision ("start beside what runs, or make room") rests on.
    /// --measure records what a serving workload holds right now.
    Vram {
        #[arg(long, value_name = "NAME")]
        measure: Option<String>,
        #[arg(long)]
        json: bool,
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
    /// Adopt an existing systemd user unit. Without --confirm this only
    /// previews: imported definition, generated service definition, checks
    /// and the digest to confirm. With --confirm DIGEST it transfers
    /// ownership (the unit keeps running; its file is not touched).
    Adopt {
        /// Bare user unit name, e.g. qwen-coder.service
        unit: String,
        /// Workload name to enroll it as.
        #[arg(long)]
        name: String,
        /// Model id the unit's endpoint must serve.
        #[arg(long)]
        model: String,
        /// Port of its OpenAI-compatible endpoint.
        #[arg(long)]
        port: u16,
        /// GPU indices it uses, comma-separated.
        #[arg(long, value_delimiter = ',', required = true)]
        gpus: Vec<u32>,
        /// Digest from the preview: transfer ownership.
        #[arg(long)]
        confirm: Option<String>,
        /// Transfer although served identity could not be checked.
        #[arg(long)]
        allow_unverified: bool,
        /// What the unit runs (ollama, lm_studio, llama_cpp, vllm, mtplx,
        /// ...): picks the readiness protocol. A vendor service may pass
        /// --model ollama / --model lmstudio for the service itself.
        #[arg(long)]
        engine_kind: Option<String>,
        /// When it holds its GPUs: at_start (evicts and is evicted) or
        /// on_demand (a vendor service that loads per request; the default
        /// for ollama and lm_studio).
        #[arg(long, value_parser = ["at_start", "on_demand"])]
        gpu_claim: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Show a workload's log from its service manager (journal). Manager
    /// lines are prefixed `systemd:`; `-- run … --` marks each new start.
    Logs {
        name: String,
        /// Newest entries to show (max 1000).
        #[arg(short = 'n', long, default_value_t = DEFAULT_LOG_LINES)]
        lines: u32,
        /// Keep printing new entries until interrupted.
        #[arg(short, long)]
        follow: bool,
        /// One JSON object per entry.
        #[arg(long)]
        json: bool,
    },
    /// Take over an adopted workload's unit file: replace the file (or the
    /// symlink itself) with the rendered definition, prove systemd still
    /// sees the identical effective definition, and keep the original for
    /// release. The unit is not restarted.
    Own { name: String },
    /// Hand an owned workload back: restore the recorded original
    /// byte-for-byte (or the link) and set ownership to external.
    Release { name: String },
    /// Staged version upgrades (plan D): stage builds a new env side by
    /// side with uv and validates it offline; promote swaps the definition
    /// as an operation with automatic rollback to the pinned N.
    Upgrade {
        #[command(subcommand)]
        cmd: UpgradeCmd,
    },
    /// llama-swap lanes as records (M4b): import the config, then change
    /// lanes through planned, confirmed writes that roll back when
    /// llama-swap does not take the new file.
    Lanes {
        #[command(subcommand)]
        cmd: LanesCmd,
    },
    /// llama.cpp (M5d): router models loaded and unloaded by verb after
    /// the fit/budget check, preset edits, and Mac llama.cpp lanes.
    Llamacpp {
        #[command(subcommand)]
        cmd: llamacpp_cmd::LlamacppCmd,
    },
    /// Installs per recipe kind (M4a): stage fetches/builds a new
    /// versioned dir; plan prints the vendor-app plan (owner-run steps,
    /// never executed by llmctl).
    Install {
        #[command(subcommand)]
        cmd: InstallCmd,
    },
    /// A workload's install recipe: what its version facts are read from.
    /// `set` attaches or replaces it and pins at once; `clear` removes it.
    Recipe {
        #[command(subcommand)]
        cmd: RecipeCmd,
    },
    /// Remove a workload's record. Nothing on the host is stopped or
    /// deleted. Without --yes it only says what would go.
    Forget {
        name: String,
        #[arg(long)]
        yes: bool,
    },
    /// Version facts of a workload's install: pin, live probe, drift. A
    /// mismatch is a finding, never auto-corrected. --repin records what
    /// the install reports now as the pin, and lists what changed: that is
    /// how a vendor upgrade or a package change is accepted.
    Version {
        name: String,
        #[arg(long)]
        repin: bool,
    },
    /// Run the daemon in the foreground (binds the control socket).
    Daemon {
        #[arg(long)]
        socket: Option<PathBuf>,
        /// Name this host's agent answers to — its enrollment name (`host
        /// add` sets it). Renames a previously bound id and the inventory's
        /// host and GPU ids once.
        #[arg(long)]
        host_id: Option<String>,
    },
    /// Relay stdin/stdout to the local control socket — the remote half of
    /// the SSH bridge (M3-E.1). The workstation runs `ssh <alias>
    /// llmctl-next bridge`; the same HTTP/1.1 API flows through the pipe.
    Bridge,
    /// Enrolled-host management (M3-D): install/upgrade/remove the agent
    /// on a Mac over ssh and record the enrollment on this workstation.
    Host {
        #[command(subcommand)]
        cmd: HostCmd,
    },
}

#[derive(Subcommand, Clone)]
enum HostCmd {
    /// Enroll a host: probe UUID/arch over ssh, provision the agent
    /// binary (reuse / --binary / --source), bootstrap its LaunchAgent,
    /// verify through the bridge, record it.
    Add {
        /// ssh alias (from ~/.ssh/config; WireGuard/LAN routes first).
        alias: String,
        /// Enrolled host name (default: the alias).
        #[arg(long)]
        name: Option<String>,
        /// Local path to a matching agent binary to install.
        #[arg(long)]
        binary: Option<PathBuf>,
        /// Local rust workspace dir: rsync + `cargo build --release` on
        /// the host itself (M3 build strategy).
        #[arg(long)]
        source: Option<PathBuf>,
    },
    /// Swap the agent binary and restart its LaunchAgent, with a protocol
    /// check before and after.
    Upgrade {
        name: String,
        #[arg(long)]
        binary: Option<PathBuf>,
        #[arg(long)]
        source: Option<PathBuf>,
    },
    /// Remove the agent (LaunchAgent, binary; data dir kept unless --purge).
    Remove {
        name: String,
        #[arg(long)]
        purge: bool,
    },
    /// Show enrolled hosts with agent version, protocol and last-seen.
    List,
}

#[derive(Subcommand, Clone)]
enum ModelCmd {
    /// Download weights into a declared store: `hf:org/name[@revision]`
    /// into an hf_cache or directory store, `ollama:tag` into an ollama
    /// store, `lms:publisher/model[@variant]` into an lmstudio store (LM
    /// Studio must be running; pass --size-gib, `lms` cannot say a size
    /// in advance). The disk is checked first (size + the store's reserve); the
    /// download runs as an operation beside swaps, shows progress, and can
    /// be cancelled (`llmctl-next cancel <id>`).
    Get {
        source: String,
        #[arg(long)]
        store: String,
        /// The workload whose journal this is filed under (default: the
        /// store's own workload).
        #[arg(long = "for", value_name = "WORKLOAD")]
        for_workload: Option<String>,
        /// Upper bound in GiB when the source cannot say its size.
        #[arg(long)]
        size_gib: Option<u64>,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        key: Option<String>,
    },
    /// Remove weights from a store: `hf:org/name`, `ollama:tag`, or
    /// `path:<relative path in the store>`. Without --confirm it shows
    /// the path, the size and who still references them; weights a
    /// workload, a profile or a lane names are never removed.
    Rm {
        target: String,
        #[arg(long)]
        store: String,
        /// The token the preview printed.
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand, Clone)]
enum StoresCmd {
    /// Declare a store.
    Add {
        name: String,
        /// hf_cache, directory, ollama or lmstudio.
        #[arg(long)]
        kind: String,
        /// Absolute path of the store's root.
        #[arg(long)]
        path: String,
        /// Free space a download must leave (default 20).
        #[arg(long)]
        reserve_gib: Option<u64>,
        /// The workload its downloads are journaled under; required for
        /// ollama and lmstudio stores (the service that downloads).
        #[arg(long)]
        workload: Option<String>,
    },
    /// Forget a store. The weights in it stay.
    Remove { name: String },
}

#[derive(Subcommand, Clone)]
enum RecipeCmd {
    /// The recipe and the extra pinned files, as JSON.
    Show { name: String },
    /// Attach or replace the recipe from a JSON file (the `recipe` object
    /// of an inventory record), then pin what the install reports.
    Set {
        name: String,
        #[arg(long)]
        file: std::path::PathBuf,
        /// Extra file pinned with the install (repeat it). Given at least
        /// once, the list replaces the current one.
        #[arg(long = "pin-file")]
        pin_files: Vec<String>,
        /// Drop every extra pinned file.
        #[arg(long, conflicts_with = "pin_files")]
        no_pin_files: bool,
    },
    /// Remove the recipe, the extra pinned files and the version pin.
    Clear { name: String },
}

#[derive(Subcommand, Clone)]
enum LanesCmd {
    /// Load one lane: shows which running lanes llama-swap will evict
    /// (from the records' matrix), requests the lane, and confirms it runs
    /// and that the eviction matched. --dry-run only predicts.
    Swap {
        workload: String,
        lane: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        key: Option<String>,
        /// Evict even while llama-swap has requests in flight.
        #[arg(long)]
        force: bool,
    },
    /// Unload one lane: llama-swap is asked to stop it, and it must leave
    /// the running list. Nothing else is evicted; the lane loads again on
    /// its next request. Held while requests are in flight unless --force.
    Stop {
        workload: String,
        lane: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        key: Option<String>,
        #[arg(long)]
        force: bool,
    },
    /// The host's memory budget against every combination the matrix
    /// allows, at the lanes' peak figures (a Mac has one pool of unified
    /// memory, no cards). --budget, --peak and --from-registry change the
    /// figures; with none it only reports. Exit 4 when a set is over.
    Memory {
        workload: String,
        /// GiB of model peaks the host can hold (0 removes the budget).
        #[arg(long)]
        budget: Option<u64>,
        /// LANE=GIB (repeat it); LANE=none removes a figure.
        #[arg(long = "peak", value_name = "LANE=GIB")]
        peaks: Vec<String>,
        /// A model registry (YAML) on that host to take peak figures from.
        #[arg(long)]
        from_registry: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// One lane's own output (the newest lines llama-swap keeps for it).
    Logs {
        workload: String,
        lane: String,
        #[arg(short = 'n', long, default_value_t = 100)]
        lines: u32,
        #[arg(long)]
        json: bool,
    },
    /// Preview (or, with --confirm, persist) the records of a llama-swap
    /// workload's config. Never writes the config.
    Import {
        workload: String,
        #[arg(long)]
        config: Option<String>,
        #[arg(long)]
        log: Option<String>,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        json: bool,
    },
    /// Records, drift against the live file, and validation.
    Show {
        workload: String,
        #[arg(long)]
        json: bool,
    },
    /// Version facts per lane: what each lane's command runs (the
    /// programs it names, and its Python env's packages) against the
    /// pin. --pin records the live facts as the pin. Exit 4 on drift.
    Version {
        workload: String,
        /// One lane instead of all.
        #[arg(long)]
        lane: Option<String>,
        #[arg(long)]
        pin: bool,
        #[arg(long)]
        json: bool,
    },
    /// Set one lane field (value parsed as JSON, else taken as a string).
    Set {
        workload: String,
        lane: String,
        field: String,
        value: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Remove one lane field.
    Unset {
        workload: String,
        lane: String,
        field: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Set a macro (an engine version swap is a macro path change).
    Macro {
        workload: String,
        name: String,
        value: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
    /// Write the records as they are (normalises the file to the render).
    Rewrite {
        workload: String,
        #[arg(long)]
        confirm: Option<String>,
        #[arg(long)]
        wait: bool,
    },
}

#[derive(Subcommand, Clone)]
enum UpgradeCmd {
    /// Build and offline-validate a new env for the workload's recipe.
    Stage {
        name: String,
        /// Target version of the recipe's primary package.
        version: String,
    },
    /// Promote a validated stage; --wait follows the operation.
    Promote {
        name: String,
        version: String,
        #[arg(long)]
        wait: bool,
    },
    /// Roll back to the last promoted upgrade's N.
    Rollback {
        name: String,
        #[arg(long)]
        wait: bool,
    },
    /// Upgrade history for the workload.
    List { name: String },
    /// Remove envs no upgrade row or the live interpreter needs.
    Gc { name: String },
}

#[derive(Subcommand, Clone)]
enum InstallCmd {
    /// Stage a new version per the workload's recipe kind: uv env
    /// (python-env), fetch+unpack (release-archive), git+build
    /// (source-build). Vendor apps refuse with a pointer to `plan`.
    Stage { name: String, version: String },
    /// The vendor-app plan: observed/pinned version, drift, and the
    /// owner-run backup/reinstall steps. Nothing is executed.
    Plan { name: String },
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
            | codes::INVALID_STATE
            | codes::DEFINITION_DRIFT
            | codes::GPU_OCCUPIED
            | codes::GPU_PROBE_FAILED
            | codes::STALE_ADOPTION
            | codes::ADOPTION_REFUSED
            | codes::ADOPTION_UNVERIFIED
            | codes::OWNERSHIP_REFUSED
            | codes::UPGRADE_REFUSED
            | codes::HOST_LOCK_BUSY
            | codes::LANE_LEASED
            | codes::REQUESTS_IN_FLIGHT
            | codes::UNSUPPORTED => 4,
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
    if let Some(name) = &cli.host {
        let hosts = enroll::load_hosts();
        let record = enroll::find_host(&hosts, name).ok_or_else(|| {
            ClientError::Unreachable {
                path: name.clone().into(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("no enrolled host {name:?}; `llmctl-next host add` first"),
                ),
            }
        })?;
        return Ok(ControlClient::connect_bridge(&record.ssh_alias));
    }
    let client = ControlClient::connect(socket_path(cli))?;
    // Inside a lane-lease holder (a longjob job that drives llmctl-next):
    // say so, or the agent would refuse the job's own commands (M4c).
    let client = match llmctl_client::lease::holder_ancestor() {
        Some(pid) => client.under_lease(pid),
        None => client,
    };
    // Called by whoever holds the host operation lock (llm-router during a
    // switch): work under that hold instead of being refused by it.
    Ok(match llmctl_client::oplock::holder_ancestor() {
        Some(pid) => client.holding_lock(pid),
        None => client,
    })
}

/// Mutations require a protocol match; reads (status, logs, ops) still
/// work across versions (M3-E.3). The named error is PROTOCOL_MISMATCH.
async fn ensure_protocol(client: &ControlClient) -> Result<(), ClientError> {
    let health = client.health().await?;
    if !health.supported_protocols.contains(&PROTOCOL_VERSION) {
        return Err(ClientError::Api {
            status: 409,
            code: codes::PROTOCOL_MISMATCH.into(),
            message: format!(
                "agent {} speaks protocol {} (supported: {:?}); this client speaks \
                 {PROTOCOL_VERSION}. Mutations refused — status/logs still work. \
                 `llmctl-next host upgrade` the agent.",
                health.daemon_version, health.protocol, health.supported_protocols
            ),
            details: None,
        });
    }
    Ok(())
}

#[tokio::main]
async fn main() -> ExitCode {
    // Rust ignores SIGPIPE, so `println!` panics when the reader goes away
    // (`llmctl-next ops | head`). A CLI should end quietly like other Unix
    // tools: restore the default action before any output.
    #[cfg(unix)]
    // SAFETY: called once at startup, before any other thread writes output;
    // `signal` with SIG_DFL has no other preconditions.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
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
        Command::Status { json, all } => {
            if all {
                cmd_status_all(cli, json).await
            } else {
                cmd_status(cli, json).await
            }
        }
        Command::Run {
            name,
            dry_run,
            wait,
            key,
            force,
            profile_env,
            model,
        } => {
            let profile = profile_env.zip(model).map(|(env_source, model)| ProfileRequest {
                // The agent resolves the path on its own host; a relative
                // one would mean this shell's directory, so make it absolute.
                env_source: std::path::absolute(&env_source)
                    .map(|p| p.display().to_string())
                    .unwrap_or(env_source),
                model,
            });
            cmd_mutate(cli, OpKind::Run, &name, dry_run, wait, key, force, profile).await
        }
        Command::Stop {
            name,
            wait,
            key,
            force,
        } => cmd_mutate(cli, OpKind::Stop, &name, false, wait, key, force, None).await,
        Command::Unload {
            name,
            model,
            wait,
            key,
        } => cmd_unload(cli, &name, &model, wait, key).await,
        Command::Engine { name, json } => cmd_engine(cli, &name, json).await,
        Command::Restart {
            name,
            dry_run,
            wait,
            key,
            force,
        } => cmd_mutate(cli, OpKind::Restart, &name, dry_run, wait, key, force, None).await,
        Command::Intent { name, json } => cmd_intent(cli, name.as_deref(), json).await,
        Command::Audit { since, all, json } => cmd_audit(cli, since.as_deref(), all, json).await,
        Command::Stores { cmd, json } => cmd_stores(cli, cmd, json).await,
        Command::Model { cmd } => cmd_model(cli, cmd).await,
        Command::Vram { measure, json } => cmd_vram(cli, measure.as_deref(), json).await,
        Command::Ops { json } => cmd_ops(cli, json).await,
        Command::Op { id, watch } => cmd_op(cli, &id, watch).await,
        Command::Cancel { id } => cmd_cancel(cli, &id).await,
        Command::Ack { id } => cmd_ack(cli, &id).await,
        Command::Logs {
            name,
            lines,
            follow,
            json,
        } => cmd_logs(cli, &name, lines, follow, json).await,
        Command::Own { name } => cmd_own(cli, &name).await,
        Command::Release { name } => cmd_release(cli, &name).await,
        Command::Upgrade { cmd } => cmd_upgrade(cli, cmd).await,
        Command::Lanes { cmd } => cmd_lanes(cli, cmd).await,
        Command::Llamacpp { cmd } => llamacpp_cmd::cmd_llamacpp(cli, cmd).await,
        Command::Install { cmd } => cmd_install(cli, cmd).await,
        Command::Version { name, repin } => cmd_version(cli, &name, repin).await,
        Command::Recipe { cmd } => cmd_recipe(cli, cmd).await,
        Command::Forget { name, yes } => cmd_forget(cli, &name, yes).await,
        Command::Adopt {
            unit,
            name,
            model,
            port,
            gpus,
            confirm,
            allow_unverified,
            engine_kind,
            gpu_claim,
            json,
        } => {
            let req = AdoptRequest {
                protocol: PROTOCOL_VERSION,
                unit,
                workload: name,
                model,
                port,
                gpus,
                confirm_digest: confirm,
                allow_unverified,
                engine_kind,
                gpu_claim,
            };
            cmd_adopt(cli, req, json).await
        }
        Command::Daemon { socket, host_id } => {
            llmctl_daemon::run_daemon(cli.data_dir.clone(), socket, host_id)
                .await
                .map_err(CliError::Daemon)?;
            Ok(0)
        }
        Command::Bridge => cmd_bridge(cli).await,
        Command::Host { cmd } => cmd_host(cli, cmd).await,
    }
}

/// The agent-side relay: local stdin/stdout <-> the control socket. Pure
/// byte pipe; HTTP framing stays with the client and daemon (decision 1).
async fn cmd_bridge(cli: &Cli) -> Result<u8, CliError> {
    use tokio::io::AsyncWriteExt as _;
    let path = socket_path(cli);
    let mut socket = tokio::net::UnixStream::connect(&path).await.map_err(|e| {
        CliError::Daemon(format!(
            "bridge: no agent socket at {}: {e}",
            path.display()
        ))
    })?;
    let (mut socket_rd, mut socket_wr) = socket.split();
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    // EOF on stdin (client gone) half-closes the socket; the daemon then
    // finishes its answer and closes, which ends the reader. Only a
    // *reader* error fails the relay — an EOF on either side is the
    // normal end.
    let to_socket = async {
        let copied = tokio::io::copy(&mut stdin, &mut socket_wr).await;
        let _ = socket_wr.shutdown().await;
        copied
    };
    let (to_res, from_res) = tokio::join!(to_socket, tokio::io::copy(&mut socket_rd, &mut stdout));
    if let Err(e) = from_res {
        eprintln!("bridge: socket -> stdout: {e}");
        return Ok(1);
    }
    if let Err(e) = to_res {
        eprintln!("bridge: stdin -> socket: {e}");
        return Ok(1);
    }
    let _ = stdout.flush().await;
    Ok(0)
}

async fn cmd_host(cli: &Cli, cmd: HostCmd) -> Result<u8, CliError> {
    let _ = cli;
    match cmd {
        HostCmd::Add {
            alias,
            name,
            binary,
            source,
        } => {
            let record = enroll::host_add(&alias, name, binary, source)
                .await
                .map_err(CliError::Daemon)?;
            println!(
                "enrolled {} (uuid {}, {} via {}): agent {} protocol {} — bridge OK",
                record.name,
                &record.io_platform_uuid[..8.min(record.io_platform_uuid.len())],
                record.arch,
                record.ssh_alias,
                record.agent_version.as_deref().unwrap_or("?"),
                record
                    .protocol
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "?".into()),
            );
            Ok(0)
        }
        HostCmd::Upgrade {
            name,
            binary,
            source,
        } => {
            enroll::host_upgrade(&name, binary, source)
                .await
                .map_err(CliError::Daemon)?;
            Ok(0)
        }
        HostCmd::Remove { name, purge } => {
            enroll::host_remove(&name, purge).map_err(CliError::Daemon)?;
            Ok(0)
        }
        HostCmd::List => {
            let hosts = enroll::load_hosts();
            if hosts.hosts.is_empty() {
                println!("no enrolled hosts (see `llmctl-next host add`)");
                return Ok(0);
            }
            for h in &hosts.hosts {
                println!(
                    "{:<12} {:<8} alias {:<14} agent {:<8} protocol {:<3} seen {}",
                    h.name,
                    h.arch,
                    h.ssh_alias,
                    h.agent_version.as_deref().unwrap_or("?"),
                    h.protocol.map(|p| p.to_string()).unwrap_or_else(|| "?".into()),
                    h.last_seen
                        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
                        .unwrap_or_else(|| "never".into()),
                );
            }
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
        let recovery = match &w.recovery_policy {
            Some(p) => format!("recover={} state={}", p, w.recovery_state),
            None if w.recovery_state != "alive" => format!("state={}", w.recovery_state),
            None => String::new(),
        };
        if recovery.is_empty() {
            println!(
                "{:<10} {:<14} gpus={:<28} {:<8} {:<8} serves={:<18} intent={:<18} {:<8} seen {}",
                w.name,
                w.model,
                w.gpus.join(","),
                w.process,
                w.readiness,
                serves,
                w.intent,
                w.ownership,
                observed
            );
        } else {
            println!(
                "{:<10} {:<14} gpus={:<28} {:<8} {:<8} serves={:<18} intent={:<18} {:<8} seen {}  {}",
                w.name, w.model, w.gpus.join(","), w.process, w.readiness, serves, w.intent,
                w.ownership, observed, recovery
            );
        }
    }
    for g in &status.gpus {
        if let Some(err) = &g.error {
            println!("gpu {:<6} occupancy unknown: {err}", g.index);
            continue;
        }
        let enrolled = if g.enrolled.is_empty() {
            "-".to_string()
        } else {
            g.enrolled.join(",")
        };
        let foreign = if g.foreign.is_empty() {
            "-".to_string()
        } else {
            g.foreign
                .iter()
                .map(|o| match &o.unit {
                    Some(u) => format!("pid {} ({}, {} MiB)", o.pid, u, o.memory_mib),
                    None => format!("pid {} (no unit, {} MiB)", o.pid, o.memory_mib),
                })
                .collect::<Vec<_>>()
                .join("; ")
        };
        println!(
            "gpu {:<6} enrolled={:<12} foreign={}",
            g.index, enrolled, foreign
        );
    }
    Ok(0)
}

/// How long `status --all` waits for one enrolled host before showing its
/// cached observations as STALE: `LLMCTL_NEXT_STATUS_TIMEOUT_S` (whole
/// seconds, at least 1), default 10.
fn remote_status_timeout() -> std::time::Duration {
    parse_status_timeout(std::env::var("LLMCTL_NEXT_STATUS_TIMEOUT_S").ok().as_deref())
}

fn parse_status_timeout(v: Option<&str>) -> std::time::Duration {
    let secs = v
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|s| *s >= 1)
        .unwrap_or(10);
    std::time::Duration::from_secs(secs)
}

/// `status --all`: local host first, then every enrolled host, each with
/// freshness. An unreachable host shows its last cached observations
/// marked STALE — never as "stopped" (M3-E.2).
async fn cmd_status_all(cli: &Cli, json: bool) -> Result<u8, CliError> {
    use llmctl_client::wire::StatusResponse;

    fn ago(t: chrono::DateTime<chrono::Utc>) -> String {
        let secs = (chrono::Utc::now() - t).num_seconds().max(0);
        if secs < 60 {
            format!("{secs} s ago")
        } else if secs < 3600 {
            format!("{} min ago", secs / 60)
        } else {
            format!("{} h ago", secs / 3600)
        }
    }

    fn block(label: &str, fresh: Option<&StatusResponse>, cached: Option<StatusResponse>, unreachable_since: Option<chrono::DateTime<chrono::Utc>>) {
        match (fresh, cached) {
            (Some(s), _) => {
                println!("host {label} ({}) · revision {} · seen {}", s.platform, s.inventory_revision, ago(s.generated_at));
                for w in &s.workloads {
                    println!(
                        "  {:<12} {:<8} {:<8} serves={:<24} intent={:<12} {}",
                        w.name,
                        w.process,
                        w.readiness,
                        w.served_model.as_deref().unwrap_or("-"),
                        w.intent,
                        w.observed_at.map(|t| ago(t)).unwrap_or_else(|| "never".into())
                    );
                }
            }
            (None, Some(s)) => {
                println!(
                    "host {label} UNREACHABLE since {} — last observations (STALE):",
                    unreachable_since
                        .map(|t| t.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                        .unwrap_or_else(|| "?".into())
                );
                for w in &s.workloads {
                    println!(
                    "  {:<12} {:<8} {:<8} serves={:<24} intent={:<12} {}  [stale]",
                    w.name,
                    w.process,
                    w.readiness,
                    w.served_model.as_deref().unwrap_or("-"),
                    w.intent,
                    w.observed_at.map(|t| ago(t)).unwrap_or_else(|| "never".into())
                );
                }
            }
            (None, None) => {
                println!("host {label} UNREACHABLE (no cached observations)");
            }
        }
    }

    // Local first (a missing local daemon is reported, not fatal).
    let local = match ControlClient::connect(socket_path(cli)) {
        Ok(c) => c.status().await.ok(),
        Err(_) => None,
    };
    let local_label = local
        .as_ref()
        .map(|s| s.host.clone())
        .unwrap_or_else(|| "local (daemon down)".into());
    block(&local_label, local.as_ref(), None, None);

    let mut hosts = enroll::load_hosts();
    let mut dirty = false;
    for h in &mut hosts.hosts {
        let client = ControlClient::connect_bridge(&h.ssh_alias);
        // A sleeping Mac does not refuse the bridge, it lets ssh wait (and a
        // Wi-Fi wake can follow a minute later). Bound the wait so the view
        // shows the cached observations as STALE instead of hanging; the
        // dropped future drops the bridge, whose ssh is killed on drop.
        let limit = remote_status_timeout();
        let answer = match tokio::time::timeout(limit, client.status()).await {
            Ok(r) => r.map_err(|e| e.to_string()),
            Err(_) => Err(format!(
                "no answer within {} s: asleep or unreachable (LLMCTL_NEXT_STATUS_TIMEOUT_S sets the limit)",
                limit.as_secs()
            )),
        };
        match answer {
            Ok(status) => {
                h.last_seen = Some(chrono::Utc::now());
                dirty = true;
                if let Some(parent) = enroll::status_cache_path(&h.name).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(
                    enroll::status_cache_path(&h.name),
                    serde_json::to_string(&status).unwrap_or_default(),
                );
                block(&h.name, Some(&status), None, None);
            }
            Err(e) => {
                let cached = std::fs::read_to_string(enroll::status_cache_path(&h.name))
                    .ok()
                    .and_then(|t| serde_json::from_str::<StatusResponse>(&t).ok());
                block(&h.name, None, cached, h.last_seen);
                eprintln!("  ({e})");
            }
        }
    }
    if dirty {
        let _ = enroll::save_hosts(&hosts);
    }
    if json {
        // JSON mode: one object per host is printed above as text; a
        // machine-readable --all lands with the remote-status tooling.
        eprintln!("note: --all --json is not implemented yet");
    }
    Ok(0)
}

async fn cmd_adopt(cli: &Cli, req: AdoptRequest, json: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    if req.confirm_digest.is_some() {
        ensure_protocol(&client).await?;
    }
    if req.confirm_digest.is_none() {
        let preview = client.adoption_preview(&req).await?;
        if json {
            println!("{}", serde_json::to_string_pretty(&preview).unwrap());
        } else {
            print_preview(&preview);
            if preview.has_failures() {
                println!("\nnot adoptable: fix the failed check(s) above, then preview again");
            } else if let Some(digest) = &preview.digest {
                println!(
                    "\nto transfer ownership:\n  llmctl-next adopt {} --name {} --model {} --port {} --gpus {} --confirm {}{}",
                    req.unit,
                    req.workload,
                    req.model,
                    req.port,
                    req.gpus.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
                    digest,
                    if preview.has_unverified() { " --allow-unverified" } else { "" }
                );
            }
        }
        return Ok(if preview.has_failures() { 4 } else { 0 });
    }
    match client.adopt(&req).await {
        Ok(resp) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&resp).unwrap());
            } else {
                println!(
                    "adopted {} as managed workload {} (inventory revision {}, definition {})",
                    req.unit,
                    req.workload,
                    resp.inventory_revision,
                    resp.preview.digest.as_deref().unwrap_or("?")
                );
            }
            Ok(0)
        }
        Err(e) => {
            if let ClientError::Api {
                details: Some(details),
                ..
            } = &e
            {
                if let Ok(preview) =
                    serde_json::from_value::<AdoptionPreview>(details["preview"].clone())
                {
                    if json {
                        println!("{}", serde_json::to_string_pretty(&preview).unwrap());
                    } else {
                        print_preview(&preview);
                        println!();
                    }
                }
            }
            Err(e.into())
        }
    }
}

fn print_preview(p: &AdoptionPreview) {
    println!("adoption preview: {} → workload {}", p.unit, p.workload);
    for c in &p.checks {
        let mark = match c.status {
            CheckStatus::Pass => "pass",
            CheckStatus::Fail => "FAIL",
            CheckStatus::Unverified => "unverified",
            CheckStatus::Warn => "warn",
        };
        println!("  {mark:<10} {:<21} {}", c.name, c.detail);
    }
    for f in &p.provenance {
        println!("  source     {} ({})", f.path, f.content_digest);
    }
    if let Some(unit) = &p.generated_unit {
        println!("\ngenerated service definition:");
        for line in unit.lines() {
            println!("  | {line}");
        }
    }
    if let Some(d) = &p.digest {
        println!("\ndefinition digest: {d}");
    }
}

#[allow(clippy::too_many_arguments)]
async fn cmd_mutate(
    cli: &Cli,
    kind: OpKind,
    name: &str,
    dry_run: bool,
    wait: bool,
    key: Option<String>,
    force: bool,
    profile: Option<ProfileRequest>,
) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    // Build the plan against fresh observations; the printed plan is what a
    // confirmation would refer to.
    let plan_resp = client
        .plan(PlanRequest {
            kind,
            workload: name.to_string(),
            profile: profile.clone(),
        })
        .await?;
    if dry_run {
        print_plan(&plan_resp.plan, &plan_resp.stop_targets);
        for note in &plan_resp.notes {
            println!("note: {note}");
        }
        return Ok(0);
    }

    // Every invocation is a new operation unless the caller passes an
    // earlier key to retry it. (run/stop used to key on the inventory
    // revision alone; the M2 pilot showed why that is wrong: the revision
    // does not move when the world does, so a `run` after a crash replayed
    // the first run's success and changed nothing.)
    let idempotency_key = key.unwrap_or_else(|| {
        format!(
            "{kind}-{name}-r{}-{}",
            plan_resp.revision,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or_default()
        )
    });
    let admit = match client
        .admit(AdmitRequest {
            protocol: PROTOCOL_VERSION,
            idempotency_key: idempotency_key.clone(),
            kind,
            workload: name.to_string(),
            expected_revision: Some(plan_resp.revision),
            expected_stop_targets: Some(plan_resp.stop_targets.clone()),
            force,
            profile: profile.clone(),
        })
        .await
    {
        Ok(admit) => admit,
        // Lost-ACK rule (M3 decision 4): a transport error after the
        // request was written is ambiguous, never "failed". Ask the agent
        // for the key before concluding anything; if the admit landed we
        // resume that operation; if the agent never saw it we retry once
        // with the same key (idempotency makes that safe).
        Err(e @ (ClientError::Disconnected(_) | ClientError::Unreachable { .. })) => {
            eprintln!("transport error during admit ({e}); resolving by key {idempotency_key}");
            let retry_client = connect(cli)?;
            match retry_client.operation_by_key(&idempotency_key).await {
                Ok(op) => {
                    println!(
                        "the admit landed: operation {} ({}) — no second effect",
                        op.id, op.idempotency_key
                    );
                    llmctl_client::wire::AdmitResponse {
                        operation: op,
                        replayed: true,
                    }
                }
                Err(ClientError::Api { code, .. }) if code == codes::UNKNOWN_OPERATION => {
                    println!("the admit never reached the agent; retrying with the same key");
                    retry_client
                        .admit(AdmitRequest {
                            protocol: PROTOCOL_VERSION,
                            idempotency_key: idempotency_key.clone(),
                            kind,
                            workload: name.to_string(),
                            expected_revision: Some(plan_resp.revision),
                            expected_stop_targets: Some(plan_resp.stop_targets.clone()),
                            force,
                            profile: profile.clone(),
                        })
                        .await?
                }
                Err(e2) => {
                    return Err(CliError::Client(ClientError::Disconnected(format!(
                        "cannot resolve whether the operation landed (key \
                         {idempotency_key}): first {e}; then {e2}. The effect is \
                         ambiguous — re-run with `--key {idempotency_key}` once \
                         the host is reachable."
                    ))));
                }
            }
        }
        Err(e) => return Err(e.into()),
    };
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
    watch_and_report(&client, &op.id, watch_budget(&plan_resp.plan)).await
}

async fn cmd_logs(
    cli: &Cli,
    name: &str,
    lines: u32,
    follow: bool,
    json: bool,
) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let mut printer = LogPrinter {
        json,
        last_run: None,
        last_file_time: None,
    };
    let mut page = client
        .logs(&LogsRequest {
            workload: name.to_string(),
            lines: Some(lines),
            after: None,
        })
        .await?;
    loop {
        for entry in &page.entries {
            if printer.print(entry).is_err() {
                return Ok(0); // reader went away (`| head`)
            }
        }
        if !follow {
            return Ok(0);
        }
        if !page.more {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => return Ok(0),
                _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
            }
        }
        page = client
            .logs(&LogsRequest {
                workload: name.to_string(),
                lines: Some(MAX_LOG_LINES),
                after: page.cursor.clone(),
            })
            .await?;
    }
}

struct LogPrinter {
    json: bool,
    last_run: Option<String>,
    /// File time last announced for lines without their own timestamp.
    last_file_time: Option<chrono::DateTime<chrono::Utc>>,
}

impl LogPrinter {
    fn print(&mut self, e: &LogEntry) -> std::io::Result<()> {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        if self.json {
            return writeln!(out, "{}", serde_json::to_string(e).unwrap());
        }
        if e.invocation.is_some() && e.invocation != self.last_run {
            let id = e.invocation.as_deref().unwrap_or_default();
            writeln!(out, "-- run {} --", &id[..id.len().min(8)])?;
            self.last_run = e.invocation.clone();
        }
        let when = e
            .timestamp
            .with_timezone(&chrono::Local)
            .format("%b %d %H:%M:%S");
        let who = match e.source {
            LogSource::Manager => "systemd: ",
            LogSource::Unit => "",
        };
        let cut = if e.truncated { " […truncated]" } else { "" };
        if e.time_estimated {
            // A shared file time on every line would read as each line's own
            // time (M3 pilot finding 5): say it once, print lines bare.
            if self.last_file_time != Some(e.timestamp) {
                writeln!(out, "-- lines carry no timestamps; log file last written {when} --")?;
                self.last_file_time = Some(e.timestamp);
            }
            return writeln!(out, "{who}{}{cut}", printable(&e.message));
        }
        writeln!(out, "{when} {who}{}{cut}", printable(&e.message))
    }
}

/// Log text comes from the workload and is untrusted: show C0/C1 control
/// characters (ESC, CSI, CR, BEL, …) as visible escapes so a log line can
/// never move the cursor, retitle or rewrite the terminal. Tab is kept.
/// `--json` needs none of this (serde escapes controls).
fn printable(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.chars().any(|c| c.is_control() && c != '\t') {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len() + 8);
    for c in text.chars() {
        if c.is_control() && c != '\t' {
            if (c as u32) < 0x100 {
                out.push_str(&format!("\\x{:02x}", c as u32));
            } else {
                out.push_str(&c.escape_unicode().to_string());
            }
        } else {
            out.push(c);
        }
    }
    std::borrow::Cow::Owned(out)
}

async fn cmd_own(cli: &Cli, name: &str) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let resp = client.own(name).await?;
    println!(
        // The digest is of the file own just wrote (the drift guard), not
        // of the one it replaced (M3 pilot finding 4).
        "own: {} → {} (written: digest {}; previous: {})",
        name, resp.path, resp.digest, resp.previous_kind
    );
    println!(
        "the unit was not restarted; release restores the previous {}",
        {
            if resp.previous_kind == "link" {
                "link"
            } else {
                "file byte-for-byte"
            }
        }
    );
    Ok(0)
}

async fn cmd_release(cli: &Cli, name: &str) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let resp = client.release(name).await?;
    println!(
        "release: {} back to {} (revision {})",
        name, resp.ownership, resp.inventory_revision
    );
    Ok(0)
}

async fn cmd_recipe(cli: &Cli, cmd: RecipeCmd) -> Result<u8, CliError> {
    use llmctl_client::wire::{RecipeSetRequest, PROTOCOL_VERSION};
    let client = connect(cli)?;
    let show = |v: &serde_json::Value| println!("{}", serde_json::to_string_pretty(v).unwrap());
    match cmd {
        RecipeCmd::Show { name } => {
            let inv = client.inventory().await?;
            let record = inv["workloads"]
                .as_array()
                .and_then(|ws| ws.iter().find(|w| w["name"] == name.as_str()));
            let Some(w) = record else {
                eprintln!("error: no workload named {name}");
                return Ok(5);
            };
            show(&serde_json::json!({
                "workload": name,
                "recipe": w["recipe"],
                "pin_files": w.get("pin_files").cloned().unwrap_or(serde_json::json!([])),
            }));
            Ok(0)
        }
        RecipeCmd::Set {
            name,
            file,
            pin_files,
            no_pin_files,
        } => {
            ensure_protocol(&client).await?;
            let text = match std::fs::read_to_string(&file) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error: reading {}: {e}", file.display());
                    return Ok(2);
                }
            };
            let recipe: serde_json::Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: {} is not JSON: {e}", file.display());
                    return Ok(2);
                }
            };
            let pin_files = if no_pin_files {
                Some(Vec::new())
            } else if pin_files.is_empty() {
                None
            } else {
                Some(pin_files)
            };
            let report = client
                .recipe_set(
                    &name,
                    &RecipeSetRequest {
                        protocol: PROTOCOL_VERSION,
                        recipe: Some(recipe),
                        pin_files,
                    },
                )
                .await?;
            show(&report);
            Ok(0)
        }
        RecipeCmd::Clear { name } => {
            ensure_protocol(&client).await?;
            let report = client
                .recipe_set(
                    &name,
                    &RecipeSetRequest {
                        protocol: PROTOCOL_VERSION,
                        recipe: None,
                        pin_files: None,
                    },
                )
                .await?;
            show(&report);
            Ok(0)
        }
    }
}

async fn cmd_forget(cli: &Cli, name: &str, yes: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let report = client
        .forget(
            name,
            &llmctl_client::wire::ForgetRequest {
                protocol: llmctl_client::wire::PROTOCOL_VERSION,
                confirm: yes,
            },
        )
        .await?;
    let notes: Vec<&str> = report["notes"]
        .as_array()
        .map(|n| n.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if report["removed"] == true {
        println!(
            "forgot {name} ({}); inventory revision {}",
            report["unit"].as_str().unwrap_or("?"),
            report["inventory_revision"]
        );
    } else {
        println!(
            "would forget {name} ({}): {}, {}. Nothing on the host is stopped or deleted.",
            report["unit"].as_str().unwrap_or("?"),
            if report["managed"] == true { "managed" } else { "not managed" },
            if report["running"] == true { "running" } else { "not running" },
        );
    }
    for n in notes {
        println!("  note: {n}");
    }
    if report["removed"] != true {
        println!("repeat with --yes to remove the record");
    }
    Ok(0)
}

async fn cmd_version(cli: &Cli, name: &str, repin: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    if repin {
        ensure_protocol(&client).await?;
        let report = client.version_repin(name).await?;
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return Ok(0);
    }
    let report = client.version_report(name).await?;
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    let drifting = report["drift"].as_array().is_some_and(|d| !d.is_empty());
    Ok(if drifting { 4 } else { 0 })
}

async fn cmd_install(cli: &Cli, cmd: InstallCmd) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    match cmd {
        InstallCmd::Stage { name, version } => {
            let resp = client.stage(&name, &version).await?;
            println!(
                "staged {} {} at {}\n  executable: {}\n  status: {}\n  validation: {}",
                name, version, resp.env, resp.interpreter, resp.status, resp.validation
            );
            Ok(0)
        }
        InstallCmd::Plan { name } => {
            let plan = client.install_plan(&name).await?;
            println!("{}", serde_json::to_string_pretty(&plan).unwrap());
            let drifting = plan["drift"].as_array().is_some_and(|d| !d.is_empty());
            Ok(if drifting { 4 } else { 0 })
        }
    }
}

async fn cmd_upgrade(cli: &Cli, cmd: UpgradeCmd) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    match cmd {
        UpgradeCmd::Stage { name, version } => {
            let resp = client.stage(&name, &version).await?;
            println!(
                "staged {} {} at {}\n  interpreter: {}\n  status: {}\n  validation: {}",
                name, version, resp.env, resp.interpreter, resp.status, resp.validation
            );
            Ok(0)
        }
        UpgradeCmd::Promote {
            name,
            version,
            wait,
        } => {
            let view = client.promote(&name, &version, wait).await?;
            let id = view["id"].as_str().unwrap_or("?");
            if wait {
                watch_and_report(&client, id, DEFAULT_WATCH).await
            } else {
                println!("promote admitted: llmctl-next op {id} --watch");
                Ok(0)
            }
        }
        UpgradeCmd::Rollback { name, wait } => {
            let view = client.rollback_upgrade(&name, wait).await?;
            let id = view["id"].as_str().unwrap_or("?");
            if wait {
                watch_and_report(&client, id, DEFAULT_WATCH).await
            } else {
                println!("rollback admitted: llmctl-next op {id} --watch");
                Ok(0)
            }
        }
        UpgradeCmd::List { name } => {
            let rows = client.list_upgrades(&name).await?;
            if rows.upgrades.is_empty() {
                println!("no upgrades for {name}");
                return Ok(0);
            }
            for r in &rows.upgrades {
                println!(
                    "{}  {:<12} {} → {}  {}  op={}",
                    r.created_at.format("%Y-%m-%d %H:%M:%S"),
                    r.status,
                    r.from_version,
                    r.to_version,
                    r.env_path,
                    r.operation_id.as_deref().unwrap_or("-")
                );
            }
            Ok(0)
        }
        UpgradeCmd::Gc { name } => {
            let resp = client.gc_upgrades(&name).await?;
            for r in &resp.removed {
                println!("removed {r}");
            }
            for k in &resp.kept {
                println!("kept    {k}");
            }
            Ok(0)
        }
    }
}

async fn cmd_stores(cli: &Cli, cmd: Option<StoresCmd>, json: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    match cmd {
        Some(StoresCmd::Add {
            name,
            kind,
            path,
            reserve_gib,
            workload,
        }) => {
            ensure_protocol(&client).await?;
            let r = client
                .store_add(&llmctl_client::wire::StoreAddRequest {
                    protocol: PROTOCOL_VERSION,
                    name: name.clone(),
                    kind,
                    path,
                    reserve_gib,
                    workload,
                })
                .await?;
            println!("store {name} declared; inventory revision {}", r["inventory_revision"]);
            return Ok(0);
        }
        Some(StoresCmd::Remove { name }) => {
            ensure_protocol(&client).await?;
            let r = client
                .store_remove(&llmctl_client::wire::StoreRemoveRequest {
                    protocol: PROTOCOL_VERSION,
                    name: name.clone(),
                })
                .await?;
            println!(
                "store {name} forgotten (its weights stay); inventory revision {}",
                r["inventory_revision"]
            );
            return Ok(0);
        }
        None => {}
    }
    let resp = client.stores().await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        return Ok(0);
    }
    if resp.stores.is_empty() {
        println!("no store declared on {} (add `stores` to the inventory)", resp.host);
    }
    for s in &resp.stores {
        println!(
            "{:<14} {:<10} {:<44} free {:<10} reserve {} GiB{}{}",
            s.name,
            s.kind,
            s.path,
            s.free_bytes
                .map(|b| format!("{:.0} GiB", b as f64 / 1073741824.0))
                .unwrap_or_else(|| "unknown".into()),
            s.reserve_gib,
            s.workload.as_ref().map(|w| format!("  for {w}")).unwrap_or_default(),
            if s.exists { "" } else { "  (does not exist yet)" }
        );
    }
    Ok(0)
}

async fn cmd_model(cli: &Cli, cmd: ModelCmd) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let (source, store, for_workload, size_gib, dry_run, wait, key) = match cmd {
        ModelCmd::Get {
            source,
            store,
            for_workload,
            size_gib,
            dry_run,
            wait,
            key,
        } => (source, store, for_workload, size_gib, dry_run, wait, key),
        ModelCmd::Rm {
            target,
            store,
            confirm,
            json,
        } => {
            let r = client
                .model_remove(&llmctl_client::wire::ModelRemoveRequest {
                    protocol: PROTOCOL_VERSION,
                    target,
                    store,
                    confirm,
                })
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
                return Ok(if r.references.is_empty() { 0 } else { 4 });
            }
            let size = r
                .bytes
                .map(|b| format!("{:.1} GiB", b as f64 / 1073741824.0))
                .unwrap_or_else(|| "size unknown".into());
            let what = r.path.clone().unwrap_or_else(|| r.target.clone());
            if r.removed {
                println!("removed {what} from store {} ({size})", r.store);
                return Ok(0);
            }
            println!("{} in store {}: {what} ({size})", r.target, r.store);
            if !r.references.is_empty() {
                println!("  not removed: it is still referenced");
                for line in &r.references {
                    println!("    {line}");
                }
                return Ok(4);
            }
            println!("  nothing references it");
            if let Some(token) = &r.confirm {
                println!(
                    "to remove it for good:\n  llmctl-next model rm {} --store {} --confirm {token}",
                    r.target, r.store
                );
            }
            return Ok(0);
        }
    };
    let resp = client
        .acquire(&llmctl_client::wire::AcquireRequest {
            protocol: PROTOCOL_VERSION,
            source,
            store,
            for_workload,
            size_gib,
            dry_run,
            idempotency_key: key,
        })
        .await?;
    let gib = |b: u64| format!("{:.1} GiB", b as f64 / 1073741824.0);
    println!("{} -> store {} ({})", resp.source, resp.store, resp.path);
    println!(
        "  needs {}, free {}, reserve {} GiB",
        resp.need_bytes.map(gib).unwrap_or_else(|| "unknown".into()),
        gib(resp.free_bytes),
        resp.reserve_gib
    );
    println!("  runs: {}", resp.command.join(" "));
    let Some(op) = resp.operation else {
        return Ok(0);
    };
    println!("admitted operation {} ({}), filed under {}", op.id, op.idempotency_key, resp.workload);
    if !wait {
        println!("progress: llmctl-next op {} --watch   cancel: llmctl-next cancel {}", op.id, op.id);
        return Ok(0);
    }
    watch_and_report(&client, &op.id, std::time::Duration::from_secs(12 * 3600 + 300)).await
}

async fn cmd_vram(cli: &Cli, measure: Option<&str>, json: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let resp = match measure {
        Some(name) => client.vram_measure(name).await?,
        None => client.vram().await?,
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&resp).unwrap());
        return Ok(0);
    }
    let gib = |m: &std::collections::BTreeMap<u32, u64>| {
        m.iter()
            .map(|(c, v)| format!("{:.1} GiB on gpu {c}", *v as f64 / 1024.0))
            .collect::<Vec<_>>()
            .join(", ")
    };
    if let Some(r) = &resp.measured {
        println!("measured {} ({}): {}", r.workload, r.model, gib(&r.per_card_mib));
    }
    match &resp.free_mib {
        Some(free) => println!("free now: {}", gib(free)),
        None => println!("free now: unknown (the cards cannot be read)"),
    }
    if resp.records.is_empty() {
        println!("no workload measured yet: every start makes room the old way");
    }
    for r in &resp.records {
        println!(
            "  {:<14} {:<34} {}  ({})",
            r.workload,
            r.model,
            gib(&r.per_card_mib),
            r.measured_at.format("%Y-%m-%d %H:%M")
        );
    }
    Ok(0)
}

/// `intent [NAME]`: exit 0 with the record(s); exit 5 when NAME matches no
/// enrolled workload or unit. A healer treats any failure as "not held".
async fn cmd_intent(cli: &Cli, name: Option<&str>, json: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let all = client.intents().await?;
    let bare = |s: &str| s.trim_end_matches(".service").to_string();
    let picked: Vec<_> = match name {
        None => all.intents.iter().collect(),
        Some(n) => all
            .intents
            .iter()
            .filter(|i| i.workload == n || bare(&i.unit) == bare(n))
            .collect(),
    };
    if let (Some(n), true) = (name, picked.is_empty()) {
        eprintln!("error: no enrolled workload or unit named {n:?} on {}", all.host);
        return Ok(5);
    }
    if json {
        let out = match (name, picked.as_slice()) {
            (Some(_), [one]) => serde_json::to_string_pretty(one),
            _ => serde_json::to_string_pretty(&picked),
        };
        println!("{}", out.unwrap());
        return Ok(0);
    }
    for i in picked {
        println!(
            "{:<14} {:<44} {}{}",
            i.workload,
            i.unit,
            i.intent,
            if i.manual_stop { "  (manual stop: leave it alone)" } else { "" }
        );
    }
    Ok(0)
}

/// `audit [--since] [--all] [--json]`: the foreign-write record (M5a).
/// Read-only; findings land through the agent's observation passes.
async fn cmd_audit(cli: &Cli, since: Option<&str>, all: bool, json: bool) -> Result<u8, CliError> {
    let since = match since {
        None => None,
        Some(raw) => Some(parse_since(raw)?),
    };
    let client = connect(cli)?;
    let mut total = 0usize;

    let report = |host: &str, resp: llmctl_client::wire::AuditResponse| {
        if json {
            println!("{}", serde_json::to_string_pretty(&resp).unwrap());
            return;
        }
        if resp.findings.is_empty() {
            println!("host {}: no foreign writes", host);
            return;
        }
        println!("host {}: {} foreign write(s)", host, resp.findings.len());
        for f in &resp.findings {
            println!(
                "  {}  {:<12} {:<10} {} -> {}{}",
                f.observed_at.format("%Y-%m-%d %H:%M:%S"),
                f.workload,
                f.what,
                fmt_seen(&f.before_state),
                fmt_seen(&f.after_state),
                f.actor
                    .as_deref()
                    .map(|a| format!("  actor={a}"))
                    .unwrap_or_default()
            );
        }
    };

    let local = client.audit(since).await?;
    total += local.findings.len();
    report(&local.host.clone(), local);

    if all {
        let hosts = enroll::load_hosts();
        for h in &hosts.hosts {
            let remote = ControlClient::connect_bridge(&h.ssh_alias);
            let answer = tokio::time::timeout(remote_status_timeout(), remote.audit(since)).await;
            match answer {
                Ok(Ok(resp)) => {
                    total += resp.findings.len();
                    report(&h.name.clone(), resp);
                }
                Ok(Err(e)) if e.to_string().contains("404") => {
                    eprintln!("host {}: audit not supported by this agent — upgrade it", h.name);
                }
                Ok(Err(e)) => eprintln!("host {}: {e}", h.name),
                Err(_) => eprintln!("host {}: no answer within the status timeout", h.name),
            }
        }
    }

    if !json {
        println!("\n{} foreign write(s){}", total, since.map(|t| format!(" since {}", t.to_rfc3339())).unwrap_or_default());
    }
    Ok(0)
}

/// `--since` accepts RFC 3339 or a bare UTC date (midnight).
fn parse_since(raw: &str) -> Result<chrono::DateTime<chrono::Utc>, CliError> {
    if let Ok(t) = chrono::DateTime::parse_from_rfc3339(raw) {
        return Ok(t.with_timezone(&chrono::Utc));
    }
    let date = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d").map_err(|e| {
        CliError::Daemon(format!("--since {raw:?}: not RFC 3339 or YYYY-MM-DD ({e})"))
    })?;
    Ok(date
        .and_hms_opt(0, 0, 0)
        .expect("midnight")
        .and_utc())
}

/// One comparison state, compact for one line: `running pid=100 model=qwen`.
fn fmt_seen(raw: &str) -> String {
    let v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return raw.to_string(),
    };
    let mut parts = Vec::new();
    if let Some(p) = v.get("process").and_then(|p| p.as_str()) {
        parts.push(p.to_string());
    }
    for key in ["pid", "model"] {
        if let Some(x) = v.get(key) {
            if !x.is_null() {
                let s = match x {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                parts.push(format!("{key}={s}"));
            }
        }
    }
    if let Some(lanes) = v.get("lanes").and_then(|l| l.as_array()) {
        let ids: Vec<String> = lanes
            .iter()
            .filter_map(|l| l.as_str().map(str::to_owned))
            .collect();
        parts.push(format!("lanes=[{}]", ids.join(",")));
    }
    parts.join(" ")
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
    }    for op in &ops.operations {
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
        return watch_and_report(&client, id, DEFAULT_WATCH).await;
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

async fn cmd_lanes(cli: &Cli, cmd: LanesCmd) -> Result<u8, CliError> {
    use llmctl_client::wire::{LaneEdit, LaneEditRequest, LaneImportRequest};
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let print_lanes = |lanes: &[llmctl_client::wire::LaneSummary]| {
        for l in lanes {
            println!(
                "  {:<28} {:<10} ttl {:<6} {:<26} {}",
                l.id,
                l.engine_kind,
                l.ttl.map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
                l.pinned.clone().unwrap_or_else(|| "(no pin)".into()),
                l.model_name.clone().unwrap_or_default()
            );
        }
    };
    let (workload, edits, confirm, wait) = match cmd {
        LanesCmd::Swap {
            workload,
            lane,
            dry_run,
            wait,
            key,
            force,
        } => {
            let resp = client
                .lanes_swap(&llmctl_client::wire::LaneSwapRequest {
                    protocol: PROTOCOL_VERSION,
                    workload,
                    lane,
                    dry_run,
                    force,
                    idempotency_key: key,
                    unload: false,
                })
                .await?;
            let list = |v: &[String]| if v.is_empty() { "nothing".to_string() } else { v.join(", ") };
            println!("{}: load lane {}", resp.workload, resp.lane);
            println!("  running now: {}", list(&resp.running));
            if resp.already_loaded {
                println!("  already running: nothing to load or evict");
            } else {
                println!("  llama-swap will evict: {}", list(&resp.evicts));
                if !resp.set.is_empty() {
                    println!("  matrix set: {} (eviction cost {})", resp.set, resp.cost);
                }
            }
            let Some(op) = resp.operation else {
                return Ok(0);
            };
            println!("admitted operation {} ({})", op.id, op.idempotency_key);
            if !wait {
                println!("progress: llmctl-next op {} --watch", op.id);
                return Ok(0);
            }
            return watch_and_report(
                &client,
                &op.id,
                std::time::Duration::from_secs(20 * 60),
            )
            .await;
        }
        LanesCmd::Stop {
            workload,
            lane,
            dry_run,
            wait,
            key,
            force,
        } => {
            let resp = client
                .lanes_swap(&llmctl_client::wire::LaneSwapRequest {
                    protocol: PROTOCOL_VERSION,
                    workload,
                    lane,
                    dry_run,
                    force,
                    idempotency_key: key,
                    unload: true,
                })
                .await?;
            let list = |v: &[String]| if v.is_empty() { "nothing".to_string() } else { v.join(", ") };
            println!("{}: stop lane {}", resp.workload, resp.lane);
            println!("  running now: {}", list(&resp.running));
            if resp.evicts.is_empty() {
                println!("  not running: nothing to stop");
                return Ok(0);
            }
            let Some(op) = resp.operation else {
                println!("  would unload: {}", resp.lane);
                return Ok(0);
            };
            println!("admitted operation {} ({})", op.id, op.idempotency_key);
            if !wait {
                println!("progress: llmctl-next op {} --watch", op.id);
                return Ok(0);
            }
            return watch_and_report(&client, &op.id, std::time::Duration::from_secs(5 * 60)).await;
        }
        LanesCmd::Memory {
            workload,
            budget,
            peaks,
            from_registry,
            json,
        } => {
            let mut parsed = Vec::new();
            for p in &peaks {
                let Some((lane, gib)) = p.split_once('=') else {
                    eprintln!("error: --peak takes LANE=GIB, not {p:?}");
                    return Ok(2);
                };
                let gib = if gib.eq_ignore_ascii_case("none") {
                    None
                } else {
                    match gib.parse::<f64>() {
                        Ok(g) => Some(g),
                        Err(_) => {
                            eprintln!("error: {gib:?} is not a number of GiB");
                            return Ok(2);
                        }
                    }
                };
                parsed.push((lane.to_string(), gib));
            }
            let r = client
                .lanes_memory(&llmctl_client::wire::LaneMemoryRequest {
                    protocol: PROTOCOL_VERSION,
                    workload,
                    budget_gib: budget,
                    peaks: parsed,
                    from_registry,
                })
                .await?;
            let code = if r.over.is_empty() { 0 } else { 4 };
            if json {
                println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
                return Ok(code);
            }
            match r.budget_gib {
                Some(b) => println!("{}: memory budget {b} GiB", r.workload),
                None => println!("{}: no memory budget", r.workload),
            }
            for (lane, gib) in &r.peaks {
                println!("  {lane:<28} peak {gib} GiB");
            }
            for line in &r.lines {
                println!("  {line}");
            }
            if r.changed {
                println!("  figures saved");
            }
            return Ok(code);
        }
        LanesCmd::Logs {
            workload,
            lane,
            lines,
            json,
        } => {
            let r = client
                .lanes_logs(&llmctl_client::wire::LaneLogsRequest {
                    protocol: PROTOCOL_VERSION,
                    workload,
                    lane,
                    lines,
                })
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
                return Ok(0);
            }
            if r.lines.is_empty() {
                println!("(llama-swap has no output for {} yet; a lane logs once it has run)", r.lane);
            }
            for l in &r.lines {
                println!("{l}");
            }
            return Ok(0);
        }
        LanesCmd::Import {
            workload,
            config,
            log,
            confirm,
            json,
        } => {
            let r = client
                .lanes_import(&LaneImportRequest {
                    protocol: PROTOCOL_VERSION,
                    workload,
                    config_path: config,
                    log_path: log,
                    confirm,
                })
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
                return Ok(0);
            }
            println!("{} ← {}", r.workload, r.config_path);
            println!("  round trip: {}", r.roundtrip);
            println!("  comments kept: {}", r.notes_kept);
            if r.validation.is_empty() {
                println!("  validation: ok");
            } else {
                println!("  validation: {}", r.validation.join("; "));
            }
            print_lanes(&r.lanes);
            if r.persisted {
                println!("records saved (config untouched)");
            } else if let Some(d) = &r.records_digest {
                println!("to save the records: llmctl-next lanes import {} --confirm {d}", r.workload);
            }
            return Ok(if r.roundtrip == "equivalent" && r.validation.is_empty() { 0 } else { 1 });
        }
        LanesCmd::Show { workload, json } => {
            let r = client.lanes_status(&workload).await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
                return Ok(0);
            }
            println!("{} ({}, revision {})", r.workload, r.config_path, r.revision);
            if r.drift {
                println!("  DRIFT: the file changed since the records (now {})", r.file_digest_now);
            }
            for (k, v) in &r.macros {
                println!("  macro {k} = {v}");
            }
            print_lanes(&r.lanes);
            if !r.validation.is_empty() {
                println!("  validation: {}", r.validation.join("; "));
            }
            for line in &r.memory {
                println!("  memory: {line}");
            }
            return Ok(if r.drift { 1 } else { 0 });
        }
        LanesCmd::Version {
            workload,
            lane,
            pin,
            json,
        } => {
            let r = client
                .lanes_version(&llmctl_client::wire::LaneVersionRequest {
                    protocol: llmctl_client::wire::PROTOCOL_VERSION,
                    workload,
                    lane,
                    pin,
                })
                .await?;
            let drifting = r.lanes.iter().filter(|l| !l.drift.is_empty()).count();
            if json {
                println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
                return Ok(if drifting > 0 { 4 } else { 0 });
            }
            let pinned = r.lanes.iter().filter(|l| l.pinned.is_some()).count();
            println!(
                "{}: {} lanes, {} pinned, {} drifting",
                r.workload,
                r.lanes.len(),
                pinned,
                drifting
            );
            for l in &r.lanes {
                let state = match (&l.pinned, l.drift.is_empty()) {
                    (None, _) => "not pinned",
                    (Some(_), true) => "matches the pin",
                    (Some(_), false) => "DRIFT",
                };
                println!(
                    "  {:<28} {:<10} {:<26} {}",
                    l.id,
                    l.engine_kind,
                    l.label.clone().unwrap_or_default(),
                    state
                );
                for d in &l.drift {
                    println!("      {}: pinned {} now {}", d.field, d.expected, d.actual);
                }
            }
            if !r.pinned_now.is_empty() {
                println!("  pinned now: {}", r.pinned_now.join(", "));
            }
            if pinned < r.lanes.len() {
                println!("  record the rest with: lanes version {} --pin", r.workload);
            }
            return Ok(if drifting > 0 { 4 } else { 0 });
        }
        LanesCmd::Set {
            workload,
            lane,
            field,
            value,
            confirm,
            wait,
        } => {
            let value = serde_json::from_str(&value).unwrap_or(serde_json::Value::String(value));
            (workload, vec![LaneEdit::SetField { lane, field, value }], confirm, wait)
        }
        LanesCmd::Unset {
            workload,
            lane,
            field,
            confirm,
            wait,
        } => (workload, vec![LaneEdit::RemoveField { lane, field }], confirm, wait),
        LanesCmd::Macro {
            workload,
            name,
            value,
            confirm,
            wait,
        } => (workload, vec![LaneEdit::SetMacro { name, value }], confirm, wait),
        LanesCmd::Rewrite {
            workload,
            confirm,
            wait,
        } => (workload, Vec::new(), confirm, wait),
    };
    let mut req = LaneEditRequest {
        protocol: PROTOCOL_VERSION,
        workload: workload.clone(),
        edits,
        confirm: None,
        idempotency_key: None,
    };
    let Some(confirm) = confirm else {
        let p = client.lanes_plan(&req).await?;
        println!("{}: {}", p.workload, p.summary);
        for c in &p.changes {
            println!("  change {c}");
        }
        if !p.bytes_changed {
            println!("  the render equals the live file: nothing to write");
        }
        if !p.validation.is_empty() {
            println!("  REFUSED by validation: {}", p.validation.join("; "));
            return Ok(1);
        }
        if p.reload_restarts.is_empty() {
            println!("  reload restarts: nothing running");
        } else {
            println!("  reload restarts running lanes: {}", p.reload_restarts.join(", "));
        }
        if !p.reload_preloads.is_empty() {
            println!("  reload preloads: {}", p.reload_preloads.join(", "));
        }
        println!("to write it: add --confirm {}", p.plan_digest);
        return Ok(0);
    };
    req.confirm = Some(confirm);
    let admit = client.lanes_apply(&req).await?;
    let op = admit.operation;
    println!("admitted operation {} ({})", op.id, op.idempotency_key);
    if !wait {
        println!("progress: llmctl-next op {} --watch", op.id);
        return Ok(0);
    }
    watch_and_report(&client, &op.id, DEFAULT_WATCH).await
}

async fn cmd_unload(
    cli: &Cli,
    name: &str,
    model: &str,
    wait: bool,
    key: Option<String>,
) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let idempotency_key = key.unwrap_or_else(|| {
        // Model tags carry ':' and '/'; keys allow [A-Za-z0-9._-] only.
        let safe: String = model
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
            .take(64)
            .collect();
        format!(
            "unload-{name}-{safe}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or_default()
        )
    });
    let req = llmctl_client::wire::UnloadRequest {
        protocol: PROTOCOL_VERSION,
        idempotency_key: idempotency_key.clone(),
        workload: name.to_string(),
        model: model.to_string(),
    };
    let admit = match client.unload(&req).await {
        Ok(a) => a,
        // Lost-ACK rule, as for run/stop: resolve by key before retrying.
        Err(e @ (ClientError::Disconnected(_) | ClientError::Unreachable { .. })) => {
            eprintln!("transport error during admit ({e}); resolving by key {idempotency_key}");
            let retry = connect(cli)?;
            match retry.operation_by_key(&idempotency_key).await {
                Ok(op) => llmctl_client::wire::AdmitResponse {
                    operation: op,
                    replayed: true,
                },
                Err(ClientError::Api { code, .. }) if code == codes::UNKNOWN_OPERATION => {
                    retry.unload(&req).await?
                }
                Err(e2) => return Err(CliError::Client(e2)),
            }
        }
        Err(e) => return Err(e.into()),
    };
    let op = admit.operation;
    if admit.replayed {
        println!("replaying existing operation {} (same idempotency key)", op.id);
    } else {
        println!("admitted operation {} ({})", op.id, op.idempotency_key);
    }
    if !wait {
        println!("progress: llmctl-next op {} --watch", op.id);
        return Ok(0);
    }
    watch_and_report(&client, &op.id, DEFAULT_WATCH).await
}

async fn cmd_engine(cli: &Cli, name: &str, json: bool) -> Result<u8, CliError> {
    let client = connect(cli)?;
    let r = client.engine_report(name).await?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
        return Ok(0);
    }
    println!("{} ({}, {} protocol): {}", r.workload, r.engine_kind, r.protocol, r.state);
    if !r.servable.is_empty() {
        println!("  servable: {}", r.servable.join(", "));
    }
    match &r.loaded {
        Some(l) if l.is_empty() => println!("  loaded:   (none)"),
        Some(l) => println!("  loaded:   {}", l.join(", ")),
        None => {}
    }
    if r.supports_unload {
        println!("  unload:   llmctl-next unload {} <model>", r.workload);
    }
    Ok(if r.state == "ready" { 0 } else { 1 })
}

/// How long `--wait` watches by default.
const DEFAULT_WATCH: std::time::Duration = std::time::Duration::from_secs(10 * 60);

/// A watch long enough for this plan: every readiness wait it can run
/// (forward and rollback) plus a margin for stops and starts, never less
/// than the default. A Strata lane needs ~615 s to load and its workload
/// allows up to an hour; a caller that waits (llm-router's handoff) must
/// not give up while the operation is still on its way.
fn watch_budget(plan: &Plan) -> std::time::Duration {
    let readiness: u64 = plan
        .steps
        .iter()
        .chain(plan.rollback.iter())
        .map(|s| match s {
            llmctl_core::PlanStep::AwaitReadiness { timeout_secs, .. } => *timeout_secs,
            _ => 0,
        })
        .sum();
    DEFAULT_WATCH.max(std::time::Duration::from_secs(readiness + 300))
}

async fn watch_and_report(
    client: &ControlClient,
    id: &str,
    budget: std::time::Duration,
) -> Result<u8, CliError> {
    let mut last_signature = String::new();
    let final_op: Option<OperationView>;
    let deadline = tokio::time::Instant::now() + budget;
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
            "\nthis operation blocks new mutations until acknowledged: llmctl-next ack {}",
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

#[cfg(test)]
mod tests {
    use super::printable;

    #[test]
    fn the_remote_status_limit_defaults_to_ten_seconds_and_rejects_nonsense() {
        use super::parse_status_timeout as p;
        use std::time::Duration;
        assert_eq!(p(None), Duration::from_secs(10));
        assert_eq!(p(Some("3")), Duration::from_secs(3));
        assert_eq!(p(Some(" 25 ")), Duration::from_secs(25));
        assert_eq!(p(Some("0")), Duration::from_secs(10));
        assert_eq!(p(Some("soon")), Duration::from_secs(10));
    }

    #[test]
    fn control_characters_in_log_text_are_made_visible() {
        assert_eq!(printable("plain\ttext é"), "plain\ttext é");
        assert_eq!(
            printable("\u{1b}]0;pwned\u{7}\u{1b}[2Jok\rX\u{9b}31m"),
            "\\x1b]0;pwned\\x07\\x1b[2Jok\\x0dX\\x9b31m"
        );
    }

    #[test]
    fn a_waited_run_watches_for_as_long_as_its_plan_can_take() {
        use super::{watch_budget, DEFAULT_WATCH};
        use llmctl_core::{identity, OpKind, OperationId, Plan, PlanStep, UnitName, WorkloadName};
        let name = WorkloadName::new("strata-fast").unwrap();
        let unit = UnitName::systemd(identity::UnitScopeMarker::User, "x.service").unwrap();
        let wait = |secs| PlanStep::AwaitReadiness {
            workload: name.clone(),
            unit: unit.clone(),
            timeout_secs: secs,
        };
        let plan = |steps: Vec<PlanStep>, rollback: Vec<PlanStep>| Plan {
            id: OperationId::generate(),
            kind: OpKind::Run,
            target: name.clone(),
            revision: 1,
            expected_model: "m".into(),
            affected_gpus: vec![],
            steps,
            rollback,
        };
        // A two-minute start still gets the default ten minutes.
        assert_eq!(watch_budget(&plan(vec![wait(120)], vec![])), DEFAULT_WATCH);
        // An hour-long cold start, plus a rollback that waits too.
        assert_eq!(
            watch_budget(&plan(vec![wait(3600)], vec![wait(600)])),
            std::time::Duration::from_secs(3600 + 600 + 300)
        );
    }
}
