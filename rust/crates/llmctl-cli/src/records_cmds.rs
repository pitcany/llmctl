//! `enroll` and `authority` (Rust M5c item 5): records by verb for
//! workloads `adopt` cannot prove — previews until `--confirm <token>`.

use llmctl_client::wire::PROTOCOL_VERSION;
use llmctl_core::{ControlAuthority, UnitScope};
use llmctl_client::catalog::{AuthorityRequest, EnrollAuthority, EnrollRequest, RecordResponse};

use super::{connect, ensure_protocol, Cli, CliError};

/// `enroll` arguments (see the clap definition in main.rs).
#[derive(clap::Args, Clone)]
pub struct EnrollArgs {
    pub name: String,
    /// Observe only: the endpoint (`http://127.0.0.1:1234`). run/stop are refused.
    #[arg(long, conflicts_with_all = ["start", "stop"])]
    pub observe: Option<String>,
    /// A vendor app's start command (absolute program, then arguments;
    /// one shell-style string, split on spaces). Needs --stop and --port.
    #[arg(long, requires_all = ["stop", "port"])]
    pub start: Option<String>,
    #[arg(long, requires = "start")]
    pub stop: Option<String>,
    /// The vendor app's log file or log directory.
    #[arg(long, requires = "start")]
    pub log: Option<String>,
    /// What runs there: ollama, lm_studio, vllm, llama_cpp, mlx_serve, …
    #[arg(long)]
    pub engine_kind: String,
    /// What the endpoint serves.
    #[arg(long)]
    pub model: String,
    #[arg(long)]
    pub port: Option<u16>,
    /// GPU index (repeatable; default every GPU of the host).
    #[arg(long = "gpu")]
    pub gpus: Vec<u32>,
    /// Holds no memory until a request loads a model.
    #[arg(long)]
    pub on_demand: bool,
    /// Supervisor adapter override (tests: `fake`).
    #[arg(long, hide = true)]
    pub engine: Option<String>,
    #[arg(long)]
    pub confirm: Option<String>,
    #[arg(long)]
    pub json: bool,
}

/// `authority` arguments: exactly one target authority.
#[derive(clap::Args, Clone)]
pub struct AuthorityArgs {
    pub name: String,
    #[arg(long, group = "to")]
    pub observe: Option<String>,
    /// A systemd unit (`--scope user|system`, default user).
    #[arg(long, group = "to")]
    pub systemd_unit: Option<String>,
    #[arg(long, default_value = "user")]
    pub scope: String,
    #[arg(long, group = "to")]
    pub launchd: Option<String>,
    /// Vendor app commands (with --stop, optional --log).
    #[arg(long, group = "to", requires = "stop")]
    pub start: Option<String>,
    #[arg(long, requires = "start")]
    pub stop: Option<String>,
    #[arg(long, requires = "start")]
    pub log: Option<String>,
    #[arg(long)]
    pub confirm: Option<String>,
    #[arg(long)]
    pub json: bool,
}

fn argv(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_string).collect()
}

fn print(resp: &RecordResponse, json: bool, verb: &str) {
    if json {
        println!("{}", serde_json::to_string_pretty(resp).unwrap());
        return;
    }
    let r = &resp.record;
    println!(
        "{} {} ({} on :{}, {}), authority {}",
        if resp.applied { verb } else { "would write" },
        resp.workload,
        r.engine_kind,
        r.port,
        r.model,
        serde_json::to_string(&r.authority).unwrap_or_default()
    );
    if let Some(p) = &resp.previous {
        println!("  was {}", serde_json::to_string(p).unwrap_or_default());
    }
    if resp.applied {
        println!("  inventory revision {}", resp.inventory_revision);
    } else {
        println!("  write it: add --confirm {}", resp.confirm);
    }
}

pub async fn cmd_enroll(cli: &Cli, a: EnrollArgs) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let authority = match (&a.observe, &a.start, &a.stop) {
        (Some(endpoint), None, None) => EnrollAuthority::ObserveOnly { endpoint: endpoint.clone() },
        (None, Some(start), Some(stop)) => EnrollAuthority::VendorCommand { start: argv(start), stop: argv(stop), log: a.log.clone() },
        _ => return Err(CliError::Daemon("name how it is controlled: --observe URL, or --start/--stop".into())),
    };
    let req = EnrollRequest {
        protocol: PROTOCOL_VERSION,
        name: a.name,
        authority,
        engine_kind: a.engine_kind,
        model: a.model,
        port: a.port,
        gpus: a.gpus,
        on_demand: a.on_demand,
        engine: a.engine,
        confirm: a.confirm,
    };
    let resp = client.enroll_record(&req).await?;
    print(&resp, a.json, "enrolled");
    Ok(0)
}

pub async fn cmd_authority(cli: &Cli, a: AuthorityArgs) -> Result<u8, CliError> {
    let client = connect(cli)?;
    ensure_protocol(&client).await?;
    let authority = if let Some(endpoint) = &a.observe {
        ControlAuthority::ObserveOnly { endpoint: endpoint.clone() }
    } else if let Some(unit) = &a.systemd_unit {
        let scope = match a.scope.as_str() {
            "user" => UnitScope::User,
            "system" => UnitScope::System,
            other => return Err(CliError::Daemon(format!("--scope {other:?}: user or system"))),
        };
        ControlAuthority::SystemdUnit { scope, unit: unit.clone() }
    } else if let Some(label) = &a.launchd {
        ControlAuthority::LaunchdService { label: label.clone() }
    } else if let (Some(start), Some(stop)) = (&a.start, &a.stop) {
        ControlAuthority::VendorCommand { start: argv(start), stop: argv(stop), log: a.log.clone() }
    } else {
        return Err(CliError::Daemon("name the new authority: --observe, --systemd-unit, --launchd or --start/--stop".into()));
    };
    let req = AuthorityRequest { protocol: PROTOCOL_VERSION, name: a.name, authority, confirm: a.confirm };
    let resp = client.set_authority(&req).await?;
    print(&resp, a.json, "changed");
    Ok(0)
}
