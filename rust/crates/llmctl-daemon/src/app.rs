//! AppState assembly, dev data layout, inventory bootstrap and the serve
//! loop (Unix socket, user-private). The dev data directory is isolated
//! from the Python llmctl install by construction.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use llmctl_core::{
    ControlAuthority, GpuId, HostId, HostInventory, Platform, UnitName, UnitScope, UnitScopeMarker,
    WorkloadDefinition, WorkloadName,
};

use crate::api;
use crate::db::Db;
use crate::engine::{Engine, FakeEngine};
use crate::ops::OpManager;

pub const DEFAULT_DATA_DIR_SUFFIX: &str = "llmctl-next-dev";

#[derive(Debug, Clone)]
pub struct DevPaths {
    pub data_dir: PathBuf,
    pub db: PathBuf,
    pub world: PathBuf,
    pub faults: PathBuf,
    pub inventory: PathBuf,
    pub socket: PathBuf,
}

impl DevPaths {
    pub fn resolve(data_dir: Option<PathBuf>) -> PathBuf {
        data_dir.unwrap_or_else(default_data_dir)
    }

    pub fn new(data_dir: impl Into<PathBuf>, socket: Option<PathBuf>) -> Self {
        let data_dir = data_dir.into();
        Self {
            socket: socket.unwrap_or_else(|| data_dir.join("control.sock")),
            db: data_dir.join("ops.sqlite3"),
            world: data_dir.join("world.json"),
            faults: data_dir.join("faults.json"),
            inventory: data_dir.join("inventory.json"),
            data_dir,
        }
    }
}

pub fn default_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LLMCTL_NEXT_DATA_DIR") {
        return PathBuf::from(dir);
    }
    dirs_state_home().join(DEFAULT_DATA_DIR_SUFFIX)
}

fn dirs_state_home() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."));
            home.join(".local").join("state")
        })
}

pub struct AppState {
    pub host: HostId,
    pub paths: DevPaths,
    pub started_at: DateTime<Utc>,
    pub ops: Arc<OpManager>,
}

/// Sample M1 inventory: fake-engine workloads only, nothing that touches
/// the real inference stack. Written to `inventory.json` on first start;
/// afterwards the file is authoritative (edit + restart, or future API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryFile {
    pub inventory: HostInventory,
}

pub fn sample_inventory(host: &HostId) -> HostInventory {
    let mk = |name: &str, gpus: &[u32], port: u16, label: String| WorkloadDefinition {
        name: WorkloadName::new(name).expect("sample name"),
        engine: "fake".into(),
        model: format!("fake/{name}"),
        gpus: gpus.iter().map(|i| GpuId::new(host.clone(), *i)).collect(),
        port,
        unit: UnitName::systemd(UnitScopeMarker::User, label).expect("sample unit"),
        authority: ControlAuthority::SystemdUnit {
            scope: UnitScope::User,
            unit: format!("llmctl-next-fake-{name}.service"),
        },
    };
    let mut inv = HostInventory {
        host: host.clone(),
        platform: Platform::Linux,
        gpus: vec![GpuId::new(host.clone(), 0), GpuId::new(host.clone(), 1)],
        workloads: vec![
            mk("coder", &[0], 8100, "llmctl-next-fake-coder.service".into()),
            mk(
                "helper",
                &[1],
                8101,
                "llmctl-next-fake-helper.service".into(),
            ),
            mk("big", &[0, 1], 8102, "llmctl-next-fake-big.service".into()),
        ],
        revision: 1,
    };
    // Platform is detected per host rather than hardcoded once Macs enroll.
    if let Some(platform) = Platform::current() {
        inv.platform = platform;
    }
    inv
}

impl AppState {
    /// Opens the journal, loads (or bootstraps) the inventory, binds the
    /// host identity, runs startup reconciliation and assembles the state.
    pub async fn open(paths: DevPaths) -> Result<Arc<Self>, String> {
        tokio::fs::create_dir_all(&paths.data_dir)
            .await
            .map_err(|e| format!("creating data dir {}: {e}", paths.data_dir.display()))?;

        let db = Db::open(&paths.db).map_err(|e| format!("opening journal: {e}"))?;

        // Stable host identity, bound once.
        let host_str = db
            .meta_get("host_id")
            .await
            .map_err(|e| e.to_string())?
            .unwrap_or_else(|| "workstation".to_string());
        db.meta_set("host_id", &host_str)
            .await
            .map_err(|e| e.to_string())?;
        let host = HostId::new(&host_str).map_err(|e| e.to_string())?;

        // Inventory: file is authoritative; sample written when absent.
        let inv = match tokio::fs::read_to_string(&paths.inventory).await {
            Ok(text) => {
                let file: InventoryFile = serde_json::from_str(&text)
                    .map_err(|e| format!("parsing {}: {e}", paths.inventory.display()))?;
                file.inventory
            }
            Err(_) => {
                let inv = sample_inventory(&host);
                let file = InventoryFile {
                    inventory: inv.clone(),
                };
                let text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
                tokio::fs::write(&paths.inventory, text)
                    .await
                    .map_err(|e| format!("writing {}: {e}", paths.inventory.display()))?;
                inv
            }
        };
        inv.validate()
            .map_err(|e| format!("inventory invalid: {e}"))?;

        let engine: Arc<dyn Engine> = Arc::new(FakeEngine::new(&paths.data_dir));
        let ops = Arc::new(OpManager::new(
            db,
            engine,
            Arc::new(inv),
            Some(paths.faults.clone()),
        ));

        // Startup reconciliation: finalize anything a previous daemon left
        // nonterminal, honestly, before serving a single mutation.
        let resolved = ops
            .reconcile_interrupted()
            .await
            .map_err(|e| format!("reconciliation failed: {e}"))?;
        if !resolved.is_empty() {
            tracing::warn!(
                operations = ?resolved,
                "reconciled interrupted operations from previous daemon run"
            );
        }

        Ok(Arc::new(Self {
            host,
            paths,
            started_at: Utc::now(),
            ops,
        }))
    }
}

/// Bind the Unix control socket with user-only permissions and serve until
/// shutdown. A stale socket file (daemon not listening) is removed; a live
/// one refuses to start — one active agent per host.
pub async fn serve(state: Arc<AppState>) -> Result<(), String> {
    let socket_path = state.paths.socket.clone();
    if socket_path.exists() {
        // Probe: is another daemon live on this socket?
        match tokio::net::UnixStream::connect(&socket_path).await {
            Ok(_) => {
                return Err(format!(
                    "another daemon is already listening on {}; refusing to start a second",
                    socket_path.display()
                ))
            }
            Err(_) => {
                tokio::fs::remove_file(&socket_path)
                    .await
                    .map_err(|e| format!("removing stale socket: {e}"))?;
            }
        }
    }
    if let Some(parent) = socket_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("creating socket dir: {e}"))?;
    }
    let listener = tokio::net::UnixListener::bind(&socket_path)
        .map_err(|e| format!("binding {}: {e}", socket_path.display()))?;
    use std::os::unix::fs::PermissionsExt;
    tokio::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))
        .await
        .map_err(|e| format!("setting socket permissions: {e}"))?;

    let app = api::router(state);
    tracing::info!(socket = %socket_path.display(), "llmctl-next daemon serving");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| format!("server error: {e}"))?;
    let _ = tokio::fs::remove_file(&socket_path).await;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
