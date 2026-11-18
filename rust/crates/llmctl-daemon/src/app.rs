//! AppState assembly, dev data layout, inventory bootstrap and the serve
//! loop (Unix socket, user-private). The dev data directory is isolated
//! from the Python llmctl install by construction.

use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use llmctl_core::{
    ControlAuthority, GpuId, HostId, HostInventory, Platform, UnitName, WorkloadDefinition,
    WorkloadName,
};

use crate::api;
use crate::db::Db;
use crate::engine::{Engine, FakeEngine, RoutedEngine};
use crate::ops::OpManager;
use crate::supervisor::Supervisor;

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
    /// The platform supervisor (systemd on Linux, launchd on macOS),
    /// reachable for adoption's read-only probes and ownership flows.
    pub supervisor: Arc<dyn Supervisor>,
    /// Serializes adoption previews' scratch-unit round trips with transfers.
    pub adopt_lock: tokio::sync::Mutex<()>,
}

/// Sample M1 inventory: fake-engine workloads only, nothing that touches
/// the real inference stack. Written to `inventory.json` on first start;
/// afterwards the file is authoritative (edit + restart, or future API).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InventoryFile {
    pub inventory: HostInventory,
}

pub fn sample_inventory(host: &HostId) -> HostInventory {
    // The dev sample keeps its original systemd-shaped unit identities on
    // every platform: the fake engine keys on that display string and
    // never talks to a manager, and the e2e suite (faults/world/gpus
    // injections) matches these strings. Real Mac agents get their own
    // inventory written at enrollment.
    let mk = |name: &str, gpus: &[u32], port: u16| WorkloadDefinition {
        name: WorkloadName::new(name).expect("sample name"),
        engine: "fake".into(),
        engine_kind: llmctl_core::EngineKind::FakeEngine,
        model: format!("fake/{name}"),
        gpus: gpus.iter().map(|i| GpuId::new(host.clone(), *i)).collect(),
        port,
        unit: sample_unit(name),
        authority: sample_authority(name),
        ownership: Default::default(),
        recovery: None,
        recipe: None,
        readiness_timeout_secs: None,
        gpu_claim: Default::default(),
        hooks: Vec::new(),
        pin_files: Vec::new(),
        observed_version: None,
    };
    let mut inv = HostInventory {
        host: host.clone(),
        platform: Platform::Linux,
        gpus: vec![GpuId::new(host.clone(), 0), GpuId::new(host.clone(), 1)],
        workloads: vec![
            mk("coder", &[0], 8100),
            mk("helper", &[1], 8101),
            mk("big", &[0, 1], 8102),
        ],
        stores: Vec::new(),
        revision: 1,
    };
    // Platform is detected per host rather than hardcoded once Macs enroll.
    if let Some(platform) = Platform::current() {
        inv.platform = platform;
    }
    inv
}

fn sample_unit(name: &str) -> UnitName {
    use llmctl_core::UnitScopeMarker;
    UnitName::systemd(
        UnitScopeMarker::User,
        format!("llmctl-next-fake-{name}.service"),
    )
    .expect("sample unit")
}

fn sample_authority(name: &str) -> ControlAuthority {
    use llmctl_core::UnitScope;
    ControlAuthority::SystemdUnit {
        scope: UnitScope::User,
        unit: format!("llmctl-next-fake-{name}.service"),
    }
}

impl AppState {
    /// Opens the journal, loads (or bootstraps) the inventory, binds the
    /// host identity, runs startup reconciliation and assembles the state.
    /// `host_id`: the name this agent is enrolled under (`host add` passes
    /// it). Without it the bound name stays, or `workstation` on first
    /// start.
    pub async fn open(paths: DevPaths, host_id: Option<String>) -> Result<Arc<Self>, String> {
        tokio::fs::create_dir_all(&paths.data_dir)
            .await
            .map_err(|e| format!("creating data dir {}: {e}", paths.data_dir.display()))?;

        let db = Db::open(&paths.db).map_err(|e| format!("opening journal: {e}"))?;

        // Stable host identity: bound once, renamed only when enrollment
        // names the agent (every agent used to stay `workstation`).
        let bound = db.meta_get("host_id").await.map_err(|e| e.to_string())?;
        let host_str = host_id
            .clone()
            .or_else(|| bound.clone())
            .unwrap_or_else(|| "workstation".to_string());
        let host = HostId::new(&host_str).map_err(|e| format!("host id {host_str:?}: {e}"))?;
        if bound.as_deref() != Some(host_str.as_str()) {
            db.meta_set("host_id", &host_str)
                .await
                .map_err(|e| e.to_string())?;
            if let Some(old) = &bound {
                tracing::info!(from = %old, to = %host_str, "host id renamed");
            }
        }

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
        // M4a migration: fill legacy engine kinds (a structural
        // normalization — the revision does not move).
        let inv = if let Some(migrated) = inv.migrated() {
            crate::engine::write_json_atomic(
                &paths.inventory,
                &InventoryFile {
                    inventory: migrated.clone(),
                },
            )
            .await
            .map_err(|e| format!("writing {}: {e}", paths.inventory.display()))?;
            migrated
        } else {
            inv
        };
        // An explicitly named agent owns its inventory's host name too
        // (mbp2's carried the drill harness's `drill-mac`).
        let inv = if host_id.is_some() && inv.host != host {
            let renamed = inv.renamed_host(&host);
            renamed
                .validate()
                .map_err(|e| format!("inventory invalid after host rename: {e}"))?;
            crate::engine::write_json_atomic(
                &paths.inventory,
                &InventoryFile {
                    inventory: renamed.clone(),
                },
            )
            .await
            .map_err(|e| format!("writing {}: {e}", paths.inventory.display()))?;
            tracing::info!(from = %inv.host, to = %host, revision = renamed.revision, "inventory host renamed");
            renamed
        } else {
            inv
        };
        inv.validate()
            .map_err(|e| format!("inventory invalid: {e}"))?;

        // Each workload names its adapter; only the platform manager's
        // workloads reach a real service manager, and only their own
        // enrolled units.
        let mut engines: std::collections::BTreeMap<String, Arc<dyn Engine>> = Default::default();
        engines.insert("fake".into(), Arc::new(FakeEngine::new(&paths.data_dir)));
        let supervisor: Arc<dyn Supervisor> = platform_supervisor(&inv);
        engines.insert(supervisor.engine_name().into(), supervisor_manager_engine(&inv));
        let engine: Arc<dyn Engine> = Arc::new(RoutedEngine::new(&inv, engines)?);
        let ops = Arc::new(
            OpManager::new(db, engine, Arc::new(inv), Some(paths.faults.clone())).with_own_ctx(
                crate::own::OwnCtx {
                    paths: paths.clone(),
                    supervisor: supervisor.clone(),
                },
            ),
        );

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

        // Re-pin legacy definition schemas: an old pin (schema 1) gets its
        // version facts probed from the live env and recorded (schema 2),
        // so it is never reported as false drift — the current world is
        // declared the baseline.
        if let Err(e) = repin_legacy_versions(&ops, &paths.inventory).await {
            tracing::warn!(error = %e, "version re-pin skipped");
        }

        Ok(Arc::new(Self {
            host,
            paths,
            started_at: Utc::now(),
            ops,
            supervisor,
            adopt_lock: tokio::sync::Mutex::new(()),
        }))
    }
}

/// The platform's supervisor + manager engine: systemd on Linux, launchd on
/// macOS. The sample inventory only enrolls fake workloads, so either
/// platform starts green without touching the real stack.
#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
pub fn manager_engine_name() -> &'static str {
    crate::systemd::ENGINE_NAME
}

#[cfg(any(target_os = "macos", llmctl_macos))]
pub fn manager_engine_name() -> &'static str {
    crate::launchd::ENGINE_NAME
}

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
fn platform_supervisor(inv: &HostInventory) -> Arc<dyn Supervisor> {
    Arc::new(crate::supervisor::SystemdSupervisor::new(
        crate::systemd::SystemdEngine::from_inventory(inv),
    ))
}

#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
fn supervisor_manager_engine(inv: &HostInventory) -> Arc<dyn Engine> {
    Arc::new(crate::systemd::SystemdEngine::from_inventory(inv))
}

#[cfg(any(target_os = "macos", llmctl_macos))]
fn platform_supervisor(inv: &HostInventory) -> Arc<dyn Supervisor> {
    Arc::new(crate::supervisor::LaunchdSupervisor::new(
        crate::launchd::LaunchdEngine::from_inventory(inv),
    ))
}

#[cfg(any(target_os = "macos", llmctl_macos))]
fn supervisor_manager_engine(inv: &HostInventory) -> Arc<dyn Engine> {
    Arc::new(crate::launchd::LaunchdEngine::from_inventory(inv))
}

/// Pin version facts for managed manager-engine workloads whose pin
/// predates schema 2 (python-env facts) or schema 3 (per-kind facts for
/// the M4a recipe kinds), writing the inventory file back (revision
/// unchanged: an idempotent normalization, not a semantic change).
async fn repin_legacy_versions(
    ops: &Arc<OpManager>,
    inventory_path: &std::path::Path,
) -> Result<(), String> {
    let inv = ops.inventory().await;
    let mut changed = false;
    let mut next = (*inv).clone();
    for i in 0..next.workloads.len() {
        let needs = {
            let w = &next.workloads[i];
            matches!(
                &w.ownership,
                llmctl_core::Ownership::Managed {
                    definition_schema: s,
                    version: v,
                    ..
                } if *s < 2 && v.is_none()
            ) && w.engine == manager_engine_name() && w.recipe.is_some()
        };
        let needs_m4 = {
            let w = &next.workloads[i];
            matches!(
                &w.ownership,
                llmctl_core::Ownership::Managed {
                    definition_schema: s,
                    version: v,
                    ..
                // No pin, or only adoption's program-digest pin (no
                // per-kind fact): the recipe was attached after adoption.
                } if *s < 3 && v.as_ref().is_none_or(|v| v.fact.is_none())
            ) && w.engine == manager_engine_name()
                && matches!(
                    &w.recipe,
                    Some(llmctl_core::InstallRecipe::ReleaseArchive(_))
                        | Some(llmctl_core::InstallRecipe::SourceBuild(_))
                        | Some(llmctl_core::InstallRecipe::VendorApp(_))
                )
        };
        // A recipe-backed install llmctl-next does not manage (ds4 on
        // m5): its facts are recorded on the record itself.
        let needs_unmanaged = {
            let w = &next.workloads[i];
            matches!(w.ownership, llmctl_core::Ownership::External)
                && w.observed_version.is_none()
                && w.engine == manager_engine_name()
                && (!w.pin_files.is_empty()
                    || matches!(
                        &w.recipe,
                        Some(llmctl_core::InstallRecipe::ReleaseArchive(_))
                            | Some(llmctl_core::InstallRecipe::SourceBuild(_))
                            | Some(llmctl_core::InstallRecipe::VendorApp(_))
                    ))
        };
        if !needs && !needs_m4 && !needs_unmanaged {
            continue;
        }
        let recipe = next.workloads[i].recipe.clone().expect("checked");
        match crate::version::probe(&recipe, &next.workloads[i]).await {
            Ok(v) => {
                let before = next.workloads[i].clone();
                let w = &mut next.workloads[i];
                let complete = crate::pin::apply_pin(w, v);
                if *w != before {
                    let pin = crate::pin::pinned_version(w);
                    tracing::info!(
                        workload = %w.name,
                        complete,
                        packages = ?pin.map(|p| &p.packages),
                        fact = ?pin.and_then(|p| p.fact.as_ref()),
                        "pinned version facts"
                    );
                    changed = true;
                }
            }
            Err(e) => {
                // No baseline possible: stay at the old schema rather
                // than pin something unprobed.
                let name = next.workloads[i].name.to_string();
                tracing::warn!(workload = %name, error = %e, "version probe failed; pin stays unmigrated");
            }
        }
    }
    if changed {
        let text = serde_json::to_string_pretty(&InventoryFile {
            inventory: next.clone(),
        })
        .map_err(|e| e.to_string())?;
        tokio::fs::write(inventory_path, text)
            .await
            .map_err(|e| format!("writing {}: {e}", inventory_path.display()))?;
        ops.update_inventory(Arc::new(next)).await;
    }
    Ok(())
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

    let app = api::router(state.clone());
    crate::recover::spawn(state);
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
