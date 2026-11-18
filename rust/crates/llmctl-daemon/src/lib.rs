//! `llmctl-daemon` — the host agent and (on the workstation) coordinator
//! role for the llmctl Rust successor (see docs/RUST-SUCCESSOR-PLAN.md).
//!
//! Layout:
//! - [`db`]: SQLite operation journal + intent store behind one bounded
//!   worker thread (rusqlite). Every journal write is its own transaction;
//!   no transaction is held across an external effect.
//! - [`engine`]: the `Engine` port and the M1 file-backed fake engine whose
//!   `world.json` is the surviving external world for tests.
//! - [`ops`]: admission (idempotency, overlap rejection, stale-plan
//!   revalidation), phase execution with an explicit effect boundary,
//!   rollback/needs_attention handling, cancellation, acknowledgement and
//!   startup reconciliation.
//! - [`coord`]: the host-wide GPU operation lock and the lane lease shared
//!   with llm-router, Python llmctl and llm-heal (M4c).
//! - [`api`]: the versioned local HTTP/JSON API over the Unix socket.
//! - [`app`]: paths, inventory bootstrap, host identity, serve loop.

pub mod adopt;
pub mod api;
pub mod app;
pub mod coord;
pub mod db;
pub mod engine;
pub mod engine_api;
pub mod lanes;
pub mod gpu;
pub mod hooks;
pub mod inflight;
pub mod logfile;
pub mod ops;
pub mod own;
pub mod plistdef;
pub mod probe_http;
pub mod profile;
pub mod recover;
pub mod supervisor;
pub mod upgrade;
pub mod version;
pub mod vram;

// Platform managers (RUST-M3-PLAN A.2): systemd/journald exist only on
// Linux, launchd only on macOS. The fake engine, journal DB, API, CLI,
// plist mapping, log-file reader and HTTP probes are platform-neutral.
//
// `llmctl_macos` is a build-time cfg (declared in Cargo.toml) that selects
// the macOS code paths *on a Linux host* so they can be type-checked here
// (cross-compiling is impossible: bundled SQLite needs a Mac toolchain):
//     RUSTFLAGS='--cfg=llmctl_macos' cargo check -p llmctl-daemon
#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
pub mod journal;
#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
pub mod systemd;
#[cfg(all(unix, target_os = "linux", not(llmctl_macos)))]
pub mod unitdef;

#[cfg(any(target_os = "macos", llmctl_macos))]
pub mod launchd;

pub use app::{serve, AppState, DevPaths, DEFAULT_DATA_DIR_SUFFIX};

/// Entry point used by the `llmctl-next daemon` CLI subcommand.
pub async fn run_daemon(
    data_dir: Option<std::path::PathBuf>,
    socket: Option<std::path::PathBuf>,
    host_id: Option<String>,
) -> Result<(), String> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let data = app::DevPaths::resolve(data_dir);
    let paths = app::DevPaths::new(data, socket);
    let state = AppState::open(paths, host_id).await?;
    serve(state).await
}
