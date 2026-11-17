//! `llmctl-client` — wire DTOs (`wire`) and the Unix-socket control client
//! (`client`). The `llmctl-next` binary lives in the `llmctl-cli` crate so
//! the daemon (which depends on these DTOs) and the CLI (which depends on
//! the daemon) never form a dependency cycle.

pub mod bridge;
pub mod client;
pub mod wire;

pub use client::{ClientError, ControlClient};
