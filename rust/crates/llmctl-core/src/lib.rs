//! `llmctl-core` — typed identities, validated configuration, plans,
//! transitions and domain errors for the llmctl successor.
//!
//! House rule (RUST-SUCCESSOR-PLAN.md, workspace table): this crate makes no
//! system calls and touches no database. Everything here must be usable from
//! a pure test with fake data. Deserialized data must be re-validated before
//! use; unknown external state must stay representable.

pub mod error;
pub mod identity;
pub mod intent;
pub mod inventory;
pub mod launch;
pub mod observation;
pub mod plan;
pub mod planner;

pub use error::CoreError;
pub use identity::{
    GpuId, HostId, IdempotencyKey, OperationId, UnitName, UnitScopeMarker, WorkloadName,
};
pub use intent::DesiredIntent;
pub use inventory::{ControlAuthority, HostInventory, Platform, UnitScope, WorkloadDefinition};
pub use launch::{EnvFileRef, ExecCommand, FieldDiff, FileProvenance, LaunchDefinition, Ownership};
pub use observation::{Observation, ProcessState, Readiness, ServedModel};
pub use plan::{OpKind, Plan, PlanStep};
pub use planner::{plan_run, plan_stop};
