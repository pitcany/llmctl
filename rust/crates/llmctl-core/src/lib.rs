//! `llmctl-core` — typed identities, validated configuration, plans,
//! transitions and domain errors for the llmctl successor.
//!
//! House rule (RUST-SUCCESSOR-PLAN.md, workspace table): this crate makes no
//! system calls and touches no database. Everything here must be usable from
//! a pure test with fake data. Deserialized data must be re-validated before
//! use; unknown external state must stay representable.

pub mod acquire;
pub mod catalog;
pub mod error;
pub mod fit;
pub mod identity;
pub mod intent;
pub mod inventory;
pub mod launch;
pub mod matrix_edit;
pub mod observation;
pub mod plan;
pub mod planner;
pub mod preset_launch;
pub mod presets;
pub mod recipe;
pub mod swap;

pub use error::CoreError;
pub use identity::{
    GpuId, HostId, IdempotencyKey, OperationId, UnitName, UnitScopeMarker, WorkloadName,
};
pub use intent::DesiredIntent;
pub use inventory::{
    ControlAuthority, GpuClaim, Hook, HookWhen, HostInventory, Platform, RecoveryPolicy,
    UnitScope, WorkloadDefinition,
};
pub use launch::{
    looks_secret, EnvFileRef, ExecCommand, FieldDiff, FileProvenance, KeepAliveValue,
    LaunchDefinition, LaunchdExtras, Ownership, PendingApply, PreviousUnitFile, ServiceFileRecord,
    VersionFact, VersionRef,
};
pub use recipe::{
    EngineKind, ExternalRecipe, InstallRecipe, PythonEnvRecipe, ReleaseArchiveRecipe,
    SourceBuildRecipe, VendorAppRecipe, VendorProbe,
};
pub use observation::{Observation, ProcessState, Readiness, ServedModel};
pub use plan::{BinaryCheck, OpKind, Plan, PlanStep};
pub use acquire::{ModelStore, Source, StoreKind};
pub use fit::{CardMib, FitDecision};
pub use planner::{
    plan_restart, plan_restart_with, plan_run, plan_run_profile, plan_run_with, plan_stop,
    plan_unload, plan_upgrade, ProfileSwap,
};
