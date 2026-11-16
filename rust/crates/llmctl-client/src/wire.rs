//! Wire DTOs for the versioned local control API (HTTP/JSON over a Unix
//! socket). Shared by the CLI and the daemon. Mutations carry a protocol
//! version; the daemon rejects unsupported versions explicitly before any
//! side effect.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use llmctl_core::{OpKind, Plan};

/// Protocol version for mutation requests. Bump on incompatible change;
/// the daemon refuses mismatched versions with `protocol_mismatch`.
pub const PROTOCOL_VERSION: u32 = 1;

// ---- views ----------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub protocol: u32,
    pub supported_protocols: Vec<u32>,
    pub daemon_version: String,
    pub host: String,
    pub pid: u32,
    pub started_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadView {
    pub name: String,
    pub engine: String,
    pub model: String,
    pub gpus: Vec<String>,
    pub port: u16,
    pub unit: String,
    /// Last accepted desired intent (durable), distinct from observation.
    pub intent: String,
    pub process: String,
    pub readiness: String,
    /// Served model identity when known, with verification flag.
    pub served_model: Option<String>,
    pub served_model_verified: bool,
    pub observed_at: Option<DateTime<Utc>>,
    /// True when the observation could not be refreshed (stale/unknown).
    pub observation_unknown: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub host: String,
    pub platform: String,
    pub inventory_revision: u64,
    pub generated_at: DateTime<Utc>,
    pub workloads: Vec<WorkloadView>,
}

// ---- planning -------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    pub kind: OpKind,
    pub workload: String,
}

/// Dry-run response: the plan the daemon would execute now, against fresh
/// observations. Offline plans are estimates; this one is revalidated again
/// at admission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanResponse {
    pub revision: u64,
    /// Workloads this plan would stop (conflicts + target restart), the
    /// destructive footprint a confirmation refers to.
    pub stop_targets: Vec<String>,
    pub plan: Plan,
}

// ---- operations -----------------------------------------------------------

/// Admission request. `idempotency_key` is host-scoped and caller-chosen
/// before sending: retrying a lost acknowledgement reuses it and queries the
/// same operation instead of spawning another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmitRequest {
    pub protocol: u32,
    pub idempotency_key: String,
    pub kind: OpKind,
    pub workload: String,
    /// Inventory revision the caller's confirmation was built against.
    pub expected_revision: Option<u64>,
    /// Destructive footprint the caller confirmed. If the fresh plan's
    /// footprint differs, admission returns a revised plan instead of
    /// executing an obsolete confirmation.
    pub expected_stop_targets: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmitResponse {
    pub operation: OperationView,
    /// True when an existing operation was returned for the same key.
    pub replayed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpStatus {
    /// Durablely admitted; not yet executing.
    Admitted,
    Executing,
    /// Cancellation requested; executor stops before the next external
    /// effect. Not successful until the executor records the outcome.
    CancelRequested,
    Succeeded,
    Failed,
    /// Owner died mid-operation; could not tell what happened.
    Interrupted,
    /// The world was left different from where it started (failed rollback
    /// or unresolvable crash window). A human must look.
    NeedsAttention,
    Canceled,
}

impl OpStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            OpStatus::Admitted => "admitted",
            OpStatus::Executing => "executing",
            OpStatus::CancelRequested => "cancel_requested",
            OpStatus::Succeeded => "succeeded",
            OpStatus::Failed => "failed",
            OpStatus::Interrupted => "interrupted",
            OpStatus::NeedsAttention => "needs_attention",
            OpStatus::Canceled => "canceled",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            OpStatus::Succeeded
                | OpStatus::Failed
                | OpStatus::Interrupted
                | OpStatus::NeedsAttention
                | OpStatus::Canceled
        )
    }
}

impl std::fmt::Display for OpStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseStatus {
    Pending,
    Running,
    /// External effect issued; result not yet written. The crash window.
    EffectPending,
    Done,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseView {
    pub index: usize,
    pub label: String,
    pub status: PhaseStatus,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationView {
    pub id: String,
    pub idempotency_key: String,
    pub kind: OpKind,
    pub workload: String,
    pub status: OpStatus,
    pub phases: Vec<PhaseView>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// Final result details (error text, timing) for terminal operations.
    pub result: Option<serde_json::Value>,
    /// Restoration outcome, recorded separately from the original failure:
    /// a model serving again after a failed switch appears here, never by
    /// relabeling the failure.
    pub recovery: Option<serde_json::Value>,
    /// Human has seen a needs_attention/interrupted outcome; unblocks
    /// mutations blocked by this operation.
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationListResponse {
    pub operations: Vec<OperationView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AckRequest {
    pub note: Option<String>,
}

// ---- errors ---------------------------------------------------------------

/// Machine-matchable error body: `{"error": {"code": ..., "message": ...}}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiErrorBody {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

pub mod codes {
    pub const PROTOCOL_MISMATCH: &str = "protocol_mismatch";
    pub const INVALID_REQUEST: &str = "invalid_request";
    pub const UNKNOWN_WORKLOAD: &str = "unknown_workload";
    pub const STALE_PLAN: &str = "stale_plan";
    pub const OVERLAPPING_OPERATION: &str = "overlapping_operation";
    pub const IDEMPOTENCY_CONFLICT: &str = "idempotency_conflict";
    pub const BLOCKED_NEEDS_ATTENTION: &str = "blocked_needs_attention";
    pub const UNKNOWN_OPERATION: &str = "unknown_operation";
    pub const INVALID_STATE: &str = "invalid_state";
    pub const DAEMON_ERROR: &str = "daemon_error";
}
