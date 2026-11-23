//! Wire DTOs for the versioned local control API (HTTP/JSON over a Unix
//! socket). Shared by the CLI and the daemon. Mutations carry a protocol
//! version; the daemon rejects unsupported versions explicitly before any
//! side effect.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use llmctl_core::{FileProvenance, LaunchDefinition, OpKind, Plan};

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
    /// `external` or `managed` (adopted, definition pinned).
    #[serde(default = "external")]
    pub ownership: String,
    /// Pinned definition digest for managed workloads.
    #[serde(default)]
    pub adopted_digest: Option<String>,
    /// What the manager says about a down unit (`alive`, `manager_retrying`,
    /// `gave_up`, `idle`, `unknown`) — the recovery decision input, shown
    /// as-is while systemd retries on its own.
    #[serde(default)]
    pub recovery_state: String,
    /// Recovery budget when configured, `max/window_secs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recovery_policy: Option<String>,
}

fn external() -> String {
    "external".into()
}

// ---- adoption ---------------------------------------------------------------

/// Preview (no `confirm_digest`) or transfer ownership of an existing user
/// unit. Transfer requires the digest from a preview: if the unit changed
/// since, the daemon answers `stale_adoption` with a fresh preview.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdoptRequest {
    pub protocol: u32,
    /// Bare user unit name, e.g. `qwen-coder.service`.
    pub unit: String,
    pub workload: String,
    pub model: String,
    pub port: u16,
    /// GPU indices on this host.
    pub gpus: Vec<u32>,
    #[serde(default)]
    pub confirm_digest: Option<String>,
    /// Transfer even if served identity could not be checked (unit stopped).
    #[serde(default)]
    pub allow_unverified: bool,
    /// What the unit runs (`ollama`, `lm_studio`, `llama_cpp`, ...): picks
    /// the readiness/identity protocol (M4b). Absent → legacy, inferred
    /// later from a recipe; probed as an OpenAI-style server.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_kind: Option<String>,
    /// `at_start` or `on_demand` (M4c). Absent → the kind's default:
    /// on_demand for Ollama and LM Studio, at_start otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_claim: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Fail,
    Unverified,
    /// Does not block this step, but a later one will refuse (e.g. `own`).
    Warn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdoptionCheck {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdoptionPreview {
    pub unit: String,
    pub workload: String,
    /// Digest to pass as `confirm_digest`; absent when nothing was imported.
    pub digest: Option<String>,
    pub definition: Option<LaunchDefinition>,
    /// The service definition llmctl would own.
    pub generated_unit: Option<String>,
    pub provenance: Vec<FileProvenance>,
    pub checks: Vec<AdoptionCheck>,
    pub inventory_revision: u64,
}

impl AdoptionPreview {
    pub fn has_failures(&self) -> bool {
        self.checks.iter().any(|c| c.status == CheckStatus::Fail)
    }

    pub fn has_unverified(&self) -> bool {
        self.checks
            .iter()
            .any(|c| c.status == CheckStatus::Unverified)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdoptResponse {
    pub preview: AdoptionPreview,
    pub inventory_revision: u64,
}

// ---- ownership (own / release) ----------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnRequest {
    pub workload: String,
}

/// What llmctl now owns after `own`: the file it wrote and what was there
/// before (the release target).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnResponse {
    pub workload: String,
    pub path: String,
    pub digest: String,
    /// `file` (bytes backed up) or `link` (re-created on release).
    pub previous_kind: String,
    pub inventory_revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseRequest {
    pub workload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseResponse {
    pub workload: String,
    /// `external` — ownership handed back, original restored.
    pub ownership: String,
    pub inventory_revision: u64,
}

// ---- staged upgrades ---------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRequest {
    pub workload: String,
    /// Target version of the recipe's primary package.
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResponse {
    pub workload: String,
    pub version: String,
    pub env: String,
    pub interpreter: String,
    pub status: String,
    pub validation: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromoteRequest {
    pub workload: String,
    pub version: String,
    /// Wait for the operation to finish before answering.
    #[serde(default)]
    pub wait: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackRequest {
    pub workload: String,
    #[serde(default)]
    pub wait: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeListResponse {
    pub upgrades: Vec<UpgradeRowView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpgradeRowView {
    pub id: String,
    pub workload: String,
    pub from_version: String,
    pub to_version: String,
    pub env_path: String,
    pub interpreter: String,
    pub status: String,
    pub detail: Option<serde_json::Value>,
    pub operation_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcResponse {
    pub workload: String,
    pub removed: Vec<String>,
    pub kept: Vec<String>,
}

// ---- installs and version reports (M4a) -------------------------------------

/// The vendor-app plan request: what runs, what is pinned, the drift
/// between them, and the owner-run backup/reinstall steps. llmctl never
/// executes any of it.
/// Attach or replace a workload's install recipe (`recipe` present) or
/// remove it (`recipe` absent). `pin_files` absent keeps the list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeSetRequest {
    pub protocol: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin_files: Option<Vec<String>>,
}

/// Remove a workload's record. Without `confirm` it only says what would
/// go and what stays on the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetRequest {
    pub protocol: u32,
    #[serde(default)]
    pub confirm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallPlanRequest {
    pub workload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionReportRequest {
    pub workload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub host: String,
    pub platform: String,
    pub inventory_revision: u64,
    pub generated_at: DateTime<Utc>,
    pub workloads: Vec<WorkloadView>,
    /// Per card: enrolled workloads (observed running) vs everyone else.
    #[serde(default)]
    pub gpus: Vec<GpuView>,
}

/// One foreign compute process on a card.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOccupantView {
    pub pid: u32,
    /// Owning unit when it maps to one; `None` = unmappable (always foreign).
    pub unit: Option<String>,
    pub memory_mib: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuView {
    pub index: u32,
    /// Enrolled workloads observed running on this card.
    pub enrolled: Vec<String>,
    /// Processes on the card that no enrolled workload accounts for.
    pub foreign: Vec<GpuOccupantView>,
    /// Set when occupancy could not be determined at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ---- planning -------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    pub kind: OpKind,
    pub workload: String,
    /// Run only: switch the workload to this env-file profile (M4c).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<ProfileRequest>,
}

/// An env-file profile switch (M4c): make `env_source` the workload's
/// `EnvironmentFile` and expect it to serve `model` afterwards. The file
/// must sit next to the env file it replaces; its contents never travel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileRequest {
    pub env_source: String,
    pub model: String,
}

/// Recorded intents, without probing anything (M4c): what the healers
/// (llm-heal, mac-heal, llmctl-warm) ask before they restart or load a
/// workload. `manual_stop` = the owner stopped it through llmctl-next;
/// hands off.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentsResponse {
    pub host: String,
    pub intents: Vec<IntentView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentView {
    pub workload: String,
    /// Bare unit or launchd label name, as the service manager knows it.
    pub unit: String,
    /// `running`, `stopped`, `unmanaged`, or `none` (never set).
    pub intent: String,
    pub manual_stop: bool,
}

/// One foreign-write finding (M5a, decision 1): a managed workload's
/// state changed and no admitted operation explains it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditFinding {
    pub id: i64,
    pub host: String,
    pub workload: String,
    /// Always `foreign_transition` in M5.
    pub kind: String,
    /// What changed: `unit`, `lanes`, or `unit+lanes`.
    pub what: String,
    /// Comparison states as compact JSON (`process`, `pid`, `model`,
    /// `lanes`).
    pub before_state: String,
    pub after_state: String,
    /// Who did it when a name exists (`llama-swap (ttl: …)`); `None` =
    /// unknown.
    pub actor: Option<String>,
    pub observed_at: DateTime<Utc>,
}

/// `GET /v1/audit?since=<rfc3339>` — the pilot's "no unexplained foreign
/// writes" evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResponse {
    pub host: String,
    pub findings: Vec<AuditFinding>,
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
    /// What the plan rested on that its steps do not show: the VRAM-fit
    /// decision and its reason (M4c).
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Declared model stores with their free space (M4c item 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoresResponse {
    pub host: String,
    pub stores: Vec<StoreView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreView {
    pub name: String,
    pub kind: String,
    pub path: String,
    pub reserve_gib: u64,
    pub workload: Option<String>,
    pub exists: bool,
    /// Free bytes on the store's filesystem; `None` when it cannot be read.
    pub free_bytes: Option<u64>,
}

/// Declare a model store (nothing is created on disk).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreAddRequest {
    pub protocol: u32,
    pub name: String,
    /// `hf_cache`, `directory`, `ollama` or `lmstudio`.
    pub kind: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reserve_gib: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreRemoveRequest {
    pub protocol: u32,
    pub name: String,
}

/// `tolerate add`: let one known process stay on the cards (M5b). Exactly
/// one of `unit` (full identity) or `cmdline` (substring) is set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TolerateAddRequest {
    pub protocol: u32,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cmdline: Option<String>,
    pub max_mib: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TolerateRemoveRequest {
    pub protocol: u32,
    pub name: String,
}

/// The processes the GPU guard tolerates on this host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleratedResponse {
    pub host: String,
    pub tolerated: Vec<llmctl_core::GpuTolerance>,
}

/// Remove weights from a store. Without `confirm` it only previews: the
/// path, the size, and whoever still references them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRemoveRequest {
    pub protocol: u32,
    /// `hf:org/name`, `ollama:tag`, or `path:<relative path in the store>`.
    pub target: String,
    pub store: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRemoveResponse {
    pub target: String,
    pub store: String,
    /// The directory or file that goes (absent for an Ollama tag).
    pub path: Option<String>,
    pub bytes: Option<u64>,
    /// Who still names these weights. Not empty = nothing is removed.
    pub references: Vec<String>,
    /// The token that confirms this preview.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
    pub removed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub freed_bytes: Option<u64>,
}

/// Download weights into a store. `dry_run` only checks and previews.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquireRequest {
    pub protocol: u32,
    /// `hf:org/name[@revision]`, `ollama:tag` or `lms:publisher/model[@variant]`.
    pub source: String,
    pub store: String,
    /// The workload whose journal this is filed under (default: the
    /// store's own).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub for_workload: Option<String>,
    /// An upper bound in GiB when the source cannot say its size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_gib: Option<u64>,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquireResponse {
    pub source: String,
    pub store: String,
    pub path: String,
    pub workload: String,
    pub need_bytes: Option<u64>,
    pub free_bytes: u64,
    pub reserve_gib: u64,
    /// The command the operation runs.
    pub command: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<OperationView>,
    #[serde(default)]
    pub replayed: bool,
}

/// Measured VRAM per workload and the cards' free memory (M4c VRAM fit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramResponse {
    /// Free MiB per card now; `None` when the cards cannot be read.
    pub free_mib: Option<std::collections::BTreeMap<u32, u64>>,
    pub records: Vec<VramRecordView>,
    /// Set by a measure request: what was just recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<VramRecordView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramRecordView {
    pub workload: String,
    pub model: String,
    pub per_card_mib: std::collections::BTreeMap<u32, u64>,
    pub measured_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramMeasureRequest {
    pub workload: String,
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
    /// Stop serving workloads even while llm-router has requests in flight
    /// on them (M4c request safety). Without it the agent waits a bounded
    /// time, then refuses with `requests_in_flight`.
    #[serde(default)]
    pub force: bool,
    /// Run only: switch the workload to this env-file profile (M4c).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<ProfileRequest>,
}

// ---- llama-swap lane records (M4b step 4) ----------------------------------------

/// Import a llama-swap workload's config into lane records. Without
/// `confirm` it only previews; with the preview's `records_digest` it
/// persists the records (the config file is never written by import).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneImportRequest {
    pub protocol: u32,
    pub workload: String,
    /// Absent → the `-config` argument of the pinned launch definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    /// Absent → the plist's stdout path. Used to confirm reloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneSummary {
    pub id: String,
    pub engine_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<i64>,
    /// The pinned version, named in one line (`mtplx 2.11.0`); absent
    /// when the lane has no pin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned: Option<String>,
}

/// Version facts per lane: the pin, what the lane's command runs now,
/// and the drift between them. With `pin`, the live facts become the pin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneVersionRequest {
    pub protocol: u32,
    pub workload: String,
    /// Absent → every lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane: Option<String>,
    #[serde(default)]
    pub pin: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneVersion {
    pub id: String,
    pub engine_kind: String,
    /// The live facts named in one line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub pinned: Option<llmctl_core::VersionRef>,
    pub live: Option<llmctl_core::VersionRef>,
    pub drift: Vec<llmctl_core::FieldDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneVersionResponse {
    pub workload: String,
    pub lanes: Vec<LaneVersion>,
    /// Lanes whose pin this call wrote or changed.
    pub pinned_now: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneImportResponse {
    pub workload: String,
    pub config_path: String,
    pub log_path: Option<String>,
    pub file_digest: String,
    /// The confirmation token; `None` when the round trip failed.
    pub records_digest: Option<String>,
    /// `equivalent`, or why the render does not round-trip.
    pub roundtrip: String,
    /// What llama-swap would refuse (empty = valid).
    pub validation: Vec<String>,
    pub lanes: Vec<LaneSummary>,
    pub notes_kept: usize,
    pub persisted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneStatusResponse {
    pub workload: String,
    pub config_path: String,
    pub log_path: Option<String>,
    pub revision: u64,
    pub file_digest: String,
    pub file_digest_now: String,
    /// The file no longer matches the records (edited by hand).
    pub drift: bool,
    pub lanes: Vec<LaneSummary>,
    pub macros: Vec<(String, String)>,
    pub validation: Vec<String>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    /// The memory budget against the matrix (`lanes memory`); empty when
    /// no budget is set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory: Vec<String>,
    /// The `matrix:` records (M5c item 4); `None` without a matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix: Option<llmctl_core::swap::Matrix>,
}

/// The memory budget of a llama-swap host and its lanes' peak figures.
/// Every field that is set changes the records; with none it only reads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneMemoryRequest {
    pub protocol: u32,
    pub workload: String,
    /// GiB of model peaks the host can hold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_gib: Option<u64>,
    /// `(lane, peak GiB)`; `None` removes the lane's figure.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub peaks: Vec<(String, Option<f64>)>,
    /// A model registry (YAML, records with `id` and
    /// `*_peak_memory_*_gb` fields) to take peak figures from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_registry: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneMemoryResponse {
    pub workload: String,
    pub budget_gib: Option<u64>,
    /// Peak GiB per lane, as recorded.
    pub peaks: std::collections::BTreeMap<String, f64>,
    pub lines: Vec<String>,
    /// Sets over the budget (not `risky*`).
    pub over: Vec<(String, u64)>,
    pub changed: bool,
}

/// One change to the lane records.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum LaneEdit {
    SetField {
        lane: String,
        field: String,
        value: serde_json::Value,
    },
    RemoveField {
        lane: String,
        field: String,
    },
    SetMacro {
        name: String,
        value: String,
    },
    /// A change to the `matrix:` (M5c item 4). Guarded by the memory
    /// budget: refused without one, and for a set that would go over it or
    /// holds a lane with no peak figure.
    Matrix {
        edit: llmctl_core::matrix_edit::MatrixEdit,
    },
    /// A new lane with its fields in llama-swap's order (M5d: a llama.cpp
    /// lane rendered from a model record).
    AddLane {
        lane: String,
        fields: Vec<(String, serde_json::Value)>,
    },
    RemoveLane {
        lane: String,
    },
}

/// Plan (no `confirm`) or apply (with the plan's digest) lane edits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneEditRequest {
    pub protocol: u32,
    pub workload: String,
    pub edits: Vec<LaneEdit>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

/// Load one lane on a llama-swap workload (M4c). `dry_run` only predicts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneSwapRequest {
    pub protocol: u32,
    pub workload: String,
    pub lane: String,
    #[serde(default)]
    pub dry_run: bool,
    /// Evict even while llama-swap has requests in flight.
    #[serde(default)]
    pub force: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    /// `lanes stop`: unload `lane` instead of loading it. Nothing else is
    /// evicted; `evicts` in the answer is the lane itself when it runs.
    #[serde(default)]
    pub unload: bool,
}

/// The newest lines of one lane's own output, as llama-swap keeps them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneLogsRequest {
    pub protocol: u32,
    pub workload: String,
    pub lane: String,
    pub lines: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneLogsResponse {
    pub workload: String,
    pub lane: String,
    pub lines: Vec<String>,
    /// Lines llama-swap had kept for the lane, before the cut to `lines`.
    pub available: usize,
}

/// What the lane records predict for the swap, and the operation when it
/// was admitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneSwapResponse {
    pub workload: String,
    pub lane: String,
    /// Lanes llama-swap has running now.
    pub running: Vec<String>,
    /// Of those, the ones the records say llama-swap evicts.
    pub evicts: Vec<String>,
    /// The matrix set it settles on (empty without a matrix).
    pub set: String,
    pub cost: i64,
    pub already_loaded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<OperationView>,
    #[serde(default)]
    pub replayed: bool,
    /// The request was a stop (`unload`), not a load.
    #[serde(default)]
    pub unload: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanePlanResponse {
    pub workload: String,
    pub summary: String,
    /// Value paths that change (`/models/x/ttl: 3600 != 600`).
    pub changes: Vec<String>,
    pub validation: Vec<String>,
    /// Lanes running now: llama-swap's reload restarts them.
    pub reload_restarts: Vec<String>,
    /// Lanes the new config preloads (`hooks.on_startup.preload`): a
    /// reload starts loading them.
    #[serde(default)]
    pub reload_preloads: Vec<String>,
    /// Consequences worth reading that do not block the write (a lane
    /// outside every matrix set, which runs alone).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    pub plan_digest: String,
    /// False when the render equals the live file (nothing to write).
    pub bytes_changed: bool,
}

/// Engine-level unload of one model from a vendor service (M4b step 1).
/// Admitted like any operation; the key makes a retry safe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnloadRequest {
    pub protocol: u32,
    pub idempotency_key: String,
    pub workload: String,
    pub model: String,
}

/// Engine-level load of one model onto a running vendor service (M5c item
/// 4, the mirror of [`UnloadRequest`]). `keep_alive_secs` is how long the
/// engine may keep the model resident (-1 = until unloaded); `force` skips
/// only the VRAM fit check, never the catalog check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadRequest {
    pub protocol: u32,
    pub idempotency_key: String,
    pub workload: String,
    pub model: String,
    #[serde(default = "default_keep_alive")]
    pub keep_alive_secs: i64,
    #[serde(default)]
    pub force: bool,
}

/// Keep until unloaded — what a preload for llm-router's readiness check
/// needs (the qwen3coder30b-start script's choice).
fn default_keep_alive() -> i64 {
    -1
}

/// What a workload's engine reports right now (M4b step 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineReport {
    pub workload: String,
    pub engine_kind: String,
    /// `openai`, `ollama` or `lmstudio`.
    pub protocol: String,
    /// `ready`, `unreachable`, `status <code>` or `garbled`.
    pub state: String,
    /// Models the engine can serve (vendor services: pulled/downloaded).
    pub servable: Vec<String>,
    /// Models in memory now; `None` when the protocol cannot say.
    pub loaded: Option<Vec<String>>,
    pub supports_unload: bool,
    /// The `load` verb can preload one model (M5c item 4; Ollama today).
    #[serde(default)]
    pub supports_load: bool,
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

// ---- logs -----------------------------------------------------------------

/// Upper bound on entries per logs request.
pub const MAX_LOG_LINES: u32 = 1000;
/// Default entries per logs request.
pub const DEFAULT_LOG_LINES: u32 = 100;

/// Read a workload's service-manager log. Without `after`: the newest
/// `lines` entries. With `after` (a cursor from a previous response): the
/// next entries after it, oldest first, so a follower pages without gaps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsRequest {
    pub workload: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lines: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogSource {
    /// The workload's own stdout/stderr.
    Unit,
    /// The service manager about the unit (started, exited, failed).
    Manager,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    /// syslog priority 0 (emerg) ..= 7 (debug), when recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<u8>,
    pub source: LogSource,
    /// One start of the unit; changes mark run boundaries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invocation: Option<String>,
    /// Lossily decoded to UTF-8 and capped; `truncated` says when cut.
    pub message: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// The line carried no time of its own: `timestamp` is the log file's
    /// last write, an upper bound shared by every line of the page.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub time_estimated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsResponse {
    pub workload: String,
    pub unit: String,
    pub entries: Vec<LogEntry>,
    /// Pass back as `after` to continue; the request's cursor when no new
    /// entries arrived, `None` only when the log is empty.
    pub cursor: Option<String>,
    /// The page is full: more entries may already be waiting.
    pub more: bool,
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
    pub const DEFINITION_DRIFT: &str = "definition_drift";
    pub const GPU_OCCUPIED: &str = "gpu_occupied";
    pub const GPU_PROBE_FAILED: &str = "gpu_probe_failed";
    pub const STALE_ADOPTION: &str = "stale_adoption";
    pub const ADOPTION_REFUSED: &str = "adoption_refused";
    pub const ADOPTION_UNVERIFIED: &str = "adoption_unverified";
    pub const OWNERSHIP_REFUSED: &str = "ownership_refused";
    pub const UPGRADE_REFUSED: &str = "upgrade_refused";
    pub const UNSUPPORTED: &str = "unsupported";
    pub const LANES_REFUSED: &str = "lanes_refused";
    /// Another controller (llm-router, Python llmctl, llm-heal) holds the
    /// host-wide GPU operation lock (M4c).
    pub const HOST_LOCK_BUSY: &str = "host_lock_busy";
    /// A tool holds a fresh lane lease this operation would disturb (M4c).
    pub const LANE_LEASED: &str = "lane_leased";
    /// llm-router has requests in flight on a workload this operation
    /// would stop; retry, or force it (M4c).
    pub const REQUESTS_IN_FLIGHT: &str = "requests_in_flight";
    /// A vendor `load` was refused before anything ran (not in the catalog,
    /// no need ever measured, or the cards cannot hold it) — M5c item 4.
    pub const LOAD_REFUSED: &str = "load_refused";
}
