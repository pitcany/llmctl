//! Wire DTOs and client calls for llama.cpp (M5d): router models
//! (import, list, edit, load/unload, facts) and the Mac llama.cpp lane
//! engine install. Kept apart from `wire.rs` so the M5 work streams touch
//! different files.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use llmctl_core::llamacpp::{ModelRecord, ModelState, RouterModel};

use crate::client::{ClientError, ControlClient};
use crate::wire::{AdmitResponse, OperationView};

/// Import a router's preset into agent records: preview without
/// `confirm`, persist with the preview's `records_digest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterImportRequest {
    pub protocol: u32,
    pub workload: String,
    /// Default: `--models-preset` from the workload's pinned launch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterImportResponse {
    pub workload: String,
    pub preset_path: String,
    pub file_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub records_digest: Option<String>,
    /// `equivalent`, or why the round trip failed.
    pub roundtrip: String,
    pub validation: Vec<String>,
    /// What is wrong with the router's own command line (autoload, -m…).
    pub argv_problems: Vec<String>,
    /// Said when the command line could not be checked (not managed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub argv_note: Option<String>,
    pub models: Vec<ModelRecord>,
    pub notes_kept: usize,
    pub persisted: bool,
}

/// llmctl-only facts about one router model (not in the INI).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelFacts {
    /// MiB per card it holds once loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub need_mib: Option<BTreeMap<u32, u64>>,
    /// `measured <time>` or `declared <time>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub need_source: Option<String>,
    /// Host RAM its CPU offload takes (GiB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_ram_gib: Option<f64>,
    /// Peak memory on a unified-memory host (GiB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_gib: Option<f64>,
    /// Free VRAM every card must have before a load (MiB), the router's
    /// version of `LLAMA_GUARD_MIN_FREE_MIB`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_free_mib: Option<u64>,
}

/// One router model: its record, what the router says now, its facts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterModelView {
    pub record: ModelRecord,
    /// `None` = the router did not answer or does not list it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<ModelState>,
    #[serde(default)]
    pub failed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
    pub facts: ModelFacts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterModelsResponse {
    pub workload: String,
    pub preset_path: String,
    pub revision: u64,
    pub drift: bool,
    pub router_answers: bool,
    pub models_max: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_gib: Option<u64>,
    pub models: Vec<RouterModelView>,
    /// Models the router lists that the records do not hold.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unknown_listed: Vec<String>,
    pub validation: Vec<String>,
    /// Model hooks declared in `<data>/llamacpp/<workload>.hooks.json`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hooks: Vec<String>,
}

/// One change to the preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum RouterEdit {
    Set { model: String, key: String, value: String },
    Remove { model: String, key: String },
    AddModel { model: RouterModel },
    RemoveModel { id: String },
}

/// Plan (no `confirm`) or apply (with the plan's digest) preset edits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterEditRequest {
    pub protocol: u32,
    pub workload: String,
    pub edits: Vec<RouterEdit>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterPlanResponse {
    pub workload: String,
    pub summary: String,
    pub changes: Vec<String>,
    pub validation: Vec<String>,
    /// Loaded models the reload unloads (changed or removed sections).
    pub reload_unloads: Vec<String>,
    pub plan_digest: String,
    pub bytes_changed: bool,
}

/// Set llmctl-only facts. `None` leaves a fact as it is; `clear` lists
/// facts to drop (`need`, `host_ram`, `peak`, `min_free`, `budget`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterFactsRequest {
    pub protocol: u32,
    pub workload: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub need_mib: Option<BTreeMap<u32, u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_ram_gib: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_gib: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_free_mib: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_gib: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clear: Vec<String>,
}

/// Load (or, with `unload`, unload) one router model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterLoadRequest {
    pub protocol: u32,
    pub workload: String,
    pub model: String,
    #[serde(default)]
    pub unload: bool,
    #[serde(default)]
    pub dry_run: bool,
    /// Stop the at-start workloads on the router's cards first when the
    /// model does not fit beside them (they come back if the load fails).
    #[serde(default)]
    pub evict: bool,
    /// Do not wait for requests in flight.
    #[serde(default)]
    pub force: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterLoadResponse {
    pub workload: String,
    pub model: String,
    pub unload: bool,
    /// `load`, `already_loaded`, `refuse`, `unload`, `not_loaded`.
    pub decision: String,
    pub reason: String,
    /// Models that hold memory on the router now.
    pub loaded_now: Vec<String>,
    /// Pre-load guard findings (a load is refused on any).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preload: Vec<String>,
    /// Workloads an `evict` load stops first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evicts: Vec<String>,
    /// Hooks that run with it (`post_load ds4-adopt`, …).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hooks: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<OperationView>,
    #[serde(default)]
    pub replayed: bool,
}

/// Stage one llama.cpp release into its own versioned dir for a llama-swap
/// workload's lanes (M5d item 5). The lane macro is not changed: the
/// answer names the `lanes macro` write that switches to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneEngineInstallRequest {
    pub protocol: u32,
    pub workload: String,
    /// The macro that names the binary (`llamacpp`).
    pub macro_name: String,
    pub version: String,
    /// Saved for this macro; later installs reuse it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<llmctl_core::ReleaseArchiveRecipe>,
    /// `sha256:<hex>` of this version's archive, added to the recipe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneEngineInstallResponse {
    pub workload: String,
    pub macro_name: String,
    pub version: String,
    pub dir: String,
    pub executable: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub archive_sha256: Option<String>,
    pub validation: serde_json::Value,
    /// What the macro names now.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<String>,
    /// The write that switches the lanes to it (re-pins them).
    pub next: String,
    /// The write that goes back, once switched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback: Option<String>,
}

impl ControlClient {
    pub async fn llamacpp_install(
        &self,
        req: &LaneEngineInstallRequest,
    ) -> Result<LaneEngineInstallResponse, ClientError> {
        self.post_json("/v1/llamacpp/install", req).await
    }

    pub async fn llamacpp_import(&self, req: &RouterImportRequest) -> Result<RouterImportResponse, ClientError> {
        self.post_json("/v1/llamacpp/import", req).await
    }

    pub async fn llamacpp_models(&self, workload: &str) -> Result<RouterModelsResponse, ClientError> {
        self.get_json(&format!("/v1/llamacpp/{workload}/models")).await
    }

    /// Without `confirm`: the plan.
    pub async fn llamacpp_plan(&self, req: &RouterEditRequest) -> Result<RouterPlanResponse, ClientError> {
        self.post_json("/v1/llamacpp/edit", req).await
    }

    /// With the plan's digest as `confirm`: the admitted write.
    pub async fn llamacpp_apply(&self, req: &RouterEditRequest) -> Result<AdmitResponse, ClientError> {
        self.post_json("/v1/llamacpp/edit", req).await
    }

    pub async fn llamacpp_facts(&self, req: &RouterFactsRequest) -> Result<RouterModelsResponse, ClientError> {
        self.post_json("/v1/llamacpp/facts", req).await
    }

    pub async fn llamacpp_load(&self, req: &RouterLoadRequest) -> Result<RouterLoadResponse, ClientError> {
        self.post_json("/v1/llamacpp/loads", req).await
    }
}
