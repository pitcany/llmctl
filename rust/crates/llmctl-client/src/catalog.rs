//! Wire DTOs and client calls for presets and the model catalog (Rust M5c).
//! Kept out of `wire.rs`/`client.rs` so parallel milestones touch
//! different files. The TUI (M5e) reads these shapes: list/show answers
//! are complete on their own (no second call needed to draw a row).

use serde::{Deserialize, Serialize};

use crate::{ClientError, ControlClient};

/// Error codes this surface adds (the shared ones live in `wire::codes`).
pub mod codes {
    pub const UNKNOWN_PRESET: &str = "unknown_preset";
    pub const PRESET_REFUSED: &str = "preset_refused";
    /// The preset's router profile env does not render what the preset
    /// says; `details.drift` lists the keys. `--sync` writes it.
    pub const PRESET_DRIFT: &str = "preset_drift";
    pub const CATALOG_REFUSED: &str = "catalog_refused";
    pub const UNKNOWN_MODEL: &str = "unknown_model";
}

// ---- presets -----------------------------------------------------------------

/// One key whose value differs between a profile env and its preset's
/// render: `(key, in the file, in the render)`; `None` = absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvDrift {
    pub key: String,
    pub file: Option<String>,
    pub render: Option<String>,
}

/// How a preset reaches a workload: the router profile env that carries
/// its render, and whether it is the one serving now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetLink {
    /// The vLLM workload this preset runs on.
    pub workload: String,
    /// `<env>.<profile>` next to the workload's env file; `None` when no
    /// file carries this preset yet (`--sync` creates `<env>.<alias>`).
    pub profile_env: Option<String>,
    /// How the profile was found: `router` (router.yaml's backend for the
    /// served name), `header` (its generated-from line), or `new`.
    pub found_by: String,
    /// Keys where the profile file differs from the render.
    pub drift: Vec<EnvDrift>,
    /// The workload's active env renders this preset and it serves the
    /// preset's served name.
    pub active: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetSummary {
    pub alias: String,
    pub served_name: String,
    pub model_id: String,
    pub quantization: String,
    pub tensor_parallel_size: i64,
    pub max_model_len: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_count_b: Option<f64>,
    /// The file it was read from (a symlink is shown as found).
    pub source_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<PresetLink>,
    /// Why the link could not be worked out (no vLLM workload, several).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link_error: Option<String>,
}

/// A preset file that does not load (listed, never hidden).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetInvalid {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetListResponse {
    /// Directories read, lowest precedence first.
    pub dirs: Vec<String>,
    pub presets: Vec<PresetSummary>,
    pub invalid: Vec<PresetInvalid>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetShowResponse {
    pub summary: PresetSummary,
    /// The file's mapping as loaded (JSON).
    pub fields: serde_json::Value,
    /// The file's bytes (presets hold no secrets).
    pub text: String,
    /// The env body this preset renders for its workload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rendered_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_error: Option<String>,
}

/// Change a preset's keys in place (comments kept). `set` values are YAML
/// text (`16`, `null`, `"0.9"`, `[a, b]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetEditRequest {
    pub protocol: u32,
    pub alias: String,
    #[serde(default)]
    pub set: Vec<(String, String)>,
    #[serde(default)]
    pub unset: Vec<String>,
    #[serde(default)]
    pub dry_run: bool,
    /// The digest a preview answered with: refuse if the file changed since.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expect_digest: Option<String>,
}

/// A new preset: a copy of `from` (its comments too) or bare `set` keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetNewRequest {
    pub protocol: u32,
    pub alias: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(default)]
    pub set: Vec<(String, String)>,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetWriteResponse {
    pub alias: String,
    pub path: String,
    /// `sha256:` of the file before the write (`missing` for a new one).
    pub before_digest: String,
    pub after_digest: String,
    /// Keys whose value changes: `(key, before, after)` as YAML text.
    pub changes: Vec<(String, Option<String>, Option<String>)>,
    /// The new text (what was, or with `dry_run` would be, written).
    pub text: String,
    /// False when the comments could not be kept (full re-dump).
    pub comments_kept: bool,
    pub written: bool,
    /// Where the previous bytes were kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backup: Option<String>,
}

/// Resolve a preset for `run <preset>`; with `sync`, first bring its
/// profile env in line with the render (values merged, comments kept).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetRunRequest {
    pub protocol: u32,
    pub alias: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload: Option<String>,
    #[serde(default)]
    pub sync: bool,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PresetRunResponse {
    pub alias: String,
    pub workload: String,
    /// The served name: the `--model` of the run.
    pub model: String,
    /// The profile to switch to; `None` when the active env already
    /// renders this preset (an ordinary run, no restart for a comment).
    pub profile_env: Option<String>,
    /// The profile file was written (sync) or would be (`dry_run`).
    pub synced: bool,
    pub drift: Vec<EnvDrift>,
}

/// Rewrite the comments that name the engine's version so they say what
/// the workload's pin says (the pin is the truth).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetSettleRequest {
    pub protocol: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload: Option<String>,
    #[serde(default)]
    pub write: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettleFile {
    pub path: String,
    pub changes: Vec<llmctl_core::preset_launch::CommentChange>,
    pub written: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetSettleResponse {
    pub workload: String,
    pub pin: llmctl_core::preset_launch::EnginePin,
    pub files: Vec<SettleFile>,
    /// Files with no matching comment, read and left alone.
    pub untouched: Vec<String>,
}

fn esc(s: &str) -> String {
    // Aliases are `[a-z0-9_.-]`; anything else is refused by the agent,
    // but never let a slash change the route.
    s.replace('/', "%2F")
}

impl ControlClient {
    pub async fn presets(&self) -> Result<PresetListResponse, ClientError> {
        self.get_json("/v1/presets").await
    }

    pub async fn preset(&self, alias: &str) -> Result<PresetShowResponse, ClientError> {
        self.get_json(&format!("/v1/presets/{}", esc(alias))).await
    }

    pub async fn preset_edit(&self, req: &PresetEditRequest) -> Result<PresetWriteResponse, ClientError> {
        self.post_json("/v1/presets/edit", req).await
    }

    pub async fn preset_new(&self, req: &PresetNewRequest) -> Result<PresetWriteResponse, ClientError> {
        self.post_json("/v1/presets/new", req).await
    }

    pub async fn preset_run(&self, req: &PresetRunRequest) -> Result<PresetRunResponse, ClientError> {
        self.post_json("/v1/presets/run", req).await
    }

    pub async fn preset_settle(
        &self,
        req: &PresetSettleRequest,
    ) -> Result<PresetSettleResponse, ClientError> {
        self.post_json("/v1/presets/settle", req).await
    }
}

// ---- model catalog ---------------------------------------------------------------

pub use llmctl_core::catalog::{CatalogModel, RowOutcome, RowReport};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogListResponse {
    pub host: String,
    /// Names the registry knows this host by (from imports).
    pub host_names: Vec<String>,
    pub revision: u64,
    pub models: Vec<CatalogModel>,
    /// Records hidden because they are disabled (without `all`).
    pub disabled_hidden: usize,
    /// Bundle rows kept verbatim for the export but not records here.
    pub unmapped: usize,
    pub profiles: usize,
}

/// Import a registry bundle (`llmctl export-registry`). Without `confirm`
/// it only reports; with the report's `confirm` token it applies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogImportRequest {
    pub protocol: u32,
    pub bundle: serde_json::Value,
    /// Registry host names that mean this host (`--as yannik-desktop`).
    #[serde(default)]
    pub host_names: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogImportResponse {
    pub rows: Vec<RowReport>,
    /// Counts by outcome: new, no_store, present, unmapped.
    pub counts: std::collections::BTreeMap<String, usize>,
    pub profiles: usize,
    /// Pass back as `confirm` to apply exactly this report.
    pub confirm: String,
    pub applied: bool,
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAddRequest {
    pub protocol: u32,
    pub name: String,
    pub engine: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<String>,
    /// Other fields, values as YAML text.
    #[serde(default)]
    pub set: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEditRequest {
    pub protocol: u32,
    /// Id, or a name that is unique on this host.
    pub model: String,
    #[serde(default)]
    pub set: Vec<(String, String)>,
    #[serde(default)]
    pub unset: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEnableRequest {
    pub protocol: u32,
    pub model: String,
    pub enabled: bool,
}

/// Delete one catalog record (M5 gap 3). Without `confirm` it is a preview:
/// the record, what still names it, and the token that applies exactly this.
/// The weights are not touched (`model rm` removes weights).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDeleteRequest {
    pub protocol: u32,
    /// Id, or a name that is unique on this host.
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDeleteResponse {
    pub model: CatalogModel,
    /// What still names the record (`workload <name>`, `preset <alias>`).
    /// A delete is refused while this is not empty.
    pub references: Vec<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The token for `confirm` (bound to the record and the catalog revision).
    pub confirm: String,
    pub deleted: bool,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelWriteResponse {
    pub model: CatalogModel,
    /// `(field, before, after)` as JSON text.
    pub changes: Vec<(String, Option<String>, Option<String>)>,
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanRequest {
    pub protocol: u32,
    /// One store (default: every declared store).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<String>,
    /// Add what is found and not cataloged as records.
    #[serde(default)]
    pub add: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScanFound {
    pub store: String,
    pub engine: String,
    /// What the engine calls it (repo, tag, key, path).
    pub source: String,
    pub path: String,
    /// The record that already points at it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScanResponse {
    pub stores: Vec<String>,
    pub found: Vec<ScanFound>,
    /// Records in a scanned store whose weights are not there.
    pub missing: Vec<CatalogModel>,
    pub added: usize,
    pub revision: u64,
}

impl ControlClient {
    pub async fn catalog_models(&self, all: bool) -> Result<CatalogListResponse, ClientError> {
        self.get_json(if all { "/v1/catalog/models?all=1" } else { "/v1/catalog/models" }).await
    }

    pub async fn catalog_model(&self, model: &str) -> Result<CatalogModel, ClientError> {
        let q: String = model
            .bytes()
            .map(|b| match b {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
                _ => format!("%{b:02X}"),
            })
            .collect();
        self.get_json(&format!("/v1/catalog/model?m={q}")).await
    }

    pub async fn catalog_import(&self, req: &CatalogImportRequest) -> Result<CatalogImportResponse, ClientError> {
        self.post_json("/v1/catalog/import", req).await
    }

    pub async fn catalog_export(&self) -> Result<serde_json::Value, ClientError> {
        self.get_json("/v1/catalog/export").await
    }

    pub async fn model_add(&self, req: &ModelAddRequest) -> Result<ModelWriteResponse, ClientError> {
        self.post_json("/v1/catalog/models/add", req).await
    }

    pub async fn model_edit(&self, req: &ModelEditRequest) -> Result<ModelWriteResponse, ClientError> {
        self.post_json("/v1/catalog/models/edit", req).await
    }

    pub async fn model_enable(&self, req: &ModelEnableRequest) -> Result<ModelWriteResponse, ClientError> {
        self.post_json("/v1/catalog/models/enable", req).await
    }

    pub async fn model_delete(&self, req: &ModelDeleteRequest) -> Result<ModelDeleteResponse, ClientError> {
        self.post_json("/v1/catalog/models/delete", req).await
    }

    pub async fn model_scan(&self, req: &ScanRequest) -> Result<ScanResponse, ClientError> {
        self.post_json("/v1/catalog/scan", req).await
    }
}

// ---- records by verb (M5c item 5) ---------------------------------------------

/// How a new record is controlled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EnrollAuthority {
    /// Watched, never moved: `run`/`stop` are refused.
    ObserveOnly { endpoint: String },
    /// A Mac vendor app's own controls (absolute programs).
    VendorCommand {
        start: Vec<String>,
        stop: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        log: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollRequest {
    pub protocol: u32,
    pub name: String,
    pub authority: EnrollAuthority,
    /// What runs there (`ollama`, `lm_studio`, `vllm`, …).
    pub engine_kind: String,
    /// What its endpoint must serve.
    pub model: String,
    /// The port it answers on (observe-only: taken from the endpoint).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// GPU indexes; default every GPU of the host.
    #[serde(default)]
    pub gpus: Vec<u32>,
    /// Holds no VRAM until a request loads a model (vendor services).
    #[serde(default)]
    pub on_demand: bool,
    /// The supervisor adapter (default: `systemd` on Linux, `launchd` on
    /// a Mac; tests use `fake`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorityRequest {
    pub protocol: u32,
    pub name: String,
    pub authority: llmctl_core::ControlAuthority,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confirm: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordResponse {
    pub workload: String,
    /// The record as it is (or with `confirm` was) written.
    pub record: llmctl_core::WorkloadDefinition,
    /// The authority before (`authority` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<llmctl_core::ControlAuthority>,
    /// Pass back as `confirm` to write exactly this.
    pub confirm: String,
    pub applied: bool,
    pub inventory_revision: u64,
}

impl ControlClient {
    pub async fn enroll_record(&self, req: &EnrollRequest) -> Result<RecordResponse, ClientError> {
        self.post_json("/v1/records/enroll", req).await
    }

    pub async fn set_authority(&self, req: &AuthorityRequest) -> Result<RecordResponse, ClientError> {
        self.post_json("/v1/records/authority", req).await
    }
}
