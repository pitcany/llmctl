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
