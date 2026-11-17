//! Engine kinds and install recipes (RUST-M4-PLAN, design decision 1):
//! every managed engine instance has an [`InstallRecipe`] of one of five
//! kinds, and every kind installs into a new versioned directory, never in
//! place. Recipes are data — the daemon maps a kind to an installer; the
//! core stays free of system calls.
//!
//! Legacy inventories (pre-M4) carry a flat python-env recipe without a
//! `kind` tag; deserialization maps those to
//! [`InstallRecipe::PythonEnv`] so old files load unchanged (see the
//! migration tests in `inventory.rs`).

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};

/// What a workload runs, independent of which supervisor adapter controls
/// it (adapters are per supervisor: systemd, launchd, fake). The kind
/// names the engine for reports and future per-engine adapters (M4b);
/// install behaviour comes from the recipe kind.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    Vllm,
    LlamaCpp,
    Ollama,
    LmStudio,
    Mtplx,
    MlxVlm,
    MlxServe,
    Strata,
    Ds4,
    FakeEngine,
    #[serde(untagged)]
    Other(String),
}

impl EngineKind {
    /// The placeholder stored when an inventory predates `engine_kind`.
    pub fn legacy() -> Self {
        EngineKind::Other("legacy".into())
    }

    pub fn is_legacy(&self) -> bool {
        matches!(self, EngineKind::Other(s) if s == "legacy")
    }

    /// Best-effort inference from a recipe/engine name, used when migrating
    /// old inventories. Unknown names stay `Other(name)` — representable,
    /// never an error.
    pub fn infer(name: &str) -> Self {
        let k = name.trim().to_ascii_lowercase().replace(['_', '-', '.'], "");
        match k.as_str() {
            "vllm" => EngineKind::Vllm,
            "llamacpp" | "llamaserver" | "llamacppserver" => EngineKind::LlamaCpp,
            "ollama" => EngineKind::Ollama,
            "lmstudio" | "lms" | "llmster" => EngineKind::LmStudio,
            "mtplx" => EngineKind::Mtplx,
            "mlxvlm" => EngineKind::MlxVlm,
            "mlxserve" => EngineKind::MlxServe,
            "strata" => EngineKind::Strata,
            "ds4" | "ds4server" => EngineKind::Ds4,
            "fake" | "fakeengine" => EngineKind::FakeEngine,
            _ => EngineKind::Other(name.trim().to_string()),
        }
    }
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineKind::Vllm => write!(f, "vllm"),
            EngineKind::LlamaCpp => write!(f, "llama_cpp"),
            EngineKind::Ollama => write!(f, "ollama"),
            EngineKind::LmStudio => write!(f, "lm_studio"),
            EngineKind::Mtplx => write!(f, "mtplx"),
            EngineKind::MlxVlm => write!(f, "mlx_vlm"),
            EngineKind::MlxServe => write!(f, "mlx_serve"),
            EngineKind::Strata => write!(f, "strata"),
            EngineKind::Ds4 => write!(f, "ds4"),
            EngineKind::FakeEngine => write!(f, "fake_engine"),
            EngineKind::Other(s) => write!(f, "{s}"),
        }
    }
}

/// How a workload's software is installed and versioned. One installer
/// model, five recipe kinds (owner decision 1):
///
/// - [`InstallRecipe::PythonEnv`] — `uv venv` + pinned packages (vLLM,
///   MTPLX, mlx-vlm, Strata's venv).
/// - [`InstallRecipe::ReleaseArchive`] — a released archive (llama.cpp
///   prebuilt, mlx-serve, Strata's prebuilt engine, `llmster`), unpacked
///   to `<root>/<version-dir>/`.
/// - [`InstallRecipe::SourceBuild`] — git checkout + build command (ds4,
///   llama.cpp from source); the build runs in a staging dir and the
///   binary is checked before the directory is promoted.
/// - [`InstallRecipe::VendorApp`] — vendor-installed app (Ollama, LM
///   Studio): llmctl records and pins the version it observes and owns
///   the service, but install/upgrade is an explicit owner-run step with
///   a tested backup/reinstall path. No silent vendor auto-update counts
///   as an upgrade.
/// - [`InstallRecipe::External`] — observe-only, as today.
//
// `Deserialize` is implemented manually below (legacy flat recipes map to
// `PythonEnv`).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InstallRecipe {
    PythonEnv(PythonEnvRecipe),
    ReleaseArchive(ReleaseArchiveRecipe),
    SourceBuild(SourceBuildRecipe),
    VendorApp(VendorAppRecipe),
    External(ExternalRecipe),
}

/// The pre-M4 flat recipe shape, verbatim (see `deserialize_flexible`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PythonEnvRecipe {
    /// `vllm`, `fakeengine`, ... — selects the offline validation step.
    pub name: String,
    /// Explicit interpreter; when absent it is derived from `ExecStart`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interpreter: Option<String>,
    /// Package names whose versions are pinned and verified.
    #[serde(default)]
    pub packages: Vec<String>,
    /// `exec` rewrites `ExecStart`; `env` sets
    /// `Environment=<NAME>_PYTHON=<interpreter>` via a trailing env file
    /// (a launcher script resolves it).
    pub apply: String,
    /// Name for `apply: "env"` (e.g. `VLLM`). Required for `env` recipes
    /// other than the vllm recipe's historical default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_prefix: Option<String>,
    /// Local wheel directory for staging (`--find-links`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub find_links: Option<String>,
    /// Absolute path to the served model's config, for offline validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_config: Option<String>,
    /// Package index URL for staging.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseArchiveRecipe {
    pub name: String,
    /// Absolute directory the versioned install dirs live in.
    pub root: String,
    /// Archive location: `http(s)://` URL or a local file path, `{version}`
    /// templated.
    pub url: String,
    /// `sha256:<hex>` of the archive (`{version}` templated). When absent
    /// the digest is recorded but not verified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    /// `tar --strip-components` (archives usually wrap one directory).
    #[serde(default = "default_strip")]
    pub strip_components: u32,
    /// Path of the engine binary inside the unpacked directory.
    pub binary: String,
    /// Argument that makes the binary print its version (`--version`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version_arg: Option<String>,
    /// What the version output must contain (default: the version).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version_pattern: Option<String>,
}

fn default_strip() -> u32 {
    1
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceBuildRecipe {
    pub name: String,
    /// Absolute directory the versioned build dirs live in.
    pub root: String,
    /// git URL (or local path) to clone from.
    pub git: String,
    /// git ref to build (default: the version being staged — a tag, branch
    /// or commit; ds4's identity is its fork SHA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_ref: Option<String>,
    /// Build command, run with `sh -c` inside the staged checkout.
    pub build: String,
    /// Path of the built binary inside the checkout.
    pub binary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version_arg: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version_pattern: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VendorAppRecipe {
    pub name: String,
    /// How the running version is observed (owner decision 2: the running
    /// server's reported version is the fact — which is how the stale m5
    /// Ollama gets caught).
    pub probe: VendorProbe,
    /// The tested backup path the owner recorded (what to copy/run before
    /// a vendor upgrade). Reported by the plan; never executed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backup_path: Option<String>,
    /// The tested reinstall command/path the owner recorded. Reported by
    /// the plan; never executed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reinstall_cmd: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "probe", rename_all = "snake_case")]
pub enum VendorProbe {
    /// `GET http://127.0.0.1:<workload port><path>` and read a JSON field
    /// (Ollama: `/api/version`, `version`).
    HttpJson {
        path: String,
        field: String,
    },
    /// Run a command; the first whitespace token of stdout is the version
    /// (`lms version`).
    Command { argv: Vec<String> },
    /// The version is a directory name under `path` (the `llmster`
    /// `<ver>/` layout).
    DirName { path: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalRecipe {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl InstallRecipe {
    pub fn name(&self) -> &str {
        match self {
            InstallRecipe::PythonEnv(r) => &r.name,
            InstallRecipe::ReleaseArchive(r) => &r.name,
            InstallRecipe::SourceBuild(r) => &r.name,
            InstallRecipe::VendorApp(r) => &r.name,
            InstallRecipe::External(r) => &r.name,
        }
    }

    /// The root directory versioned installs live under (python-env uses
    /// the daemon's own envs dir; the caller supplies it).
    pub fn install_root(&self, default_envs_root: &std::path::Path) -> std::path::PathBuf {
        match self {
            InstallRecipe::PythonEnv(_) => default_envs_root.to_path_buf(),
            InstallRecipe::ReleaseArchive(r) => std::path::PathBuf::from(&r.root),
            InstallRecipe::SourceBuild(r) => std::path::PathBuf::from(&r.root),
            InstallRecipe::VendorApp(_) | InstallRecipe::External(_) => {
                default_envs_root.to_path_buf()
            }
        }
    }

    /// Structural validation of deserialized data.
    pub fn validate(&self) -> CoreResult<()> {
        let kind = |s: &str| CoreError::Validation(format!("recipe {s}"));
        match self {
            InstallRecipe::PythonEnv(r) => {
                if r.name.trim().is_empty() {
                    return Err(kind("has an empty name"));
                }
                match r.apply.as_str() {
                    "exec" => {}
                    "env" => {
                        if r.env_prefix.is_none() && r.name != "vllm" {
                            return Err(kind(
                                "apply=env needs env_prefix (only the vllm recipe defaults)",
                            ));
                        }
                    }
                    other => {
                        return Err(CoreError::Validation(format!(
                            "recipe apply must be exec or env, not {other:?}"
                        )))
                    }
                }
            }
            InstallRecipe::ReleaseArchive(r) => {
                if r.name.trim().is_empty() || r.url.trim().is_empty() {
                    return Err(kind("needs a name and a url"));
                }
                if !std::path::Path::new(&r.root).is_absolute() {
                    return Err(kind("root must be absolute"));
                }
                if std::path::Path::new(&r.binary).is_absolute() {
                    return Err(kind("binary must be relative to the unpacked dir"));
                }
                // Supply chain (M4a security review): a fetched archive is
                // unpacked and its binary is *executed* during validation
                // and later promoted to ExecStart — so remote fetches fail
                // closed: no plaintext http at all, and https requires a
                // declared sha256 checksum. Local paths keep the
                // record-only digest.
                if r.url.starts_with("http://") {
                    return Err(kind(
                        "refuses plaintext http URLs (use https with a sha256 checksum)",
                    ));
                }
                if r.url.starts_with("https://")
                    && !r
                        .checksum
                        .as_deref()
                        .is_some_and(|c| c.starts_with("sha256:"))
                {
                    return Err(kind(
                        "https URLs require a sha256 checksum (the archive's binary runs here)",
                    ));
                }
            }
            InstallRecipe::SourceBuild(r) => {
                if r.name.trim().is_empty() || r.git.trim().is_empty() || r.build.trim().is_empty()
                {
                    return Err(kind("needs a name, git url and build command"));
                }
                if !std::path::Path::new(&r.root).is_absolute() {
                    return Err(kind("root must be absolute"));
                }
                if std::path::Path::new(&r.binary).is_absolute() {
                    return Err(kind("binary must be relative to the build dir"));
                }
            }
            InstallRecipe::VendorApp(_) | InstallRecipe::External(_) => {}
        }
        Ok(())
    }

    /// Deserialize an [`InstallRecipe`], mapping the pre-M4 flat recipe
    /// (no `kind` tag) to [`InstallRecipe::PythonEnv`].
    pub fn deserialize_flexible(json: &serde_json::Value) -> Result<Self, serde_json::Error> {
        if json.get("kind").is_some() {
            let tagged: TaggedInstallRecipe = serde_json::from_value(json.clone())?;
            Ok(tagged.into())
        } else {
            let legacy: PythonEnvRecipe = serde_json::from_value(json.clone())?;
            Ok(InstallRecipe::PythonEnv(legacy))
        }
    }
}

/// The serde-derived tagged mirror used to *parse* recipes; the public
/// enum's `Deserialize` goes through [`InstallRecipe::deserialize_flexible`]
/// so legacy flat recipes load as `PythonEnv`. Variants must stay in sync
/// with [`InstallRecipe`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TaggedInstallRecipe {
    PythonEnv(PythonEnvRecipe),
    ReleaseArchive(ReleaseArchiveRecipe),
    SourceBuild(SourceBuildRecipe),
    VendorApp(VendorAppRecipe),
    External(ExternalRecipe),
}

impl From<TaggedInstallRecipe> for InstallRecipe {
    fn from(t: TaggedInstallRecipe) -> Self {
        match t {
            TaggedInstallRecipe::PythonEnv(r) => InstallRecipe::PythonEnv(r),
            TaggedInstallRecipe::ReleaseArchive(r) => InstallRecipe::ReleaseArchive(r),
            TaggedInstallRecipe::SourceBuild(r) => InstallRecipe::SourceBuild(r),
            TaggedInstallRecipe::VendorApp(r) => InstallRecipe::VendorApp(r),
            TaggedInstallRecipe::External(r) => InstallRecipe::External(r),
        }
    }
}

// `HostInventory` (which contains `Option<InstallRecipe>`) loads legacy
// files transparently through the flexible path.
impl<'de> Deserialize<'de> for InstallRecipe {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        InstallRecipe::deserialize_flexible(&value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_flat_recipe_loads_as_python_env() {
        let json = serde_json::json!({
            "name": "vllm",
            "packages": ["vllm"],
            "apply": "env",
            "env_prefix": "VLLM"
        });
        let recipe: InstallRecipe = serde_json::from_value(json).unwrap();
        match &recipe {
            InstallRecipe::PythonEnv(r) => {
                assert_eq!(r.name, "vllm");
                assert_eq!(r.apply, "env");
                assert_eq!(r.env_prefix.as_deref(), Some("VLLM"));
            }
            other => panic!("expected python_env, got {other:?}"),
        }
        // And it re-serializes in the tagged form.
        let out = serde_json::to_value(&recipe).unwrap();
        assert_eq!(out["kind"], "python_env");
    }

    #[test]
    fn tagged_kinds_round_trip() {
        let json = serde_json::json!({
            "kind": "release_archive",
            "name": "llama.cpp",
            "root": "/home/u/builds",
            "url": "https://example.com/llama-{version}.tar.gz",
            "checksum": "sha256:ab",
            "binary": "bin/llama-server",
            "version_arg": "--version"
        });
        let recipe: InstallRecipe = serde_json::from_value(json).unwrap();
        match &recipe {
            InstallRecipe::ReleaseArchive(r) => {
                assert_eq!(r.strip_components, 1);
                assert_eq!(r.binary, "bin/llama-server");
            }
            other => panic!("expected release_archive, got {other:?}"),
        }
        let out = serde_json::to_value(&recipe).unwrap();
        let back: InstallRecipe = serde_json::from_value(out).unwrap();
        assert_eq!(recipe, back);
    }

    #[test]
    fn source_and_vendor_shapes_parse() {
        let src = serde_json::json!({
            "kind": "source_build",
            "name": "ds4",
            "root": "/Users/u/ds4-builds",
            "git": "https://github.com/pitcany/ds41-m5",
            "build": "cargo build --release",
            "binary": "target/release/ds4-server"
        });
        assert!(matches!(
            serde_json::from_value::<InstallRecipe>(src).unwrap(),
            InstallRecipe::SourceBuild(_)
        ));
        let vendor = serde_json::json!({
            "kind": "vendor_app",
            "name": "ollama",
            "probe": {"probe": "http_json", "path": "/api/version", "field": "version"}
        });
        match serde_json::from_value::<InstallRecipe>(vendor).unwrap() {
            InstallRecipe::VendorApp(r) => {
                assert_eq!(
                    r.probe,
                    VendorProbe::HttpJson {
                        path: "/api/version".into(),
                        field: "version".into()
                    }
                );
            }
            other => panic!("expected vendor_app, got {other:?}"),
        }
    }

    #[test]
    fn validation_rejects_relative_roots_and_missing_prefix() {
        let bad_root = serde_json::json!({
            "kind": "release_archive",
            "name": "x", "root": "builds", "url": "u", "binary": "b"
        });
        let recipe = InstallRecipe::deserialize_flexible(&bad_root).unwrap();
        assert!(recipe.validate().is_err());

        let bad_prefix = serde_json::json!({
            "name": "mtplx", "packages": ["mtplx"], "apply": "env"
        });
        let recipe = InstallRecipe::deserialize_flexible(&bad_prefix).unwrap();
        assert!(recipe.validate().is_err());
        // The vllm recipe's historical default stays legal.
        let vllm = serde_json::json!({
            "name": "vllm", "packages": ["vllm"], "apply": "env"
        });
        assert!(InstallRecipe::deserialize_flexible(&vllm)
            .unwrap()
            .validate()
            .is_ok());
    }

    #[test]
    fn engine_kind_infers_from_names() {
        assert_eq!(EngineKind::infer("vllm"), EngineKind::Vllm);
        assert_eq!(EngineKind::infer("llama.cpp"), EngineKind::LlamaCpp);
        assert_eq!(EngineKind::infer("mlx-serve"), EngineKind::MlxServe);
        assert_eq!(EngineKind::infer("fakeengine"), EngineKind::FakeEngine);
        assert_eq!(
            EngineKind::infer("warpdrive"),
            EngineKind::Other("warpdrive".into())
        );
    }
}
