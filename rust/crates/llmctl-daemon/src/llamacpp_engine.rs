//! The llama.cpp release for Mac lanes (M5d item 5; closes M4 carry-over 4,
//! release archives on launchd).
//!
//! On the Macs llama.cpp runs as llama-swap lanes (owner decision 8): a
//! lane's `cmd` names the binary through a macro (`${llamacpp}`), and a
//! macro write is the version swap — it re-pins every lane it moves
//! (`lane_version::follow_write`). So the install is a release-archive
//! stage keyed by that macro: the macOS arm64 archive is fetched, its
//! sha256 checked against the recipe, unpacked into a new versioned dir
//! under the recipe's root, and the binary must report the version. The
//! macro is not touched here; the answer names the `lanes macro` write
//! that switches the lanes (and the one that switches back). Old dirs
//! stay for rollback; a restage of the dir the macro names is refused.

use std::path::Path;

use llmctl_client::llamacpp::{LaneEngineInstallRequest, LaneEngineInstallResponse};
use llmctl_core::{InstallRecipe, ReleaseArchiveRecipe};

use crate::app::AppState;
use crate::lanes::{read_store, refuse, save_store, store_path, LanesError};
use crate::upgrade::{stage_release_archive, RestageGuard, UpgradeError};

/// `{version}` and `{build}` (the version without a leading `b`: release
/// `b12527` prints `version: 12527`) in a recipe's version pattern.
pub fn resolve_pattern(r: &ReleaseArchiveRecipe, version: &str) -> ReleaseArchiveRecipe {
    let mut out = r.clone();
    let build = version.strip_prefix('b').unwrap_or(version);
    out.version_pattern = r
        .version_pattern
        .as_ref()
        .map(|p| p.replace("{version}", version).replace("{build}", build));
    out
}

fn from_upgrade(e: UpgradeError) -> LanesError {
    match e {
        UpgradeError::Refused(m) => refuse(m),
        UpgradeError::Ops(e) => LanesError::Ops(e),
        other => LanesError::Io(other.to_string()),
    }
}

pub async fn install(state: &AppState, req: LaneEngineInstallRequest) -> Result<LaneEngineInstallResponse, LanesError> {
    let path = store_path(state, &req.workload);
    let mut store = read_store(&path).await?.ok_or_else(|| {
        refuse(format!(
            "{} has no lane records; `lanes import {}` first",
            req.workload, req.workload
        ))
    })?;
    if req.macro_name.is_empty() || !req.macro_name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err(refuse(format!("{:?} is not a macro name", req.macro_name)));
    }
    if req.version.is_empty() || req.version.contains(['/', ' ', '\\']) || req.version.starts_with('.') {
        return Err(refuse(format!("{:?} is not a release version", req.version)));
    }
    let mut recipe = req
        .recipe
        .clone()
        .or_else(|| store.engine_recipes.get(&req.macro_name).cloned())
        .ok_or_else(|| {
            refuse(format!(
                "no recipe for the {} engine of {}; pass one with --recipe <file.json>",
                req.macro_name, req.workload
            ))
        })?;
    if let Some(sum) = &req.checksum {
        if !sum.starts_with("sha256:") || sum.len() != 71 {
            return Err(refuse(format!("checksum {sum:?} is not sha256:<64 hex>")));
        }
        recipe.checksums.insert(req.version.clone(), sum.clone());
    }
    InstallRecipe::ReleaseArchive(recipe.clone())
        .validate()
        .map_err(|e| refuse(e.to_string()))?;
    let current = store.records.macro_value(&req.macro_name).map(str::to_owned);
    let resolved = resolve_pattern(&recipe, &req.version);
    let name = format!("{}.{}", req.workload, req.macro_name);
    let outcome = stage_release_archive(
        state,
        &name,
        &RestageGuard::LivePaths(current.iter().cloned().collect()),
        &InstallRecipe::ReleaseArchive(resolved.clone()),
        &resolved,
        &req.version,
    )
    .await
    .map_err(from_upgrade)?;
    // The recipe (with this version's digest) is kept for the next one.
    store.engine_recipes.insert(req.macro_name.clone(), recipe);
    save_store(&path, &store).await.map_err(LanesError::Io)?;

    let dir = outcome.row.env_path.clone();
    let executable = outcome.row.interpreter.clone();
    let archive_sha256 = tokio::fs::read(Path::new(&dir).join("staged.json"))
        .await
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|m| m.get("archive_sha256").and_then(|v| v.as_str()).map(str::to_owned));
    Ok(LaneEngineInstallResponse {
        workload: req.workload.clone(),
        macro_name: req.macro_name.clone(),
        version: req.version.clone(),
        next: format!("llmctl-next lanes macro {} {} {executable}", req.workload, req.macro_name),
        rollback: current
            .as_ref()
            .map(|c| format!("llmctl-next lanes macro {} {} {c}", req.workload, req.macro_name)),
        dir,
        executable,
        archive_sha256,
        validation: outcome.validation,
        current,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_pattern_takes_the_build_number() {
        let r = ReleaseArchiveRecipe {
            name: "llama.cpp".into(),
            root: "/Users/y/local-llm/engines/llama.cpp".into(),
            url: "https://github.com/ggml-org/llama.cpp/releases/download/{version}/llama-{version}-bin-macos-arm64.tar.gz".into(),
            checksum: None,
            checksums: Default::default(),
            strip_components: 1,
            binary: "llama-server".into(),
            version_arg: Some("--version".into()),
            version_pattern: Some("version: {build}".into()),
        };
        assert_eq!(resolve_pattern(&r, "b12527").version_pattern.as_deref(), Some("version: 12527"));
        assert_eq!(resolve_pattern(&r, "1.0.0").version_pattern.as_deref(), Some("version: 1.0.0"));
    }
}
