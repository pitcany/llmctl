//! Env-file profiles (M4c, the rest of plan item 3). A profile is an env
//! file whose bytes become a workload's `EnvironmentFile`: how one vLLM
//! unit serves several models (`vllm-engine.env.qwen27b-awq` copied onto
//! `vllm-engine.env`). Until now only llm-router did that copy, behind
//! llmctl-next's back — and since an adopted definition pins the env file's
//! digest, every switch read as drift.
//!
//! Here the swap is a step of the operation itself: checked against the
//! digest the plan saw, backed up next to the env file, undone on failure,
//! and followed by a re-pin (env digest + the model the workload now
//! serves) so the next operation sees no drift.
//!
//! Env contents may hold credentials. They are copied file to file and
//! never read into the journal, the inventory, the API or a log line.

use std::path::{Path, PathBuf};

use llmctl_core::{FieldDiff, HostInventory, Ownership, ProfileSwap, WorkloadDefinition, WorkloadName};
use sha2::{Digest, Sha256};

/// Where the forward step keeps the env it replaces (next to the env file,
/// same mode). One slot: it only ever needs to undo the running operation.
pub const BACKUP_SUFFIX: &str = ".llmctl-previous";

/// Marks the temp file a swap writes before the rename (`<env>.llmctl-tmp-<pid>`).
const TMP_INFIX: &str = "llmctl-tmp-";

/// A profile is a handful of `KEY=value` lines; anything larger is not one.
const MAX_PROFILE_BYTES: u64 = 1024 * 1024;

pub fn digest(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

/// `sha256:<hex>` of the file, or `missing` (the pin's own spelling).
pub async fn file_digest(path: &Path) -> String {
    match tokio::fs::read(path).await {
        Ok(bytes) => digest(&bytes),
        Err(_) => "missing".into(),
    }
}

/// Turn a request (`env_source`, `model`) into the swap the planner
/// arranges. `Ok(None)`: that profile is already in place and the workload
/// already carries that model — an ordinary run. Refusals are worded for
/// the caller.
pub async fn resolve(
    def: &WorkloadDefinition,
    env_source: &str,
    model: &str,
) -> Result<Option<ProfileSwap>, String> {
    let name = &def.name;
    let Ownership::Managed { definition, .. } = &def.ownership else {
        return Err(format!(
            "{name} is not adopted: a profile switch replaces a pinned EnvironmentFile"
        ));
    };
    let [env] = definition.environment_files.as_slice() else {
        return Err(format!(
            "{name} has {} EnvironmentFile entries; a profile switch needs exactly one",
            definition.environment_files.len()
        ));
    };
    let model = model.trim();
    if model.is_empty() {
        return Err("a profile switch needs the model the profile serves (--model)".into());
    }
    let target = PathBuf::from(&env.path);
    let source = Path::new(env_source);
    if !source.is_absolute() {
        return Err(format!("profile path {env_source:?} must be absolute"));
    }
    let canon = |p: PathBuf| async move {
        tokio::fs::canonicalize(&p)
            .await
            .map_err(|e| format!("{}: {e}", p.display()))
    };
    let source = canon(source.to_path_buf()).await?;
    let target_real = canon(target.clone()).await?;
    let meta = tokio::fs::metadata(&source)
        .await
        .map_err(|e| format!("{}: {e}", source.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a regular file", source.display()));
    }
    if meta.len() > MAX_PROFILE_BYTES {
        return Err(format!("{} is too large to be an env profile", source.display()));
    }
    // Profiles live next to the env file they replace (`services/`): the
    // request cannot make the agent copy some other file into a unit's env.
    if source.parent() != target_real.parent() {
        return Err(format!(
            "{} is not in {}; a profile sits next to the env file it replaces",
            source.display(),
            target_real.parent().map(|p| p.display().to_string()).unwrap_or_default()
        ));
    }
    if source == target_real {
        return Err(format!("{} is the active env file itself, not a profile", source.display()));
    }
    // A profile of `vllm-engine.env` is named `vllm-engine.env.<profile>`.
    // Without this, another unit's env file in the same directory (and
    // whatever it holds) could be made this unit's env.
    let env_name = target_real
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let source_name = source
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let suffix = source_name
        .strip_prefix(&env_name)
        .and_then(|rest| rest.strip_prefix('.'))
        .filter(|rest| !rest.is_empty());
    let Some(suffix) = suffix else {
        return Err(format!(
            "{} is not a profile of {env_name}: profiles are named {env_name}.<profile>",
            source.display()
        ));
    };
    if source_name.ends_with(BACKUP_SUFFIX) || suffix.starts_with(TMP_INFIX) {
        return Err(format!("{} is llmctl's own working file, not a profile", source.display()));
    }
    let bytes = tokio::fs::read(&source)
        .await
        .map_err(|e| format!("{}: {e}", source.display()))?;
    let source_digest = digest(&bytes);
    let current_digest = file_digest(&target).await;
    if current_digest == "missing" {
        return Err(format!("{} does not exist; nothing to switch from", target.display()));
    }
    if source_digest == current_digest && model == def.model {
        return Ok(None);
    }
    Ok(Some(ProfileSwap {
        backup: format!("{}{BACKUP_SUFFIX}", env.path),
        path: env.path.clone(),
        source: source.display().to_string(),
        source_digest,
        current_digest,
        model: model.to_string(),
    }))
}

/// The file half of an `ApplyEnvFile` step. Returns what happened, for the
/// journal (never the contents).
pub async fn apply_file(
    path: &str,
    source: &str,
    source_digest: &str,
    backup: Option<&str>,
) -> Result<&'static str, String> {
    use std::os::unix::fs::PermissionsExt;
    let path = Path::new(path);
    if file_digest(path).await == source_digest {
        return Ok("already in place");
    }
    let bytes = tokio::fs::read(source)
        .await
        .map_err(|e| format!("reading {source}: {e}"))?;
    let found = digest(&bytes);
    if found != source_digest {
        return Err(format!(
            "{source} is not the file this plan was made for (expected {source_digest}, found {found})"
        ));
    }
    // Env files may hold credentials. Contents never become more readable
    // by moving: the new env gets the stricter of the env file's mode and
    // the profile's own, and the backup keeps the mode of the env it copies.
    // 0600 when a mode cannot be read.
    let mode_of = |p: PathBuf| async move {
        tokio::fs::metadata(&p)
            .await
            .map(|m| m.permissions().mode() & 0o777)
            .unwrap_or(0o600)
    };
    let env_mode = mode_of(path.to_path_buf()).await;
    let new_mode = env_mode & mode_of(PathBuf::from(source)).await;
    if let Some(backup) = backup {
        let current = tokio::fs::read(path)
            .await
            .map_err(|e| format!("reading {}: {e}", path.display()))?;
        write_replacing(Path::new(backup), &current, env_mode).await?;
    }
    write_replacing(path, &bytes, new_mode).await?;
    Ok("swapped")
}

/// Write `bytes` to `path` through a private temp file and a rename, so a
/// reader never sees half an env file.
async fn write_replacing(path: &Path, bytes: &[u8], mode: u32) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let tmp = PathBuf::from(format!("{}.{TMP_INFIX}{}", path.display(), std::process::id()));
    let io = |what: &str, e: std::io::Error| format!("{what} {}: {e}", path.display());
    let _ = tokio::fs::remove_file(&tmp).await;
    {
        let mut opts = tokio::fs::OpenOptions::new();
        opts.write(true).create_new(true).mode(0o600);
        let mut file = opts.open(&tmp).await.map_err(|e| io("creating a temp file for", e))?;
        tokio::io::AsyncWriteExt::write_all(&mut file, bytes)
            .await
            .map_err(|e| io("writing", e))?;
        file.sync_all().await.map_err(|e| io("syncing", e))?;
    }
    tokio::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(mode))
        .await
        .map_err(|e| io("setting the mode of", e))?;
    if let Err(e) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(io("replacing", e));
    }
    Ok(())
}

/// The inventory with `workload` re-pinned to the env digest and model a
/// profile step put in place; `None` when it already says so.
pub fn repinned(
    inv: &HostInventory,
    workload: &WorkloadName,
    path: &str,
    env_digest: &str,
    model: &str,
) -> Option<HostInventory> {
    let mut next = inv.clone();
    let w = next.workloads.iter_mut().find(|w| &w.name == workload)?;
    let mut changed = false;
    if w.model != model {
        w.model = model.to_string();
        changed = true;
    }
    if let Ownership::Managed {
        definition, digest, ..
    } = &mut w.ownership
    {
        for env in &mut definition.environment_files {
            if env.path == path && env.content_digest != env_digest {
                env.content_digest = env_digest.to_string();
                changed = true;
            }
        }
        *digest = definition.digest();
    }
    changed.then_some(next)
}

/// True when a drift report says only this: the contents of the pinned env
/// files changed (same files, same order, same optional flags). That is
/// what a profile switch made outside llmctl-next leaves behind, and what
/// the next profile switch overwrites anyway — so it must not refuse it.
pub fn only_env_content_differs(diffs: &[FieldDiff]) -> bool {
    let shape = |v: &serde_json::Value| -> Option<Vec<(String, bool)>> {
        v.as_array()?
            .iter()
            .map(|e| Some((e.get("path")?.as_str()?.to_string(), e.get("optional")?.as_bool()?)))
            .collect()
    };
    !diffs.is_empty()
        && diffs.iter().all(|d| {
            d.field == "environment_files"
                && matches!((shape(&d.expected), shape(&d.actual)), (Some(a), Some(b)) if a == b)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llmctl_core::launch::{EnvFileRef, LaunchDefinition};
    use serde_json::json;

    fn managed(dir: &Path, env_files: &[&str]) -> WorkloadDefinition {
        let host = llmctl_core::HostId::new("workstation").unwrap();
        let mut inv = crate::app::sample_inventory(&host);
        let mut def = inv.workloads.remove(0);
        let definition = LaunchDefinition {
            environment_files: env_files
                .iter()
                .map(|f| EnvFileRef {
                    path: dir.join(f).display().to_string(),
                    optional: false,
                    content_digest: digest(b"A=1\n"),
                })
                .collect(),
            ..Default::default()
        };
        def.ownership = Ownership::Managed {
            digest: definition.digest(),
            definition: Box::new(definition),
            adopted_at: chrono::Utc::now(),
            provenance: vec![],
            service_file: None,
            version: None,
            definition_schema: 1,
            pending_apply: None,
        };
        def
    }

    #[tokio::test]
    async fn resolve_accepts_a_sibling_profile_and_refuses_everything_else() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().canonicalize().unwrap();
        std::fs::write(d.join("coder.env"), "A=1\n").unwrap();
        std::fs::write(d.join("coder.env.fast"), "A=2\n").unwrap();
        std::fs::write(d.join(format!("coder.env{BACKUP_SUFFIX}")), "A=0\n").unwrap();
        for other in ["coder.env.llmctl-tmp-4242", "helper.env", "coder.envx"] {
            std::fs::write(d.join(other), "A=7\n").unwrap();
        }
        let other = tempfile::tempdir().unwrap();
        std::fs::write(other.path().join("evil.env"), "A=9\n").unwrap();
        let def = managed(&d, &["coder.env"]);
        let p = |f: &str| d.join(f).display().to_string();

        let swap = resolve(&def, &p("coder.env.fast"), " fake/coder-fast ").await.unwrap().unwrap();
        assert_eq!(swap.path, p("coder.env"));
        assert_eq!(swap.source, p("coder.env.fast"));
        assert_eq!(swap.source_digest, digest(b"A=2\n"));
        assert_eq!(swap.current_digest, digest(b"A=1\n"));
        assert_eq!(swap.backup, p(&format!("coder.env{BACKUP_SUFFIX}")));
        assert_eq!(swap.model, "fake/coder-fast");

        // The same bytes and the same model: nothing to switch.
        std::fs::write(d.join("coder.env.same"), "A=1\n").unwrap();
        assert_eq!(resolve(&def, &p("coder.env.same"), &def.model).await.unwrap(), None);
        // The same bytes under another model name still re-pins the model.
        assert!(resolve(&def, &p("coder.env.same"), "other").await.unwrap().is_some());

        let refused = [
            (p("coder.env.fast"), "  ", "needs the model"),
            ("coder.env.fast".to_string(), "m", "must be absolute"),
            (p("nope.env"), "m", "nope.env"),
            (other.path().join("evil.env").display().to_string(), "m", "next to the env file"),
            (p("coder.env"), "m", "the active env file itself"),
            (p(&format!("coder.env{BACKUP_SUFFIX}")), "m", "own working file"),
            (p("coder.env.llmctl-tmp-4242"), "m", "own working file"),
            // Another unit's env file in the same directory is not a profile.
            (p("helper.env"), "m", "profiles are named coder.env.<profile>"),
            (p("coder.envx"), "m", "profiles are named coder.env.<profile>"),
            (d.display().to_string(), "m", "not a regular file"),
        ];
        for (src, model, why) in refused {
            let err = resolve(&def, &src, model).await.unwrap_err();
            assert!(err.contains(why), "{src}: {err}");
        }
        // Not adopted, or more than one env file: no single file to switch.
        let mut external = def.clone();
        external.ownership = Ownership::External;
        assert!(resolve(&external, &p("coder.env.fast"), "m").await.unwrap_err().contains("not adopted"));
        let two = managed(&d, &["coder.env", "b.env"]);
        assert!(resolve(&two, &p("coder.env.fast"), "m").await.unwrap_err().contains("exactly one"));
    }

    #[tokio::test]
    async fn apply_swaps_once_backs_up_and_restores() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let p = |f: &str| dir.path().join(f).display().to_string();
        std::fs::write(p("e.env"), "OLD=1\n").unwrap();
        std::fs::set_permissions(p("e.env"), std::fs::Permissions::from_mode(0o640)).unwrap();
        std::fs::write(p("e.env.new"), "NEW=1\n").unwrap();
        std::fs::set_permissions(p("e.env.new"), std::fs::Permissions::from_mode(0o600)).unwrap();
        let (old, new) = (digest(b"OLD=1\n"), digest(b"NEW=1\n"));

        // A source that changed since the plan: refused, nothing touched.
        let err = apply_file(&p("e.env"), &p("e.env.new"), &old.replace('a', "b"), Some(&p("e.bak")))
            .await
            .unwrap_err();
        assert!(err.contains("not the file this plan was made for"), "{err}");
        assert_eq!(std::fs::read_to_string(p("e.env")).unwrap(), "OLD=1\n");
        assert!(!Path::new(&p("e.bak")).exists());

        assert_eq!(apply_file(&p("e.env"), &p("e.env.new"), &new, Some(&p("e.bak"))).await, Ok("swapped"));
        assert_eq!(std::fs::read_to_string(p("e.env")).unwrap(), "NEW=1\n");
        assert_eq!(std::fs::read_to_string(p("e.bak")).unwrap(), "OLD=1\n");
        // The profile was 0600 (set below before the swap): its contents do
        // not become group-readable by landing in a 0640 env file. The
        // backup keeps the old env's mode. No temp file is left behind.
        let mode = |f: &str| std::fs::metadata(p(f)).unwrap().permissions().mode() & 0o777;
        assert_eq!((mode("e.env"), mode("e.bak")), (0o600, 0o640));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 3);

        // Again: already in place, and the backup is not overwritten with
        // the new bytes (it must still undo the first swap).
        assert_eq!(
            apply_file(&p("e.env"), &p("e.env.new"), &new, Some(&p("e.bak"))).await,
            Ok("already in place")
        );
        assert_eq!(std::fs::read_to_string(p("e.bak")).unwrap(), "OLD=1\n");

        // The restore reads the backup; a restore when nothing was swapped
        // is a no-op even if the backup is stale or missing.
        assert_eq!(apply_file(&p("e.env"), &p("e.bak"), &old, None).await, Ok("swapped"));
        assert_eq!(std::fs::read_to_string(p("e.env")).unwrap(), "OLD=1\n");
        std::fs::remove_file(p("e.bak")).unwrap();
        assert_eq!(apply_file(&p("e.env"), &p("e.bak"), &old, None).await, Ok("already in place"));
    }

    #[test]
    fn repin_moves_the_env_digest_and_the_model_together() {
        let dir = tempfile::tempdir().unwrap();
        let def = managed(dir.path(), &["coder.env"]);
        let host = llmctl_core::HostId::new("workstation").unwrap();
        let mut inv = crate::app::sample_inventory(&host);
        let name = def.name.clone();
        let path = dir.path().join("coder.env").display().to_string();
        inv.workloads[0] = def;

        let next = repinned(&inv, &name, &path, "sha256:new", "fake/coder-fast").unwrap();
        let w = next.workload(&name).unwrap();
        assert_eq!(w.model, "fake/coder-fast");
        let Ownership::Managed { definition, digest, .. } = &w.ownership else {
            panic!("still managed")
        };
        assert_eq!(definition.environment_files[0].content_digest, "sha256:new");
        assert_eq!(*digest, definition.digest());
        // Saying it twice changes nothing.
        assert_eq!(repinned(&next, &name, &path, "sha256:new", "fake/coder-fast"), None);
    }

    #[test]
    fn only_a_changed_env_content_is_tolerated_as_drift() {
        let env = |digest: &str, path: &str| json!([{"path": path, "optional": false, "content_digest": digest}]);
        let diff = |field: &str, expected, actual| FieldDiff {
            field: field.into(),
            expected,
            actual,
        };
        assert!(only_env_content_differs(&[diff(
            "environment_files",
            env("sha256:a", "/s/v.env"),
            env("sha256:b", "/s/v.env")
        )]));
        // Another env file, another field, or no drift at all: not this case.
        assert!(!only_env_content_differs(&[diff(
            "environment_files",
            env("sha256:a", "/s/v.env"),
            env("sha256:b", "/s/other.env")
        )]));
        assert!(!only_env_content_differs(&[
            diff("environment_files", env("sha256:a", "/s/v.env"), env("sha256:b", "/s/v.env")),
            diff("exec_start", json!(["a"]), json!(["b"])),
        ]));
        assert!(!only_env_content_differs(&[]));
    }
}
