//! Two config files of the Python llmctl that llmctl-next reads in place
//! (M5 close-out gaps C4 and C5): the router alias overlay
//! (`router_aliases.json`) and the model roots (`model_dirs.yaml`). Both
//! sit beside settings.yaml.
//!
//! shortcut: plain functions that the CLI calls on its own host; there is
//! no API route, so `--host` cannot reach these files. Add routes when the
//! TUI or a remote host needs them.

use std::collections::{BTreeMap, BTreeSet};
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde_yaml::Value;

use crate::presets::{atomic_write, home, settings_path};

/// Alias name to target. A `None` target is an alias that is declared
/// and not bound.
pub type Aliases = BTreeMap<String, Option<String>>;

pub fn alias_overlay_path() -> PathBuf {
    settings_path().with_file_name("router_aliases.json")
}

pub fn model_dirs_path() -> PathBuf {
    settings_path().with_file_name("model_dirs.yaml")
}

fn read_yaml(path: &Path) -> Result<Option<Value>, String> {
    match std::fs::read_to_string(path) {
        Ok(t) => serde_yaml::from_str(&t)
            .map(Some)
            .map_err(|e| format!("{}: not YAML: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

fn scalar(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// The roles Python declares, with no target, when settings.yaml has no
/// `router.aliases` (`_DEFAULT_ROUTER_ALIASES` in `llmctl/config.py`).
const DEFAULT_ROLES: [&str; 7] =
    ["reasoning", "coding", "fast", "long-context", "tutoring", "adtech", "quant"];

/// `router.aliases` of settings.yaml, else the default roles. Only that
/// subtree is kept: the file holds a token elsewhere, and it never leaves
/// this function.
fn settings_aliases(settings: &Path) -> Result<Aliases, String> {
    let doc = read_yaml(settings)?;
    let map = doc
        .as_ref()
        .and_then(|v| v.get("router")?.get("aliases")?.as_mapping());
    Ok(match map {
        Some(map) => map.iter().filter_map(|(k, t)| Some((scalar(k)?, scalar(t)))).collect(),
        None => DEFAULT_ROLES.iter().map(|role| (role.to_string(), None)).collect(),
    })
}

/// The overlay file. A missing file is empty. A file that is not a JSON
/// object is an error: Python reads it as empty and its next `set-alias`
/// then drops every alias in it.
fn read_overlay(overlay: &Path) -> Result<Aliases, String> {
    let text = match std::fs::read_to_string(overlay) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Aliases::new()),
        Err(e) => return Err(format!("{}: {e}", overlay.display())),
    };
    let map: BTreeMap<String, serde_json::Value> = serde_json::from_str(&text)
        .map_err(|e| format!("{}: not a JSON object: {e}", overlay.display()))?;
    Ok(map
        .into_iter()
        .map(|(k, v)| {
            let target = match v {
                serde_json::Value::Null => None,
                serde_json::Value::String(s) => Some(s),
                other => Some(other.to_string()),
            };
            (k, target)
        })
        .collect())
}

/// settings.yaml `router.aliases` overlaid by the overlay file, as
/// Python's `GatewayService._merged_aliases`.
pub fn aliases_at(settings: &Path, overlay: &Path) -> Result<Aliases, String> {
    let mut merged = settings_aliases(settings)?;
    merged.extend(read_overlay(overlay)?);
    Ok(merged)
}

pub fn aliases() -> Result<Aliases, String> {
    aliases_at(&settings_path(), &alias_overlay_path())
}

/// Bind `alias` to `target` in the overlay, or remove it with `None` (an
/// alias of settings.yaml then shows again). settings.yaml is never
/// written.
pub fn set_alias_at(overlay: &Path, alias: &str, target: Option<&str>) -> Result<(), String> {
    if alias.trim().is_empty() {
        return Err("the alias name is empty".into());
    }
    if let Some(dir) = overlay.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    // The lock file of Python's `set-alias`: two writers never drop each
    // other's alias. Released when `lock` drops.
    let lock_path = PathBuf::from(format!("{}.lock", overlay.display()));
    let lock = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&lock_path)
        .map_err(|e| format!("{}: {e}", lock_path.display()))?;
    // SAFETY: `lock` is an open file; flock has no other precondition.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
        let e = std::io::Error::last_os_error();
        return Err(format!("locking {}: {e}", lock_path.display()));
    }
    let mut map = read_overlay(overlay)?;
    match target {
        Some(t) => {
            map.insert(alias.to_string(), Some(t.to_string()));
        }
        None => {
            map.remove(alias);
        }
    }
    // Sorted keys, two spaces, no final newline: the bytes of Python's
    // `json.dumps(overlay, indent=2, sort_keys=True)` for ASCII text.
    let bytes = serde_json::to_vec_pretty(&map).map_err(|e| e.to_string())?;
    let mode = std::fs::metadata(overlay).map(|m| m.permissions().mode() & 0o777).unwrap_or(0o644);
    atomic_write(overlay, &bytes, mode).map_err(|e| e.to_string())
}

pub fn set_alias(alias: &str, target: Option<&str>) -> Result<(), String> {
    set_alias_at(&alias_overlay_path(), alias, target)
}

/// One enabled root of model_dirs.yaml. `path` is `None` when no field
/// gives one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRoot {
    pub name: String,
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDirs {
    pub roots: Vec<ModelRoot>,
    pub max_depth: usize,
}

fn expand_user(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None if p == "~" => home(),
        None => PathBuf::from(p),
    }
}

/// The enabled roots with the path that Python's `ModelRoot.resolve_path`
/// gives: the env var when it is set, else `default_path`, else
/// `relative_path`. A missing file has no roots.
pub fn model_dirs_at(path: &Path) -> Result<ModelDirs, String> {
    let Some(v) = read_yaml(path)? else {
        return Ok(ModelDirs { roots: Vec::new(), max_depth: 4 });
    };
    let max_depth =
        v.get("scan").and_then(|s| s.get("max_depth")).and_then(Value::as_u64).unwrap_or(4) as usize;
    let text = |r: &Value, key: &str| r.get(key).and_then(Value::as_str).map(str::to_owned);
    let roots = v
        .get("model_roots")
        .and_then(Value::as_sequence)
        .into_iter()
        .flatten()
        .filter(|r| r.get("enabled").and_then(Value::as_bool).unwrap_or(true))
        .map(|r| {
            let relative = text(r, "relative_path").unwrap_or_else(|| ".".into());
            let from_env = text(r, "env_var")
                .and_then(|key| std::env::var(key).ok())
                .filter(|base| !base.is_empty())
                .map(|base| match relative.as_str() {
                    "." => expand_user(&base),
                    rel => expand_user(&base).join(rel),
                });
            let path = from_env
                .or_else(|| text(r, "default_path").map(|p| expand_user(&p)))
                .or_else(|| (relative != ".").then(|| expand_user(&relative)));
            ModelRoot { name: text(r, "name").unwrap_or_default(), path }
        })
        .collect();
    Ok(ModelDirs { roots, max_depth })
}

pub fn model_dirs() -> Result<ModelDirs, String> {
    model_dirs_at(&model_dirs_path())
}

/// Symlinks under `root` whose target does not resolve. The walk follows
/// no link and goes `max_depth` directory levels down, as Python's
/// `iter_broken_symlinks`. A directory that cannot be read is skipped.
pub fn broken_symlinks(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut todo = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = todo.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_symlink() {
                if !path.exists() {
                    found.push(path);
                }
            } else if kind.is_dir() && depth < max_depth {
                todo.push((path, depth + 1));
            }
        }
    }
    found.sort();
    found
}

/// Broken symlinks per root name, as Python's `check_model_root_symlinks`:
/// a root with no path or no directory is skipped, and two roots on one
/// directory are swept one time (the first root gets the findings).
pub fn check_roots(dirs: &ModelDirs) -> BTreeMap<String, Vec<PathBuf>> {
    let mut seen = BTreeSet::new();
    let mut out = BTreeMap::new();
    for root in &dirs.roots {
        let Some(dir) = root.path.as_ref().and_then(|p| std::fs::canonicalize(p).ok()) else {
            continue;
        };
        if dir.is_dir() && seen.insert(dir.clone()) {
            out.insert(root.name.clone(), broken_symlinks(&dir, dirs.max_depth));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn the_overlay_wins_over_settings_and_set_writes_the_python_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let settings = dir.path().join("settings.yaml");
        let overlay = dir.path().join("router_aliases.json");
        std::fs::write(
            &settings,
            "api:\n  token: SECRET\nrouter:\n  aliases:\n    coding: from-settings\n    reasoning: null\n",
        )
        .unwrap();
        // No overlay yet: the settings aliases, an unbound one included.
        let only_settings = aliases_at(&settings, &overlay).unwrap();
        assert_eq!(only_settings["coding"].as_deref(), Some("from-settings"));
        assert_eq!(only_settings["reasoning"], None);

        set_alias_at(&overlay, "long-context", Some("qwen3.8-27b")).unwrap();
        set_alias_at(&overlay, "coding", Some("flash")).unwrap();
        // What `json.dumps(overlay, indent=2, sort_keys=True)` writes.
        assert_eq!(
            std::fs::read_to_string(&overlay).unwrap(),
            "{\n  \"coding\": \"flash\",\n  \"long-context\": \"qwen3.8-27b\"\n}"
        );
        let merged = aliases_at(&settings, &overlay).unwrap();
        assert_eq!(merged["coding"].as_deref(), Some("flash"));
        assert_eq!(merged.len(), 3);
        assert!(!format!("{merged:?}").contains("SECRET"));

        // Clearing the overlay entry shows the settings alias again.
        set_alias_at(&overlay, "coding", None).unwrap();
        assert_eq!(aliases_at(&settings, &overlay).unwrap()["coding"].as_deref(), Some("from-settings"));
        assert!(set_alias_at(&overlay, " ", Some("x")).is_err());
    }

    #[test]
    fn a_malformed_overlay_is_refused_and_left_as_it_is() {
        let dir = tempfile::tempdir().unwrap();
        let overlay = dir.path().join("router_aliases.json");
        std::fs::write(&overlay, "[\"not\", \"an object\"]").unwrap();
        assert!(aliases_at(&dir.path().join("none.yaml"), &overlay).is_err());
        assert!(set_alias_at(&overlay, "coding", Some("x")).is_err());
        assert_eq!(std::fs::read_to_string(&overlay).unwrap(), "[\"not\", \"an object\"]");

        // No `router.aliases` in settings.yaml: Python's seven roles, unbound.
        let roles = aliases_at(&dir.path().join("none.yaml"), &dir.path().join("none.json")).unwrap();
        assert_eq!(roles.len(), 7);
        assert!(roles.contains_key("quant") && roles.values().all(Option::is_none));
    }

    #[test]
    fn roots_resolve_as_python_does() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model_dirs.yaml");
        std::fs::write(
            &file,
            "model_roots:\n\
             \x20 - name: by-default\n    env_var: LLMCTL_TEST_NO_SUCH_VAR\n    relative_path: hub\n    default_path: /srv/models\n\
             \x20 - name: by-relative\n    relative_path: /srv/rel\n\
             \x20 - name: nothing\n\
             \x20 - name: off\n    enabled: false\n    default_path: /srv/off\n\
             scan:\n  max_depth: 2\n",
        )
        .unwrap();
        let dirs = model_dirs_at(&file).unwrap();
        assert_eq!(dirs.max_depth, 2);
        let got: Vec<_> = dirs.roots.iter().map(|r| (r.name.as_str(), r.path.clone())).collect();
        assert_eq!(
            got,
            vec![
                ("by-default", Some(PathBuf::from("/srv/models"))),
                ("by-relative", Some(PathBuf::from("/srv/rel"))),
                ("nothing", None),
            ]
        );
        assert_eq!(expand_user("~/x"), home().join("x"));
        assert_eq!(model_dirs_at(&dir.path().join("missing.yaml")).unwrap().roots, Vec::new());
    }

    #[test]
    fn broken_links_are_found_to_the_depth_limit_and_a_directory_is_swept_once() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        std::fs::create_dir_all(root.join("a/b")).unwrap();
        std::fs::write(root.join("real"), "x").unwrap();
        symlink(root.join("real"), root.join("good")).unwrap();
        symlink(root.join("gone"), root.join("dead")).unwrap();
        symlink(root.join("gone"), root.join("a/dead1")).unwrap();
        symlink(root.join("gone"), root.join("a/b/dead2")).unwrap();
        // A link to a directory is not followed.
        symlink(root.join("a"), root.join("loop")).unwrap();

        let root = std::fs::canonicalize(&root).unwrap();
        assert_eq!(
            broken_symlinks(&root, 4),
            vec![root.join("a/b/dead2"), root.join("a/dead1"), root.join("dead")]
        );
        // Depth 1 reads `root` and `root/a`, not `root/a/b`.
        assert_eq!(broken_symlinks(&root, 1), vec![root.join("a/dead1"), root.join("dead")]);

        let dirs = ModelDirs {
            roots: vec![
                ModelRoot { name: "first".into(), path: Some(root.clone()) },
                ModelRoot { name: "same".into(), path: Some(root.join("a/..")) },
                ModelRoot { name: "missing".into(), path: Some(root.join("nope")) },
                ModelRoot { name: "none".into(), path: None },
            ],
            max_depth: 0,
        };
        let found = check_roots(&dirs);
        assert_eq!(found.keys().collect::<Vec<_>>(), vec!["first"]);
        assert_eq!(found["first"], vec![root.join("dead")]);
    }
}
