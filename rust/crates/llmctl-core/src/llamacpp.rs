//! llama.cpp router mode as typed records (M5d, design items 1–3 and 5).
//!
//! `llama-server` started without `-m` is a router: it serves the models of
//! a preset INI (`--models-preset`, one section per model), lists them on
//! `GET /models` with a status, and loads or unloads one on
//! `POST /models/load|unload` with `{"model": id}`. llmctl-next owns that
//! INI the way it owns a llama-swap config:
//!
//! 1. **Import** ([`import`]): parse the INI into a [`RouterPreset`] — top
//!    keys, the `[*]` globals, one section per model — keeping every
//!    comment and blank line next to the key or header it precedes.
//! 2. **Prove** ([`roundtrip`]): render, parse the render, and compare what
//!    llama.cpp would read (section → key → value, last one wins) with the
//!    original. A canonical file renders byte for byte. Nothing is written
//!    from records that failed this, and every write re-proves its render
//!    ([`check_render`]).
//! 3. **Validate** ([`validate`]): the grammar llama.cpp's parser accepts
//!    (`common/preset.cpp`: keys `[a-zA-Z_][a-zA-Z0-9_.-]*`, a value ends at
//!    the first `;` or `#`) plus the rules of this deployment: every model
//!    names an absolute GGUF (or an HF repo), no key the router overrides
//!    (`host`, `port`, `alias`, `api-key`).
//!
//! The router's own command line is checked by [`check_router_argv`]: no
//! model of its own, a preset, and **`--no-models-autoload`** (owner
//! decision 9): a request for an unloaded model fails fast instead of
//! loading behind the planner. Every load is decided by [`decide_load`]
//! first — the VRAM fit on the workstation, the memory budget on a Mac —
//! and refused at `--models-max`, so the router never evicts on its own.
//!
//! [`ModelRecord`] is the catalog-facing view (M5c lists router models
//! from it). [`LlamaLane`] renders a llama.cpp lane for llama-swap on the
//! Macs (owner decision 8). This module makes no system calls.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::fit::{format_mib, CardMib, HEADROOM_MIB};
use crate::recipe::EngineKind;

/// The model name that means "the router itself" (its workload's model).
pub const SERVICE_IDENTITY: &str = "llamacpp-router";
/// The section whose keys apply to every model.
pub const GLOBAL_SECTION: &str = "*";
/// llama.cpp's default when `--models-max` is absent.
pub const DEFAULT_MODELS_MAX: u32 = 4;
/// Host RAM a CPU-offload load must leave free (GiB).
pub const HOST_RAM_HEADROOM_GIB: f64 = 4.0;

/// Keys the router removes or overwrites on load (README "Model presets").
const ROUTER_CONTROLLED: &[&str] = &["host", "port", "alias", "a", "api-key", "api-key-file"];
/// Router options that make no sense per model.
const ROUTER_ONLY: &[&str] = &[
    "models-preset",
    "models-dir",
    "models-max",
    "models-autoload",
    "no-models-autoload",
];
/// Preset-only keys (not command-line arguments).
const PRESET_ONLY: &[&str] = &["load-on-startup", "stop-timeout", "dedup-cache-models"];

// ---- records -----------------------------------------------------------------------

/// One `key = value` line with the comment and blank lines before it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub key: String,
    pub value: String,
    /// Lines before this key, verbatim: comments (`; …`, `# …`) or `""`
    /// for a blank line.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub before: Vec<String>,
    /// A trailing comment on the same line, verbatim from its marker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<String>,
}

impl Entry {
    pub fn new(key: &str, value: &str) -> Self {
        Self {
            key: key.to_string(),
            value: value.to_string(),
            before: Vec::new(),
            inline: None,
        }
    }
}

/// One `[name]` section: `*` (globals) or a model id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Section {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub before: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inline: Option<String>,
    pub entries: Vec<Entry>,
}

impl Section {
    pub fn get(&self, keys: &[&str]) -> Option<&str> {
        // Last one wins, as in llama.cpp's parser.
        self.entries
            .iter()
            .rev()
            .find(|e| keys.contains(&e.key.as_str()))
            .map(|e| e.value.as_str())
    }

    pub fn is_global(&self) -> bool {
        self.name == GLOBAL_SECTION
    }
}

/// The whole preset INI as records.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RouterPreset {
    /// Keys before the first section (`version = 1`).
    #[serde(default)]
    pub top: Vec<Entry>,
    #[serde(default)]
    pub sections: Vec<Section>,
    /// Comment and blank lines after the last key.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trailing: Vec<String>,
}

/// A model section as typed fields (the INI is the source; this is a view).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterModel {
    /// Section name = the id requests route by (`"model": id`).
    pub id: String,
    /// Absolute GGUF path, or the first shard of a split model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub load_on_startup: bool,
    /// Every other key, in file order (`ctx-size`, `n-gpu-layers`, …).
    #[serde(default)]
    pub args: Vec<(String, String)>,
}

const MODEL_KEYS: &[&str] = &["model", "m", "LLAMA_ARG_MODEL"];
const HF_KEYS: &[&str] = &["hf-repo", "hf", "hfr", "LLAMA_ARG_HF_REPO"];
const MMPROJ_KEYS: &[&str] = &["mmproj", "mm", "LLAMA_ARG_MMPROJ"];
const STARTUP_KEYS: &[&str] = &["load-on-startup"];

impl RouterModel {
    pub fn from_section(s: &Section) -> Self {
        let typed = |k: &str| {
            MODEL_KEYS.contains(&k) || HF_KEYS.contains(&k) || MMPROJ_KEYS.contains(&k) || STARTUP_KEYS.contains(&k)
        };
        Self {
            id: s.name.clone(),
            model: s.get(MODEL_KEYS).map(str::to_owned),
            hf_repo: s.get(HF_KEYS).map(str::to_owned),
            mmproj: s.get(MMPROJ_KEYS).map(str::to_owned),
            load_on_startup: s.get(STARTUP_KEYS).is_some_and(is_truthy),
            args: s
                .entries
                .iter()
                .filter(|e| !typed(&e.key))
                .map(|e| (e.key.clone(), e.value.clone()))
                .collect(),
        }
    }

    /// The section a new model renders to (canonical key order).
    pub fn to_section(&self) -> Section {
        let mut entries = Vec::new();
        if let Some(m) = &self.model {
            entries.push(Entry::new("model", m));
        }
        if let Some(h) = &self.hf_repo {
            entries.push(Entry::new("hf-repo", h));
        }
        if let Some(m) = &self.mmproj {
            entries.push(Entry::new("mmproj", m));
        }
        for (k, v) in &self.args {
            entries.push(Entry::new(k, v));
        }
        if self.load_on_startup {
            entries.push(Entry::new("load-on-startup", "true"));
        }
        Section {
            name: self.id.clone(),
            before: Vec::new(),
            inline: None,
            entries,
        }
    }

    pub fn arg(&self, keys: &[&str]) -> Option<&str> {
        self.args
            .iter()
            .rev()
            .find(|(k, _)| keys.contains(&k.as_str()))
            .map(|(_, v)| v.as_str())
    }
}

pub fn is_truthy(v: &str) -> bool {
    matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes" | "enabled")
}

fn is_boolish(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "on" | "yes" | "enabled" | "0" | "false" | "off" | "no" | "disabled"
    )
}

impl RouterPreset {
    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name == name)
    }

    pub fn section_mut(&mut self, name: &str) -> Option<&mut Section> {
        self.sections.iter_mut().find(|s| s.name == name)
    }

    pub fn global(&self) -> Option<&Section> {
        self.section(GLOBAL_SECTION)
    }

    /// Model sections in file order.
    pub fn models(&self) -> Vec<RouterModel> {
        self.sections
            .iter()
            .filter(|s| !s.is_global())
            .map(RouterModel::from_section)
            .collect()
    }

    pub fn model(&self, id: &str) -> Option<RouterModel> {
        self.section(id).filter(|s| !s.is_global()).map(RouterModel::from_section)
    }

    pub fn model_ids(&self) -> Vec<String> {
        self.sections
            .iter()
            .filter(|s| !s.is_global())
            .map(|s| s.name.clone())
            .collect()
    }

    /// Set `key` in `section`: the last line with that key is changed in
    /// place (its comments stay); a new key is appended. The section must
    /// exist.
    pub fn set(&mut self, section: &str, key: &str, value: &str) -> Result<(), String> {
        let s = self
            .section_mut(section)
            .ok_or_else(|| format!("no section [{section}] in the preset"))?;
        match s.entries.iter_mut().rev().find(|e| e.key == key) {
            Some(e) => e.value = value.to_string(),
            None => s.entries.push(Entry::new(key, value)),
        }
        Ok(())
    }

    /// Remove every line with `key` from `section`; their comments move to
    /// the next line so no note is lost. True when something was removed.
    pub fn remove_key(&mut self, section: &str, key: &str) -> Result<bool, String> {
        let s = self
            .section_mut(section)
            .ok_or_else(|| format!("no section [{section}] in the preset"))?;
        let mut carried: Vec<String> = Vec::new();
        let mut out = Vec::with_capacity(s.entries.len());
        let mut removed = false;
        for mut e in std::mem::take(&mut s.entries) {
            if e.key == key {
                removed = true;
                carried.extend(e.before);
                continue;
            }
            if !carried.is_empty() {
                let mut b = std::mem::take(&mut carried);
                b.extend(e.before);
                e.before = b;
            }
            out.push(e);
        }
        s.entries = out;
        if !carried.is_empty() {
            self.carry_after(section, carried);
        }
        Ok(removed)
    }

    /// Notes of something removed at the end of `section`: before the next
    /// section, or trailing.
    fn carry_after(&mut self, section: &str, notes: Vec<String>) {
        let i = self.sections.iter().position(|s| s.name == section);
        match i.and_then(|i| self.sections.get_mut(i + 1)) {
            Some(next) => {
                let mut b = notes;
                b.append(&mut next.before);
                next.before = b;
            }
            None => {
                let mut b = notes;
                b.append(&mut self.trailing);
                self.trailing = b;
            }
        }
    }

    /// Append a model section (a blank line before it). Refused when the
    /// id exists.
    pub fn add_model(&mut self, m: &RouterModel) -> Result<(), String> {
        if self.section(&m.id).is_some() {
            return Err(format!("[{}] exists in the preset", m.id));
        }
        let mut s = m.to_section();
        if !self.sections.is_empty() || !self.top.is_empty() {
            s.before.push(String::new());
        }
        self.sections.push(s);
        Ok(())
    }

    /// Remove a model section. Its own comment lines go with it (they
    /// described it); a blank line before it is kept for the next one.
    pub fn remove_model(&mut self, id: &str) -> Result<(), String> {
        let i = self
            .sections
            .iter()
            .position(|s| s.name == id && !s.is_global())
            .ok_or_else(|| format!("no model [{id}] in the preset"))?;
        self.sections.remove(i);
        Ok(())
    }

    /// What llama.cpp reads: section → key → value, the last line wins.
    /// Top keys belong to llama.cpp's default section (`""` here).
    pub fn effective(&self) -> BTreeMap<String, BTreeMap<String, String>> {
        let mut out: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        if !self.top.is_empty() {
            let m = out.entry(String::new()).or_default();
            for e in &self.top {
                m.insert(e.key.clone(), e.value.clone());
            }
        }
        for s in &self.sections {
            let m = out.entry(s.name.clone()).or_default();
            for e in &s.entries {
                m.insert(e.key.clone(), e.value.clone());
            }
        }
        out
    }

    /// `sha256:` over the records (comments included).
    pub fn digest(&self) -> String {
        let json = serde_json::to_vec(self).unwrap_or_default();
        format!("sha256:{:x}", Sha256::digest(&json))
    }

    pub fn notes_count(&self) -> usize {
        let count = |v: &[String]| v.iter().filter(|l| !l.is_empty()).count();
        let mut n = count(&self.trailing);
        for e in &self.top {
            n += count(&e.before) + usize::from(e.inline.is_some());
        }
        for s in &self.sections {
            n += count(&s.before) + usize::from(s.inline.is_some());
            for e in &s.entries {
                n += count(&e.before) + usize::from(e.inline.is_some());
            }
        }
        n
    }
}

// ---- parse / render ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PresetError {
    #[error("line {line}: {reason}")]
    Parse { line: usize, reason: String },
    #[error("the render is not equivalent to the original: {0}")]
    NotEquivalent(String),
    #[error("the records do not survive a render: {0}")]
    Unstable(String),
}

fn is_ident(k: &str) -> bool {
    let mut chars = k.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
}

/// Split `text ; comment` the way llama.cpp does: the value ends at the
/// first `;` or `#`, trailing blanks dropped.
fn split_comment(rest: &str) -> (&str, Option<&str>) {
    match rest.find([';', '#']) {
        Some(i) => (rest[..i].trim_end_matches([' ', '\t']), Some(&rest[i..])),
        None => (rest.trim_end_matches([' ', '\t']), None),
    }
}

/// Parse a preset INI (llama.cpp's grammar: header, `key = value`,
/// comment and blank lines).
pub fn import(text: &str) -> Result<RouterPreset, PresetError> {
    let mut p = RouterPreset::default();
    let mut pending: Vec<String> = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line_no = n + 1;
        let line = raw.trim_end_matches('\r');
        let t = line.trim_start_matches([' ', '\t']);
        if t.is_empty() {
            pending.push(String::new());
            continue;
        }
        if t.starts_with(';') || t.starts_with('#') {
            pending.push(t.trim_end_matches([' ', '\t']).to_string());
            continue;
        }
        if let Some(rest) = t.strip_prefix('[') {
            let close = rest.find(']').ok_or_else(|| PresetError::Parse {
                line: line_no,
                reason: "section header without `]`".into(),
            })?;
            let name = rest[..close].trim_matches([' ', '\t']).to_string();
            if name.is_empty() {
                return Err(PresetError::Parse {
                    line: line_no,
                    reason: "empty section name".into(),
                });
            }
            let after = rest[close + 1..].trim_start_matches([' ', '\t']);
            let inline = match after {
                "" => None,
                a if a.starts_with(';') || a.starts_with('#') => Some(a.trim_end_matches([' ', '\t']).to_string()),
                a => {
                    return Err(PresetError::Parse {
                        line: line_no,
                        reason: format!("text after the section header: {a:?}"),
                    })
                }
            };
            p.sections.push(Section {
                name,
                before: std::mem::take(&mut pending),
                inline,
                entries: Vec::new(),
            });
            continue;
        }
        let eq = t.find('=').ok_or_else(|| PresetError::Parse {
            line: line_no,
            reason: format!("not a header, key, comment or blank line: {t:?}"),
        })?;
        let key = t[..eq].trim_end_matches([' ', '\t']);
        if !is_ident(key) {
            return Err(PresetError::Parse {
                line: line_no,
                reason: format!("{key:?} is not a key llama.cpp accepts"),
            });
        }
        let rest = t[eq + 1..].trim_start_matches([' ', '\t']);
        let (value, inline) = split_comment(rest);
        let entry = Entry {
            key: key.to_string(),
            value: value.to_string(),
            before: std::mem::take(&mut pending),
            inline: inline.map(|c| c.trim_end_matches([' ', '\t']).to_string()),
        };
        match p.sections.last_mut() {
            Some(s) => s.entries.push(entry),
            None => p.top.push(entry),
        }
    }
    p.trailing = pending;
    Ok(p)
}

fn push_lines(out: &mut String, lines: &[String]) {
    for l in lines {
        out.push_str(l);
        out.push('\n');
    }
}

fn push_entry(out: &mut String, e: &Entry) {
    push_lines(out, &e.before);
    out.push_str(&e.key);
    out.push_str(" = ");
    out.push_str(&e.value);
    if let Some(c) = &e.inline {
        out.push(' ');
        out.push_str(c);
    }
    out.push('\n');
}

/// Render the records: `key = value`, one header per section, every note
/// back where it was.
pub fn render(p: &RouterPreset) -> String {
    let mut out = String::new();
    for e in &p.top {
        push_entry(&mut out, e);
    }
    for s in &p.sections {
        push_lines(&mut out, &s.before);
        out.push('[');
        out.push_str(&s.name);
        out.push(']');
        if let Some(c) = &s.inline {
            out.push(' ');
            out.push_str(c);
        }
        out.push('\n');
        for e in &s.entries {
            push_entry(&mut out, e);
        }
    }
    push_lines(&mut out, &p.trailing);
    out
}

fn diff_effective(
    a: &BTreeMap<String, BTreeMap<String, String>>,
    b: &BTreeMap<String, BTreeMap<String, String>>,
) -> Option<String> {
    for (s, keys) in a {
        let Some(other) = b.get(s) else {
            return Some(format!("[{s}] is missing"));
        };
        for (k, v) in keys {
            match other.get(k) {
                None => return Some(format!("[{s}] {k} is missing")),
                Some(w) if w != v => return Some(format!("[{s}] {k}: {v:?} became {w:?}")),
                _ => {}
            }
        }
        if let Some(k) = other.keys().find(|k| !keys.contains_key(*k)) {
            return Some(format!("[{s}] {k} appeared"));
        }
    }
    b.keys().find(|s| !a.contains_key(*s)).map(|s| format!("[{s}] appeared"))
}

/// Import, render, and prove the render reads the same to llama.cpp and
/// re-imports to the same records.
pub fn roundtrip(text: &str) -> Result<(RouterPreset, String), PresetError> {
    let p = import(text)?;
    let rendered = check_render(&p)?;
    let again = import(&rendered)?;
    if let Some(d) = diff_effective(&p.effective(), &again.effective()) {
        return Err(PresetError::NotEquivalent(d));
    }
    Ok((p, rendered))
}

/// Render and re-import: the records must come back unchanged.
pub fn check_render(p: &RouterPreset) -> Result<String, PresetError> {
    let rendered = render(p);
    let again = import(&rendered).map_err(|e| PresetError::Unstable(e.to_string()))?;
    if &again != p {
        let d = diff_effective(&p.effective(), &again.effective())
            .unwrap_or_else(|| "comments or order moved".into());
        return Err(PresetError::Unstable(d));
    }
    Ok(rendered)
}

/// What llama.cpp would refuse, or this deployment does not allow. Empty =
/// fine to write.
pub fn validate(p: &RouterPreset) -> Vec<String> {
    let mut out = Vec::new();
    let note_ok = |l: &str| l.is_empty() || l.starts_with(';') || l.starts_with('#');
    let check_entry = |out: &mut Vec<String>, where_: &str, e: &Entry| {
        if !is_ident(&e.key) {
            out.push(format!("{where_}: {:?} is not a key llama.cpp accepts", e.key));
        }
        if e.value.contains([';', '#', '\n', '\r']) {
            out.push(format!(
                "{where_} {}: a value cannot hold `;`, `#` or a newline (llama.cpp ends it there)",
                e.key
            ));
        }
        if e.value != e.value.trim() {
            out.push(format!("{where_} {}: leading or trailing blanks would be lost", e.key));
        }
        if e.before.iter().any(|l| !note_ok(l)) || e.inline.as_deref().is_some_and(|c| !note_ok(c) || c.contains('\n')) {
            out.push(format!("{where_} {}: a note must start with `;` or `#`", e.key));
        }
    };
    for e in &p.top {
        check_entry(&mut out, "top", e);
    }
    let mut seen = std::collections::BTreeSet::new();
    for s in &p.sections {
        let where_ = format!("[{}]", s.name);
        if !seen.insert(s.name.as_str()) {
            out.push(format!("{where_} appears twice"));
        }
        if s.name.contains([']', '\n', '\r', ';', '#']) || s.name != s.name.trim() {
            out.push(format!("{where_}: not a section name llama.cpp reads back"));
        }
        if s.before.iter().any(|l| !note_ok(l)) {
            out.push(format!("{where_}: a note must start with `;` or `#`"));
        }
        let mut keys = std::collections::BTreeSet::new();
        for e in &s.entries {
            check_entry(&mut out, &where_, e);
            if !keys.insert(e.key.as_str()) {
                out.push(format!("{where_} {} is set twice (llama.cpp keeps the last)", e.key));
            }
            if ROUTER_CONTROLLED.contains(&e.key.as_str()) {
                out.push(format!(
                    "{where_} {}: the router overrides this key (the section name is the model's alias)",
                    e.key
                ));
            }
            if ROUTER_ONLY.contains(&e.key.as_str()) {
                out.push(format!("{where_} {}: a router option, not a model argument", e.key));
            }
            if s.is_global() && (MODEL_KEYS.contains(&e.key.as_str()) || HF_KEYS.contains(&e.key.as_str())) {
                out.push(format!("{where_} {}: a model belongs in its own section", e.key));
            }
            if e.key == "load-on-startup" && !is_boolish(&e.value) {
                out.push(format!("{where_} load-on-startup: {:?} is not true/false", e.value));
            }
        }
        if !s.is_global() {
            let m = RouterModel::from_section(s);
            match (&m.model, &m.hf_repo) {
                (None, None) => out.push(format!("{where_}: names no `model` path (or `hf-repo`)")),
                (Some(path), _) => {
                    if !path.starts_with('/') {
                        out.push(format!("{where_} model: {path:?} is not absolute"));
                    } else if !path.ends_with(".gguf") {
                        out.push(format!(
                            "{where_} model: {path:?} is not a .gguf file (a split model is named by its first shard)"
                        ));
                    }
                }
                _ => {}
            }
            if let Some(mm) = &m.mmproj {
                if !mm.starts_with('/') {
                    out.push(format!("{where_} mmproj: {mm:?} is not absolute"));
                }
            }
        }
    }
    out
}

/// Readable `section key: before → after` lines between two presets.
pub fn changes(a: &RouterPreset, b: &RouterPreset) -> Vec<String> {
    let (ea, eb) = (a.effective(), b.effective());
    let mut out = Vec::new();
    for (s, keys) in &ea {
        match eb.get(s) {
            None => out.push(format!("[{s}] removed")),
            Some(other) => {
                for (k, v) in keys {
                    match other.get(k) {
                        None => out.push(format!("[{s}] {k}: {v} → (removed)")),
                        Some(w) if w != v => out.push(format!("[{s}] {k}: {v} → {w}")),
                        _ => {}
                    }
                }
                for (k, w) in other {
                    if !keys.contains_key(k) {
                        out.push(format!("[{s}] {k}: (new) → {w}"));
                    }
                }
            }
        }
    }
    for s in eb.keys().filter(|s| !ea.contains_key(*s)) {
        out.push(format!("[{s}] added"));
    }
    out
}

// ---- the router's own command line --------------------------------------------------

/// The router workload's launch, rendered to argv (item 1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterLaunch {
    pub binary: String,
    pub host: String,
    pub port: u16,
    /// Absolute path of the preset INI llmctl-next renders.
    pub preset: String,
    pub models_max: u32,
    /// Arguments every model inherits (`--flash-attn on`, …).
    #[serde(default)]
    pub extra: Vec<String>,
}

impl RouterLaunch {
    pub fn argv(&self) -> Vec<String> {
        let mut v = vec![
            self.binary.clone(),
            "--host".into(),
            self.host.clone(),
            "--port".into(),
            self.port.to_string(),
            "--models-preset".into(),
            self.preset.clone(),
            "--models-max".into(),
            self.models_max.to_string(),
            "--no-models-autoload".into(),
        ];
        v.extend(self.extra.iter().cloned());
        v
    }
}

fn flag_value<'a>(argv: &'a [String], names: &[&str]) -> Option<&'a str> {
    let mut found = None;
    for (i, a) in argv.iter().enumerate() {
        if let Some((k, v)) = a.split_once('=') {
            if names.contains(&k) {
                found = Some(v);
                continue;
            }
        }
        if names.contains(&a.as_str()) {
            found = argv.get(i + 1).map(String::as_str);
        }
    }
    found
}

/// `--models-preset <path>` from a router argv.
pub fn preset_path_from_argv(argv: &[String]) -> Option<String> {
    flag_value(argv, &["--models-preset"]).map(str::to_owned)
}

/// `--models-max N` from a router argv (llama.cpp's default when absent).
pub fn models_max_from_argv(argv: &[String]) -> Option<u32> {
    match flag_value(argv, &["--models-max"]) {
        None => Some(DEFAULT_MODELS_MAX),
        Some(v) => v.parse().ok(),
    }
}

/// What is wrong with a router command line for llmctl-next: it must load
/// nothing by itself. `env` is the unit's environment (`LLAMA_ARG_*`
/// stands for a flag). Empty = fine.
pub fn check_router_argv(argv: &[String], env: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    let has = |names: &[&str]| {
        argv.iter()
            .any(|a| names.contains(&a.as_str()) || a.split_once('=').is_some_and(|(k, _)| names.contains(&k)))
    };
    let env_has = |k: &str| env.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
    if has(&["-m", "--model", "-hf", "-hfr", "--hf-repo", "-mu", "--model-url", "-dr", "--docker-repo"])
        || env_has("LLAMA_ARG_MODEL").is_some()
        || env_has("LLAMA_ARG_HF_REPO").is_some()
    {
        out.push("names a model of its own: a router starts without -m (models come from the preset)".into());
    }
    if has(&["--models-dir"]) || env_has("LLAMA_ARG_MODELS_DIR").is_some() {
        out.push("--models-dir lists models the records do not hold; the preset is the one source".into());
    }
    match preset_path_from_argv(argv).or_else(|| env_has("LLAMA_ARG_MODELS_PRESET").map(str::to_owned)) {
        None => out.push("no --models-preset: the router would serve only its cache".into()),
        Some(p) if !p.starts_with('/') => out.push(format!("--models-preset {p} is not absolute")),
        _ => {}
    }
    let autoload_off = has(&["--no-models-autoload"])
        || env_has("LLAMA_ARG_MODELS_AUTOLOAD").is_some_and(|v| !is_truthy(v));
    if !autoload_off || has(&["--models-autoload"]) {
        out.push(
            "autoload is on: add --no-models-autoload, so a request cannot load a model behind the planner (decision 9)"
                .into(),
        );
    }
    match models_max_from_argv(argv) {
        None => out.push("--models-max is not a number".into()),
        Some(0) => out.push("--models-max 0 (unlimited): the planner needs a bound".into()),
        _ => {}
    }
    out
}

// ---- the router's model list ----------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Unloaded,
    Loading,
    Loaded,
    Sleeping,
    Downloading,
    #[serde(untagged)]
    Other(String),
}

impl ModelState {
    fn parse(s: &str) -> Self {
        match s {
            "unloaded" => ModelState::Unloaded,
            "loading" => ModelState::Loading,
            "loaded" => ModelState::Loaded,
            "sleeping" => ModelState::Sleeping,
            "downloading" => ModelState::Downloading,
            other => ModelState::Other(other.to_string()),
        }
    }

    /// Holds memory (or is about to): counts against `--models-max` and
    /// the budget. An unknown state counts too — unknown is never free.
    pub fn holds_memory(&self) -> bool {
        !matches!(self, ModelState::Unloaded | ModelState::Downloading)
    }
}

impl std::fmt::Display for ModelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelState::Unloaded => f.write_str("unloaded"),
            ModelState::Loading => f.write_str("loading"),
            ModelState::Loaded => f.write_str("loaded"),
            ModelState::Sleeping => f.write_str("sleeping"),
            ModelState::Downloading => f.write_str("downloading"),
            ModelState::Other(s) => f.write_str(s),
        }
    }
}

/// One row of `GET /models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterModelStatus {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub state: ModelState,
    /// The last load failed (`status.failed`).
    #[serde(default)]
    pub failed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i64>,
}

/// Parse `GET /models` (`{"data":[{"id","path","status":{"value",…}}]}`).
/// `None` when the body is not that shape.
pub fn parse_models(v: &serde_json::Value) -> Option<Vec<RouterModelStatus>> {
    let rows = v.get("data")?.as_array()?;
    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let id = r.get("id")?.as_str()?.to_string();
        let status = r.get("status");
        let state = status
            .and_then(|s| s.get("value"))
            .and_then(|s| s.as_str())
            .map(ModelState::parse)
            .unwrap_or_else(|| ModelState::Other("unknown".into()));
        out.push(RouterModelStatus {
            id,
            path: r.get("path").and_then(|p| p.as_str()).map(str::to_owned),
            state,
            failed: status
                .and_then(|s| s.get("failed"))
                .and_then(|f| f.as_bool())
                .unwrap_or(false),
            exit_code: status.and_then(|s| s.get("exit_code")).and_then(|c| c.as_i64()),
        });
    }
    Some(out)
}

// ---- the load decision (items 2 and 3) ------------------------------------------------

/// What the memory side of a load rests on.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryCheck {
    /// Workstation: the model's per-card need (measured or declared) against
    /// free VRAM on its cards, and its host-RAM share (CPU offload of MoE
    /// experts) against free RAM.
    Cards {
        need: Option<CardMib>,
        free: Option<CardMib>,
        host_ram_gib: Option<f64>,
        free_ram_gib: Option<f64>,
    },
    /// A Mac: one unified pool. Every model that holds memory plus this one
    /// must fit the budget at their peak figures.
    Budget {
        budget_gib: Option<u64>,
        need_gib: Option<f64>,
        /// Models that hold memory now, with their peak figures.
        loaded: Vec<(String, Option<f64>)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadDecision {
    /// Nothing to do: the router lists it loaded.
    AlreadyLoaded(String),
    /// Load it; the reason goes into the preview.
    Load(String),
    Refuse(String),
}

impl LoadDecision {
    pub fn reason(&self) -> &str {
        match self {
            LoadDecision::AlreadyLoaded(r) | LoadDecision::Load(r) | LoadDecision::Refuse(r) => r,
        }
    }
}

/// Decide one load. `statuses` is the router's list (`None` = it did not
/// answer). Fails closed on every unknown.
pub fn decide_load(
    model: &str,
    statuses: Option<&[RouterModelStatus]>,
    models_max: u32,
    memory: &MemoryCheck,
) -> LoadDecision {
    let Some(statuses) = statuses else {
        return LoadDecision::Refuse("the router does not answer GET /models".into());
    };
    let Some(me) = statuses.iter().find(|s| s.id == model) else {
        return LoadDecision::Refuse(format!(
            "the router does not list {model}; is it in the preset (and reloaded)?"
        ));
    };
    match me.state {
        ModelState::Loaded => return LoadDecision::AlreadyLoaded(format!("{model} is loaded already")),
        ModelState::Loading => return LoadDecision::Refuse(format!("{model} is loading already")),
        ModelState::Sleeping => return LoadDecision::AlreadyLoaded(format!("{model} is loaded (sleeping)")),
        ModelState::Downloading => return LoadDecision::Refuse(format!("{model} is still downloading")),
        _ => {}
    }
    let holding: Vec<&RouterModelStatus> =
        statuses.iter().filter(|s| s.id != model && s.state.holds_memory()).collect();
    if models_max > 0 && holding.len() as u32 >= models_max {
        return LoadDecision::Refuse(format!(
            "the router is at --models-max {models_max} ({}); unload one first, or the router would evict one by itself",
            holding.iter().map(|s| s.id.as_str()).collect::<Vec<_>>().join(", ")
        ));
    }
    match memory {
        MemoryCheck::Cards {
            need,
            free,
            host_ram_gib,
            free_ram_gib,
        } => {
            let Some(need) = need.as_ref().filter(|n| !n.is_empty()) else {
                return LoadDecision::Refuse(format!(
                    "{model} was never measured and has no declared need; declare one (`llamacpp need`) before the first load"
                ));
            };
            let Some(free) = free.as_ref().filter(|f| !f.is_empty()) else {
                return LoadDecision::Refuse("free VRAM could not be read".into());
            };
            let on_cards: CardMib = free
                .iter()
                .filter(|(c, _)| need.contains_key(c))
                .map(|(c, m)| (*c, *m))
                .collect();
            if !crate::fit::fits(Some(need), Some(free)) {
                return LoadDecision::Refuse(format!(
                    "does not fit: needs {} plus {HEADROOM_MIB} MiB headroom, free {}",
                    format_mib(need),
                    if on_cards.is_empty() { "unknown on those cards".into() } else { format_mib(&on_cards) }
                ));
            }
            let mut reason = format!(
                "fits: needs {}, free {} ({HEADROOM_MIB} MiB headroom kept)",
                format_mib(need),
                format_mib(&on_cards)
            );
            if let Some(ram) = host_ram_gib.filter(|r| *r > 0.0) {
                match free_ram_gib {
                    Some(avail) if *avail >= ram + HOST_RAM_HEADROOM_GIB => {
                        reason.push_str(&format!("; host RAM {ram:.1} GiB of {avail:.1} GiB free"));
                    }
                    Some(avail) => {
                        return LoadDecision::Refuse(format!(
                            "its CPU offload needs {ram:.1} GiB host RAM plus {HOST_RAM_HEADROOM_GIB} GiB headroom, {avail:.1} GiB free"
                        ))
                    }
                    None => return LoadDecision::Refuse("free host RAM could not be read".into()),
                }
            }
            LoadDecision::Load(reason)
        }
        MemoryCheck::Budget {
            budget_gib,
            need_gib,
            loaded,
        } => {
            let Some(budget) = budget_gib else {
                return LoadDecision::Refuse("no memory budget is set for this host".into());
            };
            let Some(need) = need_gib else {
                return LoadDecision::Refuse(format!("{model} has no peak memory figure"));
            };
            let mut total = need.max(0.0).ceil() as u64;
            for (id, peak) in loaded {
                match peak {
                    Some(g) => total += g.max(0.0).ceil() as u64,
                    None => {
                        return LoadDecision::Refuse(format!(
                            "{id} holds memory and has no peak figure: the sum is unknown"
                        ))
                    }
                }
            }
            if total <= *budget {
                LoadDecision::Load(format!("fits the memory budget: {total}G of {budget}G at peak"))
            } else {
                LoadDecision::Refuse(format!(
                    "over the memory budget: {total}G at peak > {budget}G ({} loaded)",
                    if loaded.is_empty() {
                        "nothing else".to_string()
                    } else {
                        loaded.iter().map(|(i, _)| i.as_str()).collect::<Vec<_>>().join(", ")
                    }
                ))
            }
        }
    }
}

// ---- the pre-load guard (port of llama-server-guarded) --------------------------------

/// The shard files a split GGUF needs (`<name>-00001-of-00003.gguf` →
/// all three), or just the file itself.
pub fn shard_files(model: &str) -> Vec<String> {
    let Some(stem) = model.strip_suffix(".gguf") else {
        return vec![model.to_string()];
    };
    // `…-NNNNN-of-MMMMM`
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    if let [total, "of", index, base] = parts.as_slice() {
        if total.len() == 5 && index.len() == 5 && total.bytes().all(|b| b.is_ascii_digit())
            && index.bytes().all(|b| b.is_ascii_digit())
        {
            if let Ok(n) = total.parse::<u32>() {
                return (1..=n).map(|i| format!("{base}-{i:05}-of-{total}.gguf")).collect();
            }
        }
    }
    vec![model.to_string()]
}

/// Where `hf download --local-dir` stages a model's files while they
/// arrive: `<parent of the model dir>/.cache/huggingface/download/<dir name>`.
pub fn download_stage_dir(model: &str) -> Option<String> {
    let p = std::path::Path::new(model);
    let dir = p.parent()?;
    let sub = dir.file_name()?.to_str()?;
    let root = dir.parent()?;
    Some(
        root.join(".cache/huggingface/download")
            .join(sub)
            .display()
            .to_string(),
    )
}

/// The guard's verdict from what the daemon saw on disk: missing shards,
/// `.incomplete` files in the stage dir, and free VRAM against a floor.
pub fn preload_problems(
    model: &str,
    missing: &[String],
    incomplete: &[String],
    floor_mib: Option<u64>,
    free: Option<&CardMib>,
) -> Vec<String> {
    let mut out = Vec::new();
    if !missing.is_empty() {
        let shards = shard_files(model).len();
        if shards > 1 {
            out.push(format!(
                "incomplete split model ({shards} shards expected); missing: {}",
                missing.join(", ")
            ));
        } else {
            out.push(format!("model file does not exist: {model}"));
        }
    }
    if !incomplete.is_empty() {
        out.push(format!(
            "the model is still downloading ({} .incomplete file(s) in its download stage)",
            incomplete.len()
        ));
    }
    if let Some(floor) = floor_mib.filter(|f| *f > 0) {
        match free {
            Some(free) => {
                for (card, mib) in free {
                    if *mib < floor {
                        out.push(format!(
                            "gpu {card} has only {mib} MiB free (floor {floor} MiB): something else holds VRAM"
                        ));
                    }
                }
            }
            None => out.push(format!("free VRAM could not be read (floor {floor} MiB)")),
        }
    }
    out
}

// ---- model hooks (load/unload companions) ----------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelHookWhen {
    PreLoad,
    PostLoad,
    PreUnload,
    PostUnload,
}

impl std::fmt::Display for ModelHookWhen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModelHookWhen::PreLoad => "pre_load",
            ModelHookWhen::PostLoad => "post_load",
            ModelHookWhen::PreUnload => "pre_unload",
            ModelHookWhen::PostUnload => "post_unload",
        })
    }
}

/// A companion command of one router model's load or unload (step 7: the
/// `ds4-llmctl-sync adopt/detach` pair of `gpt-oss-120b`). Declared by the
/// owner in `<data>/llamacpp/<workload>.hooks.json` only: no API registers
/// a command, as for workload hooks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHook {
    pub model: String,
    pub name: String,
    pub when: ModelHookWhen,
    pub argv: Vec<String>,
    #[serde(default = "default_hook_timeout")]
    pub timeout_secs: u64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
}

fn default_hook_timeout() -> u64 {
    60
}

/// The hooks file's shape.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHooks {
    #[serde(default)]
    pub hooks: Vec<ModelHook>,
}

impl ModelHooks {
    pub fn validate(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for h in &self.hooks {
            if h.name.trim().is_empty() || !names.insert((h.model.as_str(), h.name.as_str())) {
                out.push(format!("hook {:?} of {} is empty or used twice", h.name, h.model));
            }
            match h.argv.first() {
                None => out.push(format!("hook {} of {} has no command", h.name, h.model)),
                Some(p) if !p.starts_with('/') => out.push(format!(
                    "hook {} of {}: {p:?} is not an absolute path (hooks run without a shell or PATH lookup)",
                    h.name, h.model
                )),
                _ => {}
            }
            if h.timeout_secs == 0 || h.timeout_secs > crate::inventory::MAX_HOOK_TIMEOUT_SECS {
                out.push(format!("hook {} of {}: timeout {}s is out of range", h.name, h.model, h.timeout_secs));
            }
        }
        out
    }

    pub fn at<'a>(&'a self, model: &'a str, when: ModelHookWhen) -> impl Iterator<Item = &'a ModelHook> + 'a {
        self.hooks.iter().filter(move |h| h.model == model && h.when == when)
    }
}

// ---- catalog view ----------------------------------------------------------------------

/// One router model as the catalog sees it (M5c consumes this).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecord {
    pub workload: String,
    pub id: String,
    pub engine_kind: EngineKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    pub load_on_startup: bool,
    #[serde(default)]
    pub args: Vec<(String, String)>,
}

pub fn model_records(workload: &str, preset: &RouterPreset) -> Vec<ModelRecord> {
    preset
        .models()
        .into_iter()
        .map(|m| ModelRecord {
            workload: workload.to_string(),
            id: m.id,
            engine_kind: EngineKind::LlamaCppRouter,
            path: m.model,
            hf_repo: m.hf_repo,
            mmproj: m.mmproj,
            load_on_startup: m.load_on_startup,
            args: m.args,
        })
        .collect()
}

// ---- llama.cpp lanes for llama-swap (Macs, owner decision 8) ---------------------------

/// One llama.cpp lane: `llama-server -m <gguf>` started by llama-swap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlamaLane {
    /// llama-swap macro naming the `llama-server` binary (the pinned
    /// release-archive install), without `${}`.
    pub binary_macro: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    /// Extra flags as `(name, value)`; `"true"` renders a bare flag.
    #[serde(default)]
    pub args: Vec<(String, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl LlamaLane {
    pub fn validate(&self) -> Vec<String> {
        let mut out = Vec::new();
        let ok_token = |s: &str| !s.is_empty() && !s.contains(|c: char| c.is_whitespace() || "\"'`$\\;|&<>".contains(c));
        if !is_ident(&self.binary_macro) {
            out.push(format!("macro name {:?} is not plain", self.binary_macro));
        }
        if !self.model.starts_with('/') || !self.model.ends_with(".gguf") || !ok_token(&self.model) {
            out.push(format!("model {:?} must be an absolute .gguf path without blanks or quotes", self.model));
        }
        if let Some(m) = &self.mmproj {
            if !m.starts_with('/') || !ok_token(m) {
                out.push(format!("mmproj {m:?} must be an absolute path without blanks or quotes"));
            }
        }
        for (k, v) in &self.args {
            let k = k.trim_start_matches('-');
            if !is_ident(k) {
                out.push(format!("flag {k:?} is not a plain name"));
            }
            if ["m", "model", "port", "host", "mmproj"].contains(&k) {
                out.push(format!("flag {k} is set by the lane itself"));
            }
            if !ok_token(v) {
                out.push(format!("flag {k}: value {v:?} must be one token without quotes"));
            }
        }
        out
    }

    /// The lane's `cmd`.
    pub fn cmd(&self) -> String {
        let mut parts = vec![format!("${{{}}}", self.binary_macro), "-m".into(), self.model.clone()];
        if let Some(m) = &self.mmproj {
            parts.push("--mmproj".into());
            parts.push(m.clone());
        }
        parts.extend(["--host".into(), "127.0.0.1".into(), "--port".into(), "${PORT}".into()]);
        for (k, v) in &self.args {
            let k = k.trim_start_matches('-');
            parts.push(format!("--{k}"));
            if v != "true" {
                parts.push(v.clone());
            }
        }
        parts.join(" ")
    }

    /// The lane's fields in llama-swap's order.
    pub fn fields(&self) -> Vec<(String, serde_json::Value)> {
        let mut f = Vec::new();
        if let Some(n) = &self.name {
            f.push(("name".to_string(), serde_json::json!(n)));
        }
        f.push(("cmd".to_string(), serde_json::json!(self.cmd())));
        f.push(("proxy".to_string(), serde_json::json!("http://127.0.0.1:${PORT}")));
        if let Some(t) = self.ttl {
            f.push(("ttl".to_string(), serde_json::json!(t)));
        }
        f
    }
}

/// Keys llama.cpp reads only from a preset, never as flags.
pub fn is_preset_only(key: &str) -> bool {
    PRESET_ONLY.contains(&key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_end_at_comment_markers_like_llama_cpp() {
        let p = import("[m]\nmodel = /a.gguf ; note\nc=4096#x\n").unwrap();
        let s = p.section("m").unwrap();
        assert_eq!(s.entries[0].value, "/a.gguf");
        assert_eq!(s.entries[0].inline.as_deref(), Some("; note"));
        assert_eq!(s.entries[1].value, "4096");
        assert_eq!(s.entries[1].inline.as_deref(), Some("#x"));
    }

    #[test]
    fn shards_are_listed_from_the_first() {
        assert_eq!(
            shard_files("/m/K-UD-IQ1_S-00001-of-00003.gguf"),
            vec![
                "/m/K-UD-IQ1_S-00001-of-00003.gguf",
                "/m/K-UD-IQ1_S-00002-of-00003.gguf",
                "/m/K-UD-IQ1_S-00003-of-00003.gguf"
            ]
        );
        assert_eq!(shard_files("/m/one.gguf"), vec!["/m/one.gguf"]);
        assert_eq!(
            download_stage_dir("/home/y/models/gpt-oss-120b/x.gguf").as_deref(),
            Some("/home/y/models/.cache/huggingface/download/gpt-oss-120b")
        );
    }
}
