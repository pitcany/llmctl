//! llama-swap lanes as typed records (M4b step 4, owner decision 3).
//!
//! The Macs run every engine as a llama-swap lane, configured in one
//! hand-edited `config.yaml` watched with `-watch-config`. llmctl-next takes
//! that file over in three moves:
//!
//! 1. **Import** ([`import`]): parse the file into a [`SwapConfig`] — ordered
//!    globals, macros, the swap `matrix`, and one [`Lane`] per model with its
//!    fields in file order. Every comment is kept as a [`Notes`] entry keyed
//!    by the path of the key it precedes (or trails), so a render puts the
//!    owner's history notes back where they were.
//! 2. **Prove** ([`roundtrip`]): render the records and parse the render; the
//!    YAML value tree must equal the original's, and re-importing the render
//!    must give the same records. Nothing is written from records that failed
//!    this (the M2/M3 "compare configuration" rule), and every later write
//!    re-proves its own render ([`check_render`]).
//! 3. **Validate** ([`validate`]): the rules llama-swap itself enforces at
//!    (re)load — macros defined, preload lanes exist, the `matrix` DSL
//!    (ported from `~/local-llm/lib/swapmatrix.py`, itself a port of
//!    llama-swap v230 `matrix.go`/`matrix_dsl.go`). A config llama-swap would
//!    reject never reaches the file.
//!
//! Values are held as order-preserving YAML values (they persist as JSON in
//! file order). YAML tags, non-string keys and non-finite floats are refused
//! at import rather than approximated.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_yaml::{Mapping, Value};

use crate::recipe::EngineKind;

/// The whole llama-swap config as records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwapConfig {
    /// Top-level keys in file order (`macros`, `matrix`, `models` included).
    pub top_order: Vec<String>,
    /// Every top-level key that is not `macros`, `matrix` or `models`,
    /// verbatim (`healthCheckTimeout`, `hooks`, `groups`, ...).
    pub globals: Vec<(String, Value)>,
    /// `macros:` in file order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub macros: Vec<(String, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix: Option<Matrix>,
    pub lanes: Vec<Lane>,
    #[serde(default)]
    pub notes: Notes,
}

/// The file's comments, by position.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Notes {
    /// Before the first key.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub header: Vec<String>,
    /// Full-line comments before the key at this path (`/models/lane-a`,
    /// `/models/lane-a/ttl`, `/macros/mtplx`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub before: BTreeMap<String, Vec<String>>,
    /// A trailing `# comment` on the line of the key at this path.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub inline: BTreeMap<String, String>,
    /// After the last key.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trailing: Vec<String>,
    /// Layout the owner chose, kept so a render diffs only where records
    /// changed: scalars written in double quotes, sequences written in
    /// flow style (`[a, b]`), keys preceded by a blank line.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub quoted: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub flow: BTreeSet<String>,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub blank_before: BTreeSet<String>,
}

impl Notes {
    pub fn count(&self) -> usize {
        self.header.len()
            + self.trailing.len()
            + self.inline.len()
            + self.before.values().map(Vec::len).sum::<usize>()
    }
}

/// llama-swap v230 `matrix:` — which lanes may run together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    /// Short var id → lane id.
    pub vars: Vec<(String, String)>,
    /// Var id → eviction cost (default 1).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evict_costs: Vec<(String, Value)>,
    /// Set name → DSL (`a & b | (c & +other)`).
    pub sets: Vec<(String, String)>,
    /// Every key under `matrix:` in file order (render order).
    pub order: Vec<String>,
    /// Any other key under `matrix:`, verbatim.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra: Vec<(String, Value)>,
}

/// One llama-swap model entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lane {
    pub id: String,
    /// What the lane's `cmd` runs, inferred at import (macros expanded);
    /// `Other("forwarder")` for the mesh forwarders that proxy another host.
    pub engine_kind: EngineKind,
    /// The entry's keys in file order (`name`, `cmd`, `proxy`, `ttl`, ...).
    pub fields: Vec<(String, Value)>,
}

impl Lane {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn str_field(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    pub fn cmd(&self) -> Option<&str> {
        self.str_field("cmd")
    }

    pub fn ttl(&self) -> Option<i64> {
        self.get("ttl").and_then(Value::as_i64)
    }

    /// Set (or add, at the end) one field.
    pub fn set(&mut self, key: &str, value: Value) {
        match self.fields.iter_mut().find(|(k, _)| k == key) {
            Some((_, v)) => *v = value,
            None => self.fields.push((key.to_string(), value)),
        }
    }
}

impl SwapConfig {
    pub fn lane(&self, id: &str) -> Option<&Lane> {
        self.lanes.iter().find(|l| l.id == id)
    }

    pub fn lane_mut(&mut self, id: &str) -> Option<&mut Lane> {
        self.lanes.iter_mut().find(|l| l.id == id)
    }

    pub fn macro_value(&self, name: &str) -> Option<&str> {
        self.macros
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// A lane's `cmd` with the config's macros expanded (`${PORT}` and
    /// `${MODEL_ID}` are llama-swap's own and stay as written).
    pub fn expanded_cmd(&self, lane: &Lane) -> Option<String> {
        let mut text = lane.cmd()?.to_string();
        // A macro may name another macro; llama-swap expands until stable.
        for _ in 0..8 {
            let mut next = text.clone();
            for (k, v) in &self.macros {
                next = next.replace(&format!("${{{k}}}"), v);
            }
            if next == text {
                break;
            }
            text = next;
        }
        Some(text)
    }

    pub fn set_macro(&mut self, name: &str, value: &str) {
        match self.macros.iter_mut().find(|(k, _)| k == name) {
            Some((_, v)) => *v = value.to_string(),
            None => self.macros.push((name.to_string(), value.to_string())),
        }
        if !self.top_order.iter().any(|k| k == "macros") {
            self.top_order.insert(0, "macros".into());
        }
    }

    /// `sha256:<hex>` of the canonical JSON form (what a write confirms).
    pub fn digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self).expect("records serialize");
        format!("sha256:{:x}", Sha256::digest(bytes))
    }
}

// ---- errors --------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SwapError {
    #[error("config is not valid YAML: {0}")]
    Parse(String),
    #[error("unsupported YAML at {path}: {why}")]
    Unsupported { path: String, why: String },
    #[error("render does not round-trip: {}", .0.join("; "))]
    NotEquivalent(Vec<String>),
}

// ---- import --------------------------------------------------------------------

/// Parse a llama-swap config into records (no validation; see [`validate`]).
pub fn import(text: &str) -> Result<SwapConfig, SwapError> {
    let root = parse_value(text)?;
    let Value::Mapping(top) = &root else {
        return Err(SwapError::Unsupported {
            path: "/".into(),
            why: "top level is not a mapping".into(),
        });
    };
    let mut cfg = SwapConfig {
        top_order: Vec::new(),
        globals: Vec::new(),
        macros: Vec::new(),
        matrix: None,
        lanes: Vec::new(),
        notes: scan_notes(text),
    };
    for (key, value) in top {
        let key = key.as_str().expect("checked: string keys").to_string();
        cfg.top_order.push(key.clone());
        match key.as_str() {
            "macros" => cfg.macros = string_pairs(value, "/macros")?,
            "matrix" => cfg.matrix = Some(import_matrix(value)?),
            "models" => {
                let Value::Mapping(models) = value else {
                    return Err(SwapError::Unsupported {
                        path: "/models".into(),
                        why: "not a mapping".into(),
                    });
                };
                for (id, entry) in models {
                    let id = id.as_str().expect("checked").to_string();
                    let Value::Mapping(entry) = entry else {
                        return Err(SwapError::Unsupported {
                            path: format!("/models/{id}"),
                            why: "not a mapping".into(),
                        });
                    };
                    let fields = entry
                        .iter()
                        .map(|(k, v)| (k.as_str().expect("checked").to_string(), v.clone()))
                        .collect();
                    let mut lane = Lane {
                        id,
                        engine_kind: EngineKind::Other("unknown".into()),
                        fields,
                    };
                    lane.engine_kind = infer_lane_kind(&lane, &cfg.macros);
                    cfg.lanes.push(lane);
                }
            }
            _ => cfg.globals.push((key, value.clone())),
        }
    }
    Ok(cfg)
}

/// Parse YAML, refusing what records cannot hold faithfully.
pub fn parse_value(text: &str) -> Result<Value, SwapError> {
    let y: Value = serde_yaml::from_str(text).map_err(|e| SwapError::Parse(e.to_string()))?;
    check_supported(&y, "")?;
    Ok(y)
}

fn check_supported(y: &Value, path: &str) -> Result<(), SwapError> {
    match y {
        Value::Number(n) if n.as_f64().is_some_and(|f| !f.is_finite()) => {
            Err(SwapError::Unsupported {
                path: path.to_string(),
                why: "non-finite number".into(),
            })
        }
        Value::Sequence(seq) => seq
            .iter()
            .enumerate()
            .try_for_each(|(i, v)| check_supported(v, &format!("{path}/{i}"))),
        Value::Mapping(m) => m.iter().try_for_each(|(k, v)| match k {
            Value::String(k) => check_supported(v, &format!("{path}/{k}")),
            other => Err(SwapError::Unsupported {
                path: path.to_string(),
                why: format!("non-string key {other:?}"),
            }),
        }),
        Value::Tagged(t) => Err(SwapError::Unsupported {
            path: path.to_string(),
            why: format!("YAML tag {}", t.tag),
        }),
        _ => Ok(()),
    }
}

fn string_pairs(v: &Value, path: &str) -> Result<Vec<(String, String)>, SwapError> {
    let Value::Mapping(m) = v else {
        return Err(SwapError::Unsupported {
            path: path.into(),
            why: "not a mapping".into(),
        });
    };
    m.iter()
        .map(|(k, v)| {
            let k = k.as_str().expect("checked").to_string();
            match v {
                Value::String(s) => Ok((k, s.clone())),
                other => Err(SwapError::Unsupported {
                    path: format!("{path}/{k}"),
                    why: format!("expected a string, got {other:?}"),
                }),
            }
        })
        .collect()
}

fn import_matrix(v: &Value) -> Result<Matrix, SwapError> {
    let Value::Mapping(m) = v else {
        return Err(SwapError::Unsupported {
            path: "/matrix".into(),
            why: "not a mapping".into(),
        });
    };
    let mut out = Matrix {
        vars: Vec::new(),
        evict_costs: Vec::new(),
        sets: Vec::new(),
        order: Vec::new(),
        extra: Vec::new(),
    };
    for (k, v) in m {
        let k = k.as_str().expect("checked").to_string();
        out.order.push(k.clone());
        match k.as_str() {
            "vars" => out.vars = string_pairs(v, "/matrix/vars")?,
            "sets" => out.sets = string_pairs(v, "/matrix/sets")?,
            "evict_costs" => match v {
                Value::Mapping(c) => {
                    out.evict_costs = c
                        .iter()
                        .map(|(k, v)| (k.as_str().expect("checked").to_string(), v.clone()))
                        .collect()
                }
                _ => {
                    return Err(SwapError::Unsupported {
                        path: "/matrix/evict_costs".into(),
                        why: "not a mapping".into(),
                    })
                }
            },
            _ => out.extra.push((k, v.clone())),
        }
    }
    Ok(out)
}

/// What a lane runs, from its `cmd` with macros expanded.
pub fn infer_lane_kind(lane: &Lane, macros: &[(String, String)]) -> EngineKind {
    let Some(cmd) = lane.cmd() else {
        return EngineKind::Other("unknown".into());
    };
    let mut text = cmd.to_string();
    for (k, v) in macros {
        text = text.replace(&format!("${{{k}}}"), v);
    }
    let t = text.to_ascii_lowercase();
    if t.contains("m1-worker.sh forward") || t.contains(" forward --port") {
        EngineKind::Other("forwarder".into())
    } else if t.contains("mlx_vlm") || t.contains("mlx-vlm") {
        EngineKind::MlxVlm
    } else if t.contains("mlx-serve") {
        EngineKind::MlxServe
    } else if t.contains("mtplx") {
        EngineKind::Mtplx
    } else if t.contains("llama-server") {
        EngineKind::LlamaCpp
    } else if t.contains("ds4-server") || t.contains("/ds4") || t.contains("ds41") {
        EngineKind::Ds4
    } else if t.contains("vllm") {
        EngineKind::Vllm
    } else {
        EngineKind::Other("unknown".into())
    }
}

// ---- comments (line scanner) -------------------------------------------------------

/// A YAML key at the start of a line: `(indent, key)`.
fn line_key(line: &str) -> Option<(usize, String)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    let rest = &line[indent..];
    if rest.starts_with('#') || rest.starts_with('-') || rest.is_empty() {
        return None;
    }
    let (key, after) = if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        (stripped[..end].to_string(), &stripped[end + 1..])
    } else {
        let end = rest.find(':')?;
        let k = rest[..end].trim();
        if k.contains(' ') {
            return None;
        }
        (k.to_string(), &rest[end..])
    };
    let after = after.strip_prefix(':')?;
    if key.is_empty() || !(after.is_empty() || after.starts_with(' ')) {
        return None;
    }
    Some((indent, key))
}

/// Does this key line open a block scalar (`key: |`, `key: >-  # c`)?
fn opens_block_scalar(line: &str) -> bool {
    let code = strip_inline_comment(line).0;
    let v = code.split_once(':').map(|(_, v)| v.trim()).unwrap_or("");
    !v.is_empty() && (v.starts_with('|') || v.starts_with('>')) && v.len() <= 3
}

/// Split `code  # comment` outside quotes.
fn strip_inline_comment(line: &str) -> (&str, Option<&str>) {
    let bytes = line.as_bytes();
    let (mut sq, mut dq) = (false, false);
    for i in 0..bytes.len() {
        match bytes[i] {
            b'\'' if !dq => sq = !sq,
            b'"' if !sq => dq = !dq,
            b'#' if !sq && !dq && (i == 0 || bytes[i - 1] == b' ' || bytes[i - 1] == b'\t') => {
                return (line[..i].trim_end(), Some(line[i + 1..].trim()));
            }
            _ => {}
        }
    }
    (line, None)
}

fn comment_text(line: &str) -> String {
    let t = line.trim_start();
    let t = t.strip_prefix('#').unwrap_or(t);
    t.strip_prefix(' ').unwrap_or(t).trim_end().to_string()
}

/// Key every comment to a path: full-line comments to the next key line,
/// trailing comments to their own key line. Comments before a sequence
/// item wait for the next key (rare; never dropped).
fn scan_notes(text: &str) -> Notes {
    let mut n = Notes::default();
    let mut pending: Vec<String> = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut seen_key = false;
    let mut in_block: Option<usize> = None;
    let mut blank = false;
    for line in text.lines() {
        let li = line.len() - line.trim_start().len();
        if let Some(ind) = in_block {
            if line.trim().is_empty() || li > ind {
                continue;
            }
            in_block = None;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            // A blank line before a comment block belongs to the key the
            // block precedes.
            pending.push(comment_text(line));
            continue;
        }
        if trimmed.is_empty() {
            blank = true;
            continue;
        }
        let (_, inline) = strip_inline_comment(line);
        let was_blank = std::mem::replace(&mut blank, false);
        let blank = was_blank;
        match line_key(line) {
            Some((indent, key)) => {
                while stack.last().is_some_and(|(i, _)| *i >= indent) {
                    stack.pop();
                }
                stack.push((indent, key));
                let path: String = stack.iter().map(|(_, k)| format!("/{k}")).collect();
                // (A blank between the header comments and the first key
                // counts too.)
                if blank && (seen_key || !pending.is_empty()) {
                    n.blank_before.insert(path.clone());
                }
                let code = strip_inline_comment(line).0;
                let value = code.split_once(':').map(|(_, v)| v.trim()).unwrap_or("");
                if value.starts_with('"') {
                    n.quoted.insert(path.clone());
                } else if value.starts_with('[') {
                    n.flow.insert(path.clone());
                }
                if !pending.is_empty() {
                    if !seen_key {
                        n.header.append(&mut pending);
                    } else {
                        n.before.entry(path.clone()).or_default().append(&mut pending);
                    }
                }
                seen_key = true;
                if let Some(c) = inline {
                    n.inline.insert(path, c.to_string());
                }
                if opens_block_scalar(line) {
                    in_block = Some(indent);
                }
            }
            None => {
                if trimmed.starts_with("- \"") {
                    let path: String = stack.iter().map(|(_, k)| format!("/{k}")).collect();
                    n.quoted.insert(format!("{path}/-"));
                }
                // A sequence item or continuation line: a trailing comment
                // sticks to the enclosing key, after any it already has.
                if let Some(c) = inline {
                    let path: String = stack.iter().map(|(_, k)| format!("/{k}")).collect();
                    n.before.entry(format!("{path}/-")).or_default().push(c.to_string());
                }
            }
        }
    }
    n.trailing = pending;
    n
}

// ---- render --------------------------------------------------------------------

/// The ordered YAML tree records stand for (notes excluded).
pub fn to_value(cfg: &SwapConfig) -> Value {
    let mut top = Mapping::new();
    for key in &cfg.top_order {
        match key.as_str() {
            "macros" => {
                if !cfg.macros.is_empty() {
                    top.insert(
                        key.as_str().into(),
                        Value::Mapping(
                            cfg.macros
                                .iter()
                                .map(|(k, v)| (Value::from(k.as_str()), Value::from(v.as_str())))
                                .collect(),
                        ),
                    );
                }
            }
            "matrix" => {
                if let Some(m) = &cfg.matrix {
                    top.insert(key.as_str().into(), matrix_value(m));
                }
            }
            "models" => {
                top.insert(
                    key.as_str().into(),
                    Value::Mapping(
                        cfg.lanes
                            .iter()
                            .map(|l| {
                                (
                                    Value::from(l.id.as_str()),
                                    Value::Mapping(
                                        l.fields
                                            .iter()
                                            .map(|(k, v)| (Value::from(k.as_str()), v.clone()))
                                            .collect(),
                                    ),
                                )
                            })
                            .collect(),
                    ),
                );
            }
            other => {
                if let Some((_, v)) = cfg.globals.iter().find(|(k, _)| k == other) {
                    top.insert(other.into(), v.clone());
                }
            }
        }
    }
    Value::Mapping(top)
}

fn matrix_value(m: &Matrix) -> Value {
    let pairs = |p: &[(String, String)]| {
        Value::Mapping(
            p.iter()
                .map(|(k, v)| (Value::from(k.as_str()), Value::from(v.as_str())))
                .collect(),
        )
    };
    let mut out = Mapping::new();
    let mut order = m.order.clone();
    for k in ["vars", "evict_costs", "sets"] {
        if !order.iter().any(|o| o == k) {
            order.push(k.into());
        }
    }
    for k in order {
        match k.as_str() {
            "vars" => {
                out.insert("vars".into(), pairs(&m.vars));
            }
            "sets" => {
                out.insert("sets".into(), pairs(&m.sets));
            }
            "evict_costs" if !m.evict_costs.is_empty() => {
                out.insert(
                    "evict_costs".into(),
                    Value::Mapping(
                        m.evict_costs
                            .iter()
                            .map(|(k, v)| (Value::from(k.as_str()), v.clone()))
                            .collect(),
                    ),
                );
            }
            "evict_costs" => {}
            other => {
                if let Some((_, v)) = m.extra.iter().find(|(k, _)| k == other) {
                    out.insert(other.into(), v.clone());
                }
            }
        }
    }
    Value::Mapping(out)
}

/// Render records to llama-swap YAML: block style, two-space indents,
/// multi-line strings as literal blocks, each comment back at its key.
/// Deterministic: equal records render byte-identically.
pub fn render(cfg: &SwapConfig) -> String {
    let mut out = String::new();
    push_comments(&mut out, &cfg.notes.header, 0);
    if let Value::Mapping(top) = to_value(cfg) {
        for (k, v) in top.iter() {
            let k = k.as_str().expect("string keys");
            render_entry(&mut out, k, v, 0, &format!("/{k}"), &cfg.notes);
        }
    }
    if !cfg.notes.trailing.is_empty() {
        out.push('\n');
        push_comments(&mut out, &cfg.notes.trailing, 0);
    }
    out
}

fn push_comments(out: &mut String, lines: &[String], indent: usize) {
    let pad = " ".repeat(indent);
    for c in lines {
        if c.is_empty() {
            out.push_str(&format!("{pad}#\n"));
        } else {
            out.push_str(&format!("{pad}# {c}\n"));
        }
    }
}

fn with_inline(line: String, path: &str, notes: &Notes) -> String {
    match notes.inline.get(path) {
        Some(c) => format!("{line}  # {c}\n"),
        None => format!("{line}\n"),
    }
}

/// `key: value` at `indent` (with its comments), recursing into mappings
/// and sequences.
fn render_entry(out: &mut String, key: &str, v: &Value, indent: usize, path: &str, notes: &Notes) {
    let pad = " ".repeat(indent);
    if notes.blank_before.contains(path) {
        out.push('\n');
    }
    if let Some(before) = notes.before.get(path) {
        push_comments(out, before, indent);
    }
    let k = yaml_key(key);
    match v {
        Value::Mapping(m) if !m.is_empty() => {
            out.push_str(&with_inline(format!("{pad}{k}:"), path, notes));
            let lane_level = path.starts_with("/models/") && path.matches('/').count() == 2;
            let _ = lane_level;
            for (ck, cv) in m.iter() {
                let ck = ck.as_str().expect("string keys");
                render_entry(out, ck, cv, indent + 2, &format!("{path}/{ck}"), notes);
            }
        }
        Value::Sequence(a)
            if !a.is_empty()
                && notes.flow.contains(path)
                && a.iter().all(|i| !matches!(i, Value::Mapping(_) | Value::Sequence(_))) =>
        {
            let items: Vec<String> = a.iter().map(flow_scalar).collect();
            out.push_str(&with_inline(format!("{pad}{k}: [{}]", items.join(", ")), path, notes));
        }
        Value::Sequence(a) if !a.is_empty() => {
            out.push_str(&with_inline(format!("{pad}{k}:"), path, notes));
            for item in a {
                render_item(out, item, indent + 2, path, notes);
            }
            if let Some(tr) = notes.before.get(&format!("{path}/-")) {
                push_comments(out, tr, indent + 2);
            }
        }
        Value::String(text) if text.contains('\n') && block_safe(text) => {
            let chomp = if text.ends_with("\n\n") {
                "+"
            } else if text.ends_with('\n') {
                ""
            } else {
                "-"
            };
            out.push_str(&with_inline(format!("{pad}{k}: |{chomp}"), path, notes));
            let body = text.strip_suffix('\n').unwrap_or(text);
            for line in body.split('\n') {
                if line.is_empty() {
                    out.push('\n');
                } else {
                    out.push_str(&format!("{pad}  {line}\n"));
                }
            }
        }
        Value::String(s) if notes.quoted.contains(path) => out.push_str(&with_inline(
            format!("{pad}{k}: {}", serde_json::to_string(s).expect("string serializes")),
            path,
            notes,
        )),
        other => out.push_str(&with_inline(format!("{pad}{k}: {}", yaml_scalar(other)), path, notes)),
    }
}

/// A scalar inside `[...]`: commas and brackets force quotes there.
fn flow_scalar(v: &Value) -> String {
    match v {
        Value::String(s) if s.contains([',', '[', ']', '{', '}']) => {
            serde_json::to_string(s).expect("string serializes")
        }
        other => yaml_scalar(other),
    }
}

fn render_item(out: &mut String, v: &Value, indent: usize, path: &str, notes: &Notes) {
    let pad = " ".repeat(indent);
    match v {
        Value::Mapping(m) if !m.is_empty() => {
            for (i, (k, cv)) in m.iter().enumerate() {
                let k = k.as_str().expect("string keys");
                let mut entry = String::new();
                render_entry(&mut entry, k, cv, indent + 2, &format!("{path}/-/{k}"), notes);
                if i == 0 {
                    out.push_str(&format!("{pad}- {}", entry.trim_start()));
                } else {
                    out.push_str(&entry);
                }
            }
        }
        Value::String(s) if notes.quoted.contains(&format!("{path}/-")) => out.push_str(&format!(
            "{pad}- {}\n",
            serde_json::to_string(s).expect("string serializes")
        )),
        other => out.push_str(&format!("{pad}- {}\n", yaml_scalar(other))),
    }
}

/// A literal block can carry this string unchanged.
fn block_safe(text: &str) -> bool {
    !text.starts_with(' ')
        && !text.starts_with('\n')
        && !text.ends_with("\n\n")
        && !text.contains('\r')
        && !text.contains('\t')
        && text.split('\n').all(|l| !l.ends_with(' '))
}

/// One scalar (or an empty collection) in flow form, quoted exactly when
/// YAML needs it (double quotes, like the hand-written file).
fn yaml_scalar(v: &Value) -> String {
    match v {
        Value::Mapping(m) if m.is_empty() => "{}".into(),
        Value::Sequence(a) if a.is_empty() => "[]".into(),
        Value::String(s) => {
            let plain = serde_yaml::to_string(v).unwrap_or_default();
            let plain = plain.trim_end_matches('\n');
            if s.contains('\n') || plain.starts_with('\'') || plain.starts_with('"') || plain.contains('\n') {
                serde_json::to_string(s).expect("string serializes")
            } else {
                plain.to_string()
            }
        }
        other => serde_yaml::to_string(other)
            .unwrap_or_default()
            .trim_end_matches('\n')
            .to_string(),
    }
}

fn yaml_key(k: &str) -> String {
    let plain = !k.is_empty()
        && k.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
        && !k.chars().next().is_some_and(|c| c.is_ascii_digit() || c == '-')
        && !matches!(k, "true" | "false" | "null" | "yes" | "no" | "on" | "off" | "y" | "n");
    if plain {
        k.to_string()
    } else {
        serde_json::to_string(k).expect("key serializes")
    }
}

// ---- round trip ------------------------------------------------------------------

/// Import `text`, render the records, and prove the render parses to the
/// same YAML value tree and re-imports to the same records.
pub fn roundtrip(text: &str) -> Result<(SwapConfig, String), SwapError> {
    let original = parse_value(text)?;
    let cfg = import(text)?;
    let rendered = render(&cfg);
    let again = parse_value(&rendered)?;
    let diffs = value_diffs(&original, &again, "");
    if !diffs.is_empty() {
        return Err(SwapError::NotEquivalent(diffs));
    }
    let reimported = import(&rendered)?;
    if reimported != cfg {
        return Err(SwapError::NotEquivalent(vec![
            "re-importing the render changes the records (order or notes)".into(),
        ]));
    }
    Ok((cfg, rendered))
}

/// Prove a render of `cfg` parses back to `cfg` (every write calls this on
/// what it is about to write).
pub fn check_render(cfg: &SwapConfig) -> Result<String, SwapError> {
    let rendered = render(cfg);
    let again = import(&rendered)?;
    let diffs = value_diffs(&to_value(cfg), &to_value(&again), "");
    if !diffs.is_empty() {
        return Err(SwapError::NotEquivalent(diffs));
    }
    if again.notes != cfg.notes {
        return Err(SwapError::NotEquivalent(vec!["comments moved in the render".into()]));
    }
    Ok(rendered)
}

/// Paths where two YAML trees differ (mapping order ignored).
pub fn value_diffs(a: &Value, b: &Value, path: &str) -> Vec<String> {
    match (a, b) {
        (Value::Mapping(x), Value::Mapping(y)) => {
            let key = |k: &Value| k.as_str().map(str::to_owned).unwrap_or_else(|| format!("{k:?}"));
            let keys: BTreeSet<String> = x.keys().chain(y.keys()).map(key).collect();
            keys.into_iter()
                .flat_map(|k| match (x.get(k.as_str()), y.get(k.as_str())) {
                    (Some(p), Some(q)) => value_diffs(p, q, &format!("{path}/{k}")),
                    (Some(_), None) => vec![format!("{path}/{k}: missing from the render")],
                    (None, Some(_)) => vec![format!("{path}/{k}: only in the render")],
                    (None, None) => vec![],
                })
                .collect()
        }
        (Value::Sequence(x), Value::Sequence(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .enumerate()
            .flat_map(|(i, (p, q))| value_diffs(p, q, &format!("{path}/{i}")))
            .collect(),
        _ if a == b => vec![],
        _ => vec![format!("{path}: {a:?} != {b:?}")],
    }
}

// ---- validation (what llama-swap enforces on load) --------------------------------

/// Macros llama-swap provides itself.
const BUILTIN_MACROS: &[&str] = &["PORT", "MODEL_ID"];

/// Every problem llama-swap would refuse the config for. Empty = valid.
pub fn validate(cfg: &SwapConfig) -> Vec<String> {
    let mut errs = Vec::new();
    let ids: BTreeSet<&str> = cfg.lanes.iter().map(|l| l.id.as_str()).collect();
    if ids.len() != cfg.lanes.len() {
        errs.push("duplicate lane ids".into());
    }
    let macros: BTreeSet<&str> = cfg.macros.iter().map(|(k, _)| k.as_str()).collect();
    for lane in &cfg.lanes {
        if lane.id.trim().is_empty() {
            errs.push("a lane has an empty id".into());
        }
        match lane.cmd() {
            Some(c) if !c.trim().is_empty() => {}
            _ => errs.push(format!("lane {}: cmd is missing or empty", lane.id)),
        }
        for key in ["cmd", "cmdStop", "proxy", "checkEndpoint"] {
            if let Some(text) = lane.str_field(key) {
                for m in macro_refs(text) {
                    if !macros.contains(m.as_str()) && !BUILTIN_MACROS.contains(&m.as_str()) {
                        errs.push(format!("lane {}: {key} uses undefined macro ${{{m}}}", lane.id));
                    }
                }
            }
        }
        if let Some(ttl) = lane.get("ttl") {
            if ttl.as_i64().is_none_or(|t| t < 0) {
                errs.push(format!("lane {}: ttl must be a non-negative integer", lane.id));
            }
        }
    }
    if let Some(pre) = cfg
        .globals
        .iter()
        .find(|(k, _)| k == "hooks")
        .and_then(|(_, h)| h.get("on_startup"))
        .and_then(|o| o.get("preload"))
        .and_then(Value::as_sequence)
    {
        for p in pre {
            match p.as_str() {
                Some(id) if ids.contains(id) => {}
                _ => errs.push(format!("hooks.on_startup.preload names unknown lane {p:?}")),
            }
        }
    }
    if let Some(m) = &cfg.matrix {
        if let Err(e) = expand_matrix(m, &ids) {
            errs.push(format!("matrix: {e}"));
        }
    }
    errs
}

/// `${name}` references in a string.
fn macro_refs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("${") {
        let after = &rest[i + 2..];
        match after.find('}') {
            Some(j) => {
                out.push(after[..j].to_string());
                rest = &after[j + 1..];
            }
            None => break,
        }
    }
    out
}

// ---- matrix DSL (port of swapmatrix.py / llama-swap v230) ---------------------------

const MAX_EXPANSIONS: usize = 1000;
const DEFAULT_EVICT_COST: i64 = 1;

/// One expanded matrix set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpandedSet {
    pub name: String,
    /// Real lane ids, sorted.
    pub lanes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Ident(String),
    Ref(String),
    And,
    Or,
    Open,
    Close,
    Eof,
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.')
}

fn tokenize(dsl: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = dsl.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if let Some(t) = match c {
            '&' => Some(Tok::And),
            '|' => Some(Tok::Or),
            '(' => Some(Tok::Open),
            ')' => Some(Tok::Close),
            _ => None,
        } {
            out.push(t);
            i += 1;
        } else if c == '+' {
            i += 1;
            let start = i;
            while i < chars.len() && is_ident(chars[i]) {
                i += 1;
            }
            if i == start {
                return Err(format!("expected set name after '+' at position {start}"));
            }
            out.push(Tok::Ref(chars[start..i].iter().collect()));
        } else if is_ident(c) {
            let start = i;
            while i < chars.len() && is_ident(chars[i]) {
                i += 1;
            }
            out.push(Tok::Ident(chars[start..i].iter().collect()));
        } else {
            return Err(format!("unexpected character {c:?} at position {i}"));
        }
    }
    out.push(Tok::Eof);
    Ok(out)
}

#[derive(Debug, Clone)]
enum Node {
    Ident(String),
    Ref(String),
    And(Vec<Node>),
    Or(Vec<Node>),
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos]
    }
    fn take(&mut self) -> Tok {
        let t = self.toks[self.pos].clone();
        if t != Tok::Eof {
            self.pos += 1;
        }
        t
    }
    fn parse(mut self) -> Result<Node, String> {
        let n = self.expr()?;
        if *self.peek() != Tok::Eof {
            return Err(format!("unexpected token {:?} after expression", self.peek()));
        }
        Ok(n)
    }
    fn expr(&mut self) -> Result<Node, String> {
        let mut kids = vec![self.conj()?];
        while *self.peek() == Tok::Or {
            self.take();
            kids.push(self.conj()?);
        }
        Ok(if kids.len() == 1 { kids.pop().unwrap() } else { Node::Or(kids) })
    }
    fn conj(&mut self) -> Result<Node, String> {
        let mut kids = vec![self.atom()?];
        while *self.peek() == Tok::And {
            self.take();
            kids.push(self.atom()?);
        }
        Ok(if kids.len() == 1 { kids.pop().unwrap() } else { Node::And(kids) })
    }
    fn atom(&mut self) -> Result<Node, String> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                self.take();
                Ok(Node::Ident(s))
            }
            Tok::Ref(s) => {
                self.take();
                Ok(Node::Ref(s))
            }
            Tok::Open => {
                self.take();
                let n = self.expr()?;
                if self.take() != Tok::Close {
                    return Err("missing closing parenthesis".into());
                }
                Ok(n)
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

fn expand_node(n: &Node, refs: &BTreeMap<String, Vec<Vec<String>>>) -> Result<Vec<Vec<String>>, String> {
    match n {
        Node::Ident(s) => Ok(vec![vec![s.clone()]]),
        Node::Ref(s) => refs
            .get(s)
            .cloned()
            .ok_or_else(|| format!("unknown set reference +{s}")),
        Node::Or(kids) => {
            let mut out = Vec::new();
            for k in kids {
                out.extend(expand_node(k, refs)?);
                if out.len() > MAX_EXPANSIONS {
                    return Err(format!("DSL expansion exceeded {MAX_EXPANSIONS} combinations"));
                }
            }
            Ok(out)
        }
        Node::And(kids) => {
            let mut product: Vec<Vec<String>> = vec![vec![]];
            for k in kids {
                let right = expand_node(k, refs)?;
                if product.len() * right.len() > MAX_EXPANSIONS {
                    return Err(format!("DSL expansion exceeded {MAX_EXPANSIONS} combinations"));
                }
                product = product
                    .iter()
                    .flat_map(|l| right.iter().map(move |r| [l.clone(), r.clone()].concat()))
                    .collect();
            }
            Ok(product)
        }
    }
}

/// Every combination one DSL string allows (identifiers deduplicated, sorted).
pub fn expand_dsl(
    dsl: &str,
    refs: &BTreeMap<String, Vec<Vec<String>>>,
) -> Result<Vec<Vec<String>>, String> {
    if dsl.trim().is_empty() {
        return Err("empty DSL expression".into());
    }
    let tree = Parser { toks: tokenize(dsl)?, pos: 0 }.parse()?;
    Ok(expand_node(&tree, refs)?
        .into_iter()
        .map(|c| {
            let s: BTreeSet<String> = c.into_iter().collect();
            s.into_iter().collect()
        })
        .collect())
}

fn dsl_refs(dsl: &str) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for t in tokenize(dsl)? {
        if let Tok::Ref(r) = t {
            if !out.contains(&r) {
                out.push(r);
            }
        }
    }
    Ok(out)
}

fn dependency_order(sets: &[(String, String)]) -> Result<Vec<String>, String> {
    let names: BTreeSet<&str> = sets.iter().map(|(n, _)| n.as_str()).collect();
    let mut deps: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (name, dsl) in sets {
        let d = dsl_refs(dsl).map_err(|e| format!("set {name:?}: {e}"))?;
        if let Some(m) = d.iter().find(|r| !names.contains(r.as_str())) {
            return Err(format!("set {name:?} references undefined set {m:?}"));
        }
        deps.insert(name, d);
    }
    fn visit(
        name: &str,
        deps: &BTreeMap<&str, Vec<String>>,
        state: &mut BTreeMap<String, u8>,
        order: &mut Vec<String>,
    ) -> Result<(), String> {
        match state.get(name) {
            Some(1) => return Err(format!("circular reference detected involving set {name:?}")),
            Some(2) => return Ok(()),
            _ => {}
        }
        state.insert(name.to_string(), 1);
        for d in &deps[name] {
            visit(d, deps, state, order)?;
        }
        state.insert(name.to_string(), 2);
        order.push(name.to_string());
        Ok(())
    }
    let mut state = BTreeMap::new();
    let mut order = Vec::new();
    for (name, _) in sets {
        visit(name, &deps, &mut state, &mut order)?;
    }
    Ok(order)
}

/// Lanes that no matrix set holds. llama-swap runs such a lane alone: its
/// first request stops every lane running then (seen on m5, 2026-11-19).
/// `None` when the config has no matrix or the matrix does not expand.
pub fn lanes_outside_sets(cfg: &SwapConfig) -> Option<BTreeSet<String>> {
    let ids: BTreeSet<&str> = cfg.lanes.iter().map(|l| l.id.as_str()).collect();
    let sets = expand_matrix(cfg.matrix.as_ref()?, &ids).ok()?;
    let held: BTreeSet<&str> = sets.iter().flat_map(|s| s.lanes.iter().map(String::as_str)).collect();
    Some(ids.difference(&held).map(|s| (*s).to_owned()).collect())
}

/// Validate a matrix like llama-swap does and return its expanded sets in
/// solver order (`+ref` targets first, otherwise definition order).
pub fn expand_matrix(m: &Matrix, lanes: &BTreeSet<&str>) -> Result<Vec<ExpandedSet>, String> {
    if m.sets.is_empty() {
        return Err("matrix must define at least one set".into());
    }
    if m.vars.is_empty() {
        return Err("matrix must define at least one var".into());
    }
    let vars: BTreeMap<&str, &str> = m.vars.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    for (key, lane) in &m.vars {
        let ok = !key.is_empty() && key.len() <= 8 && key.chars().all(|c| c.is_ascii_alphanumeric());
        if !ok {
            return Err(format!("var key {key:?} must be alphanumeric and 1-8 characters"));
        }
        if !lanes.contains(lane.as_str()) {
            return Err(format!("var key {key:?} references unknown model {lane:?}"));
        }
    }
    for (key, cost) in &m.evict_costs {
        if cost.as_bool().is_some() || cost.as_i64().is_none_or(|c| c <= 0) {
            return Err(format!("evict_cost for {key:?} must be a positive integer, got {cost:?}"));
        }
        if !vars.contains_key(key.as_str()) {
            return Err(format!("evict_costs: unknown var ID {key:?}"));
        }
    }
    let dsl_of: BTreeMap<&str, &str> = m.sets.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let mut resolved: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
    let mut out = Vec::new();
    for name in dependency_order(&m.sets)? {
        let combos =
            expand_dsl(dsl_of[name.as_str()], &resolved).map_err(|e| format!("set {name:?}: {e}"))?;
        for combo in &combos {
            if let Some(u) = combo.iter().find(|i| !vars.contains_key(i.as_str())) {
                return Err(format!("set {name:?}: unknown var ID {u:?}"));
            }
            let mut lanes: Vec<String> = combo.iter().map(|i| vars[i.as_str()].to_string()).collect();
            lanes.sort();
            out.push(ExpandedSet {
                name: name.clone(),
                lanes,
            });
        }
        resolved.insert(name.clone(), combos);
        if out.len() > MAX_EXPANSIONS {
            return Err(format!(
                "total expanded combinations ({}) exceed limit of {MAX_EXPANSIONS}",
                out.len()
            ));
        }
    }
    Ok(out)
}

/// What llama-swap unloads to start `target` with `running` loaded
/// (upstream `matrixSolver.Solve`): `(evicted lanes, chosen set, cost)`.
pub fn solve(
    expanded: &[ExpandedSet],
    costs: &BTreeMap<String, i64>,
    target: &str,
    running: &[String],
) -> (Vec<String>, String, i64) {
    let cost_of = |r: &String| costs.get(r).copied().unwrap_or(DEFAULT_EVICT_COST);
    if running.iter().any(|r| r == target) {
        let set = expanded
            .iter()
            .find(|s| s.lanes.iter().any(|l| l == target) && running.iter().all(|r| s.lanes.contains(r)))
            .map(|s| s.name.clone())
            .unwrap_or_default();
        return (Vec::new(), set, 0);
    }
    let candidates: Vec<&ExpandedSet> =
        expanded.iter().filter(|s| s.lanes.iter().any(|l| l == target)).collect();
    if candidates.is_empty() {
        return (running.to_vec(), String::new(), running.iter().map(cost_of).sum());
    }
    let mut best = candidates[0];
    let mut best_cost = -1;
    for s in candidates {
        let c: i64 = running.iter().filter(|r| !s.lanes.contains(r)).map(cost_of).sum();
        if best_cost < 0 || c < best_cost {
            best = s;
            best_cost = c;
        }
    }
    (
        running.iter().filter(|r| !best.lanes.contains(r)).cloned().collect(),
        best.name.clone(),
        best_cost,
    )
}

/// What a request for one lane makes llama-swap do, read from the records.
/// Matrix sets named `risky*` may exceed the memory budget on purpose.
pub const RISKY_SET_PREFIX: &str = "risky";

/// The memory budget against every combination the matrix allows (the
/// Mac rule, ported from `llmreg.matrix_budget_lines`): unified memory
/// has no cards to fit on, so what may run together must fit one budget.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BudgetReport {
    /// `PASS` / `FAIL` / `WARN` / `INFO` lines for the owner.
    pub lines: Vec<String>,
    /// Sets over the budget that are not `risky*`: `(name, peak GiB)`.
    pub over: Vec<(String, u64)>,
    /// Lanes with no peak figure: every sum they are in is a lower bound.
    pub unknown: Vec<String>,
}

/// Sum each allowed combination at the lanes' peak figures (GiB, rounded
/// up; a forwarder weighs nothing: its engine is on another machine).
pub fn budget_report(cfg: &SwapConfig, peaks: &BTreeMap<String, f64>, budget_gib: u64) -> BudgetReport {
    let mut report = BudgetReport::default();
    let Some(matrix) = &cfg.matrix else {
        report.lines.push("INFO no matrix: llama-swap groups decide what runs together; nothing is summed".into());
        return report;
    };
    let ids: BTreeSet<&str> = cfg.lanes.iter().map(|l| l.id.as_str()).collect();
    let sets = match expand_matrix(matrix, &ids) {
        Ok(sets) => sets,
        Err(e) => {
            report.lines.push(format!("WARN the matrix does not expand ({e}); nothing is summed"));
            return report;
        }
    };
    let weight = |lane: &str| -> Option<u64> {
        let forwarder = cfg
            .lane(lane)
            .is_some_and(|l| matches!(&l.engine_kind, EngineKind::Other(k) if k == "forwarder"));
        if forwarder {
            return Some(0);
        }
        peaks.get(lane).map(|g| g.max(0.0).ceil() as u64)
    };
    let mut fitting: Vec<(u64, String)> = Vec::new();
    let total_sets = sets.len();
    for set in &sets {
        let label = format!("matrix set {} [{}]", set.name, set.lanes.join(", "));
        let mut total = 0u64;
        for lane in &set.lanes {
            match weight(lane) {
                Some(g) => total += g,
                None => {
                    report
                        .lines
                        .push(format!("WARN {label}: no peak figure for {lane} (the sum is a lower bound)"));
                    if !report.unknown.contains(lane) {
                        report.unknown.push(lane.clone());
                    }
                }
            }
        }
        if total <= budget_gib {
            fitting.push((total, label));
        } else if set.name.starts_with(RISKY_SET_PREFIX) {
            report.lines.push(format!(
                "INFO {label} peaks at {total}G > {budget_gib}G budget (accepted: {RISKY_SET_PREFIX}* set)"
            ));
        } else {
            report
                .lines
                .push(format!("FAIL {label} peaks at {total}G > {budget_gib}G budget"));
            report.over.push((set.name.clone(), total));
        }
    }
    if let Some((top, label)) = fitting.iter().max() {
        let others = if fitting.len() < total_sets { " other" } else { "" };
        report.lines.push(format!(
            "PASS matrix: {}{others} set combination(s) fit the {budget_gib}G budget (largest: {} at {top}G)",
            fitting.len(),
            label.trim_start_matches("matrix set ")
        ));
    }
    report.unknown.sort();
    report
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapPrediction {
    /// Running lanes llama-swap unloads first.
    pub evicts: Vec<String>,
    /// The matrix set it settles on (empty without a matrix).
    pub set: String,
    pub cost: i64,
    /// The lane is running already: nothing is loaded or evicted.
    pub already_loaded: bool,
}

/// Predict a lane swap (M4c): which of `running` llama-swap evicts to load
/// `target`. With a `matrix:` this is its solver; without one llama-swap
/// keeps a single lane, so everything else goes. A config that still uses
/// `groups:` is not modelled and gets no prediction rather than a wrong one.
pub fn predict_swap(
    cfg: &SwapConfig,
    target: &str,
    running: &[String],
) -> Result<SwapPrediction, String> {
    if cfg.lane(target).is_none() {
        return Err(format!("no lane {target:?} in the records"));
    }
    let already_loaded = running.iter().any(|r| r == target);
    let Some(matrix) = &cfg.matrix else {
        if cfg.globals.iter().any(|(k, _)| k == "groups") {
            return Err("this config uses `groups:`, which the lane records do not model".into());
        }
        let evicts: Vec<String> = running.iter().filter(|r| *r != target).cloned().collect();
        let evicts = if already_loaded { Vec::new() } else { evicts };
        return Ok(SwapPrediction {
            cost: evicts.len() as i64,
            evicts,
            set: String::new(),
            already_loaded,
        });
    };
    let ids: BTreeSet<&str> = cfg.lanes.iter().map(|l| l.id.as_str()).collect();
    let expanded = expand_matrix(matrix, &ids)?;
    let (evicts, set, cost) = solve(&expanded, &resolved_costs(matrix), target, running);
    Ok(SwapPrediction {
        evicts,
        set,
        cost,
        already_loaded,
    })
}

/// Eviction costs keyed by real lane id (config keys are var ids).
pub fn resolved_costs(m: &Matrix) -> BTreeMap<String, i64> {
    let vars: BTreeMap<&str, &str> = m.vars.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    m.evict_costs
        .iter()
        .filter_map(|(k, v)| {
            Some((vars.get(k.as_str()).copied().unwrap_or(k).to_string(), v.as_i64()?))
        })
        .collect()
}
