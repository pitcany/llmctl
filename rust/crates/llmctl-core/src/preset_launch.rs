//! Preset schema, launch render and router-profile env handling (Rust M5c
//! item 1). Ports of, and byte-compatible with:
//!
//! - `llmctl/presets/schema.py` (`Model`, schema v1),
//! - `llmctl/services/preset_loader.py::model_to_launch_spec`,
//! - `llmctl/integrations/vllm_env.py::render_vllm_env` and
//!   `launcher_env.py::launcher_env_lines`,
//! - `llmctl_stack/router_profiles.py` (`parse_env`, `diff_env`,
//!   `merge_env`, the generated-from header).
//!
//! Pure functions: the caller resolves host facts (the Python root, the
//! CUDA root, `HF_HOME`) and passes them in as a [`LauncherEnv`]; nothing
//! here reads the environment or a file.
//!
//! Known YAML difference (documented, not reachable from the live files):
//! PyYAML reads YAML 1.1, `serde_yaml` YAML 1.2. `yes`/`no`/`on`/`off` and
//! a float without a dot (`1e-5`) load differently; the parity fixtures
//! cover every preset file in use.

use serde::{Deserialize, Serialize};
use serde_yaml::Value;

pub const SCHEMA_VERSION: i64 = 1;

pub const CANONICAL_QUANTIZATIONS: &[&str] =
    &["awq", "bnb", "compressed-tensors", "fp8", "gguf", "gptq", "none"];

/// Every key the v1 schema knows (Python `extra="forbid"`).
pub const PRESET_FIELDS: &[&str] = &[
    "alias",
    "served_name",
    "model_id",
    "quantization",
    "vllm_quantization_flag",
    "tensor_parallel_size",
    "max_model_len",
    "family",
    "param_count_b",
    "architectures",
    "max_num_seqs",
    "max_num_batched_tokens",
    "gpu_memory_utilization",
    "kv_cache_dtype",
    "dtype",
    "trust_remote_code",
    "host",
    "port",
    "tool_parser",
    "reasoning_parser",
    "extra_args",
    "python_root",
    "spec_config",
    "model_ref",
    "tq",
    "shortcuts",
    "schema_version",
];

/// One preset, schema v1. Field for field the Python `Model`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preset {
    pub alias: String,
    pub served_name: String,
    pub model_id: String,
    pub quantization: String,
    pub vllm_quantization_flag: String,
    pub tensor_parallel_size: i64,
    pub max_model_len: i64,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub param_count_b: Option<f64>,
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub max_num_seqs: Option<i64>,
    #[serde(default)]
    pub max_num_batched_tokens: Option<i64>,
    #[serde(default)]
    pub gpu_memory_utilization: Option<f64>,
    #[serde(default)]
    pub kv_cache_dtype: Option<String>,
    #[serde(default)]
    pub dtype: Option<String>,
    #[serde(default)]
    pub trust_remote_code: Option<bool>,
    #[serde(default)]
    pub host: Option<String>,
    /// Vestigial (Python: not read by any launch path).
    #[serde(default)]
    pub port: Option<i64>,
    #[serde(default)]
    pub tool_parser: Option<String>,
    #[serde(default)]
    pub reasoning_parser: Option<String>,
    #[serde(default)]
    pub extra_args: Option<String>,
    #[serde(default)]
    pub python_root: Option<String>,
    /// vLLM `--speculative-config` as a mapping; insertion order is kept
    /// (the env renders it as compact JSON in file order, as Python does).
    #[serde(default)]
    pub spec_config: Option<Value>,
    #[serde(default)]
    pub model_ref: Option<String>,
    /// Vestigial (TurboQuant is chosen per launch).
    #[serde(default)]
    pub tq: Option<bool>,
    #[serde(default)]
    pub shortcuts: Vec<String>,
    #[serde(default = "schema_v1")]
    pub schema_version: i64,
}

fn schema_v1() -> i64 {
    SCHEMA_VERSION
}

/// `^[a-z0-9][a-z0-9_.-]*$` (Python `paths._ALIAS_RE`).
pub fn valid_alias(alias: &str) -> bool {
    let b = alias.as_bytes();
    !b.is_empty()
        && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'_' | b'.' | b'-'))
}

impl Preset {
    /// Validate a loaded YAML mapping the way the Python schema does.
    pub fn from_value(v: &Value) -> Result<Preset, String> {
        let Value::Mapping(m) = v else {
            return Err("top-level YAML must be a mapping".into());
        };
        for (k, _) in m {
            match k.as_str() {
                Some(key) if PRESET_FIELDS.contains(&key) => {}
                Some(key) => return Err(format!("unknown field {key:?} (schema v1 forbids extras)")),
                None => return Err(format!("non-string key {k:?}")),
            }
        }
        // `null` for a list field is an error in pydantic; serde would too.
        let p: Preset = serde_yaml::from_value(v.clone()).map_err(|e| e.to_string())?;
        p.validate()?;
        Ok(p)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !valid_alias(&self.alias) {
            return Err(format!(
                "alias must match '^[a-z0-9][a-z0-9_.-]*$'; got {:?}",
                self.alias
            ));
        }
        for s in &self.shortcuts {
            if !valid_alias(s) {
                return Err(format!("shortcut must match '^[a-z0-9][a-z0-9_.-]*$'; got {s:?}"));
            }
        }
        if !CANONICAL_QUANTIZATIONS.contains(&self.quantization.as_str()) {
            return Err(format!(
                "quantization must be one of {CANONICAL_QUANTIZATIONS:?}; got {:?}",
                self.quantization
            ));
        }
        if !(1..=8).contains(&self.tensor_parallel_size) {
            return Err(format!(
                "tensor_parallel_size must be in 1..8; got {}",
                self.tensor_parallel_size
            ));
        }
        if let Some(u) = self.gpu_memory_utilization {
            if !(u > 0.0 && u <= 1.0) {
                return Err(format!("gpu_memory_utilization must be in (0, 1]; got {}", py_float(u)));
            }
        }
        if self.max_model_len < 1 {
            return Err(format!("max_model_len must be >= 1; got {}", self.max_model_len));
        }
        if let Some(n) = self.max_num_seqs {
            if n < 1 {
                return Err(format!("max_num_seqs must be >= 1; got {n}"));
            }
        }
        if let Some(p) = self.port {
            if !(1..=65535).contains(&p) {
                return Err(format!("port must be in 1..65535; got {p}"));
            }
        }
        if self.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "unsupported schema_version {}; this build only reads v{SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        if let Some(sc) = &self.spec_config {
            if !matches!(sc, Value::Mapping(_)) {
                return Err("spec_config must be a mapping".into());
            }
        }
        Ok(())
    }
}

/// `settings.yaml` `vllm.defaults` (Python `VLLMDefaultsConfig`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VllmDefaults {
    pub gpus: Option<String>,
    pub tensor_parallel: i64,
    pub port: i64,
    pub host: String,
    pub max_model_len: i64,
    pub gpu_memory_utilization: f64,
    pub prefix_cache: bool,
    pub chunked_prefill: bool,
    pub nccl_p2p_disable: bool,
    pub max_num_seqs: Option<i64>,
    pub max_batched_tokens: Option<i64>,
}

impl Default for VllmDefaults {
    fn default() -> Self {
        Self {
            gpus: None,
            tensor_parallel: 2,
            port: 8003,
            host: "0.0.0.0".into(),
            max_model_len: 32768,
            gpu_memory_utilization: 0.85,
            prefix_cache: true,
            chunked_prefill: true,
            nccl_p2p_disable: false,
            max_num_seqs: None,
            max_batched_tokens: None,
        }
    }
}

impl VllmDefaults {
    /// `vllm.defaults` out of a whole `settings.yaml` value. Only that
    /// subtree is read (the file holds a token elsewhere).
    pub fn from_settings(settings: &Value) -> Result<Self, String> {
        let Some(d) = settings.get("vllm").and_then(|v| v.get("defaults")) else {
            return Ok(Self::default());
        };
        if d.is_null() {
            return Ok(Self::default());
        }
        serde_yaml::from_value(d.clone()).map_err(|e| format!("settings vllm.defaults: {e}"))
    }
}

/// The fully resolved launch (Python `VLLMLaunchSpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LaunchSpec {
    pub model: String,
    pub served_name: String,
    pub tensor_parallel: i64,
    pub gpus: String,
    pub port: i64,
    pub host: String,
    pub max_model_len: i64,
    pub gpu_memory_utilization: f64,
    pub quantization: Option<String>,
    pub kv_cache_type: Option<String>,
    pub dtype: Option<String>,
    pub trust_remote_code: Option<bool>,
    pub tool_parser: Option<String>,
    pub max_num_seqs: Option<i64>,
    pub max_batched_tokens: Option<i64>,
    pub prefix_cache: bool,
    pub chunked_prefill: bool,
    /// A mapping (rendered as compact JSON) or a string passed through.
    pub spec_config: Option<Value>,
    pub extra_args: Option<String>,
    pub nccl_p2p_disable: bool,
    pub python_root: Option<String>,
}

impl LaunchSpec {
    /// The Python model defaults for a bare `VLLMLaunchSpec(model, served_name)`.
    pub fn minimal(model: &str, served_name: &str) -> Self {
        Self {
            model: model.into(),
            served_name: served_name.into(),
            tensor_parallel: 2,
            gpus: "0,1".into(),
            port: 8003,
            host: "0.0.0.0".into(),
            max_model_len: 32768,
            gpu_memory_utilization: 0.85,
            quantization: None,
            kv_cache_type: None,
            dtype: None,
            trust_remote_code: None,
            tool_parser: None,
            max_num_seqs: None,
            max_batched_tokens: None,
            prefix_cache: true,
            chunked_prefill: true,
            spec_config: None,
            extra_args: None,
            nccl_p2p_disable: false,
            python_root: None,
        }
    }
}

/// Host facts the env render needs (Python `launcher_env_lines`). The
/// caller resolves them; `python_root` is the fallback for presets without
/// their own (llmctl-next: the prefix of the workload's pinned interpreter
/// — the pin is the truth).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LauncherEnv {
    pub python_root: Option<String>,
    pub cuda_root: String,
    pub hf_home: String,
}

/// `CUDA_VISIBLE_DEVICES` for the first `tp` cards.
pub fn gpus_for(tp: i64) -> String {
    (0..tp.max(1)).map(|i| i.to_string()).collect::<Vec<_>>().join(",")
}

/// Python `model_to_launch_spec` (no TQ override: `run --dry-run`'s render).
pub fn launch_spec(p: &Preset, d: &VllmDefaults, port_override: Option<i64>) -> LaunchSpec {
    let kv_cache_type = p
        .kv_cache_dtype
        .as_deref()
        .filter(|k| !k.is_empty() && *k != "auto")
        .map(str::to_string);
    let mut extra: Vec<String> = Vec::new();
    if let Some(r) = p.reasoning_parser.as_deref().filter(|s| !s.is_empty()) {
        extra.push(format!("--reasoning-parser {r}"));
    }
    if let Some(e) = p.extra_args.as_deref().filter(|s| !s.is_empty()) {
        extra.push(e.to_string());
    }
    LaunchSpec {
        model: p.model_id.clone(),
        served_name: p.served_name.clone(),
        tensor_parallel: p.tensor_parallel_size,
        gpus: d
            .gpus
            .clone()
            .filter(|g| !g.is_empty())
            .unwrap_or_else(|| gpus_for(p.tensor_parallel_size)),
        port: port_override.unwrap_or(d.port),
        host: p.host.clone().filter(|h| !h.is_empty()).unwrap_or_else(|| d.host.clone()),
        max_model_len: p.max_model_len,
        gpu_memory_utilization: p.gpu_memory_utilization.unwrap_or(d.gpu_memory_utilization),
        quantization: Some(p.vllm_quantization_flag.clone()),
        kv_cache_type,
        tool_parser: p.tool_parser.clone(),
        dtype: p.dtype.clone(),
        trust_remote_code: p.trust_remote_code,
        python_root: p.python_root.clone(),
        max_num_seqs: p.max_num_seqs.or(d.max_num_seqs),
        max_batched_tokens: p.max_num_batched_tokens.or(d.max_batched_tokens),
        prefix_cache: d.prefix_cache,
        chunked_prefill: d.chunked_prefill,
        spec_config: p.spec_config.clone(),
        extra_args: (!extra.is_empty()).then(|| extra.join(" ")),
        nccl_p2p_disable: d.nccl_p2p_disable,
    }
}

/// Python's falsy test for an optional string (`if spec.quantization:`).
fn truthy(s: &Option<String>) -> Option<&str> {
    s.as_deref().filter(|v| !v.is_empty())
}

/// The env value escaping of `vllm_env._env_value`.
pub fn env_value(value: &str) -> String {
    let trimmed = value.trim_end_matches(['\r', '\n']);
    let folded = trimmed.replace("\r\n", " ").replace(['\r', '\n'], " ");
    let text = folded.trim_start_matches([' ', '\t']).replace('\\', "\\\\");
    if text.starts_with('\'') || text.starts_with('"') {
        format!("\\{text}")
    } else {
        text
    }
}

/// The `vllm-engine.env` body for `spec` (Python `render_vllm_env`).
/// `Err` when no Python root is known (Python raises `LauncherEnvError`).
pub fn render_vllm_env(spec: &LaunchSpec, env: &LauncherEnv) -> Result<String, String> {
    let python_root = spec
        .python_root
        .clone()
        .filter(|r| !r.is_empty())
        .or_else(|| env.python_root.clone().filter(|r| !r.is_empty()))
        .ok_or_else(|| "no Python root: the preset sets none and the workload has no pinned interpreter".to_string())?;
    let cuda = &env.cuda_root;
    let mut lines = vec![
        format!("CUDA_VISIBLE_DEVICES={}", spec.gpus),
        "CUDA_DEVICE_ORDER=PCI_BUS_ID".to_string(),
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True".to_string(),
        format!("LD_LIBRARY_PATH={python_root}/lib:{cuda}/lib64"),
        format!("PATH={python_root}/bin:{cuda}/bin:/usr/local/bin:/usr/bin:/bin"),
        format!("HF_HOME={}", env.hf_home),
    ];
    if spec.nccl_p2p_disable {
        lines.push("NCCL_P2P_DISABLE=1".into());
    }
    if let Some(root) = truthy(&spec.python_root) {
        lines.push(format!("VLLM_PYTHON={root}/bin/python"));
    }
    lines.push(format!("VLLM_MODEL={}", spec.model));
    lines.push(format!("VLLM_SERVED_NAME={}", spec.served_name));
    lines.push(format!("VLLM_TP={}", spec.tensor_parallel));
    lines.push(format!("VLLM_PORT={}", spec.port));
    lines.push(format!("VLLM_HOST={}", spec.host));
    lines.push(format!("VLLM_MAX_LEN={}", spec.max_model_len));
    lines.push(format!("VLLM_GPU_MEM={}", py_float(spec.gpu_memory_utilization)));
    if let Some(q) = truthy(&spec.quantization) {
        lines.push(format!("VLLM_QUANT={q}"));
    }
    if let Some(k) = truthy(&spec.kv_cache_type) {
        lines.push(format!("VLLM_KV_DTYPE={k}"));
    }
    if let Some(d) = truthy(&spec.dtype) {
        lines.push(format!("VLLM_DTYPE={d}"));
    }
    if let Some(t) = spec.trust_remote_code {
        lines.push(format!("VLLM_TRUST_REMOTE={}", if t { 1 } else { 0 }));
    }
    if let Some(t) = truthy(&spec.tool_parser) {
        lines.push(format!("VLLM_TOOL_PARSER={t}"));
    }
    if let Some(n) = spec.max_num_seqs {
        lines.push(format!("VLLM_MAX_SEQS={n}"));
    }
    if let Some(n) = spec.max_batched_tokens {
        lines.push(format!("VLLM_MAX_BATCHED_TOKENS={n}"));
    }
    if !spec.prefix_cache {
        lines.push("VLLM_PREFIX_CACHE=0".into());
    }
    if !spec.chunked_prefill {
        lines.push("VLLM_CHUNKED_PREFILL=0".into());
    }
    if let Some(sc) = &spec.spec_config {
        let text = match sc {
            Value::Mapping(m) if !m.is_empty() => Some(py_json(sc)),
            Value::String(s) if !s.is_empty() => Some(s.clone()),
            Value::Null | Value::Mapping(_) | Value::String(_) => None,
            other => Some(py_json(other)),
        };
        if let Some(text) = text {
            lines.push(format!("VLLM_SPEC_CONFIG={}", env_value(&text)));
        }
    }
    if let Some(e) = truthy(&spec.extra_args) {
        lines.push(format!("VLLM_EXTRA={}", env_value(e)));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    Ok(out)
}

/// Python's `repr(float)` (shortest round-trip digits; exponent form below
/// 1e-4 and from 1e16 up).
pub fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // `{:e}` gives the shortest round-trip digits: `d[.ddd]e<exp>`.
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let body = if exp >= 0 {
            let int_len = (exp + 1) as usize;
            if digits.len() <= int_len {
                format!("{}{}.0", digits, "0".repeat(int_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            let _ = n;
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        };
        format!("{sign}{body}")
    } else {
        let m = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let es = if exp < 0 { "-" } else { "+" };
        format!("{sign}{m}e{es}{:02}", exp.abs())
    }
}

/// Python `json.dumps(v, separators=(",", ":"))` (ensure_ascii, mapping
/// order kept).
pub fn py_json(v: &Value) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

fn json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7f => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_number(n: &serde_yaml::Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        let f = n.as_f64().unwrap_or(f64::NAN);
        if f.is_nan() {
            "NaN".into()
        } else if f.is_infinite() {
            if f > 0.0 { "Infinity".into() } else { "-Infinity".into() }
        } else {
            py_float(f)
        }
    }
}

fn write_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&json_number(n)),
        Value::String(s) => json_str(s, out),
        Value::Sequence(seq) => {
            out.push('[');
            for (i, item) in seq.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Mapping(m) => {
            out.push('{');
            for (i, (k, val)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                // json.dumps turns scalar keys into strings.
                let key = match k {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
                    Value::Null => "null".into(),
                    Value::Number(n) => json_number(n),
                    other => serde_yaml::to_string(other).unwrap_or_default().trim().to_string(),
                };
                json_str(&key, out);
                out.push(':');
                write_json(val, out);
            }
            out.push('}');
        }
        Value::Tagged(t) => write_json(&t.value, out),
    }
}

// ---- router profile envs (router_profiles.py) --------------------------------

/// First line of a profile file whose values come from a preset.
pub const GENERATED_PREFIX: &str = "# Generated from llmctl preset ";

/// The marker line on top of a preset's profile file. The text is the
/// Python one, so `llmctl router-profiles` sees no missing header.
pub fn generated_header(alias: &str) -> String {
    format!(
        "{GENERATED_PREFIX}{alias} by `llmctl router-profiles --write`; edit the preset, not this file."
    )
}

/// The preset a profile file's header names, if it has one.
pub fn header_preset(text: &str) -> Option<&str> {
    let first = py_splitlines(text).into_iter().next()?;
    let rest = first.strip_prefix(GENERATED_PREFIX)?;
    rest.split_whitespace().next()
}

/// Python `str.splitlines()` (no line ends kept).
pub fn py_splitlines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let is_break = matches!(
            c,
            '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}'
        );
        if !is_break {
            continue;
        }
        out.push(&text[start..i]);
        let mut next = i + c.len_utf8();
        if c == '\r' {
            if let Some(&(j, '\n')) = chars.peek() {
                chars.next();
                next = j + 1;
            }
        }
        start = next;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// `^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$` on one line.
fn assignment(line: &str) -> Option<(&str, &str)> {
    let mut rest = line.trim_start();
    if let Some(after) = rest.strip_prefix("export") {
        if after.starts_with(char::is_whitespace) {
            let trimmed = after.trim_start();
            // `export` followed by whitespace and a valid assignment.
            if let Some(found) = bare_assignment(trimmed) {
                return Some(found);
            }
        }
    }
    rest = rest.trim_start();
    bare_assignment(rest)
}

fn bare_assignment(s: &str) -> Option<(&str, &str)> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || !(bytes[0].is_ascii_alphabetic() || bytes[0] == b'_') {
        return None;
    }
    let mut i = 1;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'=' {
        Some((&s[..i], &s[i + 1..]))
    } else {
        None
    }
}

/// An assignment's value as systemd's `EnvironmentFile=` reads it.
fn env_file_value(raw: &str) -> String {
    let v = raw.trim();
    let b = v.as_bytes();
    if b.len() >= 2 && b[0] == b[b.len() - 1] && (b[0] == b'"' || b[0] == b'\'') {
        v[1..v.len() - 1].to_string()
    } else {
        v.to_string()
    }
}

/// `KEY=VALUE` lines in order; the first of a repeated key wins.
pub fn parse_env(text: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for line in py_splitlines(text) {
        if let Some((k, v)) = assignment(line) {
            if !out.iter().any(|(have, _)| have == k) {
                out.push((k.to_string(), env_file_value(v)));
            }
        }
    }
    out
}

/// One key whose value differs: (key, in the file, in the render).
pub type EnvChange = (String, Option<String>, Option<String>);

/// Keys whose value differs between `current` and `rendered` (rendered's
/// order first, then keys only the file has).
pub fn diff_env(current: &str, rendered: &str) -> Vec<EnvChange> {
    let have = parse_env(current);
    let want = parse_env(rendered);
    let get = |list: &[(String, String)], k: &str| list.iter().find(|(x, _)| x == k).map(|(_, v)| v.clone());
    let mut keys: Vec<String> = want.iter().map(|(k, _)| k.clone()).collect();
    keys.extend(have.iter().filter(|(k, _)| get(&want, k).is_none()).map(|(k, _)| k.clone()));
    keys.into_iter()
        .filter_map(|k| {
            let (a, b) = (get(&have, &k), get(&want, &k));
            (a != b).then_some((k, a, b))
        })
        .collect()
}

/// `current` with its assignments replaced by `rendered`'s: comments and
/// blank lines stay, unchanged lines stay byte for byte, keys the render
/// no longer has are dropped, new keys are appended.
pub fn merge_env(current: &str, rendered: &str) -> String {
    let want = parse_env(rendered);
    let get = |k: &str| want.iter().find(|(x, _)| x == k).map(|(_, v)| v.as_str());
    let mut out: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for line in py_splitlines(current) {
        let Some((key, raw)) = assignment(line) else {
            out.push(line.to_string());
            continue;
        };
        let Some(value) = get(key) else { continue };
        if seen.iter().any(|s| s == key) {
            continue;
        }
        seen.push(key.to_string());
        if env_file_value(raw) == value {
            out.push(line.to_string());
        } else {
            out.push(format!("{key}={value}"));
        }
    }
    for (k, v) in &want {
        if !seen.iter().any(|s| s == k) {
            out.push(format!("{k}={v}"));
        }
    }
    let mut text = out.join("\n");
    text.push('\n');
    text
}

/// `text` with the generated-from marker on line 1 (an older marker is
/// replaced, never stacked). Python `mark_generated`.
pub fn mark_generated(text: &str, alias: &str) -> String {
    let mut lines = vec![generated_header(alias)];
    lines.extend(
        py_splitlines(text)
            .into_iter()
            .filter(|l| !l.starts_with(GENERATED_PREFIX))
            .map(str::to_string),
    );
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

/// The profile file a sync writes: values merged, header on line 1.
pub fn synced_profile(current: &str, rendered: &str, alias: &str) -> String {
    mark_generated(&merge_env(current, rendered), alias)
}

// ---- version comments follow the pin (M5c item 1) -----------------------------

/// What the workload's pin says about its engine; the comment rewrite
/// takes its words from here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnginePin {
    /// The conda env / venv name: the directory above `bin/python`.
    pub env_name: String,
    /// The engine package's pinned version (`vllm`).
    pub version: Option<String>,
}

impl EnginePin {
    /// From a pinned interpreter path (`.../envs/vllm-muse/bin/python`)
    /// and the pinned package list.
    pub fn from_pin(interpreter: &str, packages: &[(String, String)], package: &str) -> Option<Self> {
        let p = std::path::Path::new(interpreter);
        let env = p.parent()?.parent()?.file_name()?.to_str()?.to_string();
        Some(Self {
            env_name: env,
            version: packages.iter().find(|(n, _)| n == package).map(|(_, v)| v.clone()),
        })
    }

    /// The Python root (prefix) of the pinned interpreter.
    pub fn python_root(interpreter: &str) -> Option<String> {
        let p = std::path::Path::new(interpreter);
        Some(p.parent()?.parent()?.display().to_string())
    }
}

/// One rewritten comment line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommentChange {
    /// 1-based line number.
    pub line: usize,
    pub before: String,
    pub after: String,
}

fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

/// Replace the token that ends at the first of `stops` after `prefix`.
fn replace_after(line: &str, prefix: &str, stops: &[char], new: &str) -> Option<String> {
    let at = line.find(prefix)? + prefix.len();
    let rest = &line[at..];
    let end = rest.find(|c: char| stops.contains(&c)).unwrap_or(rest.len());
    if end == 0 || &rest[..end] == new {
        return None;
    }
    Some(format!("{}{}{}", &line[..at], new, &rest[end..]))
}

/// Rewrite the comment lines that name the engine's version source so they
/// say what the pin says. Deterministic patterns only, comment lines only:
///
/// - `The interpreter is the <env> env (vLLM <version>` (the env header),
/// - `Runs from the <env> interpreter` (the router.yaml comment),
/// - `Keep this file in sync with the llmctl preset <path>` (when
///   `preset_path` is given: the preset file the profile really comes from).
///
/// Everything else — rollback notes naming an older env, history — stays.
pub fn settle_version_comments(
    text: &str,
    pin: &EnginePin,
    preset_path: Option<&str>,
) -> (String, Vec<CommentChange>) {
    let mut changes = Vec::new();
    let mut out = String::with_capacity(text.len());
    for (i, piece) in text.split_inclusive('\n').enumerate() {
        let (line, end) = match piece.strip_suffix('\n') {
            Some(l) => (l, "\n"),
            None => (piece, ""),
        };
        let mut new = line.to_string();
        if is_comment(line) {
            if let Some(r) = replace_after(&new, "The interpreter is the ", &[' '], &pin.env_name) {
                new = r;
            }
            if let Some(v) = &pin.version {
                if new.contains("The interpreter is the ") {
                    if let Some(r) = replace_after(&new, " env (vLLM ", &[',', ')', ' '], v) {
                        new = r;
                    }
                }
            }
            if new.contains(" interpreter") {
                if let Some(r) = replace_after(&new, "Runs from the ", &[' '], &pin.env_name) {
                    if r.contains(&format!("Runs from the {} interpreter", pin.env_name)) {
                        new = r;
                    }
                }
            }
            if let Some(path) = preset_path {
                if let Some(r) = replace_after(
                    &new,
                    "Keep this file in sync with the llmctl preset ",
                    &[' ', ')'],
                    path,
                ) {
                    new = r;
                }
            }
        }
        if new != line {
            changes.push(CommentChange {
                line: i + 1,
                before: line.to_string(),
                after: new.clone(),
            });
        }
        out.push_str(&new);
        out.push_str(end);
    }
    (out, changes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_float_matches_python_repr() {
        for (f, want) in [
            (0.9, "0.9"),
            (0.85, "0.85"),
            (1.0, "1.0"),
            (0.1 + 0.2, "0.30000000000000004"),
            (131072.0, "131072.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789012345.6, "123456789012345.6"),
            (2.5e-7, "2.5e-07"),
            (-0.5, "-0.5"),
        ] {
            assert_eq!(py_float(f), want, "{f}");
        }
    }

    #[test]
    fn py_json_keeps_order_and_python_spelling() {
        let v: Value = serde_yaml::from_str(
            "model: m\nnum_speculative_tokens: 7\nmethod: dflash\nratio: 0.5\nflag: true\nnone: null\nlist: [1, 'é']\n",
        )
        .unwrap();
        assert_eq!(
            py_json(&v),
            r#"{"model":"m","num_speculative_tokens":7,"method":"dflash","ratio":0.5,"flag":true,"none":null,"list":[1,"\u00e9"]}"#
        );
    }

    #[test]
    fn env_value_escapes_like_vllm_env() {
        assert_eq!(env_value(r#"--a "b" \d"#), r#"--a "b" \\d"#);
        assert_eq!(env_value(r#""--a" --b"#), r#"\"--a" --b"#);
        assert_eq!(env_value("  'x'"), r"\'x'");
        assert_eq!(env_value("a\nB=1\r\n"), "a B=1");
        assert_eq!(env_value("a\u{2028}b"), "a\u{2028}b");
    }

    #[test]
    fn splitlines_is_pythons() {
        assert_eq!(py_splitlines("a\nb\r\nc\rd\u{2028}e\n"), vec!["a", "b", "c", "d", "e"]);
        assert_eq!(py_splitlines(""), Vec::<&str>::new());
        assert_eq!(py_splitlines("a\n\nb"), vec!["a", "", "b"]);
    }

    #[test]
    fn merge_keeps_comments_and_unchanged_lines() {
        let current = "# head\nA=1\n# why B\nB=\"2\"\nOLD=x\nA=dup\n";
        let rendered = "A=1\nB=3\nC=4\n";
        assert_eq!(merge_env(current, rendered), "# head\nA=1\n# why B\nB=3\nC=4\n");
        assert_eq!(
            diff_env(current, rendered),
            vec![
                ("B".into(), Some("2".into()), Some("3".into())),
                ("C".into(), None, Some("4".into())),
                ("OLD".into(), Some("x".into()), None),
            ]
        );
        // Quoted equal value is no change; a trailing comment is.
        assert!(diff_env("A=\"1\"\n", "A=1\n").is_empty());
        assert_eq!(diff_env("A=1  # n\n", "A=1\n").len(), 1);
        assert_eq!(
            synced_profile("# Generated from llmctl preset old by x\n# keep\nA=0\n", "A=1\n", "new"),
            format!("{}\n# keep\nA=1\n", generated_header("new"))
        );
        assert_eq!(header_preset(&format!("{}\nA=1\n", generated_header("qwen27b"))), Some("qwen27b"));
        assert_eq!(header_preset("A=1\n"), None);
    }

    #[test]
    fn settle_rewrites_only_the_named_patterns() {
        let pin = EnginePin { env_name: "vllm-muse".into(), version: Some("0.30.0".into()) };
        let text = "# The interpreter is the vllm-laguna env (vLLM 0.25.1, torch 2.10) since x\n\
                    # Rollback: set python_root back to vllm-laguna (kept installed)\n\
                    # vLLM 0.25.1 returns it in `reasoning`\n\
                    # input. Runs from the vllm-laguna interpreter (see the profile), not\n\
                    # Keep this file in sync with the llmctl preset ~/.config/llm-models/qwen3.8-27b.yaml\n\
                    VLLM_X=Runs from the vllm-laguna interpreter\n";
        let (out, changes) =
            settle_version_comments(text, &pin, Some("~/.config/llmctl/presets/qwen27b.yaml"));
        assert_eq!(changes.len(), 3, "{changes:#?}");
        assert!(out.starts_with("# The interpreter is the vllm-muse env (vLLM 0.30.0, torch 2.10) since x\n"));
        assert!(out.contains("# Rollback: set python_root back to vllm-laguna (kept installed)\n"));
        assert!(out.contains("# vLLM 0.25.1 returns it"));
        assert!(out.contains("# input. Runs from the vllm-muse interpreter (see the profile), not\n"));
        assert!(out.contains("llmctl preset ~/.config/llmctl/presets/qwen27b.yaml\n"));
        assert!(out.contains("VLLM_X=Runs from the vllm-laguna interpreter\n"));
        // Settled text settles to itself.
        let (again, none) = settle_version_comments(&out, &pin, Some("~/.config/llmctl/presets/qwen27b.yaml"));
        assert_eq!(again, out);
        assert!(none.is_empty());
        assert_eq!(
            EnginePin::from_pin("/h/miniconda3/envs/vllm-muse/bin/python", &[("vllm".into(), "0.30.0".into())], "vllm"),
            Some(pin)
        );
        assert_eq!(EnginePin::python_root("/h/envs/x/bin/python").as_deref(), Some("/h/envs/x"));
    }

    #[test]
    fn schema_refuses_what_python_refuses() {
        let base = "alias: x\nserved_name: x\nmodel_id: org/m\nquantization: awq\nvllm_quantization_flag: awq\ntensor_parallel_size: 2\nmax_model_len: 4096\n";
        let ok: Value = serde_yaml::from_str(base).unwrap();
        let p = Preset::from_value(&ok).unwrap();
        assert_eq!(p.schema_version, 1);
        for (extra, why) in [
            ("bogus: 1\n", "unknown field"),
            ("gpu_memory_utilization: 1.5\n", "gpu_memory_utilization"),
            ("schema_version: 2\n", "schema_version"),
            ("port: 0\n", "port"),
        ] {
            let v: Value = serde_yaml::from_str(&format!("{base}{extra}")).unwrap();
            let err = Preset::from_value(&v).unwrap_err();
            assert!(err.contains(why), "{extra}: {err}");
        }
        let bad_alias: Value = serde_yaml::from_str(&base.replace("alias: x", "alias: X")).unwrap();
        assert!(Preset::from_value(&bad_alias).unwrap_err().contains("alias"));
        let bad_q: Value = serde_yaml::from_str(&base.replace("quantization: awq\n", "quantization: int4\n")).unwrap();
        assert!(Preset::from_value(&bad_q).unwrap_err().contains("quantization"));
        let bad_tp: Value = serde_yaml::from_str(&base.replace("size: 2", "size: 9")).unwrap();
        assert!(Preset::from_value(&bad_tp).unwrap_err().contains("tensor_parallel_size"));
    }
}
