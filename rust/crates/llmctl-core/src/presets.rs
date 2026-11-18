//! Line-based editing of preset YAML that keeps comments and untouched
//! keys — the Rust port of llmctl's `llmctl/presets/yaml_edit.py` rules
//! (Rust M5c, plan decision 3: presets stay in their files; edits never
//! reformat them).
//!
//! The contract is the Python one: edit only the top-level key spans that
//! changed, keep a replaced single-line value's trailing comment, delete
//! keys that are gone, append keys that are new, and refuse (`Err`) when
//! the line edit does not reload to exactly the desired mapping — the
//! caller then falls back to a full dump, losing comments, and says so.

use serde_yaml::Value;

/// A key line: `name:`, `name: value`, `name:` with a trailing comment.
/// Same shape as Python's `_KEY_RE` (letters, digits, `_`, `.`, `-`).
fn key_of(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    if bytes.is_empty() || !(bytes[0].is_ascii_alphabetic() || bytes[0] == b'_') {
        return None;
    }
    let mut i = 1;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.' | b'-') {
            i += 1;
            continue;
        }
        if c == b':' {
            let rest = &line[i + 1..];
            let rest = rest.strip_prefix('\r').unwrap_or(rest);
            if rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t') {
                return Some(&line[..i]);
            }
        }
        return None;
    }
    None
}

/// Top-level key -> `[start, end)` line range (the key line plus its
/// continuation lines: indented values, list items). Interior blank and
/// comment lines do not extend the span but do not end it either.
fn spans(lines: &[String]) -> std::collections::BTreeMap<String, (usize, usize)> {
    let mut starts: Vec<(String, usize)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(key) = key_of(line) {
            starts.push((key.to_string(), i));
        }
    }
    let mut out = std::collections::BTreeMap::new();
    for (n, (key, start)) in starts.iter().enumerate() {
        let limit = starts.get(n + 1).map(|(_, s)| *s).unwrap_or(lines.len());
        let mut end = start + 1;
        for line in lines.iter().take(limit).skip(start + 1) {
            let body = line.trim_end_matches(['\n', '\r']);
            let first = body.as_bytes().first().copied().unwrap_or(b' ');
            if (first == b' ' || first == b'\t')
                && !body.trim().is_empty()
                && !body.trim_start().starts_with('#')
            {
                end += 1;
            } else if body == "-" || body.starts_with("- ") {
                end += 1;
            } else if body.trim().is_empty() || body.trim_start().starts_with('#') {
                continue;
            } else {
                break;
            }
        }
        out.insert(key.clone(), (*start, end));
    }
    out
}

/// Dump one `key: value` pair the way a full dump would (block style, no
/// wrapping, insertion order) — `serde_yaml` on a single-entry mapping.
fn dump_pair(key: &str, value: &Value) -> Result<String, PresetError> {
    let mut mapping = serde_yaml::Mapping::new();
    mapping.insert(Value::String(key.to_string()), value.clone());
    serde_yaml::to_string(&mapping).map_err(|e| PresetError::Render(e.to_string()))
}

/// The trailing comment of a replaced single-line value, kept in place:
/// the first ` #` whose left side still loads to the old value.
fn trailing_comment(line: &str, old_value: &Value) -> String {
    let body = line.trim_end_matches(['\n', '\r']);
    let mut from = 0;
    while let Some(rel) = body[from..].find(" #") {
        let pos = from + rel;
        let prefix = &body[..pos];
        let Some((_, value_part)) = prefix.split_once(':') else {
            from = pos + 1;
            continue;
        };
        if let Ok(parsed) = serde_yaml::from_str::<Value>(value_part) {
            if parsed.eq(old_value) {
                let start = body[..pos].trim_end().len();
                return body[start..].to_string();
            }
        }
        from = pos + 1;
    }
    String::new()
}

/// Edit `text` in place so it loads to `desired`, keeping comments and
/// untouched keys. `Err` when the line edit cannot reproduce `desired`
/// (the caller falls back to a full dump).
pub fn edit_preset_text(text: &str, desired: &Value) -> Result<String, PresetError> {
    // Python's `yaml.safe_load(text) or {}`: empty (or blank) text is an
    // empty mapping, not a refusal.
    let old: Value = if text.trim().is_empty() {
        Value::Mapping(Default::default())
    } else {
        serde_yaml::from_str(text).unwrap_or(Value::Null)
    };
    let Value::Mapping(old_map) = &old else {
        return Err(PresetError::NotMapping);
    };
    let Value::Mapping(desired) = desired else {
        return Err(PresetError::NotMapping);
    };

    let mut lines: Vec<String> = if text.is_empty() {
        Vec::new()
    } else {
        text.split('\n')
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect()
    };
    if lines.last().is_some_and(|l| l.is_empty()) && text.ends_with('\n') {
        lines.pop();
    }
    let line_map = spans(&lines);

    let mut replacements: Vec<(usize, usize, Vec<String>)> = Vec::new();
    for (key, (start, end)) in &line_map {
        let old_value = old_map.get(&Value::String(key.clone()));
        let Some(new_value) = desired.get(&Value::String(key.clone())) else {
            replacements.push((*start, *end, Vec::new())); // deleted key
            continue;
        };
        if old_value == Some(new_value) {
            continue; // unchanged: byte-identical span stays
        }
        let dumped = dump_pair(key, new_value)?;
        let mut new_lines: Vec<String> = dumped
            .split('\n')
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        if new_lines.len() == 1 && end - start == 1 {
            let comment = old_value
                .map(|v| trailing_comment(&lines[*start], v))
                .unwrap_or_default();
            if !comment.is_empty() {
                // The slice starts at its own spaces (Python concatenates
                // without adding one).
                new_lines[0] = format!("{}{}", new_lines[0], comment);
            }
        }
        replacements.push((*start, *end, new_lines));
    }
    for (start, end, new) in replacements.into_iter().rev() {
        lines.splice(start..end, new);
    }
    let mut added: Vec<(String, &Value)> = Vec::new();
    for (k, v) in desired {
        if let Value::String(key) = k {
            if !line_map.contains_key(key) {
                added.push((key.clone(), v));
            }
        }
    }
    if !added.is_empty() {
        let mut mapping = serde_yaml::Mapping::new();
        for (k, v) in &added {
            mapping.insert(Value::String(k.clone()), (*v).clone());
        }
        let dumped =
            serde_yaml::to_string(&mapping).map_err(|e| PresetError::Render(e.to_string()))?;
        for line in dumped.split('\n').filter(|l| !l.is_empty()) {
            lines.push(line.to_string());
        }
    }
    if lines.is_empty() {
        if desired.is_empty() {
            return Ok(String::new()); // the Python contract: "" edits to ""
        }
        return Err(PresetError::NotReproduced("nothing to write".into()));
    }
    let mut result = lines.join("\n");
    result.push('\n');

    // Self-check: the edit must reload to exactly `desired` (key order,
    // like Python dict equality, does not matter).
    let reloaded: Value = serde_yaml::from_str(&result).unwrap_or(Value::Null);
    // Blank reloads to Null; Python folds that into `{}` before comparing.
    let reloaded = if reloaded.is_null() {
        Value::Mapping(Default::default())
    } else {
        reloaded
    };
    let desired_value = Value::Mapping(desired.clone());
    let keys_of = |v: &Value| -> std::collections::BTreeSet<String> {
        match v {
            Value::Mapping(m) => m
                .iter()
                .filter_map(|(k, _)| k.as_str().map(str::to_owned))
                .collect(),
            _ => Default::default(),
        }
    };
    if reloaded != desired_value || keys_of(&reloaded) != keys_of(&desired_value) {
        return Err(PresetError::NotReproduced(format!(
            "line edit reloads differently ({:?} vs {:?})",
            keys_of(&reloaded),
            keys_of(&desired_value)
        )));
    }
    Ok(result)
}

#[derive(Debug, thiserror::Error)]
pub enum PresetError {
    #[error("preset is not a mapping")]
    NotMapping,
    #[error("cannot render the YAML: {0}")]
    Render(String),
    #[error("line edit cannot reproduce the mapping: {0}")]
    NotReproduced(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::Value;

    fn mapping(pairs: &[(&str, Value)]) -> Value {
        let mut m = serde_yaml::Mapping::new();
        for (k, v) in pairs {
            m.insert(Value::String(k.to_string()), v.clone());
        }
        Value::Mapping(m)
    }

    const FIXTURE: &str = "# header comment stays\n#\nalias: qwen27b  # the router alias\nserved_name: qwen3.8-27b\ntensor_parallel_size: 2\narchitectures:\n- Qwen3_5ForConditionalGeneration\n\n# a comment between keys\nspec_config:\n  method: mtp\n  num_speculative_tokens: 3\nshortcuts: []\nschema_version: 1\n";

    #[test]
    fn unchanged_keys_and_comments_survive_byte_for_byte() {
        let desired = serde_yaml::from_str::<Value>(FIXTURE).unwrap();
        let edited = edit_preset_text(FIXTURE, &desired).unwrap();
        assert_eq!(edited, FIXTURE);
    }

    #[test]
    fn a_changed_value_keeps_its_trailing_comment() {
        let desired = mapping(&[
            ("alias", Value::String("qwen27b".into())),
            ("served_name", Value::String("qwen3.8-27b-v2".into())),
        ]);
        let edited = edit_preset_text(
            "alias: qwen27b  # the router alias\nserved_name: qwen3.8-27b\n",
            &desired,
        )
        .unwrap();
        assert!(
            edited.contains("alias: qwen27b  # the router alias"),
            "{edited}"
        );
        assert!(edited.contains("served_name: qwen3.8-27b-v2\n"), "{edited}");
    }

    #[test]
    fn edits_keep_header_comments_nested_mappings_and_lists() {
        let text = FIXTURE;
        let mut desired = serde_yaml::from_str::<Value>(text).unwrap();
        if let Value::Mapping(m) = &mut desired {
            m.insert(
                Value::String("max_model_len".into()),
                Value::Number(131072.into()),
            );
            m.remove(Value::String("shortcuts".into()));
        }
        let edited = edit_preset_text(text, &desired).unwrap();
        assert!(edited.starts_with("# header comment stays"), "{edited}");
        assert!(edited.contains("# a comment between keys"), "{edited}");
        assert!(
            edited.contains("- Qwen3_5ForConditionalGeneration"),
            "{edited}"
        );
        assert!(edited.contains("max_model_len: 131072"), "{edited}");
        assert!(!edited.contains("shortcuts"), "{edited}");
        // spec_config's nested lines are untouched.
        assert!(edited.contains("  num_speculative_tokens: 3"), "{edited}");
    }

    #[test]
    fn added_keys_append_and_deleted_keys_disappear() {
        let desired = mapping(&[
            ("alias", Value::String("new".into())),
            ("extra_args", Value::String("--shutdown-timeout 30".into())),
        ]);
        let edited =
            edit_preset_text("alias: old  # kept through the edit\nremoved_key: 1\n", &desired)
                .unwrap();
        // The port keeps a single-line value's trailing comment through a
        // value change (yaml_edit.py does the same; only the full-dump
        // fallback loses comments).
        assert!(edited.contains("alias: new  # kept through the edit\n"), "{edited}");
        assert!(!edited.contains("removed_key"), "{edited}");
        assert!(
            edited.ends_with("extra_args: --shutdown-timeout 30\n"),
            "{edited}"
        );
    }

    #[test]
    fn a_multiline_value_edits_as_a_block_scalar() {
        let desired = mapping(&[("weird", Value::String("a: b\n c".into()))]);
        let edited = edit_preset_text("weird: old\n", &desired).unwrap();
        assert!(edited.starts_with("weird: |-"), "{edited}");
    }

    #[test]
    fn a_non_mapping_source_refuses_instead_of_corrupting() {
        let desired = mapping(&[("alias", Value::String("x".into()))]);
        assert!(matches!(
            edit_preset_text("- just\n- a list\n", &desired),
            Err(PresetError::NotMapping)
        ));
        // Empty to empty is the Python contract's success ("").
        assert_eq!(edit_preset_text("", &mapping(&[])).unwrap(), "");
    }

    #[test]
    fn the_real_qwen27b_preset_round_trips() {
        // The live preset's shape (header comments, inline comments,
        // nested mapping, list, empty list) with synthetic values.
        let text = "# Qwen3.8-27B (vendor block-FP8)\n#\n# QUANT CHOICE etc.\nalias: qwen27b\nserved_name: qwen3.8-27b\nmodel_id: /home/yannik/models/hf2local/Qwen3.8-27B-FP8  # hf2local cuda from Qwen/Qwen3.8-27B@1d4bf0f2\nquantization: fp8\ntensor_parallel_size: 2\nmax_model_len: 262144\narchitectures:\n- Qwen3_5ForConditionalGeneration\nmax_num_seqs: 32\ngpu_memory_utilization: 0.80\nkv_cache_dtype: fp8\ndtype: null\ntrust_remote_code: false  # the repo ships no custom modeling code\nmax_num_batched_tokens: 8192\nhost: 127.0.0.1\nspec_config:\n  method: mtp\n  num_speculative_tokens: 3\ntq: false\nshortcuts: []\nschema_version: 1\n";
        let desired = serde_yaml::from_str::<Value>(text).unwrap();
        assert_eq!(edit_preset_text(text, &desired).unwrap(), text);

        // One edit: max_model_len 262144 -> 131072; nothing else moves.
        let mut edited_desired = desired.clone();
        if let Value::Mapping(m) = &mut edited_desired {
            m.insert(
                Value::String("max_model_len".into()),
                Value::Number(131072.into()),
            );
        }
        let edited = edit_preset_text(text, &edited_desired).unwrap();
        assert!(edited.contains("max_model_len: 131072\n"), "{edited}");
        assert_eq!(edited.matches('#').count(), text.matches('#').count());
    }

    #[test]
    fn a_missing_final_newline_is_handled() {
        let desired = mapping(&[("alias", Value::String("x".into()))]);
        let edited = edit_preset_text("alias: old", &desired).unwrap();
        assert_eq!(edited, "alias: x\n");
    }
}
