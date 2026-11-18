//! M5c item 2: for every preset fixture, llmctl-next renders the launch the
//! Python llmctl renders (unit env, args, served name) and edits the file
//! to the same bytes. The expected side is `tests/fixtures/rust_parity/
//! parity.json`, written by that directory's `generate.py` from the same
//! sanitised inputs; `tests/test_rust_preset_parity.py` keeps the Python
//! side honest against the same file.

use std::path::PathBuf;

use serde_yaml::Value;

use llmctl_core::preset_launch::{launch_spec, render_vllm_env, LaunchSpec, LauncherEnv, Preset, VllmDefaults};
use llmctl_core::presets::edit_preset_text;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/rust_parity")
}

fn parity() -> Value {
    // serde_yaml reads JSON and keeps mapping order (serde_json would sort).
    let text = std::fs::read_to_string(fixture_dir().join("parity.json")).expect("parity.json");
    serde_yaml::from_str(&text).expect("parity.json parses")
}

fn s(v: &Value, k: &str) -> String {
    v.get(k).and_then(|x| x.as_str()).unwrap_or_else(|| panic!("{k} in {v:?}")).to_string()
}

fn launcher(v: &Value) -> LauncherEnv {
    LauncherEnv {
        python_root: Some(s(v, "python_root")),
        cuda_root: s(v, "cuda_root"),
        hf_home: s(v, "hf_home"),
    }
}

#[test]
fn every_preset_renders_the_python_env() {
    let p = parity();
    let env = launcher(&p["launcher"]);
    let custom: VllmDefaults = serde_yaml::from_value(p["custom_defaults"].clone()).unwrap();
    let renders = p["renders"].as_sequence().unwrap();
    assert!(renders.len() >= 18, "every preset twice");
    let mut failures = Vec::new();
    for case in renders {
        let file = s(case, "file");
        let text = std::fs::read_to_string(fixture_dir().join("presets").join(&file)).unwrap();
        let value: Value = serde_yaml::from_str(&text).unwrap();
        let preset = Preset::from_value(&value).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(preset.alias, s(case, "alias"));
        let defaults = match s(case, "defaults").as_str() {
            "default" => VllmDefaults::default(),
            _ => custom.clone(),
        };
        let port = case["port"].as_i64().unwrap();
        let spec = launch_spec(&preset, &defaults, Some(port));
        let got = render_vllm_env(&spec, &env).unwrap();
        let want = s(case, "env");
        if got != want {
            failures.push(format!("{file} ({}):\n--- rust\n{got}--- python\n{want}", s(case, "defaults")));
        }
        // The served name the run verifies is the preset's.
        assert!(got.contains(&format!("VLLM_SERVED_NAME={}\n", preset.served_name)));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn every_preset_file_round_trips_and_edits_to_the_python_bytes() {
    let p = parity();
    let edits = p["edits"].as_sequence().unwrap();
    let mut failures = Vec::new();
    let mut identities = 0;
    for case in edits {
        let name = s(case, "name");
        let text = s(case, "text");
        let desired = case["desired"].clone();
        let got = edit_preset_text(&text, &desired).ok();
        let want = case["expected"].as_str().map(str::to_string);
        if name.ends_with(":identity") {
            identities += 1;
            assert_eq!(got.as_deref(), Some(text.as_str()), "{name}: a no-op edit must keep every byte");
        }
        if got != want {
            failures.push(format!("{name}:\n--- rust\n{got:?}\n--- python\n{want:?}"));
        }
    }
    assert!(identities >= 9, "one round trip per preset file");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frozen_gpu_models_specs_render_the_same() {
    let p = parity();
    let env = launcher(&p["spec_launcher"]);
    for case in p["specs"].as_sequence().unwrap() {
        let kw = &case["spec"];
        let mut spec = LaunchSpec::minimal(&s(kw, "model"), &s(kw, "served_name"));
        if let Value::Mapping(m) = kw {
            for (k, v) in m {
                match k.as_str().unwrap() {
                    "model" | "served_name" => {}
                    "tensor_parallel" => spec.tensor_parallel = v.as_i64().unwrap(),
                    "gpus" => spec.gpus = v.as_str().unwrap().into(),
                    "port" => spec.port = v.as_i64().unwrap(),
                    "host" => spec.host = v.as_str().unwrap().into(),
                    "max_model_len" => spec.max_model_len = v.as_i64().unwrap(),
                    "gpu_memory_utilization" => spec.gpu_memory_utilization = v.as_f64().unwrap(),
                    "quantization" => spec.quantization = v.as_str().map(Into::into),
                    "kv_cache_type" => spec.kv_cache_type = v.as_str().map(Into::into),
                    "tool_parser" => spec.tool_parser = v.as_str().map(Into::into),
                    "max_num_seqs" => spec.max_num_seqs = v.as_i64(),
                    "max_batched_tokens" => spec.max_batched_tokens = v.as_i64(),
                    "spec_config" => spec.spec_config = Some(v.clone()),
                    "extra_args" => spec.extra_args = v.as_str().map(Into::into),
                    "nccl_p2p_disable" => spec.nccl_p2p_disable = v.as_bool().unwrap(),
                    "prefix_cache" => spec.prefix_cache = v.as_bool().unwrap(),
                    "chunked_prefill" => spec.chunked_prefill = v.as_bool().unwrap(),
                    other => panic!("spec field {other} not mapped"),
                }
            }
        }
        // The capture env is the caller's (no python_root in these specs).
        let got = render_vllm_env(&spec, &env).unwrap();
        assert_eq!(got, s(case, "env"), "{}", s(case, "name"));
    }
}
