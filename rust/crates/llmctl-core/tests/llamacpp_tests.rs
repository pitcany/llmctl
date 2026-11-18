//! llama.cpp router records (M5d): preset INI round trips on fixtures,
//! validation, the router's command line, `GET /models` parsing, the load
//! decision (fit, budget, `--models-max`), the pre-load guard and the Mac
//! lane render.

use std::collections::BTreeMap;

use llmctl_core::llamacpp::{
    self, check_router_argv, decide_load, import, parse_models, preload_problems, render, roundtrip,
    validate, LlamaLane, LoadDecision, MemoryCheck, ModelHookWhen, ModelHooks, ModelState,
    RouterLaunch, RouterModel, RouterModelStatus,
};
use llmctl_core::{EngineKind, GpuClaim};

fn fixture(name: &str) -> String {
    let path = format!("{}/tests/fixtures/llamacpp/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn canonical_fixtures_render_byte_for_byte() {
    for name in ["workstation-pilot.ini", "readme-example.ini", "gpt-oss-120b.ini"] {
        let text = fixture(name);
        let (records, rendered) = roundtrip(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(rendered, text, "{name} did not render byte for byte");
        assert_eq!(import(&rendered).unwrap(), records, "{name}: re-import differs");
    }
}

#[test]
fn every_note_is_kept_with_the_key_it_precedes() {
    let p = import(&fixture("workstation-pilot.ini")).unwrap();
    assert_eq!(p.notes_count(), 7);
    assert_eq!(p.top[0].key, "version");
    assert_eq!(p.top[0].before.len(), 2);
    let small = p.section("qwen3.5-0.8b").unwrap();
    assert_eq!(small.before, vec!["", "; Pilot model 1: tiny, loads in seconds."]);
    let ctx = small.entries.iter().find(|e| e.key == "ctx-size").unwrap();
    assert_eq!(ctx.value, "32768");
    assert_eq!(ctx.inline.as_deref(), Some("; small on purpose"));
    assert_eq!(p.trailing, vec!["", "; end of file"]);
}

#[test]
fn a_non_canonical_file_reads_the_same_and_renders_canonically() {
    let text = fixture("noncanonical.ini");
    let (records, rendered) = roundtrip(&text).unwrap();
    assert_ne!(rendered, text);
    assert_eq!(
        rendered,
        "; indented note\nversion = 1\n[m1] # header note\nmodel = /x/m1.gguf\nc = 2048 ;tail\n\n# end\n"
    );
    assert_eq!(records.effective(), import(&rendered).unwrap().effective());
    assert_eq!(records.model("m1").unwrap().arg(&["c"]), Some("2048"));
}

#[test]
fn records_type_the_model_sections() {
    let p = import(&fixture("workstation-pilot.ini")).unwrap();
    assert_eq!(p.model_ids(), vec!["qwen3.5-0.8b", "qwen3.6-35b-a3b"]);
    let big = p.model("qwen3.6-35b-a3b").unwrap();
    assert_eq!(
        big.model.as_deref(),
        Some("/mnt/storage/converted/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf")
    );
    assert!(big.mmproj.is_some());
    assert!(!big.load_on_startup);
    assert_eq!(big.args, vec![("ctx-size".into(), "65536".into()), ("tensor-split".into(), "50,50".into())]);
    assert!(p.model("*").is_none(), "the globals are not a model");

    let recs = llamacpp::model_records("llamacpp-router", &p);
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].engine_kind, EngineKind::LlamaCppRouter);
    assert_eq!(recs[0].workload, "llamacpp-router");
}

#[test]
fn edits_keep_notes_and_round_trip() {
    let mut p = import(&fixture("workstation-pilot.ini")).unwrap();
    p.set("qwen3.5-0.8b", "ctx-size", "16384").unwrap();
    p.set("qwen3.5-0.8b", "cache-reuse", "256").unwrap();
    assert!(p.remove_key("qwen3.5-0.8b", "parallel").unwrap());
    p.add_model(&RouterModel {
        id: "gpt-oss-120b".into(),
        model: Some("/home/yannik/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf".into()),
        hf_repo: None,
        mmproj: None,
        load_on_startup: false,
        args: vec![("n-cpu-moe".into(), "6".into())],
    })
    .unwrap();
    assert!(p.add_model(&p.model("gpt-oss-120b").unwrap()).is_err(), "a duplicate id is refused");
    let text = llamacpp::check_render(&p).unwrap();
    assert!(text.contains("ctx-size = 16384 ; small on purpose"), "{text}");
    assert!(text.contains("cache-reuse = 256\n"), "{text}");
    assert!(!text.contains("parallel = 2"), "{text}");
    // A new model goes after the last section, before the closing notes.
    assert!(
        text.ends_with("load-on-startup = false\n\n[gpt-oss-120b]\nmodel = /home/yannik/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf\nn-cpu-moe = 6\n\n; end of file\n"),
        "{text}"
    );
    assert!(validate(&p).is_empty(), "{:?}", validate(&p));
    let changes = llamacpp::changes(&import(&fixture("workstation-pilot.ini")).unwrap(), &p);
    assert!(changes.contains(&"[qwen3.5-0.8b] ctx-size: 32768 → 16384".to_string()), "{changes:?}");
    assert!(changes.contains(&"[gpt-oss-120b] added".to_string()), "{changes:?}");

    p.remove_model("qwen3.6-35b-a3b").unwrap();
    assert!(p.remove_model("qwen3.6-35b-a3b").is_err());
    assert!(p.remove_model("*").is_err(), "the globals are not a model");
    llamacpp::check_render(&p).unwrap();
}

#[test]
fn validation_names_what_llama_cpp_or_this_deployment_refuses() {
    assert!(validate(&import(&fixture("workstation-pilot.ini")).unwrap()).is_empty());
    assert!(validate(&import(&fixture("gpt-oss-120b.ini")).unwrap()).is_empty());

    // The README example: a key set twice, a section without a model path.
    let v = validate(&import(&fixture("readme-example.ini")).unwrap());
    assert!(v.iter().any(|m| m.contains("model-draft is set twice")), "{v:?}");
    assert!(v.iter().any(|m| m.contains("[ggml-org/MY-MODEL-GGUF:Q8_0]: names no `model`")), "{v:?}");

    let bad = import(
        "[*]\nmodel = /x.gguf\nmodels-max = 2\n[a]\nmodel = rel.gguf\nport = 9000\nload-on-startup = maybe\n[b]\nmodel = /x/b.bin\n",
    )
    .unwrap();
    let v = validate(&bad);
    for want in [
        "[*] model: a model belongs in its own section",
        "[*] models-max: a router option",
        "[a] model: \"rel.gguf\" is not absolute",
        "[a] port: the router overrides this key",
        "[a] load-on-startup: \"maybe\"",
        "[b] model: \"/x/b.bin\" is not a .gguf file",
    ] {
        assert!(v.iter().any(|m| m.starts_with(want)), "missing {want:?} in {v:?}");
    }

    // A value llama.cpp would cut at `;` never reaches the file.
    let mut p = import("[a]\nmodel = /a.gguf\n").unwrap();
    p.set("a", "chat-template", "x;y").unwrap();
    assert!(validate(&p).iter().any(|m| m.contains("cannot hold `;`")), "{:?}", validate(&p));
}

#[test]
fn parse_errors_name_the_line() {
    let e = import("[a]\nmodel = /a.gguf\nnot a key line\n").unwrap_err();
    assert!(e.to_string().starts_with("line 3:"), "{e}");
    assert!(import("[a\n").is_err());
    assert!(import("[]\n").is_err());
    assert!(import("9key = 1\n").is_err());
}

#[test]
fn the_router_command_line_loads_nothing_by_itself() {
    let launch = RouterLaunch {
        binary: "/home/yannik/AI/llama.cpp/build/bin/llama-server".into(),
        host: "127.0.0.1".into(),
        port: 8010,
        preset: "/home/yannik/.config/llmctl-next/llamacpp-router.ini".into(),
        models_max: 2,
        extra: vec!["--flash-attn".into(), "on".into()],
    };
    let argv = launch.argv();
    assert!(argv.contains(&"--no-models-autoload".to_string()));
    assert!(!argv.iter().any(|a| a == "-m" || a == "--model"));
    assert!(check_router_argv(&argv, &[]).is_empty(), "{:?}", check_router_argv(&argv, &[]));
    assert_eq!(llamacpp::preset_path_from_argv(&argv).as_deref(), Some(launch.preset.as_str()));
    assert_eq!(llamacpp::models_max_from_argv(&argv), Some(2));

    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let gptoss = s(&["/bin/llama-server", "--model", "/m.gguf", "--port", "8005"]);
    let v = check_router_argv(&gptoss, &[]);
    assert!(v.iter().any(|m| m.contains("names a model of its own")), "{v:?}");
    assert!(v.iter().any(|m| m.contains("no --models-preset")), "{v:?}");
    assert!(v.iter().any(|m| m.contains("autoload is on")), "{v:?}");

    let unlimited = s(&["/bin/llama-server", "--models-preset=/p.ini", "--no-models-autoload", "--models-max", "0"]);
    assert_eq!(check_router_argv(&unlimited, &[]), vec!["--models-max 0 (unlimited): the planner needs a bound"]);
    // Autoload turned back on through the environment counts.
    let argv_no_flag = s(&["/bin/llama-server", "--models-preset", "/p.ini"]);
    let env = vec![("LLAMA_ARG_MODELS_AUTOLOAD".to_string(), "0".to_string())];
    assert!(check_router_argv(&argv_no_flag, &env).is_empty());
    assert_eq!(llamacpp::models_max_from_argv(&argv_no_flag), Some(llamacpp::DEFAULT_MODELS_MAX));
    let env = vec![("LLAMA_ARG_MODEL".to_string(), "/m.gguf".to_string())];
    assert!(!check_router_argv(&launch.argv(), &env).is_empty());
}

fn models_body() -> serde_json::Value {
    serde_json::json!({"data": [
        {"id": "small", "path": "/m/small.gguf", "status": {"value": "loaded", "args": ["llama-server"]}},
        {"id": "big", "status": {"value": "unloaded", "args": ["llama-server"], "failed": true, "exit_code": 1}},
        {"id": "mid", "status": {"value": "unloaded"}},
        {"id": "odd", "status": {"value": "hibernating"}},
    ]})
}

#[test]
fn the_model_list_parses_with_failures_and_unknown_states() {
    let rows = parse_models(&models_body()).unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].state, ModelState::Loaded);
    assert_eq!(rows[0].path.as_deref(), Some("/m/small.gguf"));
    assert!(rows[1].failed && rows[1].exit_code == Some(1));
    assert_eq!(rows[3].state, ModelState::Other("hibernating".into()));
    assert!(rows[3].state.holds_memory(), "an unknown state is never free");
    assert!(!ModelState::Unloaded.holds_memory());
    assert!(parse_models(&serde_json::json!({"object": "list"})).is_none());
}

fn mib(pairs: &[(u32, u64)]) -> BTreeMap<u32, u64> {
    pairs.iter().copied().collect()
}

fn cards(need: Option<&[(u32, u64)]>, free: Option<&[(u32, u64)]>) -> MemoryCheck {
    MemoryCheck::Cards {
        need: need.map(mib),
        free: free.map(mib),
        host_ram_gib: None,
        free_ram_gib: None,
    }
}

#[test]
fn a_load_is_decided_by_the_fit_and_refused_when_it_says_no() {
    let rows = parse_models(&models_body()).unwrap();
    let fit = cards(Some(&[(0, 20000), (1, 20000)]), Some(&[(0, 30000), (1, 30000)]));
    let d = decide_load("mid", Some(&rows), 4, &fit);
    assert!(matches!(d, LoadDecision::Load(_)), "{d:?}");
    assert!(d.reason().starts_with("fits: needs 19.5 GiB on gpu 0"), "{d:?}");

    let tight = cards(Some(&[(0, 20000), (1, 20000)]), Some(&[(0, 30000), (1, 20500)]));
    let d = decide_load("mid", Some(&rows), 4, &tight);
    assert!(matches!(&d, LoadDecision::Refuse(r) if r.starts_with("does not fit")), "{d:?}");

    // Fail closed: never measured, cards unreadable, router silent, unknown id.
    for (model, statuses, mem) in [
        ("mid", Some(rows.as_slice()), cards(None, Some(&[(0, 30000)]))),
        ("mid", Some(rows.as_slice()), cards(Some(&[(0, 1000)]), None)),
        ("mid", None, cards(Some(&[(0, 1000)]), Some(&[(0, 30000)]))),
        ("nope", Some(rows.as_slice()), cards(Some(&[(0, 1000)]), Some(&[(0, 30000)]))),
    ] {
        let d = decide_load(model, statuses, 4, &mem);
        assert!(matches!(d, LoadDecision::Refuse(_)), "{model}: {d:?}");
    }
    assert!(decide_load("mid", Some(&rows), 4, &cards(None, None)).reason().contains("never measured"));

    // Already loaded: nothing to do.
    assert!(matches!(decide_load("small", Some(&rows), 4, &fit), LoadDecision::AlreadyLoaded(_)));

    // At --models-max: refused, so the router never evicts on its own
    // ("small" and the unknown-state "odd" both hold memory).
    let d = decide_load("mid", Some(&rows), 2, &fit);
    assert!(
        matches!(&d, LoadDecision::Refuse(r) if r.contains("--models-max 2 (small, odd)")),
        "{d:?}"
    );
}

#[test]
fn a_cpu_offload_load_also_needs_host_ram() {
    let rows = parse_models(&models_body()).unwrap();
    let mem = |free_ram: Option<f64>| MemoryCheck::Cards {
        need: Some(mib(&[(0, 20000)])),
        free: Some(mib(&[(0, 30000)])),
        host_ram_gib: Some(70.0),
        free_ram_gib: free_ram,
    };
    assert!(matches!(decide_load("mid", Some(&rows), 4, &mem(Some(80.0))), LoadDecision::Load(_)));
    let d = decide_load("mid", Some(&rows), 4, &mem(Some(72.0)));
    assert!(d.reason().contains("needs 70.0 GiB host RAM"), "{d:?}");
    assert!(matches!(decide_load("mid", Some(&rows), 4, &mem(None)), LoadDecision::Refuse(_)));
}

#[test]
fn on_a_mac_the_memory_budget_decides() {
    let rows = vec![
        RouterModelStatus { id: "emb".into(), path: None, state: ModelState::Loaded, failed: false, exit_code: None },
        RouterModelStatus { id: "chat".into(), path: None, state: ModelState::Unloaded, failed: false, exit_code: None },
    ];
    let budget = |b: Option<u64>, need: Option<f64>, peak: Option<f64>| MemoryCheck::Budget {
        budget_gib: b,
        need_gib: need,
        loaded: vec![("emb".into(), peak)],
    };
    let d = decide_load("chat", Some(&rows), 4, &budget(Some(48), Some(30.0), Some(2.5)));
    assert_eq!(d, LoadDecision::Load("fits the memory budget: 33G of 48G at peak".into()));
    let d = decide_load("chat", Some(&rows), 4, &budget(Some(32), Some(30.0), Some(2.5)));
    assert!(matches!(&d, LoadDecision::Refuse(r) if r.starts_with("over the memory budget: 33G")), "{d:?}");
    for mem in [budget(None, Some(1.0), Some(1.0)), budget(Some(48), None, Some(1.0)), budget(Some(48), Some(1.0), None)] {
        assert!(matches!(decide_load("chat", Some(&rows), 4, &mem), LoadDecision::Refuse(_)), "{mem:?}");
    }
}

#[test]
fn the_preload_guard_ports_llama_server_guarded() {
    let split = "/m/K/K-UD-IQ1_S-00001-of-00003.gguf";
    let p = preload_problems(split, &["K-UD-IQ1_S-00003-of-00003.gguf".into()], &[], None, None);
    assert_eq!(p, vec!["incomplete split model (3 shards expected); missing: K-UD-IQ1_S-00003-of-00003.gguf"]);
    let p = preload_problems("/m/one.gguf", &["/m/one.gguf".into()], &["x.incomplete".into()], None, None);
    assert_eq!(p.len(), 2, "{p:?}");
    assert!(p[1].contains("still downloading"));
    let p = preload_problems("/m/one.gguf", &[], &[], Some(29500), Some(&mib(&[(0, 31000), (1, 12000)])));
    assert_eq!(p, vec!["gpu 1 has only 12000 MiB free (floor 29500 MiB): something else holds VRAM"]);
    assert!(preload_problems("/m/one.gguf", &[], &[], Some(29500), None)[0].contains("could not be read"));
    assert!(preload_problems("/m/one.gguf", &[], &[], None, None).is_empty());
}

#[test]
fn model_hooks_are_validated_like_workload_hooks() {
    let hooks: ModelHooks = serde_json::from_value(serde_json::json!({"hooks": [
        {"model": "gpt-oss-120b", "name": "ds4-adopt", "when": "post_load",
         "argv": ["/usr/bin/env", "DSV4_PORT=8010", "/home/yannik/bin/ds4-llmctl-sync", "adopt"],
         "timeout_secs": 1260, "optional": true},
        {"model": "gpt-oss-120b", "name": "ds4-detach", "when": "post_unload",
         "argv": ["/usr/bin/env", "/home/yannik/bin/ds4-llmctl-sync", "detach"], "optional": true},
    ]}))
    .unwrap();
    assert!(hooks.validate().is_empty(), "{:?}", hooks.validate());
    assert_eq!(hooks.at("gpt-oss-120b", ModelHookWhen::PostLoad).count(), 1);
    assert_eq!(hooks.at("other", ModelHookWhen::PostLoad).count(), 0);
    assert_eq!(hooks.hooks[1].timeout_secs, 60);

    let bad: ModelHooks = serde_json::from_value(serde_json::json!({"hooks": [
        {"model": "m", "name": "x", "when": "pre_load", "argv": ["ds4-llmctl-sync"]},
        {"model": "m", "name": "x", "when": "pre_load", "argv": [], "timeout_secs": 0},
    ]}))
    .unwrap();
    let v = bad.validate();
    assert!(v.iter().any(|m| m.contains("not an absolute path")), "{v:?}");
    assert!(v.iter().any(|m| m.contains("used twice")), "{v:?}");
    assert!(v.iter().any(|m| m.contains("no command")), "{v:?}");
    assert!(v.iter().any(|m| m.contains("timeout 0s")), "{v:?}");
}

#[test]
fn a_mac_lane_renders_llama_server_with_one_model() {
    let lane = LlamaLane {
        binary_macro: "llamacpp".into(),
        model: "/Users/y/Models/gguf/embeddings/nomic-embed-text-v1.5.Q8_0.gguf".into(),
        mmproj: None,
        args: vec![("embedding".into(), "true".into()), ("ctx-size".into(), "8192".into())],
        ttl: Some(600),
        name: Some("Nomic embed (llama.cpp)".into()),
    };
    assert!(lane.validate().is_empty(), "{:?}", lane.validate());
    assert_eq!(
        lane.cmd(),
        "${llamacpp} -m /Users/y/Models/gguf/embeddings/nomic-embed-text-v1.5.Q8_0.gguf --host 127.0.0.1 --port ${PORT} --embedding --ctx-size 8192"
    );
    let fields = lane.fields();
    assert_eq!(fields.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), vec!["name", "cmd", "proxy", "ttl"]);

    // What the lane records infer from it: a llama.cpp lane.
    let cfg = llmctl_core::swap::import(&format!(
        "macros:\n  llamacpp: /opt/llama-b12527/llama-server\nmodels:\n  emb:\n    cmd: {}\n    proxy: http://127.0.0.1:${{PORT}}\n",
        lane.cmd()
    ))
    .unwrap();
    assert_eq!(cfg.lanes[0].engine_kind, EngineKind::LlamaCpp);

    let bad = LlamaLane {
        binary_macro: "llama cpp".into(),
        model: "rel.gguf".into(),
        mmproj: Some("/x y".into()),
        args: vec![("port".into(), "1".into()), ("c".into(), "a b".into())],
        ttl: None,
        name: None,
    };
    assert_eq!(bad.validate().len(), 5, "{:?}", bad.validate());
}

#[test]
fn the_router_is_an_on_demand_engine_with_model_unload() {
    assert_eq!(EngineKind::infer("llamacpp-router"), EngineKind::LlamaCppRouter);
    assert_eq!(EngineKind::LlamaCppRouter.to_string(), "llama_cpp_router");
    assert!(EngineKind::LlamaCppRouter.has_engine_unload());
    assert_eq!(GpuClaim::default_for(&EngineKind::LlamaCppRouter), GpuClaim::OnDemand);
    let k: EngineKind = serde_json::from_str("\"llama_cpp_router\"").unwrap();
    assert_eq!(k, EngineKind::LlamaCppRouter);
    // The single-model llama.cpp unit stays what it was.
    assert!(!EngineKind::LlamaCpp.has_engine_unload());
    let _ = render(&llamacpp::RouterPreset::default());
}

/// Item 3, the other direction: a `vllm-engine` start on the router's cards
/// plans the router's loaded models out first (the M4c on-demand swap),
/// and never stops the router itself.
#[test]
fn a_vllm_start_unloads_router_models_on_its_cards_first() {
    use llmctl_core::{plan_run, Observation, ProcessState, Readiness, ServedModel, WorkloadName};
    let wl = |name: &str, kind: &str, claim: Option<&str>, model: &str, port: u16| {
        let mut w = serde_json::json!({
            "name": name, "engine": "fake", "engine_kind": kind, "model": model,
            "gpus": [{"host": "workstation", "index": 0}, {"host": "workstation", "index": 1}],
            "port": port,
            "unit": {"scope": "user", "name": format!("{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("{name}.service")},
            "ownership": {"kind": "external"}
        });
        if let Some(c) = claim {
            w["gpu_claim"] = serde_json::json!(c);
        }
        w
    };
    let inv: llmctl_core::HostInventory = serde_json::from_value(serde_json::json!({
        "host": "workstation", "platform": "linux",
        "gpus": [{"host": "workstation", "index": 0}, {"host": "workstation", "index": 1}],
        "workloads": [
            wl("vllm-engine", "vllm", None, "fake/vllm", 8003),
            wl("llamacpp-router", "llama_cpp_router", Some("on_demand"), "llamacpp-router", 8010),
        ],
        "revision": 1
    }))
    .unwrap();
    inv.validate().unwrap();
    let obs = |name: &str, running: bool, loaded: Option<&[&str]>| Observation {
        workload: WorkloadName::new(name).unwrap(),
        unit: None,
        process: if running { ProcessState::Running } else { ProcessState::Stopped },
        readiness: if running { Readiness::Ready } else { Readiness::NotReady },
        served_model: ServedModel::Unknown,
        pid: None,
        gpus: Vec::new(),
        loaded_models: loaded.map(|m| m.iter().map(|s| s.to_string()).collect()),
        observed_at: chrono::Utc::now(),
    };
    let plan = plan_run(
        &inv,
        &[obs("vllm-engine", false, None), obs("llamacpp-router", true, Some(&["qwen3.6-35b-a3b", "qwen3.5-0.8b"]))],
        &WorkloadName::new("vllm-engine").unwrap(),
    )
    .unwrap();
    let labels: Vec<String> = plan.steps.iter().map(|s| s.label()).collect();
    assert_eq!(
        labels[..3],
        [
            "record intent: vllm-engine → Running",
            "unload qwen3.5-0.8b from llamacpp-router",
            "unload qwen3.6-35b-a3b from llamacpp-router",
        ],
        "{labels:?}"
    );
    assert!(!labels.iter().any(|l| l == "stop llamacpp-router"), "{labels:?}");
}
