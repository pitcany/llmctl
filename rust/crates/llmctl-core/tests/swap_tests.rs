//! llama-swap lane records (M4b step 4): round trip, validation, matrix.
//!
//! Fixtures are the two Macs' live configs (2026-11-17), sanitised for the
//! public repo: comments removed, home directory replaced. The comment
//! handling is covered by the synthetic config below. The live files
//! themselves (with comments) round-trip in `live_configs_round_trip`
//! (ignored; point LLMCTL_SWAP_LIVE_DIR at a directory of *.yaml).

use std::collections::{BTreeMap, BTreeSet};

use llmctl_core::swap::{self, SwapError};
use llmctl_core::EngineKind;

const M5: &str = include_str!("fixtures/llama-swap/m5-sanitized.yaml");
const MBP2: &str = include_str!("fixtures/llama-swap/mbp2-sanitized.yaml");

const COMMENTED: &str = r#"# header line one
# header line two
healthCheckTimeout: 300
logLevel: info  # chatty enough

# macro notes
macros:
  # the engine venv
  eng: "/opt/eng/bin/serve"

matrix:
  vars:
    a: lane-a
    b: lane-b
  evict_costs:
    a: 100   # keep a
  sets:
    pair: "a & b"

hooks:
  on_startup:
    preload:
      - lane-a

models:
  # lane a notes
  # second line
  lane-a:
    name: "Lane A"
    # inside lane a
    cmd: |
      bash -lc 'exec ${eng} --port ${PORT} --flag "x # not a comment"'
    proxy: http://127.0.0.1:${PORT}
    ttl: 0  # resident
  lane-b:
    cmd: ${eng} --port ${PORT}
    proxy: http://127.0.0.1:${PORT}
    capabilities:
      in: [text, image]
      tools: true
# trailing note
"#;

#[test]
fn both_mac_configs_round_trip() {
    for (name, text) in [("m5", M5), ("mbp2", MBP2)] {
        let (cfg, rendered) = swap::roundtrip(text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(!cfg.lanes.is_empty(), "{name}");
        // A render of a render is byte-stable.
        let (_, again) = swap::roundtrip(&rendered).unwrap();
        assert_eq!(rendered, again, "{name}: render is not a fixed point");
        assert!(swap::validate(&cfg).is_empty(), "{name}: {:?}", swap::validate(&cfg));
    }
}

#[test]
fn m5_lanes_are_typed_by_engine() {
    let cfg = swap::import(M5).unwrap();
    let kind = |id: &str| cfg.lane(id).unwrap().engine_kind.clone();
    assert_eq!(kind("qwen35b-m5"), EngineKind::Mtplx);
    assert_eq!(kind("qwen27b-vision-m5"), EngineKind::MlxVlm);
    assert_eq!(kind("mimo9b-m5"), EngineKind::MlxServe);
    assert_eq!(kind("glm53-m5"), EngineKind::Ds4);
    assert_eq!(kind("deepseek41-1017-mxfp4-m5"), EngineKind::Ds4);
    assert_eq!(kind("qwen35b-m1"), EngineKind::Other("forwarder".into()));
    // Lane order is file order, not alphabetical.
    assert_eq!(cfg.lanes[0].id, "deepseek41-1017-mxfp4-m5");
    assert_eq!(cfg.lane("mimo9b-m5").unwrap().ttl(), Some(0));
    assert_eq!(cfg.top_order.last().map(String::as_str), Some("models"));
}

#[test]
fn comments_survive_at_their_keys_and_round_trip() {
    let (cfg, rendered) = swap::roundtrip(COMMENTED).unwrap();
    let n = &cfg.notes;
    assert_eq!(n.header, vec!["header line one", "header line two"]);
    assert_eq!(n.inline["/logLevel"], "chatty enough");
    assert_eq!(n.before["/macros"], vec!["macro notes"]);
    assert_eq!(n.before["/macros/eng"], vec!["the engine venv"]);
    assert_eq!(n.inline["/matrix/evict_costs/a"], "keep a");
    assert_eq!(n.before["/models/lane-a"], vec!["lane a notes", "second line"]);
    assert_eq!(n.before["/models/lane-a/cmd"], vec!["inside lane a"]);
    assert_eq!(n.inline["/models/lane-a/ttl"], "resident");
    assert_eq!(n.trailing, vec!["trailing note"]);
    // The '#' inside the cmd block is content, not a comment.
    let a = cfg.lane("lane-a").unwrap();
    assert!(a.cmd().unwrap().contains("x # not a comment"));
    assert!(rendered.starts_with("# header line one\n# header line two\n"));
    assert!(rendered.contains("  # lane a notes\n  # second line\n  lane-a:\n"));
    assert!(rendered.contains("    ttl: 0  # resident\n"));
    assert!(swap::validate(&cfg).is_empty(), "{:?}", swap::validate(&cfg));
}

#[test]
fn an_edit_renders_and_proves_itself() {
    let mut cfg = swap::import(M5).unwrap();
    // The M4a carry-over: point mlx-serve at the rollback version.
    let old = cfg.macro_value("mlxserve").unwrap().to_string();
    let new = old.replace("26.10.1", "26.9.5");
    assert_ne!(old, new);
    cfg.set_macro("mlxserve", &new);
    cfg.lane_mut("qwen4b-m5").unwrap().set("ttl", 600.into());
    let rendered = swap::check_render(&cfg).unwrap();
    let back = swap::import(&rendered).unwrap();
    assert_eq!(back.macro_value("mlxserve"), Some(new.as_str()));
    assert_eq!(back.lane("qwen4b-m5").unwrap().ttl(), Some(600));
    // Nothing else moved.
    let orig = swap::import(M5).unwrap();
    assert_eq!(back.lanes.len(), orig.lanes.len());
    assert_ne!(back.digest(), orig.digest());
}

#[test]
fn validation_refuses_what_llama_swap_would() {
    let base = swap::import(COMMENTED).unwrap();

    let mut c = base.clone();
    c.lane_mut("lane-b").unwrap().set("cmd", "${nope} --port ${PORT}".into());
    let errs = swap::validate(&c);
    assert!(errs.iter().any(|e| e.contains("undefined macro ${nope}")), "{errs:?}");

    let mut c = base.clone();
    c.lanes.retain(|l| l.id != "lane-a");
    let errs = swap::validate(&c);
    assert!(errs.iter().any(|e| e.contains("preload names unknown lane")), "{errs:?}");
    assert!(errs.iter().any(|e| e.contains("references unknown model")), "{errs:?}");

    let mut c = base.clone();
    c.matrix.as_mut().unwrap().sets.push(("bad".into(), "a & +ghost".into()));
    let errs = swap::validate(&c);
    assert!(errs.iter().any(|e| e.contains("undefined set")), "{errs:?}");

    let mut c = base.clone();
    c.matrix.as_mut().unwrap().evict_costs = vec![("a".into(), 0.into())];
    assert!(swap::validate(&c).iter().any(|e| e.contains("positive integer")));

    let mut c = base.clone();
    c.lane_mut("lane-a").unwrap().set("ttl", (-1).into());
    assert!(swap::validate(&c).iter().any(|e| e.contains("ttl")));

    let mut c = base;
    c.lane_mut("lane-a").unwrap().set("cmd", "".into());
    assert!(swap::validate(&c).iter().any(|e| e.contains("cmd is missing")));
}

#[test]
fn matrix_dsl_matches_the_python_port() {
    let refs = BTreeMap::new();
    assert_eq!(
        swap::expand_dsl("a & (b | c)", &refs).unwrap(),
        vec![vec!["a", "b"], vec!["a", "c"]]
    );
    // & binds tighter than |.
    assert_eq!(
        swap::expand_dsl("a | b & c", &refs).unwrap(),
        vec![vec!["a".to_string()], vec!["b".into(), "c".into()]]
    );
    assert!(swap::expand_dsl("a & (b", &refs).is_err());
    assert!(swap::expand_dsl("a $ b", &refs).is_err());
    assert!(swap::expand_dsl("", &refs).is_err());
    assert!(swap::expand_dsl("+", &refs).is_err());
}

#[test]
fn m5_matrix_solves_like_llama_swap() {
    let cfg = swap::import(M5).unwrap();
    let m = cfg.matrix.as_ref().unwrap();
    let ids: BTreeSet<&str> = cfg.lanes.iter().map(|l| l.id.as_str()).collect();
    let sets = swap::expand_matrix(m, &ids).unwrap();
    // +m1 is defined first in dependency order; `single` expands to two.
    assert_eq!(sets[0].name, "m1");
    assert_eq!(sets.iter().filter(|s| s.name == "single").count(), 2);
    let costs = swap::resolved_costs(m);
    assert_eq!(costs.get("mimo9b-m5"), Some(&100));

    // The coding pair is up; asking for GLM (in `deep`, without mimo)
    // evicts both — mimo's cost 100 included.
    let running = vec!["mimo9b-m5".to_string(), "qwen35b-m5".to_string()];
    let (evict, set, cost) = swap::solve(&sets, &costs, "glm53-m5", &running);
    assert_eq!(set, "deep");
    assert_eq!(evict, running);
    assert_eq!(cost, 101);

    // Asking for the 4B keeps the coding pair (`coding` holds all three).
    let (evict, set, cost) = swap::solve(&sets, &costs, "qwen4b-m5", &running);
    assert_eq!((evict.len(), set.as_str(), cost), (0, "coding", 0));

    // A lane in no set runs alone.
    let (evict, set, _) = swap::solve(&sets, &costs, "qwen27b-uncensored-m5", &[]);
    assert!(evict.is_empty());
    assert_eq!(set, "risky_long");
}

#[test]
fn unsupported_yaml_is_refused_not_approximated() {
    assert!(matches!(swap::import("models: !tagged {}"), Err(SwapError::Unsupported { .. })));
    assert!(matches!(swap::import("? [a]\n: 1\n"), Err(SwapError::Unsupported { .. })));
    assert!(matches!(swap::import(": : :"), Err(SwapError::Parse(_))));
}

#[test]
#[ignore]
fn live_configs_round_trip() {
    let dir = std::env::var("LLMCTL_SWAP_LIVE_DIR").expect("set LLMCTL_SWAP_LIVE_DIR");
    let mut n = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let (cfg, rendered) =
            swap::roundtrip(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let notes = cfg.notes.count();
        let comments = text.lines().filter(|l| l.trim_start().starts_with('#')).count();
        println!(
            "{}: {} lanes, {} notes kept from {} comment lines, render {} lines, valid: {:?}",
            path.display(),
            cfg.lanes.len(),
            notes,
            comments,
            rendered.lines().count(),
            swap::validate(&cfg)
        );
        assert!(notes >= comments, "{}: comments lost", path.display());
        if let Ok(out) = std::env::var("LLMCTL_SWAP_RENDER_DIR") {
            let name = path.file_name().unwrap().to_string_lossy().replace(".yaml", ".render.yaml");
            std::fs::write(std::path::Path::new(&out).join(name), &rendered).unwrap();
        }
        n += 1;
    }
    assert!(n > 0);
}

// ---- M4c: lane swap prediction ---------------------------------------------------

const MATRIX_CONFIG: &str = r#"
matrix:
  vars:
    mimo: mimo9b
    q35: qwen35b
    q4: qwen4b
    fn: flash
    glm: glm53
  evict_costs:
    mimo: 100
  sets:
    coding: "mimo & q35 & q4"
    risky: "fn & mimo"
    deep: "glm"
models:
  mimo9b: {cmd: "a --port ${PORT}"}
  qwen35b: {cmd: "b --port ${PORT}"}
  qwen4b: {cmd: "c --port ${PORT}"}
  flash: {cmd: "d --port ${PORT}"}
  glm53: {cmd: "e --port ${PORT}"}
"#;

#[test]
fn a_lane_swap_is_predicted_from_the_matrix() {
    use llmctl_core::swap::{import, predict_swap};
    let cfg = import(MATRIX_CONFIG).unwrap();
    let run = |ids: &[&str]| ids.iter().map(|s| s.to_string()).collect::<Vec<_>>();

    // Fits beside what runs: nothing is evicted.
    let p = predict_swap(&cfg, "qwen4b", &run(&["mimo9b"])).unwrap();
    assert_eq!((p.evicts.clone(), p.set.as_str(), p.already_loaded), (vec![], "coding", false));

    // Flash shares a set with MiMo only: the 4B goes, MiMo stays.
    let p = predict_swap(&cfg, "flash", &run(&["mimo9b", "qwen4b"])).unwrap();
    assert_eq!((p.evicts, p.set.as_str(), p.cost), (run(&["qwen4b"]), "risky", 1));

    // GLM runs alone: everything goes, MiMo's cost included.
    let p = predict_swap(&cfg, "glm53", &run(&["mimo9b", "qwen4b"])).unwrap();
    assert_eq!((p.evicts, p.cost), (run(&["mimo9b", "qwen4b"]), 101));

    // Already running: nothing to do.
    let p = predict_swap(&cfg, "mimo9b", &run(&["mimo9b", "qwen4b"])).unwrap();
    assert!(p.already_loaded && p.evicts.is_empty());

    assert!(predict_swap(&cfg, "nope", &[]).unwrap_err().contains("no lane"));
}

#[test]
fn without_a_matrix_llama_swap_keeps_one_lane_and_groups_are_not_guessed() {
    use llmctl_core::swap::{import, predict_swap};
    let plain = import("models:\n  a: {cmd: \"x --port ${PORT}\"}\n  b: {cmd: \"y --port ${PORT}\"}\n").unwrap();
    let p = predict_swap(&plain, "b", &["a".to_string()]).unwrap();
    assert_eq!((p.evicts, p.set.as_str()), (vec!["a".to_string()], ""));
    assert!(predict_swap(&plain, "a", &["a".to_string()]).unwrap().evicts.is_empty());

    let grouped = import(
        "groups:\n  g: {swap: false, members: [a, b]}\nmodels:\n  a: {cmd: \"x --port ${PORT}\"}\n  b: {cmd: \"y --port ${PORT}\"}\n",
    )
    .unwrap();
    assert!(predict_swap(&grouped, "b", &["a".to_string()]).unwrap_err().contains("groups"));
}

#[test]
fn the_budget_is_summed_over_every_combination_the_matrix_allows() {
    let cfg = llmctl_core::swap::import(
        r#"matrix:
  vars:
    f: fast
    c: coder
    b: big
    r: remote
  sets:
    pair: "f & c & r"
    solo: "b"
    riskyall: "f & c & b"
models:
  fast: {cmd: "/e/mtplx serve --port ${PORT}"}
  coder: {cmd: "/e/mtplx serve --port ${PORT}"}
  big: {cmd: "/e/mtplx serve --port ${PORT}"}
  remote: {cmd: "bash -lc 'exec /a/m1-worker.sh forward --port ${PORT} --service router'"}
"#,
    )
    .unwrap();
    let peaks = |pairs: &[(&str, f64)]| -> std::collections::BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    };

    // All known, all fit except the set that is over on purpose. The
    // forwarder weighs nothing and needs no figure.
    let r = llmctl_core::swap::budget_report(&cfg, &peaks(&[("fast", 20.0), ("coder", 19.2), ("big", 50.0)]), 60);
    assert!(r.over.is_empty(), "{r:?}");
    assert!(r.unknown.is_empty(), "{r:?}");
    assert!(r.lines.iter().any(|l| l.starts_with("INFO matrix set riskyall") && l.contains("90G > 60G")), "{r:?}");
    // 19.2 rounds up to 20: pair = 40, solo = 50 is the largest fit.
    assert!(
        r.lines.iter().any(|l| l.starts_with("PASS matrix: 2 other set combination(s) fit the 60G budget")
            && l.contains("solo [big] at 50G")),
        "{r:?}"
    );

    // Over the budget: named, with its sum.
    let r = llmctl_core::swap::budget_report(&cfg, &peaks(&[("fast", 20.0), ("coder", 20.0), ("big", 61.0)]), 60);
    assert_eq!(r.over, vec![("solo".to_string(), 61)]);
    assert!(r.lines.iter().any(|l| l == "FAIL matrix set solo [big] peaks at 61G > 60G budget"), "{r:?}");

    // A lane without a figure: its sets are lower bounds, and said so.
    let r = llmctl_core::swap::budget_report(&cfg, &peaks(&[("fast", 20.0), ("big", 50.0)]), 60);
    assert_eq!(r.unknown, vec!["coder".to_string()]);
    assert!(r.lines.iter().any(|l| l.starts_with("WARN matrix set pair") && l.contains("no peak figure for coder")), "{r:?}");
}


#[test]
fn lanes_outside_every_matrix_set_are_named() {
    // Every lane of the synthetic config is in the `pair` set.
    let mut cfg = swap::import(COMMENTED).unwrap();
    assert_eq!(swap::lanes_outside_sets(&cfg), Some(BTreeSet::new()));
    // A new lane with no var and no set: llama-swap runs it alone.
    let mut c = cfg.lanes[1].clone();
    c.id = "lane-c".into();
    cfg.lanes.push(c);
    assert_eq!(swap::lanes_outside_sets(&cfg), Some(BTreeSet::from(["lane-c".to_string()])));
    // A var alone does not put a lane in a set.
    cfg.matrix.as_mut().unwrap().vars.push(("c".into(), "lane-c".into()));
    assert_eq!(swap::lanes_outside_sets(&cfg), Some(BTreeSet::from(["lane-c".to_string()])));
    cfg.matrix.as_mut().unwrap().sets.push(("solo".into(), "c".into()));
    assert_eq!(swap::lanes_outside_sets(&cfg), Some(BTreeSet::new()));
    // No matrix (groups or nothing): the question does not apply.
    assert_eq!(swap::lanes_outside_sets(&swap::import(MBP2).unwrap()), None);
}
