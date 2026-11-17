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
