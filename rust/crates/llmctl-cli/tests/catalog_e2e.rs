//! M5c item 3 end to end: a registry bundle imported into a temp agent
//! (dry-run report, confirm token, idempotent re-import), records changed
//! by verb, stores scanned, and the reverse export pinned to
//! `tests/fixtures/rust_parity/catalog_export.json` (which
//! `tests/test_rust_catalog_export.py` feeds to Python's `import-registry`).
//! `UPDATE_FIXTURES=1` rewrites that file.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

struct Agent {
    child: Child,
    _dir: tempfile::TempDir,
    root: PathBuf,
    socket: PathBuf,
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/rust_parity")
}

fn touch(p: &Path) {
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, "x").unwrap();
}

fn start() -> Agent {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    touch(&root.join("models/hf2local/Qwen3.8-27B-FP8/config.json"));
    touch(&root.join("models/hf2local/New-Model/config.json"));
    touch(&root.join("ollama/manifests/registry.ollama.ai/library/qwen3-coder/30b"));
    std::fs::create_dir_all(root.join("cache/hub/models--acme--Small-Model")).unwrap();
    let host = "workstation";
    let p = |s: &str| root.join(s).display().to_string();
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [{
            "name": "coder", "engine": "fake", "model": "fake/coder",
            "gpus": [{"host": host, "index": 0}], "port": 8100,
            "unit": {"scope": "user", "name": "llmctl-next-fake-coder.service"},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": "llmctl-next-fake-coder.service"},
            "ownership": {"kind": "external"}
        }, {
            "name": "ollama", "engine": "fake", "engine_kind": "ollama", "model": "ollama",
            "gpus": [{"host": host, "index": 0}], "port": 8101,
            "unit": {"scope": "user", "name": "llmctl-next-fake-ollama.service"},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": "llmctl-next-fake-ollama.service"},
            "ownership": {"kind": "external"}
        }],
        "stores": [
            {"name": "hf-cache", "kind": "hf_cache", "path": p("cache/hub"), "reserve_gib": 1},
            {"name": "hf2local", "kind": "directory", "path": p("models/hf2local"), "reserve_gib": 1},
            {"name": "ollama", "kind": "ollama", "path": p("ollama"), "reserve_gib": 1, "workload": "ollama"}
        ],
        "revision": 1
    }});
    std::fs::write(root.join("inventory.json"), inv.to_string()).unwrap();
    let socket = root.join("control.sock");
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&socket).arg("--data-dir").arg(&root).arg("daemon")
        .stdout(Stdio::null()).stderr(Stdio::null()).spawn().expect("spawn daemon");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket).is_err() {
        assert!(std::time::Instant::now() < deadline, "daemon did not come up");
        std::thread::sleep(Duration::from_millis(50));
    }
    Agent { child, _dir: dir, root, socket }
}

fn cli(a: &Agent, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&a.socket).args(args).output().expect("run llmctl-next");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn json(a: &Agent, args: &[&str]) -> serde_json::Value {
    let (code, out, err) = cli(a, args);
    assert_eq!(code, 0, "{args:?}: {out}{err}");
    serde_json::from_str(&out).unwrap_or_else(|e| panic!("{args:?}: {e}: {out}"))
}

fn outcome<'a>(report: &'a serde_json::Value, name: &str) -> (&'a str, Option<&'a str>, &'a str) {
    let r = report["rows"].as_array().unwrap().iter().find(|r| r["name"] == name).unwrap_or_else(|| panic!("{name}"));
    (r["outcome"].as_str().unwrap(), r["store"].as_str(), r["reason"].as_str().unwrap_or(""))
}

#[test]
fn catalog_import_verbs_scan_and_export() {
    let a = start();
    let root = a.root.display().to_string();
    // The bundle names /srv/...: point it at this agent's temp stores.
    let bundle_text = std::fs::read_to_string(fixtures().join("catalog_bundle.json")).unwrap().replace("/srv", &root);
    let bundle = a.root.join("bundle.json");
    std::fs::write(&bundle, &bundle_text).unwrap();
    let b = bundle.to_str().unwrap();

    // 1. Without --as, this host's registry name is someone else's.
    let report = json(&a, &["catalog", "import", b, "--json"]);
    let (o, _, why) = outcome(&report, "qwen3.8-27b");
    assert_eq!(o, "unmapped");
    assert!(why.contains("another host") && why.contains("--as ws-desktop"), "{why}");

    // 2. The dry run with --as: every row has an outcome and a reason.
    let report = json(&a, &["catalog", "import", b, "--as", "ws-desktop", "--json"]);
    assert_eq!(report["applied"], false);
    assert_eq!(report["counts"], serde_json::json!({"new": 6, "no_store": 3, "unmapped": 3}), "{report:#}");
    assert_eq!(report["profiles"], 2);
    assert_eq!(outcome(&report, "qwen3.8-27b"), ("new", Some("hf2local"), ""));
    assert_eq!(outcome(&report, "qwen3.8-27b-uncensored").1, Some("hf-cache"));
    assert_eq!(outcome(&report, "qwen3-coder:30b").1, Some("ollama"));
    assert_eq!(outcome(&report, "sglang-test").1, Some("hf-cache"));
    let (o, _, why) = outcome(&report, "gpt-oss-120b-mxfp4");
    assert!(o == "no_store" && why.contains("in no declared store"), "{why}");
    assert!(outcome(&report, "text-embedding-nomic-embed-text").2.contains("no lmstudio store"));
    assert_eq!(outcome(&report, "freetoken-lane").0, "no_store");
    assert!(outcome(&report, "qwen/qwen3-4b").2.contains("mac-laptop.local"));
    assert!(outcome(&report, "gpt-5-proxy").2.contains("route, not weights"));
    assert!(outcome(&report, "qwen3.8-27b-dup").2.contains("earlier in this bundle"));
    assert!(json(&a, &["model", "list", "--json"])["models"].as_array().unwrap().is_empty(), "a dry run writes nothing");

    // 3. A token that is not this report's is refused; the right one applies.
    let (code, _, err) = cli(&a, &["catalog", "import", b, "--as", "ws-desktop", "--confirm", "sha256:nope"]);
    assert_eq!(code, 4, "{err}");
    let token = report["confirm"].as_str().unwrap();
    let done = json(&a, &["catalog", "import", b, "--as", "ws-desktop", "--confirm", token, "--json"]);
    assert_eq!(done["applied"], true);

    // 4. Again: nothing new, nothing stashed twice; the host name is kept.
    let again = json(&a, &["catalog", "import", b, "--json"]);
    assert_eq!(again["counts"], serde_json::json!({"present": 10, "unmapped": 2}), "the duplicate row matches a record now: {again:#}");
    let list = json(&a, &["model", "list", "--json"]);
    assert_eq!(list["host_names"], serde_json::json!(["ws-desktop"]));
    assert_eq!(list["models"].as_array().unwrap().len(), 8);
    assert_eq!((list["disabled_hidden"].as_u64(), list["unmapped"].as_u64(), list["profiles"].as_u64()), (Some(1), Some(3), Some(2)));
    assert_eq!(json(&a, &["model", "list", "--all", "--json"])["models"].as_array().unwrap().len(), 9);

    // 5. The reverse export, pinned (paths back to /srv for the fixture).
    let out = a.root.join("export.json");
    let (code, text, err) = cli(&a, &["catalog", "export", out.to_str().unwrap()]);
    assert_eq!(code, 0, "{text}{err}");
    let exported = std::fs::read_to_string(&out).unwrap().replace(&root, "/srv");
    let exported: serde_json::Value = serde_json::from_str(&exported).unwrap();
    assert_eq!(exported["models"].as_array().unwrap().len(), 12, "every bundle row comes back");
    let pinned = fixtures().join("catalog_export.json");
    if std::env::var_os("UPDATE_FIXTURES").is_some() {
        std::fs::write(&pinned, serde_json::to_string_pretty(&exported).unwrap() + "\n").unwrap();
    }
    let want: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&pinned).unwrap()).unwrap();
    assert_eq!(exported, want, "export moved off {}", pinned.display());

    // 6. Records by verb.
    let m = json(&a, &["model", "show", "qwen3-coder:30b", "--json"]);
    assert_eq!((m["engine"].as_str(), m["store"].as_str()), (Some("ollama"), Some("ollama")));
    let new_model = a.root.join("models/hf2local/New-Model");
    let added = json(&a, &["model", "add", "mylocal", "--engine", "vllm", "--path", new_model.to_str().unwrap(), "--set", "max_context=8192", "--json"]);
    assert_eq!(added["model"]["store"], "hf2local");
    assert_eq!(added["model"]["extra"]["max_context"], 8192);
    let (code, _, err) = cli(&a, &["model", "add", "mylocal", "--engine", "vllm", "--path", "/x/y"]);
    assert_eq!(code, 4, "same engine and name: {err}");
    let (code, _, err) = cli(&a, &["model", "add", "x", "--engine", "tensorrt", "--source", "a"]);
    assert_eq!(code, 6, "{err}");
    let edited = json(&a, &["model", "edit", "mylocal", "--set", "quantization=fp8", "--unset", "max_context", "--json"]);
    assert!(edited["model"]["extra"].get("max_context").is_none());
    assert_eq!(edited["model"]["extra"]["quantization"], "fp8");
    let (code, _, err) = cli(&a, &["model", "edit", "mylocal", "--set", "bogus=1"]);
    assert_eq!(code, 6, "{err}");
    let (code, _, err) = cli(&a, &["model", "edit", "nope", "--set", "notes=x"]);
    assert_eq!(code, 5, "{err}");
    let (code, out, err) = cli(&a, &["model", "disable", "mylocal"]);
    assert_eq!(code, 0, "{out}{err}");
    assert_eq!(json(&a, &["model", "show", "mylocal", "--json"])["enabled"], false);
    let (code, out, err) = cli(&a, &["model", "enable", "mylocal"]);
    assert_eq!(code, 0, "{out}{err}");

    // 7. Scan: what the stores hold, matched to records; gone weights listed.
    let scan = json(&a, &["model", "scan", "--json"]);
    let found = |src_end: &str| {
        scan["found"].as_array().unwrap().iter().find(|f| f["source"].as_str().unwrap().ends_with(src_end)).cloned().unwrap()
    };
    assert!(found("New-Model")["record"].is_string());
    assert!(found("Qwen3.8-27B-FP8")["record"].is_string());
    assert!(found("qwen3-coder:30b")["record"].is_string());
    assert!(found("acme/Small-Model")["record"].is_string(), "the sglang record names that repo");
    assert!(scan["missing"].as_array().unwrap().is_empty());
    touch(&a.root.join("models/hf2local/Unlisted/config.json"));
    std::fs::remove_dir_all(&new_model).unwrap();
    let scan = json(&a, &["model", "scan", "--store", "hf2local", "--json"]);
    assert_eq!(scan["stores"], serde_json::json!(["hf2local"]));
    assert_eq!(scan["missing"][0]["name"], "mylocal");
    let unlisted: Vec<_> = scan["found"].as_array().unwrap().iter().filter(|f| f["record"].is_null()).collect();
    assert_eq!(unlisted.len(), 1, "{scan:#}");
    let scan = json(&a, &["model", "scan", "--store", "hf2local", "--add", "--json"]);
    assert_eq!(scan["added"], 1);
    let m = json(&a, &["model", "show", "Unlisted", "--json"]);
    assert_eq!((m["origin"].as_str(), m["store"].as_str()), (Some("scan"), Some("hf2local")));
}
