//! M5c item 5 end to end: enroll a workload `adopt` cannot prove, and
//! change a record's control authority by verb (preview, then confirm).

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

struct Agent {
    child: Child,
    _dir: tempfile::TempDir,
    socket: PathBuf,
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start() -> Agent {
    use llmctl_core::launch::LaunchDefinition;
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let host = "workstation";
    let wl = |name: &str, port: u16| {
        serde_json::json!({
            "name": name, "engine": "fake", "model": format!("fake/{name}"),
            "gpus": [{"host": host, "index": 0}], "port": port,
            "unit": {"scope": "user", "name": format!("llmctl-next-fake-{name}.service")},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": format!("llmctl-next-fake-{name}.service")},
            "ownership": {"kind": "external"}
        })
    };
    let definition = LaunchDefinition::default();
    let mut big = wl("big", 8102);
    big["ownership"] = serde_json::json!({
        "kind": "managed", "definition": definition, "digest": definition.digest(),
        "adopted_at": "2026-11-17T00:00:00Z", "provenance": []
    });
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}, {"host": host, "index": 1}],
        "workloads": [wl("coder", 8100), big],
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
    Agent { child, _dir: dir, socket }
}

fn cli(a: &Agent, args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket").arg(&a.socket).args(args).output().expect("run llmctl-next");
    (
        out.status.code().unwrap_or(-1),
        format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)),
    )
}

fn token(text: &str) -> String {
    let i = text.find("--confirm ").unwrap_or_else(|| panic!("no token in {text}"));
    text[i + 10..].split_whitespace().next().unwrap().to_string()
}

fn status(a: &Agent) -> serde_json::Value {
    let (code, out) = cli(a, &["status", "--json"]);
    assert_eq!(code, 0, "{out}");
    serde_json::from_str(&out).unwrap()
}

fn workload(a: &Agent, name: &str) -> Option<serde_json::Value> {
    status(a)["workloads"].as_array().unwrap().iter().find(|w| w["name"] == name).cloned()
}

#[test]
fn enroll_an_unadoptable_record_and_change_authority_by_verb() {
    let a = start();
    let enroll = ["enroll", "lms", "--observe", "http://127.0.0.1:9123/v1", "--engine-kind", "lm_studio", "--model", "qwen3-4b", "--on-demand", "--engine", "fake"];

    // 1. Enroll an observe-only record: a preview, then the token writes it.
    let (code, text) = cli(&a, &enroll);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("would write lms") && text.contains("observe_only"), "{text}");
    assert!(workload(&a, "lms").is_none(), "a preview writes nothing");
    let t = token(&text);
    let mut with = enroll.to_vec();
    with.extend(["--confirm", &t]);
    let (code, text) = cli(&a, &with);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("enrolled lms"), "{text}");
    let lms = workload(&a, "lms").expect("enrolled");
    assert_eq!(lms["model"], "qwen3-4b");
    //    It is watched, never moved.
    let (code, text) = cli(&a, &["run", "lms", "--dry-run"]);
    assert_ne!(code, 0, "{text}");
    assert!(text.to_lowercase().contains("observe"), "{text}");

    // 2. Refusals: the name is taken; no port; a vendor app on Linux.
    let (code, text) = cli(&a, &enroll);
    assert_eq!(code, 6, "{text}");
    let (code, text) = cli(&a, &["enroll", "x", "--observe", "http://h/v1", "--engine-kind", "vllm", "--model", "m", "--engine", "fake"]);
    assert!(code == 6 && text.contains("names no port"), "{text}");
    let (code, text) = cli(&a, &["enroll", "app", "--start", "/usr/bin/open -a App", "--stop", "/usr/bin/osascript -e quit", "--port", "9200", "--engine-kind", "ollama", "--model", "m", "--engine", "fake"]);
    assert!(code == 6 && text.contains("launchd adapter only"), "{text}");

    // 3. Authority: coder becomes observe-only, then a unit again.
    let (code, text) = cli(&a, &["authority", "coder", "--observe", "http://127.0.0.1:8100"]);
    assert_eq!(code, 0, "{text}");
    assert!(text.contains("was {\"kind\":\"systemd_unit\""), "{text}");
    let t = token(&text);
    //    A token made before the inventory moved is refused.
    let (code, text) = cli(&a, &["authority", "coder", "--observe", "http://127.0.0.1:8100", "--confirm", "sha256:stale"]);
    assert_eq!(code, 6, "{text}");
    let (code, text) = cli(&a, &["authority", "coder", "--observe", "http://127.0.0.1:8100", "--confirm", &t]);
    assert_eq!(code, 0, "{text}");
    let (code, text) = cli(&a, &["run", "coder", "--dry-run"]);
    assert_ne!(code, 0, "observe-only now: {text}");
    let (code, text) = cli(&a, &["authority", "coder", "--observe", "http://127.0.0.1:9999"]);
    assert!(code == 6 && text.contains("not the record's port"), "{text}");
    let (code, text) = cli(&a, &["authority", "coder", "--systemd-unit", "llmctl-next-fake-coder.service"]);
    assert_eq!(code, 0, "{text}");
    let t = token(&text);
    let (code, text) = cli(&a, &["authority", "coder", "--systemd-unit", "llmctl-next-fake-coder.service", "--confirm", &t]);
    assert_eq!(code, 0, "{text}");
    let (code, text) = cli(&a, &["run", "coder", "--dry-run"]);
    assert_eq!(code, 0, "a unit again: {text}");

    // 4. An adopted workload keeps the unit adoption pinned.
    let (code, text) = cli(&a, &["authority", "big", "--observe", "http://127.0.0.1:8102"]);
    assert!(code == 6 && text.contains("release big"), "{text}");
    let (code, text) = cli(&a, &["authority", "nope", "--observe", "http://127.0.0.1:1"]);
    assert_eq!(code, 5, "{text}");
}
