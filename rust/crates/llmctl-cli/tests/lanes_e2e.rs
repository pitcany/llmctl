//! llama-swap lane records end to end (M4b step 4) against a fake
//! llama-swap: a thread that watches the config like `-watch-config`, logs
//! `reloading configuration` / `configuration reloaded` (or a config
//! error), and serves `/v1/models` and `/running` from what it loaded.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const CONFIG: &str = r#"# fleet lanes
healthCheckTimeout: 300
macros:
  eng: "/opt/eng-1.0/bin/serve"

models:
  # the fast lane
  fast:
    name: "Fast"
    cmd: ${eng} --port ${PORT}
    proxy: http://127.0.0.1:${PORT}
    ttl: 3600
  big:
    cmd: ${eng} --big --port ${PORT}
    proxy: http://127.0.0.1:${PORT}
"#;

struct FakeSwap {
    port: u16,
    lanes: Arc<Mutex<Vec<String>>>,
}

/// Start the fake llama-swap on a free port, watching `config`, logging to
/// `log`. A config containing `REJECT` is refused like a bad file.
fn fake_swap(config: PathBuf, log: PathBuf) -> FakeSwap {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let lanes = Arc::new(Mutex::new(lane_ids(&std::fs::read_to_string(&config).unwrap())));
    let l2 = lanes.clone();
    std::thread::spawn(move || {
        let mut last = std::fs::read(&config).unwrap();
        loop {
            std::thread::sleep(Duration::from_millis(100));
            let Ok(now) = std::fs::read(&config) else { continue };
            if now == last {
                continue;
            }
            last = now.clone();
            let text = String::from_utf8_lossy(&now).to_string();
            let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
            writeln!(f, "[INFO] reloading configuration").unwrap();
            if text.contains("REJECT") {
                writeln!(f, "[ERROR] unable to load config: rejected by the fake").unwrap();
            } else {
                *l2.lock().unwrap() = lane_ids(&text);
                writeln!(f, "[INFO] configuration reloaded").unwrap();
            }
        }
    });
    let l3 = lanes.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = [0u8; 2048];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = if req.starts_with("GET /running") {
                r#"{"running":[{"model":"fast","state":"ready"}]}"#.to_string()
            } else {
                let ids: Vec<String> = l3
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|i| format!(r#"{{"id":"{i}"}}"#))
                    .collect();
                format!(r#"{{"data":[{}]}}"#, ids.join(","))
            };
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            );
        }
    });
    FakeSwap { port, lanes }
}

fn lane_ids(text: &str) -> Vec<String> {
    llmctl_core::swap::import(text)
        .map(|c| c.lanes.iter().map(|l| l.id.clone()).collect())
        .unwrap_or_default()
}

struct Daemon {
    child: std::process::Child,
    socket: PathBuf,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn(dir: &Path, port: u16) -> Daemon {
    std::fs::create_dir_all(dir).unwrap();
    let socket = dir.join("control.sock");
    let host = "workstation";
    let inv = serde_json::json!({
        "inventory": {
            "host": host,
            "platform": "linux",
            "gpus": [{"host": host, "index": 9}],
            "workloads": [{
                "name": "swap",
                "engine": "fake",
                "model": "fake/swap",
                "gpus": [{"host": host, "index": 9}],
                "port": port,
                "unit": {"scope": "user", "name": "llmctl-next-fake-swap.service"},
                "authority": {"kind": "systemd_unit", "scope": "user", "unit": "llmctl-next-fake-swap.service"},
                "ownership": {"kind": "external"}
            }],
            "revision": 1
        }
    });
    std::fs::write(dir.join("inventory.json"), inv.to_string()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(dir)
        .arg("daemon")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket).is_err() {
        assert!(child.try_wait().unwrap().is_none(), "daemon exited");
        assert!(std::time::Instant::now() < deadline, "daemon did not come up");
        std::thread::sleep(Duration::from_millis(50));
    }
    Daemon { child, socket }
}

fn cli(socket: &Path, args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(socket)
        .args(args)
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

fn token_after(text: &str, marker: &str) -> String {
    let i = text.find(marker).unwrap_or_else(|| panic!("no {marker:?} in: {text}"));
    text[i + marker.len()..]
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output().unwrap();
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).to_string()
}

#[test]
fn import_plan_write_confirm_commit_then_a_rejected_write_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let repo = dir.path().join("repo");
    std::fs::create_dir(&repo).unwrap();
    let config = repo.join("config.yaml");
    let log = dir.path().join("swap.log");
    std::fs::write(&config, CONFIG).unwrap();
    std::fs::write(&log, "").unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.name", "lanes-test"]);
    git(&repo, &["config", "user.email", "lanes-test@example.invalid"]);
    git(&repo, &["config", "commit.gpgsign", "false"]);
    git(&repo, &["add", "config.yaml"]);
    git(&repo, &["commit", "-q", "-m", "seed"]);

    let swap = fake_swap(config.clone(), log.clone());
    let d = spawn(&dir.path().join("agent"), swap.port);
    let cfg_s = config.to_str().unwrap();
    let log_s = log.to_str().unwrap();

    // Import: preview, then confirm. The config is untouched.
    let (ok, text) = cli(&d.socket, &["lanes", "import", "swap", "--config", cfg_s, "--log", log_s]);
    assert!(ok, "{text}");
    assert!(text.contains("round trip: equivalent"), "{text}");
    let digest = token_after(&text, "--confirm ");
    let (ok, text) = cli(
        &d.socket,
        &["lanes", "import", "swap", "--config", cfg_s, "--log", log_s, "--confirm", &digest],
    );
    assert!(ok && text.contains("records saved"), "{text}");
    assert_eq!(std::fs::read_to_string(&config).unwrap(), CONFIG);

    // Plan a version swap (macro) + a ttl change: shows the changes and the
    // running lane the reload restarts; nothing written yet.
    let (ok, text) = cli(&d.socket, &["lanes", "macro", "swap", "eng", "/opt/eng-2.0/bin/serve"]);
    assert!(ok, "{text}");
    assert!(text.contains("/macros/eng"), "{text}");
    assert!(text.contains("reload restarts running lanes: fast"), "{text}");
    let plan = token_after(&text, "--confirm ");
    assert_eq!(std::fs::read_to_string(&config).unwrap(), CONFIG);

    // A stale confirmation is refused.
    let (ok, text) = cli(
        &d.socket,
        &["lanes", "macro", "swap", "eng", "/opt/eng-3.0/bin/serve", "--confirm", &plan],
    );
    assert!(!ok && text.contains("stale confirmation"), "{text}");

    // Apply: written, reload confirmed, store and git follow.
    let (ok, text) = cli(
        &d.socket,
        &["lanes", "macro", "swap", "eng", "/opt/eng-2.0/bin/serve", "--confirm", &plan, "--wait"],
    );
    assert!(ok, "{text}");
    let now = std::fs::read_to_string(&config).unwrap();
    assert!(now.contains("eng-2.0") && now.contains("# the fast lane"), "{now}");
    assert_eq!(
        std::fs::read_to_string(repo.join("config.yaml.llmctl-previous")).unwrap(),
        CONFIG
    );
    let logline = git(&repo, &["log", "--oneline", "-1"]);
    assert!(logline.contains("llmctl-next(swap): macro eng"), "{logline}");
    assert!(git(&repo, &["status", "--porcelain", "config.yaml"]).trim().is_empty());
    let (ok, text) = cli(&d.socket, &["lanes", "show", "swap"]);
    assert!(ok && !text.contains("DRIFT") && text.contains("eng-2.0"), "{text}");

    // A write llama-swap rejects: the op fails, the previous bytes come
    // back (and are reloaded), the store stays on them.
    let before = std::fs::read_to_string(&config).unwrap();
    let (ok, text) = cli(&d.socket, &["lanes", "set", "swap", "big", "name", "REJECT"]);
    assert!(ok, "{text}");
    let plan = token_after(&text, "--confirm ");
    let (ok, text) = cli(
        &d.socket,
        &["lanes", "set", "swap", "big", "name", "REJECT", "--confirm", &plan, "--wait"],
    );
    assert!(!ok, "{text}");
    assert!(text.contains("rejected the config"), "{text}");
    assert_eq!(std::fs::read_to_string(&config).unwrap(), before);
    let (_, text) = cli(&d.socket, &["lanes", "show", "swap"]);
    assert!(!text.contains("DRIFT"), "{text}");
    assert_eq!(*swap.lanes.lock().unwrap(), vec!["fast", "big"]);

    // A hand edit is drift: planning refuses until re-imported.
    std::fs::write(&config, before.replace("ttl: 3600", "ttl: 60")).unwrap();
    let (ok, text) = cli(&d.socket, &["lanes", "set", "swap", "fast", "ttl", "120"]);
    assert!(!ok && text.contains("changed since the lane records"), "{text}");
    let (_, text) = cli(&d.socket, &["lanes", "show", "swap"]);
    assert!(text.contains("DRIFT"), "{text}");
}

#[test]
fn validation_refuses_before_anything_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.yaml");
    let log = dir.path().join("swap.log");
    std::fs::write(&config, CONFIG).unwrap();
    std::fs::write(&log, "").unwrap();
    let swap = fake_swap(config.clone(), log.clone());
    let d = spawn(&dir.path().join("agent"), swap.port);
    let cfg_s = config.to_str().unwrap();
    let (_, text) = cli(&d.socket, &["lanes", "import", "swap", "--config", cfg_s]);
    let digest = token_after(&text, "--confirm ");
    let (ok, _) = cli(&d.socket, &["lanes", "import", "swap", "--config", cfg_s, "--confirm", &digest]);
    assert!(ok);
    // An undefined macro: llama-swap would refuse it, so the plan does.
    let (ok, text) = cli(&d.socket, &["lanes", "set", "swap", "big", "cmd", "${nope} --port ${PORT}"]);
    assert!(!ok && text.contains("undefined macro"), "{text}");
    assert_eq!(std::fs::read_to_string(&config).unwrap(), CONFIG);
}
