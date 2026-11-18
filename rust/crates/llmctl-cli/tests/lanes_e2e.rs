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
    /// What `/running` lists; a chat request for a lane adds it.
    running: Arc<Mutex<Vec<String>>>,
    /// Lanes the fake drops when it loads one (the test plays the matrix).
    evict_on_load: Arc<Mutex<Vec<String>>>,
    /// The router-wide in-flight count its event stream reports.
    inflight: Arc<Mutex<u64>>,
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
    let running = Arc::new(Mutex::new(vec!["fast".to_string()]));
    let evict_on_load = Arc::new(Mutex::new(Vec::<String>::new()));
    let (r2, e2) = (running.clone(), evict_on_load.clone());
    let inflight = Arc::new(Mutex::new(0u64));
    let i2 = inflight.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = [0u8; 2048];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            if req.starts_with("GET /api/events") {
                // The first events of llama-swap's stream, then close.
                let n = *i2.lock().unwrap();
                let _ = write!(
                    s,
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\nevent:message\ndata:{{\"type\":\"logData\",\"data\":\"x\"}}\n\nevent:message\ndata:{{\"type\":\"inflight\",\"data\":\"{{\\\"total\\\":{n}}}\"}}\n\n"
                );
                continue;
            }
            let body = if req.starts_with("GET /running") {
                let rows: Vec<String> = r2
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|m| format!(r#"{{"model":"{m}","state":"ready"}}"#))
                    .collect();
                format!(r#"{{"running":[{}]}}"#, rows.join(","))
            } else if req.starts_with("POST /v1/chat/completions") {
                // Load the requested lane the way llama-swap does: evict
                // first, then run it.
                let model = req
                    .split("\"model\":\"")
                    .nth(1)
                    .and_then(|r| r.split('"').next())
                    .unwrap_or("")
                    .to_string();
                let gone = e2.lock().unwrap().clone();
                let mut run = r2.lock().unwrap();
                run.retain(|m| !gone.contains(m));
                if !run.contains(&model) {
                    run.push(model);
                }
                r#"{"choices":[{"message":{"content":"ok"}}]}"#.to_string()
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
    FakeSwap { port, lanes, running, evict_on_load, inflight }
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
        .env("LLMCTL_NEXT_INFLIGHT_WAIT_MS", "400")
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

/// M4c lane swap: the records predict what llama-swap evicts, the lane is
/// requested, and the journal says whether the eviction matched.
#[test]
fn a_lane_swap_predicts_loads_and_compares_the_eviction() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let config = root.join("config.yaml");
    std::fs::write(
        &config,
        r#"matrix:
  vars:
    f: fast
    c: coder
    b: big
  sets:
    pair: "f & c"
    solo: "b"
models:
  fast: {cmd: "a --port ${PORT}"}
  coder: {cmd: "b --port ${PORT}"}
  big: {cmd: "c --port ${PORT}"}
"#,
    )
    .unwrap();
    let log = root.join("swap.log");
    std::fs::write(&log, "").unwrap();
    let fake = fake_swap(config.clone(), log.clone());
    let d = spawn(&root.join("data"), fake.port);
    let s = &d.socket;

    // No records yet: refused with the way out.
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "coder", "--dry-run"]);
    assert!(!ok && text.contains("lanes import swap"), "{text}");
    let (ok, text) = cli(
        s,
        &["lanes", "import", "swap", "--config", config.to_str().unwrap(), "--log", log.to_str().unwrap()],
    );
    assert!(ok, "{text}");
    let digest = token_after(&text, "--confirm ");
    let (ok, text) = cli(
        s,
        &["lanes", "import", "swap", "--config", config.to_str().unwrap(), "--log", log.to_str().unwrap(), "--confirm", &digest],
    );
    assert!(ok, "{text}");

    // `coder` shares a set with the running `fast`: nothing is evicted.
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "coder", "--dry-run"]);
    assert!(ok, "{text}");
    assert!(text.contains("running now: fast") && text.contains("will evict: nothing"), "{text}");
    assert!(text.contains("matrix set: pair"), "{text}");
    assert_eq!(*fake.running.lock().unwrap(), vec!["fast"], "a dry run loads nothing");

    let (ok, text) = cli(s, &["lanes", "swap", "swap", "coder", "--wait"]);
    assert!(ok, "{text}");
    assert!(text.contains("coder running; llama-swap evicted nothing, as the records predicted"), "{text}");
    assert_eq!(*fake.running.lock().unwrap(), vec!["fast", "coder"]);

    // `big` runs alone: both go. With requests in flight an evicting swap
    // is held, then refused; a swap that evicts nothing is not.
    *fake.inflight.lock().unwrap() = 2;
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "big", "--wait"]);
    assert!(!ok && text.contains("requests_in_flight") && text.contains("(2)"), "{text}");
    assert_eq!(*fake.running.lock().unwrap(), vec!["fast", "coder"], "nothing was evicted");
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "coder", "--wait"]);
    assert!(ok && text.contains("already running"), "{text}");
    *fake.inflight.lock().unwrap() = 0;

    // Quiet again: llama-swap (the fake) does evict both.
    *fake.evict_on_load.lock().unwrap() = vec!["fast".into(), "coder".into()];
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "big", "--wait"]);
    assert!(ok, "{text}");
    assert!(text.contains("will evict: fast, coder") && text.contains("matrix set: solo"), "{text}");
    assert!(text.contains("evicted coder, fast, as the records predicted"), "{text}");
    assert_eq!(*fake.running.lock().unwrap(), vec!["big"]);

    // Already running: admitted, but nothing is loaded or evicted.
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "big", "--wait"]);
    assert!(ok && text.contains("already running"), "{text}");

    // --force evicts with requests in flight.
    *fake.inflight.lock().unwrap() = 5;
    *fake.evict_on_load.lock().unwrap() = vec!["big".into()];
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "coder", "--wait", "--force"]);
    assert!(ok && text.contains("evicted big, as the records predicted"), "{text}");
    *fake.inflight.lock().unwrap() = 0;
    *fake.evict_on_load.lock().unwrap() = vec!["coder".into()];
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "big", "--wait"]);
    assert!(ok, "{text}");

    // The records say `fast` evicts `big`; this llama-swap keeps it. The
    // lane is up, and the journal flags the wrong prediction.
    *fake.evict_on_load.lock().unwrap() = vec![];
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "fast", "--wait"]);
    assert!(ok, "{text}");
    assert!(text.contains("PREDICTION MISMATCH") && text.contains("evicted nothing but the records predicted big"), "{text}");

    // An unknown lane is refused before anything is asked of llama-swap.
    let (ok, text) = cli(s, &["lanes", "swap", "swap", "nope"]);
    assert!(!ok && text.contains("no lane"), "{text}");
}
