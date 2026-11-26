//! llama.cpp router models end to end (M5d) against a stand-in router: a
//! thread that serves `GET /models[?reload=1]` from the preset file,
//! `POST /models/load|unload` with a short `loading` phase, a load that
//! fails with exit code 1, and `--no-models-autoload` behaviour (a request
//! for an unloaded model is refused, never loads it). A load takes VRAM out
//! of the fake engine's `gpu_free.json`, so the measured need is checked
//! too.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Debug)]
struct Row {
    id: String,
    state: String,
    failed: bool,
}

struct Stand {
    port: u16,
    rows: Arc<Mutex<Vec<Row>>>,
    reloads: Arc<Mutex<u32>>,
    /// Chat requests refused because the model was not loaded.
    refused_chats: Arc<Mutex<u32>>,
}

const PRESET: &str = "; router models (test)\nversion = 1\n\n[*]\nflash-attn = on\n\n; the small one\n[small]\nmodel = MODELS/small.gguf\nctx-size = 8192 ; tiny\n\n[big]\nmodel = MODELS/big.gguf\n\n[broken]\nmodel = MODELS/broken.gguf\n";

fn ids_of(path: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    llmctl_core::llamacpp::import(&text)
        .map(|p| p.model_ids())
        .unwrap_or_default()
        .into_iter()
        // A section the stand-in "cannot read": a write must roll back.
        .filter(|id| !id.starts_with("ghost"))
        .collect()
}

fn read_request(s: &mut std::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = s.read(&mut chunk).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&buf).to_string();
        if let Some(head_end) = text.find("\r\n\r\n") {
            let len = text[..head_end]
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            if buf.len() >= head_end + 4 + len {
                break;
            }
        }
    }
    String::from_utf8_lossy(&buf).to_string()
}

fn model_of(req: &str) -> String {
    req.split("\"model\":\"").nth(1).and_then(|r| r.split('"').next()).unwrap_or("").to_string()
}

fn adjust_free(gpu_free: &Path, delta: i64) {
    let mut m: BTreeMap<String, i64> =
        serde_json::from_str(&std::fs::read_to_string(gpu_free).unwrap_or_else(|_| "{}".into())).unwrap();
    if let Some(v) = m.get_mut("0") {
        *v += delta;
    }
    std::fs::write(gpu_free, serde_json::to_string(&m).unwrap()).unwrap();
}

/// `vram`: MiB a model takes from card 0 when it loads.
fn stand_in(preset: PathBuf, gpu_free: PathBuf, vram: BTreeMap<String, i64>) -> Stand {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let rows = Arc::new(Mutex::new(
        ids_of(&preset)
            .into_iter()
            .map(|id| Row { id, state: "unloaded".into(), failed: false })
            .collect::<Vec<_>>(),
    ));
    let reloads = Arc::new(Mutex::new(0u32));
    let refused_chats = Arc::new(Mutex::new(0u32));
    let (r2, rl2, rc2) = (rows.clone(), reloads.clone(), refused_chats.clone());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let req = read_request(&mut s);
            let line = req.lines().next().unwrap_or("").to_string();
            let (status, body) = if line.starts_with("GET /models") {
                if line.contains("reload=1") {
                    *rl2.lock().unwrap() += 1;
                    let ids = ids_of(&preset);
                    let mut rows = r2.lock().unwrap();
                    rows.retain(|r| ids.contains(&r.id));
                    for id in ids {
                        if !rows.iter().any(|r| r.id == id) {
                            rows.push(Row { id, state: "unloaded".into(), failed: false });
                        }
                    }
                }
                let data: Vec<serde_json::Value> = r2
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|r| {
                        let mut st = serde_json::json!({"value": r.state});
                        if r.failed {
                            st["failed"] = serde_json::json!(true);
                            st["exit_code"] = serde_json::json!(1);
                        }
                        serde_json::json!({"id": r.id, "status": st})
                    })
                    .collect();
                ("200 OK", serde_json::json!({ "data": data }).to_string())
            } else if line.starts_with("POST /models/load") {
                let model = model_of(&req);
                let known = r2.lock().unwrap().iter().any(|r| r.id == model);
                if !known {
                    ("404 Not Found", r#"{"error":"model not found"}"#.to_string())
                } else {
                    for r in r2.lock().unwrap().iter_mut().filter(|r| r.id == model) {
                        r.state = "loading".into();
                        r.failed = false;
                    }
                    let (r3, gf, need) = (r2.clone(), gpu_free.clone(), vram.get(&model).copied().unwrap_or(0));
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(400));
                        let fails = model.starts_with("broken");
                        if !fails {
                            adjust_free(&gf, -need);
                        }
                        for r in r3.lock().unwrap().iter_mut().filter(|r| r.id == model) {
                            if fails {
                                r.state = "unloaded".into();
                                r.failed = true;
                            } else {
                                r.state = "loaded".into();
                            }
                        }
                    });
                    ("200 OK", r#"{"success":true}"#.to_string())
                }
            } else if line.starts_with("POST /models/unload") {
                let model = model_of(&req);
                let mut freed = 0;
                for r in r2.lock().unwrap().iter_mut().filter(|r| r.id == model) {
                    if r.state == "loaded" {
                        freed = vram.get(&model).copied().unwrap_or(0);
                    }
                    r.state = "unloaded".into();
                }
                adjust_free(&gpu_free, freed);
                ("200 OK", r#"{"success":true}"#.to_string())
            } else if line.starts_with("POST /v1/chat/completions") {
                // --no-models-autoload: an unloaded model is refused, not loaded.
                let model = model_of(&req);
                let loaded = r2.lock().unwrap().iter().any(|r| r.id == model && r.state == "loaded");
                if loaded {
                    ("200 OK", r#"{"choices":[{"message":{"content":"ok"}}]}"#.to_string())
                } else {
                    *rc2.lock().unwrap() += 1;
                    ("400 Bad Request", r#"{"error":{"message":"model is not loaded"}}"#.to_string())
                }
            } else {
                ("404 Not Found", "{}".to_string())
            };
            let _ = write!(
                s,
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    Stand { port, rows, reloads, refused_chats }
}

impl Stand {
    fn state(&self, id: &str) -> String {
        self.rows
            .lock()
            .unwrap()
            .iter()
            .find(|r| r.id == id)
            .map(|r| r.state.clone())
            .unwrap_or_else(|| "absent".into())
    }

    fn chat(&self, model: &str) -> u16 {
        let body = format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"hi"}}]}}"#);
        let mut s = std::net::TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        write!(
            s,
            "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut out = String::new();
        let _ = s.read_to_string(&mut out);
        out.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0)
    }
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

fn workload(name: &str, kind: &str, claim: Option<&str>, model: &str, port: u16) -> serde_json::Value {
    let unit = format!("llmctl-next-fake-{name}.service");
    let mut w = serde_json::json!({
        "name": name,
        "engine": "fake",
        "engine_kind": kind,
        "model": model,
        "gpus": [{"host": "workstation", "index": 0}, {"host": "workstation", "index": 1}],
        "port": port,
        "unit": {"scope": "user", "name": unit},
        "authority": {"kind": "systemd_unit", "scope": "user", "unit": unit},
        "ownership": {"kind": "external"}
    });
    if let Some(c) = claim {
        w["gpu_claim"] = serde_json::json!(c);
    }
    w
}

fn spawn(dir: &Path, port: u16) -> Daemon {
    std::fs::create_dir_all(dir).unwrap();
    let socket = dir.join("control.sock");
    let inv = serde_json::json!({
        "inventory": {
            "host": "workstation",
            "platform": "linux",
            "gpus": [{"host": "workstation", "index": 0}, {"host": "workstation", "index": 1}],
            "workloads": [
                workload("router", "llama_cpp_router", Some("on_demand"), "llamacpp-router", port),
                workload("vllm", "vllm", None, "fake/vllm", 18999),
            ],
            "revision": 1
        }
    });
    std::fs::write(dir.join("inventory.json"), inv.to_string()).unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
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
    let mut child = child;
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
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

fn token_after(text: &str, marker: &str) -> String {
    let i = text.find(marker).unwrap_or_else(|| panic!("no {marker:?} in: {text}"));
    text[i + marker.len()..].split_whitespace().next().unwrap().to_string()
}

struct Setup {
    _dir: tempfile::TempDir,
    agent: PathBuf,
    models: PathBuf,
    preset: PathBuf,
    stand: Stand,
    d: Daemon,
}

fn setup(vram: &[(&str, i64)]) -> Setup {
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");
    std::fs::create_dir_all(&models).unwrap();
    for f in ["small.gguf", "big.gguf", "broken.gguf", "extra.gguf"] {
        std::fs::write(models.join(f), b"GGUF").unwrap();
    }
    let preset = dir.path().join("router.ini");
    std::fs::write(&preset, PRESET.replace("MODELS", models.to_str().unwrap())).unwrap();
    let agent = dir.path().join("agent");
    std::fs::create_dir_all(&agent).unwrap();
    std::fs::write(agent.join("gpu_free.json"), r#"{"0": 30000, "1": 30000}"#).unwrap();
    let stand = stand_in(
        preset.clone(),
        agent.join("gpu_free.json"),
        vram.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    );
    let d = spawn(&agent, stand.port);
    let p = preset.to_str().unwrap().to_string();
    let (ok, text) = cli(&d.socket, &["llamacpp", "import", "router", "--preset", &p]);
    assert!(ok, "{text}");
    assert!(text.contains("round trip: equivalent") && text.contains("3 models"), "{text}");
    assert!(text.contains("not managed: its command line was not checked"), "{text}");
    let digest = token_after(&text, "--confirm ");
    let (ok, text) = cli(&d.socket, &["llamacpp", "import", "router", "--preset", &p, "--confirm", &digest]);
    assert!(ok && text.contains("records saved"), "{text}");
    Setup { _dir: dir, agent, models, preset, stand, d }
}

#[test]
fn load_unload_by_verb_with_measured_need_refusals_and_a_failed_load() {
    let s = setup(&[("small", 2500)]);
    let sock = &s.d.socket;

    let (ok, text) = cli(sock, &["llamacpp", "models", "router"]);
    assert!(ok && text.contains("router answers"), "{text}");
    assert!(text.contains("small") && text.contains("unloaded") && text.contains("need unknown"), "{text}");

    // Never measured, nothing declared: refused, nothing admitted.
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "small"]);
    assert!(!ok && text.contains("never measured"), "{text}");
    assert!(!text.contains("admitted"), "{text}");
    assert_eq!(s.stand.state("small"), "unloaded");

    // Declared need: the fit decides, the dry run admits nothing.
    let (ok, text) = cli(sock, &["llamacpp", "need", "router", "small", "--mib", "0=2000"]);
    assert!(ok && text.contains("(declared)"), "{text}");
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "small", "--dry-run"]);
    assert!(ok && text.contains("decision: load — fits: needs 2.0 GiB on gpu 0"), "{text}");
    assert_eq!(s.stand.state("small"), "unloaded");

    // The load: admitted, confirmed by GET /models, need measured.
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "small", "--wait"]);
    assert!(ok, "{text}");
    assert!(text.contains("small loaded after"), "{text}");
    assert!(text.contains("measured need 2.4 GiB on gpu 0"), "{text}");
    assert_eq!(s.stand.state("small"), "loaded");
    let (_, text) = cli(sock, &["llamacpp", "models", "router"]);
    assert!(text.contains("(measured)"), "{text}");
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "small"]);
    assert!(ok && text.contains("decision: already_loaded"), "{text}");

    // No autoload: a request for an unloaded model is refused and loads
    // nothing; the agent admits nothing for it either.
    let ops_before = cli(sock, &["ops"]).1.lines().count();
    assert_eq!(s.stand.chat("big"), 400);
    std::thread::sleep(Duration::from_millis(600));
    assert_eq!(s.stand.state("big"), "unloaded");
    assert_eq!(*s.stand.refused_chats.lock().unwrap(), 1);
    assert_eq!(cli(sock, &["ops"]).1.lines().count(), ops_before);
    assert_eq!(s.stand.chat("small"), 200);

    // The fit says no: refused.
    cli(sock, &["llamacpp", "need", "router", "big", "--mib", "0=40000"]);
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "big"]);
    assert!(!ok && text.contains("does not fit: needs 39.1 GiB on gpu 0"), "{text}");
    assert_eq!(s.stand.state("big"), "unloaded");

    // A load the router reports failed: the op fails and rolls back.
    cli(sock, &["llamacpp", "need", "router", "broken", "--mib", "0=1000"]);
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "broken", "--wait"]);
    assert!(!ok, "{text}");
    assert!(text.contains("the router reports the load of broken failed (exit code 1)"), "{text}");
    let (_, text) = cli(sock, &["llamacpp", "models", "router"]);
    assert!(text.contains("last load failed, exit 1"), "{text}");
    assert_eq!(s.stand.state("small"), "loaded", "a failed neighbour load leaves the others alone");

    // Unload: the router keeps running, the card gets its memory back.
    let (ok, text) = cli(sock, &["llamacpp", "unload", "router", "small", "--wait"]);
    assert!(ok && text.contains("small unloaded"), "{text}");
    assert_eq!(s.stand.state("small"), "unloaded");
    let free: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(s.agent.join("gpu_free.json")).unwrap()).unwrap();
    assert_eq!(free["0"], 30000);
    let (ok, text) = cli(sock, &["llamacpp", "unload", "router", "small"]);
    assert!(ok && text.contains("decision: not_loaded"), "{text}");
}

#[test]
fn preset_writes_reload_keep_notes_and_roll_back_when_the_router_does_not_take_them() {
    let s = setup(&[]);
    let sock = &s.d.socket;
    let before = std::fs::read_to_string(&s.preset).unwrap();

    // Set a key: plan first (nothing written), then confirm.
    let (ok, text) = cli(sock, &["llamacpp", "set", "router", "small", "ctx-size", "4096"]);
    assert!(ok && text.contains("[small] ctx-size: 8192 → 4096"), "{text}");
    let plan = token_after(&text, "--confirm ");
    assert_eq!(std::fs::read_to_string(&s.preset).unwrap(), before);
    let (ok, text) = cli(sock, &["llamacpp", "set", "router", "small", "ctx-size", "4096", "--confirm", &plan, "--wait"]);
    assert!(ok, "{text}");
    assert!(text.contains("reloaded; the router lists all 3 models"), "{text}");
    let now = std::fs::read_to_string(&s.preset).unwrap();
    assert!(now.contains("ctx-size = 4096 ; tiny") && now.contains("; the small one"), "{now}");
    assert_eq!(std::fs::read_to_string(format!("{}.llmctl-previous", s.preset.display())).unwrap(), before);
    assert!(*s.stand.reloads.lock().unwrap() >= 1);

    // Add a model: the router lists it after the reload.
    let extra = s.models.join("extra.gguf");
    let add = ["llamacpp", "add", "router", "extra", "--model", extra.to_str().unwrap(), "--arg", "ctx-size=2048"];
    let (ok, text) = cli(sock, &add);
    assert!(ok && text.contains("[extra] added"), "{text}");
    let plan = token_after(&text, "--confirm ");
    let mut args = add.to_vec();
    args.extend(["--confirm", &plan, "--wait"]);
    let (ok, text) = cli(sock, &args);
    assert!(ok && text.contains("lists all 4 models"), "{text}");
    assert_eq!(s.stand.state("extra"), "unloaded");

    // A write the router does not take (it never lists the new model):
    // the op fails, the previous bytes come back, records stay on them.
    let good = std::fs::read_to_string(&s.preset).unwrap();
    let ghost = ["llamacpp", "add", "router", "ghost", "--model", extra.to_str().unwrap()];
    let (ok, text) = cli(sock, &ghost);
    assert!(ok, "{text}");
    let plan = token_after(&text, "--confirm ");
    let mut args = ghost.to_vec();
    args.extend(["--confirm", &plan]);
    args.push("--wait");
    let saved = s.stand.reloads.lock().unwrap().to_owned();
    let (ok, text) = cli(sock, &args);
    assert!(!ok, "{text}");
    assert!(text.contains("missing [\"ghost\"]"), "{text}");
    assert_eq!(std::fs::read_to_string(&s.preset).unwrap(), good);
    assert!(*s.stand.reloads.lock().unwrap() > saved);
    let (_, text) = cli(sock, &["llamacpp", "models", "router"]);
    assert!(!text.contains("DRIFT") && !text.contains("ghost"), "{text}");

    // A hand edit since the import: further writes are refused.
    std::fs::write(&s.preset, format!("{good}; hand edit\n")).unwrap();
    let (ok, text) = cli(sock, &["llamacpp", "set", "router", "small", "ctx-size", "1024"]);
    assert!(!ok && text.contains("changed since the records were imported"), "{text}");

    // Invalid edits never reach the file.
    std::fs::write(&s.preset, &good).unwrap();
    let (ok, text) = cli(sock, &["llamacpp", "set", "router", "small", "port", "9000"]);
    assert!(!ok && text.contains("the router overrides this key"), "{text}");
}

#[test]
fn a_preset_write_is_refused_while_the_router_does_not_answer() {
    let Setup { _dir, agent, preset, d, .. } = setup(&[]);
    // Same records, but the router's port has nothing behind it (stopped).
    drop(d);
    let dead = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let d = spawn(&agent, dead);
    let before = std::fs::read_to_string(&preset).unwrap();
    let (ok, text) = cli(&d.socket, &["llamacpp", "set", "router", "small", "ctx-size", "4096"]);
    assert!(ok, "{text}");
    let plan = token_after(&text, "--confirm ");
    let (ok, text) = cli(&d.socket, &["llamacpp", "set", "router", "small", "ctx-size", "4096", "--confirm", &plan, "--wait"]);
    assert!(!ok && text.contains("does not answer GET /models"), "{text}");
    assert_eq!(std::fs::read_to_string(&preset).unwrap(), before);
    assert!(!Path::new(&format!("{}.llmctl-previous", preset.display())).exists());
}

#[test]
fn the_preload_guard_refuses_missing_shards_live_downloads_and_a_low_floor() {
    let s = setup(&[]);
    let sock = &s.d.socket;
    // A split model with its second shard missing.
    let split_dir = s.models.join("split");
    std::fs::create_dir_all(&split_dir).unwrap();
    let first = split_dir.join("M-00001-of-00002.gguf");
    std::fs::write(&first, b"GGUF").unwrap();
    let add = ["llamacpp", "add", "router", "split", "--model", first.to_str().unwrap()];
    let (_, text) = cli(sock, &add);
    let plan = token_after(&text, "--confirm ");
    let mut args = add.to_vec();
    args.extend(["--confirm", &plan, "--wait"]);
    let (ok, text) = cli(sock, &args);
    assert!(ok, "{text}");
    cli(sock, &["llamacpp", "need", "router", "split", "--mib", "0=1000"]);

    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "split"]);
    assert!(!ok, "{text}");
    assert!(text.contains("PRE-LOAD CHECK: incomplete split model (2 shards expected); missing: M-00002-of-00002.gguf"), "{text}");
    assert_eq!(s.stand.state("split"), "unloaded");

    // The shard arrives but a download is still staged next to it.
    std::fs::write(split_dir.join("M-00002-of-00002.gguf"), b"GGUF").unwrap();
    let stage = s.models.join(".cache/huggingface/download/split");
    std::fs::create_dir_all(&stage).unwrap();
    std::fs::write(stage.join("abc.incomplete"), b"").unwrap();
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "split"]);
    assert!(!ok && text.contains("still downloading"), "{text}");
    std::fs::remove_file(stage.join("abc.incomplete")).unwrap();

    // The VRAM floor (LLAMA_GUARD_MIN_FREE_MIB's successor).
    cli(sock, &["llamacpp", "need", "router", "split", "--min-free", "31000"]);
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "split", "--dry-run"]);
    assert!(!ok && text.contains("gpu 0 has only 30000 MiB free (floor 31000 MiB)"), "{text}");
    cli(sock, &["llamacpp", "need", "router", "split", "--clear", "min_free"]);
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "split", "--wait"]);
    assert!(ok, "{text}");
    assert_eq!(s.stand.state("split"), "loaded");
}

#[test]
fn an_evict_load_stops_the_at_start_neighbour_and_restores_it_when_the_load_fails() {
    let s = setup(&[("small", 9000)]);
    let sock = &s.d.socket;
    let (ok, text) = cli(sock, &["run", "vllm", "--wait"]);
    assert!(ok, "{text}");
    // vllm holds most of both cards.
    std::fs::write(s.agent.join("gpu_free.json"), r#"{"0": 2000, "1": 2000}"#).unwrap();
    std::fs::write(
        s.agent.join("gpus.json"),
        r#"[{"gpu":0,"pid":4242,"unit":"user/llmctl-next-fake-vllm.service","memory_mib":28000},
            {"gpu":1,"pid":4242,"unit":"user/llmctl-next-fake-vllm.service","memory_mib":28000}]"#,
    )
    .unwrap();
    cli(sock, &["llamacpp", "need", "router", "small", "--mib", "0=10000"]);
    cli(sock, &["llamacpp", "need", "router", "broken", "--mib", "0=10000"]);

    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "small"]);
    assert!(!ok && text.contains("does not fit"), "{text}");
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "small", "--evict", "--dry-run"]);
    assert!(ok, "{text}");
    assert!(text.contains("once vllm is stopped") && text.contains("stops first: vllm"), "{text}");

    // A failed load after the eviction: vllm comes back.
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "broken", "--evict", "--wait"]);
    assert!(!ok, "{text}");
    assert!(text.contains("stop vllm"), "{text}");
    let (_, status) = cli(sock, &["status"]);
    let vllm_line = status.lines().find(|l| l.contains("vllm")).unwrap_or_default().to_string();
    assert!(vllm_line.contains("running"), "{status}");

    // The real one: vllm stopped, small loaded.
    let (ok, text) = cli(sock, &["llamacpp", "load", "router", "small", "--evict", "--wait"]);
    assert!(ok, "{text}");
    assert_eq!(s.stand.state("small"), "loaded");
    let (_, status) = cli(sock, &["status"]);
    let vllm_line = status.lines().find(|l| l.contains("vllm")).unwrap_or_default().to_string();
    assert!(vllm_line.contains("stopped"), "{status}");
}
