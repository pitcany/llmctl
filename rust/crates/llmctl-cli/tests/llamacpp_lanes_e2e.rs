//! llama.cpp on the Macs (M5d items 4–5) end to end against a minimal fake
//! llama-swap: a release archive staged into its own versioned dir (local
//! archive, checksum checked, binary version checked), the lane macro
//! switched to it, a llama.cpp lane added from a model record, loaded,
//! and removed.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

const CONFIG: &str = "# mac lanes\nhealthCheckTimeout: 300\nmacros:\n  mlx: \"/opt/mlx/bin/serve\"\n\nmodels:\n  chat:\n    cmd: ${mlx} --port ${PORT}\n    proxy: http://127.0.0.1:${PORT}\n";

fn lane_ids(path: &Path) -> Vec<String> {
    llmctl_core::swap::import(&std::fs::read_to_string(path).unwrap_or_default())
        .map(|c| c.lanes.iter().map(|l| l.id.clone()).collect())
        .unwrap_or_default()
}

/// `/v1/models` from the config as it is now, `/running`, a chat request
/// loads its lane; no event stream (no in-flight count).
fn fake_swap(config: PathBuf) -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let running = Arc::new(Mutex::new(Vec::<String>::new()));
    let r2 = running.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = [0u8; 8192];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = if req.starts_with("GET /running") {
                let rows: Vec<String> =
                    r2.lock().unwrap().iter().map(|m| format!(r#"{{"model":"{m}","state":"ready"}}"#)).collect();
                format!(r#"{{"running":[{}]}}"#, rows.join(","))
            } else if req.starts_with("POST /v1/chat/completions") {
                let model = req.split("\"model\":\"").nth(1).and_then(|r| r.split('"').next()).unwrap_or("").to_string();
                let mut run = r2.lock().unwrap();
                if !run.contains(&model) {
                    run.push(model);
                }
                r#"{"choices":[{"message":{"content":"ok"}}]}"#.to_string()
            } else if req.starts_with("GET /v1/models") {
                let ids: Vec<String> = lane_ids(&config).iter().map(|i| format!(r#"{{"id":"{i}"}}"#)).collect();
                format!(r#"{{"data":[{}]}}"#, ids.join(","))
            } else {
                let _ = write!(s, "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
                continue;
            };
            let _ = write!(
                s,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (port, running)
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
    let host = "m5";
    let inv = serde_json::json!({
        "inventory": {
            "host": host,
            "platform": "linux",
            // Validation wants a device; the Mac's unified memory stands in.
            "gpus": [{"host": host, "index": 0}],
            "workloads": [{
                "name": "llama-swap",
                "engine": "fake",
                "model": "fake/swap",
                "gpus": [{"host": host, "index": 0}],
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
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

fn token_after(text: &str, marker: &str) -> String {
    let i = text.find(marker).unwrap_or_else(|| panic!("no {marker:?} in: {text}"));
    text[i + marker.len()..].split_whitespace().next().unwrap().to_string()
}

/// A release archive shaped like llama.cpp's macOS one:
/// `llama-<v>/llama-server` that prints `version: <build>` on stderr.
fn make_archive(dir: &Path, version: &str) -> (PathBuf, String) {
    let build = version.trim_start_matches('b');
    let src = dir.join(format!("src-{version}"));
    let inner = src.join(format!("llama-{version}"));
    std::fs::create_dir_all(&inner).unwrap();
    let bin = inner.join("llama-server");
    std::fs::write(&bin, format!("#!/bin/sh\necho \"version: {build} (deadbeef)\" >&2\n")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let archive = dir.join(format!("llama-{version}-bin-macos-arm64.tar.gz"));
    let ok = Command::new("tar")
        .arg("-czf")
        .arg(&archive)
        .arg("-C")
        .arg(&src)
        .arg(format!("llama-{version}"))
        .status()
        .unwrap();
    assert!(ok.success());
    let digest = format!("sha256:{:x}", Sha256::digest(std::fs::read(&archive).unwrap()));
    (archive, digest)
}

#[test]
fn a_release_archive_is_staged_and_a_llama_cpp_lane_runs_from_it() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.yaml");
    std::fs::write(&config, CONFIG).unwrap();
    let (port, running) = fake_swap(config.clone());
    let d = spawn(&dir.path().join("agent"), port);
    let cfg_s = config.to_str().unwrap();
    let (_, text) = cli(&d.socket, &["lanes", "import", "llama-swap", "--config", cfg_s]);
    let digest = token_after(&text, "--confirm ");
    let (ok, text) = cli(&d.socket, &["lanes", "import", "llama-swap", "--config", cfg_s, "--confirm", &digest]);
    assert!(ok && text.contains("records saved"), "{text}");

    let archives = dir.path().join("archives");
    std::fs::create_dir_all(&archives).unwrap();
    let (_, sum) = make_archive(&archives, "b12527");
    let root = dir.path().join("engines/llama.cpp");
    let recipe = serde_json::json!({
        "name": "llama.cpp",
        "root": root,
        "url": format!("{}/llama-{{version}}-bin-macos-arm64.tar.gz", archives.display()),
        "strip_components": 1,
        "binary": "llama-server",
        "version_arg": "--version",
        "version_pattern": "version: {build}"
    });
    let recipe_file = dir.path().join("llamacpp-recipe.json");
    std::fs::write(&recipe_file, recipe.to_string()).unwrap();
    let rf = recipe_file.to_str().unwrap();

    // A wrong checksum is refused; nothing is installed.
    let wrong = format!("sha256:{}", "0".repeat(64));
    let (ok, text) = cli(&d.socket, &["llamacpp", "install", "llama-swap", "b12527", "--recipe", rf, "--checksum", &wrong]);
    assert!(!ok && text.contains("checksum mismatch"), "{text}");

    // The right one: a versioned dir, the binary reports the build.
    let (ok, text) = cli(&d.socket, &["llamacpp", "install", "llama-swap", "b12527", "--recipe", rf, "--checksum", &sum]);
    assert!(ok, "{text}");
    assert!(text.contains(&format!("archive: {sum}")), "{text}");
    assert!(text.contains("version: 12527"), "{text}");
    assert!(text.contains("the lanes run now: (macro not set)"), "{text}");
    let exe = token_after(&text, "binary: ");
    assert!(exe.starts_with(root.to_str().unwrap()) && exe.ends_with("/llama-server"), "{exe}");
    assert!(Path::new(&exe).exists());

    // The switch is a lane macro write (re-pins the lanes it moves).
    let (ok, text) = cli(&d.socket, &["lanes", "macro", "llama-swap", "llamacpp", &exe]);
    assert!(ok, "{text}");
    let plan = token_after(&text, "--confirm ");
    let (ok, text) = cli(&d.socket, &["lanes", "macro", "llama-swap", "llamacpp", &exe, "--confirm", &plan, "--wait"]);
    assert!(ok, "{text}");

    // A restage of the version the lanes run from is refused.
    let (ok, text) = cli(&d.socket, &["llamacpp", "install", "llama-swap", "b12527", "--checksum", &sum]);
    assert!(!ok && text.contains("restaging would delete the live install"), "{text}");

    // A llama.cpp lane from a model record: one llama-server, one model.
    let gguf = dir.path().join("nomic-embed-text-v1.5.Q8_0.gguf");
    std::fs::write(&gguf, b"GGUF").unwrap();
    let lane = ["llamacpp", "lane", "llama-swap", "emb", "--model", gguf.to_str().unwrap(), "--arg", "embedding", "--ttl", "600"];
    let (ok, text) = cli(&d.socket, &lane);
    assert!(ok, "{text}");
    assert!(text.contains("cmd: ${llamacpp} -m ") && text.contains("--embedding"), "{text}");
    assert!(text.contains("add lane emb"), "{text}");
    let plan = token_after(&text, "--confirm ");
    let mut args = lane.to_vec();
    args.extend(["--confirm", &plan, "--wait"]);
    let (ok, text) = cli(&d.socket, &args);
    assert!(ok, "{text}");
    let now = std::fs::read_to_string(&config).unwrap();
    assert!(now.contains("emb:") && now.contains("# mac lanes"), "{now}");
    let (ok, text) = cli(&d.socket, &["lanes", "show", "llama-swap"]);
    assert!(ok, "{text}");
    let emb = text.lines().find(|l| l.trim_start().starts_with("emb")).unwrap_or_default();
    assert!(emb.contains("llama_cpp"), "{text}");

    // llama-swap loads it like any lane.
    let (ok, text) = cli(&d.socket, &["lanes", "swap", "llama-swap", "emb", "--wait"]);
    assert!(ok, "{text}");
    assert!(running.lock().unwrap().contains(&"emb".to_string()));

    // A second release: staged beside the first, which stays for rollback.
    let (_, sum2) = make_archive(&archives, "b12530");
    let (ok, text) = cli(&d.socket, &["llamacpp", "install", "llama-swap", "b12530", "--checksum", &sum2]);
    assert!(ok, "{text}");
    assert!(text.contains(&format!("the lanes run now: {exe}")), "{text}");
    assert!(text.contains(&format!("back, after switching: llmctl-next lanes macro llama-swap llamacpp {exe}")), "{text}");
    assert!(Path::new(&exe).exists(), "the running version is kept");

    // Remove the lane again.
    let (ok, text) = cli(&d.socket, &["llamacpp", "lane", "llama-swap", "emb", "--remove"]);
    assert!(ok && text.contains("remove lane emb"), "{text}");
    let plan = token_after(&text, "--confirm ");
    let (ok, text) =
        cli(&d.socket, &["llamacpp", "lane", "llama-swap", "emb", "--remove", "--confirm", &plan, "--wait"]);
    assert!(ok, "{text}");
    assert!(!std::fs::read_to_string(&config).unwrap().contains("emb:"));
}
