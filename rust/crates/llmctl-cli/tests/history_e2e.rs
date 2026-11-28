//! M6b item 7 end to end: `history import` of a Python DB copy into a temp
//! agent (report, token, idempotent re-import, a conflict kept as it was,
//! another host's row reported) and `history list` without secrets.

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

fn start() -> Agent {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let host = "workstation";
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux", "gpus": [{"host": host, "index": 0}],
        "workloads": [], "revision": 1
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
    (out.status.code().unwrap_or(-1), String::from_utf8_lossy(&out.stdout).into(), String::from_utf8_lossy(&out.stderr).into())
}

fn json(a: &Agent, args: &[&str]) -> serde_json::Value {
    let (code, out, err) = cli(a, args);
    assert_eq!(code, 0, "{args:?}: {out}{err}");
    serde_json::from_str(&out).unwrap_or_else(|e| panic!("{args:?}: {e}: {out}"))
}

/// A Python-shaped DB: two sessions, an operation of this host, one of a Mac.
fn snapshot(path: &Path, phase: &str) {
    let _ = std::fs::remove_file(path);
    let c = rusqlite::Connection::open(path).unwrap();
    c.execute_batch(
        "CREATE TABLE sessions(id VARCHAR, runtime VARCHAR, status VARCHAR, env JSON, created_at DATETIME);
         CREATE TABLE operations(id VARCHAR, kind VARCHAR, target VARCHAR, phase VARCHAR, owner JSON,
                                 request JSON, created_at DATETIME);
         CREATE TABLE events(id INTEGER, message VARCHAR);",
    )
    .unwrap();
    c.execute("INSERT INTO sessions VALUES('s1','VLLM','RUNNING','{\"HF_TOKEN\":\"secret\"}','2026-06-01 10:36:46')", []).unwrap();
    c.execute("INSERT INTO sessions VALUES('s2','LLAMA_CPP','STOPPED',NULL,'2026-09-27 06:49:46')", []).unwrap();
    c.execute(
        "INSERT INTO operations VALUES('o1','swap','qwen3.8-27b',?1,'{\"pid\":1,\"hostname\":\"yannik-desktop\"}','{\"preset\":\"qwen27b\"}','2026-11-25 23:57:27')",
        [phase],
    )
    .unwrap();
    c.execute(
        "INSERT INTO operations VALUES('o2','swap','x','completed','{\"pid\":2,\"hostname\":\"macbookpro.lan\"}','{}','2026-11-20 20:46:18')",
        [],
    )
    .unwrap();
    c.execute("INSERT INTO events VALUES(1,'not imported')", []).unwrap();
}

#[test]
fn a_python_db_copy_is_imported_read_only_and_only_with_its_token() {
    let a = start();
    let snap = a.root.join("py-copy.sqlite3");
    snapshot(&snap, "completed");
    let s = snap.to_str().unwrap();

    // A live DB (a WAL beside it) is refused.
    std::fs::write(a.root.join("py-copy.sqlite3-wal"), "").unwrap();
    let (code, _, err) = cli(&a, &["history", "import", s, "--as", "yannik-desktop"]);
    assert_ne!(code, 0);
    assert!(err.contains("live DB"), "{err}");
    std::fs::remove_file(a.root.join("py-copy.sqlite3-wal")).unwrap();

    let report = json(&a, &["history", "import", s, "--as", "yannik-desktop", "--json"]);
    assert_eq!(report["applied"], false);
    assert_eq!(report["counts"], serde_json::json!({"new": 3, "other_host": 1}), "{report:#}");
    let token = report["confirm"].as_str().unwrap().to_string();
    let (code, _, err) = cli(&a, &["history", "import", s, "--as", "yannik-desktop", "--confirm", "hist-0"]);
    assert_eq!(code, 4, "{err}");
    let done = json(&a, &["history", "import", s, "--as", "yannik-desktop", "--confirm", &token, "--json"]);
    assert_eq!((done["applied"].as_bool(), done["total"].as_u64()), (Some(true), Some(3)));

    let list = json(&a, &["history", "list", "--json"]);
    let ids: Vec<&str> = list["entries"].as_array().unwrap().iter().map(|e| e["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["py:yannik-desktop:o1", "py:yannik-desktop:s1", "py:yannik-desktop:s2"]);
    let text = list.to_string();
    assert!(!text.contains("secret") && !text.contains("preset"), "no env, no request body: {text}");

    // An imported row authorizes nothing: it is no operation of the agent.
    let ops = json(&a, &["ops", "--json"]);
    let ops = ops.get("operations").unwrap_or(&ops);
    assert_eq!(ops.as_array().map(Vec::len), Some(0), "{ops:#}");

    // Again: nothing new. Changed at the source: reported, the first import kept.
    let again = json(&a, &["history", "import", s, "--as", "yannik-desktop", "--json"]);
    assert_eq!(again["counts"], serde_json::json!({"present": 3, "other_host": 1}));
    snapshot(&snap, "failed");
    let changed = json(&a, &["history", "import", s, "--as", "yannik-desktop", "--json"]);
    assert_eq!(changed["counts"]["conflicting"], 1, "{changed:#}");
    let t = changed["confirm"].as_str().unwrap().to_string();
    json(&a, &["history", "import", s, "--as", "yannik-desktop", "--confirm", &t, "--json"]);
    let o1 = json(&a, &["history", "list", "--json"])["entries"][0]["summary"]["phase"].clone();
    assert_eq!(o1, "completed", "the first import is kept");
}

/// A `.backup` copy of a WAL-mode DB is itself in WAL mode. Reading it must not
/// create `-wal`/`-shm` beside it: the `--confirm` call would refuse it as a live DB.
#[test]
fn a_wal_mode_backup_copy_is_imported_and_nothing_is_written_beside_it() {
    let a = start();
    let live = a.root.join("live.sqlite3");
    snapshot(&live, "completed");
    let src = rusqlite::Connection::open(&live).unwrap();
    let mode: String = src.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0)).unwrap();
    assert_eq!(mode, "wal");
    src.execute("INSERT INTO sessions VALUES('s3','VLLM','STOPPED',NULL,'2026-11-01 00:00:00')", []).unwrap();
    let copy = a.root.join("wal copy 100%?#.sqlite3"); // a path a URI must escape
    {
        let mut dst = rusqlite::Connection::open(&copy).unwrap();
        let backup = rusqlite::backup::Backup::new(&src, &mut dst).unwrap();
        backup.run_to_completion(64, Duration::ZERO, None).unwrap();
    }
    let beside = |suffix: &str| PathBuf::from(format!("{}{suffix}", copy.display()));
    let _ = (std::fs::remove_file(beside("-wal")), std::fs::remove_file(beside("-shm")));
    assert_eq!(std::fs::read(&copy).unwrap()[18..20], [2, 2], "the copy is in WAL mode");
    assert!(a.root.join("live.sqlite3-wal").exists(), "the source is a live WAL-mode DB");
    let s = copy.to_str().unwrap();

    let report = json(&a, &["history", "import", s, "--as", "yannik-desktop", "--json"]);
    assert_eq!(report["counts"], serde_json::json!({"new": 4, "other_host": 1}), "{report:#}");
    assert!(!beside("-wal").exists() && !beside("-shm").exists(), "the dry run wrote beside the copy");
    let token = report["confirm"].as_str().unwrap().to_string();
    let done = json(&a, &["history", "import", s, "--as", "yannik-desktop", "--confirm", &token, "--json"]);
    assert_eq!((done["applied"].as_bool(), done["total"].as_u64()), (Some(true), Some(4)));
    assert!(!beside("-wal").exists() && !beside("-shm").exists(), "--confirm wrote beside the copy");
}
