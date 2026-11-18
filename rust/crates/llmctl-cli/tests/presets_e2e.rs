//! M5c item 1 end to end: presets read and written in place by the agent,
//! `run <preset>` through the ordinary profile swap, drift refused until
//! synced, comment-only differences never restart, and the version
//! comments settled to the pin.

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

const ALPHA: &str = "# alpha: the house preset\n\
alias: alpha\n\
served_name: fake/alpha  # what clients ask for\n\
model_id: /models/alpha\n\
quantization: fp8\n\
vllm_quantization_flag: fp8\n\
tensor_parallel_size: 1\n\
max_model_len: 4096\n\
# why four\n\
max_num_seqs: 4\n\
kv_cache_dtype: fp8\n\
host: 127.0.0.1\n\
reasoning_parser: qwen3\n\
spec_config:\n  method: mtp\n  num_speculative_tokens: 3\n";

fn sha(bytes: &[u8]) -> String {
    use sha2::Digest;
    format!("sha256:{:x}", sha2::Sha256::digest(bytes))
}

fn start() -> Agent {
    use llmctl_core::launch::{EnvFileRef, LaunchDefinition};
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let services = root.join("services");
    let presets = root.join("presets");
    std::fs::create_dir_all(&services).unwrap();
    std::fs::create_dir_all(&presets).unwrap();
    std::fs::write(services.join("coder.env"), "MODEL=base\n").unwrap();
    std::fs::write(presets.join("alpha.yaml"), ALPHA).unwrap();
    std::fs::write(presets.join("_shortcuts.yaml"), "not: a preset\n").unwrap();
    std::fs::write(presets.join("broken.yaml"), "alias: Broken\n").unwrap();
    std::fs::write(
        root.join("router.yaml"),
        "# llm-router\nbackends:\n  # input. Runs from the vllm-laguna interpreter (see the profile), not\n  - name: other\n    served_model: someone-else\n",
    )
    .unwrap();
    let definition = LaunchDefinition {
        environment_files: vec![EnvFileRef {
            path: services.join("coder.env").display().to_string(),
            optional: false,
            content_digest: sha(b"MODEL=base\n"),
        }],
        ..Default::default()
    };
    let host = "workstation";
    let inv = serde_json::json!({"inventory": {
        "host": host, "platform": "linux",
        "gpus": [{"host": host, "index": 0}],
        "workloads": [{
            "name": "coder", "engine": "fake", "engine_kind": "vllm", "model": "fake/coder",
            "gpus": [{"host": host, "index": 0}], "port": 8100,
            "unit": {"scope": "user", "name": "llmctl-next-fake-coder.service"},
            "authority": {"kind": "systemd_unit", "scope": "user", "unit": "llmctl-next-fake-coder.service"},
            "ownership": {
                "kind": "managed", "definition": definition, "digest": definition.digest(),
                "adopted_at": "2026-11-17T00:00:00Z", "provenance": [],
                "version": {
                    "interpreter": "/opt/envs/vllm-muse/bin/python",
                    "packages": [["torch", "2.13.0"], ["vllm", "0.30.0"]],
                    "cuda": "13.0", "lock_digest": "sha256:x", "exec_digests": []
                }
            }
        }],
        "revision": 1
    }});
    std::fs::write(root.join("inventory.json"), inv.to_string()).unwrap();
    let socket = root.join("control.sock");
    let child = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&socket)
        .arg("--data-dir")
        .arg(&root)
        .arg("daemon")
        .env("LLMCTL_NEXT_PRESET_DIRS", &presets)
        .env("LLMCTL_NEXT_SETTINGS", root.join("no-settings.yaml"))
        .env("LLMCTL_ROUTER_CONFIG", root.join("router.yaml"))
        .env("HF_HOME", "/hf")
        .env("LLMCTL_CUDA_ROOT", "/cuda")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn daemon");
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    while std::os::unix::net::UnixStream::connect(&socket).is_err() {
        assert!(std::time::Instant::now() < deadline, "daemon did not come up");
        std::thread::sleep(Duration::from_millis(50));
    }
    Agent { child, _dir: dir, root, socket }
}

fn cli(a: &Agent, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_llmctl-next"))
        .arg("--socket")
        .arg(&a.socket)
        .args(args)
        .output()
        .expect("run llmctl-next");
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

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn presets_list_run_sync_edit_new_and_settle() {
    let a = start();
    let services = a.root.join("services");
    let profile = services.join("coder.env.alpha");

    // List: alpha links to coder, no profile yet; broken is reported.
    let list = json(&a, &["preset", "list", "--json"]);
    let alpha = &list["presets"][0];
    assert_eq!(alpha["alias"], "alpha");
    assert_eq!(alpha["link"]["workload"], "coder");
    assert_eq!(alpha["link"]["found_by"], "new");
    assert!(alpha["link"]["profile_env"].is_null());
    assert_eq!(list["presets"].as_array().unwrap().len(), 1, "{list}");
    assert!(list["invalid"][0]["path"].as_str().unwrap().ends_with("broken.yaml"), "{list}");

    // The render: pinned interpreter as the Python root, workload port,
    // reasoning parser folded into VLLM_EXTRA, spec config in file order.
    let show = json(&a, &["preset", "show", "alpha", "--json"]);
    let env = show["rendered_env"].as_str().unwrap();
    for line in [
        "CUDA_VISIBLE_DEVICES=0\n",
        "LD_LIBRARY_PATH=/opt/envs/vllm-muse/lib:/cuda/lib64\n",
        "HF_HOME=/hf\n",
        "VLLM_SERVED_NAME=fake/alpha\n",
        "VLLM_PORT=8100\n",
        "VLLM_GPU_MEM=0.85\n",
        "VLLM_MAX_SEQS=4\n",
        "VLLM_SPEC_CONFIG={\"method\":\"mtp\",\"num_speculative_tokens\":3}\n",
        "VLLM_EXTRA=--reasoning-parser qwen3\n",
    ] {
        assert!(env.contains(line), "missing {line:?} in\n{env}");
    }
    assert!(!env.contains("VLLM_PYTHON="), "no python_root in the preset: {env}");

    // 1. No profile carries alpha yet: run refuses (drift), --sync writes it.
    let (code, _, err) = cli(&a, &["run", "alpha"]);
    assert_eq!(code, 4, "{err}");
    assert!(err.contains("--sync"), "{err}");
    assert!(!profile.exists());
    let (code, out, err) = cli(&a, &["run", "alpha", "--sync", "--wait"]);
    assert_eq!(code, 0, "{out}{err}");
    let text = read(&profile);
    assert!(text.starts_with("# Generated from llmctl preset alpha by "), "{text}");
    assert_eq!(read(&services.join("coder.env")), text, "the profile is the active env now");
    let list = json(&a, &["preset", "list", "--json"]);
    assert_eq!(list["presets"][0]["link"]["active"], true, "{list}");
    assert_eq!(list["presets"][0]["link"]["found_by"], "header");

    // 2. Edit in place: comments stay, the value changes, a backup is kept.
    let (code, out, err) = cli(&a, &["preset", "edit", "alpha", "--set", "max_num_seqs=8"]);
    assert_eq!(code, 0, "{out}{err}");
    let yaml = read(&a.root.join("presets/alpha.yaml"));
    assert_eq!(yaml, ALPHA.replace("max_num_seqs: 4", "max_num_seqs: 8"));
    assert_eq!(read(&a.root.join("presets/alpha.yaml.llmctl-previous")), ALPHA);
    let (code, out, _) = cli(&a, &["preset", "diff", "alpha"]);
    assert_eq!(code, 3, "{out}");
    assert!(out.contains("VLLM_MAX_SEQS: file 4 | preset 8"), "{out}");
    //    An invalid edit never reaches the file.
    let (code, _, err) = cli(&a, &["preset", "edit", "alpha", "--set", "gpu_memory_utilization=1.5"]);
    assert_eq!(code, 6, "{err}");
    assert_eq!(read(&a.root.join("presets/alpha.yaml")), yaml);
    let (code, _, err) = cli(&a, &["preset", "edit", "alpha", "--set", "alias=other"]);
    assert_ne!(code, 0, "{err}");

    // 3. A drifted profile is refused; sync writes the value, keeps the
    //    comments, and the next run switches to it.
    let mut with_comment = read(&profile);
    with_comment.push_str("# a note the owner wrote\n");
    std::fs::write(&profile, &with_comment).unwrap();
    let (code, _, err) = cli(&a, &["run", "alpha"]);
    assert_eq!(code, 4, "{err}");
    let (code, out, err) = cli(&a, &["preset", "sync", "alpha"]);
    assert_eq!(code, 0, "{out}{err}");
    let synced = read(&profile);
    assert!(synced.contains("VLLM_MAX_SEQS=8\n") && synced.contains("# a note the owner wrote\n"), "{synced}");
    let (code, out, err) = cli(&a, &["run", "alpha", "--dry-run"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("apply profile coder.env.alpha"), "{out}");
    let (code, out, err) = cli(&a, &["run", "alpha", "--wait"]);
    assert_eq!(code, 0, "{out}{err}");
    assert_eq!(read(&services.join("coder.env")), synced);

    // 4. A comment-only difference is no switch (and so no restart).
    std::fs::write(&profile, format!("{synced}# another note\n")).unwrap();
    let (code, out, err) = cli(&a, &["run", "alpha", "--dry-run"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("already renders it"), "{out}");
    assert!(!out.contains("apply profile"), "{out}");

    // 5. New from alpha: comments copied, served name changed.
    let (code, out, err) = cli(&a, &["preset", "new", "beta", "--from", "alpha", "--set", "served_name=fake/beta"]);
    assert_eq!(code, 0, "{out}{err}");
    let beta = read(&a.root.join("presets/beta.yaml"));
    assert!(beta.starts_with("# alpha: the house preset\nalias: beta\n"), "{beta}");
    assert!(beta.contains("served_name: fake/beta  # what clients ask for\n"), "{beta}");
    let (code, _, err) = cli(&a, &["preset", "new", "beta", "--from", "alpha"]);
    assert_eq!(code, 4, "exists: {err}");
    let (code, _, err) = cli(&a, &["preset", "new", "gamma", "--set", "alias=gamma"]);
    assert_eq!(code, 6, "missing required keys: {err}");
    let show = json(&a, &["preset", "show", "beta", "--json"]);
    assert!(show["rendered_env"].as_str().unwrap().contains("VLLM_SERVED_NAME=fake/beta\n"));

    // 6. Settle: the version comments follow the pin.
    std::fs::write(
        &profile,
        format!("{}# The interpreter is the vllm-laguna env (vLLM 0.25.1, torch 2.10) since x\n", read(&profile)),
    )
    .unwrap();
    let preview = json(&a, &["preset", "settle", "--json"]);
    assert_eq!(preview["pin"]["env_name"], "vllm-muse");
    let files: Vec<&str> = preview["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert_eq!(files.len(), 2, "{preview}");
    assert!(read(&a.root.join("router.yaml")).contains("vllm-laguna interpreter"), "a preview writes nothing");
    let written = json(&a, &["preset", "settle", "--write", "--json"]);
    assert!(written["files"].as_array().unwrap().iter().all(|f| f["written"] == true));
    assert!(read(&a.root.join("router.yaml")).contains("Runs from the vllm-muse interpreter"));
    assert!(read(&profile).contains("The interpreter is the vllm-muse env (vLLM 0.30.0, torch 2.10)"));
    let again = json(&a, &["preset", "settle", "--json"]);
    assert!(again["files"].as_array().unwrap().is_empty(), "{again}");

    // A workload name still runs the workload.
    let (code, out, err) = cli(&a, &["run", "coder", "--dry-run"]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(!out.contains("preset"), "{out}");
}
