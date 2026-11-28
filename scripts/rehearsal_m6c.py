#!/usr/bin/env python3
# ruff: noqa: E501 - embedded stand-in programs and check names read better unwrapped
"""M6c rehearsal: the cutover and both reversals, inside a Debian 12 container.

Run it with scripts/rehearsal_m6c.sh, never on a host: it writes the real
default paths of the user it runs as (settings.yaml, the Python DB, the
agent data dir, systemd user units). Procedure and results:
docs/RUST-M6-REHEARSAL.md.

Expects (the driver provides them): /src/llmctl, /src/AI, /src/llm-router,
/src/local-llm, /src/fixtures/{llmctl.sqlite3,presets/}, llmctl-next and
the Python llmctl and llm-router on PATH, a systemd user manager.

Exit 0 = every check passed.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

H = Path.home()
SRC = Path("/src")
CONF = H / ".config/llmctl"
SETTINGS = CONF / "settings.yaml"
DB = H / ".local/share/llmctl/llmctl.sqlite3"
LOCK = H / ".local/share/llmctl/operations.lock"
DATA = H / ".local/state/llmctl-next"
UNITS = H / ".config/systemd/user"
SERVICES = H / "services"  # profile envs and the active env of the engine
WORK = H / "m6c"  # snapshots, bundles, router/heal config, stand-in programs
BACKUPS = DATA / "deploy-backups/rehearsal-m6-cutover"
ROUTER_YAML = WORK / "router.yaml"
HEAL_YAML = WORK / "heal.yaml"
MAC_CONF = WORK / "mac-config.yaml"
DS4_FLAG = H / ".local/state/llmctl/ds4-llmctl-sync.off"
ENGINE, LANE = "vllm-engine", "reh-lane"  # unit names; workloads reh-engine, reh-lane
PORT, LANE_PORT, SWAP_PORT = 18001, 18002, 18099
NEXT = "/usr/local/bin/llmctl-next"
TERMINAL = {"succeeded", "failed", "cancelled", "needs_attention", "interrupted"}

SERVER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, model = int(os.environ["VLLM_PORT"]), os.environ["VLLM_SERVED_NAME"]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        ok = self.path in ("/v1/models", "/health")
        body = json.dumps({"object": "list", "data": [{"id": model}]}).encode() if ok else b""
        self.send_response(200 if ok else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""
# A stand-in llama-swap for the Mac llmctl drill: health, catalog, nothing running.
SWAP = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = {"/health": b"OK", "/v1/models": json.dumps({"data": [{"id": sys.argv[2]}]}).encode(),
                "/running": json.dumps({"running": []}).encode()}.get(self.path.split("?")[0], b"[]")
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""
PRESET = """alias: {alias}
served_name: {served}
model_id: {model_dir}
quantization: fp8
vllm_quantization_flag: fp8
tensor_parallel_size: 1
max_model_len: 4096
host: 127.0.0.1
python_root: /usr
"""
results = []


class Stop(Exception):
    pass


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}", flush=True)
    return ok


def need(name, ok, detail=""):
    if not check(name, ok, detail):
        raise Stop(name)


def sh(argv, env=None, timeout=600, stdin=None):
    p = subprocess.run(
        [str(a) for a in argv],
        capture_output=True,
        text=True,
        env={**ENV, **(env or {})},
        timeout=timeout,
        input=stdin,
    )
    return p.returncode, p.stdout, p.stderr


def tail(out, err, n=140):
    return " ".join((out + err).split())[-n:]


ENV = {
    k: v
    for k, v in os.environ.items()
    if not k.startswith(("LLMCTL_NEXT", "LLMCTL_CONFIG", "LLMCTL_API"))
}
ENV.update(
    {
        "LLMCTL_PLUGINS": "none",
        "NO_COLOR": "1",
        "TERM": "dumb",
        "COLUMNS": "200",
        "LLMCTL_LMS_BIN": "/bin/false",  # never an LM Studio CLI
        "LLM_UNIT_NEXT_PY": "/opt/venv/bin/python",
        "PATH": f"{WORK}/bin:/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
    }
)


def agent(*args, timeout=600, env=None):
    return sh([NEXT, "--data-dir", DATA, *args], env=env, timeout=timeout)


def agent_json(*args):
    rc, out, err = agent(*args)
    try:
        return rc, json.loads(out)
    except ValueError:
        return rc, {"_raw": tail(out, err, 300)}


def py(*args, env=None):
    return sh(["llmctl", *args], env=env)


def systemctl(*args):
    return sh(["systemctl", "--user", *args])


def unit_state(unit):
    _, out, _ = systemctl("show", unit, "-p", "ActiveState", "-p", "MainPID")
    kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return kv.get("ActiveState"), int(kv.get("MainPID") or 0)


def served(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
            return [m["id"] for m in json.load(r)["data"]]
    except (OSError, ValueError, KeyError):
        return None


def wait_for(fn, seconds=30):
    end = time.time() + seconds
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.3)
    return fn()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ops():
    _, v = agent_json("ops", "--json")
    return v if isinstance(v, list) else v.get("operations", [])


def open_ops():
    return [o for o in ops() if str(o.get("state", o.get("status", ""))).lower() not in TERMINAL]


def audit_findings():
    rc, v = agent_json("audit", "--unexplained", "--json")
    rows = v if isinstance(v, list) else v.get("findings", [])
    return rc, rows


def signature():
    """(rc, digest) of serving_signature.py over the agent's status."""
    _, status, _ = agent("status", "--json")
    rc, out, _ = sh([sys.executable, SRC / "llmctl/scripts/serving_signature.py"], stdin=status)
    try:
        return rc, json.loads(out).get("digest", "")
    except ValueError:
        return 2, ""


def pids():
    return {u: unit_state(u)[1] for u in (ENGINE, LANE)}


# ---------------------------------------------------------------- settings
def write_settings(mode):
    """mode: cutover | delegate | off (R2)."""
    head = {"cutover": "  mode: cutover\n  enabled: true\n", "delegate": "  enabled: true\n"}
    SETTINGS.write_text(
        "# Written by the M6c rehearsal. No token.\n"
        "managed_units:\n  vllm_engine:\n    enabled: true\n"
        f"    unit_name: {ENGINE}\n    default_port: {PORT}\n"
        f"    env_file_path: {SERVICES / 'engine.env'}\n    launcher_marker: null\n    user_scope: true\n"
        "llmctl_next:\n"
        + head.get(mode, "  enabled: false\n")
        + f"  cmd: [{NEXT}, --data-dir, {DATA}]\n"
        f"  units:\n    {ENGINE}.service: reh-engine\n    {LANE}.service: reh-lane\n"
    )


def write_router(only, refusal_final):
    ROUTER_YAML.write_text(
        "# Written by the M6c rehearsal.\n"
        "switch:\n"
        f"  services_dir: {SERVICES}\n  active_env: engine.env\n"
        f"  restart_cmd: [systemctl, --user, restart, {ENGINE}]\n"
        f"  op_lock_file: {LOCK}\n"
        f"  llmctl_next_cmd: [{NEXT}, --data-dir, {DATA}]\n"
        "  llmctl_next_workload: reh-engine\n"
        f"  llmctl_next_only: {'true' if only else 'false'}\n"
        f"  llmctl_next_refusal_final: {'true' if refusal_final else 'false'}\n"
        "routing:\n  default_backend: reh-a\n"
        "backends:\n"
        + "".join(
            f"  - name: reh-{x}\n    served_model: reh/model-{x}\n"
            f"    profile_env: engine.env.reh-{x}\n    upstream: http://127.0.0.1:{PORT}\n"
            for x in "ab"
        )
    )


def write_heal(report_only):
    HEAL_YAML.write_text(
        "# Written by the M6c rehearsal.\n"
        f"mode: act\nstate_dir: {WORK / 'heal-state'}\nrouter_yaml: {ROUTER_YAML}\n"
        "llmctl: /opt/venv/bin/llmctl\ndefault_model: reh/model-a\n"
        "consecutive: 1\nsettle_s: 0\nload_margin_s: 0\ngen_timeout_s: 5\nfront_door_grace_s: 0\n"
        "limits:\n  max_actions: 5\n  window_s: 600\n"
        f"op_lock_file: {LOCK}\n"
        f"llmctl_next_cmd: [{NEXT}, --data-dir, {DATA}]\n"
        f"llmctl_next_report_only: {'true' if report_only else 'false'}\n"
        "gpu_busy_mib: 8192\nbusy_process_patterns: []\nnotify:\n  enabled: false\n"
        f"extra_lanes:\n  - {{unit: {LANE}, scope: user, upstream: 'http://127.0.0.1:{LANE_PORT}', "
        "runtime: vllm, engine: false, load_timeout_s: 30}\n"  # engine: false = a plain restart (no GPU policy)
        "front_doors: []\nmac_repair:\n  enabled: false\n"
    )


def heal_run():
    rc, out, err = sh(
        [
            "/opt/venv/bin/python",
            H / "AI/heal/llm-heal.py",
            "--config",
            HEAL_YAML,
            "run",
            "--json",
        ],
        env={"LLM_UNIT_BIN": "none"},
    )
    return rc, out, err


def heal_held(unit):
    """llm-heal's recorded reason for not running a repair (state.json)."""
    try:
        return str(
            json.loads((WORK / "heal-state/state.json").read_text())["targets"][unit].get("held")
            or ""
        )
    except (OSError, ValueError, KeyError):
        return ""


def router_switch(backend):
    return sh(
        ["/opt/venv/bin/python", "-m", "llm_router.cli", "--config", ROUTER_YAML, "switch", backend]
    )


def lane_unit(verb, unit):
    return sh(
        ["bash", "-c", f'. {H}/AI/services/lanes/_unit.sh && lane_unit "$0" "$1"', verb, unit]
    )


def write_unit(unit, env_file, desc):
    (UNITS / f"{unit}.service").write_text(
        f"[Unit]\nDescription={desc}\n\n[Service]\nType=simple\nEnvironmentFile={env_file}\n"
        f"ExecStart=/usr/local/bin/python3 {WORK / 'server.py'}\n"
    )


def legacy_writers(enabled):
    """The stand-in llmctl-reconcile.timer and llmctl-api.service."""
    for u in ("llmctl-reconcile.timer", "llmctl-api.service"):
        systemctl("enable" if enabled else "disable", "--now", u)


def legacy_writer_state():
    return [unit_state(u)[0] for u in ("llmctl-reconcile.timer", "llmctl-api.service")]


# ---------------------------------------------------------------- setup
def setup(as_host):
    for d in (
        CONF / "presets",
        DB.parent,
        DATA,
        UNITS,
        SERVICES,
        WORK / "bin",
        BACKUPS,
        DS4_FLAG.parent,
    ):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC / "AI", H / "AI", symlinks=True, dirs_exist_ok=True)
    shutil.copy2(SRC / "fixtures/llmctl.sqlite3", DB)
    for p in (SRC / "fixtures/presets").glob("*.yaml"):
        shutil.copy2(p, CONF / "presets" / p.name)
    (WORK / "server.py").write_text(SERVER)
    (WORK / "swap.py").write_text(SWAP)
    for alias in "ab":
        (WORK / f"models/{alias}").mkdir(parents=True, exist_ok=True)
        (WORK / f"models/{alias}/config.json").write_text("{}\n")  # llm-router's model guard
        (CONF / f"presets/reh-{alias}.yaml").write_text(
            PRESET.format(
                alias=f"reh-{alias}",
                served=f"reh/model-{alias}",
                model_dir=WORK / f"models/{alias}",
            )
        )
    # Stand-in programs a container lacks. They record each call.
    for name, body in {
        "nvidia-smi": "exit 0",
        "launchctl": f'echo "$*" >> {WORK}/launchctl.calls; exit 0',
        "journalctl": "exit 0",
        "tailscale": "echo '{}'; exit 0",
        "sudo": f'echo "$*" >> {WORK}/sudo.calls; exit 1',
    }.items():
        f = WORK / "bin" / name
        f.write_text(f"#!/bin/sh\n{body}\n")
        f.chmod(0o755)
    # The engine and a lane, as on the workstation: user units, stand-in engines.
    (SERVICES / "engine.env").write_text(f"VLLM_PORT={PORT}\nVLLM_SERVED_NAME=reh/model-a\n")
    (SERVICES / "lane.env").write_text(f"VLLM_PORT={LANE_PORT}\nVLLM_SERVED_NAME=reh/lane\n")
    write_unit(ENGINE, SERVICES / "engine.env", "M6c stand-in vLLM engine")
    write_unit(LANE, SERVICES / "lane.env", "M6c stand-in lane")
    # The Python-side writers the quiesce stops (stand-ins: they change nothing).
    (UNITS / "llmctl-api.service").write_text(
        "[Service]\nExecStart=/bin/sleep infinity\n[Install]\nWantedBy=default.target\n"
    )
    (UNITS / "llmctl-reconcile.service").write_text(
        "[Service]\nType=oneshot\nExecStart=/bin/true\n"
    )
    (UNITS / "llmctl-reconcile.timer").write_text(
        "[Timer]\nOnUnitActiveSec=1h\nOnBootSec=1h\n[Install]\nWantedBy=timers.target\n"
    )
    (UNITS / "llmctl-next.service").write_text(
        "[Unit]\nDescription=llmctl-next agent (M6c rehearsal)\n\n[Service]\n"
        f"ExecStart={NEXT} --data-dir {DATA} daemon --host-id workstation\n"
        "Environment=LLMCTL_NEXT_LANE_LEASE=off LLMCTL_NEXT_ROUTER=off LLMCTL_LMS_BIN=/bin/false\n"
        f"Environment=LLMCTL_NEXT_SETTINGS={SETTINGS} LLMCTL_ROUTER_CONFIG={ROUTER_YAML}\n"
        f"Environment=PATH={WORK}/bin:/usr/local/bin:/usr/bin:/bin\n"
    )
    (DATA / "inventory.json").write_text(
        json.dumps(
            {
                "inventory": {
                    "host": "workstation",
                    "platform": "linux",
                    "gpus": [
                        {"host": "workstation", "index": 0},
                        {"host": "workstation", "index": 1},
                        {"host": "workstation", "index": 2},
                    ],
                    "workloads": [],
                    "revision": 1,
                }
            }
        )
    )
    MAC_CONF.write_text(
        f"router: http://127.0.0.1:{SWAP_PORT}\nnext_enabled: 1\nnext_only: 1\n"
        f"next_bin: {NEXT}\nnext_data: {DATA}\nnext_workload: reh-swap\nlogdir: {WORK}/mac-logs\n"
        f"swap_config: {WORK}/swap-config.yaml\n"
    )
    write_settings("delegate")
    write_router(only=False, refusal_final=True)
    write_heal(report_only=False)
    systemctl("daemon-reload")
    legacy_writers(True)
    for u in (ENGINE, LANE, "llmctl-next"):
        systemctl("start", u)


# ---------------------------------------------------------------- the run
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-host", required=True, help="this host's name in the Python DB")
    a = ap.parse_args()
    if not Path("/.dockerenv").exists() or H != Path("/home/tester"):
        sys.exit("refused: run this only inside the rehearsal container (scripts/rehearsal_m6c.sh)")

    setup(a.as_host)
    try:
        print("== M5f end state: the delegate state with two enrolled units", flush=True)
        need("S1 the agent answers", wait_for(lambda: agent("status", "--json")[0] == 0, 30))
        need(
            "S2 the engine serves reh/model-a, the lane serves",
            wait_for(lambda: served(PORT) == ["reh/model-a"] and served(LANE_PORT) == ["reh/lane"]),
        )
        for unit, wl, port, model in (
            (ENGINE, "reh-engine", PORT, "reh/model-a"),
            (LANE, "reh-lane", LANE_PORT, "reh/lane"),
        ):
            adopt = [
                "adopt",
                f"{unit}.service",
                "--name",
                wl,
                "--model",
                model,
                "--port",
                port,
                "--gpus",
                "0,1" if wl == "reh-engine" else "2",
                "--engine-kind",
                "vllm",
                "--readiness-timeout",
                "60",
            ]
            rc, pre = agent_json(*adopt, "--json")
            bad = [c["name"] for c in pre.get("checks", []) if c.get("status") != "pass"]
            need(
                f"S3 adopt {wl}: every preview check passes",
                rc == 0 and pre.get("digest") and not bad,
                f"not passing: {bad}" if bad else json.dumps(pre)[:200] if rc else "",
            )
            rc, out, err = agent(*adopt, "--confirm", pre["digest"])
            need(f"S4 adopt {wl} confirmed", rc == 0, tail(out, err))
        for alias in ("reh-a", "reh-b"):
            rc, out, err = agent("preset", "sync", alias, "--workload", "reh-engine")
            need(
                f"S5 preset sync {alias}",
                rc == 0 and (SERVICES / f"engine.env.{alias}").exists(),
                tail(out, err),
            )

        # ============================== forward ==============================
        print("\n== forward 1: preflight, protection, quiesce", flush=True)
        rc, sig0 = signature()
        need("F1 signature before agrees with the agent", rc == 0, sig0[:23])
        pid0 = pids()
        lock = open(LOCK, "a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check("F2 the host lock is held for the quiesce and the snapshot", True, str(LOCK))
        legacy_writers(False)
        DS4_FLAG.touch()
        write_heal(report_only=True)
        check(
            "F3 quiesce: llmctl-reconcile.timer and llmctl-api stopped and disabled",
            legacy_writer_state() == ["inactive", "inactive"],
            str(legacy_writer_state()),
        )
        rc, out, err = sh(["bash", H / "AI/services/ds4-llmctl-sync", "adopt"])
        check(
            "F4 ds4-llmctl-sync is a no-op with the flag file",
            rc == 0 and "nothing to do" in out,
            tail(out, err),
        )

        print("\n== forward 2: snapshot and export", flush=True)
        snap = BACKUPS / "llmctl.sqlite3"
        rc, out, err = sh(["sqlite3", DB, f".backup {snap}"])
        integrity = (
            sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
            .execute("pragma integrity_check")
            .fetchone()[0]
        )
        need("F5 Python DB .backup", rc == 0 and integrity == "ok", f"sha256 {sha256(snap)[:16]}")
        rc, out, err = py("export-registry", BACKUPS / "bundle.json")
        need("F6 export-registry bundle", rc == 0, tail(out, err))
        bundle = json.loads((BACKUPS / "bundle.json").read_text())
        files = sorted(
            list((CONF / "presets").glob("*.yaml"))
            + [SETTINGS, ROUTER_YAML]
            + sorted(SERVICES.glob("engine.env*"))
            + [UNITS / f"{ENGINE}.service", UNITS / f"{LANE}.service"]
        )
        sums0 = {str(f): sha256(f) for f in files}
        (BACKUPS / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sums0.items()))
        for f in ("inventory.json", "catalog.json"):
            if (DATA / f).exists():
                shutil.copy2(DATA / f, BACKUPS / f)
        check(
            "F7 sha256 list of presets, settings, router.yaml, profile envs, unit files",
            len(sums0) == len(files),
            f"{len(files)} files",
        )

        print("\n== forward 3: import and verify (lock still held)", flush=True)
        t = time.time()
        rc, dry = agent_json(
            "catalog", "import", BACKUPS / "bundle.json", "--as", a.as_host, "--json"
        )
        check(
            "F8 catalog import dry run while the lock is held",
            rc == 0 and "confirm" in dry,
            f"{time.time() - t:.1f}s {json.dumps(dry.get('counts', dry))[:140]}",
        )
        rc, done = agent_json(
            "catalog",
            "import",
            BACKUPS / "bundle.json",
            "--as",
            a.as_host,
            "--json",
            "--confirm",
            dry.get("confirm", "x"),
        )
        need(
            "F9 catalog import --confirm while the lock is held",
            rc == 0 and done.get("applied"),
            json.dumps(done.get("counts", done))[:140],
        )
        rc, again = agent_json(
            "catalog", "import", BACKUPS / "bundle.json", "--as", a.as_host, "--json"
        )
        check(
            "F10 a second dry run: nothing new",
            rc == 0 and not again.get("counts", {}).get("new"),
            json.dumps(again.get("counts"))[:140],
        )
        rc, hdry = agent_json("history", "import", snap, "--as", a.as_host, "--json")
        check(
            "F11 history import dry run",
            rc == 0 and "confirm" in hdry,
            json.dumps(hdry.get("counts", hdry))[:160],
        )
        rc, hdone = agent_json(
            "history",
            "import",
            snap,
            "--as",
            a.as_host,
            "--json",
            "--confirm",
            hdry.get("confirm", "x"),
        )
        check(
            "F12 history import --confirm",
            rc == 0 and hdone.get("applied"),
            f"total {hdone.get('total')}",
        )
        rc, hl = agent_json("history", "list", "--json")
        n_hist = len(hl.get("entries", []))
        check(
            "F13 history list holds the rows, ids py:<host>:<id>",
            n_hist == hdone.get("total")
            and all(e["id"].startswith(f"py:{a.as_host}:") for e in hl.get("entries", [])),
            f"{n_hist} rows",
        )
        check(
            "F14 the snapshot is unchanged and has no -wal beside it",
            not Path(f"{snap}-wal").exists() and integrity == "ok",
        )
        rc, out, err = agent("catalog", "export", WORK / "export-forward.json")
        exported = (
            {m.get("id") for m in json.loads((WORK / "export-forward.json").read_text())["models"]}
            if rc == 0
            else set()
        )
        mine = {m.get("id") for m in bundle["models"] if m.get("host", a.as_host) == a.as_host}
        check(
            "F15 catalog export gives back the bundle's rows of this host, by id",
            mine <= exported,
            f"{len(mine)} rows, missing {len(mine - exported)}",
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
        rc, sig1 = signature()
        check(
            "F16 reconcile: signature equals the agent record and the 'before' one",
            rc == 0 and sig1 == sig0,
            sig1[:23],
        )

        print("\n== forward 4: switch the clients", flush=True)
        write_settings("cutover")
        write_router(only=True, refusal_final=True)
        rc, out, err = sh(
            [
                "/opt/venv/bin/python",
                "-c",
                f"from llm_router.config import load_config; load_config('{ROUTER_YAML}')",
            ]
        )
        check("F17 router config with llmctl_next_only validates", rc == 0, tail(out, err))
        rc, out, _ = sh([H / "AI/bin/llm-unit", "mode"])
        check("F18 llm-unit mode says cutover", out.strip() == "cutover", out.strip())
        rc, sig2 = signature()
        check(
            "F19 signature after the switch = before; no engine pid changed",
            rc == 0 and sig2 == sig0 and pids() == pid0,
            f"{pid0} -> {pids()}",
        )
        check(
            "F20 the switch changed no preset, profile env or unit file",
            all(
                sha256(k) == v
                for k, v in sums0.items()
                if k not in (str(SETTINGS), str(ROUTER_YAML))
            ),
        )

        print("\n== forward 5: negative drills", flush=True)
        rc, out, err = agent("stop", "reh-lane", "--wait")
        need(
            "N0 agent stop reh-lane: the lane is inactive",
            rc == 0 and unit_state(f"{LANE}.service")[0] != "active",
            tail(out, err),
        )
        rc, out, err = py(
            "start-unit", f"{LANE}.service", "--yes", env={"LLMCTL_NEXT_DISABLED": "1"}
        )
        check(
            "N1 LLMCTL_NEXT_DISABLED=1: Python start-unit refused, nothing starts",
            rc != 0 and unit_state(f"{LANE}.service")[0] != "active",
            f"rc {rc}: {tail(out, err, 110)}",
        )
        n = len(ops())
        rc, out, err = py("start-unit", f"{LANE}.service", "--yes")
        check(
            "N2 Python start-unit of the stopped enrolled lane: one agent operation, the lane serves",
            rc == 0 and len(ops()) == n + 1 and wait_for(lambda: served(LANE_PORT) == ["reh/lane"]),
            f"rc {rc}, ops {n} -> {len(ops())}",
        )
        (UNITS / "reh-other.service").write_text("[Service]\nExecStart=/bin/sleep infinity\n")
        systemctl("daemon-reload")
        rc, out, err = py("start-unit", "reh-other.service", "--yes")
        check(
            "N3 Python start-unit of a unit that is not enrolled: refused, nothing starts",
            rc != 0 and unit_state("reh-other.service")[0] != "active",
            f"rc {rc}: {tail(out, err, 110)}",
        )
        rc, out, err = sh([H / "AI/bin/llm-unit", "start", "reh-other.service"])
        check(
            "N4 llm-unit start of a unit that is not enrolled: refused",
            rc != 0 and unit_state("reh-other.service")[0] != "active",
            f"rc {rc}: {tail(out, err, 110)}",
        )
        agent("stop", "reh-lane", "--wait")
        n = len(ops())
        rc, out, err = lane_unit("start", f"{LANE}.service")
        check(
            "N5 lane script lane_unit start of the stopped lane: through llm-unit to the agent",
            rc == 0 and len(ops()) == n + 1 and wait_for(lambda: served(LANE_PORT) == ["reh/lane"]),
            f"rc {rc}, ops {n} -> {len(ops())}",
        )
        before_pid, eng_pid = pids()[LANE], pids()[ENGINE]
        rc, out, err = lane_unit("start", f"{LANE}.service")
        wait_for(lambda: served(LANE_PORT) == ["reh/lane"])
        print(
            f"INFO  N5b lane_unit start of the running lane: rc {rc}, lane pid {before_pid} -> {pids()[LANE]}, "
            f"engine pid {eng_pid} -> {pids()[ENGINE]} (systemctl start would change nothing)",
            flush=True,
        )
        check("N5c ...and the engine on other cards kept its pid", pids()[ENGINE] == eng_pid)
        rc, out, err = router_switch("reh-b")
        check(
            "N6 preset switch through llm-router (agent run): rc 0, serves reh/model-b",
            rc == 0 and wait_for(lambda: served(PORT) == ["reh/model-b"]),
            tail(out, err),
        )
        pid_b = pids()[ENGINE]
        agent("stop", "reh-lane", "--wait")  # N9 needs a stopped lane
        check(
            "N7 the active env is the reh-b profile",
            (SERVICES / "engine.env").read_bytes() == (SERVICES / "engine.env.reh-b").read_bytes(),
        )
        systemctl("stop", "llmctl-next")
        t0 = time.time()
        rc, out, err = router_switch("reh-a")
        check(
            "N8 agent stopped: a router switch fails and changes nothing",
            rc != 0 and served(PORT) == ["reh/model-b"] and pids()[ENGINE] == pid_b,
            f"rc {rc}: {tail(out, err, 110)}",
        )
        rc, out, err = py("start-unit", f"{LANE}.service", "--yes")
        check(
            "N9 agent stopped: Python start-unit refused, nothing starts",
            rc != 0 and unit_state(f"{LANE}.service")[0] != "active",
            f"rc {rc}: {tail(out, err, 110)}",
        )
        time.sleep(max(0, 60 - (time.time() - t0)))  # the plan's one minute without the agent
        systemctl("start", "llmctl-next")
        need(
            "N10 the agent is back; the engines kept their pids",
            wait_for(lambda: agent("status", "--json")[0] == 0) and pids()[ENGINE] == pid_b,
        )
        rc, out, err = agent("run", "reh-lane", "--wait")
        need(
            "N10b the agent starts the lane",
            rc == 0 and wait_for(lambda: served(LANE_PORT) == ["reh/lane"]),
            tail(out, err),
        )
        # llm-heal in report-only: a crash of an owned unit is reported, not repaired.
        systemctl("kill", "-s", "KILL", f"{LANE}.service")
        wait_for(lambda: unit_state(f"{LANE}.service")[0] == "failed", 10)
        rc, out, err = heal_run()  # before the agent's 15 s watcher acts
        state_after = unit_state(f"{LANE}.service")[0]
        check(
            "N11 llm-heal (report-only) reports the crashed owned lane and repairs nothing",
            state_after == "failed" and "owned by llmctl-next" in heal_held(LANE),
            f"rc {rc}, lane {state_after}, held: {heal_held(LANE)[:90]}",
        )
        rc, out, err = agent("run", "reh-lane", "--wait")
        need(
            "N11b the agent starts the lane",
            rc == 0 and wait_for(lambda: served(LANE_PORT) == ["reh/lane"]),
            tail(out, err),
        )
        agent("status", "--json")  # an observation pass: the baseline is "running" (finding 3)
        n_find = len(audit_findings()[1])
        systemctl("stop", f"{LANE}.service")  # a raw foreign write on an enrolled unit
        agent("status", "--json")  # an observation pass records the transition
        rc, rows = audit_findings()
        check(
            "N12 audit --unexplained records the raw stop: one new finding for the lane",
            rc == 0
            and len(rows) == n_find + 1
            and any(LANE in json.dumps(r) for r in rows[:1] + rows[-1:]),
            f"{n_find} -> {len(rows)} finding(s)",
        )
        rc, out, err = agent("run", "reh-lane", "--wait")
        check(
            "N13 the agent starts the lane again",
            rc == 0 and wait_for(lambda: served(LANE_PORT) == ["reh/lane"]),
            tail(out, err),
        )
        # Finding 3 repro: stop op, run op, then a raw stop before any observation pass.
        n_find = len(audit_findings()[1])
        agent("stop", "reh-lane", "--wait")
        agent("run", "reh-lane", "--wait")
        systemctl("stop", f"{LANE}.service")
        agent("status", "--json")
        print(
            f"INFO  N13b raw stop right after an agent stop+run: findings {n_find} -> {len(audit_findings()[1])}"
            " (a recorded stop needs +1)",
            flush=True,
        )
        agent("run", "reh-lane", "--wait")
        wait_for(lambda: served(LANE_PORT) == ["reh/lane"])
        # A real lane id of the Mac registry; the stand-in llama-swap lists it.
        mac_lane = next(
            line.split(":", 1)[1].strip()
            for line in (SRC / "local-llm/registry/models.yaml").read_text().splitlines()
            if line.strip().startswith("- id:")
        )
        (WORK / "swap-config.yaml").write_text(f"models:\n  {mac_lane}:\n    cmd: /bin/true\n")
        swap = subprocess.Popen([sys.executable, WORK / "swap.py", str(SWAP_PORT), mac_lane])
        time.sleep(0.5)
        rc, out, err = sh(
            ["bash", SRC / "local-llm/bin/llmctl", "start", mac_lane],
            env={
                "LLMCTL_CONFIG": str(MAC_CONF),
                "LLMCTL_FORCE": "1",
                "LLMCTL_PY": "/opt/venv/bin/python",
            },
        )
        swap.terminate()
        check(
            # The agent has no llama-swap workload here: this is the fail-closed path.
            "N14 Mac llmctl next_only, no agent answer for llama-swap: start refused, no launchctl call",
            rc != 0
            and "refusing to touch llama-swap directly" in out + err
            and not (WORK / "launchctl.calls").exists(),
            f"rc {rc}: {tail(out, err, 120)}",
        )
        rc, out, err = agent("run", "reh-a", "--workload", "reh-engine", "--wait")
        check(
            "N15 one preset switch by the agent: serves reh/model-a",
            rc == 0 and served(PORT) == ["reh/model-a"],
            tail(out, err),
        )
        need("N16 drained", wait_for(lambda: not open_ops(), 60))
        rc, sig3 = signature()
        check("N17 signature after the drills agrees with the agent", rc == 0, sig3[:23])

        rc, added = agent_json(
            "model",
            "add",
            "m6c-rehearsal-add",
            "--engine",
            "vllm",
            "--path",
            WORK / "models/a",
            "--set",
            "notes=added in the M6c rehearsal",
            "--json",
        )
        check("N18 catalog edit: model add", rc == 0, json.dumps(added)[:100])
        vllm_rows = [
            m
            for m in bundle["models"]
            if m.get("runtime") == "vllm" and m.get("host", a.as_host) == a.as_host
        ]
        target = vllm_rows[0]["name"] if vllm_rows else None
        rc, edited = agent_json(
            "model", "edit", target, "--set", "notes=edited in the M6c rehearsal", "--json"
        )
        check("N19 catalog edit: model edit of an imported row", rc == 0, f"{target}")

        # ============================== R1 ===================================
        print("\n== R1: back to the delegate state", flush=True)
        pid1 = pids()
        write_settings("delegate")
        write_router(only=False, refusal_final=True)
        write_heal(report_only=False)
        legacy_writers(True)
        DS4_FLAG.unlink()
        check(
            "R1.1 legacy writers on again",
            legacy_writer_state() == ["active", "active"],
            str(legacy_writer_state()),
        )
        rc, out, _ = sh([H / "AI/bin/llm-unit", "mode"])
        check("R1.2 llm-unit mode says delegate", out.strip() == "delegate", out.strip())
        rc, out, err = agent("catalog", "export", WORK / "export-r1.json")
        rc2, out2, err2 = py("import-registry", WORK / "export-r1.json", "--update-models")
        check(
            "R1.3 catalog export, then Python import-registry --update-models",
            rc == 0 and rc2 == 0,
            tail(out2, err2),
        )
        rc, out, err = py("reconcile")
        check("R1.4 Python reconcile", rc == 0, tail(out, err))
        check("R1.5 the reversal changed no pid", pids() == pid1, f"{pid1} -> {pids()}")
        agent("stop", "reh-lane", "--wait")
        n = len(ops())
        rc, out, err = py("start-unit", f"{LANE}.service", "--yes")
        check(
            "R1.6 delegated write, Python: start-unit of the stopped lane goes to the agent",
            rc == 0 and len(ops()) == n + 1 and wait_for(lambda: served(LANE_PORT) == ["reh/lane"]),
            f"ops {n} -> {len(ops())}",
        )
        n = len(ops())
        rc, out, err = sh([H / "AI/bin/llm-unit", "restart", f"{LANE}.service"])
        check(
            "R1.7 delegated write, llm-unit restart of the lane goes to the agent",
            rc == 0 and len(ops()) > n and wait_for(lambda: served(LANE_PORT) == ["reh/lane"]),
            f"rc {rc}, ops {n} -> {len(ops())}",
        )
        rc, out, err = lane_unit("stop", "reh-other.service")
        check(
            "R1.8 lane script lane_unit uses systemctl again (unit not enrolled)",
            rc == 0,
            tail(out, err),
        )
        rc, out, err = router_switch("reh-b")
        check(
            "R1.9 delegated write, llm-router switch (agent run) serves reh/model-b",
            rc == 0 and wait_for(lambda: served(PORT) == ["reh/model-b"]),
            tail(out, err),
        )
        rc, out, err = sh(["bash", "-c", f'{H}/AI/services/ds4-llmctl-sync status; echo "rc=$?"'])
        check("R1.10 ds4-llmctl-sync runs again (flag removed)", "rc=0" in out, tail(out, err, 80))
        # A crash, not a stop: llm-heal repairs a failed unit, never a stopped one.
        systemctl("kill", "-s", "KILL", f"{LANE}.service")
        wait_for(lambda: unit_state(f"{LANE}.service")[0] == "failed", 10)
        rc, out, err = heal_run()  # before the agent's 15 s watcher acts
        check(
            "R1.11 llm-heal repairs on again: it restarts the crashed lane",
            wait_for(lambda: unit_state(f"{LANE}.service")[0] == "active", 20)
            and "restart" in out.lower(),
            f"{unit_state(f'{LANE}.service')[0]}: {tail(out, err, 120)}",
        )
        wait_for(lambda: served(LANE_PORT) == ["reh/lane"])
        rc, out, err = py("status", "--json")
        check(
            "R1.12 Python status sees the engine serving reh/model-b",
            rc == 0 and "reh/model-b" in out,
            tail(out, err, 80),
        )
        need("R1.13 drained", wait_for(lambda: not open_ops(), 60))

        # ============================== R2 ===================================
        print("\n== R2: full reversal", flush=True)
        rc, sig4 = signature()
        check("R2.1 signature before R2 agrees with the agent", rc == 0, sig4[:23])
        before = (pids(), served(PORT), served(LANE_PORT))
        # `catalog export` asks the agent: export before the agent stops (plan R2 step 4 before 3).
        rc, out, err = agent("catalog", "export", WORK / "export-r2.json")
        write_settings("off")
        write_router(only=False, refusal_final=False)
        systemctl("disable", "--now", "llmctl-next")
        check(
            "R2.2 the agent is stopped and disabled; the engines kept pid and model",
            unit_state("llmctl-next")[0] != "active"
            and (pids(), served(PORT), served(LANE_PORT)) == before,
            str(before[0]),
        )
        rc2, out2, err2 = py("import-registry", WORK / "export-r2.json", "--update-models")
        check(
            "R2.3 catalog export, Python import-registry --update-models",
            rc == 0 and rc2 == 0,
            tail(out2, err2),
        )
        py("export-registry", WORK / "after.json")
        after = {m["name"]: m for m in json.loads((WORK / "after.json").read_text())["models"]}
        check(
            "R2.4 Python sees the added model and the edit of an imported row",
            "m6c-rehearsal-add" in after and "M6c rehearsal" in json.dumps(after.get(target, {})),
        )
        rc, out, err = py("reconcile")
        rc2, st, _ = py("status", "--json")
        check(
            "R2.5 Python reconcile; status shows the engine serving reh/model-b",
            rc == 0 and "reh/model-b" in st,
            tail(out, err, 80),
        )
        rc, out, err = router_switch("reh-a")
        check(
            "R2.6 llm-router on its own path: switch to reh-a (env copy + systemctl restart)",
            rc == 0 and wait_for(lambda: served(PORT) == ["reh/model-a"]),
            tail(out, err),
        )
        rc, out, err = py("run", "reh-b", "--yes", "--direct", "--force")
        check(
            "R2.7 legacy Python run <preset> --direct serves reh/model-b",
            rc == 0 and wait_for(lambda: served(PORT) == ["reh/model-b"]),
            tail(out, err),
        )
        rc, out, err = sh([H / "AI/bin/llm-unit", "restart", f"{LANE}.service"])
        check(
            "R2.8 llm-unit restart of the lane: systemctl",
            rc == 0 and wait_for(lambda: served(LANE_PORT) == ["reh/lane"]),
            tail(out, err),
        )
        check(
            "R2.9 the agent journal and the deploy-backups are kept",
            (BACKUPS / "llmctl.sqlite3").exists()
            and any(DATA.glob("*.sqlite*"))
            or any(DATA.glob("journal*")),
            ", ".join(sorted(p.name for p in DATA.iterdir()))[:120],
        )
    except Stop as e:
        print(f"\nstopped at: {e}", flush=True)

    failed = [n for n, ok, _ in results if not ok]
    print(
        f"\n{len(results) - len(failed)} of {len(results)} checks passed"
        + (f"; failed: {failed}" if failed else "")
    )
    return 1 if failed or not results else 0


if __name__ == "__main__":
    sys.exit(main())
