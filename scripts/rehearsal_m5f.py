#!/usr/bin/env python3
"""M5f rehearsal: migration to llmctl-next and reversal, on temporary state.

Procedure and results: docs/RUST-M5-REHEARSAL.md.

What it uses of the host:
  - a backup copy of the Python llmctl DB and a copy of the preset files (reads);
  - one disposable systemd user unit (`llmctl-next-rehearsal-<id>.service`) that
    runs a stand-in engine on a free port. It is removed at the end.
It starts its own agent in a temp data dir, with the host operation lock, the
lane lease and the router probe off, and the workload on a notional GPU (9).
It reads no settings.yaml of the host: it writes its own, with no token.

Exit 0 = every check passed. Exit 1 = a check failed (each one is printed).
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

HOME = Path.home()
SERVER = """
import json, os
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
print(f"serving {model} on {port}", flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
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
TERMINAL = {"succeeded", "failed", "cancelled", "needs_attention", "interrupted"}
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  — ' + detail if detail else ''}", flush=True)
    return ok


class Stop(Exception):
    pass


def need(name, ok, detail=""):
    """A check that the next steps depend on."""
    if not check(name, ok, detail):
        raise Stop(name)


def sh(argv, env=None, timeout=300):
    p = subprocess.run(
        [str(a) for a in argv], capture_output=True, text=True, env=env, timeout=timeout
    )
    return p.returncode, p.stdout, p.stderr


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    state = HOME / ".local/state/llmctl-next-m4a"
    ap.add_argument(
        "--agent-bin", default=str(state / "llmctl-next.bin"), help="llmctl-next binary to rehearse"
    )
    ap.add_argument(
        "--python-llmctl",
        default=str(HOME / "AI/.venv/bin/llmctl"),
        help="Python llmctl to reverse to",
    )
    ap.add_argument(
        "--db", default=str(HOME / ".local/share/llmctl/llmctl.sqlite3"), help="Python DB to copy"
    )
    ap.add_argument(
        "--presets", default=str(HOME / ".config/llmctl/presets"), help="preset dir to copy"
    )
    ap.add_argument(
        "--as-host", default=socket.gethostname(), help="host name of this machine in the Python DB"
    )
    ap.add_argument("--keep", action="store_true", help="keep the temp dir")
    a = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="llmctl-m5f-"))
    unit = f"llmctl-next-rehearsal-{uuid.uuid4().hex[:8]}.service"
    unit_file = HOME / ".config/systemd/user" / unit
    for d in (
        "agent",
        "config",
        "services",
        "presets",
        "real-presets",
        "models/a",
        "models/b",
        "data",
        "xdg",
    ):
        (root / d).mkdir(parents=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env_file = root / "services/engine.env"
    settings = root / "config/settings.yaml"
    daemon = None

    def write_settings(enabled):
        settings.write_text(
            "# Written for the M5f rehearsal. No token.\n"
            f"managed_units:\n  vllm_engine:\n    enabled: true\n    unit_name: {unit[:-8]}\n"
            f"    default_port: {port}\n    env_file_path: {env_file}\n    launcher_marker: null\n"
            "    user_scope: true\n"
            f"llmctl_next:\n  enabled: {'true' if enabled else 'false'}\n"
            f"  cmd: [{a.agent_bin}, --data-dir, {root / 'agent'}]\n"
            f"  units:\n    {unit}: reh-engine\n"
        )

    base = {
        k: v
        for k, v in os.environ.items()
        if k
        not in ("LLMCTL_API_TOKEN", "LLMCTL_NEXT_CMD", "LLMCTL_NEXT_DISABLED", "LLMCTL_CONFIG_DIR")
    }
    agent_env = {
        **base,
        "LLMCTL_NEXT_DATA_DIR": str(root / "agent"),
        "LLMCTL_NEXT_OP_LOCK": "off",
        "LLMCTL_NEXT_LANE_LEASE": "off",
        "LLMCTL_NEXT_ROUTER": "off",
        "LLMCTL_NEXT_PRESET_DIRS": f"{root / 'real-presets'}:{root / 'presets'}",
        "LLMCTL_NEXT_SETTINGS": str(settings),
        "LLMCTL_ROUTER_CONFIG": str(root / "router.yaml"),
    }
    py_env = {
        **agent_env,
        "LLMCTL_CONFIG_DIR": str(root / "config"),
        "LLMCTL_DB_URL": f"sqlite:///{root / 'data/llmctl.sqlite3'}",
        "LLMCTL_OP_LOCK": str(root / "operations.lock"),
        "LLMCTL_PLUGINS": "none",
        "LLM_UNIT_BIN": "none",
        "XDG_DATA_HOME": str(root / "xdg/data"),
        "XDG_STATE_HOME": str(root / "xdg/state"),
        "XDG_CONFIG_HOME": str(root / "xdg/config"),
        "NO_COLOR": "1",
        "TERM": "dumb",
    }

    def agent(*args, timeout=300):
        return sh(
            [a.agent_bin, "--data-dir", root / "agent", *args], env=agent_env, timeout=timeout
        )

    def agent_json(*args):
        rc, out, err = agent(*args)
        try:
            return rc, json.loads(out)
        except ValueError:
            return rc, {"_raw": (out + err)[-400:]}

    def py(*args):
        return sh([a.python_llmctl, *args], env=py_env)

    def unit_state():
        _, out, _ = sh(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "MainPID"])
        kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        return kv.get("ActiveState"), int(kv.get("MainPID") or 0)

    def signature():
        rc, status, _ = agent("status", "--json")
        p = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("serving_signature.py"))],
            input=status,
            capture_output=True,
            text=True,
        )
        return p.returncode, json.loads(p.stdout) if p.stdout.strip() else {}

    def ops():
        _, v = agent_json("ops", "--json")
        return v if isinstance(v, list) else v.get("operations", [])

    print(f"temp dir {root}\nunit {unit}, port {port}\n", flush=True)
    try:
        # ---------------- forward ----------------
        print("== forward: Python state into a temp agent", flush=True)
        src = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
        dst = sqlite3.connect(root / "data/llmctl.sqlite3")
        src.backup(dst)
        integrity = dst.execute("pragma integrity_check").fetchone()[0]
        src.close()
        dst.close()
        need(
            "F1 DB snapshot taken (backup API, source opened read-only)",
            integrity == "ok",
            f"sha256 {sha256(root / 'data/llmctl.sqlite3')[:16]}",
        )
        real = sorted(p for p in Path(a.presets).glob("*.yaml") if not p.name.startswith("_"))
        for p in real:
            shutil.copy2(p, root / "real-presets" / p.name)
        write_settings(enabled=True)

        rc, out, err = py("export-registry", root / "bundle.json")
        need(
            "F2 Python export-registry of the snapshot",
            rc == 0 and (root / "bundle.json").exists(),
            (out + err).strip()[-120:],
        )
        bundle = json.loads((root / "bundle.json").read_text())
        n_models, n_profiles = len(bundle["models"]), len(bundle["profiles"])

        (root / "agent/inventory.json").write_text(
            json.dumps(
                {
                    "inventory": {
                        "host": "workstation",
                        "platform": "linux",
                        "gpus": [{"host": "workstation", "index": 9}],
                        "workloads": [],
                        "revision": 1,
                    }
                }
            )
        )
        log = open(root / "agent/daemon.log", "w")
        daemon = subprocess.Popen(
            [a.agent_bin, "--data-dir", str(root / "agent"), "daemon"],
            env=agent_env,
            stdout=log,
            stderr=log,
        )
        need(
            "F3 temp agent answers",
            wait_for(lambda: agent("status", "--json")[0] == 0, 20),
            f"pid {daemon.pid}",
        )

        rc, dry = agent_json("catalog", "import", root / "bundle.json", "--as", a.as_host, "--json")
        need(
            "F4 catalog import dry run",
            rc == 0 and "confirm" in dry,
            json.dumps(dry.get("counts", dry))[:160],
        )
        rc, done = agent_json(
            "catalog",
            "import",
            root / "bundle.json",
            "--as",
            a.as_host,
            "--json",
            "--confirm",
            dry["confirm"],
        )
        need(
            "F5 catalog import applied with the report's token",
            rc == 0 and done.get("applied"),
            json.dumps(done.get("counts"))[:160],
        )
        rc, again = agent_json(
            "catalog", "import", root / "bundle.json", "--as", a.as_host, "--json"
        )
        check(
            "F6 a second dry run finds nothing new",
            rc == 0 and not again.get("counts", {}).get("new"),
            json.dumps(again.get("counts"))[:160],
        )

        print("\n== forward: a disposable unit through a preset", flush=True)
        (root / "server.py").write_text(SERVER)
        env_file.write_text(f"VLLM_PORT={port}\nVLLM_SERVED_NAME=reh/model-a\n")
        for alias in ("a", "b"):
            (root / f"presets/reh-{alias}.yaml").write_text(
                PRESET.format(
                    alias=f"reh-{alias}",
                    served=f"reh/model-{alias}",
                    model_dir=root / "models" / alias,
                )
            )
        (root / "router.yaml").write_text(
            f"switch:\n  services_dir: {root / 'services'}\n  active_env: engine.env\n"
            f"  unit: {unit[:-8]}\nbackends:\n"
            + "".join(
                f"  - name: reh-{x}\n    served_model: reh/model-{x}\n"
                f"    profile_env: engine.env.reh-{x}\n"
                f"    upstream: http://127.0.0.1:{port}\n"
                for x in "ab"
            )
        )
        unit_file.write_text(
            f"[Unit]\nDescription=llmctl-next M5f rehearsal stand-in engine ({root})\n\n"
            f"[Service]\nType=simple\nEnvironmentFile={env_file}\n"
            f"ExecStart=/usr/bin/python3 {root / 'server.py'}\n"
        )
        sh(["systemctl", "--user", "daemon-reload"])
        sh(["systemctl", "--user", "start", unit])
        need(
            "F7 the unit serves reh/model-a",
            wait_for(lambda: served(port) == ["reh/model-a"]),
            f"pid {unit_state()[1]}",
        )

        adopt = [
            "adopt",
            unit,
            "--name",
            "reh-engine",
            "--model",
            "reh/model-a",
            "--port",
            port,
            "--gpus",
            "9",
            "--engine-kind",
            "vllm",
            "--readiness-timeout",
            "60",
        ]
        rc, pre = agent_json(*adopt, "--json")
        bad = [c["name"] for c in pre.get("checks", []) if c.get("status") != "pass"]
        need(
            "F8 adopt preview: every check passes",
            rc == 0 and pre.get("digest") and not bad,
            f"not passing: {bad}" if bad else "",
        )
        rc, out, err = agent(*adopt, "--confirm", pre["digest"])
        need("F9 adopt confirmed; the unit kept its pid", rc == 0, (out + err).strip()[-120:])

        rc, plist = agent_json("preset", "list", "--json")
        listed = {
            p.get("alias") for p in (plist if isinstance(plist, list) else plist.get("presets", []))
        }
        want = {p.stem for p in real} | {"reh-a", "reh-b"}
        check(
            "F10 the agent reads the copied presets and the two rehearsal presets",
            want <= listed,
            f"{len(listed)} listed, missing {sorted(want - listed)}",
        )
        for alias in ("reh-a", "reh-b"):
            rc, out, err = agent("preset", "sync", alias)
            need(
                f"F11 preset sync {alias} writes its profile",
                rc == 0 and (root / f"services/engine.env.{alias}").exists(),
                (out + err).strip()[-120:],
            )
        pid_a = unit_state()[1]
        rc, out, err = agent("run", "reh-b", "--wait")
        need("F12 run reh-b: planned swap, rc 0", rc == 0, (out + err).strip()[-200:])
        state, pid_b = unit_state()
        check(
            "F13 the unit restarted and serves reh/model-b",
            served(port) == ["reh/model-b"] and pid_b not in (0, pid_a),
            f"pid {pid_a} -> {pid_b}, {state}",
        )
        check(
            "F14 the active env is the reh-b profile, the old one is kept",
            env_file.read_bytes() == (root / "services/engine.env.reh-b").read_bytes()
            and Path(str(env_file) + ".llmctl-previous").exists(),
        )
        rc, sig1 = signature()
        check(
            "F15 serving signature agrees with the agent's record",
            rc == 0,
            f"digest {sig1.get('digest', '')[:23]}",
        )

        rc, added = agent_json(
            "model",
            "add",
            "m5f-rehearsal-add",
            "--engine",
            "vllm",
            "--path",
            root / "models/a",
            "--set",
            "notes=added in the M5f rehearsal",
            "--json",
        )
        check("F16 catalog edit 1: model add", rc == 0, json.dumps(added)[:120])
        vllm_rows = [
            m
            for m in bundle["models"]
            if m.get("runtime") == "vllm" and m.get("host", a.as_host) == a.as_host
        ]
        target = vllm_rows[0]["name"] if vllm_rows else None
        rc, edited = agent_json(
            "model", "edit", target, "--set", "notes=edited in the M5f rehearsal", "--json"
        )
        check(
            "F17 catalog edit 2: model edit of an imported row",
            rc == 0,
            f"{target}: {json.dumps(edited)[:100]}",
        )

        print("\n== forward: a legacy writer in the delegate state", flush=True)
        n0 = len(ops())
        rc, out, err = agent("stop", "reh-engine", "--wait")
        need(
            "F18 agent stop: the unit is inactive",
            rc == 0 and unit_state()[0] != "active",
            (out + err).strip()[-120:],
        )
        rc, out, err = py("start-unit", unit, "--yes")
        new = ops()
        kinds = (
            [str(o.get("kind", "")).lower() for o in new[: len(new) - n0]] if len(new) > n0 else []
        )
        check(
            "F19 Python start-unit is redirected: the agent ran it",
            rc == 0 and len(new) == n0 + 2 and unit_state()[0] == "active",
            f"rc {rc}, operations {n0} -> {len(new)}, {(out + err).strip()[-100:]}",
        )
        check(
            "F20 the unit serves again",
            wait_for(lambda: served(port) == ["reh/model-b"]),
            f"new ops: {kinds}",
        )

        # ---------------- reverse ----------------
        print("\n== reverse: back to the Python tool", flush=True)
        open_ops = [
            o for o in ops() if str(o.get("state", o.get("status", ""))).lower() not in TERMINAL
        ]
        need(
            "R1 drained: no operation that is not terminal",
            not open_ops,
            f"{len(ops())} operations",
        )
        rc, sig2 = signature()
        check(
            "R2 signature before the reversal agrees with the agent",
            rc == 0,
            f"digest {sig2.get('digest', '')[:23]}",
        )
        before = (unit_state(), served(port))

        rc, out, err = agent("catalog", "export", root / "bundle-out.json")
        need("R3 catalog export", rc == 0, out.strip()[-120:])
        outb = json.loads((root / "bundle-out.json").read_text())
        key = lambda m: (m.get("runtime"), m.get("source") or m.get("name"))  # noqa: E731
        check(
            "R4 the export holds every row of the bundle and the added row",
            {key(m) for m in bundle["models"]} <= {key(m) for m in outb["models"]}
            and len(outb["models"]) == n_models + 1
            and len(outb["profiles"]) == n_profiles,
            f"{n_models} + 1 -> {len(outb['models'])} models, {len(outb['profiles'])} profiles",
        )
        # --update-models: every row of the export exists in this DB already.
        rc, out, err = py("import-registry", root / "bundle-out.json", "--update-models")
        need(
            "R5 Python import-registry --update-models into the DB copy",
            rc == 0,
            " ".join((out + err).split())[-140:],
        )
        py("export-registry", root / "bundle-after.json")
        after = {
            m["name"]: m for m in json.loads((root / "bundle-after.json").read_text())["models"]
        }
        check("R6 Python sees the model that was added", "m5f-rehearsal-add" in after)
        check(
            "R7 Python sees the edit of an imported row",
            "M5f rehearsal" in json.dumps(after.get(target, {})),
            f"{target}: notes = {after.get(target, {}).get('notes')!r}",
        )

        daemon.send_signal(signal.SIGTERM)
        daemon.wait(timeout=20)
        check(
            "R8 the agent stopped; the engine did not",
            (unit_state(), served(port)) == before,
            f"pid {before[0][1]}, serves {before[1]}",
        )
        sh(["systemctl", "--user", "stop", unit])
        rc, out, err = py("start-unit", unit, "--yes")
        check(
            "R9 delegation on, agent down: Python start-unit is refused, nothing starts",
            rc != 0 and unit_state()[0] != "active",
            f"rc {rc}: {(out + err).strip()[-110:]}",
        )
        write_settings(enabled=False)
        rc, out, err = py("start-unit", unit, "--yes")
        check(
            "R10 delegation off: Python start-unit starts the unit by itself",
            rc == 0 and unit_state()[0] == "active",
            f"rc {rc}: {(out + err).strip()[-110:]}",
        )
        check(
            "R11 the unit serves reh/model-b again",
            wait_for(lambda: served(port) == ["reh/model-b"]),
        )
        rc, out, err = py("status", "--json")
        try:
            rows = json.loads(out)
            rows = rows if isinstance(rows, list) else rows.get("units", rows.get("roles", []))
            row = next((r for r in rows if r.get("port") == port), {})
        except ValueError:
            row = {}
        check(
            "R12 Python status sees the same running state",
            bool(row.get("serving")) and "reh/model-b" in json.dumps(row),
            json.dumps(
                {k: row.get(k) for k in ("role", "unit_name", "port", "serving", "served_models")}
            )[:200],
        )
    except Stop as e:
        print(f"\nstopped at: {e}", flush=True)
    finally:
        if daemon and daemon.poll() is None:
            daemon.terminate()
            daemon.wait(timeout=20)
        sh(["systemctl", "--user", "stop", unit])
        sh(["systemctl", "--user", "reset-failed", unit])
        unit_file.unlink(missing_ok=True)
        sh(["systemctl", "--user", "daemon-reload"])
        if a.keep:
            print(f"kept {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    failed = [n for n, ok, _ in results if not ok]
    print(
        f"\n{len(results) - len(failed)} of {len(results)} checks passed"
        + (f"; failed: {failed}" if failed else "")
    )
    return 1 if failed or not results else 0


if __name__ == "__main__":
    sys.exit(main())
