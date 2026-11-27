#!/usr/bin/env python3
"""Serving signature of one host (M6 plan, design decision 8).

Reads the agent's `status --json` for the list of workloads (name, unit,
port). Then it observes each workload directly: the service manager, the
endpoint and `nvidia-smi`. It does not ask the agent what runs.

    llmctl-next status --json | serving_signature.py
    serving_signature.py --status status.json

Output: one JSON record on stdout. `digest` covers the fields that a
cutover must not change: unit state, main pid, port, listening, served
models and the cards held. Loaded on-demand models, loaded lanes and MiB
figures are in the record but not in the digest: they change by
themselves. Exit 0 = the record agrees with the agent, 1 = it does not
(see `disagreements`), 2 = the input could not be read.

Python 3.9, standard library only (the Macs have 3.9). Changes nothing.
"""
import argparse
import datetime
import hashlib
import json
import socket
import subprocess
import sys
import urllib.request

TIMEOUT_S = 3


def sh(*argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def parse_show(text):
    """`systemctl show -p ActiveState -p MainPID` -> (active, pid or None)."""
    kv = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    pid = int(kv.get("MainPID") or 0)
    return kv.get("ActiveState") == "active", pid or None


def launchd_pid(table, label):
    """Pid of `label` in the `launchctl list` table, else None.

    A vendor app's label has a suffix (`application.<id>.<n>.<n>`), so a
    label that starts with `<label>.` counts too.
    """
    for line in table.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and (parts[2] == label or parts[2].startswith(label + ".")):
            if parts[0].isdigit():
                return int(parts[0])
    return None


def get_json(port, path):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=TIMEOUT_S) as r:
            return json.load(r)
    except (OSError, ValueError):
        return None


def listening(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT_S):
            return True
    except OSError:
        return False


def names(doc, key, field):
    """Sorted `field` of each item of `doc[key]`; None when the shape is wrong."""
    items = doc.get(key) if isinstance(doc, dict) else None
    if not isinstance(items, list):
        return None
    return sorted(str(i[field]) for i in items if isinstance(i, dict) and field in i)


def gpu_holdings():
    """{unit name: {card index: MiB}} from nvidia-smi and each pid's cgroup."""
    index = dict(
        (uuid.strip(), int(i))
        for i, uuid in (
            line.split(",") for line in sh(
                "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"
            ).splitlines() if "," in line
        )
    )
    held = {}
    query = "--query-compute-apps=pid,gpu_uuid,used_memory"
    apps = sh("nvidia-smi", query, "--format=csv,noheader,nounits")
    for line in apps.splitlines():
        try:
            pid, uuid, mib = (p.strip() for p in line.split(","))
            with open(f"/proc/{pid}/cgroup") as f:
                unit = f.read().strip().rsplit("/", 1)[-1]
            cards = held.setdefault(unit, {})
            cards[index[uuid]] = cards.get(index[uuid], 0) + int(mib)
        except (OSError, ValueError, KeyError):
            continue
    return held


def observe(workload, launchctl_table, held):
    scope, _, unit = workload["unit"].partition("/")
    if scope == "launchd":
        pid = launchd_pid(launchctl_table, unit)
        active = pid is not None
    else:
        user = ["--user"] if scope == "user" else []
        show = sh("systemctl", *user, "show", unit, "-p", "ActiveState", "-p", "MainPID")
        active, pid = parse_show(show)
    port = workload["port"]
    up = listening(port)
    loaded = None
    if up:
        ps, running = get_json(port, "/api/ps"), get_json(port, "/running")
        loaded = names(ps, "models", "name")
        if loaded is None:
            loaded = names(running, "running", "model")
    mib = held.get(unit, {})
    return {
        "name": workload["name"],
        "unit": workload["unit"],
        "active": active,
        "pid": pid,
        "port": port,
        "listening": up,
        "models": names(get_json(port, "/v1/models"), "data", "id") if up else None,
        "gpus": sorted(mib),
        "loaded": loaded,
        "gpu_mib": {str(k): v for k, v in sorted(mib.items())},
    }


STABLE = ("name", "unit", "active", "pid", "port", "listening", "models", "gpus")


def digest(records):
    stable = sorted(([r[k] for k in STABLE] for r in records), key=lambda r: r[0])
    return "sha256:" + hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def compare(agent_workloads, records):
    """Where the agent's status and the direct observation differ."""
    out = []
    seen = {r["name"]: r for r in records}
    for w in agent_workloads:
        r = seen[w["name"]]
        # A vendor app has no launchd job of its own: a listening port counts.
        up = r["active"] or r["listening"]
        if (w.get("process") == "Running") != up:
            seen_now = f"active={r['active']} listening={r['listening']}"
            out.append(f"{w['name']}: agent says {w.get('process')}, observed {seen_now}")
        if w.get("readiness") == "Ready" and not r["listening"]:
            out.append(f"{w['name']}: agent says Ready, port {r['port']} does not listen")
        # shortcut: only an endpoint that lists one id is compared. For a
        # service with many models (Ollama, LM Studio, llama-swap) the agent
        # records the service identity, not a model id. Upgrade when
        # `status --json` carries the engine kind.
        one = r["models"] if r["models"] and len(r["models"]) == 1 else None
        if w.get("served_model_verified") and one and w.get("served_model") not in one:
            served = w.get("served_model")
            out.append(f"{w['name']}: agent says it serves {served}, endpoint lists {one}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--status", help="file with the agent's `status --json` (default: stdin)")
    args = ap.parse_args()
    try:
        status = json.load(open(args.status) if args.status else sys.stdin)
        workloads = status["workloads"]
    except (OSError, ValueError, KeyError) as e:
        print(f"cannot read the agent status: {e}", file=sys.stderr)
        return 2
    table = sh("launchctl", "list") if status.get("platform") == "Macos" else ""
    held = gpu_holdings() if status.get("platform") == "Linux" else {}
    records = [observe(w, table, held) for w in workloads]
    bad = compare(workloads, records)
    json.dump(
        {
            "host": status.get("host"),
            # timezone.utc, not datetime.UTC: the Macs run this on Python 3.9.
            "taken_at": datetime.datetime.now(datetime.timezone.utc).isoformat(  # noqa: UP017
                timespec="seconds"
            ),
            "inventory_revision": status.get("inventory_revision"),
            "digest": digest(records),
            "workloads": records,
            "disagreements": bad,
        },
        sys.stdout,
        indent=1,
    )
    print()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
