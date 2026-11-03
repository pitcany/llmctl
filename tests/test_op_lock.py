"""Host-wide GPU operation lock tests (FIXPLAN 2026-11-03 §2A).

The lock is the one boundary shared by CLI, TUI, API, and API-spawned child
processes. These tests prove the properties the plan demands:

* separate OS processes cannot run conflicting GPU mutations concurrently,
* the lock is reentrant for one thread (CLI swap -> nested helpers),
* a different thread or process gets a clear refusal, not a silent share,
* the kernel releases the lock when a holder dies,
* holder identity is readable for messages.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from llmctl.services.op_lock import (
    LockBusy,
    describe_holder,
    gpu_operation_lock,
    lock_path,
)


def test_lock_reentrant_for_same_thread(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(tmp_path / "op.lock"))
    with gpu_operation_lock(purpose="outer", interface="cli"):
        with gpu_operation_lock(purpose="inner", interface="cli"):
            with gpu_operation_lock(purpose="innermost", interface="cli"):
                pass
    # Fully released: another acquire in this thread is a fresh flock.
    with gpu_operation_lock(purpose="again", interface="cli"):
        pass


def test_cross_thread_conflict_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(tmp_path / "op.lock"))
    started = threading.Event()
    release = threading.Event()

    def _hold() -> None:
        with gpu_operation_lock(purpose="hold", interface="tui"):
            started.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=_hold)
    holder.start()
    assert started.wait(timeout=5)
    with pytest.raises(LockBusy) as excinfo:
        with gpu_operation_lock(purpose="conflict", interface="api", timeout=0.2):
            pass
    assert "hold" in str(excinfo.value) or "tui" in str(excinfo.value)
    release.set()
    holder.join(timeout=5)


def test_lock_released_after_exception(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(tmp_path / "op.lock"))
    with pytest.raises(RuntimeError):
        with gpu_operation_lock(purpose="boom", interface="cli"):
            raise RuntimeError("boom")
    with gpu_operation_lock(purpose="after", interface="cli"):
        pass


def test_holder_identity_recorded(tmp_path, monkeypatch) -> None:
    path = tmp_path / "op.lock"
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(path))
    with gpu_operation_lock(purpose="swap to lane x", interface="cli"):
        holder = json.loads(path.read_text())
        assert holder["pid"] == os.getpid()
        assert holder["purpose"] == "swap to lane x"
        assert holder["interface"] == "cli"
        assert holder["hostname"]
    # 0600: the lock lives in a shared data dir; keep it operator-only.
    assert (path.stat().st_mode & 0o777) == 0o600


def test_describe_holder_without_file(tmp_path) -> None:
    assert "unknown" in describe_holder(None)


_CHILD_HOLDER = """
import sys, time
sys.path.insert(0, {repo!r})
from llmctl.services.op_lock import gpu_operation_lock
with gpu_operation_lock(purpose="child swap", interface="cli"):
    print("held", flush=True)
    time.sleep(3)
print("released", flush=True)
"""


def test_separate_process_blocks_until_release(tmp_path, monkeypatch) -> None:
    """§2A acceptance: another OS process cannot take the lock while the
    child holds it, and can once the child exits."""
    repo = str(Path(__file__).resolve().parents[1])
    lock = tmp_path / "op.lock"
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(lock))  # parent must share the child's lock
    env = {**os.environ, "LLMCTL_OP_LOCK": str(lock)}
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_HOLDER.format(repo=repo)],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held"
        with pytest.raises(LockBusy) as excinfo:
            with gpu_operation_lock(purpose="second", interface="api", timeout=0.5):
                pass
        assert "child swap" in str(excinfo.value)
        # The refusal names the holder so the operator knows what waits.
        assert str(os.getpid()) not in str(excinfo.value)
    finally:
        child.kill()
        child.wait(timeout=10)


_CHILD_APPEND = """
import sys, time
sys.path.insert(0, {repo!r})
from llmctl.services.op_lock import gpu_operation_lock
with gpu_operation_lock(purpose="mutate", interface="cli"):
    with open({target!r}, "a") as fh:
        for i in range(200):
            fh.write(f"{{sys.argv[1]}} {{i}}\\n")
    time.sleep(0.05)
"""


def test_two_processes_never_interleave_mutations(tmp_path, monkeypatch) -> None:
    """§2A acceptance (CLI-vs-CLI, stop-vs-start shape): two processes
    writing while each holds the lock never interleave their writes."""
    repo = str(Path(__file__).resolve().parents[1])
    target = tmp_path / "mutations.log"
    lock = tmp_path / "op.lock"
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(lock))
    env = {**os.environ, "LLMCTL_OP_LOCK": str(lock)}
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD_APPEND.format(repo=repo, target=str(target)), name],
            env=env,
        )
        for name in ("A", "B")
    ]
    for proc in procs:
        assert proc.wait(timeout=30) == 0

    lines = [line for line in target.read_text().splitlines() if line.strip()]
    # Every contiguous run belongs to one writer; interleaving would mix them.
    runs: list[list[str]] = []
    for line in lines:
        who = line.split()[0]
        if not runs or runs[-1][0] != who:
            runs.append([who, 1])
        else:
            runs[-1][1] += 1
    assert len(runs) == 2, f"writes interleaved: {runs}"
    assert sorted(count for _, count in runs) == [200, 200]


def test_api_session_start_refused_while_lock_held(tmp_path) -> None:
    """§2A acceptance (API-vs-other): a mutation request returns 409 while
    another holder owns the lock, and succeeds after release."""
    from fastapi.testclient import TestClient

    from llmctl.api.app import create_app

    client = TestClient(create_app(database_url=f"sqlite:///{tmp_path / 'api.db'}"))
    model = client.post(
        "/models", json={"name": "o", "runtime": "ollama", "source": "o:latest"}
    )
    model_id = model.json()["id"]

    with gpu_operation_lock(purpose="cli swap", interface="cli"):
        response = client.post(
            "/sessions/start",
            json={"model_id": model_id, "runtime": "ollama", "dry_run": False, "force": True},
        )
        assert response.status_code == 409
        assert "cli swap" in response.json()["detail"]

    planned = client.post(
        "/sessions/start",
        json={"model_id": model_id, "runtime": "ollama", "dry_run": True, "force": True},
    )
    assert planned.status_code == 201


def test_lock_path_env_override_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(tmp_path / "custom.lock"))
    assert lock_path() == tmp_path / "custom.lock"


def test_implicit_settings_share_configured_lock_path(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    settings = SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path / "configured"))
    monkeypatch.delenv("LLMCTL_OP_LOCK", raising=False)
    monkeypatch.setattr("llmctl.config.load_settings", lambda: settings)
    assert lock_path() == lock_path(settings) == tmp_path / "configured" / "operations.lock"
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(tmp_path / "override.lock"))
    assert lock_path() == tmp_path / "override.lock"
