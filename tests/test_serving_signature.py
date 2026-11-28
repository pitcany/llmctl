"""The pure parts of ``scripts/serving_signature.py`` (Rust M6a item 7)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "serving_signature", Path(__file__).resolve().parents[1] / "scripts" / "serving_signature.py"
)
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)


def _record(**over):
    base = {
        "name": "engine", "unit": "user/engine.service", "active": True, "pid": 10, "port": 8003,
        "listening": True, "models": ["m"], "gpus": [0], "loaded": None, "gpu_mib": {"0": 100},
    }
    return {**base, **over}


def test_parse_show():
    assert sig.parse_show("MainPID=4022288\nActiveState=active\n") == (True, 4022288)
    assert sig.parse_show("MainPID=0\nActiveState=inactive\n") == (False, None)
    assert sig.parse_show("") == (False, None)


def test_launchd_pid_exact_suffix_and_not_running():
    table = (
        "PID\tStatus\tLabel\n"
        "-\t0\tcom.ollama.ollama\n"
        "67153\t0\tcom.pitcany.llama-swap\n"
        "67246\t0\tapplication.ai.elementlabs.lmstudio.2797.2798\n"
        "1\t0\tcom.pitcany.llama-swap-other\n"
    )
    assert sig.launchd_pid(table, "com.pitcany.llama-swap") == 67153
    assert sig.launchd_pid(table, "application.ai.elementlabs.lmstudio") == 67246
    assert sig.launchd_pid(table, "com.ollama.ollama") is None
    assert sig.launchd_pid(table, "com.pitcany.ds4") is None


def test_names_refuses_a_wrong_shape():
    assert sig.names({"data": [{"id": "b"}, {"id": "a"}]}, "data", "id") == ["a", "b"]
    assert sig.names({"error": "unauthorized"}, "data", "id") is None
    assert sig.names(None, "data", "id") is None


def test_digest_ignores_what_changes_by_itself():
    a = [_record(), _record(name="other", pid=None, active=False)]
    b = [_record(name="other", pid=None, active=False), _record(loaded=["x"], gpu_mib={"0": 999})]
    assert sig.digest(a) == sig.digest(b)
    assert sig.digest(a) != sig.digest([_record(pid=11), a[1]])


def test_compare_reports_each_difference():
    agent = [
        {"name": "engine", "process": "Running", "readiness": "Ready",
         "served_model": "m", "served_model_verified": True},
    ]
    assert sig.compare(agent, [_record()]) == []
    # A vendor app with no launchd job still counts as running when its port listens.
    assert sig.compare(agent, [_record(active=False, pid=None)]) == []
    down = sig.compare(agent, [_record(active=False, pid=None, listening=False, models=None)])
    assert len(down) == 2 and "agent says Running" in down[0] and "does not listen" in down[1]
    assert "endpoint lists ['other']" in sig.compare(agent, [_record(models=["other"])])[0]


def test_compare_uses_the_engine_kind_and_the_pid_of_a_0_2_0_agent():
    vllm = {"name": "engine", "process": "Running", "readiness": "Ready", "served_model": "m",
            "served_model_verified": True, "engine_kind": "vllm", "pid": 10}
    assert sig.compare([vllm], [_record()]) == []
    # A vLLM endpoint that lists two ids is compared now.
    assert "endpoint lists" in sig.compare([vllm], [_record(models=["x", "y"])])[0]
    assert "says pid 10, the service manager says 11" in sig.compare([vllm], [_record(pid=11)])[0]
    ollama = {**vllm, "engine_kind": "ollama", "served_model": "ollama"}
    assert sig.compare([ollama], [_record(models=["qwen3-coder:30b"])]) == []

