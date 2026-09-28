"""A vLLM start that crashes stops the readiness wait at once.

2026-09-28: a Muse preset died 6 s into startup, but the adapter polled HTTP
for the whole readiness timeout while systemd (Restart=on-failure) relaunched
it every 15 s. That used up StartLimitBurst, so the rollback's start of the
previous preset was refused and the tutor stayed down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from llmctl.adapters.vllm_systemd import VLLMSystemdAdapter, unit_crash
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.integrations.vllm_env import VLLMLaunchSpec


@dataclass
class _Done:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/opt/python")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/tmp/hf")


def test_unit_crash_reads_systemd_states() -> None:
    assert unit_crash({}, 0) is None
    assert (
        unit_crash({"ActiveState": "activating", "SubState": "start", "NRestarts": "4"}, 4) is None
    )
    assert "failed" in unit_crash({"ActiveState": "failed"}, 0)
    assert "restart it" in unit_crash({"SubState": "auto-restart"}, 0)
    assert "1 time" in unit_crash({"ActiveState": "active", "NRestarts": "5"}, 4)


def _adapter(tmp_path: Path, show_outputs: list[str]):
    shows = iter(show_outputs)
    sleeps: list[float] = []

    def fake(argv: list[str]) -> _Done:
        if "show" in argv:
            return _Done(stdout=next(shows))
        if "cat" in argv:
            return _Done(stdout="ExecStart=/bin/vllm-launcher.sh\n")
        return _Done()

    def http_get(url: str, timeout: float) -> object:
        raise OSError("connection refused")

    clock = iter(range(0, 10_000, 5))
    runner = SystemctlRunner(runner=fake)
    runner.available = lambda: True  # type: ignore[method-assign]
    adapter = VLLMSystemdAdapter(
        env_file_path=tmp_path / "vllm-tp.env",
        systemctl=runner,
        clock=lambda: float(next(clock)),
        sleep=sleeps.append,
        http_get=http_get,
    )
    return adapter, sleeps


def _spec() -> VLLMLaunchSpec:
    return VLLMLaunchSpec(model="m", served_name="m", port=8003)


def test_a_crashed_unit_ends_the_wait_with_the_reason(tmp_path: Path) -> None:
    adapter, sleeps = _adapter(
        tmp_path,
        [
            "ActiveState=active\nSubState=running\nNRestarts=0\n",  # before the restart
            "ActiveState=activating\nSubState=start\nNRestarts=0\n",  # still loading
            "ActiveState=activating\nSubState=auto-restart\nNRestarts=0\n",  # died
        ],
    )
    result = adapter.restart_with_spec(_spec(), timeout_s=900, poll_interval_s=3)
    assert result.ready is False
    assert "about to restart it" in (result.error or "")
    assert len(sleeps) <= 3, "kept waiting after the unit was known dead"


def test_restarts_from_before_the_swap_are_not_a_crash(tmp_path: Path) -> None:
    adapter, _ = _adapter(
        tmp_path, ["NRestarts=2\n"] + ["ActiveState=activating\nNRestarts=2\n"] * 50
    )
    result = adapter.restart_with_spec(_spec(), timeout_s=60, poll_interval_s=3)
    assert result.ready is False
    assert "did not become ready within 60s" in (result.error or "")
