"""Unit tests for :mod:`llmctl.adapters.vllm_systemd`.

Verifies the full lifecycle of the managed-systemd adapter without
touching real systemd, real filesystem (beyond ``tmp_path``), or real
HTTP. The systemctl runner, clock, sleep, and HTTP getter are all
injectable.
"""

from __future__ import annotations

import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from llmctl.adapters.vllm_systemd import (
    LegacyUnitError,
    VLLMSystemdAdapter,
)
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.integrations.vllm_env import VLLMLaunchSpec


@dataclass
class _FakeCompleted:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


@pytest.fixture(autouse=True)
def _pin_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMCTL_PYTHON_ROOT", "/opt/python")
    monkeypatch.setenv("LLMCTL_CUDA_ROOT", "/usr/local/cuda")
    monkeypatch.setenv("HF_HOME", "/tmp/hf")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)


def _make_systemctl(
    *,
    cat_body: str = "ExecStart=/bin/vllm-launcher.sh\n",
    is_active: str = "inactive",
    restart_returncode: int = 0,
    available: bool = True,
) -> tuple[SystemctlRunner, list[list[str]]]:
    """Build an injected SystemctlRunner with scripted responses."""
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> _FakeCompleted:
        calls.append(list(argv))
        # argv is [systemctl, verb, unit] for read; [sudo, systemctl, verb, unit] for write
        verb = argv[-2]
        if verb == "is-active":
            return _FakeCompleted(stdout=is_active + "\n")
        if verb == "cat":
            return _FakeCompleted(stdout=cat_body)
        if verb == "restart":
            return _FakeCompleted(
                returncode=restart_returncode,
                stderr="boom" if restart_returncode else "",
            )
        return _FakeCompleted()

    runner = SystemctlRunner(runner=fake)
    # `available()` consults shutil.which; force True/False via subclass-ish wrapper
    runner.available = lambda: available  # type: ignore[method-assign]
    return runner, calls


def _make_http_get(*, fails: int = 0) -> tuple[Callable[[str, float], object], dict[str, int]]:
    """Return an http_get that fails ``fails`` times then succeeds.

    Counts each call so the test can assert poll cadence.
    """
    state = {"calls": 0}

    def get(url: str, timeout: float) -> object:
        state["calls"] += 1
        if state["calls"] <= fails:
            raise urllib.error.URLError("not ready")
        return object()

    return get, state


def test_write_env_creates_parent_dir(tmp_path: Path) -> None:
    env_path = tmp_path / "nested" / "services" / "vllm-tp.env"
    adapter = VLLMSystemdAdapter(env_file_path=env_path)
    spec = VLLMLaunchSpec(model="m", served_name="s")
    path, body = adapter.write_env(spec)
    assert path == env_path
    assert env_path.is_file()
    assert env_path.read_text() == body


def test_restart_with_spec_happy_path(tmp_path: Path) -> None:
    runner, calls = _make_systemctl()
    get, state = _make_http_get()
    sleeps: list[float] = []

    adapter = VLLMSystemdAdapter(
        env_file_path=tmp_path / "vllm-tp.env",
        systemctl=runner,
        sleep=sleeps.append,
        http_get=get,
    )
    spec = VLLMLaunchSpec(model="m", served_name="s")
    result = adapter.restart_with_spec(spec)

    assert result.ready is True
    assert result.error is None
    assert (tmp_path / "vllm-tp.env").read_text() == result.env_body
    # systemctl call sequence: cat (guard) then restart
    verbs = [c[-2] for c in calls]
    assert verbs == ["cat", "restart"]
    # http_get was invoked exactly once (succeeds on first try)
    assert state["calls"] == 1
    # The initial 5s pre-poll sleep is always issued
    assert sleeps and sleeps[0] == 5.0


def test_restart_with_spec_polls_until_ready(tmp_path: Path) -> None:
    runner, _ = _make_systemctl()
    get, state = _make_http_get(fails=3)
    clock_ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    sleeps: list[float] = []

    adapter = VLLMSystemdAdapter(
        env_file_path=tmp_path / "env",
        systemctl=runner,
        clock=lambda: next(clock_ticks),
        sleep=sleeps.append,
        http_get=get,
    )
    result = adapter.restart_with_spec(VLLMLaunchSpec(model="m", served_name="s"))
    assert result.ready is True
    # 3 failing pings + 1 successful = 4 calls
    assert state["calls"] == 4
    # Initial 5s + 3 retry sleeps
    assert sleeps[0] == 5.0
    assert sleeps.count(5.0) >= 4  # default poll_interval_s is 5


def test_restart_with_spec_times_out_when_never_ready(tmp_path: Path) -> None:
    runner, _ = _make_systemctl()

    def always_fail(url: str, timeout: float) -> object:
        raise urllib.error.URLError("never ready")

    clock_state = [0.0]

    def clock() -> float:
        clock_state[0] += 6.0  # each poll pretends 6s passed
        return clock_state[0]

    adapter = VLLMSystemdAdapter(
        env_file_path=tmp_path / "env",
        systemctl=runner,
        clock=clock,
        sleep=lambda s: None,
        http_get=always_fail,
    )
    result = adapter.restart_with_spec(
        VLLMLaunchSpec(model="m", served_name="s"),
        timeout_s=10.0,
        poll_interval_s=1.0,
    )
    assert result.ready is False
    assert result.error is not None
    assert "did not become ready" in result.error


def test_restart_failure_surfaces_error_with_stderr(tmp_path: Path) -> None:
    runner, _ = _make_systemctl(restart_returncode=1)
    adapter = VLLMSystemdAdapter(
        env_file_path=tmp_path / "env",
        systemctl=runner,
    )
    result = adapter.restart_with_spec(
        VLLMLaunchSpec(model="m", served_name="s"),
        wait_for_ready=False,
    )
    assert result.ready is False
    assert result.error is not None
    assert "systemctl restart vllm-tp failed" in result.error
    assert "boom" in result.error


def test_legacy_unit_guard_raises(tmp_path: Path) -> None:
    """The pre-launcher ExecStart unit must be refused."""
    runner, _ = _make_systemctl(cat_body="ExecStart=/usr/bin/python -m vllm ...\n")
    adapter = VLLMSystemdAdapter(env_file_path=tmp_path / "env", systemctl=runner)
    with pytest.raises(LegacyUnitError, match="launcher-based unit"):
        adapter.ensure_launcher_unit()
    with pytest.raises(LegacyUnitError):
        adapter.restart_with_spec(VLLMLaunchSpec(model="m", served_name="s"), wait_for_ready=False)


def test_legacy_unit_guard_passes_for_launcher_unit(tmp_path: Path) -> None:
    runner, _ = _make_systemctl(cat_body="ExecStart=/home/yannik/AI/scripts/vllm-launcher.sh\n")
    adapter = VLLMSystemdAdapter(env_file_path=tmp_path / "env", systemctl=runner)
    adapter.ensure_launcher_unit()  # no raise


def test_legacy_unit_guard_skipped_when_systemctl_unavailable(tmp_path: Path) -> None:
    """In containers without systemd, the guard should be a no-op."""
    runner, _ = _make_systemctl(cat_body="<<<does not matter>>>", available=False)
    adapter = VLLMSystemdAdapter(env_file_path=tmp_path / "env", systemctl=runner)
    adapter.ensure_launcher_unit()  # no raise


def test_legacy_unit_guard_skipped_when_unit_missing(tmp_path: Path) -> None:
    """Missing unit (cat returns empty) should not raise — the start
    call will surface a clearer error."""
    runner, _ = _make_systemctl(cat_body="")
    adapter = VLLMSystemdAdapter(env_file_path=tmp_path / "env", systemctl=runner)
    adapter.ensure_launcher_unit()  # no raise


def test_stop_proxies_to_try_stop(tmp_path: Path) -> None:
    runner, calls = _make_systemctl(is_active="active")
    adapter = VLLMSystemdAdapter(env_file_path=tmp_path / "env", systemctl=runner)
    assert adapter.stop() is True
    verbs = [c[-2] for c in calls]
    assert "is-active" in verbs
    assert "stop" in verbs


def test_wait_for_ready_false_skips_polling(tmp_path: Path) -> None:
    runner, _ = _make_systemctl()

    def boom(url: str, timeout: float) -> object:
        raise AssertionError("http_get should not be invoked")

    adapter = VLLMSystemdAdapter(
        env_file_path=tmp_path / "env",
        systemctl=runner,
        sleep=lambda s: None,
        http_get=boom,
    )
    result = adapter.restart_with_spec(
        VLLMLaunchSpec(model="m", served_name="s"),
        wait_for_ready=False,
    )
    assert result.ready is True
    assert result.error is None
