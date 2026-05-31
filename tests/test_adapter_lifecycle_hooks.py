"""Verify the pre/post-start hook plumbing on :class:`VLLMSystemdAdapter`.

Phase 3 added a stackable hook mechanism so integrations (hermes verify,
harbor preflight) can run at the right lifecycle moments without
entangling the adapter with their specifics. These tests pin:

* Hook ordering (pre-start before write_env; post-start after readiness)
* Skip semantics (no post-start when readiness fails or restart fails)
* Stacking (multiple hooks run in declaration order)
"""

from __future__ import annotations

import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from llmctl.adapters.vllm_systemd import VLLMSystemdAdapter
from llmctl.config import ManagedUnitConfig
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


def _systemctl_ok(*, restart_returncode: int = 0) -> SystemctlRunner:
    """Build a SystemctlRunner that scripts a launcher-marker pass + restart."""
    def fake(argv: list[str]) -> _FakeCompleted:
        verb = argv[-2]
        if verb == "cat":
            return _FakeCompleted(stdout="ExecStart=/bin/vllm-launcher.sh\n")
        if verb == "is-active":
            return _FakeCompleted(stdout="inactive\n")
        if verb == "restart":
            return _FakeCompleted(returncode=restart_returncode)
        return _FakeCompleted()

    runner = SystemctlRunner(runner=fake)
    runner.available = lambda: True  # type: ignore[method-assign]
    return runner


def _http_get_ready() -> Callable[[str, float], object]:
    """An HTTP getter that always succeeds."""
    return lambda url, timeout: object()


def _http_get_never_ready() -> Callable[[str, float], object]:
    """An HTTP getter that always raises ``URLError``."""
    def get(url: str, timeout: float) -> object:
        raise urllib.error.URLError("never ready")

    return get


def test_pre_start_runs_before_write_env(tmp_path: Path) -> None:
    """A pre-start hook seeing the spec should fire before we write the env file."""
    timeline: list[str] = []

    def hook(spec: VLLMLaunchSpec) -> None:
        timeline.append("pre")
        assert spec.served_name == "test"
        # env file does not exist yet at pre-start time
        assert not (tmp_path / "env").exists()

    adapter = VLLMSystemdAdapter(
        ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="vllm-launcher.sh"),
        env_file_path=tmp_path / "env",
        systemctl=_systemctl_ok(),
        sleep=lambda _: None,
        http_get=_http_get_ready(),
        pre_start_hooks=[hook],
    )
    adapter.restart_with_spec(
        VLLMLaunchSpec(model="m", served_name="test"),
        wait_for_ready=False,
    )
    assert timeline == ["pre"]
    # env file exists after the call
    assert (tmp_path / "env").exists()


def test_post_start_runs_after_readiness(tmp_path: Path) -> None:
    """Post-start hooks should fire only after the readiness poll succeeds."""
    timeline: list[str] = []

    def pre(spec: VLLMLaunchSpec) -> None:
        timeline.append("pre")

    def post(spec: VLLMLaunchSpec) -> None:
        timeline.append("post")
        assert spec.port == 8003

    adapter = VLLMSystemdAdapter(
        ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="vllm-launcher.sh"),
        env_file_path=tmp_path / "env",
        systemctl=_systemctl_ok(),
        sleep=lambda _: None,
        http_get=_http_get_ready(),
        pre_start_hooks=[pre],
        post_start_hooks=[post],
    )
    result = adapter.restart_with_spec(VLLMLaunchSpec(model="m", served_name="s"))
    assert result.ready is True
    assert timeline == ["pre", "post"]


def test_post_start_skipped_when_not_ready(tmp_path: Path) -> None:
    """Readiness timeout -> post-start hooks must not fire."""
    timeline: list[str] = []
    clock_state = [0.0]

    def clock() -> float:
        clock_state[0] += 6.0
        return clock_state[0]

    adapter = VLLMSystemdAdapter(
        ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="vllm-launcher.sh"),
        env_file_path=tmp_path / "env",
        systemctl=_systemctl_ok(),
        clock=clock,
        sleep=lambda _: None,
        http_get=_http_get_never_ready(),
        post_start_hooks=[lambda _spec: timeline.append("post")],
    )
    result = adapter.restart_with_spec(
        VLLMLaunchSpec(model="m", served_name="s"),
        timeout_s=10.0,
        poll_interval_s=1.0,
    )
    assert result.ready is False
    assert timeline == []  # post-start did not fire


def test_post_start_skipped_when_restart_fails(tmp_path: Path) -> None:
    """Restart-call failure should short-circuit both poll and post-start."""
    timeline: list[str] = []
    adapter = VLLMSystemdAdapter(
        ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="vllm-launcher.sh"),
        env_file_path=tmp_path / "env",
        systemctl=_systemctl_ok(restart_returncode=1),
        sleep=lambda _: None,
        http_get=_http_get_ready(),
        pre_start_hooks=[lambda _spec: timeline.append("pre")],
        post_start_hooks=[lambda _spec: timeline.append("post")],
    )
    result = adapter.restart_with_spec(VLLMLaunchSpec(model="m", served_name="s"))
    assert result.ready is False
    # pre-start fired (it precedes restart); post-start did not
    assert timeline == ["pre"]


def test_post_start_fires_when_wait_for_ready_false(tmp_path: Path) -> None:
    """When the caller opts out of readiness polling, post-start still fires
    (it's the caller's signal that they've intentionally skipped verification)."""
    timeline: list[str] = []
    adapter = VLLMSystemdAdapter(
        ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="vllm-launcher.sh"),
        env_file_path=tmp_path / "env",
        systemctl=_systemctl_ok(),
        post_start_hooks=[lambda _spec: timeline.append("post")],
    )
    result = adapter.restart_with_spec(
        VLLMLaunchSpec(model="m", served_name="s"),
        wait_for_ready=False,
    )
    assert result.ready is True
    assert timeline == ["post"]


def test_multiple_hooks_run_in_order(tmp_path: Path) -> None:
    """Hooks fire in declaration order — important when one hook's output
    informs the next (e.g. preflight stop then Harbor pin check)."""
    timeline: list[str] = []
    pre_hooks = [
        lambda _s: timeline.append("pre-1"),
        lambda _s: timeline.append("pre-2"),
        lambda _s: timeline.append("pre-3"),
    ]
    post_hooks = [
        lambda _s: timeline.append("post-1"),
        lambda _s: timeline.append("post-2"),
    ]
    adapter = VLLMSystemdAdapter(
        ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="vllm-launcher.sh"),
        env_file_path=tmp_path / "env",
        systemctl=_systemctl_ok(),
        sleep=lambda _: None,
        http_get=_http_get_ready(),
        pre_start_hooks=pre_hooks,
        post_start_hooks=post_hooks,
    )
    adapter.restart_with_spec(VLLMLaunchSpec(model="m", served_name="s"))
    assert timeline == ["pre-1", "pre-2", "pre-3", "post-1", "post-2"]


def test_pre_start_exception_aborts_lifecycle(tmp_path: Path) -> None:
    """A pre-start exception must propagate — the lifecycle is canceled.

    Integrations that detect a fatal condition (e.g. WebUI pin orphan
    the user explicitly rejected) should raise; the adapter shouldn't
    swallow it and proceed silently.
    """
    written: list[str] = []
    adapter = VLLMSystemdAdapter(
        ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="vllm-launcher.sh"),
        env_file_path=tmp_path / "env",
        systemctl=_systemctl_ok(),
        sleep=lambda _: None,
        http_get=_http_get_ready(),
        pre_start_hooks=[
            lambda _spec: written.append("first-fired"),
            lambda _spec: (_ for _ in ()).throw(RuntimeError("user aborted")),
            lambda _spec: written.append("should-not-fire"),
        ],
    )
    with pytest.raises(RuntimeError, match="user aborted"):
        adapter.restart_with_spec(VLLMLaunchSpec(model="m", served_name="s"))
    assert written == ["first-fired"]
    # env file was not written
    assert not (tmp_path / "env").exists()


def test_no_hooks_by_default(tmp_path: Path) -> None:
    """An adapter constructed without hooks behaves identically to Phase 1."""
    adapter = VLLMSystemdAdapter(
        ManagedUnitConfig(unit_name="vllm-tp", launcher_marker="vllm-launcher.sh"),
        env_file_path=tmp_path / "env",
        systemctl=_systemctl_ok(),
        sleep=lambda _: None,
        http_get=_http_get_ready(),
    )
    result = adapter.restart_with_spec(VLLMLaunchSpec(model="m", served_name="s"))
    assert result.ready is True
    assert adapter.pre_start_hooks == []
    assert adapter.post_start_hooks == []
