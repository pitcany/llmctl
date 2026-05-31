"""Unit tests for :mod:`llmctl.integrations.systemctl`.

Tests the argv shape, sudo-prepending behaviour, and the convenience
verbs (start/stop/restart/is_active/cat/try_stop). No real systemctl
invocation — the runner is injected so we can assert exactly what
would be executed.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import pytest

from llmctl.integrations.systemctl import SystemctlRunner, SystemctlVerb


@dataclass
class _FakeCompleted:
    """Lightweight stand-in for ``subprocess.CompletedProcess``."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class _Recorder:
    """Capture the argv each call would have invoked."""

    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, argv: list[str]) -> _FakeCompleted:
        self.calls.append(list(argv))
        return _FakeCompleted(self.returncode, self.stdout, self.stderr)


def test_write_verbs_prepend_sudo() -> None:
    rec = _Recorder()
    runner = SystemctlRunner(runner=rec)
    runner.start("vllm-tp")
    runner.stop("vllm-tp")
    runner.restart("vllm-tp")
    assert rec.calls == [
        ["sudo", "systemctl", "start", "vllm-tp"],
        ["sudo", "systemctl", "stop", "vllm-tp"],
        ["sudo", "systemctl", "restart", "vllm-tp"],
    ]


def test_read_only_verbs_skip_sudo() -> None:
    rec = _Recorder(stdout="active\n")
    runner = SystemctlRunner(runner=rec)
    runner.is_active("vllm-tp")
    runner.cat("vllm-tp")
    runner.run(SystemctlVerb.STATUS, "vllm-tp")
    assert rec.calls == [
        ["systemctl", "is-active", "vllm-tp"],
        ["systemctl", "cat", "vllm-tp"],
        ["systemctl", "status", "vllm-tp"],
    ]


def test_is_active_parses_stdout() -> None:
    rec = _Recorder(stdout="active\n")
    runner = SystemctlRunner(runner=rec)
    assert runner.is_active("vllm-tp") is True

    rec = _Recorder(stdout="inactive\n")
    runner = SystemctlRunner(runner=rec)
    assert runner.is_active("vllm-tp") is False


def test_cat_returns_empty_on_error() -> None:
    rec = _Recorder(returncode=1, stderr="No such unit\n")
    runner = SystemctlRunner(runner=rec)
    assert runner.cat("does-not-exist") == ""


def test_try_stop_returns_false_when_inactive() -> None:
    """Inactive units shouldn't trigger a stop call."""

    class StatefulRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def __call__(self, argv: list[str]) -> _FakeCompleted:
            self.calls.append(list(argv))
            if argv[-2] == "is-active":
                return _FakeCompleted(stdout="inactive\n")
            return _FakeCompleted()

    rec = StatefulRunner()
    runner = SystemctlRunner(runner=rec)
    assert runner.try_stop("vllm-tp") is False
    assert rec.calls == [["systemctl", "is-active", "vllm-tp"]]  # no stop issued


def test_try_stop_issues_stop_when_active() -> None:
    """Active units should be stopped, returning True."""
    state = {"active": True}

    def runner(argv: list[str]) -> _FakeCompleted:
        if argv[-2] == "is-active":
            return _FakeCompleted(stdout="active\n" if state["active"] else "inactive\n")
        if argv[-2] == "stop":
            state["active"] = False
            return _FakeCompleted()
        return _FakeCompleted()

    r = SystemctlRunner(runner=runner)
    assert r.try_stop("vllm-tp") is True


def test_result_ok_reflects_returncode() -> None:
    rec = _Recorder(returncode=0)
    result = SystemctlRunner(runner=rec).start("vllm-tp")
    assert result.ok is True

    rec = _Recorder(returncode=5, stderr="Permission denied\n")
    result = SystemctlRunner(runner=rec).start("vllm-tp")
    assert result.ok is False
    assert "Permission denied" in result.stderr


def test_extra_args_appended() -> None:
    rec = _Recorder()
    runner = SystemctlRunner(runner=rec)
    runner.run(SystemctlVerb.START, "vllm-tp", "--no-block")
    assert rec.calls[-1] == ["sudo", "systemctl", "start", "vllm-tp", "--no-block"]


def test_default_runner_uses_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no runner is injected, real ``subprocess.run`` is invoked.

    We don't want to actually shell out, so we patch ``subprocess.run``
    to verify the call path.
    """
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    SystemctlRunner().start("vllm-tp")
    assert captured["argv"] == ["sudo", "systemctl", "start", "vllm-tp"]
    # capture_output/text/check are the safety contract
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is False
