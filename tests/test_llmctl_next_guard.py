"""The llmctl-next delegation guard (Rust M5b item 1, plan decision 2).

A write verb on a unit llmctl-next owns goes to the llmctl-next CLI; an
agent that is down or refuses is an honest failure, never a silent
fallback; unowned units keep the systemctl path; one setting reverses it.
The stand-in llmctl-next records its argv so the tests can see exactly
what the guard ran.
"""

from __future__ import annotations

import shlex
import stat
import subprocess
from pathlib import Path

import pytest

from llmctl.integrations.llmctl_next import NextDelegation, load_delegation
from llmctl.integrations.systemctl import SystemctlRunner, SystemctlVerb

STANDIN = """#!/usr/bin/env python3
import sys, pathlib, os
pathlib.Path(sys.argv[2]).open('a').write(' '.join(sys.argv[3:]) + '\\n')
os._exit(int(sys.argv[1]))
"""


@pytest.fixture()
def standin(tmp_path: Path) -> Path:
    """Stand-in llmctl-next: ``standin <rc> <marker> <guard argv...>``."""
    script = tmp_path / "standin"
    script.write_text(STANDIN)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _cmd(standin: Path, tmp_path: Path, rc: int) -> tuple[str, ...]:
    """A delegation cmd prefix that exits ``rc`` and logs the guard argv."""
    return (str(standin), str(rc), str(tmp_path / f"argv-{rc}.txt"))


def _argv(tmp_path: Path, rc: int) -> list[str]:
    return (tmp_path / f"argv-{rc}.txt").read_text().splitlines()


def _fake_systemctl(seen: list[list[str]]):
    def runner(argv: list[str]) -> subprocess.CompletedProcess:
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    return runner


def test_write_verb_on_owned_unit_is_delegated(tmp_path, standin):
    delegation = NextDelegation(
        enabled=True, cmd=_cmd(standin, tmp_path, 0), units={"vllm-engine.service": "vllm-engine"}
    )
    seen: list[list[str]] = []
    runner = SystemctlRunner(runner=_fake_systemctl(seen), next_delegation=delegation)

    result = runner.run(SystemctlVerb.START, "vllm-engine.service", user=True)

    assert result.ok, result.stderr
    assert seen == [], "systemctl must not run when the unit is delegated"
    assert _argv(tmp_path, 0) == ["run vllm-engine --wait"]

    # The settings map may key by the bare name too.
    bare = NextDelegation(
        enabled=True, cmd=_cmd(standin, tmp_path, 0), units={"vllm-engine": "vllm-engine"}
    )
    runner2 = SystemctlRunner(runner=_fake_systemctl(seen), next_delegation=bare)
    assert runner2.stop("vllm-engine.service").ok
    assert _argv(tmp_path, 0) == ["run vllm-engine --wait", "stop vllm-engine --wait"]


def test_refused_agent_down_is_an_honest_failure_not_a_fallback(tmp_path, standin):
    delegation = NextDelegation(
        enabled=True, cmd=_cmd(standin, tmp_path, 3), units={"gpt-oss-120b.service": "gpt-oss-120b"}
    )
    seen: list[list[str]] = []
    runner = SystemctlRunner(runner=_fake_systemctl(seen), next_delegation=delegation)

    result = runner.run(SystemctlVerb.STOP, "gpt-oss-120b.service")

    assert not result.ok and result.returncode == 3
    assert "refused before anything changed" in result.stderr, result.stderr
    assert "Not falling back to systemctl" in result.stderr
    assert "llmctl_next.enabled: false" in result.stderr  # the reversal lever
    assert seen == [], "a refusal must not silently write through systemctl"


def test_admitted_but_failed_operation_is_a_failure_with_detail(tmp_path, standin):
    delegation = NextDelegation(
        enabled=True, cmd=_cmd(standin, tmp_path, 7), units={"vllm-engine.service": "vllm-engine"}
    )
    runner = SystemctlRunner(
        runner=_fake_systemctl([]), next_delegation=delegation
    )
    result = runner.run(SystemctlVerb.RESTART, "vllm-engine.service")
    assert not result.ok and result.returncode == 7
    assert "did not succeed" in result.stderr


def test_unowned_unit_keeps_the_systemctl_path(tmp_path, standin):
    delegation = NextDelegation(
        enabled=True, cmd=_cmd(standin, tmp_path, 0), units={"vllm-engine.service": "vllm-engine"}
    )
    seen: list[list[str]] = []
    runner = SystemctlRunner(runner=_fake_systemctl(seen), next_delegation=delegation)

    assert runner.run(SystemctlVerb.START, "some-other-unit.service", user=True).ok
    assert seen and seen[0][:3] == ["systemctl", "--user", "start"], seen
    assert not (tmp_path / "argv-0.txt").exists(), "the stand-in must not run for an unowned unit"

    # Read verbs never delegate, even for owned units.
    assert runner.run(SystemctlVerb.IS_ACTIVE, "vllm-engine.service").ok
    assert len(seen) == 2 and seen[1][0] == "systemctl"


def test_disabled_or_absent_settings_delegate_nothing(tmp_path, monkeypatch):
    settings = tmp_path / "settings.yaml"
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path))

    settings.write_text("app:\n  name: llmctl\n")
    assert load_delegation().enabled is False

    seen: list[list[str]] = []
    runner = SystemctlRunner(runner=_fake_systemctl(seen))
    assert runner.stop("vllm-engine.service", user=True).ok
    assert seen[0][:3] == ["systemctl", "--user", "stop"]

    settings.write_text(
        "llmctl_next:\n  enabled: false\n  units: {vllm-engine.service: vllm-engine}\n"
    )
    assert load_delegation().enabled is False

    monkeypatch.setenv("LLMCTL_NEXT_DISABLED", "1")
    settings.write_text(
        "llmctl_next:\n  enabled: true\n  units: {vllm-engine.service: vllm-engine}\n"
    )
    assert load_delegation().enabled is False


def test_settings_shape_and_env_cmd_override(tmp_path, monkeypatch):
    settings = tmp_path / "settings.yaml"
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path))
    settings.write_text(
        "llmctl_next:\n"
        "  enabled: true\n"
        "  cmd: [/from/settings, --data-dir, /data]\n"
        "  units: {vllm-engine.service: vllm-engine}\n"
    )
    delegation = load_delegation()
    assert delegation.enabled
    assert delegation.cmd == ("/from/settings", "--data-dir", "/data")
    assert delegation.workload_for("vllm-engine.service") == "vllm-engine"

    monkeypatch.setenv("LLMCTL_NEXT_CMD", "/tmp/standin --data-dir /tmp/x")
    assert load_delegation().cmd == ("/tmp/standin", "--data-dir", "/tmp/x")


def test_cutover_routes_owned_writes_and_refuses_unowned_writes(tmp_path, monkeypatch, standin):
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path))
    (tmp_path / "settings.yaml").write_text(
        "llmctl_next:\n  mode: cutover\n  enabled: false\n  units: {engine.service: engine}\n"
    )
    monkeypatch.setenv("LLMCTL_NEXT_CMD", shlex.join(_cmd(standin, tmp_path, 0)))
    seen: list[list[str]] = []
    runner = SystemctlRunner(runner=_fake_systemctl(seen))

    assert runner.start("engine", user=True).ok
    refused = runner.restart("unowned.service", user=True)
    assert refused.returncode == 5
    assert "cutover" in refused.stderr and "llmctl-next restart unowned" in refused.stderr
    assert _argv(tmp_path, 0) == ["run engine --wait"]
    assert seen == []

    assert runner.run(SystemctlVerb.IS_ACTIVE, "engine", user=True).ok
    assert seen == [["systemctl", "--user", "is-active", "engine"]]


def test_cutover_refuses_environment_bypass(tmp_path, monkeypatch, standin):
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path))
    (tmp_path / "settings.yaml").write_text(
        "llmctl_next:\n  mode: cutover\n  enabled: true\n  units: {engine: engine}\n"
    )
    monkeypatch.setenv("LLMCTL_NEXT_DISABLED", "1")
    monkeypatch.setenv("LLMCTL_NEXT_CMD", shlex.join(_cmd(standin, tmp_path, 0)))
    seen: list[list[str]] = []
    runner = SystemctlRunner(runner=_fake_systemctl(seen))

    refused = runner.stop("engine", user=True)
    assert refused.returncode == 4
    assert "LLMCTL_NEXT_DISABLED=1" in refused.stderr and "cutover" in refused.stderr
    assert not (tmp_path / "argv-0.txt").exists()
    assert seen == []


def test_both_unit_name_forms_hit_the_map(tmp_path, standin):
    """Bare (`vllm-engine`) and suffixed (`vllm-engine.service`) lookups
    both resolve against either key form — a near-miss key must never
    silently fall through to systemctl (security review 2026-11-18)."""
    suffixed = NextDelegation(
        enabled=True, cmd=_cmd(standin, tmp_path, 0), units={"vllm-engine.service": "vllm-engine"}
    )
    bare = NextDelegation(
        enabled=True, cmd=_cmd(standin, tmp_path, 0), units={"vllm-engine": "vllm-engine"}
    )
    for delegation in (suffixed, bare):
        seen: list[list[str]] = []
        runner = SystemctlRunner(runner=_fake_systemctl(seen), next_delegation=delegation)
        assert runner.stop("vllm-engine").ok, delegation.units
        assert runner.stop("vllm-engine.service").ok, delegation.units
        assert seen == [], "both forms must delegate regardless of the key form"

@pytest.mark.parametrize("preset", [False, True])
@pytest.mark.parametrize("rc", [0, 4])
def test_delegation_hands_off_and_reacquires_host_lock(tmp_path, monkeypatch, preset, rc):
    import fcntl

    from llmctl.integrations import llmctl_next
    from llmctl.services.op_lock import gpu_operation_lock, held_by_us

    path = tmp_path / "operations.lock"
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(path))
    config = NextDelegation(enabled=True, units={"engine": "engine"})

    def runner(argv, **kwargs):
        assert not held_by_us()
        with path.open("r+") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return subprocess.CompletedProcess(argv, rc, "", "refused" if rc else "")

    monkeypatch.setattr(llmctl_next.subprocess, "run", runner)
    with gpu_operation_lock(purpose="test delegation", interface="pytest"):
        if preset:
            result = llmctl_next.run_preset("profile", "engine", config)
        else:
            result = llmctl_next.delegate("start", "engine", config)
        assert result[0] == rc
        assert held_by_us()
        with path.open("r+") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_delegation_aborts_when_caller_cannot_reacquire_lock(tmp_path, monkeypatch):
    import fcntl

    from llmctl.integrations import llmctl_next
    from llmctl.services import op_lock

    path = tmp_path / "operations.lock"
    monkeypatch.setenv("LLMCTL_OP_LOCK", str(path))
    original_handoff = op_lock.handed_off
    monkeypatch.setattr(llmctl_next, "handed_off", lambda **_: original_handoff(timeout=0))
    config = NextDelegation(enabled=True, units={"engine": "engine"})
    path.touch()
    with path.open("r+") as contender:
        def runner(argv, **kwargs):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return subprocess.CompletedProcess(argv, 3, "", "agent disconnected")

        monkeypatch.setattr(llmctl_next.subprocess, "run", runner)
        with op_lock.gpu_operation_lock(purpose="test", interface="pytest"):
            with pytest.raises(op_lock.LockBusy):
                llmctl_next.delegate("start", "engine", config)
            assert not op_lock.held_by_us()
