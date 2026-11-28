"""Cutover mode (llmctl M6b): every lifecycle write goes through the llmctl-next guard.

`llmctl stop <lane>`, the TUI swap helpers and the rollback paths used to run raw
`systemctl` / `sudo systemctl` / lane scripts. In cutover mode an enrolled unit goes to
the agent and any other unit is refused; nothing else runs. With the mode absent the
old argv runs unchanged. A stand-in llmctl-next records what the guard ran.
"""

from __future__ import annotations

import shlex
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmctl import cli, cli_swap
from llmctl.services import lane_swap, stop_serving
from llmctl.services.lane_swap import LaneSwapDeps, LaneSwapResult
from llmctl.services.lanes import Lane, SwapState, build_catalog
from llmctl.services.session_swap import swap_to_session
from llmctl.services.systemd import SystemdUnit, install_unit

STANDIN = """#!/usr/bin/env python3
import sys, pathlib, os
pathlib.Path(sys.argv[2]).open('a').write(' '.join(sys.argv[3:]) + '\\n')
os._exit(int(sys.argv[1]))
"""

ENROLLED = Lane(name="gptoss", served_model="gpt-oss", upstream="http://127.0.0.1:1",
                start_cmd=("gptoss-start", "--take-gpus"), unit="gpt-oss-120b.service")
OTHER = Lane(name="glm53", served_model="glm", upstream="http://127.0.0.1:2",
             start_cmd=("glm53-start", "--take-gpus"), unit="glm-5.3-flash.service")


class Recorder:
    """LaneSwapDeps that record every command and never run one."""

    def __init__(self) -> None:
        self.ran: list[list[str]] = []
        self.logs: list[str] = []
        self.deps = LaneSwapDeps(
            run_script=lambda argv, timeout: self.ran.append(list(argv)) or 0,
            run_quiet=lambda argv: self.ran.append(list(argv)) or 0,
            probe=lambda upstream: ["gpt-oss", "glm"],
            sleep=lambda s: None,
            log=self.logs.append,
            record=lambda key, seconds: None,
            remember_vram=lambda key, lane: None,
        )


def _settings(tmp_path: Path, monkeypatch, body: str, rc: int = 0) -> Path:
    """A temp settings.yaml and a stand-in agent; returns the agent's argv log."""
    (tmp_path / "settings.yaml").write_text(body)
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("LLMCTL_NEXT_DISABLED", raising=False)
    script = tmp_path / "standin"
    script.write_text(STANDIN)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    marker = tmp_path / "agent.txt"
    monkeypatch.setenv("LLMCTL_NEXT_CMD", shlex.join([str(script), str(rc), str(marker)]))
    return marker


CUTOVER = (
    "llmctl_next:\n  mode: cutover\n"
    "  units: {gpt-oss-120b.service: gpt-oss-120b, vllm-tp: vllm-engine}\n"
)
#: Delegate mode with the same map: lane_swap never delegated here, and must not start to.
DELEGATE = CUTOVER.replace("mode: cutover", "enabled: true")


@pytest.fixture
def cutover(tmp_path, monkeypatch) -> Path:
    return _settings(tmp_path, monkeypatch, CUTOVER)


def _agent(marker: Path) -> list[str]:
    return marker.read_text().splitlines() if marker.exists() else []


# ---- (a) cutover: an enrolled unit goes to the agent, no systemctl ---------------

def test_cutover_stop_of_an_enrolled_lane_goes_to_the_agent(cutover, monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(lane_swap, "LaneSwapDeps", lambda: rec.deps)  # what real_deps builds
    deps = stop_serving.real_deps(settings=None)
    assert deps.stop_lane("gpt-oss-120b.service") is True
    assert _agent(cutover) == ["stop gpt-oss-120b --wait"]
    assert rec.ran == []


def test_cutover_swap_to_lane_is_one_agent_run(cutover):
    rec = Recorder()
    result = lane_swap.swap_to_lane(
        ENROLLED, other_active=[OTHER], vllm_active=True, vllm_unit="vllm-tp", deps=rec.deps
    )
    assert result.ok, result.error
    assert _agent(cutover) == ["run gpt-oss-120b --wait"]
    assert rec.ran == [], "no lane script, no systemctl: the agent plans the stops"


def test_cutover_swap_to_vllm_stops_nothing_itself(cutover):
    rec = Recorder()

    class Started:
        ok = True

    started, lanes = lane_swap.swap_to_vllm([ENROLLED, OTHER], Started, deps=rec.deps)
    assert started.ok and lanes.ok
    assert rec.ran == [] and _agent(cutover) == []


def test_cutover_rollback_starts_units_through_the_agent(cutover):
    rec = Recorder()
    result = LaneSwapResult(ok=False)
    lane_swap.restore_previous(
        [ENROLLED], restart_vllm=True, vllm_unit="vllm-tp", deps=rec.deps, result=result
    )
    assert _agent(cutover) == ["run vllm-engine --wait", "run gpt-oss-120b --wait"]
    assert rec.ran == [] and result.restored == ["vllm-tp", "gpt-oss-120b.service"]


def test_cutover_run_of_an_enrolled_lane_reaches_the_agent(cutover, monkeypatch):
    state = SwapState(
        catalog=build_catalog({}, [ENROLLED, OTHER], vllm_port=8003),
        serving=set(), active_units=set(), vllm_unit="vllm-tp", vllm_active=False,
    )
    monkeypatch.setattr(cli, "gather_state", lambda settings: state)
    monkeypatch.setattr(cli, "_confirm_state_change", lambda *a, **k: None)
    monkeypatch.setattr(cli_swap, "router_up", lambda url: False)
    rec = Recorder()
    monkeypatch.setattr(lane_swap, "LaneSwapDeps", lambda: rec.deps)

    res = CliRunner().invoke(cli.app, ["run", "gptoss"])
    assert res.exit_code == 0, res.output
    assert "Ready" in res.output
    assert _agent(cutover) == ["run gpt-oss-120b --wait"] and rec.ran == []

    # A lane with no workload: refused as before, nothing runs.
    res = CliRunner().invoke(cli.app, ["run", "glm53"])
    assert res.exit_code == 1 and "llmctl-next" in res.output
    assert _agent(cutover) == ["run gpt-oss-120b --wait"] and rec.ran == []


# ---- (b) cutover: a unit with no workload is refused, no systemctl ---------------

def test_cutover_refuses_a_unit_with_no_workload(cutover):
    rec = Recorder()
    assert lane_swap.stop_user_unit("glm-5.3-flash.service", rec.deps) is False
    result = lane_swap.swap_to_lane(
        OTHER, other_active=[], vllm_active=False, vllm_unit="vllm-tp", deps=rec.deps
    )
    assert not result.ok and "llmctl-next run glm-5.3-flash --wait" in (result.error or "")
    assert rec.ran == [] and _agent(cutover) == []
    assert any("llmctl-next stop glm-5.3-flash --wait" in line for line in rec.logs)


def test_cutover_agent_refusal_is_a_failure_without_fallback(tmp_path, monkeypatch):
    marker = _settings(tmp_path, monkeypatch, CUTOVER, rc=4)
    rec = Recorder()
    assert lane_swap.stop_user_unit("gpt-oss-120b.service", rec.deps) is False
    assert _agent(marker) == ["stop gpt-oss-120b --wait"] and rec.ran == []


def test_cutover_refuses_a_session_start(cutover):
    rec = Recorder()
    started: list[int] = []
    result = swap_to_session(
        "some-gguf", placement=None, start=started.append, stop=lambda sid: None,
        restart=lambda sid: None, gpu_sessions=[], active_lanes=[ENROLLED], vllm_active=True,
        vllm_unit="vllm-tp", deps=rec.deps,
    )
    assert not result.ok and "llmctl-next" in (result.error or "")
    assert started == [] and rec.ran == [] and _agent(cutover) == []


def test_cutover_install_unit_does_not_enable_or_start(cutover, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    unit = SystemdUnit(name="x-test.service", content="[Unit]\n", user=True)
    report = install_unit(unit, dry_run=True)
    assert not any("enable" in action for action in report.actions)


# ---- (c) mode absent, or delegate mode: the old argv exactly ---------------------

@pytest.mark.parametrize("body", ["", DELEGATE])
def test_without_cutover_the_old_commands_run_unchanged(tmp_path, monkeypatch, body):
    marker = _settings(tmp_path, monkeypatch, body)
    rec = Recorder()
    assert lane_swap.stop_user_unit("gpt-oss-120b.service", rec.deps) is True
    assert rec.ran == [["systemctl", "--user", "stop", "gpt-oss-120b.service"]]

    rec = Recorder()
    result = lane_swap.swap_to_lane(
        ENROLLED, other_active=[OTHER], vllm_active=False, vllm_unit="vllm-tp", deps=rec.deps
    )
    assert result.ok
    assert rec.ran == [
        ["systemctl", "--user", "stop", "glm-5.3-flash.service"],
        ["gptoss-start", "--take-gpus"],
    ]

    rec = Recorder()
    lane_swap.restore_previous(
        [OTHER], restart_vllm=True, vllm_unit="vllm-tp", deps=rec.deps,
        result=LaneSwapResult(ok=False),
    )
    assert rec.ran == [
        ["sudo", "-n", "systemctl", "start", "vllm-tp"],
        ["glm53-start", "--take-gpus"],
    ]

    rec = Recorder()
    started, lanes = lane_swap.swap_to_vllm([OTHER], lambda: None, deps=rec.deps)
    assert rec.ran[0] == ["systemctl", "--user", "stop", "glm-5.3-flash.service"]
    assert _agent(marker) == []

    unit = SystemdUnit(name="x-test.service", content="[Unit]\n", user=True)
    actions = install_unit(unit, dry_run=True).actions
    assert "systemctl --user enable --now x-test.service" in actions


def test_without_cutover_a_session_swap_stops_vllm_with_sudo(tmp_path, monkeypatch):
    _settings(tmp_path, monkeypatch, "")
    rec = Recorder()

    class Placement:
        problem = None
        cards = 1

    def refuse(cards: int):
        raise RuntimeError("scheduler refused")

    swap_to_session(
        "some-gguf", placement=Placement(), start=refuse, stop=lambda sid: None,
        restart=lambda sid: None, gpu_sessions=[], active_lanes=[], vllm_active=True,
        vllm_unit="vllm-tp", deps=rec.deps,
    )
    assert rec.ran == [
        ["sudo", "-n", "systemctl", "stop", "vllm-tp"],
        ["sudo", "-n", "systemctl", "start", "vllm-tp"],
    ]
