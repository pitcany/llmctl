"""`llmctl vllm` must not report readiness it never checked.

`--no-wait` skips the `/v1/models` poll entirely. The command still stopped
ollama and the plugin companions on the way in, so an operator who reads
"vLLM ready -- serving X" two seconds later has been told something nobody
verified: at that point the process has usually not even bound the port.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmctl import cli
from llmctl.adapters.vllm_systemd import ManagedRestartResult
from llmctl.integrations.vllm_env import VLLMLaunchSpec
from llmctl.services.vllm_orchestrator import OrchestratorResult


def _result(ready: bool | None, *, error: str | None = None) -> OrchestratorResult:
    return OrchestratorResult(
        spec=VLLMLaunchSpec(model="m", served_name="qwen3.6-27b", port=8003),
        restart=ManagedRestartResult(
            env_path=Path("/tmp/fake.env"), env_body="", ready=ready, error=error
        ),
    )


@pytest.fixture(autouse=True)
def _no_confirm(monkeypatch) -> None:
    """The confirmation prompt is not what these tests are about."""
    monkeypatch.setattr(cli, "_confirm_state_change", lambda *a, **k: None)


def _invoke(monkeypatch, result: OrchestratorResult, *args: str):
    monkeypatch.setattr(cli, "start_vllm_tp", lambda *a, **k: result, raising=False)
    return CliRunner().invoke(cli.app, ["vllm", "llama-3.3-70b", *args])


def test_no_wait_does_not_claim_ready(monkeypatch) -> None:
    """The headline claim must be absent when nothing was probed."""
    res = _invoke(monkeypatch, _result(None), "--no-wait")

    assert res.exit_code == 0, res.output
    assert "vLLM ready" not in res.output, res.output
    assert "not checked" in res.output.lower(), res.output
    # The operator still needs to know what was asked for.
    assert "qwen3.6-27b" in res.output


def test_waited_and_ready_still_says_ready(monkeypatch) -> None:
    """The honest success path is unchanged."""
    res = _invoke(monkeypatch, _result(True))

    assert res.exit_code == 0, res.output
    assert "vLLM ready" in res.output
    assert "qwen3.6-27b" in res.output


def test_probed_and_not_ready_is_a_failure(monkeypatch) -> None:
    """A real readiness timeout must stay a non-zero exit."""
    res = _invoke(
        monkeypatch, _result(False, error="vLLM did not become ready within 300s")
    )

    assert res.exit_code == 1, res.output
    assert "did not become ready" in res.output


def test_no_wait_is_not_treated_as_a_failure(monkeypatch) -> None:
    """Opting out of the probe is a choice, not an error: exit 0, no red."""
    res = _invoke(monkeypatch, _result(None), "--no-wait")

    assert res.exit_code == 0, res.output
    assert "did not complete cleanly" not in res.output


# --- the failure path and the exit-code contract ------------------------------
#
# `llmctl vllm` is the most destructive command in the tool -- it stops ollama
# and the plugin companions, rewrites the unit env file and restarts the GPU pair
# -- and had no CLI tests at all. These cover the contract README documents.


def test_failed_launch_says_what_was_left_stopped(monkeypatch) -> None:
    """The launch stopped ollama to free VRAM and then failed. Say so.

    The TUI has always disclosed this; the CLI discarded `fleet_stopped`, so an
    operator only found out by noticing ollama was dead.
    """
    res = _result(False, error="vLLM did not become ready within 300s")
    res.fleet_stopped = ["ollama.service"]

    out = _invoke(monkeypatch, res)

    assert out.exit_code == 1, out.output
    assert "still stopped" in out.output.lower(), out.output
    assert "ollama.service" in out.output


def test_failed_launch_reports_rollback_and_next_commands(monkeypatch) -> None:
    """After a rollback the CLI says what came back and what is still on the operator."""
    from llmctl.services.vllm_orchestrator import RollbackReport

    res = _result(False, error="vLLM did not become ready within 300s")
    res.fleet_stopped = ["ollama", "vllm-tp"]
    res.rollback = RollbackReport(
        env_restored=True,
        env_path=Path("/tmp/vllm-tp.env"),
        tp_stopped=True,
        restarted=["vllm-tp"],
        restart_failed=["ollama"],
    )

    out = _invoke(monkeypatch, res)

    assert out.exit_code == 1, out.output
    assert "restored previous" in out.output
    assert "started again: vllm-tp" in out.output
    assert "sudo systemctl start ollama" in out.output
    assert "journalctl -u vllm-tp" in out.output


def test_fleet_preflight_failure_is_reported(monkeypatch) -> None:
    res = _result(None)
    res.fleet_failed = ["ollama.service"]

    out = _invoke(monkeypatch, res)

    assert out.exit_code == 1, out.output
    assert "fleet preflight failed" in out.output


def test_unknown_preset_exits_2(monkeypatch) -> None:
    """Usage errors are exit 2, per the documented exit-code contract."""
    from llmctl.services.vllm_orchestrator import UnknownPresetError

    def boom(*a, **k):
        raise UnknownPresetError("no-such-preset")

    monkeypatch.setattr(cli, "start_vllm_tp", boom, raising=False)
    out = CliRunner().invoke(cli.app, ["vllm", "no-such-preset"])

    assert out.exit_code == 2, out.output
    assert "no-such-preset" in out.output


def test_missing_model_path_exits_2(monkeypatch) -> None:
    """A safety refusal is exit 2, not 1."""
    from llmctl.services.vllm_orchestrator import MissingModelPathError

    def boom(*a, **k):
        raise MissingModelPathError("/mnt/storage/gone")

    monkeypatch.setattr(cli, "start_vllm_tp", boom, raising=False)
    out = CliRunner().invoke(cli.app, ["vllm", "llama-3.3-70b"])

    assert out.exit_code == 2, out.output
    assert "Refusing to start" in out.output


def test_declining_the_confirmation_is_exit_0(monkeypatch) -> None:
    """Saying "no" is not an error."""
    import typer

    def decline(*a, **k):
        raise typer.Exit(0)

    monkeypatch.setattr(cli, "_confirm_state_change", decline)
    monkeypatch.setattr(
        cli, "start_vllm_tp", lambda *a, **k: pytest.fail("must not launch"),
        raising=False,
    )

    out = CliRunner().invoke(cli.app, ["vllm", "llama-3.3-70b"])

    assert out.exit_code == 0, out.output


def test_dry_run_prints_nothing_and_exits_0(monkeypatch) -> None:
    res = _result(None)
    res.dry_run = True

    out = _invoke(monkeypatch, res, "--dry-run")

    assert out.exit_code == 0, out.output
    assert "vLLM ready" not in out.output
