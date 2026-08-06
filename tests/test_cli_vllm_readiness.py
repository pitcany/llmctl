"""`llmctl vllm` must not report readiness it never checked.

`--no-wait` skips the `/v1/models` poll entirely. The command still stopped
ollama and the Harbor container on the way in, so an operator who reads
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
