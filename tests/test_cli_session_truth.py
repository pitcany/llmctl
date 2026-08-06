"""The CLI must report what a session command actually did.

Two lies lived here. ``restart`` terminated the process *and relaunched it*
(`SessionService.restart` calls `_terminate_record` then `_launch_record`) while
printing "Restart planned <id>; no process launched." -- no pid, no endpoint, no
error, and exit 0 even when the relaunch failed. And ``stop`` announced success
for a process that had survived SIGTERM and SIGKILL.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llmctl import cli
from llmctl.db import RuntimeName, SessionStatus
from llmctl.schemas import Session


def _session(status: SessionStatus, **kw) -> Session:
    return Session(
        id="sess-1",
        model_id="m1",
        runtime=RuntimeName.VLLM,
        status=status,
        **kw,
    )


class _StubService:
    """Stands in for SessionService; records what the CLI asked for."""

    result: Session | None = None

    def __init__(self, db, *a, **k) -> None:
        pass

    def stop(self, session_id, *, stop_unit=False):
        return type(self).result

    def restart(self, session_id):
        return type(self).result


@pytest.fixture(autouse=True)
def _wire(monkeypatch, tmp_path):
    """No confirmation prompt, no real database, stubbed service."""
    monkeypatch.setattr(cli, "_confirm_state_change", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_session", lambda: _NullCtx())
    monkeypatch.setattr(cli, "SessionService", _StubService)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# --- restart ------------------------------------------------------------------


def test_restart_reports_the_process_it_launched() -> None:
    _StubService.result = _session(
        SessionStatus.RUNNING, pid=4242, endpoint_url="http://127.0.0.1:8003"
    )

    res = CliRunner().invoke(cli.app, ["restart", "sess-1"])

    assert res.exit_code == 0, res.output
    assert "no process launched" not in res.output, "restart does launch a process"
    assert "4242" in res.output, res.output
    assert "8003" in res.output, res.output


def test_restart_exits_non_zero_when_the_relaunch_failed() -> None:
    _StubService.result = _session(
        SessionStatus.FAILED, error="port 8003 already in use"
    )

    res = CliRunner().invoke(cli.app, ["restart", "sess-1"])

    assert res.exit_code == 1, res.output
    assert "port 8003 already in use" in res.output


def test_restart_reports_a_still_loading_endpoint_honestly() -> None:
    _StubService.result = _session(SessionStatus.STARTING, pid=99)

    res = CliRunner().invoke(cli.app, ["restart", "sess-1"])

    assert res.exit_code == 0, res.output
    assert "starting" in res.output.lower()


def test_restart_says_so_when_it_really_did_not_launch() -> None:
    """A plan-only row (no stored launch plan) is the one honest 'planned'."""
    _StubService.result = _session(SessionStatus.PLANNED)

    res = CliRunner().invoke(cli.app, ["restart", "sess-1"])

    assert res.exit_code == 0, res.output
    assert "no process launched" in res.output


def test_restart_unknown_session_is_a_usage_error() -> None:
    _StubService.result = None

    res = CliRunner().invoke(cli.app, ["restart", "ghost"])

    assert res.exit_code == 2, res.output
    assert "ghost" in res.output


def test_restart_asks_for_confirmation(monkeypatch) -> None:
    """It interrupts whatever is serving; that deserves the same gate as stop."""
    asked: list[tuple] = []
    monkeypatch.setattr(
        cli,
        "_confirm_state_change",
        lambda action, **kw: asked.append((action, kw)),
    )
    _StubService.result = _session(SessionStatus.RUNNING, pid=1)

    res = CliRunner().invoke(cli.app, ["restart", "sess-1"])

    assert res.exit_code == 0, res.output
    assert asked, "restart never asked for confirmation"
    assert "sess-1" in asked[0][0]


def test_restart_yes_flag_skips_the_prompt(monkeypatch) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(
        cli,
        "_confirm_state_change",
        lambda action, **kw: seen.append(kw["assume_yes"]),
    )
    _StubService.result = _session(SessionStatus.RUNNING, pid=1)

    res = CliRunner().invoke(cli.app, ["restart", "sess-1", "--yes"])

    assert res.exit_code == 0, res.output
    assert seen == [True]


# --- stop ---------------------------------------------------------------------


def test_stop_reports_a_survivor_as_a_failure() -> None:
    """A process that outlived SIGKILL is not a successful stop."""
    _StubService.result = _session(
        SessionStatus.DEGRADED,
        pid=4242,
        error="vLLM process 4242 did not terminate cleanly.",
    )

    res = CliRunner().invoke(cli.app, ["stop", "sess-1"])

    assert res.exit_code == 1, res.output
    # Rich wraps the cell at 80 columns, so assert on tokens that survive it.
    assert "not stopped" in res.output
    assert "did not terminate" in res.output
    assert "4242" in res.output


def test_stop_clean_is_unchanged() -> None:
    _StubService.result = _session(SessionStatus.STOPPED)

    res = CliRunner().invoke(cli.app, ["stop", "sess-1"])

    assert res.exit_code == 0, res.output
    assert "stopped" in res.output.lower()
