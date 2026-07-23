"""Regression tests for the adversarial-review fixes.

Each test pins one confirmed finding: pid persistence across the readiness
wait, FAILED-pid hygiene, probe debounce, config merge/exclude_unset, settings
path resolution, secret redaction, --json flag propagation, doctor check
isolation, systemctl timeout, and systemd unit rendering.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from sqlmodel import Session as DBSession
from typer.testing import CliRunner

from llmctl.adapters.llama_cpp import LlamaCppAdapter
from llmctl.cli import _redact_secrets, app
from llmctl.config import RuntimeConfig, Settings, settings_file_path
from llmctl.db import (
    RuntimeName,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import LaunchPlan

runner = CliRunner()
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class _FakeSupervisor:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive

    def launch(self, command, env=None, cwd=None, log_name=None):
        from llmctl.db import utcnow
        from llmctl.telemetry.process import LaunchResult

        return LaunchResult(pid=7777, command=list(command), log_path=None, started_at=utcnow())

    def is_running(self, pid):
        return self.alive


def _plan() -> LaunchPlan:
    return LaunchPlan(
        runtime=RuntimeName.LLAMA_CPP,
        command=["llama-server"],
        endpoint_url="http://127.0.0.1:9999",
        dry_run=False,
    )


def test_on_spawn_fires_before_readiness_wait() -> None:
    """The pid reaches the callback before the (possibly long) readiness wait."""
    config = RuntimeConfig(readiness_timeout_s=0.1, readiness_poll_interval_s=0.05)
    adapter = LlamaCppAdapter(config, _FakeSupervisor(alive=True))  # type: ignore[arg-type]

    async def never_alive(_endpoint):
        return False

    adapter._endpoint_alive = never_alive  # type: ignore[method-assign]
    seen: list[tuple[int, str | None]] = []
    session = asyncio.run(adapter.start(_plan(), on_spawn=lambda pid, log: seen.append((pid, log))))
    assert seen == [(7777, None)]
    assert session.status == SessionStatus.STARTING


def test_interrupted_readiness_wait_leaves_pid_in_db(tmp_path: Path) -> None:
    """If the process dies mid-start-call, the DB still knows the spawned pid."""
    from llmctl.schemas import SessionStartRequest
    from llmctl.services.sessions import SessionService

    url = f"sqlite:///{tmp_path / 'interrupt.sqlite3'}"
    init_db(url)
    with DBSession(get_engine(url)) as db:
        service = SessionService(db)

        class InterruptedAdapter:
            async def start(self, plan, on_spawn=None):
                if on_spawn is not None:
                    on_spawn(4242, "/tmp/x.log")
                raise KeyboardInterrupt

        service.router.get_adapter = lambda runtime: InterruptedAdapter()  # type: ignore[method-assign]
        request = SessionStartRequest(
            model_id="m", runtime=RuntimeName.PYTHON_SCRIPT, dry_run=False, force=True
        )
        with pytest.raises(KeyboardInterrupt):
            service.start(request)
        records = service.list_sessions()
        assert records[-1].pid == 4242  # recoverable: reconcile/stop can find it


def test_failed_startup_clears_pid() -> None:
    """A FAILED start does not leave a dead (reusable) pid in the session."""
    config = RuntimeConfig(readiness_timeout_s=0.5, readiness_poll_interval_s=0.05)
    adapter = LlamaCppAdapter(config, _FakeSupervisor(alive=False))  # type: ignore[arg-type]

    async def never_alive(_endpoint):
        return False

    adapter._endpoint_alive = never_alive  # type: ignore[method-assign]
    session = asyncio.run(adapter.start(_plan()))
    assert session.status == SessionStatus.FAILED
    assert session.pid is None


def test_owned_probe_debounce_keeps_running_on_single_blip(tmp_path: Path) -> None:
    """One failed probe must not demote RUNNING; the retry answers."""
    from llmctl.db import SessionKind, SessionRecord
    from llmctl.services.sessions import SessionService

    url = f"sqlite:///{tmp_path / 'blip.sqlite3'}"
    init_db(url)
    answers = iter([None, ["m"]])
    with DBSession(get_engine(url)) as db:
        service = SessionService(db, probe=lambda u, t: next(answers))
        service.router.supervisor = _FakeSupervisor(alive=True)  # type: ignore[assignment]
        record = SessionRecord(
            runtime=RuntimeName.LLAMA_CPP,
            status=SessionStatus.RUNNING,
            kind=SessionKind.OWNED,
            pid=7777,
            endpoint_url="http://127.0.0.1:9999",
        )
        db.add(record)
        db.commit()
        assert service.reconcile() == 0
        db.refresh(record)
        assert record.status == SessionStatus.RUNNING


def test_runtime_config_partial_override_keeps_port_range() -> None:
    """Setting one YAML field must not clobber runtime-specific defaults."""
    settings = Settings.model_validate(
        {"runtimes": {"llama_cpp": {"readiness_timeout_s": 30.0}}}
    )
    merged = settings.runtime_config("llama_cpp")
    from llmctl.config import default_runtime_configs

    assert merged.readiness_timeout_s == 30.0
    assert merged.port_range == default_runtime_configs()["llama_cpp"].port_range


def test_settings_file_path_fresh_install(tmp_path: Path, monkeypatch) -> None:
    missing_dir = tmp_path / "not-created-yet"
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(missing_dir))
    assert settings_file_path() == missing_dir / "settings.yaml"

    yaml_file = tmp_path / "custom.yaml"
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(yaml_file))
    assert settings_file_path() == yaml_file  # explicit file path, absent or not


def test_redaction_covers_auth_shapes() -> None:
    payload = {
        "runtimes": {
            "ollama": {
                "env": {
                    "AUTHORIZATION": "Bearer sk-123",
                    "PRIVATE_KEY": "MIIE...",
                    "HF_TOKEN": "hf_abc",
                    "PORT": "8003",
                }
            }
        },
        "scheduler": {"require_auth_token": True},
    }
    redacted = _redact_secrets(payload)
    env = redacted["runtimes"]["ollama"]["env"]
    assert env["AUTHORIZATION"] == "********"
    assert env["PRIVATE_KEY"] == "********"
    assert env["HF_TOKEN"] == "********"
    assert env["PORT"] == "8003"
    # booleans are never masked, even under secret-looking keys
    assert redacted["scheduler"]["require_auth_token"] is True


def test_runtimes_group_json_flag_reaches_inspect(tmp_path: Path, monkeypatch) -> None:
    import json

    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'cli.sqlite3'}")
    result = runner.invoke(app, ["runtimes", "--json", "inspect", "python_script"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["runtime"] == "python_script"


def test_doctor_isolates_crashing_check(monkeypatch) -> None:
    import llmctl.services.doctor as doctor_mod

    def boom(_cfg):
        raise ValueError("model_dirs.yaml is not a mapping")

    monkeypatch.setattr(doctor_mod, "_check_drift", boom)
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    report = doctor_mod.run_doctor()
    assert any(c["name"] == "doctor:drift" for c in report["failures"])
    # the other checks still ran
    assert report["passed"] or report["warnings"]


def test_systemctl_run_times_out_cleanly(monkeypatch) -> None:
    from llmctl.integrations.systemctl import SystemctlRunner, SystemctlVerb

    def slow_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", slow_run)
    result = SystemctlRunner().run(SystemctlVerb.IS_ACTIVE, "vllm-tp")
    assert result.returncode == 124
    assert "timed out" in result.stderr


def test_api_unit_warns_on_public_host_without_allow() -> None:
    from llmctl.services.systemd import render_api_unit

    settings = Settings.model_validate({"api": {"host": "0.0.0.0"}})
    unit = render_api_unit(settings)
    assert any("allow_public_bind" in warning for warning in unit.warnings)

    loopback = render_api_unit(Settings.model_validate({}))
    assert loopback.warnings == []


def test_launch_modal_escape_returns_none(tmp_path, monkeypatch) -> None:
    from textual.app import App

    from llmctl.tui._modals import LaunchPlanModal

    plan = LaunchPlan(runtime=RuntimeName.PYTHON_SCRIPT, command=["echo"], dry_run=True)

    async def _run() -> None:
        outcomes: list[object] = []

        class Harness(App[None]):
            pass

        harness = Harness()
        async with harness.run_test() as pilot:
            harness.push_screen(LaunchPlanModal(plan), outcomes.append)
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert outcomes == [None]

    asyncio.run(_run())
