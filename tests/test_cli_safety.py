"""Tests for CLI safety wiring: confirmations, bind guard, GPU-pinning fallback.

Covers the previously dead ``scheduler.require_confirmation_for_*`` settings
(now honored, TTY-gated, ``--yes``-bypassed), the control-plane API public-bind
guard, and the EnvironmentFile fallback for GPU pinning of root-owned units.
"""

from __future__ import annotations

import subprocess

import pytest
import typer
from typer.testing import CliRunner

from llmctl.cli import _confirm_state_change, app
from llmctl.services.unit_gpus import unit_gpu_ids

runner = CliRunner()


def test_confirm_skipped_when_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    called = False

    def confirm(_msg: str) -> bool:
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(typer, "confirm", confirm)
    _confirm_state_change("x", required=False, assume_yes=False)
    assert not called


def test_confirm_skipped_with_yes_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(
        typer, "confirm", lambda _msg: pytest.fail("must not prompt with --yes")
    )
    _confirm_state_change("x", required=True, assume_yes=True)


def test_confirm_skipped_when_not_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        typer, "confirm", lambda _msg: pytest.fail("must not prompt without a TTY")
    )
    _confirm_state_change("x", required=True, assume_yes=False)


def test_confirm_decline_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda _msg: False)
    with pytest.raises(typer.Exit) as excinfo:
        _confirm_state_change("x", required=True, assume_yes=False)
    assert excinfo.value.exit_code == 0


def test_confirm_accept_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda _msg: True)
    _confirm_state_change("x", required=True, assume_yes=False)


def test_serve_refuses_public_bind(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from pathlib import Path

    import uvicorn

    # Isolate from the invoking user's real config, and make an (incorrect)
    # fall-through into a real bind fail loudly instead of hanging the suite.
    configs = Path(__file__).resolve().parents[1] / "configs"
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(configs))
    monkeypatch.setenv("LLMCTL_DB_URL", f"sqlite:///{tmp_path / 'serve.sqlite3'}")
    monkeypatch.setattr(
        uvicorn, "run", lambda *a, **k: pytest.fail("guard must fire before binding")
    )
    result = runner.invoke(app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 2
    assert "Refusing to bind" in result.output


def _fake_systemctl_run(main_pid: str, env_files_value: str):
    def run(argv, capture_output=True, text=True, timeout=None, check=False):
        if "--property=MainPID" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=main_pid, stderr="")
        if "--property=EnvironmentFiles" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=env_files_value, stderr="")
        raise AssertionError(f"unexpected argv: {argv}")

    return run


def test_unit_gpu_ids_falls_back_to_environment_file(tmp_path) -> None:
    env_file = tmp_path / "vllm-tp.env"
    env_file.write_text("PORT=8003\nCUDA_VISIBLE_DEVICES=0,1\n")
    ids = unit_gpu_ids(
        "vllm-tp.service",
        run=_fake_systemctl_run("4242\n", f"{env_file} (ignore_errors=no)\n"),
        read_environ=lambda pid: None,  # simulate root-owned /proc environ
    )
    assert ids == [0, 1]


def test_unit_gpu_ids_fallback_missing_file_returns_empty(tmp_path) -> None:
    ids = unit_gpu_ids(
        "vllm-tp.service",
        run=_fake_systemctl_run("4242\n", f"{tmp_path / 'gone.env'} (ignore_errors=no)\n"),
        read_environ=lambda pid: None,
    )
    assert ids == []


def test_tui_action_worker_surfaces_errors(tmp_path, monkeypatch) -> None:
    """A failing TUI action notifies instead of crashing the worker."""
    import asyncio
    from pathlib import Path

    from llmctl.config import load_settings
    from llmctl.tui.app import MissionControlApp

    configs = Path(__file__).resolve().parents[1] / "configs"
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(configs))
    base = load_settings()
    settings = base.model_copy(deep=True)
    settings.database.url = f"sqlite:///{tmp_path / 'tui.db'}"
    monkeypatch.setattr("llmctl.tui._data.load_settings", lambda: settings, raising=False)

    async def _run() -> None:
        app = MissionControlApp()
        async with app.run_test() as pilot:
            await pilot.press("s")  # Sessions screen
            await pilot.pause()
            screen = app.screen

            def boom() -> None:
                raise ValueError("adopted sessions are stopped via systemd")

            screen.run_action_worker(boom, lambda _result: pytest.fail("must not run"))
            await pilot.pause(0.2)
            await app.workers.wait_for_complete()
            # The app is still alive and the error surfaced as a notification.
            assert any(
                "adopted sessions" in n.message for n in app._notifications
            )

    asyncio.run(_run())
