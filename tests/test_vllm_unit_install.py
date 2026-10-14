"""A stranger can serve vLLM: llmctl ships the launcher and writes the unit.

Before this, `llmctl run` restarted a `vllm-tp` unit that only existed on
the author's machine, driven by a launcher script in that repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmctl.cli import app
from llmctl.config import ManagedUnitConfig
from llmctl.services.systemd import install_unit, launcher_path, render_vllm_unit


def test_the_shipped_launcher_exists_and_is_executable() -> None:
    path = launcher_path()
    assert path.name == "vllm-launcher.sh"
    assert os.access(path, os.X_OK)
    text = path.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    # Host-neutral: no author paths, interpreter comes from the env file's PATH.
    assert "/home/yannik" not in text
    assert 'VLLM_PYTHON:-$(command -v python3' in text


def test_render_vllm_unit_points_at_the_launcher_and_env_file(tmp_path: Path) -> None:
    unit = ManagedUnitConfig(unit_name="vllm-tp", env_file_path=tmp_path / "vllm-tp.env")
    rendered = render_vllm_unit(
        unit, user_name="alice", launcher=Path("/opt/llmctl/vllm-launcher.sh")
    )

    assert rendered.name == "vllm-tp.service"
    assert rendered.user is False  # system scope: the GPUs are machine-wide
    assert "ExecStart=/opt/llmctl/vllm-launcher.sh" in rendered.content
    assert f"EnvironmentFile=-{tmp_path / 'vllm-tp.env'}" in rendered.content
    assert "User=alice" in rendered.content
    assert "Restart=on-failure" in rendered.content
    assert "WantedBy=multi-user.target" in rendered.content
    assert rendered.warnings == []


def test_render_vllm_unit_warns_when_the_marker_would_refuse_it(tmp_path: Path) -> None:
    """The adapter refuses a unit whose ExecStart lacks launcher_marker."""
    unit = ManagedUnitConfig(
        unit_name="vllm-tp", env_file_path=tmp_path / "e", launcher_marker="custom-launch.sh"
    )
    rendered = render_vllm_unit(unit, user_name="alice", launcher=Path("/opt/vllm-launcher.sh"))
    assert rendered.warnings and "launcher_marker" in rendered.warnings[0]


def test_install_dry_run_for_a_system_unit_writes_nothing(tmp_path: Path) -> None:
    unit = render_vllm_unit(
        ManagedUnitConfig(unit_name="vllm-tp", env_file_path=tmp_path / "e"),
        user_name="alice",
        launcher=Path("/opt/vllm-launcher.sh"),
    )
    report = install_unit(unit, dry_run=True)
    assert report.written is False and report.dry_run is True
    assert report.unit_path == "/etc/systemd/system/vllm-tp.service"
    assert report.actions[0] == "sudo tee /etc/systemd/system/vllm-tp.service"
    assert "sudo systemctl daemon-reload" in report.actions


def test_cli_install_vllm_unit_defaults_to_a_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("AI_HOME", raising=False)
    monkeypatch.delenv("LLMCTL_VLLM_ENV_FILE", raising=False)

    res = CliRunner().invoke(app, ["install-vllm-unit"], env={"COLUMNS": "200"})

    assert res.exit_code == 0, res.output
    assert "DRY RUN" in res.output
    assert str(launcher_path()) in res.output
    assert str(tmp_path / "state" / "llmctl" / "units" / "vllm-tp.env") in res.output
    assert "--no-dry-run" in res.output


def test_cli_install_vllm_unit_rejects_an_unknown_role(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path / "cfg"))
    res = CliRunner().invoke(app, ["install-vllm-unit", "--role", "nope"])
    assert res.exit_code == 2
    assert "nope" in res.output and "vllm-tp" in res.output
