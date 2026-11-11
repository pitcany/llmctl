"""`install_unit` reports `written` only when the unit file was written."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from llmctl.services import systemd
from llmctl.services.systemd import SystemdUnit, install_unit

_UNIT = "[Service]\nExecStart=/bin/true\n"


def test_failed_sudo_tee_is_not_reported_as_written(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd, stderr="sudo: a password is required")

    monkeypatch.setattr(systemd.subprocess, "run", _run)
    report = install_unit(SystemdUnit(name="x.service", content=_UNIT, user=False))
    assert calls == [["sudo", "tee", "/etc/systemd/system/x.service"]]
    assert report.written is False
    assert report.enabled is False
    message = " ".join(report.messages)
    assert "a password is required" in message
    assert "systemctl failed" not in message


def test_missing_sudo_is_not_reported_as_written(monkeypatch: pytest.MonkeyPatch) -> None:
    def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(systemd.subprocess, "run", _run)
    report = install_unit(SystemdUnit(name="x.service", content=_UNIT, user=False))
    assert report.written is False
    assert "wrote the unit file" not in " ".join(report.messages)


def test_user_unit_without_systemctl_is_still_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "units" / "x.service"
    monkeypatch.setattr(systemd, "unit_install_path", lambda _unit: target)

    def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    monkeypatch.setattr(systemd.subprocess, "run", _run)
    report = install_unit(SystemdUnit(name="x.service", content=_UNIT, user=True))
    assert target.read_text() == _UNIT
    assert report.written is True
    assert report.enabled is False
    assert "wrote the unit file" in " ".join(report.messages)


def test_failed_reload_after_a_good_write_keeps_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "units" / "x.service"
    monkeypatch.setattr(systemd, "unit_install_path", lambda _unit: target)

    def _run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="Failed to connect to bus")

    monkeypatch.setattr(systemd.subprocess, "run", _run)
    report = install_unit(SystemdUnit(name="x.service", content=_UNIT, user=True))
    assert report.written is True
    assert "systemctl failed: Failed to connect to bus" in " ".join(report.messages)
