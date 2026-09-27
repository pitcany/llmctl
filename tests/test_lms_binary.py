"""LM Studio rows read "ready" when `lms` is installed but not on PATH.

`~/.lmstudio/bin` reaches PATH only through `~/.bashrc`, so a TUI started
from tmux, ssh or a launcher could not find `lms` and every LM Studio row in
the Models screen read "no binary".
"""

from __future__ import annotations

from pathlib import Path

from llmctl.adapters import lmstudio
from llmctl.services.backends import detect_backends


def _install_lms(home: Path) -> Path:
    lms = home / ".lmstudio" / "bin" / "lms"
    lms.parent.mkdir(parents=True)
    lms.write_text("#!/bin/sh\n")
    lms.chmod(0o755)
    return lms


def test_lms_binary_falls_back_to_the_install_location(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert lmstudio.lms_binary() is None
    lms = _install_lms(tmp_path)
    assert lmstudio.lms_binary() == str(lms)


def test_lmstudio_backend_is_available_without_lms_on_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    lms = _install_lms(tmp_path)

    def unreachable(url: str, timeout: float):
        raise OSError("connection refused")

    rows = {r["backend"]: r for r in detect_backends(http_get=unreachable)}
    assert rows["lmstudio"]["available"] is True
    assert rows["lmstudio"]["path"] == str(lms)
