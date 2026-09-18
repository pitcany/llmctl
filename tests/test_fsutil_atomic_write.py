"""Tests for :func:`llmctl._fsutil.atomic_write_text`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llmctl._fsutil import atomic_write_text


def test_writes_new_file_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "vllm-tp.env"
    atomic_write_text(target, "A=1\n")
    assert target.read_text() == "A=1\n"
    assert [p.name for p in target.parent.iterdir()] == ["vllm-tp.env"]


def test_preserves_existing_mode(tmp_path: Path) -> None:
    target = tmp_path / "f.env"
    target.write_text("old\n")
    os.chmod(target, 0o640)
    atomic_write_text(target, "new\n")
    assert target.read_text() == "new\n"
    assert target.stat().st_mode & 0o777 == 0o640


def test_follows_symlink_instead_of_replacing_it(tmp_path: Path) -> None:
    real = tmp_path / "legacy" / "p.yaml"
    real.parent.mkdir()
    real.write_text("old\n")
    link = tmp_path / "p.yaml"
    link.symlink_to(real)
    atomic_write_text(link, "new\n")
    assert link.is_symlink()
    assert real.read_text() == "new\n"


def test_failed_write_keeps_old_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "f.env"
    target.write_text("old\n")

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "new\n")
    assert target.read_text() == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["f.env"]
