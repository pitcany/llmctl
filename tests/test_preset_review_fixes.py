"""Fixes for two review findings on the P2 devlab run (2026-11-04)."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from llmctl.presets import migrate, store
from llmctl.presets.schema import Model

MINIMAL = {
    "alias": "alpha",
    "served_name": "alpha-served",
    "model_id": "Org/Alpha-7B",
    "quantization": "none",
    "vllm_quantization_flag": "none",
    "tensor_parallel_size": 1,
    "max_model_len": 8192,
}


def _write(directory: Path, comment: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "alpha.yaml"
    path.write_text(comment + yaml.safe_dump(MINIMAL, sort_keys=False))
    return path


def test_failed_legacy_swap_leaves_no_unrecorded_canonical_file(tmp_path, monkeypatch):
    legacy_dir, canonical_dir = tmp_path / "legacy", tmp_path / "canonical"
    legacy = _write(legacy_dir)
    original = legacy.read_bytes()

    def boom(_dest, _target):
        raise OSError("disk full")

    monkeypatch.setattr(migrate, "_replace_with_symlink", boom)
    with pytest.raises(OSError):
        migrate.apply(legacy_dir, canonical_dir)
    assert not (canonical_dir / "alpha.yaml").exists(), "unrecorded canonical file left behind"
    assert legacy.read_bytes() == original and not legacy.is_symlink()


def test_full_rewrite_fallback_is_logged(tmp_path, monkeypatch, caplog):
    path = _write(tmp_path, comment="# context\n")

    def cannot_edit(_text, _desired):
        raise ValueError("unusual layout")

    monkeypatch.setattr(store, "edit_preset_text", cannot_edit)
    model = Model.model_validate(MINIMAL).model_copy(update={"max_model_len": 4096})
    with caplog.at_level(logging.WARNING, logger=store.log.name):
        store.save_preset(model, config_dir=tmp_path)
    assert yaml.safe_load(path.read_text())["max_model_len"] == 4096
    assert any("comments are not kept" in r.getMessage() for r in caplog.records)
