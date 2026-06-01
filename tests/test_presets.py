"""Tests for llmctl's internal preset schema and store."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

from llmctl.presets import Model, PresetSchemaError, load_all, load_one
from llmctl.presets.paths import default_preset_dir, user_config_dir
from llmctl.presets.store import migrate_legacy_presets


def _valid_model_kwargs() -> dict[str, object]:
    return {
        "alias": "x",
        "served_name": "x",
        "model_id": "org/x",
        "quantization": "awq",
        "vllm_quantization_flag": "awq_marlin",
        "tensor_parallel_size": 2,
        "max_model_len": 32768,
    }


def _write_preset(directory: Path, alias: str, body: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{alias}.yaml"
    if body is None:
        body = """
        alias: x
        served_name: x
        model_id: org/x
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """
    path.write_text(textwrap.dedent(body).strip() + "\n")
    return path


def test_public_api_exports_expected_symbols() -> None:
    import llmctl.presets as presets

    assert presets.__all__ == [
        "Model",
        "PresetSchemaError",
        "CANONICAL_QUANTIZATIONS",
        "user_config_dir",
        "default_preset_dir",
        "load_all",
        "load_one",
    ]


def test_model_constructs_and_tracks_explicit_fields() -> None:
    model = Model(
        **_valid_model_kwargs(),
        max_num_seqs=None,
    )

    assert model.max_num_seqs is None
    assert "max_num_seqs" in model.model_fields_set
    assert "gpu_memory_utilization" not in model.model_fields_set


def test_model_rejects_invalid_alias() -> None:
    with pytest.raises(PresetSchemaError):
        Model(**(_valid_model_kwargs() | {"alias": "X-INVALID"}))


def test_model_rejects_invalid_quantization() -> None:
    with pytest.raises(PresetSchemaError):
        Model(**(_valid_model_kwargs() | {"quantization": "bogus"}))


def test_paths_respect_xdg_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    assert default_preset_dir() == tmp_path / "xdg" / "llmctl" / "presets"
    assert user_config_dir() == tmp_path / "xdg" / "llm-models"


def test_load_all_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert load_all(tmp_path) == {}


def test_load_all_reads_valid_preset(tmp_path: Path) -> None:
    _write_preset(tmp_path, "x")

    models = load_all(tmp_path)

    assert models == {"x": Model(**_valid_model_kwargs())}


def test_load_all_skips_underscore_prefixed_files(tmp_path: Path) -> None:
    (tmp_path / "_shortcuts.yaml").write_text("not: a preset\n")
    _write_preset(tmp_path, "x")

    assert list(load_all(tmp_path)) == ["x"]


def test_load_all_skips_malformed_files(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _write_preset(tmp_path, "x")
    (tmp_path / "broken.yaml").write_text("alias: [")

    with caplog.at_level(logging.WARNING):
        models = load_all(tmp_path)

    assert list(models) == ["x"]
    assert "skipping malformed preset" in caplog.text


def test_load_one_missing_returns_none(tmp_path: Path) -> None:
    assert load_one("missing", tmp_path) is None


def test_load_all_legacy_overrides_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _write_preset(
        default_preset_dir(),
        "x",
        """
        alias: x
        served_name: new
        model_id: org/new
        quantization: awq
        vllm_quantization_flag: awq_marlin
        tensor_parallel_size: 2
        max_model_len: 32768
        """,
    )
    _write_preset(user_config_dir(), "x")

    assert load_all()["x"].model_id == "org/x"


def test_migrate_legacy_presets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    legacy = _write_preset(user_config_dir(), "x")

    with caplog.at_level(logging.INFO):
        assert migrate_legacy_presets() == 1

    migrated = default_preset_dir() / "x.yaml"
    assert migrated.is_symlink()
    assert migrated.resolve() == legacy
    assert "migrated 1 presets" in caplog.text


def test_migrate_legacy_presets_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    _write_preset(user_config_dir(), "x")

    assert migrate_legacy_presets() == 1
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert migrate_legacy_presets() == 0

    assert "migrated" not in caplog.text
