"""End-to-end tests for the new model/profile management CLI commands.

Uses CliRunner with input piping so the interactive prompt-driven commands
(``model add``, ``profile create``) can be exercised without a TTY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llmctl.cli import app
from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.schemas import ModelCreate, ProfileCreate
from llmctl.services.profiles import ProfileService
from llmctl.services.registry import RegistryService
from sqlmodel import Session


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated config + DB per test; returns the data dir for inspection."""
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    data_dir.mkdir()
    config_dir.mkdir()
    db_url = f"sqlite:///{data_dir / 'llmctl.sqlite3'}"
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("LLMCTL_DB_URL", db_url)
    init_db(db_url)
    return tmp_path


def _engine_session(tmp_path: Path) -> Session:
    db_url = os.environ["LLMCTL_DB_URL"]
    return Session(get_engine(db_url))


def test_model_add_non_interactive(cli_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "model",
            "add",
            "--non-interactive",
            "--name",
            "llama3-70b",
            "--backend",
            "vllm",
            "--path",
            "/srv/models/llama3-70b",
            "--quantization",
            "awq-int4",
            "--max-context",
            "32768",
            "--estimated-vram",
            "44.5",
            "--tags",
            "coder,reasoning",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Registered model" in result.output

    with _engine_session(cli_env) as db:
        models = RegistryService(db).list_models()
    assert len(models) == 1
    assert models[0].name == "llama3-70b"
    assert models[0].max_context == 32768
    assert models[0].tags == ["coder", "reasoning"]


def test_model_show_clone_edit_disable_enable_delete(cli_env: Path) -> None:
    runner = CliRunner()
    with _engine_session(cli_env) as db:
        model = RegistryService(db).add_model(
            ModelCreate(name="m1", runtime=RuntimeName.VLLM, source="/x", path="/x")
        )
    assert model.id is not None

    result = runner.invoke(app, ["model", "show", "m1"])
    assert result.exit_code == 0
    assert "m1" in result.output

    result = runner.invoke(app, ["model", "clone", "m1", "m1-copy"])
    assert result.exit_code == 0
    assert "Cloned" in result.output

    result = runner.invoke(app, ["model", "edit", "m1-copy", "--notes", "via cli"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["model", "disable", "m1-copy"])
    assert result.exit_code == 0
    assert "Disabled" in result.output

    # Default listing should hide the disabled clone.
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "m1-copy" not in result.output

    result = runner.invoke(app, ["model", "enable", "m1-copy"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["models"])
    assert "m1-copy" in result.output

    result = runner.invoke(app, ["model", "delete", "m1-copy"])
    assert result.exit_code == 0
    assert "Soft-deleted" in result.output


def test_model_delete_files_only_when_flagged(cli_env: Path) -> None:
    artifact = cli_env / "weights.gguf"
    artifact.write_bytes(b"abc")
    runner = CliRunner()
    with _engine_session(cli_env) as db:
        RegistryService(db).add_model(
            ModelCreate(
                name="gguf",
                runtime=RuntimeName.LLAMA_CPP,
                source=str(artifact),
                path=str(artifact),
            )
        )

    # Without --delete-files: artifact must survive.
    result = runner.invoke(app, ["model", "delete", "gguf"])
    assert result.exit_code == 0
    assert artifact.exists(), "artifact must NOT be removed without --delete-files"


def test_profile_lifecycle(cli_env: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "create",
            "--non-interactive",
            "--name",
            "custom-fast",
            "--backend",
            "vllm",
            "--tensor-parallel",
            "1",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Created profile" in result.output

    result = runner.invoke(app, ["profiles"])
    assert "custom-fast" in result.output

    result = runner.invoke(app, ["profile", "show", "custom-fast"])
    assert "tensor_parallel_size" in result.output

    result = runner.invoke(
        app, ["profile", "edit", "custom-fast", "--max-model-len", "16384"]
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app, ["profile", "clone", "custom-fast", "custom-fast-2"]
    )
    assert result.exit_code == 0

    out_path = cli_env / "custom-fast.yaml"
    result = runner.invoke(
        app, ["profile", "export", "custom-fast", str(out_path)]
    )
    assert result.exit_code == 0
    assert out_path.exists()
    data = yaml.safe_load(out_path.read_text())
    assert data["name"] == "custom-fast"
    assert data["parameters"]["max_model_len"] == 16384

    # Round-trip: import the file back into a fresh profile name.
    data["name"] = "custom-fast-imported"
    out_path.write_text(yaml.safe_dump(data))
    result = runner.invoke(app, ["profile", "import", str(out_path)])
    assert result.exit_code == 0
    assert "custom-fast-imported" in result.output

    result = runner.invoke(app, ["profile", "delete", "custom-fast-2"])
    assert result.exit_code == 0


def test_profile_edit_rejects_validation_errors(cli_env: Path) -> None:
    runner = CliRunner()
    with _engine_session(cli_env) as db:
        ProfileService(db).create_profile(
            ProfileCreate(name="warnable", runtime=RuntimeName.VLLM)
        )
    result = runner.invoke(
        app,
        [
            "profile",
            "edit",
            "warnable",
            "--gpu-memory-utilization",
            "1.5",  # invalid: out of range
        ],
    )
    assert result.exit_code != 0
    assert "validation errors" in result.output


def test_export_and_import_registry_round_trip(cli_env: Path) -> None:
    runner = CliRunner()
    with _engine_session(cli_env) as db:
        reg = RegistryService(db)
        reg.add_model(
            ModelCreate(name="export-me", runtime=RuntimeName.VLLM, source="/e", path="/e")
        )
        ProfileService(db).create_profile(
            ProfileCreate(name="export-profile", runtime=RuntimeName.VLLM)
        )

    out_path = cli_env / "bundle.json"
    result = runner.invoke(app, ["export-registry", str(out_path)])
    assert result.exit_code == 0
    assert out_path.exists()
    bundle = json.loads(out_path.read_text())
    assert bundle["version"] == 1
    assert any(m["name"] == "export-me" for m in bundle["models"])
    assert any(p["name"] == "export-profile" for p in bundle["profiles"])

    # Fresh DB: should import everything.
    fresh_db = cli_env / "fresh.sqlite3"
    fresh_url = f"sqlite:///{fresh_db}"
    init_db(fresh_url)
    os.environ["LLMCTL_DB_URL"] = fresh_url

    result = runner.invoke(app, ["import-registry", str(out_path)])
    assert result.exit_code == 0
    assert "Imported" in result.output

    with Session(get_engine(fresh_url)) as db:
        names = {m.name for m in RegistryService(db).list_models()}
        profile_names = {p.name for p in ProfileService(db).list_profiles()}
    assert "export-me" in names
    assert "export-profile" in profile_names


def test_resolve_gpu_mode_handles_spaces() -> None:
    """``--gpus "0, 1"`` must classify as explicit, not pass through as a mode."""
    from llmctl.cli_registry import _resolve_gpu_mode

    assert _resolve_gpu_mode("0") == "explicit"
    assert _resolve_gpu_mode("0,1") == "explicit"
    assert _resolve_gpu_mode("0, 1") == "explicit"
    assert _resolve_gpu_mode(" 0 , 1 ") == "explicit"
    assert _resolve_gpu_mode("auto") == "auto"
    assert _resolve_gpu_mode("most-free") == "most-free"


def test_scan_dry_run_does_not_persist(cli_env: Path) -> None:
    runner = CliRunner()
    # No model dirs configured — should be a clean no-op.
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    with _engine_session(cli_env) as db:
        assert RegistryService(db).list_models() == []
