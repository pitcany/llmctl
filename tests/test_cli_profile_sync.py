"""Smoke test for ``llmctl profile sync``.

The sync command re-seeds the seven shipped defaults from
``configs/profiles.yaml`` without touching user-created profiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session
from typer.testing import CliRunner

from llmctl.cli import app
from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.schemas import ProfileCreate
from llmctl.services.profiles import ProfileService

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_url = f"sqlite:///{tmp_path / 'profiles.sqlite3'}"
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(CONFIGS))
    monkeypatch.setenv("LLMCTL_DB_URL", db_url)
    init_db(db_url)
    return tmp_path


def test_profile_sync_loads_defaults_and_preserves_custom(cli_env: Path) -> None:
    runner = CliRunner()
    with Session(get_engine(f"sqlite:///{cli_env / 'profiles.sqlite3'}")) as db:
        ProfileService(db).create_profile(
            ProfileCreate(name="user-custom", runtime=RuntimeName.VLLM)
        )

    result = runner.invoke(app, ["profile", "sync"])
    assert result.exit_code == 0, result.output
    assert "Synced" in result.output

    with Session(get_engine(f"sqlite:///{cli_env / 'profiles.sqlite3'}")) as db:
        names = {p.name for p in ProfileService(db).list_profiles()}

    expected_defaults = {
        "fast",
        "coding",
        "reasoning",
        "long-context",
        "quant",
        "adtech",
        "tutoring",
    }
    assert expected_defaults.issubset(names)
    assert "user-custom" in names
