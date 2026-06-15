"""Tests for the `llmctl model prune` CLI command."""

from __future__ import annotations

from sqlmodel import Session, select
from typer.testing import CliRunner

from llmctl import cli_registry
from llmctl.db import ModelRecord, ModelStatus, RuntimeName, get_engine, init_db


def _seed_missing(url: str, name: str, runtime: RuntimeName) -> None:
    with Session(get_engine(url)) as db:
        db.add(
            ModelRecord(
                name=name, runtime=runtime, source=name,
                status=ModelStatus.MISSING, active=True,
            )
        )
        db.commit()


def _status(url: str, name: str) -> ModelStatus:
    with Session(get_engine(url)) as db:
        record = db.exec(select(ModelRecord).where(ModelRecord.name == name)).first()
        assert record is not None
        return record.status


def test_model_prune_soft_deletes_missing(tmp_path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path}/cli.db"
    init_db(url)
    _seed_missing(url, "ghost", RuntimeName.OLLAMA)
    monkeypatch.setattr(cli_registry, "_session", lambda: Session(get_engine(url)))

    result = CliRunner().invoke(cli_registry.model_app, ["prune", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Pruned 1" in result.output
    assert _status(url, "ghost") == ModelStatus.DELETED


def test_model_prune_no_missing_is_noop(tmp_path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path}/cli_empty.db"
    init_db(url)
    monkeypatch.setattr(cli_registry, "_session", lambda: Session(get_engine(url)))

    result = CliRunner().invoke(cli_registry.model_app, ["prune", "--yes"])

    assert result.exit_code == 0, result.output
    assert "No missing models" in result.output
