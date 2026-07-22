"""CLI smoke tests."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from llmctl.cli import app

_SKIP_HELP_RENDER_ON_CI = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason=(
        "Typer/Rich help rendering on GH Actions runners drops flag names "
        "from the captured stdout regardless of COLUMNS; behavior is covered "
        "locally and by the parser-level tests."
    ),
)


def test_cli_help_lists_required_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = result.output
    for command in [
        "scan",
        "models",
        "gpus",
        "sessions",
        "add-model",
        "delete-model",
        "start",
        "stop",
        "restart",
        "logs",
        "bench",
        "tui",
        "serve",
        "generate-systemd",
        "adopt",
        "adopt-managed",
        "detach",
        "reconcile",
    ]:
        assert command in output


@_SKIP_HELP_RENDER_ON_CI
def test_cli_adopt_help_includes_endpoint_and_runtime() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["adopt", "--help"])
    assert result.exit_code == 0
    for token in ("--endpoint", "--runtime", "--unit", "--served-name"):
        assert token in result.output


@_SKIP_HELP_RENDER_ON_CI
def test_cli_adopt_managed_help_lists_all_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["adopt-managed", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output
    assert "vllm-tp" in result.output


def test_cli_adopt_managed_requires_role_or_all() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["adopt-managed"])
    assert result.exit_code != 0
    assert "role" in result.output.lower() or "--all" in result.output


def test_cli_adopt_managed_rejects_role_and_all_together() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["adopt-managed", "vllm-tp", "--all"])
    assert result.exit_code != 0


def test_cli_detach_help_mentions_adopted() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["detach", "--help"])
    assert result.exit_code == 0
    assert "adopted" in result.output.lower()


# -- AdoptError handling in stop/restart -------------------------------------


def _cli_db_with_adopted_row(tmp_path) -> tuple[str, str]:
    """Create a SQLite DB with a single RUNNING+ADOPTED row; return (url, id)."""
    from sqlmodel import Session

    from llmctl.db import (
        RuntimeName,
        SessionKind,
        SessionRecord,
        SessionStatus,
        get_engine,
        init_db,
        utcnow,
    )

    db_url = f"sqlite:///{tmp_path / 'cli.sqlite3'}"
    init_db(db_url)
    with Session(get_engine(db_url)) as db:
        record = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.RUNNING,
            kind=SessionKind.ADOPTED,
            endpoint_url="http://127.0.0.1:8003",
            port=8003,
            served_name="llama-3.3-70b",
            systemd_unit="vllm-tp.service",
            adopted_at=utcnow(),
            launch_plan={
                "runtime": "vllm",
                "command": [],
                "endpoint_url": "http://127.0.0.1:8003",
                "dry_run": False,
            },
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return db_url, record.id


def _patch_session(monkeypatch, db_url: str) -> None:
    """Force the CLI's _session() to point at our temp DB."""
    from sqlmodel import Session

    import llmctl.cli as cli_mod
    from llmctl.db import get_engine

    monkeypatch.setattr(cli_mod, "_session", lambda: Session(get_engine(db_url)))


def test_cli_stop_adopted_prints_refusal_and_exits_nonzero(tmp_path, monkeypatch) -> None:
    db_url, session_id = _cli_db_with_adopted_row(tmp_path)
    _patch_session(monkeypatch, db_url)
    runner = CliRunner()
    result = runner.invoke(app, ["stop", session_id])
    assert result.exit_code == 1
    assert "refused" in result.output.lower()
    assert "adopted" in result.output.lower()


def test_cli_restart_adopted_prints_refusal_and_exits_nonzero(tmp_path, monkeypatch) -> None:
    db_url, session_id = _cli_db_with_adopted_row(tmp_path)
    _patch_session(monkeypatch, db_url)
    runner = CliRunner()
    result = runner.invoke(app, ["restart", session_id])
    assert result.exit_code == 1
    assert "refused" in result.output.lower()
    assert "adopted" in result.output.lower()


# -- reconcile command -------------------------------------------------------


def test_cli_install_systemd_refuses_adopted_single_id(tmp_path, monkeypatch) -> None:
    """install-systemd <adopted_id> refuses with exit 1 and a clear message."""
    db_url, session_id = _cli_db_with_adopted_row(tmp_path)
    _patch_session(monkeypatch, db_url)
    runner = CliRunner()
    result = runner.invoke(app, ["install-systemd", "--session-id", session_id])
    assert result.exit_code == 1
    assert "refused" in result.output.lower()
    assert "adopted" in result.output.lower()


def test_cli_generate_systemd_session_refuses_adopted(tmp_path, monkeypatch) -> None:
    """generate-systemd-session refuses adopted with exit 1."""
    db_url, session_id = _cli_db_with_adopted_row(tmp_path)
    _patch_session(monkeypatch, db_url)
    runner = CliRunner()
    result = runner.invoke(app, ["generate-systemd-session", session_id])
    assert result.exit_code == 1
    assert "refused" in result.output.lower()
    assert "adopted" in result.output.lower()


def test_cli_install_systemd_all_skips_adopted(tmp_path, monkeypatch) -> None:
    """install-systemd --all logs a skip for adopted and continues."""
    db_url, _adopted_id = _cli_db_with_adopted_row(tmp_path)
    _patch_session(monkeypatch, db_url)
    runner = CliRunner()
    result = runner.invoke(app, ["install-systemd", "--all"])
    # Exit 0 (skipping isn't a failure); message names the adopted skip
    # and the "no owned sessions" trailer.
    assert result.exit_code == 0
    assert "skip adopted" in result.output.lower()
    assert "no owned sessions" in result.output.lower()


def test_cli_reconcile_reports_no_changes_on_empty_db(tmp_path, monkeypatch) -> None:
    """An empty DB has nothing to reconcile; the command exits 0 cleanly."""
    from llmctl.db import init_db

    db_url = f"sqlite:///{tmp_path / 'recon.sqlite3'}"
    init_db(db_url)
    _patch_session(monkeypatch, db_url)
    runner = CliRunner()
    result = runner.invoke(app, ["reconcile"])
    assert result.exit_code == 0
    assert "no changes" in result.output.lower()


def _isolate_validate(tmp_path, monkeypatch) -> str:
    """Point every source `validate` reads at a temp dir; return the DB url.

    Covers presets (XDG), settings + model_dirs (LLMCTL_CONFIG_DIR), the
    registry DB, and systemd — the port check must not probe the live
    stack from a unit test.
    """
    from llmctl.db import init_db

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("LLMCTL_CONFIG_DIR", str(tmp_path / "config" / "llmctl"))
    monkeypatch.setattr(
        "llmctl.integrations.systemctl.SystemctlRunner.available", lambda self: False
    )
    db_url = f"sqlite:///{tmp_path / 'validate.sqlite3'}"
    init_db(db_url)
    _patch_session(monkeypatch, db_url)
    return db_url


def test_cli_validate_passes_on_a_clean_host(tmp_path, monkeypatch) -> None:
    """Nothing configured means nothing broken — exit 0, not a false alarm."""
    _isolate_validate(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["validate"])
    assert result.exit_code == 0
    assert "validation passed" in result.output.lower()


def test_cli_validate_reports_findings_and_exits_nonzero(tmp_path, monkeypatch) -> None:
    """A preset pointing at a deleted checkpoint must fail the command."""
    _isolate_validate(tmp_path, monkeypatch)
    presets = tmp_path / "config" / "llmctl" / "presets"
    presets.mkdir(parents=True)
    (presets / "stale.yaml").write_text(
        "alias: stale\n"
        "served_name: stale\n"
        f"model_id: {tmp_path / 'deleted-checkpoint'}\n"
        "quantization: fp8\n"
        "vllm_quantization_flag: fp8\n"
        "tensor_parallel_size: 2\n"
        "max_model_len: 4096\n"
    )
    result = CliRunner().invoke(app, ["validate"])
    assert result.exit_code == 1
    assert "preset-model-missing" in result.output
    assert "stale" in result.output
