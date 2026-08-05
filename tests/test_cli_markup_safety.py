"""CLI output must never parse process text as Rich console markup.

The mirror of ``test_tui_markup_safety.py`` for the Typer front-end. Rich's
``Console.print`` runs console markup over every ``str`` it is handed, and the
text llmctl prints is bracket-rich: ``[/INST]`` chat-template echoes, ``[rank0]:``
torch prefixes, ``[/mnt/...]`` paths. An unbalanced closing tag raises
``MarkupError`` -- a traceback instead of the log, exactly when the operator is
debugging a failed launch -- and a stray ``[word]`` is silently deleted, showing
a log that differs from the file on disk.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from sqlmodel import Session
from typer.testing import CliRunner

from llmctl import cli
from llmctl.db import EventLevel, get_engine, init_db
from llmctl.services.sessions import SessionService

#: A chat-template token vLLM echoes at startup -- the hard-crash case.
HOSTILE_TAIL = "RuntimeError: bad dir [/opt/models/x] template [/INST]"
#: Torch distributed prefix -- the silent-deletion case.
RANK_TAIL = "ERROR [rank0]: CUDA out of memory"


def _isolated_session(tmp_path, monkeypatch):
    """Point the CLI at a throwaway SQLite file instead of the real registry."""
    url = f"sqlite:///{tmp_path}/markup.db"
    init_db(url)
    monkeypatch.setattr(cli, "_session", lambda: Session(get_engine(url)))
    return url


def test_logs_with_closing_tag_does_not_crash(tmp_path, monkeypatch) -> None:
    """'[/INST]' in a tail must print, not raise MarkupError."""
    _isolated_session(tmp_path, monkeypatch)
    monkeypatch.setattr(SessionService, "tail_log", lambda self, sid, lines=50: HOSTILE_TAIL)

    result = CliRunner().invoke(cli.app, ["logs", "sess-1"])

    assert result.exit_code == 0, result.output
    assert result.exception is None, repr(result.exception)
    assert "[/INST]" in result.output, f"closing tag was eaten: {result.output!r}"


def test_logs_preserves_rank_prefix(tmp_path, monkeypatch) -> None:
    """'[rank0]:' is the text the operator needs; it must not be deleted."""
    _isolated_session(tmp_path, monkeypatch)
    monkeypatch.setattr(SessionService, "tail_log", lambda self, sid, lines=50: RANK_TAIL)

    result = CliRunner().invoke(cli.app, ["logs", "sess-1"])

    assert result.exit_code == 0, result.output
    assert "[rank0]:" in result.output, f"rank prefix was eaten as markup: {result.output!r}"


def test_logs_missing_session_still_reports_cleanly(tmp_path, monkeypatch) -> None:
    """The not-found path is unchanged by the escaping fix."""
    _isolated_session(tmp_path, monkeypatch)
    monkeypatch.setattr(SessionService, "tail_log", lambda self, sid, lines=50: None)

    result = CliRunner().invoke(cli.app, ["logs", "ghost"])

    assert result.exit_code == 2, result.output
    assert "ghost" in result.output


def test_events_table_survives_bracketed_message(tmp_path, monkeypatch) -> None:
    """Event messages embed log lines, so they carry the same brackets."""
    _isolated_session(tmp_path, monkeypatch)
    event = SimpleNamespace(
        created_at=datetime(2026, 8, 5, 12, 0, 0),
        level=EventLevel.ERROR,
        category="session",
        message=HOSTILE_TAIL,
    )
    monkeypatch.setattr(
        "llmctl.services.events.list_events", lambda db, limit=50: [event]
    )

    result = CliRunner().invoke(cli.app, ["logs"])

    assert result.exit_code == 0, result.output
    assert result.exception is None, repr(result.exception)
    # Rich wraps the table cell at the column width, so assert on a token that
    # survives wrapping rather than the whole line.
    assert "INST" in result.output, f"message was eaten as markup: {result.output!r}"
