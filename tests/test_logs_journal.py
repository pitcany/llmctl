"""journalctl log surface for ADOPTED sessions (`llmctl logs <adopted-id>`)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from sqlmodel import Session

from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.integrations.journalctl import JournalctlRunner, JournalResult
from llmctl.services.sessions import SessionService


class _StubJournal(JournalctlRunner):
    """JournalctlRunner that replays canned (rc, stdout, stderr) responses."""

    def __init__(self, responses: list[tuple[int, str, str]], *, present: bool = True) -> None:
        super().__init__(runner=self._record)
        self.calls: list[list[str]] = []
        self._responses = list(responses)
        self._present = present

    def available(self) -> bool:
        return self._present

    def _record(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        rc, out, err = self._responses.pop(0)
        return subprocess.CompletedProcess(argv, rc, out, err)


def _adopted_service(
    tmp_path: Path, journal: JournalctlRunner, *, unit: str | None = "vllm-tp"
) -> tuple[Session, SessionService, str]:
    """Return a DB-backed service plus the id of one adopted session."""
    url = f"sqlite:///{tmp_path / 'journal.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    service = SessionService(
        db,
        probe=lambda _u, _t: ["ornith-35b"],
        gpu_ids_for_unit=lambda _u: [],
        journalctl=journal,
    )
    session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit=unit)
    return db, service, session.id


def test_adopted_tail_log_reads_system_journal(tmp_path: Path) -> None:
    journal = _StubJournal([(0, "line-1\nline-2\n", "")])
    db, service, sid = _adopted_service(tmp_path, journal)
    try:
        assert service.tail_log(sid, lines=7) == "line-1\nline-2\n"
        assert journal.calls == [
            ["journalctl", "-u", "vllm-tp", "-n", "7", "--no-pager", "-o", "short-iso"]
        ]
    finally:
        db.close()


def test_adopted_tail_log_falls_back_to_user_journal(tmp_path: Path) -> None:
    journal = _StubJournal([(0, "-- No entries --\n", ""), (0, "user-line\n", "")])
    db, service, sid = _adopted_service(tmp_path, journal)
    try:
        assert service.tail_log(sid) == "user-line\n"
        assert journal.calls[1][:2] == ["journalctl", "--user"]
    finally:
        db.close()


def test_adopted_tail_log_empty_when_no_journal_entries(tmp_path: Path) -> None:
    journal = _StubJournal([(0, "", ""), (0, "-- No entries --\n", "")])
    db, service, sid = _adopted_service(tmp_path, journal)
    try:
        assert service.tail_log(sid) == ""
        assert len(journal.calls) == 2
    finally:
        db.close()


def test_adopted_tail_log_surfaces_journalctl_failure(tmp_path: Path) -> None:
    journal = _StubJournal([(1, "", "Hint: You are currently not seeing messages.")])
    db, service, sid = _adopted_service(tmp_path, journal)
    try:
        content = service.tail_log(sid)
        assert content is not None
        assert "journalctl -u vllm-tp failed" in content
        assert "not seeing messages" in content
        # A failed system query must not trigger the user-scope fallback.
        assert len(journal.calls) == 1
    finally:
        db.close()


def test_adopted_tail_log_reports_missing_journalctl(tmp_path: Path) -> None:
    journal = _StubJournal([], present=False)
    db, service, sid = _adopted_service(tmp_path, journal)
    try:
        content = service.tail_log(sid)
        assert content is not None
        assert "journalctl not found" in content
        assert journal.calls == []
    finally:
        db.close()


def test_adopted_tail_log_without_unit_stays_empty(tmp_path: Path) -> None:
    journal = _StubJournal([])
    db, service, sid = _adopted_service(tmp_path, journal, unit=None)
    try:
        assert service.tail_log(sid) == ""
        assert journal.calls == []
    finally:
        db.close()


def test_tail_unit_clamps_lines_to_at_least_one() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    JournalctlRunner(runner=runner).tail_unit("vllm-tp", lines=0)
    assert "-n" in calls[0]
    assert calls[0][calls[0].index("-n") + 1] == "1"


def test_tail_unit_times_out_cleanly(monkeypatch) -> None:
    def slow_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", slow_run)
    result = JournalctlRunner().tail_unit("vllm-tp", lines=5)
    assert result.returncode == 124
    assert "timed out" in result.stderr
    assert not result.has_entries


def test_journal_timeout_diagnostic_is_not_doubled(tmp_path: Path) -> None:
    """The service-level message must not repeat the journalctl command echo."""
    journal = _StubJournal([(124, "", "timed out after 30s")])
    db, service, sid = _adopted_service(tmp_path, journal)
    try:
        content = service.tail_log(sid)
        assert content == "journalctl -u vllm-tp failed: timed out after 30s"
    finally:
        db.close()


def test_journal_result_has_entries_semantics() -> None:
    assert JournalResult(0, "real log line\n", "").has_entries
    assert not JournalResult(0, "-- No entries --\n", "").has_entries
    assert not JournalResult(0, "   \n", "").has_entries
    assert not JournalResult(1, "output despite failure", "boom").has_entries
