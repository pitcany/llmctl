"""Phase-2 adopt flow: SessionService.adopt, reconcile probe, stop/restart refusal."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from sqlmodel import Session, select

from llmctl.db import RuntimeName, SessionKind, SessionRecord, SessionStatus, get_engine, init_db
from llmctl.integrations.systemctl import SystemctlRunner
from llmctl.services.sessions import AdoptError, AmbiguousSessionError, SessionService


def _make_service(
    tmp_path: Path,
    probe: Callable[[str, float], list[str] | None],
    *,
    db_name: str = "adopt.sqlite3",
    systemctl: SystemctlRunner | None = None,
) -> tuple[Session, SessionService]:
    """Wire up an isolated DB-backed SessionService with the supplied probe."""
    url = f"sqlite:///{tmp_path / db_name}"
    init_db(url)
    db = Session(get_engine(url))
    service = SessionService(db, probe=probe, systemctl=systemctl)
    return db, service


def test_adopt_inserts_running_adopted_record(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda url, _t: ["llama-3.3-70b"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        assert session.status == SessionStatus.RUNNING
        assert session.kind == SessionKind.ADOPTED
        assert session.served_name == "llama-3.3-70b"
        assert session.endpoint_url == "http://127.0.0.1:8003"
        assert session.port == 8003
        assert session.adopted_at is not None
        assert session.started_at is not None
        assert session.systemd_unit is None
        # round-trip through DB
        record = db.exec(select(SessionRecord).where(SessionRecord.id == session.id)).one()
        assert record.kind == SessionKind.ADOPTED
        assert record.served_name == "llama-3.3-70b"
        assert record.endpoint_url == "http://127.0.0.1:8003"
    finally:
        db.close()


def test_adopt_uses_first_probed_model_when_no_served_name(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["first", "second"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        assert session.served_name == "first"
    finally:
        db.close()


def test_adopt_with_explicit_served_name_and_unit(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["upstream-id"])
    try:
        session = service.adopt(
            RuntimeName.VLLM,
            "http://127.0.0.1:8003",
            served_name="llama-3.3-70b",
            systemd_unit="vllm-tp.service",
        )
        assert session.served_name == "llama-3.3-70b"
        assert session.systemd_unit == "vllm-tp.service"
    finally:
        db.close()


def test_adopt_trailing_slash_is_normalized(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003/")
        assert session.endpoint_url == "http://127.0.0.1:8003"
    finally:
        db.close()


def test_adopt_probe_failure_raises(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: None)
    try:
        with pytest.raises(AdoptError, match="Probe of"):
            service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        assert db.exec(select(SessionRecord)).all() == []
    finally:
        db.close()


def test_adopt_empty_probe_raises(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: [])
    try:
        with pytest.raises(AdoptError):
            service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
    finally:
        db.close()


def test_adopt_empty_endpoint_raises(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        with pytest.raises(AdoptError, match="non-empty endpoint_url"):
            service.adopt(RuntimeName.VLLM, "")
    finally:
        db.close()


def test_adopt_duplicate_active_endpoint_refuses(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        first = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        assert first.status == SessionStatus.RUNNING
        with pytest.raises(AdoptError, match="already tracked"):
            service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
    finally:
        db.close()


def test_reconcile_adopted_probe_failure_marks_stopped(tmp_path: Path) -> None:
    probe_responses: list[list[str] | None] = [["m"], None]

    def probe(_url: str, _t: float) -> list[str] | None:
        return probe_responses.pop(0) if probe_responses else None

    db, service = _make_service(tmp_path, probe)
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        changed = service.reconcile()
        assert changed == 1
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.STOPPED
        assert refreshed.kind == SessionKind.ADOPTED
        assert refreshed.error is not None
        assert "no longer responds" not in (refreshed.error or "")  # internal detail
        assert "Adopted endpoint" in (refreshed.error or "")
    finally:
        db.close()


def test_reconcile_adopted_probe_success_revives_stopped(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        # Force-stop the adopted record directly (simulating a prior outage).
        record = db.exec(select(SessionRecord).where(SessionRecord.id == session.id)).one()
        record.status = SessionStatus.STOPPED
        record.error = "Adopted endpoint failed to respond on /v1/models."
        db.add(record)
        db.commit()
        changed = service.reconcile()
        assert changed == 1
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.RUNNING
        assert refreshed.error is None
        assert refreshed.kind == SessionKind.ADOPTED
    finally:
        db.close()


def test_stop_adopted_refuses(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        with pytest.raises(AdoptError, match="adopted"):
            service.stop(session.id)
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.RUNNING  # unchanged
    finally:
        db.close()


def test_restart_adopted_refuses(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        with pytest.raises(AdoptError, match="adopted"):
            service.restart(session.id)
    finally:
        db.close()


def _recording_systemctl(
    returncode: int = 0, stderr: str = ""
) -> tuple[list[list[str]], SystemctlRunner]:
    """A SystemctlRunner whose subprocess calls are captured, not executed."""
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, "", stderr)

    return calls, SystemctlRunner(runner=fake)


def test_stop_adopted_with_unit_and_flag_stops_unit(tmp_path: Path) -> None:
    calls, runner = _recording_systemctl()
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp"
        )
        result = service.stop(session.id, stop_unit=True)
        assert result is not None
        assert result.status == SessionStatus.STOPPED
        # the row is kept (detach deletes; stop marks stopped)
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.STOPPED
        # exactly one `systemctl stop vllm-tp` was issued
        assert any("stop" in argv and "vllm-tp" in argv for argv in calls)
    finally:
        db.close()


def test_stop_adopted_user_unit_uses_user_scope(tmp_path: Path) -> None:
    """A user-scope unit must be stopped with ``--user`` and without sudo.

    The llama.cpp servers (deepseek-v4-flash-0731, gpt-oss-120b) are user
    units. Stopping them as ``sudo systemctl stop`` looks for a system unit
    of that name and fails, and the NOPASSWD contract does not cover them.
    """
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        # Only the user manager has this unit loaded.
        if "show" in argv:
            state = "loaded" if "--user" in argv else "not-found"
            return subprocess.CompletedProcess(argv, 0, f"{state}\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    runner = SystemctlRunner(runner=fake)
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(
            RuntimeName.LLAMA_CPP, "http://127.0.0.1:8005", systemd_unit="gpt-oss-120b"
        )
        result = service.stop(session.id, stop_unit=True)
        assert result is not None
        assert result.status == SessionStatus.STOPPED

        stops = [argv for argv in calls if "stop" in argv]
        assert stops == [["systemctl", "--user", "stop", "gpt-oss-120b"]]
        assert "sudo" not in stops[0]
    finally:
        db.close()


def test_stop_adopted_flag_without_unit_still_refuses(tmp_path: Path) -> None:
    calls, runner = _recording_systemctl()
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")  # no unit
        with pytest.raises(AdoptError, match="adopted"):
            service.stop(session.id, stop_unit=True)
        assert calls == []  # never shelled out to systemctl
    finally:
        db.close()


def test_restart_adopted_with_unit_and_flag_restarts_unit(tmp_path: Path) -> None:
    """`restart --systemd` mirrors `stop --systemd` for a system-scope unit."""
    calls, runner = _recording_systemctl()
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp"
        )
        result = service.restart(session.id, restart_unit=True)
        assert result is not None
        # STARTING, not RUNNING: systemd accepted the restart but the endpoint
        # answers only once the model is loaded. reconcile promotes it.
        assert result.status == SessionStatus.STARTING
        assert result.error is None
        restarts = [argv for argv in calls if "restart" in argv]
        assert restarts == [["sudo", "systemctl", "restart", "vllm-tp"]]
    finally:
        db.close()


def test_restart_adopted_user_unit_uses_user_scope(tmp_path: Path) -> None:
    """The llama.cpp servers are user units — no sudo, `--user` instead."""
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if "show" in argv:
            state = "loaded" if "--user" in argv else "not-found"
            return subprocess.CompletedProcess(argv, 0, f"{state}\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    db, service = _make_service(
        tmp_path, lambda u, _t: ["m"], systemctl=SystemctlRunner(runner=fake)
    )
    try:
        session = service.adopt(
            RuntimeName.LLAMA_CPP, "http://127.0.0.1:8005", systemd_unit="gpt-oss-120b"
        )
        result = service.restart(session.id, restart_unit=True)
        assert result is not None
        assert result.status == SessionStatus.STARTING
        restarts = [argv for argv in calls if "restart" in argv]
        assert restarts == [["systemctl", "--user", "restart", "gpt-oss-120b"]]
        assert "sudo" not in restarts[0]
    finally:
        db.close()


def test_restart_adopted_revives_a_stopped_row(tmp_path: Path) -> None:
    """`systemctl restart` starts an inactive unit, so a stopped row comes back.

    This is the half of the symmetry that `stop --systemd` alone left missing:
    without it an operator could take an adopted unit down through llmctl but
    not bring it back.
    """
    calls, runner = _recording_systemctl()
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp"
        )
        stopped = service.stop(session.id, stop_unit=True)
        assert stopped is not None and stopped.status == SessionStatus.STOPPED

        revived = service.restart(session.id, restart_unit=True)
        assert revived is not None
        assert revived.status == SessionStatus.STARTING
        assert revived.stopped_at is None
    finally:
        db.close()


def test_restart_adopted_flag_without_unit_still_refuses(tmp_path: Path) -> None:
    calls, runner = _recording_systemctl()
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")  # no unit
        with pytest.raises(AdoptError, match="adopted"):
            service.restart(session.id, restart_unit=True)
        assert calls == []  # never shelled out to systemctl
    finally:
        db.close()


def test_restart_adopted_systemctl_failure_raises(tmp_path: Path) -> None:
    _calls, runner = _recording_systemctl(returncode=1, stderr="Failed to restart")
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp"
        )
        with pytest.raises(AdoptError, match="failed"):
            service.restart(session.id, restart_unit=True)
        # a failed restart must not claim the session is coming back
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.RUNNING
    finally:
        db.close()


def test_stop_adopted_systemctl_failure_raises(tmp_path: Path) -> None:
    _calls, runner = _recording_systemctl(returncode=1, stderr="Failed to stop")
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp"
        )
        with pytest.raises(AdoptError, match="failed"):
            service.stop(session.id, stop_unit=True)
        # the failed stop must not have flipped the row to STOPPED
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.RUNNING
    finally:
        db.close()


def test_reconcile_probes_fresh_each_call(tmp_path: Path) -> None:
    """No probe cache: each reconcile pass probes adopted endpoints fresh.

    The earlier TTL cache was removed because (a) on the API path each
    request constructed a new SessionService so the TTL never engaged,
    and (b) within a single reconcile pass each row is iterated once so
    dedup adds nothing. Fresh probes guarantee ground truth and avoid
    the stale-liveness window the cache introduced.
    """
    call_count = {"n": 0}

    def probe(_url: str, _t: float) -> list[str] | None:
        call_count["n"] += 1
        return ["m"]

    db, service = _make_service(tmp_path, probe)
    try:
        service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        n_after_adopt = call_count["n"]
        for _ in range(3):
            service.reconcile()
        # adopt() probed once; each reconcile probes once.
        assert call_count["n"] == n_after_adopt + 3
    finally:
        db.close()


def test_list_sessions_does_not_reconcile(tmp_path: Path) -> None:
    """list_sessions is a pure DB read — no probe, no PID check, no transition.

    Phase-2 used to call reconcile() from list_sessions, which made the
    API path probe every adopted endpoint synchronously per request.
    This proves the call is gone: after adopt(), if we directly flip the
    record to STOPPED behind the service's back, list_sessions returns
    STOPPED with no probe attempt.
    """
    call_count = {"n": 0}

    def probe(_url: str, _t: float) -> list[str] | None:
        call_count["n"] += 1
        return ["m"]

    db, service = _make_service(tmp_path, probe)
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        baseline = call_count["n"]
        record = db.exec(select(SessionRecord).where(SessionRecord.id == session.id)).one()
        record.status = SessionStatus.STOPPED
        db.add(record)
        db.commit()
        sessions = service.list_sessions()
        # Probe count did not change.
        assert call_count["n"] == baseline
        # And the DB state shows through unchanged.
        assert len(sessions) == 1
        assert sessions[0].status == SessionStatus.STOPPED
    finally:
        db.close()


def test_cleanup_still_calls_reconcile(tmp_path: Path) -> None:
    """cleanup() must keep its explicit reconcile call so dead rows get marked."""
    probed: list[str] = []

    def probe(url: str, _t: float) -> list[str] | None:
        probed.append(url)
        return None  # endpoint is down

    db, service = _make_service(tmp_path, probe)
    try:
        # Adopt with a fake probe that returns ["m"] just for the adopt() call,
        # then swap the probe to None so cleanup -> reconcile flips it.
        service._probe = lambda u, _t: ["m"]
        service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        service._probe = probe  # now reports endpoint down
        report = service.cleanup()
        assert report["dead_marked"] == 1
        assert "http://127.0.0.1:8003" in probed
    finally:
        db.close()


def test_detach_removes_adopted_record(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        removed = service.detach(session.id)
        assert removed is not None
        assert removed.id == session.id
        assert removed.kind == SessionKind.ADOPTED
        assert service.get_session(session.id) is None
    finally:
        db.close()


def test_detach_unknown_session_returns_none(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        assert service.detach("does-not-exist") is None
    finally:
        db.close()


def test_detach_owned_session_refuses(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        # Insert an OWNED row directly (no spawn path needed for this assertion).
        record = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.STOPPED,
            kind=SessionKind.OWNED,
            endpoint_url="http://127.0.0.1:9999",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        with pytest.raises(AdoptError, match="not adopted"):
            service.detach(record.id)
        # Record is untouched.
        assert service.get_session(record.id) is not None
    finally:
        db.close()


def test_detach_then_readopt_succeeds(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        first = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        service.detach(first.id)
        second = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        assert second.id != first.id
        assert second.kind == SessionKind.ADOPTED
        assert second.status == SessionStatus.RUNNING
    finally:
        db.close()


def test_normalize_endpoint_url_folds_loopback_aliases() -> None:
    """Unit: localhost <-> 127.0.0.1 + scheme/host casing + trailing slash."""
    from llmctl.services.sessions import SessionService

    norm = SessionService._normalize_endpoint_url
    assert norm("http://localhost:8003") == "http://127.0.0.1:8003"
    assert norm("http://LOCALHOST:8003/") == "http://127.0.0.1:8003"
    assert norm("HTTP://127.0.0.1:8003") == "http://127.0.0.1:8003"
    assert norm("http://127.0.0.1:8003") == "http://127.0.0.1:8003"
    # IPv6 loopback is intentionally not folded (different protocol family).
    assert "127.0.0.1" not in norm("http://[::1]:8003")


def test_adopt_normalizes_localhost_at_storage(tmp_path: Path) -> None:
    """adopt() with localhost URL stores 127.0.0.1; alias collision blocked."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        first = service.adopt(RuntimeName.VLLM, "http://localhost:8003")
        assert first.endpoint_url == "http://127.0.0.1:8003"
        # Adopting the alias must now refuse.
        with pytest.raises(AdoptError, match="already tracked"):
            service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        # Inverse direction also refuses.
        with pytest.raises(AdoptError, match="already tracked"):
            service.adopt(RuntimeName.VLLM, "http://LOCALHOST:8003/")
    finally:
        db.close()


def test_adopt_refuses_against_planned_row_at_same_url(tmp_path: Path) -> None:
    """A PLANNED row (e.g. a dry-run start) reserves the URL — adopt must refuse.

    Bugbot flagged: the original duplicate-check only fired on ACTIVE +
    STOPPED-ADOPTED, letting a PLANNED row slip through; a second RUNNING
    adopted record at the same endpoint would then fork gateway routing.
    """
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        planned = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.PLANNED,
            kind=SessionKind.OWNED,
            endpoint_url="http://127.0.0.1:8003",
            port=8003,
        )
        db.add(planned)
        db.commit()
        with pytest.raises(AdoptError, match="already tracked"):
            service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        rows = db.exec(
            select(SessionRecord).where(SessionRecord.endpoint_url == "http://127.0.0.1:8003")
        ).all()
        assert len(rows) == 1
        assert rows[0].status == SessionStatus.PLANNED
    finally:
        db.close()


def test_adopt_allowed_after_owned_failed_at_same_url(tmp_path: Path) -> None:
    """FAILED OWNED rows are terminal and must not block a fresh adopt."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        failed = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.FAILED,
            kind=SessionKind.OWNED,
            endpoint_url="http://127.0.0.1:8003",
            port=8003,
            error="prior crash",
        )
        db.add(failed)
        db.commit()
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        assert session.status == SessionStatus.RUNNING
        assert session.kind == SessionKind.ADOPTED
    finally:
        db.close()


def test_adopt_duplicate_stopped_adopted_refuses(tmp_path: Path) -> None:
    """Re-adopting a URL with a STOPPED ADOPTED row refuses and names detach.

    Without this check, the auto-revival path in reconcile() would resurrect
    the prior STOPPED row right after the new RUNNING row was inserted,
    leaving two RUNNING records pointing at the same endpoint_url and making
    gateway routing ambiguous.
    """
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        first = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        # Force-flip the prior adopted row to STOPPED without going through
        # reconcile so the test isn't fighting the auto-revive path.
        record = db.exec(select(SessionRecord).where(SessionRecord.id == first.id)).one()
        record.status = SessionStatus.STOPPED
        db.add(record)
        db.commit()
        with pytest.raises(AdoptError, match="auto-revive"):
            service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        # Only one record persists at this URL.
        rows = db.exec(
            select(SessionRecord).where(SessionRecord.endpoint_url == "http://127.0.0.1:8003")
        ).all()
        assert len(rows) == 1
        assert rows[0].id == first.id
    finally:
        db.close()


def test_reconcile_query_skips_owned_stopped(tmp_path: Path) -> None:
    """OWNED STOPPED rows must not be probed (they have no endpoint to probe).

    Greptile flagged the widened reconcile query loading all stopped session
    history; this asserts that an OWNED STOPPED row coexisting with an
    ADOPTED RUNNING row never triggers a probe call.
    """
    probe_targets: list[str] = []

    def probe(url: str, _t: float) -> list[str] | None:
        probe_targets.append(url)
        return ["m"]

    db, service = _make_service(tmp_path, probe)
    try:
        # Plant an OWNED STOPPED row that reconcile must not consider.
        owned = SessionRecord(
            runtime=RuntimeName.VLLM,
            status=SessionStatus.STOPPED,
            kind=SessionKind.OWNED,
            endpoint_url="http://127.0.0.1:9999",
        )
        db.add(owned)
        db.commit()
        service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        probe_targets.clear()
        service.reconcile()
        # Reconcile probed only the adopted URL, not the OWNED STOPPED row's URL.
        assert probe_targets == ["http://127.0.0.1:8003"]
    finally:
        db.close()


def test_list_sessions_round_trips_adopt_fields(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["llama-3.3-70b"])
    try:
        service.adopt(
            RuntimeName.VLLM,
            "http://127.0.0.1:8003",
            served_name="llama-3.3-70b",
            systemd_unit="vllm-tp.service",
        )
        sessions = service.list_sessions()
        assert len(sessions) == 1
        s = sessions[0]
        assert s.kind == SessionKind.ADOPTED
        assert s.served_name == "llama-3.3-70b"
        assert s.systemd_unit == "vllm-tp.service"
        assert s.adopted_at is not None
        assert s.endpoint_url == "http://127.0.0.1:8003"
    finally:
        db.close()


def test_reconcile_leaves_a_starting_adopted_row_alone_while_loading(
    tmp_path: Path,
) -> None:
    """A restarted unit is loading; an unavailable probe must not call it STOPPED.

    `restart --systemd` records STARTING precisely because the endpoint cannot
    answer for minutes. Marking it STOPPED with "endpoint failed to respond"
    would report the opposite of what happened.
    """
    probes: list[list[str] | None] = [["m"], None]

    def probe(_url: str, _t: float) -> list[str] | None:
        return probes.pop(0) if probes else None

    calls, runner = _recording_systemctl()
    db, service = _make_service(tmp_path, probe, systemctl=runner)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp"
        )
        restarted = service.restart(session.id, restart_unit=True)
        assert restarted is not None and restarted.status == SessionStatus.STARTING

        assert service.reconcile() == 0  # nothing changed
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.STARTING
        assert refreshed.error is None
    finally:
        db.close()


def test_reconcile_promotes_a_starting_adopted_row_once_it_answers(
    tmp_path: Path,
) -> None:
    """The first successful probe moves STARTING -> RUNNING."""
    calls, runner = _recording_systemctl()
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        session = service.adopt(
            RuntimeName.VLLM, "http://127.0.0.1:8003", systemd_unit="vllm-tp"
        )
        restarted = service.restart(session.id, restart_unit=True)
        assert restarted is not None and restarted.status == SessionStatus.STARTING

        assert service.reconcile() == 1
        refreshed = service.get_session(session.id)
        assert refreshed is not None
        assert refreshed.status == SessionStatus.RUNNING
        assert refreshed.error is None
    finally:
        db.close()


def test_resolve_accepts_a_unique_id_prefix(tmp_path: Path) -> None:
    """`llmctl sessions` elides the id, so a prefix must be usable."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        found = service.get_session(session.id[:8])
        assert found is not None
        assert found.id == session.id
    finally:
        db.close()


def test_resolve_accepts_the_served_name(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(
            RuntimeName.LLAMA_CPP, "http://127.0.0.1:8005", served_name="gpt-oss-120b"
        )
        found = service.get_session("gpt-oss-120b")
        assert found is not None
        assert found.id == session.id
    finally:
        db.close()


def test_resolve_prefers_a_live_session_over_a_stopped_namesake(tmp_path: Path) -> None:
    """A stopped historical row must not shadow the server answering now."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        old = service.adopt(
            RuntimeName.LLAMA_CPP, "http://127.0.0.1:8005", served_name="gpt-oss-120b"
        )
        record = db.exec(select(SessionRecord).where(SessionRecord.id == old.id)).one()
        record.status = SessionStatus.STOPPED
        db.add(record)
        db.commit()
        live = service.adopt(
            RuntimeName.LLAMA_CPP, "http://127.0.0.1:8006", served_name="gpt-oss-120b"
        )

        found = service.get_session("gpt-oss-120b")
        assert found is not None
        assert found.id == live.id
    finally:
        db.close()


def test_resolve_refuses_to_guess_between_two_live_namesakes(tmp_path: Path) -> None:
    """Guessing here would stop the wrong server."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        service.adopt(
            RuntimeName.LLAMA_CPP, "http://127.0.0.1:8005", served_name="twin"
        )
        service.adopt(
            RuntimeName.LLAMA_CPP, "http://127.0.0.1:8006", served_name="twin"
        )
        with pytest.raises(AmbiguousSessionError, match="matches 2 sessions"):
            service.get_session("twin")
    finally:
        db.close()


def test_resolve_exact_id_wins_over_any_other_interpretation(tmp_path: Path) -> None:
    """A full id must never be reinterpreted as a prefix or a name."""
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        found = service.get_session(session.id)
        assert found is not None and found.id == session.id
    finally:
        db.close()


def test_resolve_unknown_selector_is_not_found_not_an_error(tmp_path: Path) -> None:
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        service.adopt(RuntimeName.VLLM, "http://127.0.0.1:8003")
        assert service.get_session("no-such-thing") is None
    finally:
        db.close()


def test_stop_accepts_a_served_name(tmp_path: Path) -> None:
    """The ergonomics that matter: stop by name, not by UUID."""
    calls, runner = _recording_systemctl()
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        service.adopt(
            RuntimeName.LLAMA_CPP,
            "http://127.0.0.1:8005",
            served_name="gpt-oss-120b",
            systemd_unit="gpt-oss-120b.service",
        )
        result = service.stop("gpt-oss-120b", stop_unit=True)
        assert result is not None
        assert result.status == SessionStatus.STOPPED
    finally:
        db.close()


def test_stop_survives_the_unit_detaching_its_own_row(tmp_path: Path) -> None:
    """`systemctl stop` runs the unit's ExecStopPost, which detaches the row.

    The UPDATE then matches zero rows and SQLAlchemy raises StaleDataError.
    That delete is correct behaviour — the caller asked for the unit to go
    away and it did — so the stop must report success, not a traceback.
    """
    db, service = _make_service(tmp_path, lambda u, _t: ["m"])

    url = f"sqlite:///{tmp_path / 'adopt.sqlite3'}"

    def fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if "stop" in argv:
            # Stand in for ExecStopPost -> `ds4-llmctl-sync detach`, which runs
            # in a *separate process*. Deleting over a second connection is what
            # makes the in-flight UPDATE match zero rows, as it does in reality.
            with Session(get_engine(url)) as other:
                for row in other.exec(select(SessionRecord)).all():
                    other.delete(row)
                other.commit()
        return subprocess.CompletedProcess(argv, 0, "", "")

    service._systemctl = SystemctlRunner(runner=fake)
    try:
        session = service.adopt(
            RuntimeName.LLAMA_CPP,
            "http://127.0.0.1:8005",
            served_name="gpt-oss-120b",
            systemd_unit="gpt-oss-120b.service",
        )
        result = service.stop(session.id, stop_unit=True)
        assert result is not None
        assert result.status == SessionStatus.STOPPED
    finally:
        db.close()


def test_detach_logs_the_resolved_id_not_the_selector(tmp_path: Path) -> None:
    """A prefix or name in the event log leaves it ambiguous after the fact."""
    from llmctl.db import EventRecord

    db, service = _make_service(tmp_path, lambda u, _t: ["m"])
    try:
        session = service.adopt(
            RuntimeName.LLAMA_CPP, "http://127.0.0.1:8005", served_name="gpt-oss-120b"
        )
        service.detach("gpt-oss-120b")  # resolved by served name

        messages = [e.message for e in db.exec(select(EventRecord)).all()]
        detached = [m for m in messages if "Detached adopted session" in m]
        assert detached, messages
        assert session.id in detached[-1]
        assert "gpt-oss-120b " not in detached[-1].split("(")[0]
    finally:
        db.close()


def _scoped_systemctl(
    *, user_unit: bool, active: bool = False, start_rc: int = 0
) -> tuple[list[list[str]], SystemctlRunner]:
    """A runner answering LoadState/is-active per scope, recording every call."""
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        in_user = "--user" in argv
        if "show" in argv:
            known = in_user if user_unit else not in_user
            return subprocess.CompletedProcess(
                argv, 0, ("loaded" if known else "not-found") + "\n", ""
            )
        if "is-active" in argv:
            return subprocess.CompletedProcess(
                argv, 0, ("active" if active else "inactive") + "\n", ""
            )
        if "start" in argv:
            return subprocess.CompletedProcess(argv, start_rc, "", "boom" if start_rc else "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    return calls, SystemctlRunner(runner=fake)


def test_start_unit_uses_user_scope_without_sudo(tmp_path: Path) -> None:
    calls, runner = _scoped_systemctl(user_unit=True)
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        scope, was_active = service.start_unit("gpt-oss-120b")
        assert (scope, was_active) == ("user", False)
        starts = [a for a in calls if "start" in a]
        assert starts == [["systemctl", "--user", "start", "gpt-oss-120b"]]
        assert "sudo" not in starts[0]
    finally:
        db.close()


def test_start_unit_uses_sudo_for_a_system_unit(tmp_path: Path) -> None:
    calls, runner = _scoped_systemctl(user_unit=False)
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        scope, _ = service.start_unit("vllm-tp")
        assert scope == "system"
        starts = [a for a in calls if "start" in a]
        assert starts == [["sudo", "systemctl", "start", "vllm-tp"]]
    finally:
        db.close()


def test_start_unit_reports_an_already_active_unit_without_starting(
    tmp_path: Path,
) -> None:
    calls, runner = _scoped_systemctl(user_unit=True, active=True)
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        scope, was_active = service.start_unit("gpt-oss-120b")
        assert (scope, was_active) == ("user", True)
        assert not [a for a in calls if "start" in a]  # nothing issued
    finally:
        db.close()


def test_start_unit_refuses_an_unknown_unit(tmp_path: Path) -> None:
    """Neither manager knows it — say so instead of shelling out."""
    calls: list[list[str]] = []

    def fake(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if "show" in argv:
            return subprocess.CompletedProcess(argv, 0, "not-found\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    db, service = _make_service(
        tmp_path, lambda u, _t: ["m"], systemctl=SystemctlRunner(runner=fake)
    )
    try:
        with pytest.raises(AdoptError, match="No systemd unit named"):
            service.start_unit("ghost")
        assert not [a for a in calls if "start" in a]
    finally:
        db.close()


def test_start_unit_surfaces_a_failed_start(tmp_path: Path) -> None:
    _calls, runner = _scoped_systemctl(user_unit=True, start_rc=1)
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        with pytest.raises(AdoptError, match="failed"):
            service.start_unit("gpt-oss-120b")
    finally:
        db.close()


def test_start_unit_creates_no_session_row(tmp_path: Path) -> None:
    """The unit adopts itself from ExecStartPost; a row here would race it."""
    _calls, runner = _scoped_systemctl(user_unit=True)
    db, service = _make_service(tmp_path, lambda u, _t: ["m"], systemctl=runner)
    try:
        service.start_unit("gpt-oss-120b")
        assert db.exec(select(SessionRecord)).all() == []
    finally:
        db.close()
