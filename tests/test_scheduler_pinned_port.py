"""A profile may pin its port, so both launch paths agree on the endpoint.

Some models are reachable two ways: a systemd unit that hardcodes a port, and
llmctl's own scheduler path (`llmctl start --profile`). Without a pin the
scheduler allocates from ``scheduler.port_ranges``, so the same model answered on
8005 when systemd started it and on whatever was free when llmctl did — and
anything downstream pinned to the documented port broke depending on who
launched it.

An explicit pin is honoured exactly: taken means **refused**, never silently
moved. Silently relocating is the failure this exists to prevent, and a session
whose recorded port is not where the server is listening is the same class of
bug as a status that does not match reality.
"""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from llmctl.db import (
    RuntimeName,
    SessionKind,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import SessionStartRequest
from llmctl.services.scheduler import SchedulerService


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'ports.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def _request(port: int | None) -> SessionStartRequest:
    params: dict = {"n_gpu_layers": 999}
    if port is not None:
        params["port"] = port
    return SessionStartRequest(
        model_id="m1",
        runtime=RuntimeName.LLAMA_CPP,
        dry_run=True,
        parameters=params,
    )


def _occupy(db: Session, port: int) -> None:
    """An active session already holding ``port``."""
    db.add(
        SessionRecord(
            runtime=RuntimeName.LLAMA_CPP,
            status=SessionStatus.RUNNING,
            kind=SessionKind.OWNED,
            port=port,
            endpoint_url=f"http://127.0.0.1:{port}",
            pid=4242,
        )
    )
    db.commit()


def test_a_pinned_port_is_used_verbatim(tmp_path: Path) -> None:
    db = _db(tmp_path)
    try:
        plan = SchedulerService(db).create_launch_plan(_request(8005))
        assert plan.port == 8005
        assert plan.endpoint_url.endswith(":8005")
    finally:
        db.close()


def test_a_pinned_port_survives_being_outside_the_configured_range(tmp_path: Path) -> None:
    """An explicit pin beats the range: explicit beats implicit."""
    db = _db(tmp_path)
    try:
        plan = SchedulerService(db).create_launch_plan(_request(9321))
        assert plan.port == 9321
    finally:
        db.close()


def test_a_taken_pinned_port_is_refused_not_reassigned(tmp_path: Path) -> None:
    """The whole point. Moving the port silently is what this prevents."""
    db = _db(tmp_path)
    try:
        _occupy(db, 8005)
        plan = SchedulerService(db).create_launch_plan(_request(8005))

        assert plan.port == 8005, "must not silently relocate to a free port"
        assert any("8005" in r for r in plan.refusal_reasons), plan.refusal_reasons
    finally:
        db.close()


def test_without_a_pin_the_range_is_used_as_before(tmp_path: Path) -> None:
    db = _db(tmp_path)
    try:
        plan = SchedulerService(db).create_launch_plan(_request(None))
        assert plan.port is not None
        assert not any("pinned" in r for r in plan.refusal_reasons)
    finally:
        db.close()


def test_without_a_pin_a_taken_port_still_moves_on(tmp_path: Path) -> None:
    """Unpinned launches keep their existing scan-for-free behaviour."""
    db = _db(tmp_path)
    try:
        first = SchedulerService(db).create_launch_plan(_request(None))
        _occupy(db, first.port)

        second = SchedulerService(db).create_launch_plan(_request(None))

        assert second.port != first.port
        # Only port concerns are in scope here; the stub model id draws an
        # unrelated "no model path/source" refusal.
        assert not any("port" in r.lower() for r in second.refusal_reasons)
    finally:
        db.close()
