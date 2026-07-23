"""Generic OpenAI-endpoint adoption: RuntimeName.OPENAI as an adopt-only runtime."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlmodel import Session

from llmctl.api.gateway import create_gateway_app
from llmctl.config import RouterSettings, Settings
from llmctl.db import (
    RuntimeName,
    SessionKind,
    SessionStatus,
    get_engine,
    init_db,
    utcnow,
)
from llmctl.schemas import SessionStartRequest
from llmctl.services.scheduler import SchedulerError, SchedulerService
from llmctl.services.sessions import SessionService


def _service(tmp_path: Path, probe) -> tuple[Session, SessionService]:
    url = f"sqlite:///{tmp_path / 'openai.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    service = SessionService(db, probe=probe, gpu_ids_for_unit=lambda _u: [])
    return db, service


def test_adopt_openai_endpoint_without_unit(tmp_path: Path) -> None:
    db, service = _service(tmp_path, lambda _u, _t: ["mac-qwen-235b"])
    try:
        session = service.adopt(RuntimeName.OPENAI, "http://100.71.53.54:8080")
        assert session.kind == SessionKind.ADOPTED
        assert session.status == SessionStatus.RUNNING
        assert session.runtime == RuntimeName.OPENAI
        assert session.served_name == "mac-qwen-235b"
        assert session.systemd_unit is None
        assert session.gpu_ids == []
    finally:
        db.close()


def test_openai_adopted_reconcile_probe_lifecycle(tmp_path: Path) -> None:
    """A remote endpoint dying and returning flips the row STOPPED -> RUNNING."""
    answers: dict[str, list[str] | None] = {"models": ["remote-model"]}
    db, service = _service(tmp_path, lambda _u, _t: answers["models"])
    try:
        session = service.adopt(RuntimeName.OPENAI, "http://10.0.0.9:9000")
        answers["models"] = None
        assert service.reconcile() == 1
        refreshed = service.get_session(session.id)
        assert refreshed is not None and refreshed.status == SessionStatus.STOPPED

        answers["models"] = ["remote-model"]
        assert service.reconcile() == 1
        revived = service.get_session(session.id)
        assert revived is not None and revived.status == SessionStatus.RUNNING
    finally:
        db.close()


def test_scheduler_refuses_openai_launch_plan(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'sched.sqlite3'}"
    init_db(url)
    with Session(get_engine(url)) as db:
        scheduler = SchedulerService(db, Settings())
        with pytest.raises(SchedulerError, match="adopt-only"):
            scheduler.create_launch_plan(
                SessionStartRequest(model_id="whatever", runtime=RuntimeName.OPENAI)
            )


def test_gateway_routes_to_openai_adopted_session(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'gateway.sqlite3'}"
    init_db(db_url)
    settings = Settings()
    settings.database.url = db_url
    settings.paths.config_dir = tmp_path / "cfg"
    settings.paths.config_dir.mkdir(exist_ok=True)
    settings.router = RouterSettings(aliases={})

    from llmctl.db import SessionRecord

    with Session(get_engine(db_url)) as db:
        db.add(
            SessionRecord(
                runtime=RuntimeName.OPENAI,
                status=SessionStatus.RUNNING,
                kind=SessionKind.ADOPTED,
                endpoint_url="http://100.71.53.54:8080",
                health_url="http://100.71.53.54:8080/v1/models",
                port=8080,
                served_name="mac-qwen-235b",
                adopted_at=utcnow(),
                started_at=utcnow(),
            )
        )
        db.commit()

    client = TestClient(create_gateway_app(settings, database_url=db_url))
    with respx.mock(assert_all_called=True) as router:
        router.post("http://100.71.53.54:8080/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"index": 0}]})
        )
        response = client.post(
            "/v1/chat/completions",
            json={"model": "mac-qwen-235b", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.headers["x-llmctl-route"] == "explicit"
