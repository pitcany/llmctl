"""Health must report what is actually loaded, not only what llmctl launched.

Reported live on 2026-08-26: ``llmctl health`` showed ``Serving: nothing`` for
every runtime while Ollama had qwen3.8:27b resident and a vLLM unit held 51 GB
of VRAM. Both had been started outside llmctl, so no SessionRecord existed and
the DB-only lookup had nothing to report. The column claimed to answer "what is
actually serving right now"; it answered "what did I start".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from sqlmodel import Session

from llmctl.db import (
    ModelStatus,
    RuntimeName,
    SessionRecord,
    SessionStatus,
    get_engine,
    init_db,
)
from llmctl.schemas import Model
from llmctl.services.health import HealthService


class _Adapter:
    """Adapter double: answers health_check and list_loaded_models only."""

    def __init__(self, loaded: list[Model] | None, *, raises: bool = False) -> None:
        self._loaded = loaded
        self._raises = raises

    async def health_check(self) -> Any:
        from llmctl.schemas import AdapterStatus, HealthState

        return AdapterStatus(state=HealthState.OK, message="binary found")

    async def list_loaded_models(self) -> list[Model] | None:
        if self._raises:
            raise RuntimeError("probe exploded")
        return self._loaded


class _Router:
    def __init__(self, adapters: dict[RuntimeName, _Adapter]) -> None:
        self.adapters = adapters

    def list_runtimes(self) -> list[RuntimeName]:
        return list(self.adapters)

    def get_adapter(self, runtime: RuntimeName) -> _Adapter:
        return self.adapters[runtime]


def _model(name: str, runtime: RuntimeName) -> Model:
    return Model(name=name, runtime=runtime, source=name, status=ModelStatus.DISCOVERED)


def _db_with(tmp_path: Path, *records: SessionRecord) -> Session:
    url = f"sqlite:///{tmp_path / 'probe.sqlite3'}"
    init_db(url)
    db = Session(get_engine(url))
    for r in records:
        db.add(r)
    db.commit()
    return db


def test_loaded_model_started_outside_llmctl_is_reported() -> None:
    """The exact reported bug: model resident in Ollama, no SessionRecord."""
    router = _Router({RuntimeName.OLLAMA: _Adapter([_model("qwen3.8:27b", RuntimeName.OLLAMA)])})
    data = HealthService(router=router).get_health()
    ollama = data["runtimes"]["ollama"]

    assert ollama["serving_count"] == 1
    entry = ollama["sessions"][0]
    assert entry["served_name"] == "qwen3.8:27b"
    assert entry["managed"] is False


def test_managed_session_is_not_duplicated_by_the_probe(tmp_path: Path) -> None:
    """A model llmctl launched must appear once, still flagged managed."""
    router = _Router({RuntimeName.OLLAMA: _Adapter([_model("qwen3.8:27b", RuntimeName.OLLAMA)])})
    db = _db_with(
        tmp_path,
        SessionRecord(
            runtime=RuntimeName.OLLAMA,
            status=SessionStatus.RUNNING,
            served_name="qwen3.8:27b",
            endpoint_url="http://127.0.0.1:11434",
        ),
    )
    try:
        data = HealthService(router=router, db=db).get_health()
        ollama = data["runtimes"]["ollama"]
        assert ollama["serving_count"] == 1
        assert ollama["sessions"][0]["managed"] is True
    finally:
        db.close()


def test_unsupported_probe_reports_nothing_and_does_not_crash() -> None:
    """``None`` means the runtime cannot answer — not that it is idle."""
    router = _Router({RuntimeName.LLAMA_CPP: _Adapter(None)})
    data = HealthService(router=router).get_health()
    assert data["runtimes"]["llama_cpp"]["serving_count"] == 0


def test_probe_failure_degrades_instead_of_failing_health() -> None:
    """A raising adapter must not take down the whole health call."""
    router = _Router({RuntimeName.OLLAMA: _Adapter(None, raises=True)})
    data = HealthService(router=router).get_health()
    assert data["runtimes"]["ollama"]["serving_count"] == 0
    assert data["state"] is not None


def test_probe_works_without_a_database() -> None:
    """Probing needs no DB: unmanaged models show even when session state is off."""
    router = _Router({RuntimeName.VLLM: _Adapter([_model("Qwen3-27B", RuntimeName.VLLM)])})
    data = HealthService(router=router).get_health()
    assert data["session_state_available"] is False
    assert data["runtimes"]["vllm"]["serving_count"] == 1
    assert data["runtimes"]["vllm"]["sessions"][0]["managed"] is False


class _FakeLmStudio:
    """Captures which path the adapter probes and returns a canned payload."""

    def __init__(self, payloads: dict[str, object]) -> None:
        self.payloads = payloads
        self.requested: list[str] = []

    async def get_json(self, path: str) -> tuple[bool, object | None, str | None]:
        self.requested.append(path)
        if path not in self.payloads:
            return False, None, "404"
        return True, self.payloads[path], None


def test_lmstudio_reports_only_loaded_models() -> None:
    """``/v1/models`` lists the catalog; only ``state == loaded`` is resident.

    Observed 2026-08-26: every LM Studio model read as serving because the
    OpenAI listing includes downloaded-but-unloaded models.
    """
    from llmctl.adapters.lmstudio import LMStudioAdapter

    adapter = LMStudioAdapter()
    fake = _FakeLmStudio(
        {
            "/api/v0/models": {
                "data": [
                    {"id": "qwen3.8", "state": "not-loaded"},
                    {"id": "qwen3-4b", "state": "loaded"},
                    {"id": "deepseek-r1-8b", "state": "not-loaded"},
                ]
            }
        }
    )
    adapter._get_json = fake.get_json  # type: ignore[method-assign]

    loaded = asyncio.run(adapter.list_loaded_models())

    assert loaded is not None
    assert [m.name for m in loaded] == ["qwen3-4b"]
    assert "/api/v0/models" in fake.requested


def test_lmstudio_probe_returns_none_when_state_endpoint_absent() -> None:
    """Without ``/api/v0/models`` there is no way to tell loaded from listed.

    Reporting the catalog would be worse than admitting ignorance, so the
    adapter must answer ``None`` (unsupported) rather than every model.
    """
    from llmctl.adapters.lmstudio import LMStudioAdapter

    adapter = LMStudioAdapter()
    fake = _FakeLmStudio({"/v1/models": {"data": [{"id": "qwen3.8"}]}})
    adapter._get_json = fake.get_json  # type: ignore[method-assign]

    assert asyncio.run(adapter.list_loaded_models()) is None
