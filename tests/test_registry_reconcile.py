"""Tests for scan-time reconciliation and pruning of vanished models."""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from llmctl.db import ModelRecord, ModelStatus, RuntimeName, get_engine, init_db
from llmctl.schemas import AdapterStatus, HealthState, Model, ModelCreate
from llmctl.services.registry import RegistryService


class FakeAdapter:
    """Adapter stub with controllable health and discovery."""

    def __init__(self, runtime: RuntimeName, *, healthy: bool, models: list[Model]) -> None:
        self._runtime = runtime
        self._healthy = healthy
        self._models = models

    async def health_check(self) -> AdapterStatus:
        state = HealthState.OK if self._healthy else HealthState.UNAVAILABLE
        return AdapterStatus(runtime=self._runtime, state=state, message="")

    async def discover_models(self) -> list[Model]:
        return list(self._models)


class FakeRouter:
    """Router stub returning fixed adapters."""

    def __init__(self, adapters: dict[RuntimeName, FakeAdapter]) -> None:
        self._adapters = adapters

    def list_runtimes(self) -> list[RuntimeName]:
        return list(self._adapters)

    def get_adapter(self, runtime: RuntimeName) -> FakeAdapter:
        return self._adapters[runtime]


def _model(name: str) -> Model:
    return Model(
        name=name,
        runtime=RuntimeName.OLLAMA,
        source=name,
        status=ModelStatus.DISCOVERED,
    )


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path}/reconcile.db"
    init_db(url)
    with Session(get_engine(url)) as session:
        yield session


def _status(db: Session, name: str) -> ModelStatus:
    record = db.exec(select(ModelRecord).where(ModelRecord.name == name)).first()
    assert record is not None
    return record.status


def _scan(db: Session, *, healthy: bool, models: list[Model]) -> None:
    adapter = FakeAdapter(RuntimeName.OLLAMA, healthy=healthy, models=models)
    router = FakeRouter({RuntimeName.OLLAMA: adapter})
    RegistryService(db, router).scan()


def test_scan_marks_vanished_discovered_as_missing(db: Session) -> None:
    a, b = _model("a"), _model("b")
    _scan(db, healthy=True, models=[a, b])
    assert _status(db, "a") == ModelStatus.DISCOVERED
    assert _status(db, "b") == ModelStatus.DISCOVERED
    _scan(db, healthy=True, models=[a])  # b is gone
    assert _status(db, "a") == ModelStatus.DISCOVERED
    assert _status(db, "b") == ModelStatus.MISSING


def test_scan_skips_reconcile_when_adapter_unhealthy(db: Session) -> None:
    a, b = _model("a"), _model("b")
    _scan(db, healthy=True, models=[a, b])
    _scan(db, healthy=False, models=[])  # daemon down -> discover empty
    assert _status(db, "a") == ModelStatus.DISCOVERED
    assert _status(db, "b") == ModelStatus.DISCOVERED


def test_scan_does_not_touch_registered_models(db: Session) -> None:
    RegistryService(db).add_model(
        ModelCreate(name="manual", runtime=RuntimeName.OLLAMA, source="manual")
    )
    _scan(db, healthy=True, models=[])  # healthy, nothing discovered
    assert _status(db, "manual") == ModelStatus.REGISTERED


def test_rediscovery_restores_missing_to_discovered(db: Session) -> None:
    a, b = _model("a"), _model("b")
    _scan(db, healthy=True, models=[a, b])
    _scan(db, healthy=True, models=[a])  # b -> MISSING
    assert _status(db, "b") == ModelStatus.MISSING
    _scan(db, healthy=True, models=[a, b])  # b reappears
    assert _status(db, "b") == ModelStatus.DISCOVERED
