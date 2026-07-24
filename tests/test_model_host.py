"""Tests for per-device model host attribution.

Covers the ``lms`` CLI device-map enrichment, the LM Studio adapter wiring, and
the registry treating ``host`` as part of a model's identity (with a migration
bridge for pre-host NULL rows).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from sqlmodel import Session, select

from llmctl.adapters import lmstudio
from llmctl.adapters.lmstudio import LMStudioAdapter, lms_device_map
from llmctl.db import ModelRecord, ModelStatus, RuntimeName, get_engine, init_db
from llmctl.schemas import AdapterStatus, HealthState, Model
from llmctl.services import registry as registry_module
from llmctl.services.registry import RegistryService

# --- lms_device_map ---------------------------------------------------------

_LINK_STATUS = {
    "status": "online",
    "deviceIdentifier": "deskid",
    "deviceName": "yannik-desktop",
    "peers": [
        {"deviceIdentifier": "mbp2id", "deviceName": "macbookpro.lan"},
        {"deviceIdentifier": "m5id", "deviceName": "Yanniks-MacBook-Pro.local"},
    ],
}

_LS_LISTING = [
    {"modelKey": "local-model", "deviceIdentifier": None},
    {"modelKey": "mbp2-model", "deviceIdentifier": "mbp2id"},
    {"modelKey": "m5-model", "deviceIdentifier": "m5id"},
    {"modelKey": "orphan-model", "deviceIdentifier": "unknownid"},
]


def _fake_lms(monkeypatch, status, listing) -> None:
    def fake_run(args, timeout):  # noqa: ARG001 - signature parity
        if args[:2] == ["link", "status"]:
            return status
        if args[:1] == ["ls"]:
            return listing
        return None

    monkeypatch.setattr(lmstudio, "_run_lms_json", fake_run)


def test_lms_device_map_resolves_devices(monkeypatch) -> None:
    _fake_lms(monkeypatch, _LINK_STATUS, _LS_LISTING)
    mapping = lms_device_map()
    assert mapping["local-model"] == "yannik-desktop"  # null identifier -> local
    assert mapping["mbp2-model"] == "macbookpro.lan"
    assert mapping["m5-model"] == "Yanniks-MacBook-Pro.local"
    # An identifier with no matching peer resolves to nothing, so it is omitted.
    assert "orphan-model" not in mapping


def test_lms_device_map_empty_when_cli_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(lmstudio, "_run_lms_json", lambda args, timeout: None)
    assert lms_device_map() == {}


# --- LMStudioAdapter enrichment ---------------------------------------------


def _http_factory(payload):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    def factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))

    return factory


def test_adapter_attributes_host_from_device_map() -> None:
    payload = {"data": [{"id": "mbp2-model"}, {"id": "local-model"}]}
    adapter = LMStudioAdapter(
        client_factory=_http_factory(payload),
        device_map_fn=lambda: {"mbp2-model": "macbookpro.lan"},
    )
    models = asyncio.run(adapter.discover_models())
    by_name = {m.name: m for m in models}
    assert by_name["mbp2-model"].host == "macbookpro.lan"
    # A model absent from the map keeps its (unset) host; the registry defaults it.
    assert by_name["local-model"].host is None


def test_adapter_discovery_survives_device_map_failure() -> None:
    payload = {"data": [{"id": "some-model"}]}

    def boom() -> dict[str, str]:
        raise RuntimeError("lms exploded")

    adapter = LMStudioAdapter(client_factory=_http_factory(payload), device_map_fn=boom)
    models = asyncio.run(adapter.discover_models())
    assert [m.name for m in models] == ["some-model"]
    assert models[0].host is None


# --- registry identity ------------------------------------------------------


class _FakeAdapter:
    def __init__(self, models: list[Model]) -> None:
        self._models = models

    async def health_check(self) -> AdapterStatus:
        return AdapterStatus(runtime=RuntimeName.LMSTUDIO, state=HealthState.OK, message="")

    async def discover_models(self) -> list[Model]:
        return list(self._models)

    @property
    def last_discovery_ok(self) -> bool:
        return True


class _FakeRouter:
    def __init__(self, adapter: _FakeAdapter) -> None:
        self._adapter = adapter

    def list_runtimes(self) -> list[RuntimeName]:
        return [RuntimeName.LMSTUDIO]

    def get_adapter(self, _runtime: RuntimeName) -> _FakeAdapter:
        return self._adapter


def _lms_model(name: str, host: str | None) -> Model:
    return Model(
        name=name,
        runtime=RuntimeName.LMSTUDIO,
        source=name,
        host=host,
        status=ModelStatus.DISCOVERED,
    )


@pytest.fixture
def db(tmp_path):
    url = f"sqlite:///{tmp_path}/host.db"
    init_db(url)
    with Session(get_engine(url)) as session:
        yield session


def _rows(db: Session, name: str) -> list[ModelRecord]:
    return list(db.exec(select(ModelRecord).where(ModelRecord.name == name)).all())


def _scan(db: Session, models: list[Model]) -> None:
    RegistryService(db, _FakeRouter(_FakeAdapter(models))).scan()


def test_same_model_on_two_hosts_is_two_rows(db: Session) -> None:
    _scan(
        db,
        [
            _lms_model("shared-embed", "macbookpro.lan"),
            _lms_model("shared-embed", "Yanniks-MacBook-Pro.local"),
        ],
    )
    hosts = sorted(r.host for r in _rows(db, "shared-embed"))
    assert hosts == ["Yanniks-MacBook-Pro.local", "macbookpro.lan"]


def test_hostless_model_defaults_to_local_host(db: Session, monkeypatch) -> None:
    monkeypatch.setattr(registry_module, "local_host", lambda: "test-desktop")
    _scan(db, [_lms_model("loopback-model", None)])
    rows = _rows(db, "loopback-model")
    assert len(rows) == 1
    assert rows[0].host == "test-desktop"


def test_pre_host_null_row_is_adopted_not_duplicated(db: Session) -> None:
    # Simulate a row registered before the host column existed.
    db.add(
        ModelRecord(
            name="legacy",
            runtime=RuntimeName.LMSTUDIO,
            source="legacy",
            host=None,
            status=ModelStatus.DISCOVERED,
            active=True,
        )
    )
    db.commit()
    _scan(db, [_lms_model("legacy", "macbookpro.lan")])
    rows = _rows(db, "legacy")
    assert len(rows) == 1  # adopted in place, not duplicated
    assert rows[0].host == "macbookpro.lan"


def test_per_device_missing_is_independent(db: Session) -> None:
    _scan(
        db,
        [
            _lms_model("m", "macbookpro.lan"),
            _lms_model("m", "Yanniks-MacBook-Pro.local"),
        ],
    )
    # macbookpro.lan goes offline; only the M5 Max still reports the model.
    _scan(db, [_lms_model("m", "Yanniks-MacBook-Pro.local")])
    status_by_host = {r.host: r.status for r in _rows(db, "m")}
    assert status_by_host["macbookpro.lan"] == ModelStatus.MISSING
    assert status_by_host["Yanniks-MacBook-Pro.local"] == ModelStatus.DISCOVERED
