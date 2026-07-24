"""Tests for scan-time reconciliation and pruning of vanished models."""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from llmctl.db import ModelRecord, ModelStatus, RuntimeName, get_engine, init_db
from llmctl.schemas import AdapterStatus, HealthState, Model, ModelCreate
from llmctl.services.registry import RegistryService


class FakeAdapter:
    """Adapter stub with controllable health and discovery."""

    def __init__(
        self,
        runtime: RuntimeName,
        *,
        healthy: bool,
        models: list[Model],
        discovery_ok: bool = True,
    ) -> None:
        self._runtime = runtime
        self._healthy = healthy
        self._models = models
        self._discovery_ok = discovery_ok

    async def health_check(self) -> AdapterStatus:
        state = HealthState.OK if self._healthy else HealthState.UNAVAILABLE
        return AdapterStatus(runtime=self._runtime, state=state, message="")

    async def discover_models(self) -> list[Model]:
        return list(self._models)

    @property
    def last_discovery_ok(self) -> bool:
        return self._discovery_ok


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


def _scan(db: Session, *, healthy: bool, models: list[Model], discovery_ok: bool = True) -> None:
    adapter = FakeAdapter(
        RuntimeName.OLLAMA, healthy=healthy, models=models, discovery_ok=discovery_ok
    )
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


def test_scan_skips_reconcile_when_discovery_call_failed(db: Session) -> None:
    a, b = _model("a"), _model("b")
    _scan(db, healthy=True, models=[a, b])
    # Daemon healthy (version OK) but the tags/list call failed -> discover() returns []
    # but last_discovery_ok is False, so reconcile must be skipped (no false MISSING).
    _scan(db, healthy=True, models=[], discovery_ok=False)
    assert _status(db, "a") == ModelStatus.DISCOVERED
    assert _status(db, "b") == ModelStatus.DISCOVERED


def test_scan_keeps_row_whose_path_still_exists_on_disk(db: Session, tmp_path) -> None:
    """A row whose recorded checkpoint is still on disk is never MISSING.

    Single-model servers (the vllm-tp unit) report only the currently
    served model, so absence from one scan means rotation, not
    disappearance. MISSING is reserved for artifacts that vanished.
    """
    ckpt = tmp_path / "ornith-ckpt"
    ckpt.mkdir()
    on_disk = Model(
        name="ornith",
        runtime=RuntimeName.OLLAMA,
        source="ornith",
        path=str(ckpt),
        status=ModelStatus.DISCOVERED,
    )
    other = _model("other")
    _scan(db, healthy=True, models=[on_disk, other])
    _scan(db, healthy=True, models=[other])  # ornith rotated out, still on disk
    assert _status(db, "ornith") == ModelStatus.DISCOVERED


def test_scan_keeps_row_whose_path_is_not_a_local_filesystem_path(
    db: Session, tmp_path, monkeypatch
) -> None:
    """A non-absolute ``path`` (an HF repo id) is not judgeable from disk.

    vLLM reports ``root`` as the ``--model`` value, which is a Hugging Face
    repo id whenever the server was pointed at the hub rather than a local
    checkpoint. Testing such a value with ``Path.exists()`` asks a question
    about the current working directory, not about the model.
    """
    repo_id = Model(
        name="hub-model",
        runtime=RuntimeName.OLLAMA,
        source="hub-model",
        path="deepreinforce-ai/Ornith-1.0-35B-FP8",
        status=ModelStatus.DISCOVERED,
    )
    _scan(db, healthy=True, models=[repo_id])
    _scan(db, healthy=True, models=[])
    assert _status(db, "hub-model") == ModelStatus.DISCOVERED

    # ...and the verdict must not change with the process working directory.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deepreinforce-ai").mkdir()
    _scan(db, healthy=True, models=[])
    assert _status(db, "hub-model") == ModelStatus.DISCOVERED


def test_scan_flags_row_whose_path_vanished(db: Session, tmp_path) -> None:
    """A recorded path that no longer exists does not shield the row."""
    gone = Model(
        name="gone",
        runtime=RuntimeName.OLLAMA,
        source="gone",
        path=str(tmp_path / "never-created"),
        status=ModelStatus.DISCOVERED,
    )
    _scan(db, healthy=True, models=[gone])
    _scan(db, healthy=True, models=[])
    assert _status(db, "gone") == ModelStatus.MISSING


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


def _seed_missing(db: Session, name: str, runtime: RuntimeName) -> None:
    db.add(
        ModelRecord(
            name=name,
            runtime=runtime,
            source=name,
            status=ModelStatus.MISSING,
            active=True,
        )
    )
    db.commit()


def test_prune_missing_transitions_to_deleted_and_returns_count(db: Session) -> None:
    _seed_missing(db, "g1", RuntimeName.OLLAMA)
    _seed_missing(db, "g2", RuntimeName.OLLAMA)
    count = RegistryService(db).prune_missing()
    assert count == 2
    assert _status(db, "g1") == ModelStatus.DELETED
    assert _status(db, "g2") == ModelStatus.DELETED
    # Pruned rows are hidden from the default listing.
    names = {m.name for m in RegistryService(db).list_models()}
    assert "g1" not in names and "g2" not in names


def test_prune_missing_runtime_filter(db: Session) -> None:
    _seed_missing(db, "oll", RuntimeName.OLLAMA)
    _seed_missing(db, "lms", RuntimeName.LMSTUDIO)
    count = RegistryService(db).prune_missing(RuntimeName.OLLAMA)
    assert count == 1
    assert _status(db, "oll") == ModelStatus.DELETED
    assert _status(db, "lms") == ModelStatus.MISSING
