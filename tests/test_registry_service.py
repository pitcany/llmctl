"""Tests for RegistryService CRUD: add, update, clone, enable/disable, delete."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import Session

from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.schemas import ModelCreate, ModelUpdate
from llmctl.services.registry import RegistryService


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'reg.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def _make(service: RegistryService, **overrides) -> str:
    payload = ModelCreate(
        name=overrides.pop("name", "test-model"),
        runtime=overrides.pop("runtime", RuntimeName.VLLM),
        source=overrides.pop("source", "/data/test"),
        **overrides,
    )
    model = service.add_model(payload)
    assert model.id is not None
    return model.id


def test_add_and_get_round_trip(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = RegistryService(db)
        model_id = _make(
            service,
            max_context=32768,
            parameter_count=70_000_000_000,
            notes="primary",
            tags=["coder"],
        )
        fetched = service.get_model(model_id)
        assert fetched is not None
        assert fetched.max_context == 32768
        assert fetched.parameter_count == 70_000_000_000
        assert fetched.notes == "primary"
        assert fetched.active is True
        assert fetched.tags == ["coder"]


def test_update_model_partial(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = RegistryService(db)
        model_id = _make(service, notes="old")
        updated = service.update_model(
            model_id,
            ModelUpdate(notes="new", tags=["adtech", "fast"]),
        )
        assert updated is not None
        assert updated.notes == "new"
        assert updated.tags == ["adtech", "fast"]


def test_clone_model_copies_all_fields(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = RegistryService(db)
        model_id = _make(service, max_context=8192, tags=["a"])
        cloned = service.clone_model(model_id, new_name="test-model-2")
        assert cloned is not None
        assert cloned.id != model_id
        assert cloned.name == "test-model-2"
        assert cloned.max_context == 8192
        assert cloned.tags == ["a"]


def test_enable_disable_filters_default_list(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = RegistryService(db)
        a = _make(service, name="model-a")
        b = _make(service, name="model-b", source="/data/b")
        assert service.disable_model(a) is True
        names = {m.name for m in service.list_models()}
        assert names == {"model-b"}
        with_inactive = {m.name for m in service.list_models(include_inactive=True)}
        assert with_inactive == {"model-a", "model-b"}
        assert service.enable_model(a) is True
        assert {m.name for m in service.list_models()} == {"model-a", "model-b"}
        # Silence unused-variable warning while keeping the second model in
        # the registry so the include_inactive case is meaningful.
        assert b


def test_delete_soft_removes_from_default_list(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = RegistryService(db)
        model_id = _make(service, name="ephemeral")
        assert service.delete_model(model_id) is True
        assert service.get_model(model_id) is None
        assert all(m.name != "ephemeral" for m in service.list_models())


def test_delete_with_files_removes_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "weights.gguf"
    artifact.write_bytes(b"x" * 8)
    with _db(tmp_path) as db:
        service = RegistryService(db)
        model_id = _make(
            service,
            name="gguf",
            runtime=RuntimeName.LLAMA_CPP,
            source=str(artifact),
            path=str(artifact),
        )
        assert artifact.exists()
        assert service.delete_model(model_id, delete_files=True) is True
        assert not artifact.exists()


def test_delete_without_files_preserves_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "keep.gguf"
    artifact.write_bytes(b"y" * 4)
    with _db(tmp_path) as db:
        service = RegistryService(db)
        model_id = _make(
            service,
            name="keep",
            runtime=RuntimeName.LLAMA_CPP,
            source=str(artifact),
            path=str(artifact),
        )
        assert service.delete_model(model_id) is True
        assert artifact.exists()


def test_find_by_id_and_name(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = RegistryService(db)
        model_id = _make(service, name="unique-name")
        assert service.find(model_id).id == model_id  # type: ignore[union-attr]
        assert service.find("unique-name").id == model_id  # type: ignore[union-attr]
        assert service.find("nope") is None


def test_find_ambiguous_name_raises(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = RegistryService(db)
        _make(service, name="dual", runtime=RuntimeName.VLLM, source="/v")
        _make(service, name="dual", runtime=RuntimeName.LLAMA_CPP, source="/l")
        with pytest.raises(ValueError, match="ambiguous"):
            service.find("dual")


def test_list_models_includes_null_active_rows(tmp_path: Path) -> None:
    """Models migrated before the ``active`` column existed have NULL there.

    ``apply_migrations`` adds the column without ``DEFAULT TRUE``, so old
    rows keep NULL. The default listing must treat NULL as active — a strict
    ``active != False`` filter would silently hide every pre-migration model.
    """
    import sqlite3

    db_file = tmp_path / "reg.sqlite3"
    with _db(tmp_path) as db:
        model_id = _make(RegistryService(db), name="legacy")

    # Simulate a pre-migration row by nulling ``active`` directly in SQLite.
    conn = sqlite3.connect(str(db_file))
    conn.execute("UPDATE models SET active = NULL WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()

    with _db(tmp_path) as db:
        visible_names = {m.name for m in RegistryService(db).list_models()}
    assert "legacy" in visible_names
