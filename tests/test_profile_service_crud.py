"""Tests for ProfileService CRUD, validation, and YAML round-trip."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlmodel import Session

from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.schemas import ProfileCreate, ProfileUpdate
from llmctl.services.profiles import ProfileService, validate_profile

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'prof.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def test_create_and_get_profile(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = ProfileService(db)
        created = service.create_profile(
            ProfileCreate(
                name="custom",
                runtime=RuntimeName.VLLM,
                description="custom",
                tensor_parallel_size=2,
                max_model_len=16384,
                gpu_memory_utilization=0.85,
                dtype="float16",
                extra_args=["--enable-prefix-caching"],
                environment_variables={"NCCL_P2P_DISABLE": "0"},
            )
        )
        assert created.id is not None
        round_trip = service.get_by_id(created.id)
        assert round_trip is not None
        assert round_trip.tensor_parallel_size == 2
        assert round_trip.extra_args == ["--enable-prefix-caching"]


def test_create_rejects_duplicate_name(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = ProfileService(db)
        service.create_profile(ProfileCreate(name="dup", runtime=RuntimeName.VLLM))
        with pytest.raises(ValueError, match="already exists"):
            service.create_profile(ProfileCreate(name="dup", runtime=RuntimeName.VLLM))


def test_update_profile_partial(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = ProfileService(db)
        created = service.create_profile(
            ProfileCreate(name="edit-me", runtime=RuntimeName.VLLM, max_model_len=4096)
        )
        updated = service.update_profile(
            created.id, ProfileUpdate(max_model_len=32768, dtype="bfloat16")
        )
        assert updated is not None
        assert updated.max_model_len == 32768
        assert updated.dtype == "bfloat16"


def test_clone_profile(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = ProfileService(db)
        original = service.create_profile(
            ProfileCreate(
                name="src",
                runtime=RuntimeName.VLLM,
                tensor_parallel_size=2,
                extra_args=["--foo"],
            )
        )
        cloned = service.clone_profile(original.id, "dst")
        assert cloned is not None
        assert cloned.id != original.id
        assert cloned.tensor_parallel_size == 2
        assert cloned.extra_args == ["--foo"]


def test_delete_profile(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = ProfileService(db)
        created = service.create_profile(
            ProfileCreate(name="gone", runtime=RuntimeName.VLLM)
        )
        assert service.delete_profile(created.id) is True
        assert service.get_by_id(created.id) is None


def test_yaml_round_trip_preserves_parameters(tmp_path: Path) -> None:
    os.environ["LLMCTL_CONFIG_DIR"] = str(CONFIGS)
    try:
        with _db(tmp_path) as db:
            service = ProfileService(db)
            service.sync_from_yaml()
            long_ctx = service.get_by_name("long-context")
            assert long_ctx is not None
            exported = service.export_to_dict(long_ctx)
            assert exported["parameters"]["tensor_parallel_size"] == 2
            assert exported["parameters"]["max_model_len"] == 131072
            assert exported["runtime"] == "vllm"
    finally:
        del os.environ["LLMCTL_CONFIG_DIR"]


def test_import_from_dict_creates_new(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = ProfileService(db)
        imported = service.import_from_dict(
            {
                "name": "imported",
                "runtime": "vllm",
                "description": "via import",
                "parameters": {
                    "tensor_parallel_size": 1,
                    "max_model_len": 8192,
                    "custom_knob": "value",
                },
            }
        )
        assert imported.id is not None
        assert imported.tensor_parallel_size == 1
        assert imported.parameters["custom_knob"] == "value"


def test_import_from_dict_updates_existing(tmp_path: Path) -> None:
    with _db(tmp_path) as db:
        service = ProfileService(db)
        service.create_profile(
            ProfileCreate(name="x", runtime=RuntimeName.VLLM, max_model_len=1024)
        )
        updated = service.import_from_dict(
            {"name": "x", "runtime": "vllm", "parameters": {"max_model_len": 9999}}
        )
        assert updated.max_model_len == 9999


def test_validate_flags_bad_inputs() -> None:
    issues = validate_profile(
        ProfileCreate(
            name="oops",
            runtime=RuntimeName.LLAMA_CPP,
            tensor_parallel_size=2,
            max_model_len=-1,
            gpu_memory_utilization=1.5,
        )
    )
    fields = {(issue.field, issue.severity) for issue in issues}
    assert ("tensor_parallel_size", "warning") in fields
    assert ("max_model_len", "error") in fields
    assert ("gpu_memory_utilization", "error") in fields


def test_validate_passes_reasonable_profile() -> None:
    issues = validate_profile(
        ProfileCreate(
            name="ok",
            runtime=RuntimeName.VLLM,
            tensor_parallel_size=2,
            max_model_len=32768,
            gpu_memory_utilization=0.85,
        )
    )
    assert issues == []
