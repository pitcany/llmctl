"""Tests for the profile service and default profiles."""

from __future__ import annotations

import os
from pathlib import Path

from sqlmodel import Session

from llmctl.db import RuntimeName, get_engine, init_db
from llmctl.services.profiles import ProfileService

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _db(tmp_path: Path) -> Session:
    url = f"sqlite:///{tmp_path / 'prof.sqlite3'}"
    init_db(url)
    return Session(get_engine(url))


def test_sync_and_list_default_profiles(tmp_path: Path) -> None:
    os.environ["LLMCTL_CONFIG_DIR"] = str(CONFIGS)
    try:
        with _db(tmp_path) as db:
            service = ProfileService(db)
            synced = service.sync_from_yaml()
            names = {profile.name for profile in synced}
            assert {
                "fast",
                "coding",
                "reasoning",
                "long-context",
                "tutoring",
                "adtech",
                "quant",
            }.issubset(names)
            # Idempotent re-sync should not duplicate.
            again = service.sync_from_yaml()
            assert len(again) == len(synced)
    finally:
        del os.environ["LLMCTL_CONFIG_DIR"]


def test_get_by_name_returns_runtime(tmp_path: Path) -> None:
    os.environ["LLMCTL_CONFIG_DIR"] = str(CONFIGS)
    try:
        with _db(tmp_path) as db:
            service = ProfileService(db)
            long_context = service.get_by_name("long-context")
            assert long_context is not None
            assert long_context.runtime == RuntimeName.VLLM
            assert long_context.parameters["tensor_parallel_size"] == 2

            tutoring = service.get_by_name("tutoring")
            assert tutoring is not None
            assert tutoring.runtime == RuntimeName.LLAMA_CPP
    finally:
        del os.environ["LLMCTL_CONFIG_DIR"]


def test_get_by_name_missing(tmp_path: Path) -> None:
    os.environ["LLMCTL_CONFIG_DIR"] = str(CONFIGS)
    try:
        with _db(tmp_path) as db:
            assert ProfileService(db).get_by_name("does-not-exist") is None
    finally:
        del os.environ["LLMCTL_CONFIG_DIR"]
