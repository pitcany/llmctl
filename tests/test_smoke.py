"""Smoke tests for scaffold imports and database setup."""

from __future__ import annotations

from pathlib import Path

from sqlmodel import inspect

from llmctl.api.app import create_app
from llmctl.config import load_model_dirs, load_profiles, load_settings
from llmctl.db import get_engine, init_db
from llmctl.services.router import RuntimeRouter


def test_config_examples_load() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    settings = load_settings(root / "settings.yaml")
    model_dirs = load_model_dirs(root / "model_dirs.yaml")
    profiles = load_profiles(root / "profiles.yaml")
    assert settings.app.name == "llm-mission-control"
    assert model_dirs.model_roots
    assert profiles.profiles


def test_runtime_router_has_expected_adapters() -> None:
    router = RuntimeRouter()
    assert {runtime.value for runtime in router.list_runtimes()} == {
        "vllm",
        "llama_cpp",
        "lmstudio",
        "ollama",
        "python_script",
    }


def test_db_schema_creation(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'test.sqlite3'}"
    init_db(db_url)
    engine = get_engine(db_url)
    tables = set(inspect(engine).get_table_names())
    assert {"models", "sessions", "profiles", "benchmarks", "events"}.issubset(tables)


def _collect_route_paths(routes: object) -> set[str]:
    """Collect endpoint paths from a FastAPI route list.

    Newer FastAPI (>=0.137) stops flattening ``include_router`` routes into
    ``app.routes``; it inserts a path-less ``_IncludedRouter`` whose
    ``original_router`` holds the real routes. Older versions flatten them and
    have no such wrapper. Recursing through ``original_router`` handles both,
    so the assertion is version-agnostic.
    """
    paths: set[str] = set()
    for route in routes:  # type: ignore[attr-defined]
        path = getattr(route, "path", None)
        if path is not None:
            paths.add(path)
        original = getattr(route, "original_router", None)
        if original is not None and hasattr(original, "routes"):
            paths |= _collect_route_paths(original.routes)
    return paths


def test_api_app_routes(tmp_path: Path) -> None:
    app = create_app(database_url=f"sqlite:///{tmp_path / 'api.sqlite3'}")
    routes = _collect_route_paths(app.routes)
    assert "/health" in routes
    assert "/models" in routes
    assert "/models/{model_id}" in routes
    assert "/sessions" in routes
    assert "/sessions/start" in routes
    assert "/sessions/{session_id}/stop" in routes
    assert "/sessions/{session_id}/restart" in routes
    assert "/gpus" in routes
    assert "/benchmarks" in routes
    assert "/benchmarks/run" in routes
