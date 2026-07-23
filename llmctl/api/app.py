"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Generator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from llmctl.api import (
    routes_benchmarks,
    routes_gpu,
    routes_models,
    routes_profiles,
    routes_sessions,
)
from llmctl.api.deps import get_db_session
from llmctl.config import Settings, load_settings
from llmctl.db import SQLModel, apply_migrations, get_engine
from llmctl.services.health import HealthService


def create_app(settings: Settings | None = None, database_url: str | None = None) -> FastAPI:
    """Create and configure the FastAPI app."""
    effective_settings = settings or load_settings()
    effective_database_url = database_url or effective_settings.database_url
    engine = get_engine(effective_database_url)
    SQLModel.metadata.create_all(engine)
    apply_migrations(engine)
    app = FastAPI(
        title="LLM Mission Control",
        version="0.1.0",
        description="Local-first LLM runtime control plane scaffold.",
    )

    if effective_settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=effective_settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def session_dependency() -> Generator[Session, None, None]:
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db_session] = session_dependency

    @app.get("/health", tags=["health"])
    def health() -> dict[str, object]:
        """Return service health."""
        return HealthService(effective_settings).get_health()

    @app.get("/doctor", tags=["health"])
    def doctor() -> dict[str, object]:
        """Return the structured doctor report (same checks as `llmctl doctor`).

        Legacy summary keys (backends/gpu_count/...) are kept so existing
        dashboards don't break; the pass/warn/fail report is authoritative.
        """
        from llmctl.services.backends import detect_backends
        from llmctl.services.doctor import run_doctor
        from llmctl.telemetry.gpu import get_gpu_info, nvml_available

        sched = effective_settings.scheduler
        report = run_doctor(effective_settings)
        report.update(
            {
                "backends": detect_backends(effective_settings),
                "gpu_count": len(get_gpu_info()),
                "nvml_available": nvml_available(),
                "safe_mode": effective_settings.app.safe_mode,
                "scheduler": {
                    "gpu_policy": sched.gpu_policy,
                    "safety_margin_gb": sched.safety_margin_gb,
                    "allow_public_bind": sched.allow_public_bind,
                    "default_host": sched.default_host,
                },
            }
        )
        return report

    app.include_router(routes_models.router)
    app.include_router(routes_profiles.router)
    app.include_router(routes_sessions.router)
    app.include_router(routes_gpu.router)
    app.include_router(routes_benchmarks.router)
    return app


app = create_app()
