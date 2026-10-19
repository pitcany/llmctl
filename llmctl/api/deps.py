"""FastAPI dependency helpers."""

from __future__ import annotations

from collections.abc import Generator

from sqlmodel import Session

from llmctl.config import Settings
from llmctl.services.swap_jobs import SwapJobs


def get_db_session() -> Generator[Session, None, None]:
    """Database session dependency overridden by the app factory.

    The placeholder keeps route modules independent from the app factory and
    avoids circular imports.
    """
    raise RuntimeError("Database dependency has not been configured by create_app().")
    yield  # pragma: no cover


def get_settings() -> Settings:
    """Effective settings dependency overridden by the app factory."""
    raise RuntimeError("Settings dependency has not been configured by create_app().")


def get_swap_jobs() -> SwapJobs:
    """Per-app swap job registry overridden by the app factory."""
    raise RuntimeError("Swap jobs dependency has not been configured by create_app().")
