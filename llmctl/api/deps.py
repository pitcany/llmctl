"""FastAPI dependency helpers."""

from __future__ import annotations

from collections.abc import Generator

from sqlmodel import Session


def get_db_session() -> Generator[Session, None, None]:
    """Database session dependency overridden by the app factory.

    The placeholder keeps route modules independent from the app factory and
    avoids circular imports.
    """
    raise RuntimeError("Database dependency has not been configured by create_app().")
    yield  # pragma: no cover
