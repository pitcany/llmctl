"""Schema API route: the vocabularies and models of the running llmctl."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from llmctl.schema_export import build_schema

router = APIRouter(prefix="/schema", tags=["schema"])


@router.get("")
def get_schema() -> dict[str, Any]:
    """Every state vocabulary, the session transition table, the phase
    renderings and the API models as JSON Schema (``docs/STATE-MACHINES.md``)."""
    return build_schema()
