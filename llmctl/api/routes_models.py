"""Model registry API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from llmctl.api.deps import get_db_session
from llmctl.schemas import Model, ModelCreate
from llmctl.services.registry import RegistryService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[Model])
def list_models(db: Session = Depends(get_db_session)) -> list[Model]:
    """List registered models."""
    return RegistryService(db).list_models()


@router.post("", response_model=Model, status_code=status.HTTP_201_CREATED)
def add_model(payload: ModelCreate, db: Session = Depends(get_db_session)) -> Model:
    """Register a model."""
    return RegistryService(db).add_model(payload)


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(model_id: str, db: Session = Depends(get_db_session)) -> None:
    """Soft-delete a model."""
    deleted = RegistryService(db).delete_model(model_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
