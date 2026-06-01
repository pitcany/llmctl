"""Model registry API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from llmctl.api.deps import get_db_session
from llmctl.schemas import Model, ModelCreate, ModelUpdate
from llmctl.services.registry import RegistryService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[Model])
def list_models(
    include_inactive: bool = False,
    db: Session = Depends(get_db_session),
) -> list[Model]:
    """List registered models. ``?include_inactive=true`` adds disabled rows."""
    return RegistryService(db).list_models(include_inactive=include_inactive)


@router.post("", response_model=Model, status_code=status.HTTP_201_CREATED)
def add_model(payload: ModelCreate, db: Session = Depends(get_db_session)) -> Model:
    """Register a model."""
    return RegistryService(db).add_model(payload)


@router.get("/{model_id}", response_model=Model)
def get_model(model_id: str, db: Session = Depends(get_db_session)) -> Model:
    """Fetch a single model by id."""
    model = RegistryService(db).get_model(model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return model


@router.put("/{model_id}", response_model=Model)
def update_model(
    model_id: str,
    updates: ModelUpdate,
    db: Session = Depends(get_db_session),
) -> Model:
    """Partial-update a model record."""
    updated = RegistryService(db).update_model(model_id, updates)
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    return updated


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_model(
    model_id: str,
    delete_files: bool = False,
    db: Session = Depends(get_db_session),
) -> None:
    """Soft-delete a model. ``?delete_files=true`` also removes the artifact."""
    deleted = RegistryService(db).delete_model(model_id, delete_files=delete_files)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
