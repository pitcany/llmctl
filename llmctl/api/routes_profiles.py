"""Profile API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from llmctl.api.deps import get_db_session
from llmctl.schemas import Profile, ProfileCreate, ProfileUpdate, ValidationIssue
from llmctl.services.profiles import ProfileService

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("", response_model=list[Profile])
def list_profiles(db: Session = Depends(get_db_session)) -> list[Profile]:
    """List launch profiles."""
    return ProfileService(db).list_profiles()


@router.post("", response_model=Profile, status_code=status.HTTP_201_CREATED)
def create_profile(
    payload: ProfileCreate, db: Session = Depends(get_db_session)
) -> Profile:
    """Create a new launch profile."""
    service = ProfileService(db)
    issues = service.validate(payload)
    if any(issue.severity == "error" for issue in issues):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "Profile failed validation",
                "issues": [issue.model_dump() for issue in issues],
            },
        )
    try:
        return service.create_profile(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc


@router.get("/{profile_id}", response_model=Profile)
def get_profile(profile_id: str, db: Session = Depends(get_db_session)) -> Profile:
    """Fetch a single profile by id or name."""
    profile = ProfileService(db).find(profile_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )
    return profile


@router.put("/{profile_id}", response_model=Profile)
def update_profile(
    profile_id: str,
    updates: ProfileUpdate,
    db: Session = Depends(get_db_session),
) -> Profile:
    """Partial-update a profile record."""
    service = ProfileService(db)
    existing = service.find(profile_id)
    if existing is None or existing.id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )
    issues = service.validate(updates)
    if any(issue.severity == "error" for issue in issues):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "Profile failed validation",
                "issues": [issue.model_dump() for issue in issues],
            },
        )
    try:
        updated = service.update_profile(existing.id, updates)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    assert updated is not None
    return updated


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_profile(profile_id: str, db: Session = Depends(get_db_session)) -> None:
    """Delete a profile."""
    service = ProfileService(db)
    existing = service.find(profile_id)
    if existing is None or existing.id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )
    service.delete_profile(existing.id)


@router.post("/{profile_id}/validate", response_model=list[ValidationIssue])
def validate_profile_endpoint(
    profile_id: str,
    updates: ProfileUpdate,
    db: Session = Depends(get_db_session),
) -> list[ValidationIssue]:
    """Return validation issues for a hypothetical profile update.

    Lets the TUI/web client preview warnings before issuing PUT.
    """
    service = ProfileService(db)
    existing = service.find(profile_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found"
        )
    return service.validate(updates)
