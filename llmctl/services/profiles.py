"""Profile service.

Manages reusable launch profiles. Profiles are defined in
``configs/profiles.yaml`` and synced into the database so they can be referenced
by id from sessions. Lookups by name are supported for CLI ergonomics.
"""

from __future__ import annotations

from sqlmodel import Session, select

from llmctl.config import load_profiles
from llmctl.db import ProfileRecord, RuntimeName, utcnow
from llmctl.schemas import Profile


def record_to_profile(record: ProfileRecord) -> Profile:
    """Convert a profile record into the API schema."""
    return Profile(
        id=record.id,
        name=record.name,
        runtime=record.runtime,
        description=record.description,
        parameters=record.parameters,
        gpu_policy=record.gpu_policy,
        safety=record.safety,
    )


class ProfileService:
    """Service for profile persistence, sync, and lookup."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def sync_from_yaml(self) -> list[Profile]:
        """Upsert profiles defined in ``profiles.yaml`` into the database."""
        config = load_profiles()
        for entry in config.profiles:
            name = entry.get("name")
            runtime = entry.get("runtime")
            if not name or not runtime:
                continue
            self._upsert(
                name=str(name),
                runtime=RuntimeName(runtime),
                description=entry.get("description"),
                parameters=entry.get("parameters", {}) or {},
                gpu_policy=entry.get("gpu_policy", {}) or {},
                safety=entry.get("safety", {}) or {},
            )
        records = self.db.exec(select(ProfileRecord)).all()
        return [record_to_profile(record) for record in records]

    def list_profiles(self) -> list[Profile]:
        """List all profiles, syncing from YAML first when the table is empty."""
        records = self.db.exec(select(ProfileRecord)).all()
        if not records:
            self.sync_from_yaml()
            records = self.db.exec(select(ProfileRecord)).all()
        return [record_to_profile(record) for record in records]

    def get_by_name(self, name: str) -> Profile | None:
        """Return a profile by name, syncing from YAML when missing."""
        record = self.db.exec(
            select(ProfileRecord).where(ProfileRecord.name == name)
        ).first()
        if record is None:
            self.sync_from_yaml()
            record = self.db.exec(
                select(ProfileRecord).where(ProfileRecord.name == name)
            ).first()
        return record_to_profile(record) if record else None

    def _upsert(
        self,
        name: str,
        runtime: RuntimeName,
        description: str | None,
        parameters: dict,
        gpu_policy: dict,
        safety: dict,
    ) -> None:
        """Insert or update a profile record by unique name."""
        existing = self.db.exec(
            select(ProfileRecord).where(ProfileRecord.name == name)
        ).first()
        if existing is not None:
            existing.runtime = runtime
            existing.description = description
            existing.parameters = parameters
            existing.gpu_policy = gpu_policy
            existing.safety = safety
            existing.updated_at = utcnow()
            self.db.add(existing)
            self.db.commit()
            return
        record = ProfileRecord(
            name=name,
            runtime=runtime,
            description=description,
            parameters=parameters,
            gpu_policy=gpu_policy,
            safety=safety,
        )
        self.db.add(record)
        self.db.commit()
