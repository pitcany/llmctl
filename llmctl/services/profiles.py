"""Profile service.

Manages reusable launch profiles. Profiles are seeded from
``configs/profiles.yaml`` and persisted in the database so they can be
referenced by id from sessions, edited from the CLI/TUI/API, and exported back
to YAML for sharing. Lookups by name are supported for CLI ergonomics.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from llmctl.config import load_profiles
from llmctl.db import ProfileRecord, RuntimeName, utcnow
from llmctl.schemas import Profile, ProfileCreate, ProfileUpdate, ValidationIssue

# Fields on ProfileRecord that originate as promoted top-level launch knobs.
_PROMOTED_FIELDS = (
    "tensor_parallel_size",
    "max_model_len",
    "gpu_memory_utilization",
    "dtype",
    "quantization",
)


def record_to_profile(record: ProfileRecord) -> Profile:
    """Convert a profile record into the API schema."""
    return Profile(
        id=record.id,
        name=record.name,
        runtime=record.runtime,
        description=record.description,
        tensor_parallel_size=record.tensor_parallel_size,
        max_model_len=record.max_model_len,
        gpu_memory_utilization=record.gpu_memory_utilization,
        dtype=record.dtype,
        quantization=record.quantization,
        extra_args=list(record.extra_args or []),
        environment_variables=dict(record.environment_variables or {}),
        scheduler_preferences=dict(record.scheduler_preferences or {}),
        parameters=record.parameters,
        gpu_policy=record.gpu_policy,
        safety=record.safety,
    )


def _extract_promoted(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return a view of promoted launch knobs found in a ``parameters`` dict.

    The promoted columns are a typed convenience view: keys remain in
    ``parameters`` for backward compatibility with code that reads the full
    blob. This helper just copies out the subset that has typed columns.
    """
    return {key: parameters[key] for key in _PROMOTED_FIELDS if key in parameters}


class ProfileService:
    """Service for profile persistence, sync, lookup, and CRUD."""

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
            raw_params = entry.get("parameters", {}) or {}
            promoted = _extract_promoted(raw_params)
            self._upsert(
                name=str(name),
                runtime=RuntimeName(runtime),
                description=entry.get("description"),
                promoted=promoted,
                parameters=dict(raw_params),
                gpu_policy=entry.get("gpu_policy", {}) or {},
                safety=entry.get("safety", {}) or {},
                extra_args=list(entry.get("extra_args", []) or []),
                environment_variables=dict(entry.get("environment_variables", {}) or {}),
                scheduler_preferences=dict(
                    entry.get("scheduler_preferences", {}) or {}
                ),
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

    def get_by_id(self, profile_id: str) -> Profile | None:
        """Return a profile by id."""
        record = self.db.get(ProfileRecord, profile_id)
        return record_to_profile(record) if record else None

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

    def find(self, name_or_id: str) -> Profile | None:
        """Return a profile by id first, then by exact name match."""
        direct = self.get_by_id(name_or_id)
        if direct is not None:
            return direct
        return self.get_by_name(name_or_id)

    def create_profile(self, payload: ProfileCreate) -> Profile:
        """Persist a brand-new profile.

        Raises ValueError if the name already exists.
        """
        existing = self.db.exec(
            select(ProfileRecord).where(ProfileRecord.name == payload.name)
        ).first()
        if existing is not None:
            raise ValueError(f"Profile name '{payload.name}' already exists")
        record = ProfileRecord(
            name=payload.name,
            runtime=payload.runtime,
            description=payload.description,
            tensor_parallel_size=payload.tensor_parallel_size,
            max_model_len=payload.max_model_len,
            gpu_memory_utilization=payload.gpu_memory_utilization,
            dtype=payload.dtype,
            quantization=payload.quantization,
            extra_args=list(payload.extra_args),
            environment_variables=dict(payload.environment_variables),
            scheduler_preferences=dict(payload.scheduler_preferences),
            parameters=dict(payload.parameters),
            gpu_policy=dict(payload.gpu_policy),
            safety=dict(payload.safety),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record_to_profile(record)

    def update_profile(
        self, profile_id: str, updates: ProfileUpdate
    ) -> Profile | None:
        """Apply a partial update to a profile record."""
        record = self.db.get(ProfileRecord, profile_id)
        if record is None:
            return None
        data = updates.model_dump(exclude_unset=True)
        if "name" in data and data["name"] != record.name:
            clash = self.db.exec(
                select(ProfileRecord).where(ProfileRecord.name == data["name"])
            ).first()
            if clash is not None and clash.id != record.id:
                raise ValueError(f"Profile name '{data['name']}' already exists")
        for field_name, value in data.items():
            setattr(record, field_name, value)
        record.updated_at = utcnow()
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record_to_profile(record)

    def delete_profile(self, profile_id: str) -> bool:
        """Remove a profile record. Returns False if no such profile exists."""
        record = self.db.get(ProfileRecord, profile_id)
        if record is None:
            return False
        self.db.delete(record)
        self.db.commit()
        return True

    def clone_profile(self, profile_id: str, new_name: str) -> Profile | None:
        """Duplicate a profile under a new name."""
        record = self.db.get(ProfileRecord, profile_id)
        if record is None:
            return None
        clash = self.db.exec(
            select(ProfileRecord).where(ProfileRecord.name == new_name)
        ).first()
        if clash is not None:
            raise ValueError(f"Profile name '{new_name}' already exists")
        clone = ProfileRecord(
            name=new_name,
            runtime=record.runtime,
            description=record.description,
            tensor_parallel_size=record.tensor_parallel_size,
            max_model_len=record.max_model_len,
            gpu_memory_utilization=record.gpu_memory_utilization,
            dtype=record.dtype,
            quantization=record.quantization,
            extra_args=list(record.extra_args or []),
            environment_variables=dict(record.environment_variables or {}),
            scheduler_preferences=dict(record.scheduler_preferences or {}),
            parameters=dict(record.parameters),
            gpu_policy=dict(record.gpu_policy),
            safety=dict(record.safety),
        )
        self.db.add(clone)
        self.db.commit()
        self.db.refresh(clone)
        return record_to_profile(clone)

    def validate(
        self, profile: Profile | ProfileCreate | ProfileUpdate
    ) -> list[ValidationIssue]:
        """Return non-fatal validation issues for a profile payload."""
        return validate_profile(profile)

    def export_to_dict(self, profile: Profile) -> dict[str, Any]:
        """Return a YAML-friendly mapping for a single profile.

        ``parameters`` already contains the promoted fields (the columns are a
        typed view, not a replacement) so we only re-merge fields explicitly
        set on the Profile but missing from the blob, e.g. for profiles built
        through ``create_profile`` rather than synced from YAML.
        """
        merged_parameters = dict(profile.parameters)
        for field_name in _PROMOTED_FIELDS:
            value = getattr(profile, field_name)
            if value is not None and field_name not in merged_parameters:
                merged_parameters[field_name] = value
        return {
            "name": profile.name,
            "runtime": profile.runtime.value,
            "description": profile.description,
            "parameters": merged_parameters,
            "extra_args": list(profile.extra_args),
            "environment_variables": dict(profile.environment_variables),
            "scheduler_preferences": dict(profile.scheduler_preferences),
            "gpu_policy": dict(profile.gpu_policy),
            "safety": dict(profile.safety),
        }

    def import_from_dict(self, data: dict[str, Any]) -> Profile:
        """Create or update a profile from a YAML-style mapping."""
        name = data.get("name")
        runtime = data.get("runtime")
        if not name or not runtime:
            raise ValueError("Profile import requires 'name' and 'runtime'")
        raw_params = data.get("parameters", {}) or {}
        promoted = _extract_promoted(raw_params)
        payload = ProfileCreate(
            name=str(name),
            runtime=RuntimeName(runtime),
            description=data.get("description"),
            tensor_parallel_size=promoted.get("tensor_parallel_size"),
            max_model_len=promoted.get("max_model_len"),
            gpu_memory_utilization=promoted.get("gpu_memory_utilization"),
            dtype=promoted.get("dtype"),
            quantization=promoted.get("quantization"),
            extra_args=list(data.get("extra_args", []) or []),
            environment_variables=dict(data.get("environment_variables", {}) or {}),
            scheduler_preferences=dict(data.get("scheduler_preferences", {}) or {}),
            parameters=dict(raw_params),
            gpu_policy=dict(data.get("gpu_policy", {}) or {}),
            safety=dict(data.get("safety", {}) or {}),
        )
        existing = self.db.exec(
            select(ProfileRecord).where(ProfileRecord.name == payload.name)
        ).first()
        if existing is None:
            return self.create_profile(payload)
        updates = ProfileUpdate(**payload.model_dump())
        result = self.update_profile(existing.id, updates)
        if result is None:
            # Concurrent delete between our SELECT and the UPDATE inside
            # update_profile. Don't ``assert`` (stripped under ``python -O``);
            # raise a typed error the caller can translate to a 409/retry.
            raise RuntimeError(
                f"Profile '{payload.name}' was deleted concurrently during import"
            )
        return result

    def _upsert(
        self,
        name: str,
        runtime: RuntimeName,
        description: str | None,
        promoted: dict[str, Any],
        parameters: dict[str, Any],
        gpu_policy: dict[str, Any],
        safety: dict[str, Any],
        extra_args: list[str],
        environment_variables: dict[str, str],
        scheduler_preferences: dict[str, Any],
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
            existing.extra_args = extra_args
            existing.environment_variables = environment_variables
            existing.scheduler_preferences = scheduler_preferences
            for key in _PROMOTED_FIELDS:
                if key in promoted:
                    setattr(existing, key, promoted[key])
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
            extra_args=extra_args,
            environment_variables=environment_variables,
            scheduler_preferences=scheduler_preferences,
            tensor_parallel_size=promoted.get("tensor_parallel_size"),
            max_model_len=promoted.get("max_model_len"),
            gpu_memory_utilization=promoted.get("gpu_memory_utilization"),
            dtype=promoted.get("dtype"),
            quantization=promoted.get("quantization"),
        )
        self.db.add(record)
        try:
            self.db.commit()
        except IntegrityError:
            # Concurrent sync_from_yaml: another thread inserted the same name
            # between our SELECT and our COMMIT. Roll back and treat as an
            # update against whatever the other thread persisted.
            self.db.rollback()
            existing = self.db.exec(
                select(ProfileRecord).where(ProfileRecord.name == name)
            ).first()
            if existing is not None:
                existing.runtime = runtime
                existing.description = description
                existing.parameters = parameters
                existing.gpu_policy = gpu_policy
                existing.safety = safety
                existing.extra_args = extra_args
                existing.environment_variables = environment_variables
                existing.scheduler_preferences = scheduler_preferences
                for key in _PROMOTED_FIELDS:
                    if key in promoted:
                        setattr(existing, key, promoted[key])
                existing.updated_at = utcnow()
                self.db.add(existing)
                self.db.commit()


def validate_profile(
    profile: Profile | ProfileCreate | ProfileUpdate,
) -> list[ValidationIssue]:
    """Non-fatal validation for a profile.

    Profile editing should warn but not block: a user may legitimately want to
    save a profile aimed at a host with more GPUs than the current machine, or
    intended to be paired with a future model. ``severity="error"`` issues are
    structural (e.g. negative tp), not host-specific.
    """
    issues: list[ValidationIssue] = []
    runtime = getattr(profile, "runtime", None)
    tp = getattr(profile, "tensor_parallel_size", None)
    max_len = getattr(profile, "max_model_len", None)
    gpu_util = getattr(profile, "gpu_memory_utilization", None)

    if tp is not None:
        if tp < 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="tensor_parallel_size",
                    message="tensor_parallel_size must be >= 1",
                )
            )
        elif tp > 1 and runtime in {
            RuntimeName.LLAMA_CPP,
            RuntimeName.OLLAMA,
            RuntimeName.LMSTUDIO,
        }:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="tensor_parallel_size",
                    message=(
                        f"tensor_parallel_size={tp} has no effect for runtime "
                        f"'{runtime.value if runtime else '?'}'"
                    ),
                )
            )

    if max_len is not None and max_len < 1:
        issues.append(
            ValidationIssue(
                severity="error",
                field="max_model_len",
                message="max_model_len must be >= 1",
            )
        )

    if gpu_util is not None:
        if gpu_util <= 0 or gpu_util > 1:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="gpu_memory_utilization",
                    message="gpu_memory_utilization must be in (0, 1]",
                )
            )
        elif gpu_util > 0.95:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="gpu_memory_utilization",
                    message=(
                        "gpu_memory_utilization > 0.95 leaves little room "
                        "for activations; OOM risk on big contexts"
                    ),
                )
            )

    name = getattr(profile, "name", None)
    if name is not None and not name.replace("-", "").replace("_", "").isalnum():
        issues.append(
            ValidationIssue(
                severity="warning",
                field="name",
                message="profile name should be alphanumeric with - or _ only",
            )
        )

    return issues
