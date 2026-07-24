"""Model registry service.

Manages model records and runtime discovery. ``scan`` queries every runtime
adapter for available models and upserts them into the registry, marking newly
found models as ``DISCOVERED``. CRUD methods (``add``, ``update``, ``clone``,
``enable``/``disable``, ``delete``) cover the management surface exposed to the
CLI, TUI, and API.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlmodel import Session, select

from llmctl.adapters.base import RuntimeAdapter
from llmctl.db import ModelRecord, ModelStatus, RuntimeName, utcnow
from llmctl.schemas import HealthState, Model, ModelCreate, ModelUpdate
from llmctl.services.router import RuntimeRouter


def record_to_model(record: ModelRecord) -> Model:
    """Convert a database record into an API schema."""
    return Model(
        id=record.id,
        name=record.name,
        runtime=record.runtime,
        source=record.source,
        path=record.path,
        format=record.format,
        quantization=record.quantization,
        size_bytes=record.size_bytes,
        estimated_vram_gb=record.estimated_vram_gb,
        max_context=record.max_context,
        parameter_count=record.parameter_count,
        notes=record.notes,
        default_profile_id=record.default_profile_id,
        active=record.active if record.active is not None else True,
        tags=record.tags,
        metadata=record.metadata_json,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


class RegistryService:
    """Service interface for model registration, discovery, and CRUD."""

    def __init__(self, db: Session, router: RuntimeRouter | None = None) -> None:
        self.db = db
        self._router = router

    @property
    def router(self) -> RuntimeRouter:
        """Return the runtime router, constructing a default on first use."""
        if self._router is None:
            self._router = RuntimeRouter()
        return self._router

    def scan(self) -> list[Model]:
        """Discover models from all runtime adapters, persist them, and reconcile.

        For each runtime reachable this pass, any previously auto-discovered
        model no longer reported is flagged ``MISSING``. Runtimes whose adapter
        is not healthy are skipped entirely, so a down daemon never false-flags
        its models.
        """
        for runtime in self.router.list_runtimes():
            adapter = self.router.get_adapter(runtime)
            try:
                reachable, discovered = asyncio.run(self._probe(adapter))
            except Exception:
                reachable, discovered = False, []
            for model in discovered:
                self._upsert(model)
            if reachable:
                self._reconcile_missing(runtime, discovered)
        return self.list_models()

    @staticmethod
    async def _probe(adapter: RuntimeAdapter) -> tuple[bool, list[Model]]:
        """Return ``(reachable, discovered)`` for a single adapter.

        Health check and discovery run in the same coroutine. A runtime counts
        as reachable only when health is OK *and* the discovery call itself
        succeeded: HTTP adapters return ``[]`` on a failed listing just as for
        an empty catalog, so treating a failed listing as "empty" would let
        reconcile wrongly flag every model MISSING. When not reachable,
        discovery results are discarded and ``[]`` is returned so reconcile is
        skipped entirely.
        """
        health = await adapter.health_check()
        if health.state != HealthState.OK:
            return False, []
        discovered = await adapter.discover_models()
        if getattr(adapter, "last_discovery_ok", True) is False:
            return False, []
        return True, discovered

    def _reconcile_missing(self, runtime: RuntimeName, discovered: list[Model]) -> None:
        """Flag auto-discovered models of ``runtime`` absent from ``discovered``.

        Only ``DISCOVERED`` rows are touched, so manually registered models
        (``REGISTERED``) are never affected. Rediscovery later restores a row
        via :meth:`_upsert` (``MISSING -> DISCOVERED``).

        Rows are additionally exempt unless the on-disk artifact is provably
        gone — see :meth:`_artifact_is_gone`. Single-model servers (the
        vllm-tp unit) report only the currently served model, so absence from
        one scan means the unit rotated, not that the checkpoint vanished.
        ``MISSING`` is reserved for artifacts that are actually gone.
        """
        discovered_keys = {(runtime, m.source or m.name) for m in discovered}
        stale = self.db.exec(
            select(ModelRecord).where(
                ModelRecord.runtime == runtime,
                ModelRecord.status == ModelStatus.DISCOVERED,
            )
        ).all()
        changed = False
        for record in stale:
            if (record.runtime, record.source or record.name) in discovered_keys:
                continue
            if not self._artifact_is_gone(record):
                continue
            record.status = ModelStatus.MISSING
            record.updated_at = utcnow()
            self.db.add(record)
            changed = True
        if changed:
            self.db.commit()

    @staticmethod
    def _artifact_is_gone(record: ModelRecord) -> bool:
        """Whether ``record``'s artifact is provably absent from local disk.

        Three cases, and only the last is a disappearance:

        * **No path** — nothing to check, so the runtime's own report decides
          (ollama and LM Studio rows carry no path; this is their normal path).
        * **Non-absolute path** — not a local filesystem artifact at all.
          vLLM reports ``root`` as the ``--model`` value, which is a Hugging
          Face repo id (``org/name``) whenever the server was pointed at the
          hub. ``Path("org/name").exists()`` answers a question about the
          process working directory, not about the model, so never judge on it.
        * **Absolute path** — a real location; its absence is real evidence.
        """
        raw = (record.path or "").strip()
        if not raw:
            return True
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            return False
        return not candidate.exists()

    def scan_discovered_only(self) -> list[Model]:
        """Return *new* adapter-discovered models without persisting them.

        Used by ``llmctl scan`` (without ``--import``) so the user can review
        candidates before committing them. Filters out anything already in
        the registry — matching the ``(runtime, source or name)`` upsert key
        used by ``_upsert`` — so the dry-run preview shows only the diff,
        not every model the adapter could find.
        """
        discovered: list[Model] = []
        for runtime in self.router.list_runtimes():
            adapter = self.router.get_adapter(runtime)
            try:
                results = asyncio.run(adapter.discover_models())
            except Exception:
                results = []
            discovered.extend(results)
        existing = self.db.exec(
            select(ModelRecord).where(ModelRecord.status != ModelStatus.DELETED)
        ).all()
        existing_keys = {
            (record.runtime, record.source or record.name) for record in existing
        }
        return [
            model
            for model in discovered
            if (model.runtime, model.source or model.name) not in existing_keys
        ]

    def list_models(self, include_inactive: bool = False) -> list[Model]:
        """List non-deleted models. Inactive rows are hidden by default.

        ``apply_migrations`` adds the ``active`` column without a SQL
        ``DEFAULT TRUE`` clause, so rows registered before this migration
        keep ``active IS NULL``. SQL's three-valued logic makes
        ``active != FALSE`` evaluate to NULL (falsy in WHERE) for those
        rows, which would silently hide every pre-migration model. Filter
        explicitly: include ``TRUE`` and ``NULL``, exclude only ``FALSE``.
        """
        statement = select(ModelRecord).where(ModelRecord.status != ModelStatus.DELETED)
        if not include_inactive:
            statement = statement.where(
                (ModelRecord.active.is_(True)) | (ModelRecord.active.is_(None))
            )
        records = self.db.exec(statement).all()
        return [record_to_model(record) for record in records]

    def get_model(self, model_id: str) -> Model | None:
        """Return a model by id (active or inactive, but not deleted)."""
        record = self.db.get(ModelRecord, model_id)
        if not record or record.status == ModelStatus.DELETED:
            return None
        return record_to_model(record)

    def find(self, name_or_id: str) -> Model | None:
        """Return a model by id first, then by exact name match.

        Raises ValueError when ``name_or_id`` is a name shared across multiple
        runtimes — the caller must disambiguate with the model id.
        """
        direct = self.db.get(ModelRecord, name_or_id)
        if direct is not None and direct.status != ModelStatus.DELETED:
            return record_to_model(direct)
        matches = self.db.exec(
            select(ModelRecord).where(
                ModelRecord.name == name_or_id,
                ModelRecord.status != ModelStatus.DELETED,
            )
        ).all()
        if not matches:
            return None
        if len(matches) > 1:
            runtimes = ", ".join(sorted({m.runtime.value for m in matches}))
            raise ValueError(
                f"Model name '{name_or_id}' is ambiguous across runtimes "
                f"[{runtimes}]; pass the model id instead."
            )
        return record_to_model(matches[0])

    def add_model(self, payload: ModelCreate) -> Model:
        """Register a model record."""
        record = ModelRecord(
            name=payload.name,
            runtime=payload.runtime,
            source=payload.source,
            path=payload.path,
            format=payload.format,
            quantization=payload.quantization,
            estimated_vram_gb=payload.estimated_vram_gb,
            max_context=payload.max_context,
            parameter_count=payload.parameter_count,
            notes=payload.notes,
            default_profile_id=payload.default_profile_id,
            active=True,
            tags=payload.tags,
            metadata_json=payload.metadata,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record_to_model(record)

    def update_model(self, model_id: str, updates: ModelUpdate) -> Model | None:
        """Apply a partial update to a model record."""
        record = self.db.get(ModelRecord, model_id)
        if not record or record.status == ModelStatus.DELETED:
            return None
        data = updates.model_dump(exclude_unset=True)
        if "metadata" in data:
            record.metadata_json = data.pop("metadata") or {}
        for field_name, value in data.items():
            setattr(record, field_name, value)
        record.updated_at = utcnow()
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record_to_model(record)

    def clone_model(self, model_id: str, new_name: str) -> Model | None:
        """Duplicate a model record under a new name."""
        record = self.db.get(ModelRecord, model_id)
        if not record or record.status == ModelStatus.DELETED:
            return None
        clone = ModelRecord(
            name=new_name,
            runtime=record.runtime,
            source=record.source,
            path=record.path,
            format=record.format,
            quantization=record.quantization,
            size_bytes=record.size_bytes,
            estimated_vram_gb=record.estimated_vram_gb,
            max_context=record.max_context,
            parameter_count=record.parameter_count,
            notes=record.notes,
            default_profile_id=record.default_profile_id,
            active=True,
            tags=list(record.tags),
            metadata_json=dict(record.metadata_json),
            status=ModelStatus.REGISTERED,
        )
        self.db.add(clone)
        self.db.commit()
        self.db.refresh(clone)
        return record_to_model(clone)

    def enable_model(self, model_id: str) -> bool:
        """Mark a model as active."""
        return self._set_active(model_id, True)

    def disable_model(self, model_id: str) -> bool:
        """Mark a model as inactive (hidden from default listings)."""
        return self._set_active(model_id, False)

    def delete_model(self, model_id: str, *, delete_files: bool = False) -> bool:
        """Soft-delete a model record.

        ``delete_files=True`` additionally removes the on-disk artifact at
        ``record.path``. This is intentionally opt-in: callers must explicitly
        request file deletion (matching the ``--delete-files`` CLI flag).
        Missing paths and unexpected I/O errors are silently skipped.
        """
        record = self.db.get(ModelRecord, model_id)
        if not record:
            return False
        if delete_files and record.path:
            self._delete_artifact(record.path)
        record.status = ModelStatus.DELETED
        record.updated_at = utcnow()
        self.db.add(record)
        self.db.commit()
        return True

    def prune_missing(
        self,
        runtime: RuntimeName | None = None,
        *,
        ids: list[str] | None = None,
    ) -> int:
        """Soft-delete (status ``DELETED``) every ``MISSING`` model.

        Optionally restrict to a single ``runtime``, or to an explicit ``ids``
        allow-list. The allow-list lets a caller bind the set it showed the
        user, so a concurrent scan flagging new rows cannot widen the prune
        beyond what was confirmed. Returns the count pruned.
        """
        statement = select(ModelRecord).where(ModelRecord.status == ModelStatus.MISSING)
        if runtime is not None:
            statement = statement.where(ModelRecord.runtime == runtime)
        if ids is not None:
            statement = statement.where(ModelRecord.id.in_(ids))  # type: ignore[attr-defined]
        records = self.db.exec(statement).all()
        for record in records:
            record.status = ModelStatus.DELETED
            record.updated_at = utcnow()
            self.db.add(record)
        if records:
            self.db.commit()
        return len(records)

    def _set_active(self, model_id: str, value: bool) -> bool:
        record = self.db.get(ModelRecord, model_id)
        if not record or record.status == ModelStatus.DELETED:
            return False
        record.active = value
        record.updated_at = utcnow()
        self.db.add(record)
        self.db.commit()
        return True

    @staticmethod
    def _delete_artifact(path_str: str) -> None:
        """Remove a file or directory referenced by a model record.

        Best-effort: missing or unreadable paths leave the registry update
        unchanged. Symlinks are unlinked, not followed.
        """
        try:
            target = Path(path_str)
            if not target.exists() and not target.is_symlink():
                return
            if target.is_symlink() or target.is_file():
                target.unlink()
                return
            if target.is_dir():
                for sub in sorted(
                    target.rglob("*"), key=lambda p: len(p.parts), reverse=True
                ):
                    if sub.is_symlink() or sub.is_file():
                        sub.unlink()
                    elif sub.is_dir():
                        sub.rmdir()
                target.rmdir()
        except OSError:
            return

    def _upsert(self, model: Model) -> None:
        """Insert a discovered model or refresh an existing record."""
        key_source = model.source or model.name
        existing = self.db.exec(
            select(ModelRecord).where(
                ModelRecord.runtime == model.runtime,
                ModelRecord.source == key_source,
            )
        ).first()
        if existing is not None:
            if existing.status == ModelStatus.MISSING:
                existing.status = ModelStatus.DISCOVERED
            existing.path = model.path or existing.path
            existing.format = model.format or existing.format
            existing.quantization = model.quantization or existing.quantization
            existing.size_bytes = model.size_bytes or existing.size_bytes
            existing.metadata_json = {**existing.metadata_json, **model.metadata}
            existing.updated_at = utcnow()
            self.db.add(existing)
            self.db.commit()
            return
        record = ModelRecord(
            name=model.name,
            runtime=model.runtime,
            source=key_source,
            path=model.path,
            format=model.format,
            quantization=model.quantization,
            size_bytes=model.size_bytes,
            tags=model.tags,
            metadata_json=model.metadata,
            status=ModelStatus.DISCOVERED,
            active=True,
        )
        self.db.add(record)
        self.db.commit()
