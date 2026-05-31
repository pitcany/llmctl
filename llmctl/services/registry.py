"""Model registry service.

Manages model records and runtime discovery. ``scan`` queries every runtime
adapter for available models and upserts them into the registry, marking newly
found models as ``DISCOVERED``.
"""

from __future__ import annotations

import asyncio

from sqlmodel import Session, select

from llmctl.db import ModelRecord, ModelStatus, utcnow
from llmctl.schemas import Model, ModelCreate
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
        tags=record.tags,
        metadata=record.metadata_json,
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


class RegistryService:
    """Service interface for model registration and discovery."""

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
        """Discover models from all runtime adapters and persist them."""
        for runtime in self.router.list_runtimes():
            adapter = self.router.get_adapter(runtime)
            try:
                discovered = asyncio.run(adapter.discover_models())
            except Exception:
                discovered = []
            for model in discovered:
                self._upsert(model)
        return self.list_models()

    def list_models(self) -> list[Model]:
        """List non-deleted models."""
        records = self.db.exec(
            select(ModelRecord).where(ModelRecord.status != ModelStatus.DELETED)
        ).all()
        return [record_to_model(record) for record in records]

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
            tags=payload.tags,
            metadata_json=payload.metadata,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record_to_model(record)

    def delete_model(self, model_id: str) -> bool:
        """Soft-delete a model record."""
        record = self.db.get(ModelRecord, model_id)
        if not record:
            return False
        record.status = ModelStatus.DELETED
        record.updated_at = utcnow()
        self.db.add(record)
        self.db.commit()
        return True

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
        )
        self.db.add(record)
        self.db.commit()
