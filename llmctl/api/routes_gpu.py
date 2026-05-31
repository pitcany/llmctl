"""GPU API routes."""

from __future__ import annotations

from fastapi import APIRouter

from llmctl.schemas import GPUInfo
from llmctl.telemetry.gpu import get_gpu_info

router = APIRouter(prefix="/gpus", tags=["gpu"])


@router.get("", response_model=list[GPUInfo])
def list_gpus() -> list[GPUInfo]:
    """Return current GPU telemetry or an empty list when unavailable."""
    return get_gpu_info()
