"""Swap API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from llmctl.api.deps import get_settings, get_swap_jobs
from llmctl.config import Settings
from llmctl.services import swap_jobs, swap_view
from llmctl.services.swap_jobs import JobBusy, SwapJobs

router = APIRouter(prefix="/swap", tags=["swap"])


class SwapRow(BaseModel):
    number: int
    name: str
    model: str
    engine: str
    kind: str
    cards: Any
    state: str
    last_load_s: Any = None
    endpoint: str | None = None
    aliases: list[str]


class SwapJobSummary(BaseModel):
    job_id: str
    name: str
    status: str
    started_at: str


class SwapJobDetail(BaseModel):
    job_id: str
    name: str
    model: str
    status: str
    exit_code: int | None = None
    output: list[str]
    started_at: str
    finished_at: str | None = None


class SwapList(BaseModel):
    rows: list[SwapRow]
    gateway_url: str
    job: SwapJobSummary | None = None


@router.get("", response_model=SwapList)
def list_swap(
    settings: Settings = Depends(get_settings),
    jobs: SwapJobs = Depends(get_swap_jobs),
) -> dict[str, Any]:
    """The swap list, the gateway URL, and the running swap job (if any)."""
    from llmctl.services import lanes

    state = lanes.gather_state(settings)
    running = jobs.running()
    return {
        "rows": swap_view.rows_json(state, swap_view.gateway_aliases(settings)),
        "gateway_url": swap_view.gateway_url(settings.router),
        "job": running.summary() if running else None,
    }


@router.get("/jobs/{job_id}", response_model=SwapJobDetail)
def get_swap_job(job_id: str, jobs: SwapJobs = Depends(get_swap_jobs)) -> dict[str, Any]:
    """One swap job with its captured output."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Swap job not found")
    return job.to_dict()


class RunRequest(BaseModel):
    name: str
    dry_run: bool = False


@router.post("/run")
def run_swap(
    body: RunRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
    jobs: SwapJobs = Depends(get_swap_jobs),
) -> dict[str, Any]:
    """Swap to a lane, preset or session; starts a background job."""
    from llmctl.services import lanes

    state = lanes.gather_state(settings)
    catalog = state.catalog
    target = lanes.resolve(body.name, catalog)
    if target is None and body.name.isdecimal() and 1 <= int(body.name) <= len(catalog):
        target = catalog[int(body.name) - 1]
    if target is None:
        close = lanes.close_matches(body.name, catalog)[:3]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No swap target matches {body.name!r}.{hint}",
        )

    preview = state.preview(target)
    info: dict[str, Any] = {
        "name": target.key,
        "model": target.model,
        "engine": target.engine,
        "kind": target.kind,
        "endpoint": swap_view.endpoint_for(state, target),
        "preview": preview.lines(),
    }
    if body.dry_run or (preview.already_serving and target.kind in ("lane", "session")):
        return {**info, "already_serving": preview.already_serving}
    try:
        job = jobs.start(target.key, target.model, swap_jobs.build_run_argv(target.key))
    except JobBusy as busy:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Swap job {busy.job.job_id} is already running",
        ) from busy
    response.status_code = status.HTTP_202_ACCEPTED
    return {**info, "job_id": job.job_id, "status": "running"}
