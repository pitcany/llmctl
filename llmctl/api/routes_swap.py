"""Swap API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from llmctl.api.deps import get_settings, get_swap_jobs
from llmctl.config import Settings
from llmctl.services import stop_serving, swap_jobs, swap_view
from llmctl.services.op_lock import LockBusy
from llmctl.services.operations import IdempotencyConflict, OperationPhase
from llmctl.services.swap_jobs import (
    JobBusy,
    OperationConflict,
    StopInProgress,
    SwapJobs,
)

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
    operation_id: str | None = None
    phase: str | None = None


class SwapJobDetail(BaseModel):
    job_id: str
    name: str
    model: str
    status: str
    exit_code: int | None = None
    output: list[str]
    started_at: str
    finished_at: str | None = None
    operation_id: str | None = None
    phase: str | None = None
    recovery: dict[str, Any] = Field(default_factory=dict)


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
    summary = running.summary() if running else None
    if running is not None and summary is not None:
        record = jobs.operation_for(running)
        if record is not None:
            summary["phase"] = record.phase
    return {
        "rows": swap_view.rows_json(state, swap_view.gateway_aliases(settings)),
        "gateway_url": swap_view.gateway_url(settings.router),
        "job": summary,
    }


@router.get("/jobs/{job_id}", response_model=SwapJobDetail)
def get_swap_job(job_id: str, jobs: SwapJobs = Depends(get_swap_jobs)) -> dict[str, Any]:
    """One swap job with its captured output and operation phase."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Swap job not found")
    detail = job.to_dict()
    record = jobs.operation_for(job)
    if record is not None:
        detail["phase"] = record.phase
        detail["recovery"] = record.recovery or {}
    return detail


class RunRequest(BaseModel):
    name: str
    dry_run: bool = False
    #: Optional idempotency key: repeating it with the same request returns
    #: the existing operation instead of starting a second switch; reusing
    #: it for a different request is refused (FIXPLAN §2B).
    idempotency_key: str | None = None


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
    # The length cap keeps int() away from Python's 4300-digit limit (a 500).
    if (
        target is None
        and body.name.isdecimal()
        and len(body.name) <= 6
        and 1 <= int(body.name) <= len(catalog)
    ):
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
        outcome = jobs.start_with_operation(
            target.key,
            target.model,
            swap_jobs.build_run_argv(target.key),
            idempotency_key=body.idempotency_key,
            previous_state={
                "serving": [
                    {"kind": row.kind, "key": row.key, "model": row.model}
                    for row in catalog
                    if row.key in state.serving
                ]
            },
        )
    except JobBusy as busy:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Swap job {busy.job.job_id} is already running",
        ) from busy
    except StopInProgress as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except OperationConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{exc}. It survived this API process; wait for it or "
            "check `llmctl status` before retrying.",
        ) from exc
    if outcome.replayed and outcome.operation is not None:
        # Same key, same request: the existing operation is the answer.
        return {
            **info,
            "job_id": outcome.operation.job_id,
            "operation_id": outcome.operation.id,
            "phase": outcome.operation.phase,
            "status": "already-started",
        }
    response.status_code = status.HTTP_202_ACCEPTED
    return {
        **info,
        "job_id": outcome.job.job_id if outcome.job else None,
        "operation_id": outcome.operation.id if outcome.operation else None,
        "phase": outcome.operation.phase if outcome.operation else None,
        "status": "running",
    }


class StopRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str | None = None
    all_: bool = Field(False, alias="all")
    dry_run: bool = False


@router.post("/stop")
def stop_swap(
    body: StopRequest,
    settings: Settings = Depends(get_settings),
    jobs: SwapJobs = Depends(get_swap_jobs),
) -> dict[str, Any]:
    """Stop what is serving: one named item, or everything with ``all``."""
    from llmctl.services import lanes

    items = stop_serving.serving_now(lanes.gather_state(settings))
    if body.name is not None:
        items = stop_serving.pick(items, body.name)
        if not items:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Nothing serving matches {body.name!r}",
            )
    if not items:
        return {"items": [], "results": [], "ok": True}
    if len(items) > 1 and body.name is None and not body.all_:
        labels = "; ".join(item.label for item in items)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Several things are serving; pass a name or all: {labels}",
        )
    shown = [
        {
            "kind": item.kind,
            "label": item.label,
            "ref": item.ref,
            "cards": lanes.format_gpus(item.gpus),
        }
        for item in items
    ]
    if body.dry_run:
        return {"items": shown, "results": [], "ok": True}
    store = jobs.store
    record = None
    if store is not None:
        # §2B: persist the stop intent before the first side effect.
        record, _ = store.begin(
            kind="stop",
            target=", ".join(item.label for item in items),
            interface="api",
            purpose="stop serving",
            request={"items": [{"kind": i.kind, "ref": i.ref} for i in items]},
        )
    try:
        with jobs.stopping():
            outcome = stop_serving.stop_items(items, stop_serving.real_deps(settings))
    except JobBusy as busy:
        if record is not None:
            store.finish(record.id, OperationPhase.FAILED, error="a swap job was running")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Swap job {busy.job.job_id} is running",
        ) from busy
    except LockBusy as exc:
        # Another process (CLI swap, TUI action) holds the GPU operation
        # lock; the stop cannot run concurrently with it.
        if record is not None:
            store.finish(record.id, OperationPhase.FAILED, error=str(exc))
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    results = [{"label": i.label, "ref": i.ref, "stopped": ok} for i, ok in outcome]
    ok = all(r["stopped"] for r in results)
    if record is not None:
        if ok:
            store.finish(record.id, OperationPhase.COMPLETED)
        else:
            # Half-stopped: the world differs from where we started.
            store.finish(
                record.id,
                OperationPhase.NEEDS_ATTENTION,
                error="some items did not stop",
                recovery={
                    "not_stopped": [r["label"] for r in results if not r["stopped"]],
                    "guidance": "check `llmctl status`; something may still hold the GPUs.",
                },
            )
    return {"items": shown, "results": results, "ok": ok}
