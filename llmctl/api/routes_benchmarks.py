"""Benchmark API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from llmctl.api.deps import get_db_session
from llmctl.db import BenchmarkKind
from llmctl.schemas import BenchmarkResult, BenchmarkRunRequest
from llmctl.services.benchmarks import BenchmarkService

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[BenchmarkResult])
def list_benchmarks(
    model_id: str | None = Query(default=None, description="Filter by model ID."),
    session_id: str | None = Query(default=None, description="Filter by session ID."),
    kind: BenchmarkKind | None = Query(default=None, description="Filter by kind."),
    limit: int | None = Query(default=None, ge=1, description="Max rows (newest first)."),
    db: Session = Depends(get_db_session),
) -> list[BenchmarkResult]:
    """List benchmark results, optionally filtered by model/session/kind."""
    return BenchmarkService(db).list_results(
        model_id=model_id, session_id=session_id, kind=kind, limit=limit
    )


@router.get("/{benchmark_id}", response_model=BenchmarkResult)
def get_benchmark(
    benchmark_id: str, db: Session = Depends(get_db_session)
) -> BenchmarkResult:
    """Fetch a single benchmark result by ID."""
    result = BenchmarkService(db).get_result(benchmark_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark not found")
    return result


@router.post("/run", response_model=BenchmarkResult, status_code=status.HTTP_201_CREATED)
def run_benchmark(
    payload: BenchmarkRunRequest, db: Session = Depends(get_db_session)
) -> BenchmarkResult:
    """Run a benchmark (real streaming execution; failures persist as success=False)."""
    return BenchmarkService(db).run(payload)


@router.post(
    "/sweep", response_model=list[BenchmarkResult], status_code=status.HTTP_201_CREATED
)
def sweep_benchmark(
    payload: BenchmarkRunRequest, db: Session = Depends(get_db_session)
) -> list[BenchmarkResult]:
    """Run a concurrency sweep, persisting one result per level."""
    return BenchmarkService(db).run_sweep(payload)


@router.post("/{benchmark_id}/rerun", response_model=BenchmarkResult)
def rerun_benchmark(
    benchmark_id: str, db: Session = Depends(get_db_session)
) -> BenchmarkResult:
    """Re-run a stored benchmark, reusing its prompts and parameters."""
    result = BenchmarkService(db).rerun(benchmark_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Benchmark not found")
    return result
