"""Benchmark API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from llmctl.api.deps import get_db_session
from llmctl.schemas import BenchmarkResult, BenchmarkRunRequest
from llmctl.services.benchmarks import BenchmarkService

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=list[BenchmarkResult])
def list_benchmarks(db: Session = Depends(get_db_session)) -> list[BenchmarkResult]:
    """List benchmark results."""
    return BenchmarkService(db).list_results()


@router.post("/run", response_model=BenchmarkResult, status_code=status.HTTP_201_CREATED)
def run_benchmark(
    payload: BenchmarkRunRequest, db: Session = Depends(get_db_session)
) -> BenchmarkResult:
    """Run a benchmark (real streaming execution with a mock fallback)."""
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
