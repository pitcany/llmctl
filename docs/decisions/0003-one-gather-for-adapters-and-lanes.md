# 0003. One gather for adapters and lanes

- Status: proposed (recorded automatically by devlab run `20261015-210947-health-reality-2-d158` for project `llmctl`)
- Date: 2026-10-16

## Decision

Build one coroutine list (adapter health_checks + one lane_probe per lane) and await asyncio.gather(..., return_exceptions=True) inside the existing run_sync call. A lane exception counts as False.

## Rationale

The spec requires that lanes are probed concurrently with the adapters and that one slow lane doesn't serialize the others. Reusing run_sync keeps the existing event-loop handling.

## Alternatives considered

- A separate run_sync for lanes (adds latency)
- A thread pool
