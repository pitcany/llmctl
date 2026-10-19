# 0021. Atomic one-job-at-a-time check

- Status: proposed (recorded automatically by devlab run `20261019-142021-swap-api-cdc6` for project `llmctl`)
- Date: 2026-10-19

## Decision

SwapJobs.start checks for a running job and inserts the new one under a single lock acquisition, and raises JobBusy(job) when busy. The route maps JobBusy to 409.

## Rationale

FastAPI runs sync routes in a thread pool, so a separate check followed by an insert could race and start two swaps at once.

## Alternatives considered

- The route checks running() first and then calls start()
