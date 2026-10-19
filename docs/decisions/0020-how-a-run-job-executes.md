# 0020. How a run job executes

- Status: proposed (recorded automatically by devlab run `20261019-142021-swap-api-cdc6` for project `llmctl`)
- Date: 2026-10-19

## Decision

The job runs `python -m llmctl.main run KEY --yes` in a subprocess, through swap_jobs.run_argv on a daemon thread. Output is captured line by line with NO_COLOR=1 and COLUMNS=200.

## Rationale

The spec requires it. Every CLI swap path, rollback included, is reused unchanged, and nothing that is Rich-console bound runs in the API process.

## Alternatives considered

- Call cli_swap.run_lane_swap and the other paths in-process with a recording console
