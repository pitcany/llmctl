# 0011. Avoid merge conflicts between parallel tasks

- Status: proposed (recorded automatically by devlab run `20261018-152154-ux-truth-trio-1af1` for project `llmctl`)
- Date: 2026-10-18

## Decision

Each task has its own test file (test_unit_rows / test_health_reality / test_cli_swap / new test_cli_ls). The suggestion helper goes in services/lanes.py instead of cli_swap.py. T2/T3/T4 each change only their own command function in cli.py.

## Rationale

Lets T1-T4 run in parallel worktrees with merges that do not conflict.

## Alternatives considered

- Run the cli.py tasks one after another
