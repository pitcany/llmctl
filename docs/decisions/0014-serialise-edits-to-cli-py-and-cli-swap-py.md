# 0014. Serialise edits to cli.py and cli_swap.py

- Status: proposed (recorded automatically by devlab run `20261018-214800-endpoints-ids-stop-9ef5` for project `llmctl`)
- Date: 2026-10-19

## Decision

Chain the tasks that touch these files: aliases (T2) → ls (T4) → stop (T7) for cli.py, and ls (T4) → run (T5) for cli_swap.py.

## Rationale

Parallel worktrees must not edit the same file. The service-level work (T1, T3, T6) still runs in parallel.

## Alternatives considered

- Run them all in parallel and resolve merges
