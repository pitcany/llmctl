# 0013. Where the shared table helper lives

- Status: proposed (recorded automatically by devlab run `20261018-214800-endpoints-ids-stop-9ef5` for project `llmctl`)
- Date: 2026-10-19

## Decision

A new module llmctl/cli_tables.py. `ls` stays on its own render_table this run.

## Rationale

A new file lets T1 run without touching cli.py or cli_swap.py. Leaving `ls` alone means its output cannot change.

## Alternatives considered

- Put it in cli.py next to _id_column
- Move render_table onto it now
