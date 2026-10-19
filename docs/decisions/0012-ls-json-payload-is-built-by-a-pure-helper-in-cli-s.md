# 0012. ls --json payload is built by a pure helper in cli_swap.py

- Status: proposed (recorded automatically by devlab run `20261018-152154-ux-truth-trio-1af1` for project `llmctl`)
- Date: 2026-10-18

## Decision

cli_swap.rows_json(state) builds the list; ls_cmd just emits it via _emit_json.

## Rationale

Keeps the table and the JSON reading the same state_of/cards_label/last_load_s values, and is testable without the CLI.

## Alternatives considered

- Build dicts inline in ls_cmd
