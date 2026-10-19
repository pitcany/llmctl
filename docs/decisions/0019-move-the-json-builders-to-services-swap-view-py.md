# 0019. Move the JSON builders to services/swap_view.py

- Status: proposed (recorded automatically by devlab run `20261019-142021-swap-api-cdc6` for project `llmctl`)
- Date: 2026-10-19

## Decision

Move rows_json, gateway_aliases and _aliases_for into llmctl/services/swap_view.py, and re-import them in cli_swap. Add gateway_url and endpoint_for there.

## Rationale

Without the move the API layer would have to import the CLI module, which pulls in typer, rich and prompts. One builder for both callers guarantees that GET /swap rows are identical to ls --json.

## Alternatives considered

- Import cli_swap from the API and leave the builders where they are (ADR 0012)
