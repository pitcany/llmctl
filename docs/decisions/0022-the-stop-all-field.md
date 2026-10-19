# 0022. The stop 'all' field

- Status: proposed (recorded automatically by devlab run `20261019-142021-swap-api-cdc6` for project `llmctl`)
- Date: 2026-10-19

## Decision

The pydantic model accepts the JSON key 'all' through a field alias, for example all_: bool = Field(False, alias='all'), with populate_by_name.

## Rationale

This keeps the documented body shape without shadowing the builtin in Python code.

## Alternatives considered

- Name the field all and accept the ruff A003 suppression
