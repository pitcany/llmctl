# 0017. Cards for a stop preview

- Status: proposed (recorded automatically by devlab run `20261018-214800-endpoints-ids-stop-9ef5` for project `llmctl`)
- Date: 2026-10-19

## Decision

Each Serving item stores a Gpus footprint and is formatted with format_gpus.

## Rationale

Reuses the ls Cards notation (all / 0 / 1 / 0+1). A session with unknown gpu_ids reads 'all' and fails closed, as parse_gpus does.

## Alternatives considered

- Show the session's planned placement text
