# 0015. Session status values in aliases --json

- Status: proposed (recorded automatically by devlab run `20261018-214800-endpoints-ids-stop-9ef5` for project `llmctl`)
- Date: 2026-10-19

## Decision

'unbound' when target is None, 'healthy' when healthy, otherwise 'bound'.

## Rationale

These are the spec's three states, mapped from the existing table logic.

## Alternatives considered

- (none recorded)
