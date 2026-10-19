# 0024. Header placement

- Status: proposed (recorded automatically by devlab run `20261019-161309-swap-ui-c841` for project `llmctl`)
- Date: 2026-10-19

## Decision

Send nosniff on all three files and the spec's CSP on /ui (adding it to the JS and CSS responses is harmless). Mark the routes include_in_schema=False.

## Rationale

This meets the spec exactly and keeps the OpenAPI document focused on the data API.

## Alternatives considered

- A global security-headers middleware (would change headers on existing JSON routes)
