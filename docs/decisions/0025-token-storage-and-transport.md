# 0025. Token storage and transport

- Status: proposed (recorded automatically by devlab run `20261019-161309-swap-ui-c841` for project `llmctl`)
- Date: 2026-10-19

## Decision

Keep the bearer token in sessionStorage under one key and send it as `Authorization: Bearer` on every /swap fetch. The token form is revealed on any 401.

## Rationale

This is the spec's choice. sessionStorage limits the token's lifetime to the tab, and same-origin relative fetches mean CORS is not involved.

## Alternatives considered

- localStorage (forbidden)
- Cookies (would need CSRF handling)
