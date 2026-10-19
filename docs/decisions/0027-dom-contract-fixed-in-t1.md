# 0027. DOM contract fixed in T1

- Status: proposed (recorded automatically by devlab run `20261019-161309-swap-ui-c841` for project `llmctl`)
- Date: 2026-10-19

## Decision

T1 writes the final index.html markup with every id (rows table, serving box, confirm panel, job panel, error area, token form). Later tasks only style it or wire it up.

## Rationale

This gives the CSS and JS tasks a stable contract and keeps the inline-content checks on index.html meaningful from the first task.

## Alternatives considered

- Let JS build the whole page (index.html becomes almost empty, but harder to style and test)
