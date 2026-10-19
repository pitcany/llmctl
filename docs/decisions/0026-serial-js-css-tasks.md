# 0026. Serial JS/CSS tasks

- Status: proposed (recorded automatically by devlab run `20261019-161309-swap-ui-c841` for project `llmctl`)
- Date: 2026-10-19

## Decision

T2 (CSS), T3 (JS core) and T4 (JS actions) run in sequence. T5 (docs) runs in parallel after T1.

## Rationale

All three add tests to the one mandated test file (tests/test_api_ui.py), and T3 and T4 both edit app.js. Running them in parallel would cause merge conflicts.

## Alternatives considered

- Put CSS checks in a separate test file (conflicts with the constraint that tests go in tests/test_api_ui.py)
