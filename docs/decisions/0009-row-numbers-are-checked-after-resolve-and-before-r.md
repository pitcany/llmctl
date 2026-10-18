# 0009. Row numbers are checked after resolve() and before registry routing

- Status: proposed (recorded automatically by devlab run `20261018-152154-ux-truth-trio-1af1` for project `llmctl`)
- Date: 2026-10-18

## Decision

An exact key/model/alias match wins, then a decimal number in range picks that row, and an out-of-range number exits 2 at once. Only non-numeric unknown names reach _swap_registry_model and the vLLM path.

## Rationale

Meets rules 6/7: a bad number must never start or stop anything, and treating it as a preset name would only end in a confusing error.

## Alternatives considered

- Try registry routing before numbers (a registered model named '12' would win, but numbers would behave unpredictably)
