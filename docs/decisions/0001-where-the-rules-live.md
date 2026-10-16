# 0001. Where the rules live

- Status: proposed (recorded automatically by devlab run `20261015-210947-health-reality-2-d158` for project `llmctl`)
- Date: 2026-10-16

## Decision

Add a post-processing step in HealthService.get_health (e.g. _apply_engine_rules) that runs after adapter results, sessions and loaded models are merged, and that touches only runtimes whose value is in ENGINE_NAMES.

## Rationale

Rules F and A-dedup need the merged sessions list, and doing the overlay in one place keeps non-engine runtimes byte-identical.

## Alternatives considered

- Change the adapters' health_check (not allowed by the constraints)
- Change the rules inside _runtime_health before sessions are known
