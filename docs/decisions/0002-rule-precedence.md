# 0002. Rule precedence

- Status: proposed (recorded automatically by devlab run `20261015-210947-health-reality-2-d158` for project `llmctl`)
- Date: 2026-10-16

## Decision

For an engine runtime: A (an answering lane) > F/B (live sessions keep the adapter state but drop 'not running'; with no sessions and a lane, B applies) > D/C (source 'missing' and no sessions) > E (wording fix on whatever message remains when the source is 'missing'). An engine with no EngineStatus row keeps the adapter result.

## Rationale

Matches the spec's ordering ('no lane serves it', 'nothing else is serving it') and never reports a runtime with a live session as not_used.

## Alternatives considered

- Let C apply even when unmanaged sessions exist
