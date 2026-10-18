# 0008. Lane tagging happens in _apply_engine_rules for every session that matches a lane

- Status: proposed (recorded automatically by devlab run `20261018-152154-ux-truth-trio-1af1` for project `llmctl`)
- Date: 2026-10-18

## Decision

Any session in an engine runtime whose served_name equals the served_model of a lane of that runtime gets 'lane': <first such lane's name>. The managed flag is left alone.

## Rationale

The loaded gemma4 entry comes from _loaded_by_runtime before the lane rules run. Tagging there reports it once with its lane and keeps serving_count/unmanaged_count unchanged, as the spec requires.

## Alternatives considered

- Tag only the sessions rule A adds
- Mark lane-owned models managed: True (forbidden: changes unmanaged_count semantics)
