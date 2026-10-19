# 0010. De-duplicate the /api/ps names in unit_rows

- Status: proposed (recorded automatically by devlab run `20261018-152154-ux-truth-trio-1af1` for project `llmctl`)
- Date: 2026-10-18

## Decision

served keeps order but has no duplicates, because probe_ollama_loaded returns both `name` and `model` for each entry.

## Rationale

Otherwise `status` would print every loaded model twice.

## Alternatives considered

- Change probe_ollama_loaded itself (also used by serving_keys; not needed)
