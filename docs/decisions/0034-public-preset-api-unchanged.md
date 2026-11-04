# 0034. Public preset API unchanged

- Status: proposed (recorded automatically by devlab run `20261104-043018-one-preset-store-ee2e` for project `llmctl-p2`)
- Date: 2026-11-04

## Decision

New helpers are reached through their modules (llmctl.presets.sources, llmctl.presets.migrate); llmctl.presets.__all__ is not extended.

## Rationale

test_public_api_exports_expected_symbols asserts the exact __all__ list, and the spec asks for no public API expansion.

## Alternatives considered

- Export new functions from llmctl.presets
