# 0029. Fallback when the line edit cannot express the change

- Status: proposed (recorded automatically by devlab run `20261104-043018-one-preset-store-ee2e` for project `llmctl-p2`)
- Date: 2026-11-04

## Decision

If the edited text does not reload to the intended mapping (unusual layout, flow-style blocks), rewrite the whole file as a safe_dump of only the set keys, keeping the leading comment block; always validate and compare before the atomic write.

## Rationale

Correctness of values comes first (spec C7); comments are kept wherever the line edit works, and an invalid write is never produced.

## Alternatives considered

- Raise an error like save_to_preset does (would break TUI saves)
