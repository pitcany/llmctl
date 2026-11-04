# 0031. Alias and winner semantics for the read-only scan

- Status: proposed (recorded automatically by devlab run `20261104-043018-one-preset-store-ee2e` for project `llmctl-p2`)
- Date: 2026-11-04

## Decision

The alias comes from the file's validated `alias` key, as the loaders do. The winner is the last valid file in the sequence canonical(sorted) + legacy(sorted). Invalid files appear only in `problems`.

## Rationale

Reproduces load_all_records precedence exactly without calling it, so `effective` always matches what llmctl actually loads.

## Alternatives considered

- Key by filename stem (could disagree with the loaders)
