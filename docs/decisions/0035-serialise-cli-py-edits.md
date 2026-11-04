# 0035. Serialise cli.py edits

- Status: proposed (recorded automatically by devlab run `20261104-043018-one-preset-store-ee2e` for project `llmctl-p2`)
- Date: 2026-11-04

## Decision

T2 owns the conversion of `presets` into a group and the settings --json flag. T3 depends on T2 and only adds the migrate subcommand.

## Rationale

Avoids merge conflicts in cli.py between parallel worktrees (ADR 0014).

## Alternatives considered

- Put migrate CLI in a separate cli_presets.py module created by T2
