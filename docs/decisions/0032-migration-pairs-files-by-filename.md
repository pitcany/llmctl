# 0032. Migration pairs files by filename

- Status: proposed (recorded automatically by devlab run `20261104-043018-one-preset-store-ee2e` for project `llmctl-p2`)
- Date: 2026-11-04

## Decision

For each legacy file, the canonical counterpart is <canonical_dir>/<same filename>. The reported alias is the validated alias, or the filename stem for invalid files.

## Rationale

migrate_legacy_presets and save_preset both pair by <alias>.yaml; filename pairing makes the relative-symlink layout predictable.

## Alternatives considered

- Pair by alias key across arbitrary filenames
