# 0033. Migration uses its own no-follow atomic byte writes

- Status: proposed (recorded automatically by devlab run `20261104-043018-one-preset-store-ee2e` for project `llmctl-p2`)
- Date: 2026-11-04

## Decision

Migration writes bytes to a temp file in the destination dir, sets the mode and uses os.replace on the link path itself. It also creates the new legacy symlink as a temp symlink followed by os.replace. It does not use atomic_write_text for these steps.

## Rationale

atomic_write_text follows symlinks: writing the canonical path while it is a symlink to the legacy file would overwrite the legacy file and could create a cycle.

## Alternatives considered

- Unlink then write (not atomic)
