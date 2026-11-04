# 0030. save_preset target choice with symlinks

- Status: proposed (recorded automatically by devlab run `20261104-043018-one-preset-store-ee2e` for project `llmctl-p2`)
- Date: 2026-11-04

## Decision

Target order is unchanged: canonical real file > legacy file > new canonical file. A canonical symlink to the legacy file is not a canonical real file, so the legacy file gets the write; atomic_write_text follows symlinks, so links stay links. Nothing is unlinked, in the config_dir branch too.

## Rationale

Matches spec C9 exactly and keeps load precedence results consistent, because the legacy file wins anyway.

## Alternatives considered

- Keep replacing canonical symlinks with real files (deletes the legacy file, forbidden)
