# 0006. Reuse doctor helpers

- Status: proposed (recorded automatically by devlab run `20261015-210947-health-reality-2-d158` for project `llmctl`)
- Date: 2026-10-16

## Decision

Import _model_users and _name_models from llmctl.services.doctor rather than copying them.

## Rationale

The spec says health must reuse doctor's rules, not re-derive them.

## Alternatives considered

- Duplicate the helpers in health.py
