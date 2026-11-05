# 0028. Comment-preserving writer lives in a new presets module, not in run_overrides

- Status: proposed (recorded automatically by devlab run `20261104-043018-one-preset-store-ee2e` for project `llmctl-p2`)
- Date: 2026-11-04

## Decision

Implement the line editor for save_preset in llmctl/presets/yaml_edit.py and leave services/run_overrides.save_to_preset untouched.

## Rationale

The spec forbids changing save_to_preset's behaviour; save_preset must also handle removed keys and nested blocks, which the run --save editor does not. A separate module avoids regressions and keeps the presets package free of service imports.

## Alternatives considered

- Generalise save_to_preset and route both through it
- Use ruamel.yaml round-tripping (new dependency, forbidden)
