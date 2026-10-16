# 0005. Keeping get_health fast with engine_statuses

- Status: proposed (recorded automatically by devlab run `20261015-210947-health-reality-2-d158` for project `llmctl`)
- Date: 2026-10-16

## Decision

The default engines value calls engine_statuses(settings, lanes=lanes, run=<stub that returns a failed CompletedProcess>) so that health never pays for version probes (up to 5 s each), and computes model_users lazily, only when some engine row is 'missing'.

## Rationale

Health doesn't display versions. The source/binary detection, including _really_installed, is unchanged, so the rules are still doctor's.

## Alternatives considered

- Call engine_statuses(settings, lanes=lanes) literally and accept the version-probe latency
