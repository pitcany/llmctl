# 0016. Session endpoints in ls --json

- Status: proposed (recorded automatically by devlab run `20261018-214800-endpoints-ids-stop-9ef5` for project `llmctl`)
- Date: 2026-10-19

## Decision

Session rows report endpoint null (their upstream is "").

## Rationale

The spec allows null. Looking up the running session for every ls row is extra I/O for little gain; `run` on a serving session (T5) does show its endpoint.

## Alternatives considered

- Join gpu_sessions endpoint_url into ls rows
