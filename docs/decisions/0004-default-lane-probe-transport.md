# 0004. Default lane probe transport

- Status: proposed (recorded automatically by devlab run `20261015-210947-health-reality-2-d158` for project `llmctl`)
- Date: 2026-10-16

## Decision

Use the HTTP client the adapters already depend on (httpx.AsyncClient) with timeout=2.0 against f"{lane.upstream}/v1/models", returning True only on HTTP 200.

## Rationale

Adds no new dependency. The URL comes only from configured lanes (localhost upstreams), and nothing from the response body is used.

## Alternatives considered

- urllib in a thread
