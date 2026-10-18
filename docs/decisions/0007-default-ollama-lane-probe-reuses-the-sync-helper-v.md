# 0007. Default Ollama lane probe reuses the sync helper via asyncio.to_thread

- Status: proposed (recorded automatically by devlab run `20261018-152154-ux-truth-trio-1af1` for project `llmctl`)
- Date: 2026-10-18

## Decision

For runtime == 'ollama', _default_lane_probe awaits asyncio.to_thread(probe_ollama_loaded, lane.upstream, LANE_PROBE_TIMEOUT_S) and checks lane.served_model membership; None (unreachable) means not answering.

## Rationale

The spec requires reusing probe_ollama_loaded. to_thread keeps the concurrent gather in _runtime_health non-blocking, and tests can monkeypatch one function.

## Alternatives considered

- Reimplement /api/ps parsing with httpx.AsyncClient
- Probe /api/ps once per upstream and share it across lanes
