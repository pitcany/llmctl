# 0018. Where the routes get Settings and the job registry

- Status: proposed (recorded automatically by devlab run `20261019-142021-swap-api-cdc6` for project `llmctl`)
- Date: 2026-10-19

## Decision

Add get_settings() and get_swap_jobs() placeholders in llmctl/api/deps.py. create_app overrides them with effective_settings and with one SwapJobs instance per app, the same pattern get_db_session already uses.

## Rationale

Router modules stay independent of the app factory and avoid circular imports. Each create_app, so each test, gets a fresh registry with no global state to reset. Tests can pass explicit settings.

## Alternatives considered

- A module-level global registry plus load_settings() in each route
- A router factory build_router(settings, jobs)
