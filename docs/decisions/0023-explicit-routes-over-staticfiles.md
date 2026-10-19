# 0023. Explicit routes over StaticFiles

- Status: proposed (recorded automatically by devlab run `20261019-161309-swap-ui-c841` for project `llmctl`)
- Date: 2026-10-19

## Decision

A dedicated router with three fixed routes, each mapping a constant filename to a media type and reading bytes through importlib.resources.files('llmctl.api') / 'static'.

## Rationale

The spec requires no StaticFiles mount. With fixed names, no request input reaches the filesystem, so traversal and directory listing are impossible by construction. importlib.resources works from the installed wheel.

## Alternatives considered

- StaticFiles mount (forbidden by spec)
- A catch-all /ui/{name} route with a whitelist (more surface for no gain)
