# rust/ — the llmctl successor (Rust)

Implementation of [docs/RUST-SUCCESSOR-PLAN.md](../docs/RUST-SUCCESSOR-PLAN.md).
Status: **first implementation batch complete** (see the plan's "First
implementation batch" section). This is `llmctl-next`, a dev-stage control
plane that runs entirely against a fake engine — nothing here touches the
installed inference stack or the Python llmctl data.

## Layout

| Crate | Role |
| --- | --- |
| `crates/llmctl-core` | Typed identities, desired intent, observations, plans, the pure planner, domain errors. No syscalls, no DB. |
| `crates/llmctl-client` | Wire DTOs (`wire`) shared by CLI and daemon, and the Unix-socket HTTP client. |
| `crates/llmctl-daemon` | Host agent: SQLite operation journal (bounded worker thread), versioned local API over the Unix socket, operation admission/execution/reconciliation, the `Engine` port and the file-backed fake engine. |
| `crates/llmctl-cli` | The `llmctl-next` binary. Thin composition root over client + daemon. |

Deviation from the plan's 3-crate table: the CLI binary lives in a fourth,
tiny crate (`llmctl-cli`) because the plan puts wire DTOs in
`llmctl-client` and the daemon needs those DTOs — a binary in
`llmctl-client` that also embeds the daemon would create a dependency
cycle. Same responsibilities, one extra crate boundary; revisit if the
layout hurts.

## Try it

```bash
cd rust
cargo test --workspace              # unit + e2e (fake engine, temp dirs only)
cargo run -p llmctl-cli -- daemon   # foreground, ~/.local/state/llmctl-next-dev
cargo run -p llmctl-cli -- status
cargo run -p llmctl-cli -- run coder --dry-run
cargo run -p llmctl-cli -- run coder --wait
cargo run -p llmctl-cli -- ops
LLMCTL_NEXT_DATA_DIR=/tmp/next-dev cargo run -p llmctl-cli -- daemon   # isolated dir
```

The dev data directory (default `~/.local/state/llmctl-next-dev`, or
`$LLMCTL_NEXT_DATA_DIR`) holds `inventory.json` (bootstrapped with three
fake workloads: `coder` gpu0, `helper` gpu1, `big` gpu0+1), `world.json`
(the fake engine's surviving external state), `faults.json` (test hooks),
`ops.sqlite3` (the journal) and `control.sock` (0600).

## What is proven by tests (`crates/llmctl-cli/tests/e2e.rs`)

The batch's exit criteria, each against a spawned daemon + temp dir:

1. **A second client sees the same operation** — same idempotency key
   replays the same op; the same key with a different request is refused.
2. **An interrupted client leaves it observable** — an acknowledgement
   lost to a client timeout does not cancel the op; a second client finds,
   watches and retries it idempotently.
3. **Daemon restart reconciles honestly** — a SIGKILL mid-effect never
   yields a silent success: interrupted with observations attached, or
   needs_attention blocking mutations until `op ack`; the provably-safe
   case (only observation steps remained) auto-resolves.
4. **No test touches the real stack** — fake engine + temp dirs only.

Plus the mutation contract: protocol-version rejection before any
mutation, stale destructive footprint → 409 + revised plan, overlap
rejection, failed run → recorded restoration (separate from the failure),
failed restoration → needs_attention + blocking + ack, phase-boundary
cancellation, manual-stop intent persistence, and CLI exit codes
(0 ok · 1 failed · 3 no daemon · 4 conflict · 5 unknown · 6 invalid).

## Fake-engine fault hooks (`faults.json`)

```json
{
  "fail_effects": [{"unit": "user/llmctl-next-fake-coder.service",
                    "kind": "start", "mode": "before"|"after"}],
  "effect_delay_ms": 0,
  "readiness_delay_ms": 0,
  "serve_wrong_model": false,
  "admit_delay_ms": 0
}
```

`mode: "before"` fails without the effect; `"after"` applies the effect
then reports failure — the crash-between-effect-and-result window
reconciliation must resolve. Rules are consumed in order when unit and
kind match.

## Measured (release build, 2026-11-15, this machine)

- `--version` p95 **4 ms** over 20 runs (plan's provisional budget: <100 ms).
- Fast-fail path (no daemon → exit 3) p95 5 ms.
- Release binary: 8.3 MB.

## Not yet here (per the plan's milestones)

Real engine adapters (M2+), SSH transport and Mac agents (M3), installers
and version pinning (M4), TUI and config/preset parity (M5), migration of
Python data (M5/M6). The YAML reader choice is deliberately still open —
pick a maintained crate against current docs before config parity lands.
