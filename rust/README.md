# rust/ — the llmctl successor (Rust)

Implementation of [docs/RUST-SUCCESSOR-PLAN.md](../docs/RUST-SUCCESSOR-PLAN.md).
Status: **first implementation batch complete** (see the plan's "First
implementation batch" section). This is `llmctl-next`, a dev-stage control
plane that runs against a fake engine by default — nothing here touches the
installed inference stack or the Python llmctl data. M2 has started: a
`systemd` adapter drives user units, but only units an inventory workload
explicitly enrolls with `engine: "systemd"` (see below).

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

## systemd adapter (M2 spike, `crates/llmctl-daemon/src/systemd.rs`)

Each inventory workload names its adapter (`engine: "fake" | "systemd"`);
`RoutedEngine` dispatches by unit, so an adapter never sees a unit it was
not enrolled for. The systemd adapter adopts an *existing* unit — the unit
file stays the configuration authority:

- process state from `systemctl --user show` (manager unreachable →
  `Unknown`, never `Stopped`; `activating/auto-restart` → `Stopped`);
- readiness + served-model identity from `GET 127.0.0.1:<port>/v1/models`;
- `start`/`stop` on user units only (system scope → `Unsupported`); effect
  identity carries systemd's `InvocationID`; a timed-out start/stop is
  `Unconfirmed` and handled like the after-effect fault.

Opt-in tests against disposable `llmctl-next-spike-*.service` units
(written, enrolled and removed per test):

```bash
cargo test -p llmctl-cli --test systemd_spike -- --ignored --test-threads=1
```

They prove run → Ready → stop, restart (new main PID, re-verified; a
stopped unit is refused rather than started), early exit fails as "exited
before readiness" (not a timeout), and a wrong served model fails
verification.

### Restart (`llmctl-next restart NAME [--dry-run] [--wait]`)

The run plan of an observed-running workload — stop, start, await
readiness, re-verify the served model — journaled as kind `restart`. A
stopped or `Unknown` workload is refused with `invalid_state` (exit 4):
restarting something not known to run would be a silent `run`. Unlike
run/stop, the default idempotency key is unique per invocation (a restart
is an event); pass `--key` with an earlier key to retry that attempt.

### Adoption (`llmctl-next adopt`, `crates/llmctl-daemon/src/{adopt,unitdef}.rs`)

```bash
llmctl-next adopt foo.service --name coder --model org/model --port 8003 --gpus 0,1
# → preview: checks, generated service definition, digest, exact confirm command
llmctl-next adopt foo.service … --confirm sha256:… [--allow-unverified]
```

- **Import** reads the effective definition through systemd's D-Bus API
  (`busctl --json`): exact argv arrays (`systemctl show` loses argument
  boundaries), env, `EnvironmentFile=` paths + content digests (contents
  never stored), type, restart, stop timeout, kill mode, unit-file provenance.
- **Equivalence** is proven by systemd itself: the generated definition is
  loaded under a scratch `llmctl-next-verify-*.service` (never started,
  always removed) and re-imported; any differing field fails the check.
- **Checks**: enrollment (unit/name/port free, GPUs exist), unit loaded,
  no credential-like inline `Environment=` names (move them to an
  `EnvironmentFile=`; values are never echoed), definition round trip,
  served identity (`/v1/models`; a stopped unit is *unverified*, which
  transfer accepts only with `--allow-unverified`).
- **Transfer** re-previews under the admission lock (refused while another
  operation is active or unacknowledged), requires the confirmed digest to
  still match (`stale_adoption` + fresh preview otherwise), then enrolls the
  workload as `ownership: managed` with the digest pinned in
  `inventory.json` (revision bumped). The unit keeps running; its file is
  not touched.
- **Drift guard**: `run`/`restart` of a managed workload re-imports the unit
  and refuses with `definition_drift` (exit 4, differing fields listed)
  before anything is journaled; `stop` stays allowed.

The opt-in `systemd_spike` tests cover adopt → stop → run → daemon
restart, stale confirm, drift blocking run without touching the unit, the
unverified gate, and inline secrets (refused, never echoed).

## Measured (release build, 2026-11-15, this machine)

- `--version` p95 **4 ms** over 20 runs (plan's provisional budget: <100 ms).
- Fast-fail path (no daemon → exit 3) p95 5 ms.
- Release binary: 8.3 MB.

## Not yet here (per the plan's milestones)

The rest of M2 (logs, writing/owning the generated unit file,
version pin + staged upgrade, real-model pilot), SSH transport and Mac agents (M3), installers
and version pinning (M4), TUI and config/preset parity (M5), migration of
Python data (M5/M6). The YAML reader choice is deliberately still open —
pick a maintained crate against current docs before config parity lands.
