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
export LLMCTL_NEXT_DATA_DIR=/tmp/next-dev   # always: the default is the live agent's dir
cargo run -p llmctl-cli -- daemon   # foreground, fake engine
cargo run -p llmctl-cli -- status
cargo run -p llmctl-cli -- run coder --dry-run
cargo run -p llmctl-cli -- run coder --wait
cargo run -p llmctl-cli -- ops
```

The data directory (`$LLMCTL_NEXT_DATA_DIR`, else `~/.local/state/llmctl-next`,
which is the live agent's dir on an enrolled host: never develop against the
default) holds `inventory.json` (bootstrapped with three
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

They prove run → Ready → stop, logs (output, run markers, follow across a
restart, unknown cursor → 400), restart (new main PID, re-verified; a
stopped unit is refused rather than started), early exit fails as "exited
before readiness" (not a timeout), and a wrong served model fails
verification.

### Restart (`llmctl-next restart NAME [--dry-run] [--wait]`)

The run plan of an observed-running workload — stop, start, await
readiness, re-verify the served model — journaled as kind `restart`. A
stopped or `Unknown` workload is refused with `invalid_state` (exit 4):
restarting something not known to run would be a silent `run`. As for
every verb, the default idempotency key is unique per invocation; pass
`--key` with an earlier key to retry that attempt. (run/stop once keyed on
the inventory revision alone, so a `run` after a crash replayed the first
run's success without starting anything — found in the M2 pilot.)

### Logs (`llmctl-next logs NAME [-n 100] [-f] [--json]`)

Read from the journal (`journalctl --output=json`) through the daemon
(`POST /v1/logs`, read-only, no admission lock, so logs stay readable
while an operation runs). The workload's output and the manager's lines
about it (`systemd: Started …`, `Main process exited …`) are interleaved;
`-- run <id> --` marks each new start (the invocation id). `-f` pages
forward from a cursor without gaps; `--json` prints one entry per line.
Non-UTF-8 output is decoded lossily, messages are capped at 16 KiB, a page
is at most 1000 entries. Engines without a log source (the fake engine)
answer `unsupported` (exit 4), never an empty log.

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

## The rest of M2 (docs/RUST-M2-PLAN.md, items A–D, 2026-11-16)

### Foreign GPU occupancy guard (A)

Before a plan starts a unit, the affected cards are asked who is on them:
`nvidia-smi` compute apps + the index→UUID map, pids mapped to units
through `/proc/<pid>/cgroup`. A process that is neither the target nor an
enrolled workload the plan already stops refuses the operation with
`gpu_occupied` (409, exit 4) before anything is journaled; a probe that
cannot answer refuses too (`gpu_probe_failed`) — unknown is never "free".
The probe is part of each engine's world: the fake engine reads a
`gpus.json` hook file, systemd probes nvidia-smi, and engines with nothing
enrolled are not asked. `status` shows per card enrolled vs foreign users.

### Bounded recovery (B)

A single background watcher observes enrolled workloads (default 15 s;
`LLMCTL_NEXT_WATCH_INTERVAL_MS` for tests). Policy per workload, off by
default (`recovery: {max_attempts, window_secs, min_interval_secs}` in the
inventory; adopted workloads stay off until the owner turns it on):

- systemd retries first: `activating/auto-restart` is only reported
  (`status` shows `manager_retrying`); llmctl acts when the manager gave
  up (`failed`, start-limit-hit — a start then does `reset-failed` first)
  or the unit is down and not retrying.
- A recovery is an ordinary admitted `run` (key `recover-<workload>-<n>`):
  admission, the drift guard, the GPU guard, the journal and rollback come
  for free. The budget is SQLite-persisted (`recoveries` table), so a
  daemon restart does not reset it; attempts are spaced by
  `min_interval_secs` (default 30).
- Never recovered: manual stops (durable intent), unmanaged workloads,
  unknown manager state, drifted definitions, foreign GPU occupants,
  active/unacknowledged operations, and the daemon's first interval.
- Budget exhausted → one unacknowledged `needs_attention` marker per
  episode (`recover-stalled-…`), blocking mutations until `op ack`.

The stop-timeout row of the exit checklist: `LLMCTL_NEXT_SYSTEMD_EFFECT_TIMEOUT_MS`
makes the effect timeout configurable; a stop that cannot confirm is
`Unconfirmed` (never a fake success) and the op fails naming it.

### Owning the unit file (C: `own` / `release`)

Adoption pins the definition and leaves the unit file alone; `own` goes
further — llmctl writes the unit file itself:

- Preconditions: no drop-ins (listed when refused), no drift, no active
  operation. The rendered unit (header naming llmctl-next and the digest)
  replaces the file **or the symlink itself** atomically — never writing
  through a link, never touching the repo copy it pointed at.
- Prove-or-restore: `daemon-reload` + re-import must equal the pinned
  definition, or the recorded original goes back byte-for-byte (regular
  file bytes are backed up under `<data-dir>/ownership/`, links by target)
  and the operation refuses. The unit is never restarted — a running
  engine keeps its MainPID.
- Once owned, the drift check short-circuits on the file digest; a hand
  edit is `definition_drift` naming `service_file`.
- `release` restores the original byte-identical (or relinks) and hands
  ownership back to external — the Python-llmctl hand-back path.
- `apply_definition` is the staged-upgrade primitive (below).

### Version pin + staged upgrade with rollback (D)

Version ownership is not auto-update. Per workload, an inventory `recipe`
(`name`, `packages`, `apply: exec|env`, optional `find_links`, `index_url`,
`model_config`) enables it:

- **Version reference** (`VersionRef`, pinned at `definition_schema: 2`):
  interpreter, package versions as the env reports them, CUDA, a
  `pip freeze` lock digest, and content digests of non-system `Exec*`
  files (the launcher script). Legacy schema-1 pins are re-pinned at daemon
  start from the live env — never reported as false drift. The drift guard
  covers it: an in-place `pip install -U` or a launcher edit is
  `definition_drift` (`version.*` fields) even though the unit file is
  byte-identical.
- **Stage** (`llmctl-next upgrade stage NAME VERSION`): a new env at
  `<data-dir>/envs/<name>-<ver>-<hash>` built with `uv venv` + `uv pip
  install` (≥10 GiB free checked first, install log kept), then offline
  validation: imports, reported versions, the served model's config. The
  manifest is written last — an interrupted or failed stage is never
  promotable. The serving env is never touched.
- **Promote** (`upgrade promote NAME VERSION`): an admitted `upgrade`
  operation — stop, `ApplyDefinition` (prove-or-restore), start, await
  readiness, verify served identity, then `VerifyVersion` (the running env
  must report exactly the target versions). Failure rolls back to N
  automatically (definition, start, readiness, identity); a failed
  rollback is `needs_attention`, as always.
- **Rollback / history / gc**: `upgrade rollback NAME` applies the stashed
  N (definition + env recorded in the `upgrades` table at promote time);
  `upgrade list NAME` shows the history; `upgrade gc NAME` drops envs no
  row or the live interpreter needs. Old envs, shared weights and HF
  caches are never touched by anything else.

Proven by an opt-in spike that builds a real `fakeengine` 1.0 / 2.0 /
2.0.1-broken package into a temp wheel index and drives: adopt → own →
recipe+re-pin → stage 2.0 (validated) → promote (unit serves 2.0, history
`promoted`) → rollback (serves 1.0) → broken 2.0.1 fails validation and is
never promotable → a manifest-less stage is not promotable → an in-place
env change is drift. The real vLLM staging (recipe `vllm`) runs in pilot
window 2 with the owner.

## Measured (release build, 2026-11-15, this machine)

- `--version` p95 **4 ms** over 20 runs (plan's provisional budget: <100 ms).
- Fast-fail path (no daemon → exit 3) p95 5 ms.
- Release binary: 8.3 MB.

## Not yet here (per the plan's milestones)

E — the real-model pilot on `vllm-engine2` (two windows with the owner
present; see [docs/RUST-M2-PLAN.md](../docs/RUST-M2-PLAN.md)), SSH
transport and Mac agents (M3), installers and version pinning (M4), TUI
and config/preset parity (M5), migration of Python data (M5/M6). The YAML
reader choice is deliberately still open — pick a maintained crate against
current docs before config parity lands.
