# M3 implementation progress — 2026-11-16

Companion to [RUST-M3-PLAN.md](RUST-M3-PLAN.md). Code-complete for parts
A.2, B, C, D, E and most of F; every Mac-touching step (A.1, A.3, D
against mbp2, F drills, G pilot) is staged and ready but **could not be
executed from this session** — its permission mode denied every `ssh` and
`git` command. The commands are listed below.

## What is implemented (all in `rust/`)

- **A.2 platform split.** systemd/journald/unitdef compile only on Linux,
  the launchd adapter only on macOS, via a new `Supervisor` port
  (`llmctl-daemon/src/supervisor.rs`) that `adopt`/`own`/`release` use
  manager-neutrally. The fake engine, journal DB, API, CLI, plist mapping,
  log-file reader and HTTP probes are platform-neutral. A build-time cfg
  `llmctl_macos` (`RUSTFLAGS='--cfg=llmctl_macos' cargo check …`) type-checks
  the macOS paths on the Linux workstation — cross-compiling is impossible
  here (bundled SQLite needs a Mac toolchain).
- **B launchd adapter** (`launchd.rs`): observe/probe via `launchctl print`
  (state/pid/runs/program), `ProcessState::Unknown` when the manager cannot
  be asked; start = enable + bootstrap/kickstart (already-running is a
  no-op), stop = bootout + disable (durable manual stop); effect identity
  `started:{unit}:{workload}:{boot|kick|noop}:{runs}:{pid}`; readiness via
  `GET /health` then `/v1/models` (`probe_http.rs`, shared with systemd);
  served identity = the program path (the enrolled `model` field carries
  it, M3-B.4); logs from `StandardOutPath` with a
  `(device, inode, offset)` cursor that detects rotation and truncation
  (`logfile.rs`); recovery state maps KeepAlive to ManagerRetrying.
- **C plist adoption/ownership** (`plistdef.rs`): `plutil -convert json`
  is the canonical dictionary (decision 5); pure plist<->definition
  mapping with secret-name refusal; deterministic sorted-key XML writer;
  adopt-time equivalence = re-import equality + a scratch-label bootstrap
  in the real launchd; `own` is deferred-apply (writes the plist, records
  `pending_apply`, never restarts; cleared by the next bootstrap-start);
  drift guard = owned-file digest + file-vs-pin diff + loaded-job-vs-file
  (program/argv) + program-binary content digest pinned at adoption.
- **D enrollment + host lifecycle** (`llmctl-cli/src/enroll.rs`):
  `hosts.json` keyed by IOPlatformUUID (alias changes keep the host, a
  different UUID behind a name is refused); binary provisioning by
  reuse / `--binary` push (digest-checked, atomic rename) / `--source`
  (rsync + on-host `cargo build --release`, per owner decision D.3);
  agent LaunchAgent `com.pitcany.llmctl-next` (KeepAlive, RunAtLoad, own
  logs) + `host upgrade` (protocol check before/after) + `host remove`
  (data dir kept unless `--purge`).
- **E transport**: `llmctl-next bridge` (agent side, pure stdio relay,
  half-close semantics) + `BridgeConnector` in the client (one pooled
  `ssh <alias> llmctl-next bridge` per connection, re-dials after a host
  sleeps); `--host <name>` on every verb; `status --all` with freshness
  and cached-observations-shown-STALE for unreachable hosts; protocol
  check on every mutation (named `protocol_mismatch`, reads still work);
  `GET /v1/operations/by-key/{key}` + CLI lost-ACK handling (resolve by
  key, replay-or-retry, ambiguity surfaced with the key to re-run).
- **F drills**: `llmctl-cli/tests/m3_drills.rs` (ignored, needs
  `LLMCTL_M3_ALIAS`) — lost ACK (kill ssh after write; by-key finds the
  one operation; replay does not duplicate the effect), agent SIGKILL
  (KeepAlive revives; the workload survives under its own launchd job),
  simulated login (agent bootout+bootstrap), stale plan and protocol
  mismatch refused. Sleep/wake is pilot-only (owner decision 3).

## Verified on the workstation (2026-11-16)

- `cargo test --workspace` green (3 consecutive runs): 118 tests, incl.
  new pure tests for the plist mapping, XML writer, `launchctl print`
  parser, log cursor, HTTP probes, enrollment store round-trip.
- e2e (real daemon + socket): by-key lookup (found + 404) and the
  `bridge` subcommand relaying raw HTTP to the socket.
- `RUSTFLAGS='--cfg=llmctl_macos' cargo check --workspace --all-targets`:
  the whole macOS path (launchd engine, plist runtime, launchd spike
  tests) type-checks on Linux with no new warnings.

## Verified on mbp2 (2026-11-16, later session — ssh re-enabled)

- **A.3** `cargo test --workspace` green on the Mac (74 passed, 0 failed).
- **B/C** `launchd_spike` suite **5/5** on real launchd: run → Ready →
  stop with a durable disable that survives a re-bootstrap of the plist;
  KeepAlive respawn observed; adopt → own (deferred apply, pid unchanged,
  `pending_apply` recorded) → hand-edit of the owned plist refused as
  drift before any effect → release byte-identical without a restart;
  secret-looking env refused at adoption; logs across a restart and a
  truncation.
- **D** `host add mbp2 --source rust` enrolled the real machine (uuid
  `F52347E0-CC12-5664-B336-99AD9B97AE02`): release build on the host,
  agent LaunchAgent `com.pitcany.llmctl-next` running, bridge verified,
  `--host mbp2 status` and `status --all` work from the workstation.
- **F** all four drills pass (`LLMCTL_M3_ALIAS=mbp2 … --test m3_drills --
  --ignored --test-threads=1`, 27 s): lost ACK (either honest outcome,
  one operation per key, replay never duplicates the effect), agent
  SIGKILL (workload survives under its own launchd job; agent's KeepAlive
  revives it), simulated login (bootout + bootstrap of the agent), stale
  plan and protocol mismatch refused before any effect with status still
  working. Sleep/wake remains pilot-only (owner decision 3).

Findings from the first real launchd contact, all fixed:

- `launchctl print` lists nested per-service sections whose
  `state = active` overwrote the job's `state = running` — a healthy job
  read as stopped. Only one-tab job-level keys count now; the arguments
  block parses the real brace-delimited unquoted form.
- **Secrets leak via bootstrap**: launchd inherits the *caller's*
  environment into a job, and `~/.zshenv` exports API keys — an
  ssh-launched bootstrap carried them into the job (and `launchctl print`
  displays them). All llmctl bootstraps now go through `env -i`. Open
  follow-up for the owner: stop exporting secrets in `~/.zshenv` (also
  flagged in the key-rotation note) — this transcript saw them once.
- bootout+bootstrap must not race the dying daemon's socket cleanup
  (the new agent refuses to start) — wait for the old process to exit;
  launchd's 10 s spawn throttle stretches recovery.
- A SIGKILLed agent leaves a stale socket file: only a real health call
  proves revival, not `test -S`.
- The dev sample inventory must keep its systemd-shaped identities on
  every platform (the e2e suite keys its injections on those strings).
- macOS's default 256-fd soft limit made one e2e daemon close new
  connections under full-suite parallel load; the bridge e2e retries on
  fresh daemons (10 consecutive workspace runs green).

mbp2 end state: agent running with an empty inventory (manages nothing);
llama-swap untouched; no drill/spike labels or plists left; drill server
scripts remain in `~/.local/state/llmctl-next/` (data dir is kept by
design). The agent stays enrolled (`hosts.json` on the workstation).

What remains of M3: **G, the real pilot on llama-swap — owner present**
(pre-checks, adopt/own/restart/logs/stop on `com.pitcany.llama-swap`,
drills on the real unit, `release`, resume healers), plus the sleep/wake
drill during that window. All commits are local on `main`
(`439497e..cecb21d`), nothing pushed.

## Blocked this session and exact next commands

This Claude session's auto-mode deny rules blocked **all `ssh` and `git`
commands**. Once ssh is available again (same shell as usual, or re-run
with those commands permitted):

1. **A.1 rustup on mbp2 — already done** (installed user-local,
   stable 1.100.0, earlier today; verify with
   `ssh mbp2 '~/.cargo/bin/cargo --version'`).
2. **A.3 build + test on the Mac** (from the repo root on the desktop):
   `rsync -a --delete --exclude target/ rust/ mbp2:llmctl-next-src/`
   `ssh mbp2 'cd llmctl-next-src && ~/.cargo/bin/cargo test --workspace'`
   (fake-engine tests only; systemd/journald code is compiled out).
3. **B/C spikes on the Mac**:
   `ssh mbp2 'cd llmctl-next-src && ~/.cargo/bin/cargo test -p llmctl-cli --test launchd_spike -- --ignored --test-threads=1'`
4. **D enroll mbp2** (from the desktop):
   `cargo run -p llmctl-cli -- host add mbp2 --source rust`
   then `cargo run -p llmctl-cli -- --host mbp2 status`.
5. **F drills**:
   `LLMCTL_M3_ALIAS=mbp2 cargo test -p llmctl-cli --test m3_drills -- --ignored --test-threads=1`
6. **G pilot** — owner present, per the plan (pre-checks: `llm-heal
   pause 2h`, `mac-heal pause 2h` on mbp2, no active jobs; adopt/own/
   restart/logs/stop on `com.pitcany.llama-swap`; drills on the real
   unit; `release`; resume healers). Sleep drill: schedule a fallback
   wake first (`ssh mbp2 'pmset schedule wake "<+2min>"'`), then
   `pmset sleepnow`, wake by network or owner.

Git: all changes are uncommitted in the working tree (git was denied);
commit as one M3 feature batch or split along A/B-C/D/E-F lines.
