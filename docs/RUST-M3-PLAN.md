# Rust successor — M3 plan (remote execution)

Status 2026-11-17: A–G done on mbp2. The llama-swap pilot passed adopt,
own, apply, logs, stop/run, release (byte-identical) and the drills on the
real unit ([pilot report](RUST-M3-PILOT-2026-11-17.md)); the sleep/wake
drill could not run then (Acronis blocked sleep on mbp2); **it passed on
2026-11-18** ([drill report](RUST-M3-SLEEP-DRILL-2026-11-18.md)), so M3 is complete. A–F
evidence: [progress note](RUST-M3-PROGRESS-2026-11-16.md). Owner answers
below.
Parent: [RUST-SUCCESSOR-PLAN.md](RUST-SUCCESSOR-PLAN.md)
(milestone table, M3 row). Builds on M2 ([M2 plan](RUST-M2-PLAN.md),
[pilot](RUST-M2-PILOT-2026-11-16.md)) and the answered M0 questions
([capability matrix](RUST-M0-CAPABILITY-MATRIX.md)).

M3 deliverable (parent): *macOS agent/supervisor adapter, SSH
transport/enrollment, remote install and operations, local CLI targeting
remote hosts.* Exit criterion, verbatim: *Linux-to-Mac management works;
disconnect/lost ACK, sleep/reconnect, host/agent restart, stale plans and
protocol mismatch do not duplicate effects or imply false success.*

## Scope

**In M3:** one supervised workload per Mac, **llama-swap** (the launchd
agent `com.pitcany.llama-swap`), managed from the workstation over SSH: the
launchd adapter, Mac adoption, agent install and enrollment, the SSH
transport, and the failure drills the exit criterion names.

**Not in M3 (moved to M4, "required engines and coordination"):** engine
versions under llama-swap (ds4 builds, MTPLX/mlx-vlm venvs, mlx-serve
runtimes — owner decision 2026-11-16 that llmctl-next owns them stands),
taking over llama-swap config generation from the Mac llmctl, and the
Ollama / LM Studio adapters on macOS. M3 proves the remote mechanics on the
one unit every Mac engine already runs under.

## Facts that shape the plan (checked 2026-11-16, read-only)

- **Macs:** M5 (`Mac17,6`, M5 Max 128 GB, macOS 27.0.1) and mbp2
  (`MacBookPro18,4`, macOS 27.2.1), both arm64, both reachable as
  `ssh m5` / `ssh mbp2` (BatchMode, keys, WireGuard/LAN first). Each has a
  stable `IOPlatformUUID` for enrollment.
- **No Rust toolchain on either Mac** (Command Line Tools on the M5, Xcode
  on mbp2). The workspace bundles SQLite through a C build, so cross-checking
  from Linux without the macOS SDK will not link: build on a Mac.
- **llama-swap is the Mac supervisor of every engine** (`mlx-serve`, MTPLX,
  `mlx-vlm`, ds4), `KeepAlive` + `RunAtLoad`, regular plist file in
  `~/Library/LaunchAgents/` (not a symlink), logs to
  `~/Library/Logs/llama-swap/llama-swap.{out,err}.log`, `GET :8099/health`
  answers `OK`. Its `EnvironmentVariables` hold only `HOME`, `PATH`
  (`HF_HOME` on the M5): no secrets to refuse.
- **Its config is generated.** *(Wrong — corrected 2026-11-17 in the
  [M4 plan](RUST-M4-PLAN.md): both configs are hand-edited and committed
  to git; `llmconf.py` only stores the path.)* The Mac llmctl (`~/local-llm/bin/llmctl`,
  repo `pitcany/local-llm`) writes llama-swap's `config.yaml` from its
  registry (`lib/llmconf.py`) and adds health checks, a memory budget,
  downloads and benchmarks. llmctl-next must not hand-edit that file in M3.
- **Other actors that can restart or load things on a Mac:**
  - the desktop's `llm-heal` repairs Mac lanes over SSH; `mac-heal` (every
    120 s) takes over while the desktop is down — both have `pause 3h` /
    `resume`;
  - `llmctl-warm` (M5 only, every 600 s) loads `mimo9b-m5` through the Mac
    llmctl;
  - `lms-stray-watch` (M5, every 30 s) watches LM Studio only.
- **launchd semantics (M0 Q5 spike on mbp2):** identity = label + `runs` +
  PID + start time; `KeepAlive` respawns SIGKILL *and* SIGTERM, so stop =
  `bootout`, and a durable manual stop also needs `disable`; exit cause is
  reported; agents live in `gui/501`, which exists only while the user is
  logged in at the console (`user/501` bootstrap is refused).
- **No reload-without-restart on launchd.** Changing a loaded agent's plist
  takes `bootout` + `bootstrap`, which restarts llama-swap and with it every
  engine it spawned. systemd's `daemon-reload` had no equivalent cost.
- **Sleep:** the M5 has `sleep 0` (never idle-sleeps; power assertions held);
  mbp2 has `sleep 1`. Both have wake-on-network (`womp 1`).

## Design decisions (delegated defaults, per the parent plan)

1. **Transport = a stdio bridge over SSH.** The client runs
   `ssh <alias> ~/.local/bin/llmctl-next bridge`, and `bridge` relays
   stdin/stdout to the agent's local Unix socket. The same HTTP API and DTOs
   flow through it. It works through the existing `~/.ssh/mesh-route`
   ProxyCommand, needs no listening port, leaves no stale forwarded socket
   files, and keeps SSH as transport only (no remote shell logic).
2. **The Mac agent is a LaunchAgent** (`com.pitcany.llmctl-next`, `gui/501`,
   `KeepAlive`), with its own data dir, SQLite journal and inventory, the
   same binary as Linux. It inherits the console-login limit; documented,
   not worked around.
3. **Enrollment** records `{host_id: IOPlatformUUID, ssh alias, agent
   version, protocol}` on the workstation. An address or alias change keeps
   the host; a different UUID behind the same alias is refused.
4. **Host-scoped idempotency keys** chosen by the caller before sending
   (`<host_id>/<verb>-<name>-r<rev>-<ms>`). After any transport error the
   client asks the agent for that key before considering a retry: a lost
   ACK is ambiguous, never "failed to launch".
5. **Launchd equivalence = plist semantics.** launchd has no drop-ins or
   merging, so the imported definition is the canonical plist dictionary; the
   loaded job (`launchctl print`) is compared with the file to detect "edited
   but not reloaded".
6. **`own` on launchd is deferred-apply.** It writes the rendered plist and
   records `pending_apply`; the change takes effect at the next admitted
   `restart`, never silently. `release` restores the original bytes the same
   way.

## Remaining work, in order

### A. Build and run llmctl-next on macOS (~2–3 h)

1. Install a user-local Rust toolchain (`rustup`, no sudo) on the pilot Mac
   (owner decision 2).
2. Split platform code behind `cfg`: systemd, journal, nvidia-smi and
   `/proc` cgroup code compile only on Linux; the fake engine, journal DB,
   API, CLI and bridge everywhere.
3. `cargo test --workspace` green on the Mac (fake engine, temp dirs).

### B. launchd adapter (~1 day)

1. Observe: `launchctl print gui/<uid>/<label>` → state, PID, `runs`, last
   exit / terminating signal; start time from the PID; manager unreachable →
   `Unknown`, never `Stopped`.
2. Effects: start = `bootstrap` (or `kickstart` if loaded); stop = `bootout`;
   durable manual stop = `bootout` + `disable`, and `run` re-`enable`s; restart
   = `kickstart -k` (the job's own `KeepAlive` must not race a stop).
3. Effect identity = label + `runs` + PID start time (the systemd
   `InvocationID` analogue).
4. Readiness: `GET /health` on the workload's port; identity: the expected
   program path + version where the engine exposes one.
5. Logs: the plist's `StandardOutPath` / `StandardErrorPath`, cursor =
   (device, inode, offset), so rotation and truncation are detected; same
   wire format and escaping as the journal reader.

Tests: disposable launchd spikes on the Mac (a tiny HTTP server under a
throwaway label, like the systemd spikes): run → Ready → stop; KeepAlive
respawn reported as such; SIGKILL vs bootout; durable stop survives
`bootstrap` of the plist; logs follow across a restart and a truncation.

### C. Mac adoption of llama-swap (~half a day)

1. Import: `plutil -convert json` of the plist → launchd launch definition
   (ProgramArguments, EnvironmentVariables with the secret-name check,
   WorkingDirectory, KeepAlive, RunAtLoad, Std*Path, ThrottleInterval).
2. Equivalence: render → canonical plist equal to the source; loaded job
   matches the file.
3. Drift guard: plist bytes digest + loaded-job comparison + content digest
   of the program binary (llama-swap is a single static binary: an in-place
   upgrade is drift).
4. `own` / `release` with deferred apply (decision 6) and byte-identical
   restore.

Tests: disposable label adopted, owned, released byte-identical; an edited
but not reloaded plist reported as drift; a secret-looking env name refused.

### D. Agent install and enrollment over SSH (~half a day)

1. `llmctl-next host add <alias>`: read `IOPlatformUUID` and arch over SSH,
   copy the matching binary to `~/.local/bin/llmctl-next` (atomic rename,
   digest checked), write and `bootstrap` the agent's LaunchAgent, wait for
   its socket, record the enrollment.
2. `host upgrade` (agent binary swap, protocol check before and after) and
   `host remove` (bootout, remove plist and binary; data dir kept unless
   asked).
3. Build strategy for the Mac binary: built on the Mac itself in M3 (decision
   from fact "no toolchain"); release builds are an M7 concern.

### E. SSH transport and remote CLI (~1 day)

1. `llmctl-next bridge` (agent side) and a bridge connector in
   `llmctl-client` (spawned `ssh`, framed HTTP over its stdio).
2. `--host <name>` on every verb; `status --all` shows each host with
   freshness (`seen 12 s ago`, `unreachable since …` → observations marked
   stale, never "stopped").
3. Protocol check on connect; mismatched versions refuse mutations with a
   named error and still allow `status`.
4. Retry rules from decision 4, including `--wait` resuming by key after a
   reconnect.

### F. Failure drills (exit criterion, disposable first) (~half a day)

Against a disposable launchd workload on the Mac, driven from the
workstation:

| Drill | How | Pass |
|---|---|---|
| Lost ACK | kill the `ssh` bridge right after the admit request is written | retry by key finds the one operation; no second effect |
| Disconnect mid-operation | drop the bridge during `AwaitReadiness` | the agent finishes; reconnect shows the true result |
| Agent restart | SIGKILL the Mac agent mid-operation | reconciles as `Interrupted` honestly; workload survives (own launchd job) |
| Host restart (simulated) | boot out and bootstrap the agent's LaunchAgent, as login does (no reboot: owner decision 3) | agent comes back; observations stale until then |
| Sleep / wake | `pmset sleepnow` on the pilot Mac, wake by network or owner (pilot) | stale while asleep; no duplicate effect after wake; in-flight ops resolved by key |
| Stale plan | admit with an old inventory revision | refused before any effect |
| Protocol mismatch | an older client binary | mutation refused by name, `status` works |

### G. Real pilot on one Mac (owner present, ~1–2 h)

1. Pre-checks: `llm-heal pause 2h` (desktop), `mac-heal pause 2h` (pilot
   Mac), `llmctl-warm` booted out for the window (M5 only), no active jobs
   on the Mac.
2. Enroll the Mac, adopt llama-swap (preview, confirm), `own` (deferred
   apply), `restart` to apply (all Mac lanes reload — the owner accepts that
   outage), `logs`, stop / run.
3. Drills from F on the real unit: lost ACK, agent SIGKILL, simulated login
   start, sleep/wake, protocol mismatch.
4. `release`, resume the healers and `llmctl-warm`, confirm the Mac llmctl
   and the router drive llama-swap as before.
5. Write up as `docs/RUST-M3-PILOT-<date>.md`.

## Exit checklist (maps to the parent criterion)

- [x] Linux-to-Mac management: enroll, adopt, run / stop / restart / logs
  from the workstation — disposable and pilot
- [x] Disconnect / lost ACK: no duplicate effect, no false success —
  disposable and pilot (both branches on the real unit)
- [x] Sleep / reconnect: stale while asleep, resolved by key after wake — pilot
  (done 2026-11-18 on mbp2 after a fix to `status --all`, `7c06963`;
  [drill report](RUST-M3-SLEEP-DRILL-2026-11-18.md))
- [x] Host / agent restart: agent SIGKILL reconciles honestly (disposable and
  pilot); login start simulated by bootout + bootstrap of the agent (pilot).
  A real reboot is not exercised (owner decision 3).
- [x] Stale plans and protocol mismatch refused before any effect —
  disposable and pilot

Carry-overs from M2 — fixed 2026-11-16 (disposable-unit spike
`staged_upgrade_promotes_verifies_and_rolls_back`):
- `rollback` straight after `promote --wait` was refused until a 2 s
  history poller caught up. History rows now settle from their operation on
  every read (compare-and-set, no poller).
- Restaging a version overwrote its history row. Each stage now adds its
  own row; restaging the env that is serving, or the env the last promote
  would roll back to, is refused (it would delete that env).
- Found while testing: after a successful promote the version pin stayed
  on the old env, so the next `restart` was refused as version drift, and
  a rollback could expect the wrong version. The executor now re-pins the
  version facts once `VerifyVersion` passes; promote records the version
  it replaces at promote time.
- `upgrade gc` (fixed 2026-11-17) never removed anything: it kept the env
  of every row it had not discarded, and no row was ever discarded. Worse,
  `envs/` is shared by every workload and gc deleted any entry it did not
  keep, so a gc of one workload removed another workload's envs (the spike
  shows the old code deleting `adopted2-1.0-…`). gc now touches only this
  workload's entries (name and hash recomputed), keeps the live env and
  every env a validated, promoted or in-flight row points at (including
  each promote's rollback target), removes rolled-back, failed, superseded
  and interrupted stages, marks their rows `discarded`, and runs under the
  same lock as stage, promote and rollback.

## Owner decisions (2026-11-16)

1. **Pilot Mac: mbp2** (decided). Fewer actors than the M5 (no
   `llmctl-warm`, no `lms-stray-watch`); a llama-swap restart interrupts
   fewer lanes.
2. **Rust toolchain on mbp2: yes** (decided) — user-local `rustup`.
3. **Sleep drill: yes; reboot drill: no** (decided). The "host restart" row
   is therefore covered by an agent SIGKILL plus a simulated login start (the
   agent's LaunchAgent booted out and bootstrapped again, as login does), not
   by a real reboot; the exit checklist records that limit.
