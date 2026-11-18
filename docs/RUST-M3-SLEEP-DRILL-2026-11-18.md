# Rust successor — M3 sleep/wake drill on mbp2 (2026-11-18)

Plan: [RUST-M3-PLAN.md](RUST-M3-PLAN.md), drill table row "Sleep / wake"
and exit row "Sleep / reconnect". It could not run in the
[M3 pilot](RUST-M3-PILOT-2026-11-17.md) because Acronis held
`PreventSystemSleep` on mbp2 (finding 6 there). Run on the owner's
request before M5 starts. Window 07:55–08:09 PST; llm-heal and mac-heal
paused 07:56, resumed 08:09.

**Result: the exit row is met, with one fix and two limits written
below.** Stale while asleep (after fix `7c06963`), no duplicate effect
after wake, and an operation that was running when sleep began is
resolved by its key afterwards.

## Pre-checks

- Acronis idle: `pmset -g assertions` showed no `PreventSystemSleep`
  holder (only idle-sleep holds, which `pmset sleepnow` overrides).
  `schedul2` was running but not backing up.
- mbp2 on AC (96 %), `womp 1`, `tcpkeepalive 1`. Agent pid 45131,
  llama-swap pid 2328 (authority `external` since the M3 release), no lane
  loaded.
- The in-flight operation is a lane load (`lanes swap`), because
  llama-swap is observe-only now; `qwen35b-m1` takes about 30 s to load.

## Sequence and evidence

Times are PST; mbp2's power log (`pmset -g log`) is the source for every
sleep and wake.

| Time | What | Evidence |
|---|---|---|
| 07:57:35 | Admit `lanes swap llama-swap qwen35b-m1 --key sleepdrill-20261118 --wait` from the workstation | operation `3693b0e6…` admitted, phase 0 `EffectPending` |
| 07:57:39 | `pmset sleepnow` over ssh | "Sleeping now..." |
| 07:57:44 | **Sleep begins** ("Entering Sleep state due to 'Software Sleep'") | power log; powerd delayed the sleep notifications by 28 s |
| 07:58:08 | The load finishes while the Mac goes down | result `succeeded`, effect `lane-load:llama-swap:qwen35b-m1`; the waiting client got it |
| ~07:58:12 | Kernel asleep | ssh times out on every route at 07:58:45; ping is answered (the Wi-Fi card answers for the Mac) |
| 07:59:08 | `--host mbp2 status` (old binary) | **hung 69 s**, then answered fresh: its ssh attempt woke the Mac |
| 08:00:15 | DarkWake, reason `wifibt` (network wake, no display) | power log; agent pid 45131 and llama-swap pid 2328 unchanged, `qwen35b-m1` ready |
| 08:01:00 | Maintenance sleep, by itself, 45 s after the DarkWake | power log |
| 08:01:21 | (before that) lane unloaded for round 2 | `lanes stop` succeeded |
| 08:01:34 | Round 2: keyed admit + `pmset sleepnow` | the Mac was **already asleep** (08:01:00), so the admit never reached the agent: `kex_exchange_identification: Connection closed`; a stale ssh ControlMaster socket broke the next attempt too. No operation was created (`ops` has no `sleepdrill2` entry), no lane loaded |
| 08:04:37 | `status --all` with the fix (`7c06963`) while asleep | **10.3 s**, then `host mbp2 UNREACHABLE since … — last observations (STALE)`, workloads tagged `[stale]`, never "stopped" |
| 08:07:27 | DarkWake, reason `wifibt` (the wait loop's ssh) | power log; ssh answers from 08:07:30 |
| 08:07:54 | Re-admit with the round 1 key `sleepdrill-20261118` | **same operation `3693b0e6…`**, original result (`completed_at` 15:58:08 UTC); llama-swap still lists nothing running, so the lane was **not loaded again** |

## Exit row, checked

| Requirement | Result |
|---|---|
| Stale while asleep | met after `7c06963`: `status --all` gives each enrolled host 10 s (`LLMCTL_NEXT_STATUS_TIMEOUT_S`), then shows its cached observations as STALE. Before the fix the view hung about 70 s and woke the Mac |
| No duplicate effect after wake | met: the keyed re-admit returned the existing operation; no second load |
| In-flight operations resolved by key | met for an operation that was running when sleep began (07:57:44 → finished 07:58:08); its record and result survived three sleeps |
| Agent and workload survive sleep | met: agent pid 45131 and llama-swap pid 2328 the same before and after three sleeps |
| Admission while asleep | no false success: the transport failed before the agent, nothing was admitted |

## Findings

1. **Fixed, `7c06963`:** a sleeping Mac does not refuse the bridge; ssh
   (through `mesh-route`, three rounds) waits and a Wi-Fi wake follows a
   minute later. `status --all` now bounds that wait. The single-host
   `--host <h> status` still waits for the transport. That's on purpose:
   the caller asked for that host. The M5 TUI must use the bounded call.
2. **Network wake works, as DarkWake only.** Every wake today was `wifibt`
   (an ssh attempt), lasting 11–45 s before maintenance sleep. The agent
   answers during a DarkWake. The owner's hand wake was not needed.
3. **A stale ssh ControlMaster socket after a sleep** breaks the next
   connection (`mux_client_request_session: … Broken pipe`, then
   "disabling multiplexing"). Harmless, but it makes the first call after
   a wake fail. `ssh -O exit` clears it.
4. The pilot's sudo limit still holds: no `pmset schedule wake` fallback.

## Limits, said plainly

- No operation was in flight across **deep** sleep: the load finished 4 s
  before the kernel went down (the 28 s sleep-notification delay let it
  finish). Rust's `Instant` on macOS does not advance during sleep, so a
  readiness wait would resume rather than time out. That is from the
  platform docs, not observed here.
- The Mac was never fully woken (display on) during the drill, only
  DarkWakes.
- `UNREACHABLE since` prints the host's last-seen time, not the moment
  the call failed.

## State after the drill

Healers resumed 08:09 (`repairs resumed` on both). mbp2: no lane loaded,
llama-swap pid 2328, agent pid 45131, both unchanged. Workstation unchanged.
