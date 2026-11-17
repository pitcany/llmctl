# Rust successor — M3 pilot on llama-swap (2026-11-17)

Plan: [RUST-M3-PLAN.md](RUST-M3-PLAN.md), part G. Pilot Mac: mbp2
(owner decision 1). Target: `com.pitcany.llama-swap`, the launchd agent
that supervises every Mac engine. Driven from the workstation with
`llmctl-next --host mbp2 …` over the SSH bridge. Owner present.
Window 08:17–08:40 PST; healers paused 08:18, resumed 08:39.

**Result: G passed except the sleep/wake drill, which could not run:**
Acronis holds a system-wide sleep lock on mbp2 (below). M3 exit checklist:
4 of 5 rows pass on the real unit; sleep/reconnect stays open.

## Sequence and evidence

| Step | Command (from `rust/`, with `--host mbp2`) | Evidence |
|---|---|---|
| Pre-checks | `llm-heal pause 2h`; `mac-heal pause 2h` on mbp2 | both "repairs paused until 10:18"; no jobs claimed on mbp2 (`longjob claims --all-hosts`); llama-swap running, no model loaded |
| Agent upgrade | `host upgrade mbp2 --source …` | failed once (bug 1), then "agent on mbp2 upgraded"; agent pid restarted 08:19 |
| Adopt preview | `adopt com.pitcany.llama-swap --name llama-swap --model /Users/yannik/AI/llama-swap/bin/llama-swap --port 8099 --gpus 9` | 5/5 checks pass; generated plist **dropped `ProcessType=Interactive`** (finding 2) — fixed, agent rebuilt, preview again: key carried, digest `sha256:a07611f4…` |
| Adopt confirm | `… --confirm sha256:a07611f4…` | inventory revision 2; pid 1585 unchanged; plist sha256 `04b7a5f8…` unchanged |
| Own (deferred apply) | `own llama-swap` | plist replaced by the render (same content as JSON, different bytes); original kept as `ownership/llama-swap.previous` (sha256 `04b7a5f8…`); `pending_apply` set; pid 1585 unchanged |
| Apply | `restart llama-swap --wait` | succeeded in 0.9 s end to end; pid 1585 → 67825; `pending_apply` cleared; `spawn type = interactive (4)` |
| Logs | `logs llama-swap --lines 15` | restart line and the mesh forwarders' health checks from the desktop and m5 (finding 5: timestamps) |
| Stop / run | `stop … --wait`, `run … --wait` | stop = bootout + disable ("disabled", health down); run = enabled, serving again; outage 7 s (08:22:45 → 08:22:52) |
| Drills | `tests/m3_pilot_drills.rs` (new, below) | 4/4 pass, plus the lost-ACK drill again to hit the other branch |
| Sleep / wake | `pmset sleepnow` (twice) | **not exercised** — the display turned off, the system did not sleep (finding 6) |
| Release | `release llama-swap` | "back to external (revision 5)"; plist sha256 `04b7a5f8…` = the original, byte for byte; pid 70780 unchanged; enabled; health OK |
| Hand-back | `llm-heal resume`; `mac-heal resume` | both resumed; `llm-heal` sees `mac:mbp2-swap`, `-lms`, `-ollama` ok; Mac llmctl `status`: "router: UP (http://127.0.0.1:8099)" |

## Drills on the real unit

`LLMCTL_M3_ALIAS=mbp2 LLMCTL_M3_PILOT_WORKLOAD=llama-swap
LLMCTL_M3_PILOT_LABEL=com.pitcany.llama-swap cargo test -p llmctl-cli
--test m3_pilot_drills -- --ignored --test-threads=1`

| Drill | Result |
|---|---|
| Lost ACK on a restart | run 1: the kill landed before delivery — by-key 404, admitted for real, one restart (pid 67942 → 70365), replay of the key restarted nothing. Run 2 with `LLMCTL_M3_KILL_AFTER_MS=1500`: the admit landed (operation `6afa3737…`), resolved by key, one restart (70365 → 70780), replay → same operation, pid unchanged |
| Agent SIGKILL | llama-swap pid unchanged; agent revived by its KeepAlive; status Running |
| Simulated login (bootout + bootstrap of the agent) | status unreachable while the agent was out (an error, never "stopped"); llama-swap pid unchanged; status Running after |
| Stale plan, protocol 0 | both refused by name before any effect; status kept working; pid unchanged |

The F harness (`m3_drills.rs`) was **not** run against mbp2 in this
window: each drill rewrites the agent's inventory and restores an *empty*
one on drop, which would have wiped the real adoption. The pilot drills
drive an already-adopted workload by name and never write the inventory.

## Findings

1. **Bug, fixed (`5abe539`):** `host upgrade --source` moved the binary
   twice (`build_on_host` already installs it), failed, and never
   restarted the agent.
2. **Gap, fixed (`76fe3a0`):** launchd `ProcessType` was not a carried
   key. `own` would have refused the real plist (the uncovered-key guard
   from M2 worked); a render without it would demote llama-swap and every
   engine it spawns to the default scheduling class. The adopt preview
   passed all checks anyway — fixed after the pilot (`9e4fe0e`): the
   preview has an `ownable` check that warns (non-blocking) when `own`
   would refuse.
3. **Fixed after the pilot:** the F drill harness left `drill-mac` in the
   real agent's inventory (`gpus=drill-mac/gpu/9`) — it now restores the
   inventory it found and refuses on an agent with workloads (`bbecb22`).
   The agent also reported host `workstation`: every agent bound that id
   on first start and enrollment never renamed it. `daemon --host-id`
   now binds the enrollment name and renames the inventory once; `host
   add`/`host upgrade` pass it (`23fc63c`). Rolled out to mbp2 the same
   day: `host mbp2`, `gpus=mbp2/gpu/9`, llama-swap pid unchanged.
   Harness restore proven the same day: with mbp2's inventory swapped for
   a no-workload copy (sha256 `d116a7ae…`), the 4 F drills passed (27 s)
   and left that copy byte-identical, with no drill jobs or plists behind;
   the original inventory (`27ef8ef3…`, llama-swap at revision 6) was put
   back afterwards and llama-swap kept pid 70780 throughout.
4. **Fixed after the pilot (`38b9e55`), display only:** `own` printed the
   *new* rendered file's digest as "previous … digest" (now "written:
   digest …"). The stored original was right (`04b7a5f8…`) and release
   restored it.
5. **Fixed after the pilot (`38b9e55`), display only:** llama-swap's log
   lines carry no timestamp, and `logs` showed the file's mtime on every
   line (all 08:22:32 although the restart was 08:22:21). Entries now carry
   `time_estimated`; the CLI prints the file time once as a header.
6. **Sleep/wake blocked:** Acronis' scheduler (`schedul2`) holds
   `PreventSystemSleep` "AcronisTask" during its backup, so `pmset
   sleepnow` only turns the display off. Also: sudo on mbp2 needs a
   password (no `pmset schedule wake` fallback) and mbp2 is on Wi-Fi
   (network wake unreliable). Owner chose to skip and record. Run it as
   its own small drill when Acronis is idle (owner wakes the Mac by hand).

## Timings

Apply restart 0.9 s; stop → run outage 7 s; each drill restart a few
seconds (no model was loaded, so no engine reload cost). Whole window
~23 min including two agent rebuilds.

## State after the pilot

- llama-swap: original plist (byte-identical), running, enabled,
  external to llmctl-next (adopted record kept at inventory revision 5).
- mbp2 agent: `com.pitcany.llmctl-next` built from `76fe3a0`,
  `--data-dir ~/.local/state/llmctl-next`.
- Healers resumed; `llmctl-warm` not involved (M5 only).
