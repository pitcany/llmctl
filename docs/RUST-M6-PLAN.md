# Rust successor — M6 plan (cutover)

Status 2026-11-27: **M6a done** (read-only). The result is in
[RUST-M6-AUTHORITY.md](RUST-M6-AUTHORITY.md): the authority map, the
serving signature of the three hosts, and nine findings. The M5 pilot
ended on 2026-11-27 with seven counted days; the close-out walk is in
[RUST-M5-WORKFLOWS.md](RUST-M5-WORKFLOWS.md#close-out-walk-2026-11-27).
The text below is the 2026-11-22 plan and is not changed.

Status 2026-11-22: **owner decisions taken.** The owner took the
recommendation in each of the 16 owner decisions ("Take your
recommendations", 2026-11-22). M6 is not started. M5 is in its pilot: clean days 1 and 2 of 7 are closed.
The earliest M5 exit is 12:52 PST 2026-11-27. No agent is redeployed before
that time.
Parent: [RUST-SUCCESSOR-PLAN.md](RUST-SUCCESSOR-PLAN.md) (milestone table,
M6 row; "Cutover and reversal"; "Compatibility and storage"). Builds on M5
([plan](RUST-M5-PLAN.md), [progress](RUST-M5-PROGRESS.md),
[workflow matrix](RUST-M5-WORKFLOWS.md)). M7's distribution half
([plan](RUST-M7-PLAN.md)) is built on branch `m7a-distribution` and merges
after M6.

M6 deliverable (parent): *Quiesced controllers per host, verified imports,
client/router switch, exact builds and compatible agent versions.* Exit
criterion, verbatim: *One authority per managed resource, preserved
IDs/config/history, observed serving state matches, and reversal restores
the previous control paths.*

How to read this file. Each fact has a label:

- **checked**: read on 2026-11-22 with the command or file that is named.
- **inferred**: a conclusion from checked facts. It is not measured.
- **not verified**: listed in "Facts that are not verified".

Terms. One term has one meaning in this file.

| Term | Meaning |
|---|---|
| agent | the llmctl-next daemon on one host |
| managed resource | a workload, a llama-swap lane, a llama.cpp router model, a preset file or a catalog record that an agent has in its records |
| legacy writer | a program that can change a managed resource and is not the agent |
| quiesce | make a legacy writer unable to change a managed resource |
| redirect | make a legacy writer ask the agent to make the change |
| delegate state | the M5 state: legacy writers redirect for enrolled units and use their old path for all other units |
| cutover state | the M6 state: no legacy writer has an old path |
| default model | `qwen27b` on `vllm-engine` (the runbooks also call it the tutor) |
| window | a period, approved by the owner, in which the default model can be away |

## What M6 means in practice

At the end of M6:

1. For each managed resource, the agent on its host is the only program
   that changes it.
2. Each legacy writer is retired, redirected or read-only. One switch per
   tool selects the state, as in M5.
3. The catalog, the presets, the settings and the Python history are
   preserved. Each snapshot has a sha256 record.
4. The three hosts run one recorded build with one protocol version.
5. A rehearsed procedure restores the delegate state. A second rehearsed
   procedure gives the authority back to the legacy controllers in full.

M6 changes no serving port, no model file, no engine environment and no
unit file of a serving workload. The switch itself restarts no engine.

**Not in M6:** removal of the Python llmctl, the Mac llmctl and the lane
scripts (after the M6 exit, with a separate owner approval); a rename of
`llmctl-next` to `llmctl` (decision 4); the move of the workstation data
dir out of `llmctl-next-m4a` (after the M7 merge, as the M7 plan says);
the M7 merge; the web dashboard; SGLang and FreeToken; the workspace
`cargo fmt` commit.

## Before M6, in this order

| # | Item | State on 2026-11-22 | Blocks M6? |
|---|---|---|---|
| 1 | **M5 exit checks.** Seven clean days. Walk the workflow matrix. Record each open row as passed or as an owner-accepted gap. | Days 1 and 2 closed (checked: `RUST-M5-PROGRESS.md`). The matrix is dated 2026-11-18 and is out of date. | **Yes.** |
| 2 | **Rollback fix deploy and flash adoption.** Follow `RUST-FLASH-LANE-RUNBOOK.md` on branch `fix/run-rollback-order`. | Built and tested. Not merged, not deployed (checked: `git log main..fix/run-rollback-order`, 4 commits). | **Yes.** M6 adopts three more lanes. With build `55c83de`, a failed lane start can leave the default model failed. |
| 3 | **M5d step 7** (`gpt-oss-120b` into the llama.cpp router). | Not run. Four blockers are open (checked: branch `docs/m5d-step7-runbook`). The owner decided on 2026-11-22 to switch after the M5 exit. | **No, if the owner takes decision 2.** Recommended: do it after the M6 exit. |
| 4 | **M5f rehearsal** (migration and reversal on temporary state). | Not done. `docs/RUST-M5-REHEARSAL.md` does not exist on `main` (checked: `git ls-tree main docs/`). | **Yes.** The parent requires a rehearsal of both directions before live use. M6c extends it. |
| 5 | **M5c leftovers.** (a) `hf` on m5. (b) Vendor `load` verb for the `qwen3coder30b` gap (B4). (c) Live use of `lanes matrix` on m5. (d) A store for `~/models/gpt-oss-120b`. | (a) `hf` is installed on m5 (checked: `ls /opt/homebrew/bin/hf` over ssh); the progress file says it is open. (b) Built for Ollama on 2026-11-22 on branch `worktree-m5-leftovers` (`779accb`), not merged, not deployed. (c), (d) Open. | (a), (c), (d): no. (b): **yes**, it becomes M6b item 5. |
| 6 | **M5e leftovers.** (a) TUI Run on presets and the Catalog screen. (b) mbp2 screens seen live. | (a) Inferred done: the TUI source has catalog code and no "pending" text (checked: `git grep` in `rust/crates/llmctl-tui/src`). (b) Open. | (a): **yes, verify it.** M6 turns off the Python TUI. (b): no, do it in M6a. |
| 7 | **M5b leftover.** Move the llmctl-stack pin and `uv.lock` to the deployed Python commit. | Open (checked: progress file). | No. M6d does it with the M6 Python build. |

Item 2 builds from `main` after the merge. Thus the workstation gets the
`rusqlite` fix with item 2, and the Macs get it with the M6 deploy
(inferred from the runbook's merge step).

Item 2 has a RAM precondition. Flash needs available RAM plus 11 GiB to be
at least 45 GiB. On 2026-11-22 at 17:37 PST, 33 GiB were available and
swap was full (checked: `free -g`). The precondition was not met.

Work that needs no deploy can start before the M5 exit: M6a (read-only),
M6b (repository work) and M6c (rehearsal in a container).

## Facts that shape the plan (checked 2026-11-22, read-only)

Commands used: `git log`, `git show`, `git grep`, `git rev-parse`;
`llmctl-next --help`, `status`, `status --all`, `host list`, `audit --all
--since`, `ops`, `preset list`, `model list`; `systemctl --user status`,
`cat`, `list-units`, `list-unit-files`, `list-timers`, `show`;
`nvidia-smi`; `free -g`; `sha256sum`; over ssh on the Macs: `sw_vers`,
`shasum`, `launchctl list` (labels only), `PlistBuddy Print
:ProgramArguments`, `grep next_enabled`, `df`. `settings.yaml`, env files
and `launchctl print` were not read or run.

**Branches** (checked: `git rev-parse`, `git log`, `git merge-base`):

| Branch | Tip | Relation to `main` (`0af3743`) |
|---|---|---|
| `fix/run-rollback-order` | `e22f216` | 4 commits ahead, base `62b94a7` |
| `m7a-distribution` | `bc2ab07` | 24 commits ahead, 3 behind |
| `docs/m5d-step7-runbook` | `0a0f093` | 2 documentation commits ahead |
| `fix/router-load-evict-on-demand` | `d826548` | 1 commit ahead (step 7 blocker B2) |
| `fix/router-model-lanes` | `412b8ca` | 1 commit ahead (step 7 blocker B3) |

`main` holds the `rusqlite` fix (`7f2d1cd`) and the launchd race fix
(`62b94a7`). No live agent runs them.

**Builds that run now:**

| Host | Binary | sha256 (first 16) | Build | Agent, protocol | Inventory revision |
|---|---|---|---|---|---|
| workstation | `~/.local/state/llmctl-next-m4a/llmctl-next.bin` | `10d15ca9a1697ca3` | `55c83de` | 0.1.0, 1 | 48 |
| mbp2 (macOS 27.2.1) | `~/.local/bin/llmctl-next` | `273f2847742447b2` | `55c83de` | 0.1.0, 1 | 14 |
| m5 (macOS 27.0.1) | `~/.local/bin/llmctl-next` | `273f2847742447b2` | `55c83de` | 0.1.0, 1 | 17 |

Checked: `sha256sum`, `shasum`, `host list`, `status --all`. The sha256
values equal the `55c83de` rows in `RUST-REVIEW-DEPLOYMENT-2026-11-20.md`.
The workstation agent runs since 23:36 PST 2026-11-20 (checked: `systemctl
--user status llmctl-next`). Each build prints `llmctl-next 0.1.0`, thus
the version text cannot identify a build (checked: `--version`).

**Workloads in the agents' records** (checked: `status --all`, 2026-11-23
01:37 UTC):

| Host | Workload | State | Recorded intent |
|---|---|---|---|
| workstation | `vllm-engine` | running, serves `qwen3.8-27b` | running |
| workstation | `ollama` | running | running |
| workstation | `llmster` | running | running |
| workstation | `llamacpp-router` | running, no model loaded | unmanaged |
| workstation | `gpt-oss-120b` | stopped | stopped (manual) |
| workstation | `ollama-gpu1` | stopped | stopped (manual) |
| workstation | `strata-fast` | stopped | stopped |
| mbp2 | `llama-swap` | running | running |
| mbp2 | `ollama` | running | running |
| mbp2 | `lm-studio` | running | unmanaged |
| m5 | `llama-swap` | running | unmanaged |
| m5 | `ollama` | running | running |
| m5 | `ds4` | stopped | unmanaged |
| m5 | `lm-studio` | stopped | stopped (manual) |

**Units that no agent owns** (checked: `systemctl --user list-unit-files`,
`show -p Conflicts -p Restart`): `qwen3.8-flash-next-strata` (lane
`flash`), `qwen3.8-flash-next-uncensored-strata` (`flash-uncensored`),
`deepseek-v4.1-flash-strata` (`dsv41`), `glm-5.3-flash` (`glm53`) and
`vllm-engine2`. All are linked user units and inactive. Each lane unit has
a `Conflicts=` list that names other lanes. `gpt-oss-120b`, `ollama-gpu1`,
`vllm-engine2` and `llamacpp-router` have `Restart=on-failure`.

**llm-router** (checked: `git log` in its checkout, `grep` of
`config/router.yaml` for backend names, `unit:` and `llmctl_next` keys
only). The checkout is on `main` at `e015584`.
`switch.llmctl_next_refusal_final` is `true`. The five vLLM backends go to
the agent through `llmctl_next_workload: vllm-engine`. `flash-fast` names
`llmctl_next: strata-fast`. `gptoss` names `llmctl_next: gpt-oss-120b`.
`flash`, `dsv41`, `flash-uncensored`, `glm53` and `qwen3coder30b` have no
`llmctl_next` key. The file has `stop_first`, `evict_ollama` and an idle
revert of 3600 s. On an agent exit code 3 or 5, the router uses its own
switch path (checked: `RUST-M5-PROGRESS.md`, row 2026-11-20).

**Audit since the pilot start** (checked: `audit --all --since
2026-11-20T20:52:00Z` at 2026-11-23 01:37 UTC): workstation 0, mbp2 0, m5
32. The 32 are llama-swap lane changes with an unchanged pid, and four pid
changes at 04:07 and 15:35 UTC on 2026-11-22 (the m5 restarts that the
progress file explains).

**CLI data dir** (checked: `llmctl-next --help`, `git grep` in
`rust/crates`): the default is `$LLMCTL_NEXT_DATA_DIR`, else
`~/.local/state/llmctl-next-dev` (`llmctl-daemon/src/app.rs:22`). The
workstation agent uses `~/.local/state/llmctl-next-m4a`. The Macs use
`~/.local/state/llmctl-next`. `llmctl-next` is not on `PATH` on the
workstation (checked: `command -v`).

**Deploy path** (checked: `host --help`, `host upgrade --help`, `systemctl
--user cat llmctl-next`): `host add` and `host upgrade` are for the Macs.
The workstation agent is replaced by hand (`cp`, `systemctl --user`). The
unit file is a symlink to `~/AI/services/user/llmctl-next.service`.

**Legacy services that run** (checked: `systemctl --user is-active`,
`list-timers`): `llmctl-api` (Python llmctl 0.3.0, port 8088),
`llm-router`, `llm-heal.timer` (every 2 minutes),
`llmctl-reconcile.timer` (about every 30 s, inferred from one read),
`longjob-runner`,
`router-profiles-check.timer` (hourly, read-only),
`deepseek-v41-readiness.timer`.

**Scripts with a raw `systemctl` start, stop or restart** (checked:
`grep -c` in `~/AI/services/lanes`, `~/AI/bin`, `~/AI/heal`):
`strata-start`, `strata-stop`, `glm53-start`, `glm53-stop`, `gptoss-start`,
`coder1-start`, `gemma1-start`, `qwen3coder1-start`, `qwen3coder30b-start`,
`dsv41-q`, and `heal/llm-heal.py`. Only `bin/llm-unit` and
`heal/llm-heal.py` name `llmctl-next`.

**Macs** (checked over ssh): m5 has `next_enabled: 1` in
`~/.config/llmctl/config.yaml`. mbp2 has no `next_enabled` line.
LaunchAgents that can change an engine: on m5 `com.pitcany.llama-swap`,
`ollama`, `mac-heal`, `llmctl-warm` (`llmctl warm --guarded mimo9b-m5`),
`lms-stray-watch`; on mbp2 `com.pitcany.llama-swap`, `mac-heal` and the
vendor job `com.ollama.ollama`. `llm-standby` on both Macs is a standby
gateway on port 4001. It changes no engine. mbp2's disk is 94 % full, 116
GiB free.

**Presets and catalog** (checked: `preset list`, `model list`): five
presets, `qwen27b` is `serving`, four are `ready`. The workstation catalog
has 54 models and keeps 20 bundle rows for the export.

## Legacy writers and what M6 does with each

Checked against the M5 plan, the workflow matrix, `INFERENCE-INVENTORY.md`
and the live reads above. Rows marked **new** are not in the M5 list of
legacy writers.

**Workstation**

| # | Legacy writer | Write path | M6 disposition |
|---|---|---|---|
| W1 | Python llmctl CLI and TUI | service layer to `systemctl`; delegates enrolled units since M5b | Cutover mode (M6b item 1): redirect for enrolled units, refuse for all other units |
| W2 | `llmctl-api` on port 8088 | the same service layer | Stop and disable (decision 5) |
| W3 | `llm-unit` | Python guard, else `systemctl` | Keep as the one shell entry for unit verbs. It follows W1's mode |
| W4 | `llmctl take-gpus` (llmctl-stack) | `stop_blockers`; the old path for units that are not enrolled, and on an `ImportError` | Follows W1's mode. Remove the `ImportError` fallback |
| W5 | llm-router switches | handoff to the agent; its own path (`stop_first`, `restart_cmd`, `start_cmd`) for five backends and on exit 3 or 5 | `switch.llmctl_next_only` (M6b item 2). Each backend names a workload |
| W6 | llm-router idle revert and `evict_ollama` **(new)** | stops a lane and unloads Ollama models by itself | Go through the agent, with W5 |
| W7 | `llm-heal` | asks `intent`, then repairs with `systemctl`; fails open | Report-only for agent-owned workloads (decision 8) |
| W8 | `llm-heal` `mac_repair` **(new)** | repairs Mac launchd jobs over ssh | Report-only for agent-owned labels |
| W9 | `longjob` engine stop and start | `llm-unit` | No change. Covered by W3 (drill 5 passed 2026-11-21) |
| W10 | hf2local | `longjob --stop-engine`; without `longjob`: `llm-unit`, else `sudo -n systemctl` (checked: `gpu_handoff.py`) | No change with `longjob`. Remove the `LLM_UNIT_BIN=none` path for enrolled units |
| W11 | modellab | takes the lane lease; no unit write was found (checked: `grep`, not a full read) | Read it in M6a. Redirect to `llm-unit` if a write is found |
| W12 | `strata-start` and the other lane scripts (list above) | raw `systemctl` | Redirect each unit verb to `llm-unit` (M6b item 6) |
| W13 | `restore-vllm`, `vllm-on-gpus`, `scripts/recover-vllm.sh`, `scripts/vllm-ab*.sh`, `scripts/run-quant-ab.sh` **(new)** | named in `INFERENCE-INVENTORY.md`; not read here | Read in M6a. Redirect to `llm-unit` or retire |
| W14 | Study scripts, for example mathlab `run_study.sh` **(new)** | raw `systemctl --user` (11 foreign writes on 2026-11-19/20) | Redirect to `longjob --stop-engine`. A gate check finds new ones (M6b item 6) |
| W15 | `llm-backend-guard` **(new)** | made agent operations on 2026-11-23 (checked: progress file) | Read in M6a. No change expected |
| W16 | `ds4-llmctl-sync` hooks **(new as a writer)** | writes session rows in the Python DB from six units | No-op by a flag file in M6. Remove the hooks after the M6 exit (design decision 5) |
| W17 | `llmctl-reconcile.timer` **(new as a writer)** | writes session rows in the Python DB | Stop and disable |
| W18 | systemd `Conflicts=` and `Restart=` **(new)** | systemd stops or restarts a unit by itself | M6a proves that each `Conflicts=` list is inside the planner's stop set. `Restart=` stays as M4 accepted it |
| W19 | Unowned lanes `flash`, `flash-uncensored`, `dsv41`, `glm53` | the router's own path and the lane scripts. The agent's `guard_gpus` refuses a run while they hold cards | Adopt (decision 6). `flash` is adopted before M6 |
| W20 | `vllm-engine2` **(new)** | Python `vllm_slots.place` | Adopt as a stopped workload, or retire (decision 6) |
| W21 | `qwen3coder30b` to `ollama-gpu1` (B4) | `qwen3coder30b-start` ends with `systemctl start ollama-gpu1.service` | Vendor `load` verb and a router handoff (decision 7) |
| W22 | Bypass switches **(new)** | `LLMCTL_NEXT_DISABLED=1`, `LLM_UNIT_BIN=none` | Refused in cutover mode. The reversal uses the setting |
| W23 | The owner or an agent session at a shell | `systemctl --user` by hand | Cannot be quiesced. `audit` finds it. Update the runbooks and the agent skills that describe engine starts |

**Macs**

| # | Legacy writer | Write path | M6 disposition |
|---|---|---|---|
| M1 | Mac `local-llm` llmctl | llama-swap directly; redirects owned lanes on m5 (`next_enabled: 1`); off on mbp2 | Cutover mode on both Macs: redirect, and refuse when the agent does not answer |
| M2 | llama-swap request and TTL swaps | llama-swap loads and unloads lanes by itself | **Accepted exception** (decision 16). `audit` puts them in a class of their own (M6b item 4) |
| M3 | `mac-heal` | kickstarts llama-swap and Ollama; LM Studio through `lms server` | Report-only for agent-owned labels (decision 8) |
| M4 | `llmctl-warm` (m5) | sends a warm request; loads no lane since M4c | No change. Prove it again in M6a |
| M5 | `lms-stray-watch` (m5) **(new)** | watches LM Studio; its action was not read | Read in M6a. `lm-studio` on m5 has a manual stop on record |
| M6 | Vendor updaters (brew Ollama on m5, Ollama.app on mbp2) **(new)** | replace an engine binary | Not a lifecycle write. `version` reports the drift |

## Design decisions (defaults; the owner decisions are at the end)

1. **The switch is configuration only.** M6 code lands with every switch
   off. A deploy with the switches off behaves as the delegate state. The
   switch to the cutover state changes settings and restarts llm-router.
   It restarts no engine. Thus the reversal to the delegate state is also
   configuration only.
2. **Deploy, then adopt, then switch.** The three steps have different
   risks. Each has its own check and its own reversal.
3. **Host order for the switch: mbp2, m5, workstation.** mbp2 has the
   fewest legacy writers. The workstation has the most and is last. The
   deploy order is different: the workstation gets the build first,
   because `host upgrade` for the Macs must run from a build at or after
   `62b94a7`.
4. **One build, one protocol.** All three hosts get one commit of `main`.
   It contains the rollback fix, `7f2d1cd` and `62b94a7`. M6b makes only
   additive wire changes, thus the protocol stays 1 and a host with the
   older build still answers during the rolling deploy (inferred from the
   M5 deploys, which worked this way).
5. **No unit file of a managed workload changes in M6.** A change makes a
   `definition_drift` refusal until the record is pinned again. Thus the
   `ds4-llmctl-sync` hooks stay in the six unit files, and a flag file
   makes the script a no-op. This is a change to the M5 plan text, which
   removes the hooks at M6. The removal moves to after the M6 exit.
6. **A lane lease and the host lock protect the switch.** On the
   workstation, take the lane lease for the switch. llm-router then
   refuses switches and `llm-heal` refuses repairs (checked: the header
   of `llmctl-client/src/lease.rs`). Hold the shared host lock for the
   quiesce and snapshot steps, so the snapshot is consistent. M6c finds
   out if an import needs the lock; if it does, release the lock first.
   The Macs have no lease. Their llama-swap request swaps continue; they
   change no record.
7. **The Python DB is frozen, not migrated.** After the quiesce, take one
   consistent snapshot. Rust reads only the snapshot copy. No program
   writes the live Python DB after the snapshot, until a reversal.
8. **The serving signature is the proof of "state matches".** A script
   records, per workload: unit state, main pid, port, served model as the
   endpoint reports it, and GPU or memory holdings; per Mac: the loaded
   lanes. It reads the service manager, the endpoints and `nvidia-smi`
   directly, not the agent. The exit needs three equal records: before the
   switch, after the switch, and the agent's `status --json`.
9. **Each quiesce has a negative drill.** A setting is not proof. For each
   row of the writer tables, try the old write once on a harmless target
   and record that it is redirected or refused.

## Remaining work, in order

Effort figures are working time for one session. Windows and calendar
days are extra.

### M6a. Recheck and authority map (~1 day, read-only)

1. Do the parent's recheck on each host: installed revisions, units,
   hooks, timers, launchd jobs, runtime paths and actual ownership.
2. Write `docs/RUST-M6-AUTHORITY.md`: one row per managed resource and per
   program that can change it, with the write path, the M6 disposition and
   the proof. Start from the writer tables above.
3. Read the scripts that this plan did not read: W11, W13, W15, M4, M5.
   Find the consumers of port 8088. `llm-heal` can have it as a front
   door; if so, remove that entry at the switch, or `llm-heal` restarts
   the stopped service.
4. For each owned unit, compare its `Conflicts=` list with the planner's
   stop set for that workload.
5. List each workload with intent `unmanaged` (`llamacpp-router`, m5
   `llama-swap`, m5 `ds4`, mbp2 `lm-studio`). Decide the intent to record.
   Find the verb that records it without a restart.
6. Find out if mbp2's `llama-swap` is agent-owned now. The Mac llmctl
   guard on mbp2 waits for that.
7. Write the serving signature script (design decision 8). Run it on all
   three hosts. It must agree with `status --all`.
8. See the TUI on mbp2 live, and the Run and Catalog screens with presets
   (M5e leftovers).

Check: the authority map has no row without a disposition. The signature
agrees with the agents on all three hosts. No live system is changed.

### M6b. Code for the cutover (~4–5.5 days, repository work)

Each item ships with its switch off.

1. **Python llmctl cutover mode** (`llmctl_next.mode: cutover`). A write
   verb on an enrolled unit goes to the agent, as today. A write verb on
   any other unit is refused with the reason and the `llmctl-next` verb to
   use. `LLMCTL_NEXT_DISABLED=1` is refused. `take-gpus` loses its
   `ImportError` fallback. Raise `_TIMEOUT_S` (1200 s) above the longest
   readiness timeout (1800 s) plus the rollback. Read verbs do not change.
2. **llm-router `switch.llmctl_next_only`.** With it, exit codes 3 and 5
   fail the switch with the agent's reason. Idle revert and Ollama
   eviction go through the agent. The router's own stop and start
   commands stay in the file for the reversal, but no code path reaches
   them.
3. **Healers.** A setting in `heal.yaml` and in `mac-heal.json`: for an
   agent-owned workload or label, observe and report only.
4. **Agent, small changes.** `audit --unexplained`: a lane change with an
   unchanged pid on a llama-swap workload is class `engine_scheduler` and
   is not counted. The data dir default (decision 3). The version (decision
   13).
5. **Agent, vendor `load` verb** (decision 7): `load <workload> <model>`
   for Ollama and LM Studio. It is the mirror of `unload`: it checks the
   model against the catalog, plans the GPU claim as M4c does for an
   on-demand claim, loads, and confirms through the engine's loaded list.
   The router backend `qwen3coder30b` then names the workload and the
   model. **Status 2026-11-22:** the Ollama half is built on branch
   `worktree-m5-leftovers` (`779accb`, not merged). LM Studio is not in
   it. M6b keeps the review, the merge, LM Studio if a router lane needs
   it, and the router handoff, which waits for `fix/run-rollback-order`.
6. **Scripts.** Each unit verb in the lane scripts goes through
   `llm-unit`. A gate check in the `~/AI` repository fails when a script
   has a raw `systemctl` start, stop or restart of an enrolled unit.
   `ds4-llmctl-sync` exits 0 when the flag file exists.
7. **Legacy history import** (decision 9): `history import <snapshot>`
   reads a copy of the Python DB. It keeps session and operation rows
   verbatim and read-only, split by owner host, with ids in the form
   `py:<host>:<id>`. It prints a report: imported, unsupported,
   conflicting. An imported row never authorizes an action.
8. **Mac llmctl cutover mode** (`next_only: 1` beside `next_enabled`): a
   start or stop of a lane that the agent does not own is refused.

Tests: stand-in binaries for items 1, 2, 3 and 8, as in M5b (redirected,
refused, no old path, bypass refused). Agent e2e for items 4, 5 and 7 with
the fake engine and a sanitized DB fixture. Workspace suite, the
real-systemd suite (22 of 22, serial), the Python suite in four shards and
the router suite. Do not run the tests in `~/AI/packages/llmctl-stack`
from a checkout older than `9391cb27`.

Check: all suites pass. With every switch off, the behavior equals the
build that M5 proved, apart from the rollback fix.

### M6c. Rehearsal on temporary state (~1–1.5 days)

Run in a Debian 12 container with systemd, as the rollback fix did. No
real service manager call on a host. Use a settings file that is written
for the rehearsal and holds no token.

1. Forward: start from the M5f end state. Quiesce the stand-in writers.
   Take the snapshot. Run `catalog import` (dry run, then `--confirm`) and
   `history import`. Turn the switches on. Run each negative drill. Run
   one preset switch on a disposable unit. Compare the signature.
2. Reverse to the delegate state (R1 below). Run one delegated write
   through each legacy writer.
3. Reverse in full (R2 below). Run `catalog export`, then
   Python `import-registry` into the temporary DB. Check that the Python
   side sees the running state and the catalog edits.
4. Write the result and the exact commands into
   `docs/RUST-M6-REHEARSAL.md`. M6f runs that file.

Check: both directions pass twice in a row from a fresh container.

### M6d. Deploy the M6 build, switches off (~0.5–1 day)

Do this after the M5 exit and after item 2 of "Before M6".

1. Build one commit of `main` in release mode. Record the commit, the
   `rustc` version and the sha256 of each binary.
2. Preflight on each host: no operation that is not terminal, no
   `needs_attention` row, no longjob, no lane lease, no router request in
   flight, host lock free.
3. Workstation: back up the binary, `inventory.json`, `catalog.json` and a
   journal snapshot into `deploy-backups/<date>-m6/`. Hold the host lock.
   Stop the agent, replace the binary, start the agent.
4. Macs, from the new workstation binary: `host upgrade mbp2 --source`,
   then `host upgrade m5 --binary` with mbp2's build. Back up each Mac's
   binary, inventory and journal first, as on 2026-11-20.
5. On each host, count the operations and findings that the agent can
   read. The counts must equal the counts before the deploy. On the Macs
   this is the first live use of the new bundled SQLite.
6. Install the Python build into `~/AI/.venv` (`uv pip --no-deps`),
   restart `llmctl-api`, move the llmctl-stack pin and `uv.lock`.
7. Merge the llm-router change. Restart llm-router when it has no request
   in flight. Deploy the healer and script changes.

Check: `host list` shows one agent version and protocol 1 on both Macs.
The three sha256 values equal the record. The serving signature is
unchanged. `audit --all` has no new finding. One delegated write through
Python llmctl and one router switch pass in the delegate state.

### M6e. Adopt what no agent owns (~1–2 days, with windows)

1. For each lane that decision 6 keeps (`flash-uncensored`, `glm53`,
   `dsv41`): preview `adopt` with a `--readiness-timeout` that fits the
   lane; confirm; add `llmctl_next:` to its router backend; add its unit
   to `llmctl_next.units` with an append command that does not print the
   file. Then one round trip in a window (`longjob start <name>
   --router`): switch to the lane through llm-router, wait for the model
   check, send one request, switch back, send one request.
2. `vllm-engine2`: adopt as a stopped workload, or disable the unit
   (decision 6).
3. `qwen3coder30b`: switch its backend to the `load` handoff. One round
   trip in a window.
4. Record the intents that M6a item 5 decided.

Check: every backend in `router.yaml` names a workload. `status` lists
every unit that llm-router or Python llmctl can address. `audit --all
--unexplained` is 0 after each round trip.

### M6f. Switch to the cutover state, per host (~1 day)

Run `docs/RUST-M6-REHEARSAL.md`. The order is mbp2, m5, workstation. The
table shows what is specific to each host.

| Step | mbp2 | m5 | workstation |
|---|---|---|---|
| 1. Preflight (as M6d step 2) and signature "before" | yes | yes | yes, in a window |
| 2. Take the protection (design decision 6) | host lock | host lock | lane lease, then host lock |
| 3. Quiesce | `mac-heal` report-only | `mac-heal` report-only; act on the M6a result for `lms-stray-watch` | stop and disable `llmctl-reconcile.timer` and `llmctl-api`; create the `ds4-llmctl-sync` flag file; `llm-heal` report-only, also for `mac_repair` |
| 4. Snapshot and export | Mac registry and `config.yaml` copies; the commit of the llama-swap config repository; `inventory.json`, `catalog.json`, journal snapshot | the same | Python DB `.backup`; `llmctl export-registry` bundle; sha256 list of the presets, `settings.yaml`, `router.yaml`, the profile envs and the owned unit files; `inventory.json`, `catalog.json`, journal snapshot |
| 5. Import and verify | `catalog import --as <its name>` dry run: every row `present`, or review each `new` row; then `--confirm` | the same | the same, then `catalog export` must give back the bundle's rows; `history import` on each owner host, with its report |
| 6. Reconcile | signature equals `status --json` | the same | the same; `preset list`: five presets, one `serving`, no drift |
| 7. Switch the clients | `next_enabled: 1` and `next_only: 1` (if M6a item 6 says the lanes are owned) | `next_only: 1` | `llmctl_next.mode: cutover`; `switch.llmctl_next_only: true`; restart llm-router with no request in flight |
| 8. Release the lock after step 4 and the lease after step 7; signature "after" | yes | yes | yes |
| 9. Negative drills | Mac llmctl start and stop of one small lane: redirected. With the agent stopped for one minute: refused | the same | Python `llmctl stop` of an enrolled unit that is stopped already (`ollama-gpu1`): one agent operation, nothing stops. `llm-unit` on a unit that is not enrolled: refused. `LLMCTL_NEXT_DISABLED=1`: refused. Agent stopped for one minute: a router switch fails and changes nothing. A disposable enrolled unit, stopped with raw `systemctl`: `audit` records one finding for that unit, and `llm-heal` reports it and repairs nothing |
| 10. Live reversal drill (decision 11) | R1, one delegated write, forward again | no | R1 for the settings only, one delegated write, forward again |

Use the `--as` names of the 2026-11-19 import (`RUST-M5-PROGRESS.md`).
The snapshot directory is `<agent data dir>/deploy-backups/<date>-m6-cutover/`.

Check per host: signature before = signature after = agent record; no
engine pid changed because of the switch; every negative drill has the
expected result; `audit --all --unexplained` is 0, apart from the one
finding that the drill on the disposable unit makes on purpose.

### M6g. Acceptance window and exit (7 calendar days)

1. Daily: `audit --all --unexplained` and `ops` on all three hosts. Write
   one line into `docs/RUST-M6-PROGRESS.md`: date, workflows used,
   operations, findings, anything that needed a legacy tool.
2. A day counts with zero unexplained foreign writes on all hosts. A bad
   day restarts the count after the cause is fixed (decision 12).
3. Update `docs/OPERATIONS.md`, `QUICKSTART.md` and `USER-GUIDE.md`. Check
   `~/AI/docs/ARTIFACT-INDEX.md` for published pages that describe the
   Python llmctl as the tool for starts and stops.
4. At the end, walk the exit checklist with the owner.

## Rules for the work

- No agent redeploy on any host before the M5 exit.
- Do not print `settings.yaml` or an env file. Change `settings.yaml` with
  an append or a line edit, and read back only the parsed section names.
- Do not run `launchctl print` on the Macs. It prints the environment.
- For a window with router switches, use `longjob start <name> --router`.
  A `--gpu` or `queue` lease blocks the switch.
- After a change to a pinned install, run `version <workload> --repin`.
  Without it the agent refuses the next start.
- Do not deploy the llm-router branch `fix/lane-script-lock-handoff`.
- Do not run the tests in `~/AI/packages/llmctl-stack` against a live
  agent.

## Calendar (inferred from the estimates, not a commitment)

The M5 exit is on 2026-11-27 at the earliest. If M6a, M6b and M6c are done
before it, the M6 deploy can follow the rollback fix window by one day.
The workstation switch is then possible 5 to 7 working days after the M5
exit, and the M6 exit 7 calendar days after the switch.

## Downtime per step

"Default model away" means that `qwen27b` does not answer. Times marked
inferred have no measurement in the `RUST-*` files.

| Step | What is away | Estimate | Source |
|---|---|---|---|
| Rollback fix deploy and `flash` round trip (before M6) | default model | 13–20 min; 30 min or more on a failure | checked: runbook. Flash cold loads took 674 s to 17 min; the readiness timeout is 1800 s |
| M6a, M6b, M6c | nothing | 0 | no live change |
| M6d workstation deploy | agent, llm-router and `llmctl-api` restart; no engine | less than 1 min; a request in flight at the router restart fails | checked: 2026-11-20 deployment record |
| M6d Mac upgrades | the Mac agent only | less than 1 min per Mac, after a build of 1–6 min on mbp2 | checked: M7 progress (build 1 min 17 s) |
| M6e `flash-uncensored` round trip | default model | 15–25 min | inferred: the same engine and size class as `flash`, plus about 2 min for the default model to return (95–115 s measured) |
| M6e `glm53` round trip | default model | 10–25 min | inferred |
| M6e `dsv41` round trip | default model; needs about 100 GB of RAM | 20–40 min | inferred; check RAM first |
| M6e `vllm-engine2` adoption | nothing | 0 | it is stopped |
| M6e `qwen3coder30b` round trip | default model, if the fit evicts it | 3–6 min | inferred from drill 3 (2 min) plus a model load |
| M6f mbp2 and m5 switch | nothing; lanes stay loaded | 0 | no engine restart |
| M6f workstation switch | llm-router restart, one time; three times with the reversal drill | seconds each | inferred |
| M6f drill "agent stopped" | the agent for 1 min; one switch request fails on purpose | no model is away | the agent's unit file: a stop of the agent stops no engine |
| Reversal R1 | llm-router restart | seconds | inferred |
| Reversal R2 | llm-router restart; agent stop | no model is away | inferred |
| Step 7, if it runs before M6 | default model | 10–20 min in one window, plus 1–2 days of work for the four blockers | inferred |

Sum for the default model in M6 with all four lanes kept: about 1 to 2
hours, in four or five windows (inferred).

## Reversal

Both procedures are rehearsed in M6c. Reverse the configuration before
the code. Keep every agent journal: do not restore a journal snapshot over
a journal that has newer operations.

**When to reverse:** an unexplained foreign write whose cause is not
fixed in one day; a serving signature that does not match; an agent
`needs_attention` state that blocks daily work; the owner's decision.

### R1. Back to the delegate state (the M5 state)

The agent keeps its authority over enrolled units. Legacy writers get
their old path back for all other units.

| Host | State to restore |
|---|---|
| workstation | `settings.yaml`: `llmctl_next.mode` absent, `llmctl_next.enabled: true`, the `units:` list unchanged. `router.yaml`: `switch.llmctl_next_only: false`, `switch.llmctl_next_refusal_final: true`; restart llm-router. `heal.yaml`: repairs on (restore the backup). `llmctl-reconcile.timer` and `llmctl-api.service` enabled and active. The `ds4-llmctl-sync` flag file removed |
| m5 | `config.yaml`: `next_only` absent, `next_enabled: 1`. `mac-heal.json`: the backup |
| mbp2 | `config.yaml`: no `next_only` and no `next_enabled` line. `mac-heal.json`: the backup |

Then: run `catalog export` and Python `import-registry`, so the Python DB
has the catalog edits made since the snapshot. Run `llmctl reconcile`.
Compare Python `llmctl status` with the serving signature. Run one
delegated write.

The lanes adopted in M6e stay adopted. To undo one adoption: remove its
`llmctl_next:` key and its `units:` line, restart llm-router, then
`forget <lane> --yes`. This stops nothing.

### R2. Full reversal (the legacy controllers have the authority)

This is the parent's reversal. It goes back further than the M5 state: no
agent changes an engine after it. Follow the parent's order.

1. **Drain.** `ops` shows no operation that is not terminal, on each host.
2. **Record what runs.** Take the serving signature.
3. **Stop the agent's authority.** For each workload whose unit file the
   agent took over with `own`, run `release` first. It restores the
   original file and restarts nothing. A workload that was promoted to
   another version cannot be released (an M4 limit): leave its file.
   `settings.yaml`: `llmctl_next.enabled: false`. m5 and mbp2:
   `next_enabled: 0`. Stop and disable the agent service on the
   workstation; boot out the agent's LaunchAgent on the Macs (do not use
   `host remove`: it deletes the binary). A stop of the agent stops no
   engine. With the agent away, llm-router uses its own switch path; set
   `switch.llmctl_next_refusal_final: false` and `llmctl_next_only: false`
   and restart it.
4. **Restore or reverse-export.** The presets are the same files; restore
   nothing. Run `catalog export`, then `import-registry`. Do not copy the
   DB snapshot over the live Python DB: it lost nothing, it only stopped.
5. **Reconcile.** `llmctl reconcile`; compare `llmctl status` with the
   signature of step 2.
6. **Turn the legacy tools on.** R1's list for the timer, `llmctl-api`,
   the healers and the flag file.
7. **Keep the audit trail.** Keep each agent data dir and the
   `deploy-backups` directories.

### Binary reversal

Stop the agent, restore the previous binary from `deploy-backups`, start
it. The journal keeps its additive columns. Restore `inventory.json` only
if no admitted operation changed the inventory since the backup. On the
Macs, use `host upgrade <host> --binary <previous file>` from a
workstation binary at or after `62b94a7`. The previous workstation
binaries are in the m4a data dir (`llmctl-next.bin.pre-*`; checked: `ls`).

## Order relative to M7

1. M6 builds and deploys from `main`. `m7a-distribution` stays a branch
   until the M6 exit (M7 decision 5). Rebase it after each M6 merge to
   `main`.
2. M6 uses `host upgrade` as `main` has it (`62b94a7`). It does not use
   `agent install`.
3. After the M6 exit: the format commit on `main`; rebase and merge M7
   (for `rust/Cargo.lock`, take the branch's file); write the workstation
   `agent.toml` keys before the M7 binary starts; `agent install --adopt`
   on mbp2 and m5; rewire `host add` and `host upgrade` onto `agent
   install` (M7b); then the move of the workstation data dir.
4. Three M6 choices touch M7 files: the data dir default (decision 3; M7f
   found that a plain `status` looks in the dev data dir), the version
   (decision 13; the release archive names use it), and any new source
   path (the M7a gate test scans for owner paths). Resolve them at the
   rebase.
5. Step 7, SGLang and FreeToken come after the M7 merge, each with its own
   plan or runbook.

## Exit checklist (maps to the parent criterion)

| Criterion | Evidence required |
|---|---|
| Quiesced controllers per host | each row of the writer tables has its disposition applied and its negative drill recorded (M6f step 9) |
| Verified imports | `catalog import` reports only `present` rows after the final import on each host; `catalog export` gives back the bundle's rows by id; the `history import` report has no row dropped without a reason |
| Client and router switch | `llmctl_next.mode: cutover`; `switch.llmctl_next_only: true`; every router backend names a workload; `next_only: 1` on the Macs |
| Exact builds, compatible agent versions | one commit; the sha256 of each binary equals the record and the running executable; `host list` shows one version and protocol 1 |
| One authority per managed resource | the authority map has one writer per row, or the accepted exception M2; `audit --all --unexplained` is 0 for seven days in a row |
| Preserved IDs, config and history | catalog ids unchanged (export compared by id); the sha256 list of the presets and config files is equal before and after the switch; workload names and operation ids unchanged; the Python DB snapshot and each agent journal snapshot exist with a sha256 |
| Observed serving state matches | on each host: signature before = signature after = agent record; no engine restart is caused by the switch |
| Reversal restores the previous control paths | R1 and R2 pass twice on temporary state (M6c); the live R1 drills of M6f pass; the state tables in "Reversal" are complete for all three hosts |

## Facts that are not verified

- The content of `~/.config/llmctl/settings.yaml` (not read: it holds a
  token). The plan takes `llmctl_next.enabled: true` and the six `units:`
  lines from the progress file.
- Which llm-router commit the running process has. Only the checkout was
  read.
- If `origin` has moved. No `git fetch` was run.
- If mbp2's `llama-swap` is agent-owned now.
- The last use of each unowned lane (`journalctl` gave no entry).
- The cold load times of `flash-uncensored`, `glm53`, `dsv41` and of
  `gpt-oss-120b` in the llama.cpp router.
- If a binary with the old bundled SQLite can open a journal that the new
  one wrote. The M7 progress file says the format is backward compatible;
  no test ran on a copy of a live journal.
- Which verb records an intent without a restart.
- What `lms-stray-watch` does, and the scripts of rows W11, W13 and W15.
- The consumers of port 8088.
- The M5e leftover (TUI Run on presets, Catalog): inferred from the
  source, not seen live.

## Differences between the documents and the live state

1. `RUST-M5-PROGRESS.md` says that `hf` on m5 is open. `hf` is installed
   there.
2. The M5 plan says the pilot lines go into `docs/RUST-M5-PILOT.md`. That
   file does not exist. The days are rows named "M5b clean day" in the
   progress file.
3. The M5 pilot rule says a day counts when the owner used llmctl-next on
   that day. Day 1 is counted, and the progress file says that no agent
   operation ran on any host in those 24 hours.
4. The M5 exit checklist has the row "`gpt-oss-120b` served by the router
   through llm-router". The owner decided on 2026-11-22 to switch after
   the M5 exit. The row needs an accepted-gap note.
5. The workflow matrix names inventory revision 43 and six workloads. The
   live values are 48 and seven.
6. The flash runbook says the branch is not pushed. The local
   remote-tracking ref `origin/fix/run-rollback-order` equals its tip. The
   M7 progress file says the same of `62b94a7`, which is in `origin/main`.
7. The workstation agent was replaced at 23:36 PST 2026-11-20. That is
   about 11 hours after the pilot count started (12:52 PST). The rule "no
   redeploy before the M5 exit" is dated 2026-11-21, thus no rule was
   broken, but pilot day 1 covers two builds.
8. The M5 plan says that M6 removes the `ds4-llmctl-sync` hooks from the
   six units. This plan moves the removal to after the M6 exit (design
   decision 5).

## Owner decisions

Decided 2026-11-22: the owner took each recommendation ("Take your
recommendations"). In each item, the recommended option is the decision.
The other option stays here as the record of what was not chosen.
Vault: ADR-068.

1. **Split M6 into M6a–M6g** as above, a check at the end of each, about
   9–12 working days plus the 7-day window. *Recommended.*
2. **M5d step 7:** after the M6 exit (recommended: it has four open
   blockers, three of them are faults in programs that M6 quiesces, and
   its one-week way back would overlap the M6 window; `gpt-oss-120b`
   stays its own agent-owned unit through M6), or before M6 (the cutover
   then freezes the final topology, but M6 starts 1–2 days later and
   legacy programs get fixes that M6 makes obsolete).
3. **CLI default data dir:** change the default to
   `~/.local/state/llmctl-next`, the layout of the Macs and of M7, and on
   the workstation make that path a symlink to the `llmctl-next-m4a` dir
   (recommended: a plain `llmctl-next status` then works on every host,
   and no data moves), or keep `llmctl-next-dev` as the default and set
   `LLMCTL_NEXT_DATA_DIR` in each shell profile (no code change, but a
   call without the variable does not reach the agent), or move the
   workstation data
   dir now (more risk in the cutover; the M7 plan puts it later). Risk of
   the recommended choice: a test or script that calls `llmctl-next` with
   no data dir then reaches the live agent. M6a must search for such
   calls.
4. **Command name:** keep `llmctl-next` for the Rust tool and `llmctl` for
   the Python tool through M6 (recommended: scripts call `llmctl
   take-gpus`, `llmctl run` and `llmctl export-registry`; a rename breaks
   the reversal), or rename at M6.
5. **Python llmctl and port 8088 after the switch:** keep the Python
   llmctl installed and in cutover mode; stop and disable `llmctl-api` and
   `llmctl-reconcile.timer` (recommended: the dashboard would show session
   rows that no program updates), or keep `llmctl-api` running as a
   read-only view.
6. **Unowned units:** adopt `flash-uncensored`, `glm53` and `dsv41`, each
   with a live round trip, and adopt `vllm-engine2` as a stopped workload
   (recommended, unless the owner names a lane that is no longer used), or
   retire the lanes that are not used: disable the unit and remove the
   router backend (cheaper, no window).
7. **`qwen3coder30b` (B4):** build the vendor `load` verb, about 1.5 days
   (recommended: it closes the gap for every Ollama and LM Studio model
   and gives the router one handoff path), or change
   `qwen3coder30b-start` to call `llmctl-next run ollama-gpu1` and keep
   its own preload request (half a day; the unit has one authority, the
   model load stays outside the planner).
8. **Healers:** report-only for agent-owned workloads; the agent's bounded
   recovery is the only retry loop (recommended: the parent forbids two
   retry loops on one workload), or the healers call `llmctl-next
   restart` (a second retry policy, but repairs continue when the agent's
   budget is spent).
9. **Python history:** `history import` into each owner host's agent as
   read-only rows with a report, about 1.5 days (recommended: the parent
   asks for preserved ids and history and for a migration report), or keep
   only the frozen DB snapshot with its sha256 (no code; the history is
   then readable only with SQLite tools).
10. **Host order:** switch mbp2, then m5, then the workstation, each on
    its own day (recommended), or all three in one window.
11. **Live reversal drill:** run R1 live on mbp2, and R1 for the settings
    on the workstation, during M6f (recommended: the exit criterion names
    the reversal, and neither drill restarts an engine), or accept the
    rehearsal on temporary state as the only proof.
12. **Acceptance window:** seven clean days in a row, a bad day restarts
    the count (recommended: the M5 rule), or three days.
13. **Version of the M6 build:** raise the workspace version to 0.2.0, so
    `host list` shows which hosts run the cutover build (recommended), or
    keep 0.1.0 and identify builds by sha256 only.
14. **M7 merge:** after the M6 exit, which is the end of the acceptance
    window (recommended: a fix or a reversal in the window then deploys
    from a `main` without M7), or directly after the M6f switch (M7
    reaches the hosts one week earlier).
15. **Removal of the legacy tools** (Python llmctl, `llmctl-api`, the
    `ds4-llmctl-sync` hooks and script, the lane scripts' old paths):
    not in M6; a separate owner approval after the M6 exit (recommended:
    R2 needs them), or a fixed date now.
16. **llama-swap request and TTL swaps on the Macs:** accept them as an
    exception to "one authority" and count them in their own audit class
    (recommended: llama-swap is the Macs' scheduler by the M5 decision 8),
    or turn request loading off and load lanes only by verb.

**Owner actions that M6 needs:** approve each window (M6e, M6f on the
workstation); approve each edit of `settings.yaml`; free RAM before the
`flash` and `dsv41` round trips.
