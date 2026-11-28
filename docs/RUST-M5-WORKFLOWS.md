# M5 workflow matrix — what a day looks like, per host

Status 2026-11-27: **close-out walk done**, see
[Close-out walk](#close-out-walk-2026-11-27) at the end. Of 28 rows, 13 pass with no gap. The owner
decided the six gaps on 2026-11-27: gaps 1, 4 and 5 are built (three
verbs, on main since `6d1ff10`, not deployed), gaps 2, 3 and 6 are
accepted. No open gap is left. The tables below are the 2026-11-18 text
and are not changed.

Status 2026-11-18: written in M5a from the facts in
[RUST-M5-PLAN.md](RUST-M5-PLAN.md) (checked read-only the same day) plus a
read of the tools named below. Every row names the llmctl-next path that
exists today or the M5 phase that closes the gap. The exit check walks this
table: *every row has a passing llmctl-next path on its host, or an
owner-accepted gap.*

Sources read: `llmctl/cli.py` (Python CLI surface), `llmctl/tui/`
(screens), `~/local-llm/bin/llmctl` on m5 (Mac CLI, usage block and
`cmd_*` functions), `~/AI/packages/llmctl-stack/llmctl_stack/take_gpus.py`,
`~/AI/tests/test_llm_heal.py` + `test_mac_heal.py` (healer behavior since
M4), the live inventory `~/.local/state/llmctl-next-m4a/inventory.json`
(revision 43, six managed workloads), and `llmctl-next --help` verbs.

## Legend

- ✅ path exists in llmctl-next today
- ◐ partial (named limitation) · ❌ gap → the M5 phase that closes it
- Hosts: **WS** workstation, **m5**, **mbp2** (Mac agents enrolled, protocol 1)

## Daily reads (no writes)

| # | Workflow | Today | llmctl-next | Status |
|---|---|---|---|---|
| R1 | One look at the whole day: workloads, readiness, GPUs | Python TUI `d` (`llmctl tui`, 111 uses), `llmctl status` (72) | `status` / `status --all` (all three hosts, freshness) | ✅ CLI; TUI Hosts screen M5e |
| R2 | Health of engines/endpoints | `llmctl health` (106) | `status` readiness + `vram` | ◐ no aggregated pass/fail yet; TUI M5e |
| R3 | Sessions / what ran | `llmctl sessions`, `:8088` dashboard | `ops`/`op` (durable operations) | ◐ Python session rows (incl. adopted ones from `ds4-llmctl-sync`) stay in the Python DB until M6 |
| R4 | GPU/memory occupancy | `llmctl status`, `nvidia-smi` | `status` per-card views (enrolled vs foreign), `vram` | ✅ |
| R5 | Mac memory budget, loaded lanes, swap pressure | Mac `llmctl status`/`memory`/`budget` | `lanes memory`, `lanes status` | ◐ `lanes status` exists; budget verb parity folded into M5c item 4 |
| R6 | Logs of a workload/lane | `journalctl`, Mac `llmctl logs [-f]` | `logs [--follow]`, `lanes logs` | ✅ |
| R7 | Foreign writes since yesterday | none (invisible) | `audit [--since]` | ❌ M5a (this phase) |

## Switching and lifecycle (writes)

| # | Workflow | Today | llmctl-next | Status |
|---|---|---|---|---|
| W1 | Switch the default vLLM profile | TUI Run screen, `llmctl swap/vllm` (11/24), `ccl` via llm-router | `run --profile-env` (planned, confirmed, rolled back); llm-router hands the three workloads over already | ◐ `run <preset>` by name ❌ M5c item 1 |
| W2 | Load/stop a lane (Strata, gpt-oss) | llm-router switch path | `lanes swap/stop` (admitted, confirmed by llama-swap) | ✅ (remaining router backends hand over in M5b item 4) |
| W3 | Stop a workload for a while (must stay down) | `llmctl stop` (4) | `stop` with manual-stop protection (survives eviction/rollback) | ✅ |
| W4 | Restart/unload a model | `llmctl` restart, Ollama evictions | `restart`, `unload` with in-flight safety | ✅ |
| W5 | Start a lane/model on a Mac | Mac `llmctl start MODEL` | `lanes swap` on the Mac agents | ◐ budget guard as write-blocker M5c item 4 |
| W6 | `take-gpus` before a heavy job | `llmctl-stack take-gpus <lane> --take` (stops units itself) | owned units stop through llmctl-next (M5b, 2026-11-18); the `qwen3coder30b-start` script's own `systemctl start ollama-gpu1` still writes directly (B4 gap) | ◐ |
| W7 | gpt-oss-120b start/stop | own unit, llm-router `gptoss` backend | `run/stop gpt-oss-120b` (owned) | ✅ until it moves into the llama.cpp router (M5d step 7) |

## Config, presets, catalog (writes)

| # | Workflow | Today | llmctl-next | Status |
|---|---|---|---|---|
| C1 | Edit/list presets | `llmctl presets` (14) | — | ❌ M5c item 1 (same files, comments kept) |
| C2 | New model into the catalog (download) | `llmctl model add/hf get` | `model get`, `stores add` | ◐ list/show/add/edit/scan ❌ M5c item 3 |
| C3 | Remove a model from disk+catalog | `llmctl delete-model` | `model rm` | ✅ |
| C4 | Router aliases | `llmctl aliases/set-alias` | — | ❌ covered with C1 (M5c: `~/.config/llmctl/router_aliases.json` read/write) |
| C5 | Model dirs | `~/.config/llmctl/model_dirs.yaml` | `stores` | ◐ parity with the YAML file in M5c item 3 |
| C6 | Mac registry (models.yaml) | Mac `llmctl download/discover/list/info` | lane store per workload + `lanes import` | ◐ catalog verbs M5c item 3; `hf` on m5 (decision 7) M5c |

## Background writers (things that act without the owner)

| # | Actor | Today | M5 shape |
|---|---|---|---|
| B1 | Python llmctl CLI/TUI/`:8088` | `sessions/swap_jobs/units` → systemd directly for non-handed-over workloads | guard in `SystemctlRunner.run` (2026-11-18): owned workload → llmctl-next `--wait`, refusal (not fallback) when the agent is down; reversal `llmctl_next.enabled: false`. Enabled after the M5a baseline day |
| B2 | llmctl-stack `take-gpus` | `systemctl --user stop` / `llm-unit stop` / `sudo -n systemctl stop` | `stop_blockers` delegates owned units to llmctl-next stop (2026-11-18, `26e52b8d` in llmctl-stack); same settings section; refusal, not fallback |
| B3 | Mac llmctl (m5, mbp2) | `llama-swap` loads through its own path (`start/stop` on the router) | `next_enabled: 1` routes owned lanes through `llmctl-next lanes swap/stop` (2026-11-18, `5dd7415` in local-llm); agent down = refusal with the `next_enabled: 0` lever. mbp2 stays off: its llama-swap workload is not agent-owned yet |
| B4 | llm-router | own switch path for not-handed-over backends | Remaining owned-unit backend: `qwen3coder30b` (→ `ollama-gpu1`). **Not handed over**: `run ollama-gpu1` starts the service but does not preload `qwen3-coder:30b`, which the router's readiness waits for — a handover would break the lane. Owner-accepted gap until a run can preload an Ollama model (M5c revisit) |
| B5 | llm-heal / mac-heal | asks the agent's `intent` before repairs (since M4), then repairs through systemctl/launchctl itself; fail open | repairs that go through the agent are operations; direct repairs show as findings in `audit` (M5a) — full delegation is not in M5 |
| B6 | llmctl-warm | never loads a lane (M4c); `--guarded` no-ops when the router is down | no change needed in M5 |
| B7 | `ds4-llmctl-sync` | records adopted rows in the Python session table only; starts/stops nothing | kept all M5; retires at M6 with the Python reader |
| B8 | `llmctl-reconcile.timer` | writes session rows only | not a lifecycle write; untouched |

## M4 carry-overs folded into this matrix

1. **Enroll an unadoptable workload / change authority by verb** — today
   `adopt` refuses what it cannot prove; the verb pair lands in M5c item 5.
   Rows W2/W7 gain models with no systemd unit of their own.
2. **Llama-swap matrix verb** (several lanes loaded on purpose, budget as a
   write guard) — M5c item 4; row W5/R5.
3. **Profiles with the VRAM fit** — `run --profile-env` plans against the
   fit since M4c; preset parity (M5c item 1) makes it the default path for
   W1.
4. **Release archives on launchd** — recipe exists; a real launchd install
   happens with llama.cpp on the Macs (M5d item 6).
5. **Per-lane in-flight** — `lanes swap` counts router in-flight since M4c;
   the llama.cpp router models get the same safety in M5d item 2.

## What the pilot day uses (M5g)

A counted day touches, at minimum: one read row (R1/R4/R5/R7), one write
row (W1–W5), and `audit` clean (R7) on every host that day. Rows still ❌
at pilot time must be owner-accepted gaps and say so here.

## Close-out walk 2026-11-27

The M5 pilot ended at 20:52 UTC 2026-11-27 with seven counted days
([pilot log](RUST-M5-PILOT.md)). This section walks each row of the
tables above. Each result is one of:

- **pass**: the llmctl-next path exists on its host and has evidence.
- **gap, accepted**: an owner decision already covers it. The decision is
  named.
- **gap, open**: no owner decision covers it. The owner must accept it or
  ask for the work.

All three agents run build `ccabf75` (workstation sha256 `208bdf1c…`, Macs
`f559fbc8…`, protocol 1). Each check on 2026-11-27 was read-only: `status
--all`, `audit --all`, `preset list`, `preset diff`, `model list`, `model
scan`, `stores`, `vram`, `intent`, `logs`, `lanes memory`, `lanes matrix`,
`lanes show`, `run … --dry-run`, `tui --read-only` and `tui --bench 5`.
Nothing on a host was changed.

### Daily reads

| # | Result | Evidence |
|---|---|---|
| R1 | pass | `status --all` shows the three hosts, each seen 0 to 1 s ago. The TUI Hosts screen shows the same 15 workloads on the three agents (seen live 2026-11-27, mbp2 included). |
| R2 | pass; gap built 2026-11-27 | Readiness per workload is in `status`, `engine` and the TUI. The verb `health [--all] [--json]` gives one pass or fail per host: a workload passes when the observation equals its recorded intent, and exit 1 means not healthy. It reaches the hosts with the M6d deploy. |
| R3 | gap, accepted | `ops` and `op` show the agent's operations (168 rows in the TUI). The Python session rows stay in the Python DB. M6 decision 9 imports them (`history import`, M6b item 7). |
| R4 | pass | `status` shows the enrolled and the foreign processes per card. `vram` shows the free memory and four measured workloads. The TUI GPUs screen shows the same for the three hosts. |
| R5 | pass | `lanes memory llama-swap` on m5: budget 104 GiB, 14 lane peaks, 7 set combinations fit, 3 `risky*` sets are over the budget and accepted. On mbp2: budget 46 GiB, no matrix. The row names `lanes status`; the verb is `lanes show`. |
| R6 | pass | `logs vllm-engine` shows the journal of the unit. `lanes logs` exists for a lane. |
| R7 | pass | `audit --all --since` is the evidence of each pilot day. After the pilot end it shows 0 foreign writes on the three hosts. |

### Switching and lifecycle

| # | Result | Evidence |
|---|---|---|
| W1 | pass in the pilot; **refused on 2026-11-27, gap, open** | The router switched `vllm-engine` through the agent on pilot days 2, 3, 4 and 6 and on 2026-11-27 (22:38 UTC, succeeded). `run <preset>` and the TUI Run screen on presets are in the deployed build. No pilot line shows a use of `run <preset>` by name. On 2026-11-27 at 23:42 UTC each run of `vllm-engine` is refused with `definition_drift` (finding 1 below). Finding 1 was resolved the same day: `run qwen27b-awq --dry-run` gives a plan (a preset switch by name). A plain `run vllm-engine` is still refused on `environment_files`. |
| W2 | pass for owned lanes; gap, accepted for the others | m5: 24 `LaneSwap` operations on pilot day 2. Workstation: `strata-fast`, `flash` (round trip on 2026-11-27) and `gpt-oss-120b` go through the agent. `flash-uncensored`, `dsv41` and `glm53` are not adopted: M6 decision 6, M6e item 1. |
| W3 | pass | A manual stop of `vllm-engine` and `gpt-oss-120b` held from 18:09 UTC 2026-11-24 through pilot day 5. No recovery and no healer started them. |
| W4 | pass | `restart` ran on `llamacpp-router` on pilot day 3. Router models were unloaded by verb on days 3 and 4. `unload` exists for Ollama and LM Studio. |
| W5 | pass on m5; gap, accepted on mbp2 | m5 has `next_enabled: 1` and its lane changes are agent operations. The memory budget guards a matrix write (e2e test); `lanes matrix` was read live on m5, and no matrix write was made live. mbp2 has no `next_enabled` line: M6f step 7 turns it on. |
| W6 | pass for the owned units; gap, accepted for the script | Drill 3 passed on 2026-11-21: `take-gpus` stopped `vllm-engine` through the agent. `qwen3coder30b-start` still starts `ollama-gpu1` with `systemctl`: row B4. |
| W7 | pass; gap, accepted for the router move | `run` and `stop` of `gpt-oss-120b` ran on pilot days 3 and 4. The move into the llama.cpp router (M5d step 7) comes after the M6 exit: M6 decision 2. |

### Config, presets, catalog

| # | Result | Evidence |
|---|---|---|
| C1 | pass | `preset list` shows five presets, `qwen27b` is `serving`, four are `ready`. `preset diff qwen27b` says `in sync; serving now`. `preset settle --write` ran live on 2026-11-18. The parity test holds 18 renders and 58 edits byte for byte against Python. |
| C2 | pass, with one open gap | The registry is imported on the three agents (54, 14 and 8 models). `model list` and `model scan` ran live on 2026-11-27. **Gap, open:** the catalog revision is still 1 on the workstation, thus `model add`, `model edit` and `model scan --add` were not used live. They have e2e tests. |
| C3 | pass, with one open gap | `model rm` is in the build since M4 and has tests. **Gap, open:** no pilot line shows a live use. |
| C4 | gap built 2026-11-27 | `alias list` and `alias set` read and write `~/.config/llmctl/router_aliases.json` over settings.yaml `router.aliases`, with the lock file and the bytes of Python `set-alias`. The file last changed on 2026-10-13. Its consumer, the Python gateway (`llmctl-gateway`, port 9000), does not run. The file is not a managed resource in the M6 plan. |
| C5 | gap built 2026-11-27 | `stores` shows five stores with their free space. `stores roots` reads `model_dirs.yaml`, shows each root with the stores that cover it, and does the broken-symlink check of Python `llmctl validate`. That file has only roots with an empty `runtimes` list. |
| C6 | pass, with one open gap | The Mac catalogs hold the imported rows. `hf` is installed on m5 (`/opt/homebrew/bin/hf`, 2.4.0, since 2026-11-16; the progress log says it is open). Lane stores exist on both Macs (`lanes show`: 2 lanes on mbp2, revision 5; 14 lanes on m5, revision 11). **Gap, open:** `model get` was not used live on a Mac. |

### Background writers

| # | Result | Evidence |
|---|---|---|
| B1 | pass, with finding 4 | The guard is on since 2026-11-20 (`llmctl_next.enabled: true`, read through the guard's own parser). Seven days with 0 legacy writes. The unit map has seven units and does not name `llamacpp-router.service` (finding 4 below). |
| B2 | pass | Drill 3, 2026-11-21. |
| B3 | pass on m5; gap, accepted on mbp2 | As W5. |
| B4 | gap, accepted | The `load` verb is in the deployed build. The router backend `qwen3coder30b` has no `llmctl_next` key. M6 decision 7, M6e item 3. |
| B5 | pass as designed; gap, accepted | The healers ask `intent` first. A direct repair shows in `audit`. M6 decision 8 makes them report-only. |
| B6 | pass | No change in M5. M6a item 3 reads `llmctl-warm` again. |
| B7 | pass | Kept. M6 makes it a no-op with a flag file. |
| B8 | pass | `llmctl-reconcile.timer` is active and writes session rows only. M6 stops it. |

### M4 carry-overs

| # | Result | Evidence |
|---|---|---|
| 1 | pass | `enroll` and `authority` are in the build (`records_e2e`). |
| 2 | pass, no live write | `lanes matrix` shows the m5 matrix (14 vars, 8 sets). The budget guards the write in the e2e test. |
| 3 | pass | `run --profile-env` plans against the fit. `run <preset>` is the same swap. |
| 4 | pass | Both Macs have the macro `llamacpp` on the staged release `b12527`. Each Mac answered one request from a llama.cpp lane on 2026-11-19. |
| 5 | pass | Built in M5d item 2 (progress log). Not checked again here. |

### M5 exit checklist

| Criterion | Result |
|---|---|
| Minimal multi-host TUI | **pass.** Six of the seven screens seen live against the three agents (Hosts, Run, Operations, GPUs/memory, Catalog, Audit; not Logs). `tui --bench 5`: key to frame p50 0.35 ms, hosts refresh p50 263 ms, p95 472 ms. |
| Config/preset parity | **pass for parity; gap, open for "used in the pilot".** No pilot line shows `run <preset>` by name. |
| Catalog operations | **pass for import and export; gap, open for live `add` and `rm`.** `scan` ran live on 2026-11-27. |
| llama.cpp on every host, several models | **pass, with one accepted gap.** Router models were loaded and unloaded by verb on the workstation. Each Mac answered from a pinned llama.cpp lane. `gpt-oss-120b` in the router: M6 decision 2. |
| Migration/reversal rehearsal | **pass.** Both directions passed two times on 2026-11-27, 33 of 33 checks each ([RUST-M5-REHEARSAL.md](RUST-M5-REHEARSAL.md)). The owner asked for it before M6b. It found one reversal fault, fixed on the same branch (`import-registry --update-models`). M6c extends it. |
| Daily workflows across hosts | **this walk.** Of 28 rows, 13 pass with no gap. 12 have only accepted gaps (R3, W2, W5, W6, W7, B3, B4, B5, and since 2026-11-27 W1, C2, C3, C6). 3 had an open gap that is now built and not yet deployed (R2, C4, C5). |
| No legacy lifecycle writes | **pass.** Pilot log, seven days. |
| Seven consecutive days | **pass.** Pilot log. |

### Gaps that need the owner's acceptance

**Owner decision, 2026-11-27.** The six gaps were put to the owner with
the words "accept all, or name the numbers you want built". The answer
was "Build 1, 4, and 5". Thus gaps 1, 4 and 5 are built (below), and gaps
2, 3 and 6 are accepted as proposed. No open gap is left. Later the same
day the owner changed gap 6 from accepted to done: see item 6.

1. **R2:** no aggregated health verb. Proposed: accept; use `status` and
   the TUI, and Python `llmctl health` for the one-line answer.
2. **W1:** `run <preset>` by name has no pilot use. Proposed: accept, and
   use it one time in the M6 acceptance window. First resolve finding 1.
3. **C2, C3, C6:** `model add`, `model edit`, `model rm` and a Mac `model
   get` were not used live. Proposed: accept; the M6 acceptance window
   uses each one time.
4. **C4:** router aliases stay in the Python tool. Proposed: accept; it
   is a config file that no agent record names.
5. **C5:** `model_dirs.yaml` stays a Python-only file. Proposed: accept;
   it feeds only the Python broken-symlink check.
6. **Rehearsal (M5f):** done in M6c. Proposed: accept; M6c runs both
   directions two times before any live step. **Changed by the owner on
   2026-11-27 ("Do a rehearsal before M6b"): the M5-level rehearsal ran
   the same day and passed, see [RUST-M5-REHEARSAL.md](RUST-M5-REHEARSAL.md).
   The gap is closed, not accepted.**

### Live checks required to close M5

The owner asked on 2026-11-27 to start M6b and close M5 after gaps 2 and 3
pass. Run both checks on the existing agents before the M6d deploy. Record
the date, host and result for each check in [RUST-M5-PROGRESS.md](RUST-M5-PROGRESS.md).

- [ ] **Gap 2 (W1):** In an approved workstation window, record the serving
  model and intent. Preview a named preset switch and the return to the
  original preset. Run the named switch with `--wait`, verify readiness
  and restore the original preset. Record both operation IDs and compare
  the final serving model, port and intent with the initial state.
- [ ] **Gap 3 (C2, C3, C6):** Use disposable inputs for live `model add`
  and `model edit`. Confirm that `model rm` targets only disposable
  weights before applying its removal token. Run `model get` for a small
  model on a Mac in a declared store. Record the catalog IDs, affected
  paths, results and cleanup.

After both checks pass, mark M5 done in its plan and the parent plan.

### Built for gaps 1, 4 and 5 (2026-11-27)

Three verbs of the `llmctl-next` CLI (branch `m5-gap-verbs`, merged into
main as `6d1ff10`). They add
no API route and no wire field, thus the protocol stays 1. They are **not
deployed**: they reach the hosts with the M6d deploy.

| Gap | Verb | What it does |
|---|---|---|
| 1 (R2) | `health [--all] [--json]` | One line per host: `healthy` or `UNHEALTHY`, and one `FAIL` line per workload whose observation does not equal its recorded intent. Intent `unmanaged` is skipped. With `--all`, a host that does not answer fails. Exit 0 = healthy, 1 = not. |
| 4 (C4) | `alias list [--json]`, `alias set <alias> <target>` | The merged view of Python (`router.aliases` of settings.yaml, else the seven default roles, overlaid by `router_aliases.json`). A target is `healthy` when a workload that is running and ready has it as its name or as its served model. `alias set` takes the lock file of Python `set-alias` and writes the same bytes; `-` removes a binding. A file that is not a JSON object is refused (Python reads it as empty and then loses its aliases). |
| 5 (C5) | `stores roots [--json]` | Each enabled root of `model_dirs.yaml` with its path (resolved as Python does), the declared stores above or below it, and each broken symlink under it (the same depth limit as Python). Exit 1 when a broken symlink exists. |

Limits. `alias` and `stores roots` read files on the host where the
command runs; with `--host` they refuse. `alias` has no live consumer
today: the Python gateway that resolves aliases does not run.

Read-only check with a debug build against the live workstation agent
(2026-11-27, no write):

- `health`: `host workstation: UNHEALTHY (2 of 8 fail; 6 ok, 0
  unmanaged)`, with `llmster` and `llamacpp-router` (intent running,
  observed Stopped, `gave_up`). This is finding 3 below.
- `alias list`: seven aliases; `coding` is bound, `long-context` and
  `tutoring` are healthy, four roles are unbound. Python `llmctl aliases`
  gives the same seven rows.
- `stores roots`: five roots, 0 broken symlinks. Python `llmctl validate`
  also reports 0 findings.

### Findings of the walk

These are live facts from 2026-11-27. No fix was applied.

1. **`vllm-engine` cannot be started through the agent now.** `run
   vllm-engine --dry-run`, `run qwen27b --dry-run` and the router's form
   (`run vllm-engine --profile-env … --model …  --dry-run`) each answer
   `409 definition_drift: … differing: environment_files,
   version.lock_digest`. Cause: at 15:23 PST 2026-11-27 an editable
   `pip install --user` of `agentkit` (from a worktree below
   `~/.custom-harness/worktrees/`) wrote `anyio 4.16.0`, `httpx 0.29.0`,
   `httpcore`, `h11` and `typing_extensions` into
   `~/.local/lib/python3.12/site-packages`. The `vllm-muse` interpreter is
   Python 3.12.14 and has the user site on. Its `pip freeze` now lists
   those packages, thus the lock digest is not the pinned one. The same
   packages come before the env's own `anyio 4.15.1` and `httpx 0.28.1` on
   the next start of the engine. The engine that runs now (pid 4022288,
   started 14:38 PST) is not changed. Effect: the next router switch to a
   vLLM profile is refused, and `llmctl_next_refusal_final` is `true`.
   The `environment_files` difference alone is a content difference that
   a profile switch accepts.

   **Resolved 2026-11-27 16:0x PST (owner: "Remove packages").** The
   directory `~/.local/lib/python3.12` and the script `~/.local/bin/httpx`
   were moved to `~/.local/lib/python3.12.removed-2026-11-27` (nothing was
   deleted; no process had the files open). After the move: `version
   vllm-engine` shows no drift and the live lock digest equals the pin
   (`635b205d…`). The router's form (`run vllm-engine --profile-env …
   --model … --dry-run`) and a switch to another preset (`run qwen27b-awq
   --dry-run`) give a plan. The serving signature digest is the same as
   before (`2e267a06…`) and `audit` shows 0 foreign writes. What stays:
   `run vllm-engine --dry-run` and `run qwen27b --dry-run` (the preset
   that serves now) still answer `definition_drift … environment_files`.
   The pinned digest of `vllm-engine.env` is older than the file (the
   profile envs were edited at 19:50 PST 2026-11-24), and only a profile
   switch accepts a content difference (`ops.rs:349`). It can come back:
   the longjob `ch-portfolio-m4b` (custom-harness) runs until about 18:25
   PST and the first run of that job made the install.
2. **An LM Studio server runs outside its unit on the workstation.** A
   process `llmster` (pid 3820672, started 18:43 PST 2026-11-26, in
   `session-3.scope`) listens on 127.0.0.1:1234. `llmster.service` is
   inactive and the agent's record says Stopped. The agent reads the
   loaded models from that port.
3. **The pilot notes say `llmster` stays down by owner choice.** Its
   recorded intent is `running` and its recovery state is `gave_up`. The
   same holds for `llamacpp-router`. The record does not say what the
   owner decided.
4. **The Python guard's unit map does not name `llamacpp-router.service`.**
   The agent owns that workload. A Python `llmctl` write to that unit
   goes to `systemctl` directly.
