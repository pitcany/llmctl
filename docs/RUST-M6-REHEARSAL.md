# Rust successor — M6c rehearsal (the cutover and both reversals)

Status 2026-11-28: **passed twice in a row from a fresh container**,
73 of 73 checks each run, at llmctl `__SHA__`. The rehearsal found three
facts that change the M6f procedure or need an owner decision (findings
1 to 3). No live host changed.

Plan text: [RUST-M6-PLAN.md](RUST-M6-PLAN.md), M6c, and the "Reversal"
section. M6c extends the M5f rehearsal
([RUST-M5-REHEARSAL.md](RUST-M5-REHEARSAL.md)). M6f runs the procedure at
the end of this file.

## How to run it

```bash
scripts/rehearsal_m6c.sh --runs 2
```

Options: `--runs N` (fresh containers, one after the other; default 2),
`--keep` (keep the last container), `--agent-bin FILE` (a static
`llmctl-next`; default: build one from `HEAD` in `rust:1.100.0-alpine`
into `rust/dist/m6c-<sha>/`), `--dev` (take `scripts/rehearsal_m6c.py`
from the working tree; never for a recorded run). Environment:
`LLMCTL_M6C_AI_REF`, `LLMCTL_M6C_ROUTER_REF`, `LLMCTL_M6C_LOCAL_LLM`
(`host:repo`, default `m5:local-llm`), `LLMCTL_M6C_LOCAL_LLM_REF`. Exit 0 =
every check of every run passed. One run takes about 2 minutes, of which
60 s is the drill with the agent stopped. The log goes to
`rust/dist/m6c-rehearsal-<sha>.log`.

## What the rehearsal touches

- On the host it reads: the committed sources (`git archive`, never a
  working tree) of llmctl, `~/AI` (`bin`, `heal`, `services`,
  `packages/llmctl-stack`), llm-router and local-llm (over ssh from m5);
  the Python DB through the SQLite backup API (source opened read-only);
  the preset files. It reads no `settings.yaml`, no env file and no token.
- It builds the image `llmctl-m6c-rehearsal`: `python:3.12-slim-bookworm`
  (Debian 12) with systemd as PID 1, a lingering user `tester`, the Python
  llmctl and llm-router of those commits in `/opt/venv`, and the static
  agent in `/usr/local/bin`. The image holds the DB copy. It stays local.
- Each run starts a fresh container with `--network none`, a private
  cgroup namespace and no host D-Bus. Every service-manager call runs in
  the container. The container is removed after the run.
- In the container the script uses the real default paths of `tester`, as
  on the workstation: `~/.config/llmctl/settings.yaml`, the Python DB,
  `~/.local/state/llmctl-next`, the host lock
  `~/.local/share/llmctl/operations.lock`, `~/AI/bin/llm-unit`,
  `~/AI/services/lanes/_unit.sh`, `~/.config/systemd/user`.

The stand-ins:

| Real thing | Stand-in in the container |
|---|---|
| `vllm-engine` (TP 2, cards 0 and 1) | user unit `vllm-engine.service`, a Python HTTP server that serves `VLLM_SERVED_NAME` from its env file; workload `reh-engine` |
| a GPU lane | user unit `reh-lane.service`, the same server; workload `reh-lane` on card 2 |
| two presets on the engine | `reh-a`, `reh-b` (served `reh/model-a`, `reh/model-b`), plus the five real preset files |
| `llmctl-reconcile.timer`, `llmctl-api.service` | units that run `true` and `sleep` |
| llm-router | the real switcher of `ca42338` through `python -m llm_router.cli switch` (no HTTP server) |
| `llm-heal` | the real `llm-heal.py run` with a rehearsal `heal.yaml` |
| llama-swap on a Mac | a Python HTTP server for `/health`, `/v1/models`, `/running` |
| `nvidia-smi`, `journalctl`, `tailscale`, `sudo`, `launchctl` | scripts in `~/m6c/bin` that do nothing (`launchctl` and `sudo` record each call) |

The agent runs as the user unit `llmctl-next.service` with
`LLMCTL_NEXT_LANE_LEASE=off` and `LLMCTL_NEXT_ROUTER=off`. The host lock
is on: the agent, Python llmctl, llm-router and `llm-heal` share it.

## The start state (the M5 end state)

| Check | What it proves | Result |
|---|---|---|
| S1, S2 | The agent answers; both units serve | pass |
| S3, S4 | `adopt` of both units, every preview check `pass` | pass |
| S5 | `preset sync reh-a` and `reh-b` write the profile envs | pass. Two vLLM workloads need `--workload reh-engine` |

## Forward: the cutover

| Check | M6f step | What it proves | Result |
|---|---|---|---|
| F1 | 1 | Signature "before" agrees with the agent | pass (`sha256:b5ffbca9…`) |
| F2 | 2 | The host lock is held (`flock`) | pass |
| F3 | 3 | Quiesce: the timer and `llmctl-api` stopped and disabled | pass |
| F4 | 3 | `ds4-llmctl-sync` with the flag file: exit 0, nothing done | pass |
| F5, F6 | 4 | `sqlite3 <db> ".backup <copy>"`; `export-registry` bundle | pass: 74 models, 2 profiles |
| F7 | 4 | sha256 list of the presets, `settings.yaml`, `router.yaml`, profile envs, owned unit files | pass: 14 files |
| F8 to F10 | 5 | `catalog import --as <host>`: dry run, `--confirm`, a second dry run — all while the lock is held | pass: 54 rows, then 54 `present`; 20 unmapped |
| F11 to F14 | 5 | `history import` dry run and `--confirm` of the `.backup` copy; `history list`; no `-wal` beside the copy | pass: 34 rows `py:<host>:<id>` |
| F15 | 5 | `catalog export` gives back the bundle's rows of this host, by id | pass: 53 of 53 |
| F16 | 6 | Signature equals the agent record and "before" | pass |
| F17, F18 | 7 | `llmctl_next.mode: cutover`; `switch.llmctl_next_only: true` validates; `llm-unit mode` = `cutover` | pass |
| F19, F20 | 8 | Signature "after" = "before"; no engine pid changed; no preset, profile env or unit file changed | pass |

The 54 rows are `no_store` because the container declares no store. On
the live agent the 2026-11-19 import gave 53 `new` and 1 `no_store`.

## Forward: the negative drills (M6f step 9)

| Check | Drill | Result |
|---|---|---|
| N1 | `LLMCTL_NEXT_DISABLED=1 llmctl start-unit <stopped enrolled unit>` | pass: exit 1, "cutover refuses LLMCTL_NEXT_DISABLED=1", nothing starts |
| N2 | Python `start-unit` of the stopped enrolled lane | pass: one agent operation, the lane serves |
| N3 | Python `start-unit` of a unit that is not enrolled | pass: exit 1, names the `llmctl-next` verb |
| N4 | `llm-unit start` of a unit that is not enrolled | pass: exit 5, nothing starts |
| N5 | A lane script's `lane_unit start` of the stopped lane | pass: through `llm-unit` to the agent, one operation |
| N5b | `lane_unit start` of the **running** lane | information: the lane restarts (finding 2) |
| N5c | ...the engine on other cards keeps its pid | pass |
| N6, N7 | llm-router switch to `reh-b` with `llmctl_next_only` | pass: one `llmctl-next run reh-engine --profile-env … --model reh/model-b`; the active env is the profile |
| N8 | Agent stopped: a router switch | pass: exit 1, nothing changes |
| N9 | Agent stopped: Python `start-unit` | pass: refused, nothing starts |
| N10 | The agent back after one minute; the engine kept its pid | pass |
| N11 | A crash (`kill -s KILL`) of the owned lane; `llm-heal run` with `llmctl_next_report_only: true` | pass: held, "report only: reh-lane is owned by llmctl-next, whose recovery is the only retry loop" |
| N12 | A raw `systemctl --user stop` of the enrolled lane | pass: `audit --unexplained` has one new finding |
| N13b | The same raw stop right after an agent stop and run | information: no finding (finding 3) |
| N14 | Mac `llmctl start <lane>` with `next_only: 1` and no agent answer for the llama-swap workload (the Mac bash script of `19d46dc`, a real registry lane) | pass: exit 1, "refusing to touch llama-swap directly", no `launchctl` call |
| N15, N16 | One preset switch by the agent (`run reh-a --workload reh-engine --wait`); drained | pass |
| N17 | Signature agrees with the agent | pass |
| N18, N19 | Catalog edits: `model add`, `model edit` of an imported row | pass |

## R1: back to the delegate state

| Check | What it proves | Result |
|---|---|---|
| R1.1, R1.2 | `mode` removed, `llmctl_next_only: false`, `refusal_final: true`, healer repairs on, timer and `llmctl-api` on, flag file removed; `llm-unit mode` = `delegate` | pass |
| R1.3 | `catalog export`, Python `import-registry --update-models` | pass: "Imported 1 models, updated 1 models" |
| R1.4, R1.5 | Python `reconcile`; no pid changed | pass |
| R1.6 | Delegated write, Python: `start-unit` of the stopped lane | pass: one agent operation |
| R1.7 | Delegated write, `llm-unit restart` of the lane | pass: one agent operation |
| R1.8 | `lane_unit` uses `systemctl` again | pass |
| R1.9 | Delegated write, llm-router switch to `reh-b` | pass: through the agent |
| R1.10 | `ds4-llmctl-sync` runs again | pass |
| R1.11 | A crash of the lane: `llm-heal` restarts it | pass |
| R1.12, R1.13 | Python `status` shows the engine serving `reh/model-b`; drained | pass |

## R2: full reversal

| Check | What it proves | Result |
|---|---|---|
| R2.1 | Signature before R2 agrees with the agent | pass |
| R2.2 | `catalog export` (agent still up), then `llmctl_next.enabled: false`, router `llmctl_next_only: false` and `refusal_final: false`, agent stopped and disabled: the engines keep pid and model | pass |
| R2.3, R2.4 | Python `import-registry --update-models`; Python sees the added model and the edited row | pass |
| R2.5 | Python `reconcile`; `status` shows the engine serving `reh/model-b` | pass |
| R2.6 | llm-router on its own path: env copy and `systemctl --user restart` | pass: serves `reh/model-a` |
| R2.7 | Legacy Python `run reh-b --direct` | pass: serves `reh/model-b` |
| R2.8 | `llm-unit restart` of the lane: plain `systemctl` | pass |
| R2.9 | The agent journal and `deploy-backups` are kept | pass |

## Findings

1. **R2 must export before it stops the agent.** `catalog export` asks
   the running agent (`client.catalog_export()`). The plan's R2 stops the
   agent in step 3 and exports in step 4. The rehearsal exports first
   (R2.2). Change for M6f and the plan: run `catalog export` before step 3.
2. **A delegated `start` of a running unit restarts it** (N5b, every run).
   In cutover, `lane_unit start` → `llm-unit start` → `llmctl-next run
   <workload>`. For a running workload the agent stops and starts it
   (operation phases "stop reh-lane", "start reh-lane"). `systemctl
   start` of an active unit does nothing. The real lane scripts check
   `is-active` before they start (`strata-start`), so they are not hit.
   Python `start-unit` checks "already active" first, so it is not hit.
   Any other caller of `llm-unit start` on a running enrolled unit
   restarts the engine. **Owner decision:** make `llm-unit start` a no-op
   for an active unit, or accept it and list it in the runbook.
3. **`audit` misses a raw stop that follows an agent stop and run with no
   observation pass in between** (N13b, every run). `audit::explains()`
   accepts any terminal effect after the baseline with the same process
   state. The old "stopped" effect of the agent's stop explains the later
   raw stop. The window is one watcher interval (15 s) after the run.
   **Owner decision:** compare only with the newest effect of the workload
   (or set the baseline in `record_effects`), in M6 or after the exit.
   It does not block M6f: the window is short and the M6f drill takes a
   status pass before the raw stop.
4. **The imports do not need the host lock released** (plan design
   decision 6 asked this). `catalog import` and `history import` run while
   another process holds the lock (F8 to F12).
5. **llm-router checks the profile's model before it hands off to the
   agent**, also with `llmctl_next_only`: a model directory needs a
   `config.json`. A switch to a profile whose model is missing is refused
   by the router; the agent is not asked. This is read-only and correct.
6. **`llm-heal` never restarts a crashed engine lane** ("the router starts
   it again on the next request"), and it never repairs a clean stop. The
   rehearsal lane is `engine: false` in `heal.yaml`, so N11 and R1.11 test
   the report-only hold and its reversal, not the GPU policy.
7. **Python `start-unit` of an active unit exits 0 before the cutover
   guard**, also with `LLMCTL_NEXT_DISABLED=1`. Nothing is written, so this
   is not a bypass. N1 uses a stopped unit.

## What this rehearsal does not cover

- Real engines, real cards, `nvidia-smi`, the router HTTP server and its
  in-flight probe (`LLMCTL_NEXT_ROUTER=off`), the lane lease
  (`LLMCTL_NEXT_LANE_LEASE=off`).
- `take-gpus`, the vendor `load` verb, Ollama, LM Studio.
- `llm-heal`'s `mac_repair` and the Mac healer; launchd. Of Mac
  `next_only`, only the fail-closed refusal: a stand-in llama-swap fails
  the agent's `served_identity` adoption check, so the agent has no
  llama-swap workload here. The owner check ("not a lane of llmctl-next
  workload") and the redirect of an owned lane are covered by the
  local-llm tests (M6b item 8) and the M6f drill on mbp2.
- `release` after `own`: the rehearsal adopts, it does not own.
- The M6f drill on `ollama-gpu1` (stop of a stopped enrolled unit).

M6f covers these on the live hosts.

## Procedure for M6f (from the checks above)

Workstation. Run each step; compare with the check named.

1. Signature "before": `llmctl-next status --json | scripts/serving_signature.py` (F1).
2. Take the lane lease, then the host lock (F2).
3. Quiesce: `systemctl --user disable --now llmctl-reconcile.timer
   llmctl-api.service`; `touch ~/.local/state/llmctl/ds4-llmctl-sync.off`;
   `llmctl_next_report_only: true` in `heal.yaml` (F3, F4).
4. Snapshot into `<data dir>/deploy-backups/<date>-m6-cutover/`:
   `sqlite3 ~/.local/share/llmctl/llmctl.sqlite3 ".backup <dir>/llmctl.sqlite3"`;
   `llmctl export-registry <dir>/bundle.json`; the sha256 list;
   `inventory.json`, `catalog.json` (F5 to F7).
5. Import with the lock still held: `llmctl-next catalog import
   <dir>/bundle.json --as <name> --json`, then `--confirm <token>`, then a
   dry run again; `llmctl-next history import <dir>/llmctl.sqlite3 --as
   <name> --json`, then `--confirm <token>`; `catalog export` and compare
   by id (F8 to F15).
6. Release the lock. Signature again (F16).
7. Switch: `llmctl_next.mode: cutover` in `settings.yaml`;
   `switch.llmctl_next_only: true` in `router.yaml`; restart llm-router
   with no request in flight; `llm-unit mode` prints `cutover` (F17, F18).
8. Release the lease. Signature "after" = "before"; no pid changed; the
   sha256 list is equal apart from `settings.yaml` and `router.yaml` (F19, F20).
9. Negative drills N1 to N14, each on a unit that can be away. Take a
   `status` pass before a raw stop (finding 3).

R1: restore the settings of the "Reversal" table, then `catalog export`,
`llmctl import-registry <file> --update-models`, `llmctl reconcile`, one
delegated write per legacy writer (R1.1 to R1.13).

R2: drain; signature; **`catalog export` first** (finding 1); then the
settings, the agent stop, `import-registry --update-models`, `reconcile`,
one write per legacy writer on its own path (R2.1 to R2.9).
