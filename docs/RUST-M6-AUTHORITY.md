# Rust successor — M6a recheck and authority map

Status 2026-11-27: M6a is done, read-only. No live system was changed.
The check of M6a has two parts. Part 1 holds: each row of the authority
map has a disposition. Part 2 holds on mbp2 and m5 and does not hold on
the workstation: the serving signature and the agent differ for `llmster`
(finding 2). Nine findings need an owner decision or a change to the M6
plan; they are in [Findings](#findings).

Plan: [RUST-M6-PLAN.md](RUST-M6-PLAN.md), section M6a. The writer numbers
(W1 to W23, M1 to M6) are the plan's numbers.

Each fact has a label, as in the plan:

- **checked**: read on 2026-11-27 (23:20 to 23:55 UTC) with the command
  that is named.
- **inferred**: a conclusion from checked facts.

Commands used: `llmctl-next` `status --all`, `status --json`, `host list`,
`intent`, `audit --all`, `ops`, `preset list`, `preset diff`, `model
list`, `model scan`, `stores`, `vram`, `version`, `lanes memory`, `lanes
matrix`, `lanes show`, `run … --dry-run`, `tui --read-only`, `tui --bench`;
`systemctl --user show`, `cat`, `is-active`, `is-enabled`, `list-timers`;
`sha256sum`, `ss -ltnp`, `nvidia-smi`; over ssh on the Macs: `shasum`,
`sw_vers`, `launchctl list` (labels and pids only), and a read of the
agent's `inventory.json`. `settings.yaml` was read only through the Python
guard's own parser (`load_delegation()`), which returns the `llmctl_next`
section and no token. `launchctl print` and env files were not read.

## 1. Recheck per host (M6a item 1)

### Builds and paths (checked)

| Host | Agent binary | sha256 (first 16) | Build | Version, protocol | Inventory revision | Data dir |
|---|---|---|---|---|---|---|
| workstation | `~/.local/state/llmctl-next-m4a/llmctl-next.bin` | `208bdf1c4cb29829` | `ccabf75` | 0.1.0, 1 | 55 | `~/.local/state/llmctl-next-m4a` |
| mbp2 (macOS 27.2.1) | `~/.local/bin/llmctl-next` | `f559fbc83b43196b` | `ccabf75` | 0.1.0, 1 | 14 | `~/.local/state/llmctl-next` |
| m5 (macOS 27.0.1) | `~/.local/bin/llmctl-next` | `f559fbc83b43196b` | `ccabf75` | 0.1.0, 1 | 17 | `~/.local/state/llmctl-next` |

- Workstation: the running executable (`/proc/<pid>/exe`) has the same
  sha256 as the file. The agent runs since 14:09:58 PST 2026-11-27. Its
  unit file is a symlink to `~/AI/services/user/llmctl-next.service`.
- `llmctl-next` is not on `PATH` on the workstation. The path
  `~/.local/state/llmctl-next` does not exist there, thus the symlink of
  decision 3 has no conflict.
- m5 is still on macOS 27.0.1. The 27.2.1 install is an owner task.
- Python llmctl in `~/AI/.venv` is 0.3.0, installed from a wheel that was
  built from `55c83de`. The llmctl-stack pin names `d7f2304`. The pin and
  the installed build differ (M6d step 6 moves the pin).
- llm-router: the checkout is on `4928efb` (committed 15:19 PST). The
  process runs since 14:11:02 PST. Inferred: the process has the `flash`
  wiring, because the agent ran `Run flash` for the router at 14:27 PST.
- `main` is at `6e243c0`. It holds `ccabf75` and the M6 plan.

### Legacy services and timers on the workstation (checked)

| Unit | State | What it does | M6 |
|---|---|---|---|
| `llmctl-api.service` | active, enabled | Python llmctl 0.3.0, port 8088 | W2: stop and disable |
| `llm-router.service` | active, enabled | switches lanes | W5, W6 |
| `llm-heal.timer` | active, every 2 min | repairs units | W7, W8 |
| `llmctl-reconcile.timer` | active, about every 30 s | writes Python session rows | W17: stop and disable |
| `longjob-runner.service` | active, enabled | runs queued jobs | W9: no change |
| `router-profiles-check.timer` | active, hourly | read-only check | none |
| `deepseek-v41-readiness.timer` | active, weekly | readiness check | none |
| `llm-roster-status.timer` | active, every minute | runs `~/AI/mesh/roster-status-push.sh` | **not in the plan**; not read, inferred read-only from its name |
| `llm-mesh-health.timer` | active, every 5 min | writes `~/AI/logs/mesh-health.json` | **not in the plan**; a report, no write to a unit (checked: its `ExecStart`) |

`llm-backend-guard` is not a systemd unit (`not-found`).

### Hooks in unit files (checked: `systemctl --user cat`)

`ds4-llmctl-sync adopt` (ExecStartPost) and `detach` (ExecStopPost) are in
six units: `gpt-oss-120b`, `qwen3.8-flash-next-fast-strata`,
`qwen3.8-flash-next-strata`, `qwen3.8-flash-next-uncensored-strata`,
`deepseek-v4.1-flash-strata`, `glm-5.3-flash`. The agent owns the first
three. This equals the plan (W16, design decision 5).

### The Python guard's unit map (checked: `load_delegation()`)

`enabled: true`. Seven units: `gpt-oss-120b.service`, `llmster.service`,
`ollama-gpu1.service`, `ollama.service`,
`qwen3.8-flash-next-fast-strata.service`,
`qwen3.8-flash-next-strata.service`, `vllm-engine.service`. The map does
not name `llamacpp-router.service` (finding 4).

### llm-router wiring (checked: parsed `router.yaml`, key names only)

`switch`: `llmctl_next_workload: vllm-engine`, `llmctl_next_refusal_final:
true`, `stop_first: [vllm-engine2.service]`, `evict_ollama:
[http://127.0.0.1:11434]`. There is no `llmctl_next_only` key (M6b item 2
adds it).

| Backend | Unit | Goes to the agent as | Has `start_cmd` |
|---|---|---|---|
| `qwen27b`, `qwen27b-uncensored`, `qwen27b-heretic`, `qwen27b-awq`, `qwen27b-unc-awq` | `vllm-engine` | workload `vllm-engine`, profile env | no |
| `flash` | `qwen3.8-flash-next-strata` | `flash` | yes |
| `flash-fast` | `qwen3.8-flash-next-fast-strata` | `strata-fast` | yes |
| `gptoss` | `gpt-oss-120b` | `gpt-oss-120b` | yes |
| `flash-uncensored` | `qwen3.8-flash-next-uncensored-strata` | **no key** | yes |
| `dsv41` | `deepseek-v4.1-flash-strata` | **no key** | yes |
| `glm53` | `glm-5.3-flash` | **no key** | yes |
| `qwen3coder30b` | `ollama-gpu1.service` | **no key** | yes |

## 2. Authority map: managed resources

One row per managed resource. "Writers now" names each program that can
change the resource on 2026-11-27. "Proof" names the M6 step that proves
the disposition.

### Workstation workloads (checked: `status --json`, `intent`, the tables above)

All eight have ownership `managed` and a systemd user unit.

| Workload | Unit, port, cards | State, intent | Writers now | M6 disposition | Proof |
|---|---|---|---|---|---|
| `vllm-engine` | `vllm-engine.service`, 8003, 0+1 | Running, running | agent; llm-router (through the agent); Python llmctl, `llm-unit`, `take-gpus`, `longjob` (through the guard); `llm-heal` (direct `systemctl`); the W13 scripts and study scripts (raw `systemctl`); systemd `Restart=on-failure` | agent only. W1, W4, W5, W7, W13, W14 as in the plan | M6f step 9 drills; finding 1 (resolved for the router's path) |
| `ollama` | `ollama.service`, 11434, 0 | Running, running | agent; the guard writers; llm-router `evict_ollama` (unloads models directly); `llm-heal`; `Restart=always` | agent only. W6: the eviction goes through the agent | M6f step 9 |
| `llmster` | `llmster.service`, 1234, 0+1 | Stopped, **running**, `gave_up` | agent; the guard writers; **a `llmster` process outside the unit holds the port** (finding 2) | agent only, after finding 2 and the intent of section 5 | signature equal |
| `llamacpp-router` | `llamacpp-router.service`, 8012, 0+1 | Stopped, **running**, `gave_up` | agent (unit, router models, preset INI); Python llmctl **directly** (not in the unit map, finding 4) | agent only. Add the unit to `llmctl_next.units` | drill: `llm-unit` on the unit is redirected |
| `gpt-oss-120b` | `gpt-oss-120b.service`, 8005, 0+1 | Stopped, stopped (manual) | agent; llm-router `gptoss` (through the agent; `gptoss-start` stays in the file); the guard writers; `ds4-llmctl-sync` hooks (session rows); systemd `Conflicts=` of six other units | agent only. W12, W16 | M6f step 9 |
| `ollama-gpu1` | `ollama-gpu1.service`, 11435, 1 | Stopped, stopped (manual) | agent; the guard writers; **`qwen3coder30b-start` with raw `systemctl start`** (W21) | agent only. Router `load` handoff (M6e item 3) | M6e round trip |
| `strata-fast` | `qwen3.8-flash-next-fast-strata.service`, 8009, 0+1 | Stopped, stopped | agent; llm-router `flash-fast` (through the agent); `strata-start`, `strata-stop` (raw `systemctl`); hooks | agent only. W12 | M6f step 9 |
| `flash` | `qwen3.8-flash-next-strata.service`, 8007, 0+1 | Stopped, stopped | as `strata-fast`, backend `flash` | agent only. W12 | round trip passed 2026-11-27 |

### Workstation units that no agent owns (checked: `systemctl --user show`)

| Unit | State | Writers now | M6 disposition |
|---|---|---|---|
| `qwen3.8-flash-next-uncensored-strata` | inactive, linked | llm-router own path, `strata-start`, hooks | adopt (M6e item 1) |
| `deepseek-v4.1-flash-strata` | inactive, linked | llm-router own path, `dsv41-q`, hooks | adopt (M6e item 1) |
| `glm-5.3-flash` | inactive, linked | llm-router own path, `glm53-start`, `glm53-stop`, hooks | adopt (M6e item 1) |
| `vllm-engine2` | inactive, linked | llm-router `stop_first`; Python `vllm_slots.place` | adopt as a stopped workload (M6e item 2) |

A unit `inkling2-mini` is in the pilot log and in `vram.json`. It is not
in the inventory and has no unit file now (`not-found`).

### Mac workloads (checked: each agent's `inventory.json`, `status --json`, `launchctl list`)

| Host | Workload | Label, port | Ownership, authority | State, intent | Writers now | M6 disposition |
|---|---|---|---|---|---|---|
| mbp2 | `llama-swap` | `com.pitcany.llama-swap`, 8099 | external, `launchd_service` | Running, running | agent; Mac llmctl **directly** (no `next_enabled` line); llama-swap request and TTL swaps; `mac-heal` | M1: `next_enabled: 1` and `next_only: 1`. M2: accepted exception. M3: report-only. See section 6 |
| mbp2 | `lm-studio` | `application.ai.elementlabs.lmstudio`, 1234 | external, `vendor_command` (`lms server start`/`stop`) | Running (the port listens; no launchd label), unmanaged | the owner (the app); `mac-heal` (`lms server`) | intent of section 5; M3 |
| mbp2 | `ollama` | `com.ollama.ollama`, 11434 | external, `vendor_command` (`open -ga Ollama` / `pkill`) | Running, running | agent; the vendor app and its updater (M6); `mac-heal` | M3; M6: `version` reports drift |
| m5 | `llama-swap` | `com.pitcany.llama-swap`, 8099 | managed, `launchd_service` | Running, **unmanaged** | agent; Mac llmctl (through the agent, `next_enabled: 1`); llama-swap request and TTL swaps; `mac-heal`; `llmctl-warm` | M1: `next_only: 1`. M2, M3, M4. Intent of section 5 |
| m5 | `ollama` | `com.pitcany.ollama`, 11434 | managed, `launchd_service` | Running, running | agent; `mac-heal`; brew updater (M6) | M3 |
| m5 | `ds4` | `com.pitcany.ds4`, 8788 | external, `observe_only` | Stopped, unmanaged | none through the agent (observe-only); llama-swap starts it as a lane | intent of section 5 |
| m5 | `lm-studio` | `application.ai.elementlabs.lmstudio`, 1234 | external, `vendor_command` | Stopped, stopped (manual) | the owner; `mac-heal`; `lms-stray-watch` (M5) | M3, M5 |

### Lanes, router models, presets, catalog (checked)

| Resource | Where | Records | Writers now | M6 disposition |
|---|---|---|---|---|
| llama-swap lanes, m5 | `~/AI/llama-swap/config.yaml` | lane store revision 11, 14 lanes, matrix with 14 vars and 8 sets, budget 104 GiB | agent (`lanes`, `llamacpp lane`); Mac llmctl (through the agent); an editor by hand | agent only; M1 |
| llama-swap lanes, mbp2 | `~/AI/llama-swap/config.yaml` | lane store revision 5, 2 lanes, no matrix, budget 46 GiB | agent; Mac llmctl **directly** | agent only; M1 |
| llama.cpp router models | `llamacpp-router` preset INI | agent records (`llamacpp import`) | agent (`llamacpp load/unload/set/add/remove`) | agent only. The INI is in no git repository (pilot day 3) |
| Presets | `~/.config/llmctl/presets/*.yaml`, five files | `preset list`: `qwen27b` serving, four ready, no drift | agent (`preset edit/new/sync`); Python `llmctl presets`; an editor | agent for writes; the Python read verbs stay |
| Profile envs | `~/AI/services/vllm-engine.env.*` | linked from the presets | agent (`preset sync`, profile switch); `scripts/vllm-ab*.sh` (see section 3) | agent only |
| Catalog | `<data dir>/catalog.json` per host | 54, 14 and 8 models; revision 1 on the workstation | agent (`model …`, `catalog import`) | agent only. The Python DB is frozen (design decision 7) |
| Router aliases | `~/.config/llmctl/router_aliases.json` | none in the agent | Python `llmctl set-alias` | **not a managed resource.** Stays with the Python tool (M5 gap C4) |

## 3. Scripts the plan did not read, and port 8088 (M6a item 3)

A subagent read these files on 2026-11-27. The lines marked "checked" were
read again in the main session. The other lines are from its report.

### Workstation

| Writer | File | What it writes | Path | M6 disposition |
|---|---|---|---|---|
| W11 modellab | `~/AI/lab/modellab/` | Takes the lane lease (`lease.py`). **Does a unit write:** `llmctl swap <name> --yes` (`runner.py:43`, checked) with `--swap` or `--restore`. **Unloads and loads Ollama models directly** (`meta.py:347` `keep_alive: 0`, checked; `meta.py:330` load). No `systemctl` write | Python llmctl; direct Ollama API | The swap follows W1. The direct Ollama calls: see finding 9. The plan's text "no unit write was found" is not correct |
| W13 `restore-vllm` | `~/AI/services/lanes/restore-vllm` | stop `ollama`, `reset-failed` and start `vllm-engine`, start `ollama` | `llm-unit` | follows W3. No change |
| W13 `vllm-on-gpus` | `~/AI/services/lanes/vllm-on-gpus` | nothing (`is-active` only) | read-only | none |
| W13 `recover-vllm.sh` | `~/AI/scripts/recover-vllm.sh` | `llm-unit restart vllm-engine` | `llm-unit` | follows W3 |
| W13 `vllm-ab.sh`, `vllm-ab-queue.sh` | `~/AI/scripts/` | restart `vllm-engine` per arm (`llm-unit`); **overwrites the live `vllm-engine.env`** (`cp -p`, checked); **unloads each Ollama model directly** (line 134, checked); writes the lane lease; pauses `llm-heal` | `llm-unit`; direct file write; direct Ollama API | Redirect the env change to a preset or a profile switch, or retire the script. The env overwrite makes an `environment_files` difference on the agent's record |
| W13 `run-quant-ab.sh` | `~/AI/scripts/run-quant-ab.sh` | `llmctl swap --yes <preset>` per arm and on exit | Python llmctl | follows W1 |
| not in the plan: `pp-vs-tp-ab.sh` | `~/AI/scripts/pp-vs-tp-ab.sh` | `llm-unit restart vllm-engine`; rewrites `vllm-engine.env` | `llm-unit`; direct file write | as `vllm-ab.sh` |
| W15 `llm-backend-guard` | `~/AI/bin/llm-backend-guard` | `llm-unit stop vllm-engine` (argument `ollama`), `llm-unit start vllm-engine` (argument `vllm`); **unloads each Ollama model directly** (lines 44-45, checked). Only caller: `ccl`, after a `y` on a tty | `llm-unit`; direct Ollama API | Unit verbs: no change. Ollama unload: finding 9 |
| W12 lane scripts (read again) | `~/AI/services/lanes/` | raw `systemctl --user start` or `stop` of the lane unit: `strata-start:51,69`, `strata-stop:23`, `glm53-start:57`, `glm53-stop:23`, `gptoss-start:52`, `coder1-start:115` (checked). `coder1-start`, `gemma1-start`, `qwen3coder1-start` and `qwen3coder30b-start` are one file (checked: `cmp`); it also loads the model with `keep_alive: -1` on port 11435. `dsv41-q:136` restarts `llm-router` | raw `systemctl`; direct Ollama API | M6b item 6, as in the plan |
| W14 study scripts (list) | `~/AI/scripts/finetune/qwen-refusal-run/*.sh`, `rag/eval/run-critic-window.sh`, `rag/eval/run_prm_window.sh` | raw `systemctl --user` start and stop of `llm-router`; engine verbs through `llm-unit`; **a raw fallback `systemctl --user restart ollama`** in the two `rag/eval` scripts | mixed | the gate check of M6b item 6 must find the raw fallback |
| `llm-unit` itself | `~/AI/bin/llm-unit` | `start`, `stop`, `restart` of one unit go to the Python guard. **`reset-failed`, `kill`, `enable`, `disable` are never delegated** | raw `systemctl` for those four | Decide in M6b item 1 if cutover mode refuses `kill` on an enrolled unit. `reset-failed` changes no serving state |
| `llm-heal` | `~/AI/heal/llm-heal.py` | asks `intent` (line 485), then `systemctl restart` per lane, `reset-failed`, and restarts of front doors | raw `systemctl` | W7, as in the plan |

Other engine writers that use only `llm-unit` and are not in the plan:
`apply-spec-rollout.sh`, `rag/qdrant_migration_finish.sh`,
`rag/gpu_reindex_pipeline.py`, `rag/gpu_reindex_direct.py`,
`rag/reingest_new_books.py`. They follow W3.

A unit name that the plan does not have: `dsv41-q` sets
`deepseek-v4.1-flash-dsv41-q-strata.service`. That unit is not loaded now
(checked: `LoadState=not-found`).

No systemd unit, timer or cron entry calls a W13 script.

### Macs

| Writer | Where | What it does | M6 disposition |
|---|---|---|---|
| M4 `llmctl-warm` (m5) | LaunchAgent, every 600 s: `llmctl warm --guarded mimo9b-m5` | Sends one 1-token request to a lane that is loaded. `cmd_warm` refuses a lane that is not loaded ("warm never loads a lane", line 710, checked). `--guarded` skips on seven conditions and exits 0. Inferred: the loaded check and the request are not atomic; if llama-swap unloads the lane between them, the request loads it (class M2) | no change, as the plan says |
| M5 `lms-stray-watch` (m5) | LaunchAgent, every 30 s: `~/.local/bin/lms-stray-watch.sh` | **Log only.** No kill, no `lms server stop`, no unload (checked: `grep -c` gives 0). It appends process chains to `~/Library/Logs/lms-stray-watch.log` | none. Not a writer |
| not in the plan: `bench-gate-a.py` (m5) | `~/local-llm/scripts/` | `llmctl-next … lanes stop llama-swap <lane>` | through the agent. No change |

LaunchAgents with an LLM role (labels `com.pitcany.*`): m5 has
`llama-swap`, `ollama`, `mac-heal`, `llmctl-next`, `llmctl-warm`,
`lms-stray-watch`, `llm-standby`, `llm-gateway`, `llm-gateway-env`. mbp2
has `llama-swap`, `mac-heal`, `llmctl-next`, `llm-standby`, `llm-gateway`,
`llm-gateway-env`, and the vendor job `com.ollama.ollama`. No key is in a
`ProgramArguments` value. `llm-gateway` is a forwarder and changes no
engine (inferred from its program).

`~/.config/llmctl/config.yaml`: m5 has `next_enabled: 1` and no
`next_only`. mbp2 has neither line. The `~/local-llm` checkouts differ: m5
is on `f7cea28`, mbp2 on `5dd7415`.

### Consumers of port 8088

| Consumer | What it does with the port | At the switch |
|---|---|---|
| `llmctl-api.service` | serves it (`llmctl serve --port 8088`, `Restart=always`, enabled). No other unit requires it | stop and disable |
| Tailscale Serve | `:12443` on the tailnet is a proxy to `127.0.0.1:8088` (checked) | the page has no backend after the stop. Remove the serve entry or leave it; owner choice |
| `llm-heal` | **not a consumer.** `front_doors` in `heal.yaml` names `litellm`, `llm-router`, `llm-mesh-forwarder`, `hermes-gateway` (checked). Thus `llm-heal` does not restart `llmctl-api` | nothing to remove |
| `llm-router`, `llmctl-reconcile.timer` | comments only; the timer feeds the dashboard's rows | none |
| Documents | `~/AI/CLAUDE.md` (the redeploy step restarts `llmctl-api`), `~/AI/docs/RUNBOOK.md`, `GATEWAY-QUICKSTART.md`, three published artifact sources | update in M6g step 3 |

No Caddy, nginx or dashboard config names the port. No file on the Macs
names it. Browser bookmarks were not checked.

### Calls of `llmctl-next` with no data dir (decision 3)

Each program that calls the agent passes a data dir: llm-router
(`router.yaml`), `llm-heal` (`heal.yaml`), `mac-heal` (`mac-heal.json`),
the Mac llmctl (`llmconf.py`), the ssh bridge (`bridge.rs:24`), and the
Python guard (checked: `load_delegation()` returns the `llmctl-next-m4a`
binary with `--data-dir`). Each Rust test spawns the binary with a
temporary `--socket` or `--data-dir`. The llmctl-stack tests set
`LLMCTL_NEXT_CMD` to a path that does not exist.

Two places rely on the default (finding 8):

1. `rust/crates/llmctl-cli/src/enroll.rs:53-62` (checked): `hosts_path()`
   and `status_cache_path()` use `default_data_dir()` and ignore
   `--data-dir` and `--socket`. The verbs `--host`, `host list/add/remove`,
   `status --all`, `audit --all` and `tui` use them.
2. `rust/README.md:32-36` (checked): the developer commands start a
   daemon and run `run coder --wait` with no data dir.

Three files pin the path `llmctl-next-m4a`: `router.yaml`, `heal.yaml`
and `~/AI/services/user/llmctl-next.service`. They stay correct while the
symlink of decision 3 points at that dir.

## 4. `Conflicts=` against the planner's stop set (M6a item 4)

The planner stops each running workload that holds a card of the target
(`gpu_conflicts`, `llmctl-core/src/planner.rs`). It keeps a neighbour only
when the VRAM fit says the target fits beside it, and the fit refuses that
when the target's `Conflicts=` names the neighbour (`fit.rs`, `decide`).

| Owned unit | `Conflicts=` (without `shutdown.target`) | Each named unit that is owned shares a card with it? |
|---|---|---|
| `vllm-engine`, `ollama`, `llmster`, `llamacpp-router`, `gpt-oss-120b` | none | no list |
| `ollama-gpu1` (card 1) | `gpt-oss-120b`, `glm-5.3-flash` | `gpt-oss-120b`: yes (0+1) |
| `strata-fast` (0+1) | `gpt-oss-120b`, `flash`, `flash-uncensored`, `glm-5.3-flash` | yes, yes |
| `flash` (0+1) | `gpt-oss-120b`, `strata-fast`, `flash-uncensored`, `glm-5.3-flash` | yes, yes |

Result (checked for the lists, inferred for the conclusion): each owned
unit that an owned unit's `Conflicts=` names is in the planner's stop set
for that start. No owned pair can be stopped by systemd behind the plan.

Two limits:

1. `flash-uncensored`, `dsv41` and `glm53` are in these lists and are not
   owned. A start of one of them through the router's own path makes
   systemd stop `gpt-oss-120b`, `flash`, `strata-fast` or `ollama-gpu1`
   with no agent operation. `audit` records it. M6e item 1 closes this.
2. The agent reads only the forward direction (`systemctl show
   --property=Conflicts`, `llmctl-daemon/src/systemd.rs`). systemd also
   stops a running unit that names the target in its own list. Example:
   `gpt-oss-120b` has an empty list, and four units name it. This has no
   effect today, because each such pair shares a card and no pair fits
   beside the other on a 32 GiB card (inferred from `vram`: 28 and 30 GiB
   per card). It is a latent gap (finding 6).

## 5. Intents to record (M6a item 5)

The plan lists four workloads with intent `unmanaged`. Since then
`llamacpp-router` got the intent `running` from a `run`. Two more rows
have an intent that does not match what the owner wants.

| Host | Workload | Now | Proposed intent | Verb |
|---|---|---|---|---|
| workstation | `llamacpp-router` | Stopped, intent running, `gave_up` | stopped (manual), if it stays down by owner choice | `stop llamacpp-router`: the unit is stopped, nothing restarts |
| workstation | `llmster` | Stopped, intent running, `gave_up` | stopped (manual), after finding 2 | `stop llmster` |
| m5 | `llama-swap` | Running, unmanaged | running | **no verb without a restart** (below) |
| m5 | `ds4` | Stopped, unmanaged, observe-only | keep `unmanaged`: llama-swap starts it as a lane, the agent cannot | none |
| mbp2 | `lm-studio` | Running, unmanaged | keep `unmanaged`: the owner starts the app by hand | none |

**Which verb records an intent without a restart** (the plan's open
fact): none for a running workload. `run` on a running target plans `stop`
then `start` (checked: `--host m5 run llama-swap --dry-run` shows both
steps; `plan_run` in `planner.rs` says so). `stop` on a stopped workload
records the manual stop and stops nothing. The intent is written only by
the `PersistIntent` step of a run, stop or restart plan
(`llmctl-daemon/src/ops.rs`). Thus m5 `llama-swap` needs one of:

- a small verb in M6b item 4 that records an intent as an operation with
  the one step `PersistIntent` (recommended), or
- a `restart` in a window: the loaded lanes unload and load again on the
  next request.

## 6. Is mbp2's `llama-swap` agent-owned? (M6a item 6)

Partly (checked: mbp2 `inventory.json`, `lanes show`):

- The agent has the workload with authority `launchd_service` and intent
  `running`. The lane store exists (revision 5, 2 lanes, the `llamacpp`
  macro). Thus `lanes swap` and `lanes stop` work there.
- The ownership is `external`: the launchd job was not adopted, so the
  record has no pinned definition and no digest. On m5 the same workload
  is `managed`.
- `~/.config/llmctl/config.yaml` on mbp2 has no `next_enabled` line.

For M6f step 7 on mbp2: the lanes are records, so `next_enabled: 1` and
`next_only: 1` can be set. Recommended first: `adopt` the launchd job in
M6e, as on m5. An adoption restarts nothing.

## 7. Serving signature (M6a item 7)

Script: [`scripts/serving_signature.py`](../scripts/serving_signature.py)
(Python 3.9, standard library; test `tests/test_serving_signature.py`).
It takes the agent's `status --json` for the list of workloads. Then it
reads `systemctl show` or `launchctl list`, each port, and `nvidia-smi`
with each pid's cgroup.

```bash
# workstation
B=~/.local/state/llmctl-next-m4a/llmctl-next.bin; D=~/.local/state/llmctl-next-m4a
$B --data-dir $D status --json | python3 scripts/serving_signature.py
# a Mac, from the workstation; nothing is copied to the Mac
ssh m5 'python3 - --status <(~/.local/bin/llmctl-next --data-dir ~/.local/state/llmctl-next status --json)' < scripts/serving_signature.py
```

Result on 2026-11-27 at 23:52 UTC (checked; three runs gave the same
digests):

| Host | Digest (first 16) | Agrees with the agent |
|---|---|---|
| workstation | `2e267a06a969793d` | **no**: `llmster: agent says Stopped, observed active=False listening=True` (finding 2). The other seven agree: `vllm-engine` pid 4022288 serves `qwen3.8-27b` on 8003 and holds 26.9 GiB on each card; `ollama` pid 4255 has `qwen3-embedding:0.6b` loaded |
| mbp2 | `8b3851b99d4311e2` | yes: `llama-swap` pid 67153, no lane loaded; `lm-studio` and `ollama` listen and have no launchd pid |
| m5 | `c0dd9c97434b8f3c` | yes: `llama-swap` pid 1920 with lane `mimo9b-m5` loaded; `ollama` pid 1907; `ds4` and `lm-studio` down |

Limits of the comparison (finding 7):

- `status --json` has no pid and no engine kind. The script can compare
  the process state, the port and, for an endpoint that lists one model,
  the served model. It cannot compare the pid with the agent.
- For Ollama, LM Studio and llama-swap the agent records the service
  identity, not a model id. The script does not compare those.
- A vendor app has no launchd job of its own. The script counts a
  listening port as running there.

## 8. TUI seen live (M6a item 8, M5e leftovers)

Checked with `tui --read-only` in a detached terminal, 190 columns:

- **Hosts:** the three hosts with 8, 3 and 4 workloads, each seen 2 to 3 s
  ago. mbp2 is on the screen.
- **Run:** the five presets of the workstation (`qwen27b` serving), then
  the workloads of each host.
- **Catalog:** 76 rows; `workstation rev 1 · 54 models · 20 bundle-only`.
- **GPUs/memory:** the cards with enrolled and foreign processes and four
  measured workloads; mbp2 budget 46 GiB with no lane loaded; m5 budget
  104 GiB with `mimo9b-m5` loaded (20 GiB peak). The first answer took
  more than 3 s and less than 8 s.
- **Operations:** 168 rows. **Audit:** 103 rows.
- The Logs screen was not opened. No write was tried.

`tui --bench 5`: key to frame p50 0.35 ms, p95 1.27 ms; hosts refresh p50
263 ms, p95 472 ms.

## Findings

Nothing here was fixed. Each needs the owner or a change to the M6 plan.

1. **`vllm-engine` cannot be started through the agent now** (checked).
   Each `run` of it answers `409 definition_drift … differing:
   environment_files, version.lock_digest`. At 15:23 PST 2026-11-27 an
   editable `pip install --user` of `agentkit` (source:
   `~/.custom-harness/worktrees/20261127-152329-964862`) wrote `anyio
   4.16.0`, `httpx 0.29.0`, `httpcore`, `h11` and `typing_extensions` into
   `~/.local/lib/python3.12/site-packages`. The `vllm-muse` interpreter
   (Python 3.12.14, user site on) sees them, so its `pip freeze` digest is
   not the pinned one, and the user site comes before the env's own `anyio
   4.15.1` and `httpx 0.28.1`. The serving process is not changed. The
   next router switch to a vLLM profile is refused, and the refusal is
   final. The owner chooses: remove the six user-site packages (the pin
   then holds again), or accept them with `version vllm-engine --repin`.
   A later change that keeps the user site out of each engine
   (`PYTHONNOUSERSITE=1`) changes a unit file, so it is not for M6
   (design decision 5).

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
   **Mechanism found and closed, 2026-11-27 16:20 PST** (checked).
   The writer was a custom-harness agent session (`20261127-152329-964862`,
   model `qwen27b-nothink`): `pip install -e .` was refused by PEP 668,
   then it ran the same with `--break-system-packages`, which is a user
   install. The harness shell sandbox (`bwrap --dev-bind / /`) keeps
   `$HOME` writable. The engine processes started at 14:38 and the
   directory was made at 15:23:41, so the running engine did not load the
   packages. Three changes, none of them a repin or a restart:
   (a) `~/.local/lib/python3.12` is a read-only guard directory (mode
   0555); a user install for Python 3.12 fails with a permission error.
   (b) `~/AI/bin/engine-pins-check` runs each hour
   (`engine-pins-check.timer`, `~/AI` `34d5fa69`): it fails when a
   workload has version drift or when its Python reads a user site that
   holds packages. (c) custom-harness branch
   `fix/isolate-python-user-base` (`addb55c`, not merged while the
   portfolio job runs) gives each harness process a private
   `PYTHONUSERBASE`. Open for after the M6 exit: `PYTHONNOUSERSITE=1` in
   the engine definitions. It changes a pinned definition (design
   decision 5), and the agent's `pip freeze` probe must then run with the
   same variable.
2. **An LM Studio server runs outside its unit** (checked: `ss -ltnp`,
   `/proc/<pid>/cgroup`). Process `llmster`, pid 3820672, started 18:43
   PST 2026-11-26 in `session-3.scope`, listens on 127.0.0.1:1234.
   `llmster.service` is inactive. The agent's record says Stopped and the
   agent reads loaded models from that port. This is a writer outside the
   map: a shell start of `llmster` or `lms server start` (W23).
3. **Two intents do not match the owner's choice** (`llmster`,
   `llamacpp-router`: section 5).
4. **`llamacpp-router.service` is not in the Python guard's unit map.**
   The owner approves each edit of `settings.yaml`.
5. **No verb records an intent without a restart** (section 5). Proposed:
   add it to M6b item 4.
6. **The agent reads `Conflicts=` in one direction only** (section 4).
   Latent. Proposed: read `ConflictedBy` too in M6b item 4, or accept.
7. **`status --json` has no pid and no engine kind** (section 7). Design
   decision 8 wants three equal records. Proposed: add both fields in M6b
   item 4 (an additive wire change), so the script compares them.

8. **The default data dir change of decision 3 has two unsafe places**
   (section 3). With the new default, a developer who follows
   `rust/README.md` starts a second daemon on the live data dir, and a
   test agent that runs `status --all` with only `--socket` reads and
   writes the live `hosts.json`. Proposed for M6b item 4: make the
   enrollment store follow `--data-dir`, and change the README commands
   to set `LLMCTL_NEXT_DATA_DIR`.
9. **Four programs unload or load Ollama models with no agent
   operation:** `llm-backend-guard`, `vllm-ab.sh`, modellab, and the
   `coder1-start` file (section 3). The plan has this class only for
   llm-router (W6). Inferred: `audit` does not count a model change inside
   a running Ollama as a unit write. Proposed: redirect them to
   `llmctl-next unload` and `load` in M6b item 6, or accept them as M2 is
   accepted for llama-swap.

Also for the plan, with no decision needed:

- W11: the plan says modellab has no unit write. It calls `llmctl swap`.
- `llm-unit` never delegates `reset-failed`, `kill`, `enable`, `disable`.
- Tailscale Serve `:12443` points at port 8088.

- mbp2 `llama-swap` is not adopted (section 6). Add the adoption to M6e.
- Two timers are not in the plan's list: `llm-roster-status.timer` and
  `llm-mesh-health.timer`. Read `roster-status-push.sh` in M6b.
- m5 is on macOS 27.0.1.
