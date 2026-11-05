# P3 pilot trace (2026-11-05)

P3 step 1 of `CONSOLIDATION-PLAN-2026-11-04.md` §7, for the two pilots chosen
from `INFERENCE-INVENTORY.md`. Read-only except the code fix in §4. Labels:
(obs) observed on the desktop today, (src) read in source, (graph) from the
graphify graph of this repo (`graphify-out/`, built 2026-11-03, llmctl only; it
does not cover `~/AI`, so the router and lane scripts were read directly).

## 1. Pilot choice

| Pilot | Why this one |
| --- | --- |
| `qwen27b-awq` (vLLM preset, one card) | On-demand, not the tutor default, no shim depends on it; a mistake costs least. |
| `gptoss` (command-backed llama.cpp lane) | Most launch definitions for one process (4 in the inventory). |

## 2. `qwen27b-awq`: preview → launch → recovery

| Stage | Path |
| --- | --- |
| Preview | `llmctl run qwen27b-awq --dry-run` (obs): places it on GPU 1, would write `~/AI/services/vllm-engine.env` and restart `vllm-engine`. |
| Control | `llmctl run` → `cli.py` dispatch (src, `cli.py:2294`). A placed one-card preset always takes the **direct** path: the router's vLLM profiles take both cards. A request through the router (or an unplaced run) takes the **router** path. |
| Render (direct) | `start_vllm_tp` → `_build_spec` → `render_vllm_env` → `launcher_env_lines` (graph: `render_vllm_env` has 1 production caller, `render_preset_env`; src for `start_vllm_tp`). |
| Render (router) | `Switcher._switch_vllm_profile` copies `services/vllm-engine.env.qwen27b-awq` onto `vllm-engine.env` (src, `switcher.py:1048`). |
| Readiness | direct: `adapter.restart_with_spec(wait_for_ready=...)`; router: `_wait_healthy` on `/v1/models`. |
| Rollback | direct: captured previous env; router: in-memory snapshot, `.previous` written only after a healthy switch. |
| Recovery | healer + router `reset_failed` + systemd (inventory §6: 7 restart paths). |

**Duplicated facts.** The 4 router profile envs are hand-synced copies of the
presets. `llmctl router-profiles` (llmctl-stack) compares them: **all 4 match
today** (obs, exit 0; values not printed). This closes the inventory's first
"unverified" item. Nothing stops them drifting again.

**Duplicated policy.** Two switch implementations write the same active env:

| Step | Router `_switch_vllm_profile` | llmctl `_run_lifecycle` |
| --- | --- | --- |
| Model check | `model_available` | `_check_model_path` |
| Free GPUs | `stop_first` list + `evict_ollama` | fleet preflight + companions |
| Write env | copy profile file | render preset |
| Restart + ready | `restart_cmd` → `_wait_healthy` | `restart_with_spec` |
| Rollback | snapshot + `.previous` | captured previous env |

Both are in use: one-card placement exists only in llmctl, request-driven swaps
only in the router. Merging them is P5 (hybrid lifecycle), not this pilot.

## 3. `gptoss`: preview → launch → recovery

| Stage | Path |
| --- | --- |
| Preview | `llmctl run gpt-oss-120b-mxfp4 --dry-run` (obs): "served by the gptoss lane", starts `gpt-oss-120b.service` via `~/bin/gptoss-start --take-gpus`. |
| Control | `serve.route` → `lane_for_model` (src, `serve.py:134`) → router `/admin/switch` when up, else `run_lane_swap`. |
| Launch | `gptoss-start`: frees vLLM/Ollama/engine2 on cards 0,1, then starts the user unit (src). |
| Definition | unit `ExecStart` holds every tuned flag (obs): ctx 131072, `--n-cpu-moe 6`, `--tensor-split 56,44`, port 8005. |
| Session | unit `ExecStartPost` → `ds4-llmctl-sync adopt`; `ExecStopPost` → detach (obs). |
| Recovery | `Restart=on-failure`, `StartLimitBurst=3` (obs). |

**Launch definitions (4).** Unit `ExecStart` (authoritative); SQL profile
`gpt-oss-120b-cpumoe` ("Mirrors services/gpt-oss-120b.service"); YAML export
`~/AI/services/gpt-oss-120b-cpumoe.profile.yaml`; SQL profile `gpt-oss-120b-swap`
(port pin only, the model's `default_profile_id`).

## 4. Bug found and fixed (branch `fix/p3-lane-owned-sessions`)

`llmctl plan gpt-oss-120b-mxfp4` (obs, before the fix): profile `-`, **port
8000** (the DeepSeek lane's), **1 GPU**, command without any of the unit's
flags. It was refused only because the GPUs were busy; `start --force`, or a
free card, would have launched a second llama-server on the lane's weights.
Two causes:

1. `start`/`plan` ignored the model's `default_profile_id` (`cli.py`
   `_build_start_request`), so the `:8005` pin never applied. `run` and
   `llmctl model plan` already applied it.
2. Only `llmctl run` checked `lane_for_model`. `start`, `plan`, POST
   `/sessions` and the TUI start went straight to the scheduler.

Fix: the scheduler refuses a llama.cpp model a lane serves and names the lane
(`llmctl run <lane>`); `start`/`plan` fall back to the default profile. Tests:
`tests/test_scheduler_lane_owned.py`.

## 5. Proposed migrations (need approval; none done)

| # | Change | Effect on live values | Rollback |
| --- | --- | --- | --- |
| M1 | Mark the 4 profile envs generated: `router-profiles --write` adds a `# generated from preset <x>; edit the preset` header; run the read-only check on the existing reconcile timer so drift is logged. | None (comment + check only). | `git` in `~/AI`. |
| G1 | Archive SQL profile `gpt-oss-120b-cpumoe` and `services/gpt-oss-120b-cpumoe.profile.yaml` (`llmctl profile export` to `services/.attic-20261105/` first, consistent SQLite backup). Keep `gpt-oss-120b-swap` (the default link). | None: nothing launches from them (`run` → lane; `start` now refuses). | re-import the export; restore the SQLite backup. |

Counts after M1 + G1: editable launch definitions for live workloads **13 → 7**
(the 4 profile envs become generated; gptoss 4 → 2). Unit copies, GPU
hand-over implementations and restart paths are unchanged by this pilot.
