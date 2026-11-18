# Rust successor — M4c progress (coordination)

Parent: [RUST-M4-PLAN.md](RUST-M4-PLAN.md), sub-milestone M4c (seven items).
M4b: [RUST-M4B-PROGRESS-2026-11-17.md](RUST-M4B-PROGRESS-2026-11-17.md).

## Done

| Date | Piece | Evidence |
|---|---|---|
| 2026-11-17 | **On-demand GPU claim** (item 2, from the M4b window finding): `gpu_claim: at_start \| on_demand` per workload; adoption defaults Ollama and LM Studio to on_demand (`adopt --gpu-claim`). An on-demand service stops no conflicts and skips the foreign-occupancy guard at start; other plans never stop it; a plan that needs a card where it has a model loaded is refused with the unload to run. `dc9b260` | planner test, e2e (`on_demand_claims_pass_the_guard_and_name_the_unload`); live on the workstation pilot daemon: `llmster` and `ollama` marked on_demand — `run llmster --dry-run` and `restart vllm-engine --dry-run` each touch only their own unit |
| 2026-11-17 | **LM Studio device filter**: engine report and unload use `lms ls/ps --json` (null `deviceIdentifier` = this machine) | unit test; live: `engine llmster` shows nothing loaded locally while mbp2's 36.9 GB model is loaded via LM Link; `unload llmster <that model>` succeeded as a no-op and the model stayed loaded on mbp2 |
| 2026-11-17 | **Swap plans with rollback** (item 3, GPU part): a run or restart of an at_start workload is one plan — unload what running on-demand services hold on the target's cards, stop the at_start conflicts, start, verify. Rollback restores the evicted at_start set; unloaded models are not reloaded (the service loads them on its next request). Only models that hold VRAM are named: the daemon asks a service only when its unit occupies a target card (occupancy probe), and Ollama's list is `/api/ps` with `size_vram` > 0 (LM Studio: everything loaded on this device). Unknown lists plan nothing; the occupancy guard still refuses. `8ab1d56`, `2fdb72f` | 3 planner tests, 2 engine tests, e2e `swap_plan_unloads_on_demand_models_then_evicts_and_rolls_back` (unload → evict → failed start → restored; then success); suite 27 e2e / 48 daemon / 20 planner green, macOS check clean. Live on the pilot daemon (longjob `m4c-swap-dryrun`, 7 s): with `qwen3-embedding:0.6b` in RAM (`size_vram` 0) `restart vllm-engine --dry-run` names no unload; with `qwen3-embedding-0.6b-gpu` on GPU 0 (2.4 GB) it lists `unload … from ollama` before `stop vllm-engine`; after the unload the step is gone |
| 2026-11-17 | **Legacy flock + lane lease** (design decision 6). Every GPU operation (run, stop, restart, upgrade, unload on a workload with cards) takes the flock on `~/.local/share/llmctl/operations.lock` at admission — waits 5 s, then refuses `host_lock_busy` naming the real holder (`/proc/locks`, not a stale record) — and holds it until the operation is terminal. Under a fresh lane lease of a live holder it refuses `lane_leased`, unless the lease is on that workload or the caller runs inside the holder (CLI process ancestry → `llmctl-under-lease` header). Real engines on Linux only; `LLMCTL_NEXT_OP_LOCK` / `LLMCTL_NEXT_LANE_LEASE` = path or `off`. `0662a51` | 4 lease tests, 4 lock tests, e2e `gpu_operations_take_the_host_lock_and_honour_the_lane_lease`; suite 28 e2e / 52 daemon green. Live on the pilot daemon: with `flock(1)` holding the lock an unload was refused after 5 s (`pid … via flock (not recorded; lock file names pid 2004553, which does not hold it)`); free, the record read `interface: llmctl-next, purpose: unload ollama` and `longjob gpu` showed the lock free again; inside longjob `m4c-lease-check` (lease on GPU 0) the unload ran, the same command from outside was refused `lane_leased: longjob (pid …)`, and passed after the job |
| 2026-11-17 | **Request safety** (item 6). Before an operation stops a serving workload — or unloads a model from one as part of a run/restart — the agent asks llm-router (`GET /running`, `idle_revert.in_flight`) for requests in flight there, waits up to 20 s, then refuses `requests_in_flight` naming the busy workloads. `--force` on run/stop/restart cuts them. Only a positive count blocks; a router that is down or reports no counts blocks nothing. `LLMCTL_NEXT_ROUTER` = port or `off`, `LLMCTL_NEXT_INFLIGHT_WAIT_MS` = bound. `9def2a0` | 3 unit tests, e2e `stops_wait_for_in_flight_requests_or_refuse_without_force` (stand-in router: refused after the bound, admitted when requests finish inside it, eviction counted, forced admitted at once). Live, read-only: the daemon's reader got `{8003: 1}` from the real router during one `ccl --ask local` request and `{}` after it (`live_router_reports_in_flight_counts`, ignored test) |
| 2026-11-17 | **Lock pass for a caller that holds the flock** (needed by item 4). The CLI detects that an ancestor holds the operation lock (`/proc/locks`) and sends `llmctl-lock-holder`; the agent verifies the claim against the real holder and runs under that hold instead of refusing it. The caller must keep the lock until the operation is terminal (`--wait`). `08797eb` | e2e (true claim admitted and the holder's record untouched; false claim refused). Live: `flock -x operations.lock llmctl-next unload --wait ollama …` succeeded and the lock record did not change |
| 2026-11-17 | **Router handoff, lane backends** (item 4, part). llm-router (`6ce79fc` in `~/AI/llm-router`): with `switch.llmctl_next_cmd` set and a `start_cmd` backend naming its workload (`llmctl_next`), a switch runs `llmctl-next run <workload> --wait --force` instead of the start script. Exit 3/4/5/6 (agent down, refused at admission, workload unknown, protocol) falls back to the script; a failed operation is a failed switch. **Off by default; `config/router.yaml` is unchanged.** Also `4121863`: `--wait` now watches for the plan's readiness waits plus 5 min (was a flat 10 min, shorter than a Strata cold start) | 10 router tests (`tests/test_switcher_llmctl_next.py`); router suite 356 pass, 1 known failure (`test_every_backend_profile_is_tracked_in_the_repo`). Live: `run strata-fast --dry-run` on the pilot daemon shows the plan a handoff of `flash-fast` would run (stop vllm-engine, start, verify; rollback restarts vllm-engine). No handoff has run for real |
| 2026-11-17 | **Env-file profiles** (rest of item 3). `run <workload> --profile-env <file> --model <name>`: stop, make that file the workload's `EnvironmentFile` (old env kept as `<env>.llmctl-previous`), start, verify the profile's model, re-pin the workload (env digest + model, a new inventory revision). On failure the old env goes back; a target that was serving is restarted on its previous model and verified. A profile run tolerates drift that is only "the env contents changed" (what llm-router's own switch leaves behind). The profile must be a regular file named `<env file>.<profile>` next to it; the workload must be adopted with exactly one `EnvironmentFile`; the new env takes the stricter of the two file modes; contents never enter the journal, inventory, API or logs. `e735ba1`, `7310d60` | 3 planner tests, 4 profile tests, e2e `a_profile_run_swaps_the_env_repins_and_rolls_back` (switch, restart on the new pin, failed start restored and re-verified, five refusals); suite 30 e2e / 59 daemon / 23 planner green. Live, dry run only: `run vllm-engine --profile-env ~/AI/services/vllm-engine.env.qwen27b-awq --model qwen3.8-27b-awq --dry-run` lists stop, apply profile, start, verify `qwen3.8-27b-awq`, with the restore-and-restart rollback; `vllm-engine2.env` and `/etc/hostname` are refused; the env file's digest was unchanged afterwards |
| 2026-11-17 | **Router handoff, vLLM profiles** (item 4, complete in code). llm-router `1c1c566`: with `switch.llmctl_next_cmd` and `switch.llmctl_next_workload`, a profile switch is one `llmctl-next run <workload> --profile-env <profile> --model <served> --wait --force`; exit 3/4/5/6 falls through to the router's own switch; a failed operation fails the switch and the router's rollback does not run on top of llmctl-next's. **Off by default.** | 17 router tests in `tests/test_switcher_llmctl_next.py`; router suite 363 pass, 1 known failure. No handoff has run for real |

## Next, in order

1. **Enable the handoff and run the pilot** (owner's go). In
   `~/AI/llm-router/config/router.yaml`: `switch.llmctl_next_cmd`
   (the binary plus `--data-dir`), `switch.llmctl_next_workload:
   vllm-engine`, and `llmctl_next: strata-fast` / `gpt-oss-120b` on the
   `flash-fast` / `gptoss` backends; restart llm-router. Then one real
   switch through the router with a rollback forced (stops the tutor for
   minutes; start the window with `longjob start NAME --router`). Before
   that the pilot daemon should run under a unit: today it is a detached
   process, and while it is down every switch falls back to the old path.
2. **Companion hooks** (item 5), **manual-stop protection against the
   healers** (item 7, including "a swap's rollback never restarts a
   manual stop"), **model acquisition** (item 1), **VRAM fit** (rest
   of item 2).

## Open points

- **Swap plans, rest of item 3:** profiles (env-file sets) and llama-swap
  groups are not in the swap plan yet; no real swap has been executed on
  the workstation (dry runs only) — that is the M4c pilot.
- **Unload then start is not atomic:** a request can load a model again
  between the unload step and the target's start. The start then fails on
  VRAM and the plan rolls back. Request safety (item 6) closes this.
- **Rollback does not reload models.** Decided here, not by the owner:
  reloading a large model while the evicted engines restart could itself
  fail on VRAM. Say so if a reload step is wanted.
- **Ollama swaps on its own:** the main Ollama runs
  `OLLAMA_MAX_LOADED_MODELS=1`, so loading one model evicts the other (seen
  in the live check: the test model replaced `qwen3-embedding:0.6b`).
- The Mac agents were not upgraded for this change (no cards there, so
  the plan is the same).
- **Dry runs do not look at the lock or the lease:** a preview changes
  nothing; the refusal comes at admission.
- **Recovery under a lease:** the watcher's recovery run is refused like
  any other operation and logs `recovery refused` each tick until the
  lease ends (the same stand-down llm-heal does).
- `tests/launchd_spike.rs` does not compile under `--cfg=llmctl_macos`
  (`WorkloadDefinition` gained `engine_kind` in M4a). Not caused by M4c;
  the daemon-only macOS check is clean.
- **Request safety, limits:** an explicit `unload` is exempt (it names
  one model on purpose). `upgrade promote` waits like a stop but has no
  force flag. Requests that bypass llm-router are not counted.
- **Handoff and the router's own stops:** the router still runs its
  `stop_first` and Ollama eviction for vLLM switches, and the lane start
  scripts still stop engines themselves when the handoff falls back.
- **Profiles, limits:** one backup slot per env file (it only undoes
  the running operation); workloads with several `EnvironmentFile`
  entries cannot switch; a switch made by the router's own path still
  leaves the pin stale until the next profile run re-pins it (plain
  `run`/`restart` of that workload are refused as drift meanwhile, as
  before). No real profile switch has been executed on the workstation.
