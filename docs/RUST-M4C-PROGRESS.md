# Rust successor — M4c progress (coordination)

Parent: [RUST-M4-PLAN.md](RUST-M4-PLAN.md), sub-milestone M4c (seven items).
M4b: [RUST-M4B-PROGRESS-2026-11-17.md](RUST-M4B-PROGRESS-2026-11-17.md).

## Done

| Date | Piece | Evidence |
|---|---|---|
| 2026-11-17 | **On-demand GPU claim** (item 2, from the M4b window finding): `gpu_claim: at_start \| on_demand` per workload; adoption defaults Ollama and LM Studio to on_demand (`adopt --gpu-claim`). An on-demand service stops no conflicts and skips the foreign-occupancy guard at start; other plans never stop it; a plan that needs a card where it has a model loaded is refused with the unload to run. `dc9b260` | planner test, e2e (`on_demand_claims_pass_the_guard_and_name_the_unload`); live on the workstation pilot daemon: `llmster` and `ollama` marked on_demand — `run llmster --dry-run` and `restart vllm-engine --dry-run` each touch only their own unit |
| 2026-11-17 | **LM Studio device filter**: engine report and unload use `lms ls/ps --json` (null `deviceIdentifier` = this machine) | unit test; live: `engine llmster` shows nothing loaded locally while mbp2's 36.9 GB model is loaded via LM Link; `unload llmster <that model>` succeeded as a no-op and the model stayed loaded on mbp2 |
| 2026-11-17 | **Swap plans with rollback** (item 3, GPU part): a run or restart of an at_start workload is one plan — unload what running on-demand services hold on the target's cards, stop the at_start conflicts, start, verify. Rollback restores the evicted at_start set; unloaded models are not reloaded (the service loads them on its next request). Only models that hold VRAM are named: the daemon asks a service only when its unit occupies a target card (occupancy probe), and Ollama's list is `/api/ps` with `size_vram` > 0 (LM Studio: everything loaded on this device). Unknown lists plan nothing; the occupancy guard still refuses. `8ab1d56`, `2fdb72f` | 3 planner tests, 2 engine tests, e2e `swap_plan_unloads_on_demand_models_then_evicts_and_rolls_back` (unload → evict → failed start → restored; then success); suite 27 e2e / 48 daemon / 20 planner green, macOS check clean. Live on the pilot daemon (longjob `m4c-swap-dryrun`, 7 s): with `qwen3-embedding:0.6b` in RAM (`size_vram` 0) `restart vllm-engine --dry-run` names no unload; with `qwen3-embedding-0.6b-gpu` on GPU 0 (2.4 GB) it lists `unload … from ollama` before `stop vllm-engine`; after the unload the step is gone |

## Next, in order

1. **Legacy flock + lane lease** (design decision 6): take
   `~/.local/share/llmctl/operations.lock` and honour the lane lease for
   every GPU operation on the workstation.
2. **Request safety** (item 6) and **router handoff** (item 4): in-flight
   counts from llm-router before a stop/swap; the router calls llmctl-next
   for owned workloads.
3. **Companion hooks** (item 5), **manual-stop protection against the
   healers** (item 7), **model acquisition** (item 1), **VRAM fit** (rest
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
