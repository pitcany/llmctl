# Rust successor — M4c progress (coordination)

Parent: [RUST-M4-PLAN.md](RUST-M4-PLAN.md), sub-milestone M4c (seven items).
M4b: [RUST-M4B-PROGRESS-2026-11-17.md](RUST-M4B-PROGRESS-2026-11-17.md).

## Done

| Date | Piece | Evidence |
|---|---|---|
| 2026-11-17 | **On-demand GPU claim** (item 2, from the M4b window finding): `gpu_claim: at_start \| on_demand` per workload; adoption defaults Ollama and LM Studio to on_demand (`adopt --gpu-claim`). An on-demand service stops no conflicts and skips the foreign-occupancy guard at start; other plans never stop it; a plan that needs a card where it has a model loaded is refused with the unload to run. `dc9b260` | planner test, e2e (`on_demand_claims_pass_the_guard_and_name_the_unload`); live on the workstation pilot daemon: `llmster` and `ollama` marked on_demand — `run llmster --dry-run` and `restart vllm-engine --dry-run` each touch only their own unit |
| 2026-11-17 | **LM Studio device filter**: engine report and unload use `lms ls/ps --json` (null `deviceIdentifier` = this machine) | unit test; live: `engine llmster` shows nothing loaded locally while mbp2's 36.9 GB model is loaded via LM Link; `unload llmster <that model>` succeeded as a no-op and the model stayed loaded on mbp2 |

## Next, in order

1. **Swap plans with rollback** (item 3): one plan = unload on-demand
   models on the target's cards, stop at_start conflicts, start, verify,
   roll back to the previous set — needs loaded-model facts in observations.
2. **Legacy flock + lane lease** (design decision 6): take
   `~/.local/share/llmctl/operations.lock` and honour the lane lease for
   every GPU operation on the workstation.
3. **Request safety** (item 6) and **router handoff** (item 4): in-flight
   counts from llm-router before a stop/swap; the router calls llmctl-next
   for owned workloads.
4. **Companion hooks** (item 5), **manual-stop protection against the
   healers** (item 7), **model acquisition** (item 1), **VRAM fit** (rest
   of item 2).
