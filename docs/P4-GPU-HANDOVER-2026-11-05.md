# P4: GPU hand-over, survey and proposal (2026-11-05)

Inventory §6 counted 11 implementations of "free the cards, then start an
engine" (plus `Conflicts=` as a 12th, declarative layer). This is the survey
behind a proposal; nothing here is changed yet. (obs) observed today, (src)
read in source.

## 1. What the 11 really are

| # | Implementation | Direction | Notes |
| --- | --- | --- | --- |
| 1 | router `stop_first` list | lanes → vLLM | hand-kept unit list in `router.yaml` |
| 2 | router `evict_ollama` | Ollama → vLLM | |
| 3 | llmctl `stop_lanes_for_vllm` / fleet preflight | lanes → vLLM | direct path only |
| 4 | llmctl `run_lane_swap` | → lane | calls the lane's start script |
| 5 | `strata-start` (126 lines) | → Strata lane | also serves `strata-fast`, `strata-uncensored`, `dsv41`, which are 20–25-line wrappers (src) |
| 6 | `glm53-start` (112) | → GLM lane | |
| 7 | `gptoss-start` (100) | → gpt-oss lane | |
| 8 | `coder1-start` (158) | → GPU1 Ollama lane | also `qwen3coder30b-start`, `gemma1-start` links |
| 9 | `restore-vllm` (34) | lane → vLLM | pauses Ollama while vLLM loads |
| 10 | `vllm-on-gpus` | helper | already shared by #5–#8 |
| 11 | `llm-backend-guard` | `ccl` only | not a GPU swap path; drop from the count |

So the lane side is **4 copies of one block** (#5–#8): detect what holds the
cards → refuse without `--take-gpus` → stop `vllm-engine` (sudo -n),
`vllm-engine2`, sibling lanes, resident Ollama models → poll `nvidia-smi`
until the cards drain → start the unit → wait for health.

## 2. Findings (no live bugs)

- **"Which lane holds which cards" is written down five times:** `router.yaml`
  `gpus:`, unit `Conflicts=`, each script's sibling list, router `stop_first`,
  llmctl's lane catalog. They agree today (obs).
- **Sibling lists look incomplete but are not:** `glm53-start` omits the fast
  Strata lane and DeepSeek; `gptoss-start` stops no Strata/GLM lane. systemd
  `Conflicts=` is bidirectional and the union of the units' lines covers every
  lane pair; the one pair it misses (Strata ↔ `ollama-gpu1`) is stopped
  explicitly by `strata-start` (src).
- **Drain checks differ** (any-card max vs per-card, 4000 vs 2000 MiB, 20 vs
  30 s). For the two-card lanes the any-card check is equivalent.

## 3. Proposal

**Step A — one lane-side helper (4 → 1).** `services/lanes/lane-take-gpus
LANE [--take|--share]`, implemented once (Python, in llmctl-stack, which already
parses `router.yaml` lanes and their `gpus:`), exposed as `llmctl lane
take-gpus LANE`. It derives the units to stop from `router.yaml` (lanes whose
`gpus:` overlap, plus `vllm-on-gpus` engines), evicts Ollama only where the
script did, drains per card with one threshold, and returns the same exit codes
(0 free, 3 busy without `--take`, 1 a stop failed). The four scripts keep their
own start/health/readiness code and call it in place of their detect/stop/drain
block.
- Must **not** take the GPU operation lock: the router holds it while it runs a
  lane's `start_cmd` (consolidation plan §7 step 2: no router → wrapper →
  router loop, no recursive lock).
- Pilot: `gptoss-start` first; others only after a live swap passes.

**Step B — one sibling list.** Generate the router's `stop_first` from the
`router.yaml` lanes (same source as Step A) and keep `Conflicts=` as the
systemd-level backstop, documented as derived.

**Step C — leave alone:** llmctl's direct vLLM path (one-card placement only it
does) and `restore-vllm` (one caller pattern, 34 lines).

Counts after A+B: lane-side implementations 4 → 1; places that list lane/card
ownership 5 → 2 (`router.yaml` + `Conflicts=` backstop).

## 4. Verification plan

- Unit tests for the helper with stubbed `systemctl`/`nvidia-smi`/`ollama`.
- Live, one window: `llmctl run gptoss` from the tutor, then back
  (`llmctl run qwen27b`), comparing journal lines and timings with the
  2026-11-04 acceptance run. Costs ~10 min of GPU time with the tutor down.

## 5. Live pilot (2026-11-07)

`gptoss-start` now calls `llmctl take-gpus gptoss [--take-gpus|--share-gpus]`
in place of its inline stop/evict/wait block (`~/AI` `60a947cb`). Run as the
longjob job `p4-pilot-2`, 12:09:42 → 12:17:25.

| Step | Result | Baseline |
| --- | --- | --- |
| `llmctl run gptoss -y` (tutor → gptoss) | rc 0, 330 s; `vllm-engine` stopped by `sudo systemctl stop` from the router's tree (the script's only stop path is now take-gpus); gpt-oss on :8005 answered "OK"; GPUs 29.3 / 28.2 GB | 4 m 54 s (last load) |
| `llmctl run qwen27b -y` (gptoss → tutor) | rc 0, 99 s; `gpt-oss-120b.service` stopped, `vllm-engine` started; the tutor on :8003 answered "OK"; GPUs 26.6 / 26.0 GB | 97 s (2026-11-04) |

Observations:

- vLLM logged `EngineDeadError` during its drain stop at 12:09:54. The unit still
  stopped cleanly in 11 s ("Deactivated successfully"). The same error appears
  in 9 of 92 `vllm-engine` stops since 2026-10-20, so the change did not cause it.
- The first attempt (`p4-pilot`) ran under `longjob --gpu all`. The router
  refused at once: the job's own lane lease blocks every swap, and llmctl
  `run` for a lane goes through the router. Nothing was stopped. **Run swap
  tests as `longjob start NAME --router`, which claims the router but holds no lease.**

Pilot passed. Next: roll out to `glm53-start`, `strata-start` and `coder1-start`,
then Step B.

## 6. Rollout (2026-11-07/08)

`glm53-start`, `strata-start` and `coder1-start` now call `llmctl take-gpus`
the way `gptoss-start` does (`~/AI` `e4a48131`; 132 lines of copied
detect/stop/drain code removed). Two helper changes came with it (`e6ffe914`):
a lane resolves by its unit too (`strata-start` serves four lanes and knows
only its unit), and the raw VRAM check is skipped when the lane's own unit is
already up (no pointless 30 s wait when `ollama-gpu1` holds GPU 1). 133 package
tests pass. `strata-start` no longer reads `STRATA_GPUS`; its cards come from
`router.yaml` (all four Strata lanes say `0 1`).

Live runs, all through `llmctl run LANE -y` (router `/admin/switch`):

| Step | take-gpus hand-over | Load | Answer |
| --- | --- | --- | --- |
| tutor → glm53 (`p4-rollout`, 14:51) | `vllm-engine` stopped, cards free in 13 s | 17 m 21 s (last: 1 m 00 s) | "OK" |
| glm53 → flash | `glm-5.3-flash` stopped in 16 s | **failed**: Strata's 900 s READY timeout | — |
| flash (down) → qwen3coder30b | nothing to stop | 22 s | "OK" |
| qwen3coder30b → tutor | `ollama-gpu1` stopped (`Conflicts=`/router) | 92 s | "OK" |
| tutor → flash (`p4-flash-retest-2`, 01:32) | `vllm-engine` stopped, unit started in 11 s | 12 m 11 s (last: 1 m 29 s) | "OK" |
| flash → tutor | flash stopped | 113 s | "OK" |

Observations:

- Every hand-over took 11–16 s. The slow loads were the Strata engines reading
  their weights from disk (~20 MB/s) with swap full (14.9 / 14.9 GB) and the
  page cache cold: the 1–1.5 min baselines were warm-cache loads. The flash
  failure was its own load timeout, not take-gpus; the removed `strata-start`
  block held no RAM or cache logic.
- A *queued* `longjob queue NAME --router` job still takes the lane lease when
  the runner starts it, so the 22:00 retest (`p4-flash-retest`) was refused
  after 1 s with nothing stopped. Start router swap tests directly
  (`longjob start NAME --router`).
- Open: the full swap slows every Strata cold load; with the 900 s READY
  timeout, a cold flash start can fail. Worth its own look.
  Closed 2026-11-08: `STRATA_ENGINE_READY_S=1500` on the flash units
  (`~/AI` `aca3d232`). The next flash start (10:15:50) reached READY in
  674 s on the first try with swap at 13/14 GiB.

Rollout passed. Next: Step B (generate the router's `stop_first` from the
`router.yaml` lanes).
