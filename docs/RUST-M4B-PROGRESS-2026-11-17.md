# Rust successor — M4b progress (engine adapters and llama-swap, 2026-11-17)

Parent: [RUST-M4-PLAN.md](RUST-M4-PLAN.md), sub-milestone M4b. M4a review
first ([pilot](RUST-M4A-PILOT-2026-11-17.md)): sound, one regression found
and fixed (below). Code: `2e9af4d`..`35a3925` on `main`.

**Status: code for steps 1, 4 and 5 done and proven live on both Macs;
steps 2 and 3 (Linux units, system → user unit move) wait for GPU windows
and owner decisions — see "Open".**

## M4a review

- Suite green; the systemd spike suite (20 tests, disposable user units)
  found one regression: `6c69a1e` re-validated a staged python-env before
  promote with an **empty model config**, so every vLLM promote would have
  been refused. Fixed in `2e9af4d` (stage and promote now share
  `validation_model`); spikes 20/20.
- Carried into M4b as planned: the m5 mlx-serve swap (needs lane records)
  and the launchd tiny-archive/C spikes.

## What was built

| Step | Piece | Evidence |
|---|---|---|
| 1 | `engine_api`: readiness + servable/loaded models per protocol — OpenAI `/v1/models` (vLLM, llama.cpp, MTPLX, mlx-vlm, mlx-serve, Strata, ds4), Ollama `/api/version`+`/api/tags`+`/api/ps`, LM Studio `/api/v0/models` (`state`); a vendor workload may name the service itself (`ollama`, `lmstudio`) | unit tests against fake servers; systemd + launchd probes use it |
| 1 | `adopt --engine-kind` (picks the protocol, recorded on the workload) | m5 Ollama adopted with it (below) |
| 1 | `unload <workload> <model>`: admitted op (`unload` kind, one `UnloadModel` step, re-planned under the admission lock, no rollback), confirmed through the engine's loaded list; refused for single-model engines before anything journals | planner tests, e2e refusal test, **live on m5 Ollama** |
| 1 | `engine <workload>`: protocol, state, servable, loaded | live on m5 Ollama |
| 4 | `swap` records: import → ordered globals/macros/matrix/lanes (engine kind inferred from `cmd`), every comment and the owner's quoting/flow/blank-line layout kept by key path; `roundtrip` proof (render parses to the same tree **and** re-imports to the same records); `validate` = llama-swap's own refusals incl. the matrix DSL ported from `swapmatrix.py`, plus the eviction solver | 8 tests on sanitised fixtures of both Macs; the live commented configs round-trip (243/24 comments kept; renders differ from the hand-written files by 16 and 1 whitespace/comment-position lines) |
| 4 | `lanes import / show / set / unset / macro / rewrite`: import previews and persists on the preview digest (config untouched); edits plan (changes, validation, lanes a reload restarts and preloads) and write on the plan digest as a `lane_write` op — refuse a file changed since the plan, atomic write (previous bytes kept as `config.yaml.llmctl-previous`), wait for llama-swap's `configuration reloaded` (an `[ERROR]` about the config fails it), check `/v1/models`, re-import the store, `git commit` the file in its repo; rollback writes the previous bytes and confirms the same way; a hand edit is drift and planning refuses until re-import | e2e against a fake llama-swap (write+commit, rejected write rolled back, drift, validation); **live on mbp2** |
| 5 | Mac vendor service: m5 Ollama re-enrolled from observe-only (M4a) to managed via `adopt --engine-kind ollama` (no restart), vendor-app recipe re-attached and pinned | `install plan ollama`: running 0.34.3, pinned 0.34.3, no drift |

## Live evidence (2026-11-17, ~13:40–13:50 PST)

Agents on mbp2 and m5 upgraded to the M4b build (`host upgrade`, built on
mbp2, relayed to m5). Nothing was loaded in either llama-swap.

- **mbp2 lanes:** import — round trip equivalent, 26 comments kept, valid,
  2 lanes (`mtplx`). `lanes rewrite` (normalise to the render) → operation
  `c7b1c1cc…`: llama-swap logged `configuration reloaded`, `/v1/models`
  listed both lanes, config repo commit `7269714`
  (`llmctl-next(llama-swap): re-render`), diff = one blank line; `/health`
  OK.
- **m5 llama-swap enrolled:** M4a had recorded `mlxserve` on llama-swap's
  launchd label (a stopgap before lane records), which blocked adoption.
  Inventory backed up (`inventory.json.bak-m4b-20261117`), stopgap record
  removed (its recipe is in the backup and the M4a note), llama-swap adopted
  as `llama-swap` (managed, pid unchanged, all six checks pass).
- **m5 lanes:** import — equivalent, 243 comments kept, valid, 13 lanes
  typed `ds4` ×2, `mtplx` ×6, `mlx_vlm`, `mlx_serve`, `forwarder` ×3. No
  write on m5 yet (open item 1).
- **m5 Ollama unload:** loaded `nomic-embed-text` via `/api/embed`, then
  `unload ollama nomic-embed-text:latest --wait` → succeeded
  (`unloaded:launchd/com.pitcany.ollama:…`), loaded list empty after.

Fixes found by the live run (each committed): unload keys with `:`
(model tags) were refused as idempotency keys; a recipe attached after
adoption kept adoption's program-digest pin and read as drift — startup now
re-pins when the pin lacks a per-kind fact.

## Open (needs the owner or a GPU window)

1. ~~m5 mlx-serve 26.10.1 → 26.9.5 swap~~ — **done 13:55 PST** (owner
   go-ahead): `lanes macro llama-swap mlxserve <26.9.5 path>` → plan = one
   macro change, reload preloads `mimo9b-m5`, `qwen35b-m5`; op succeeded
   (`configuration reloaded`, 13 lanes listed), config repo commit
   `2c6d705` (18 lines: the macro plus the render's whitespace/comment
   normalisation). mimo9b-m5 runs `mlx-serve-26.9.5`, health check passed,
   a chat request answered `ok`; qwen35b-m5 healthy. The lane's `name`
   label was corrected the same way at 13:57 (`lanes set … name`, commit
   `2a476af`, 1-line diff; the reload restarted both running lanes, no
   traffic at the time). Swap back = the same `lanes macro` command with
   the 26.10.1 path.
2. **Step 2, Linux units:** Strata (one lane) and `ollama-gpu1` need the
   cards (vLLM TP=2 holds both; the GPU guard refuses rightly) → a longjob
   `--stop-engine` window. `llmster` under a new user unit stops the
   unsupervised process on :1234 first.
3. **Step 3, system → user units** (`vllm-engine`, main `ollama`): sudo,
   one GPU window, llm-router restart command + llm-heal targets + sudoers
   changed together; system units disabled (not deleted) for a week.
4. **LM Studio on the Macs:** no launchd job to adopt (app-run server);
   enroll observe-only and record the console-login limit.
5. **Capability-matrix rows** per engine × host as disposable spikes
   (launch, readiness, stop, logs), and the launchd tiny-archive/C spikes.
6. Per-lane recipes/versions: today an engine version swap is a macro path
   change (decision 3 of the plan); typed per-lane recipe records are not
   built.
