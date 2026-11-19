# Rust successor — M5 progress log

Work log for [RUST-M5-PLAN.md](RUST-M5-PLAN.md). Rows are dated; newest
at the bottom of each section. Status line first.

**Status 2026-11-18 (end of day):** M5a built, deployed to all three
agents, baseline day running. M5b's guards built, tested and committed on
every writer (Python guard `8f30377`, take-gpus `26e52b8d` in llmctl-stack,
Mac llmctl `5dd7415` in local-llm); the workstation setting flips after
the baseline day. `~/AI/.venv` and `llmctl-api` run the guard build.
M5c started: the preset line editor landed (`aae79aa`).

## M5a — workflow matrix and the foreign-write record

| Date | What | Verified |
|---|---|---|
| 2026-11-18 | **Workflow matrix** `docs/RUST-M5-WORKFLOWS.md`: reads, writes, config/catalog, background writers, M4 carry-overs, pilot-day rows. Facts read from the tools named in it | owner review pending; the exit check walks it |
| 2026-11-18 | **`foreign_transition` findings** (`9472099` docs, `1829c31` code): every observation pass (status call, 15 s watcher tick) compares each managed workload against the last recorded state (process, pid, served model; llama-swap lanes where a lane store exists). Unexplained → finding in the journal DB (30-day retention), `llmctl-next audit [--since] [--all] [--json]` + `GET /v1/audit?since=`. Agent restart keeps findings and baselines (SQLite). `Observation` carries the manager's main pid (systemd `MainPID`, launchd `pid =`); unknown never concludes; a llama-swap that cannot be asked skips the lane comparison; a TTL vanish names `llama-swap (ttl: …)` as the actor | daemon e2e: `a_foreign_flip_is_a_finding_and_an_operated_flip_is_not` (op-driven flip → no finding; foreign stop → finding; foreign restart with new pid → finding; external workload → never a finding; CLI verb + `--since`), `agent_restart_keeps_findings_without_duplicating`; unit tests for the comparison rules; full workspace suites green (e2e 42, real-systemd spike 22 serial, core/client/daemon, lanes, drills) |
| 2026-11-18 | **Live deploy**: release binary installed as the pilot agent (`llmctl-next.service`, m4a data dir; old binary kept at `llmctl-next.bin.pre-m5a` in /tmp), Mac agents via `host upgrade mbp2 --source`, `host upgrade m5 --binary` (sha256 `796209fb…` both) | `audit --all` answers on all three hosts, zero findings, baselines recorded; `status` unchanged |
| 2026-11-18→ | **Baseline day** running since ~18:32 UTC. Nothing is delegated yet (M5b guards ship disabled), so the baseline counts today's real foreign writes | check `llmctl-next audit --all` after ≥ 24 h; that count is the M5b before-picture |

Known limits (by design, said in `rust/crates/llmctl-daemon/src/audit.rs`):
a manager crash-restart (new pid, still running) records as a pid change
with no actor; a silent engine-side model unload is not caught (probes
cannot distinguish it from unknown); the actor is only named for
llama-swap TTLs.

## M5b — delegation of the legacy writers

All guards ship **off**; each reverses with one setting. Enable the
workstation section after the M5a baseline day completes (24 h clean or
the count accepted):

```yaml
llmctl_next:
  enabled: true
  cmd: [/home/yannik/.local/state/llmctl-next-m4a/llmctl-next.bin, --data-dir, /home/yannik/.local/state/llmctl-next-m4a]
  units:
    vllm-engine.service: vllm-engine
    gpt-oss-120b.service: gpt-oss-120b
    ollama.service: ollama
    ollama-gpu1.service: ollama-gpu1
    llmster.service: llmster
    qwen3.8-flash-next-fast-strata.service: strata-fast
```

| Date | What | Verified |
|---|---|---|
| 2026-11-18 | **Python guard** (`8f30377`, `llmctl/integrations/llmctl_next.py` + `SystemctlRunner.run`): write verbs on a delegated unit go to `llmctl-next run/stop/restart <workload> --wait`; rc 3/4/5/6 → honest failure naming the reversal setting, never a systemctl fallback; read verbs never delegate; CLI, TUI and `:8088` inherit it (one chokepoint). Unit lookups normalize bare and `.service` forms (security review 2026-11-18: a near-miss key must never defeat the gate). `LLMCTL_NEXT_CMD`/`LLMCTL_NEXT_DISABLED` for tests | `tests/test_llmctl_next_guard.py` 7/7 with a stand-in binary (delegated argv, refusal on rc 3 with no fallback, rc 7 ≠ refused, unowned → systemctl, both unit-name forms, disabled/absent settings, env override); full Python suite 1815 passed |
| 2026-11-18 | **take-gpus guard** (llmctl-stack `26e52b8d`): `stop_blockers` delegates owned units to `llmctl-next stop`; ImportError (old llmctl pin) falls back to the old path for everything | stack tests: owned → agent argv only; rc 3 → failed stop, no systemctl; unowned → `systemctl --user stop`. 16 pre-existing take_gpus tests unchanged-green |
| 2026-11-18 | **Mac llmctl guard** (local-llm `5dd7415`, m5 + mbp2): `next_enabled: 1` routes `start/stop` of an owned lane through `llamctl-next lanes swap/stop --wait`; `--dry-run` untouched (exits in start_guard); agent unreachable → refusal with the lever | live on m5: T1 bin missing → die; T3 agent up, owned lane → delegated ("not running: nothing to stop", exit 0); T4 agent down (rc 3) → "refusing to touch llama-swap directly… next_enabled: 0". mbp2 merged, default off (its llama-swap workload is not agent-owned yet) |
| 2026-11-18 | **Router item 4 — not done, owner-accepted gap**: `qwen3coder30b` maps to owned `ollama-gpu1`, but `run ollama-gpu1` does not preload `qwen3-coder:30b`; the router's Ollama readiness reads `/api/ps`, so a handover breaks the lane. Stayed on the script; its take-gpus stops now delegate (item 2), its final `systemctl start ollama-gpu1.service` remains a foreign write when that lane is used | reasoning in B4 of the workflow matrix; revisit with M5c catalog work |

### Left to do in M5b

1. Enable the workstation `llmctl_next:` section (snippet above) after
   the baseline day; enable `next_enabled: 1` on m5 at the same time.
2. One day of normal use with `audit` on: the foreign-write count for
   owned workloads should fall to zero.

Done the same evening: `~/AI/.venv` reinstalled from the guard commit
(`uv pip --no-deps git+file:///home/yannik/Projects/llmctl@8f30377`),
`llmctl-api` restarted and healthy with the guard importable (disabled).
When the owner pushes `8f30377`, move the llmctl-stack pin
(`packages/llmctl-stack/pyproject.toml`, git rev `c802e7c…`) to it and
re-run `uv lock` in `~/AI`, so fresh installs resolve the same build.

## M5c — preset/config parity and catalog operations

| Date | What | Verified |
|---|---|---|
| 2026-11-18 | **Preset line editor** (`aae79aa`, `llmctl-core/src/presets.rs`): the Rust port of `llmctl/presets/yaml_edit.py` — span editing of top-level keys, trailing comments kept through value changes, deletions and appends, self-check refusing a line edit that does not reload to the desired mapping (fallback-to-dump stays the caller's job, as in Python) | core unit tests: byte-identical round trip of the live qwen27b preset's shape, comment/list/nested-mapping preservation, add/delete, block-scalar multiline, non-mapping refusal, empty-to-empty contract |
| 2026-11-18 | **Preset reader/writer + verbs** (item 1): `llmctl-core/src/preset_launch.rs` ports the v1 schema, `model_to_launch_spec`, `render_vllm_env` (Python float/JSON spelling, systemd escaping) and router_profiles' `parse/diff/merge_env` + generated header; agent `presets.rs` reads `presets/*.yaml` (canonical then legacy dir, as Python), `settings.yaml` `vllm.defaults` only, router.yaml's `profile_env` backends. Verbs `preset list/show/diff/edit/new/sync/settle`, `run <preset>` (implied when the name is no workload; `--preset`, `--sync`, `--workload`). API `GET /v1/presets[/{alias}]`, `POST /v1/presets/{edit,new,run,settle}` (DTOs in `llmctl_client::catalog` for M5e). `run <preset>` = ordinary profile swap (plan, fit, rollback); a profile whose values drifted is refused (`preset_drift`, exit 4) until `--sync`/`preset sync` merges the render in (comments kept, previous bytes at `.llmctl-previous`); a comment-only difference is no switch, so no restart. **vLLM version source settled:** the render's Python-root fallback is the workload's pinned interpreter, and `preset settle [--write]` rewrites the three comment patterns (env header "The interpreter is the … env (vLLM …", router.yaml "Runs from the … interpreter", "Keep this file in sync with the llmctl preset <path>") to the pin; settings.yaml's `runtimes.vllm.binary` (vllm-serve) is not read by any llmctl-next path. Live files not rewritten here (the agent with this verb is not deployed) | `presets_e2e` (temp agent: list/invalid file, render facts, drift refused → `--sync` run swaps, edit keeps comments + backup, invalid edit never written, sync keeps owner comments, comment-only = no switch, `new --from` keeps comments, settle preview/write/idempotent, workload names still run workloads); core unit tests (py float/JSON, escaping, splitlines, merge/diff, settle patterns, schema refusals); full workspace suite green |
| 2026-11-18 | **Parity check** (item 2): `tests/fixtures/rust_parity/` holds sanitised copies of the 5 live presets and the 4 loader-test shapes (`/home/yannik` → `/home/owner`); `generate.py` records what Python renders (each preset with built-in and with non-default `vllm.defaults`), what `edit_preset_text` makes of six intended changes per file plus the `test_preset_save_keeps.py` inputs, and the frozen `vllm_env__*` spec captures, into `parity.json`. Rust `llmctl-core/tests/preset_parity.rs` must match every entry byte for byte; `tests/test_rust_preset_parity.py` re-runs the generator and fails when Python moves off the file. **Found and fixed two bugs in the `aae79aa` line editor**: spans counted `end += 1` where Python sets `end = j + 1` (an indented line after a comment/blank got a short span), and replacements were spliced in alphabetical key order instead of line order (a panic on `save_keeps`' comments case) | cargo `preset_parity` 3/3: 18 renders, 58 edits (9 byte-identical round trips), 8 specs; pytest `test_rust_preset_parity` 2/2 plus `test_preset_save_keeps`/`test_vllm_env_parity` green |
| 2026-11-18 | **Catalog** (item 3): `llmctl-core/src/catalog.rs` (records, row mapping, reverse export) + agent `catalog.rs` (`<data>/catalog.json` per host). `catalog import <bundle> [--as HOSTNAME]` reads a Python registry bundle (`llmctl export-registry`), dry run first: per row `new` / `no_store` / `present` / `unmapped` with the reason (another host's row, an OpenAI endpoint, an unknown runtime, a duplicate in the bundle); only the report's `--confirm` token applies (bound to bundle, host names and catalog revision). Unmapped rows and the launch profiles are kept verbatim, so `catalog export <file>` gives back the whole bundle. `model list/show/add/edit/enable/disable` per host (`--host` routes to a Mac agent); `model scan [--store] [--add]` walks hf_cache/directory/ollama/lmstudio stores for weights no record names and records whose weights are gone. `model get/rm` and `stores` unchanged. API under `/v1/catalog/…`. The Python DB is never written | `catalog_e2e` (temp agent: host mapping and `--as`, report counts and reasons, bad token refused, idempotent re-import, list/disabled/unmapped counts, add/dup/invalid/edit/unset/disable/enable, scan matches + gone + `--add`, export pinned to `tests/fixtures/rust_parity/catalog_export.json`); pytest `test_rust_catalog_export.py`: that export through `import-registry` in a temp DB gives the same rows and profiles as importing the original bundle, and keeps every row verbatim. **Live, read-only:** a copy of the real registry (DB `.backup` → `export-registry`: 74 live rows; the other 68 of 142 are `deleted` tombstones Python's export skips) into a temp agent with the real store declarations: new 53, no_store 1 (`gpt-oss-120b` GGUF under `~/models/gpt-oss-120b`, no store declared there), unmapped 20 (Mac LM Studio rows: 13 `Yanniks-MacBook-Pro.local`, 7 `macbookpro.lan` — import those on the Mac agents with `--as`); export → `import-registry` in a temp DB → same 74 rows and 2 profiles. `model scan` on that agent: 103 weights in the stores, 85 not cataloged, 0 gone |
| 2026-11-18 | **Mac lanes: matrix verb** (item 4, M4 carry-over 2): `llmctl-core/src/matrix_edit.rs` + `LaneEdit::Matrix`; `lanes matrix <w>` shows vars, costs, sets and the budget lines; `lanes matrix <w> var|rm-var|set|rm-set|cost …` plans through the ordinary lane plan → `--confirm` → `lane_write` operation (atomic write, reload confirmed, rollback). **The memory budget now guards the write:** a matrix change is refused without a recorded budget, for a new or changed set (not `risky*`) holding a lane with no peak figure (the sum would be a lower bound), and — as before for any lane edit — for a set that would go over. A config on `groups:` is refused (no silent conversion). `LaneStatusResponse` carries the matrix (for M5e) | lanes e2e `the_matrix_changes_by_verb_and_the_budget_guards_the_write` (fake llama-swap: no budget → refused; missing peak → refused; 65G > 50G refused, `risky-all` allowed; apply writes + reloads; eviction cost; var still in a set refused by validation; remove set); core unit tests (render, new matrix before `models:`, groups refused, budget sees the new set); full workspace suite green. Not run against m5/mbp2 (no agent deploy in this session) |
| 2026-11-18 | **Records by verb** (item 5, M4 carry-over 1): agent `records.rs`, DTOs in `llmctl_client::catalog`. `enroll <name> --observe URL` (observe-only: watched, `run`/`stop` refused) or `--start/--stop [--log] --port` (a Mac vendor app's own controls; Linux refuses it, the launchd adapter implements it), with `--engine-kind`, `--model`, `--gpu`, `--on-demand`; the unit identity is synthesised (`llmctl-next-record-<name>.service` / `com.pitcany.llmctl-next.record.<name>`). `authority <name> --observe | --systemd-unit [--scope] | --launchd | --start/--stop` changes an external record's authority (the unit identity follows a manager-backed one; an observe endpoint must match the record's port). Both preview first; the token (record + inventory revision) applies through `commit_inventory` (quiesced, validated, new revision). An adopted workload's authority is refused (`release` first) | `records_e2e` (temp agent: observe-only preview/confirm, run refused, name taken / no port / vendor app on Linux refused, authority to observe-only and back with a stale token refused and a port mismatch refused, adopted workload refused, unknown workload exit 5); unit test for endpoint ports and engine-kind spellings; full workspace suite green (`--no-fail-fast`) |
| 2026-11-18 | **Router item 4 revisited (from M5b)** — still not handed over, now with a smaller gap. The catalog names `qwen3-coder:30b` on the `ollama` store (it is one of the 53 rows the real-registry import maps), so "which model, on which store" is answered. What is still missing is the load itself: `run ollama-gpu1` starts the service but loads no model, and the router's Ollama readiness waits on `/api/ps`. Needed: a `load <workload> <model>` verb for vendor services (the mirror of `unload`: Ollama `POST /api/generate` with `keep_alive`, confirmed through `/api/ps`, the model checked against the catalog, the GPU claim planned as for M4c's on-demand claim), then the router backend can switch to `llmctl_next_workload: ollama-gpu1` plus that load. Until then the lane stays on `qwen3coder30b-start`; its final `systemctl start ollama-gpu1.service` remains the one known foreign write | read-only: router.yaml backend `qwen3coder30b` (`start_cmd` script, `unit: ollama-gpu1.service`), catalog rehearsal above |
| 2026-11-18 | **M5c check (this branch, not deployed)**: cargo workspace suite green (`--no-fail-fast`: core 52 + planner 27 + swap 11 + parity 3, daemon 93, client 4, e2e 42, lanes 5, presets/catalog/records 1 each); real-systemd spike suite `--ignored --test-threads=1` 22/22; full Python suite in 4 shards 1,820 passed, 1 skipped. Known flake, not from this branch: `e2e::bridge_subcommand_relays_http_to_the_socket` fails about 1 run in 10 on the base commit `bd90afa` too (the daemon closes a raw request unanswered) | logs in the session; base-commit check in a throwaway worktree |

### Left to do in M5c

1. Deploy (owner): merge, install the agent build on the workstation (the
   preset and catalog verbs need it), then `preset settle --write` to put
   the pin's words (vllm-muse, vLLM 0.30.0) into the env header, the
   profiles and router.yaml's comment; `preset list` should show every
   preset `ready` or `serving`.
2. Import the real registry on each agent: workstation
   `catalog import <bundle> --as yannik-desktop`, the Macs with
   `--host m5|mbp2 … --as Yanniks-MacBook-Pro.local|macbookpro.lan`
   (which name is which Mac: check before importing). Optionally declare
   a store for `~/models/gpt-oss-120b` (the one no-store row).
3. Owner decision 7: install `hf` on m5 (not done here: no host changes
   in this session).
4. Router item 4: the vendor `load` verb (row above).
5. Live use of `lanes matrix` on m5 (budget 104 GiB) once the agent there
   runs this build.

## M5d — llama.cpp on every host, several models per server

Built on branch `claude/m5d-llamacpp-c8b84c` with stand-in tests; **not
deployed, no live pilot yet** (that needs agents redeployed from merged
main and GPU windows via `longjob start --router`). Step 7 (`gpt-oss-120b`
into the router) is prepared as the runbook below, not run.

| Date | Item | What | Verified |
|---|---|---|---|
| 2026-11-18 | 1. `llamacpp-router` workload, model records | `llmctl-core/src/llamacpp.rs` (`1394e45`): preset INI → `RouterPreset` records (top keys, `[*]`, one section per model, every comment and blank line kept) with typed `RouterModel` views; render + round-trip proof (what llama.cpp reads must not change; canonical files render byte for byte); validation against llama.cpp's own parser rules (`common/preset.cpp`: key grammar, a value ends at `;`/`#`) plus deployment rules (absolute `.gguf`, no `host`/`port`/`alias`/`api-key`, no router options per model). `RouterLaunch::argv()` renders the router's own command line (no `-m`, `--models-preset`, `--models-max`, `--no-models-autoload`); `check_router_argv` refuses a model of its own, `--models-dir`, autoload on (flag or `LLAMA_ARG_MODELS_AUTOLOAD`), `--models-max 0`. `EngineKind::LlamaCppRouter` is an on-demand engine with model unload. `ModelRecord` is the catalog-facing type M5c consumes (`llamacpp::model_records`) | core `llamacpp_tests` 17/17: byte round trip of 3 fixtures (pilot preset, llama.cpp README example, the gpt-oss section), a non-canonical file (CRLF, `key=value`, indented notes) reads the same, edits keep notes, validation messages, argv checks |
| 2026-11-18 | 2. Verbs over the router API | `llmctl-next llamacpp import/models/load/unload/set/unset/add/remove/need/logs` (`37f2093`). `load`/`unload` admit a `router_model` operation (`POST /models/load|unload`, confirmed by `GET /models`; a load the router reports failed fails the step with its exit code and rolls back with an unload). Preset edits admit a `router_write` (atomic write, `<preset>.llmctl-previous`, `GET /models?reload=1`, every new model listed and every removed one gone, rollback to the previous bytes the same way, git commit when the preset lives in a work tree). Request safety: unloads wait on llm-router's in-flight count for the router's port; `--force` skips it | stand-in router e2e `llamacpp_e2e` 4/4 (see item 6) |
| 2026-11-18 | 2–3. Loads only after the fit (decision 9) | `decide_load`: workstation = per-card need (measured, or declared with `llamacpp need --mib`) against free VRAM with the 1 GiB headroom, plus host RAM for a CPU-offload model (`--host-ram`); Mac = every model holding memory plus this one within the budget at peak; refused at `--models-max` (the router would otherwise evict by itself); fails closed on every unknown (never measured, VRAM unreadable, router silent). After each load the VRAM drop is recorded as the measured need. Pre-load guard (port of `llama-server-guarded`): every shard of a split GGUF present, no `.incomplete` in the HF download stage, per-card VRAM floor (`--min-free`, successor of `LLAMA_GUARD_MIN_FREE_MIB`) | core decision tests (fit, CPU offload, budget, models-max, fail-closed); e2e: never-measured refused, does-not-fit refused, guard refusals for a missing shard / live download / floor |
| 2026-11-18 | 3. Planned against `vllm-engine` | Both directions. A router load that does not fit refuses with the reason; `--evict` plans the at-start workloads on the router's cards out first (intent `stopped`, stop) and starts them again if the load fails, using what each holds on the cards now (`gpu_occupancy`) to predict the fit. A `vllm-engine` start on the router's cards plans the router's loaded models out first (on-demand unload, the M4c swap rule; `f2be49d`), never stopping the router | core `a_vllm_start_unloads_router_models_on_its_cards_first`; daemon `the_llamacpp_router_lists_loads_and_unloads_one_model`; e2e `an_evict_load_stops_the_at_start_neighbour_and_restores_it_when_the_load_fails` |
| 2026-11-18 | 4. llama.cpp lanes on the Macs (decision 8) | `llmctl-next llamacpp lane <workload> <lane> --model <gguf> [--macro llamacpp] [--arg embedding] [--ttl N]` renders `${llamacpp} -m <gguf> --host 127.0.0.1 --port ${PORT} …` and plans it as a lane write (new `LaneEdit::AddLane`/`RemoveLane`, `ccf31b0`); the lane infers as `llama_cpp`, so `lanes swap/stop/version/memory` work unchanged | e2e `llamacpp_lanes_e2e`: lane planned, written, listed as `llama_cpp`, loaded by `lanes swap`, removed |
| 2026-11-18 | 5. Release-archive install on the Macs (closes M4 carry-over 4) | `llmctl-next llamacpp install <workload> <tag> [--recipe r.json] [--checksum sha256:…]` stages the macOS arm64 release into `<root>/<id>/` through the existing release-archive stager (remote archive refused without a sha256; checksum checked; binary must print `version: <build>`; the dir the lanes run from is never restaged). The recipe is kept per lane macro in the lane store. The switch is the printed `lanes macro` write, which re-pins the lanes it moves; the answer also prints the write back. brew's 1.0.0 stays installed, unused | e2e: wrong checksum refused, right one staged and version-checked, macro switched, live dir restage refused, a second release staged beside the first (kept for rollback); unit test for the `{build}` pattern |
| 2026-11-18 | 6. Tests | `llamacpp_e2e` (stand-in router: load, unload, status, reload, a failed load, refused loads, no autoload — a request for an unloaded model is refused and nothing loads it), `llamacpp_lanes_e2e`, core `llamacpp_tests`, daemon engine test | `cargo test` in `rust/`: core 41 + llamacpp 17 + planner 27 + swap 11, daemon 90, client 4, e2e 42, lanes 4, llamacpp 4, llamacpp lanes 1; real-systemd spike 22/22 (`--ignored --test-threads=1`); macOS cfg check clean; clippy: no new warnings against the branch base. `coord::tests::the_lock_excludes_a_second_holder_and_names_the_first` failed once in a parallel run and passed in 4 reruns (flock release timing; not touched here) |
| 2026-11-18 | 7. `gpt-oss-120b` into the router | Runbook below. Waits for the router pilot and the zero-foreign-write day after M5b is enabled (no earlier than 2026-11-20) | not run |

Recipe for the Macs (save as a file, pass with `--recipe` once; later
installs reuse it; take each release's sha256 from the release page or
`gh api repos/ggml-org/llama.cpp/releases/tags/<tag> --jq '.assets[]|select(.name|test("macos-arm64"))|.digest'`):

```json
{
  "name": "llama.cpp",
  "root": "/Users/<user>/local-llm/engines/llama.cpp",
  "url": "https://github.com/ggml-org/llama.cpp/releases/download/{version}/llama-{version}-bin-macos-arm64.tar.gz",
  "strip_components": 1,
  "binary": "llama-server",
  "version_arg": "--version",
  "version_pattern": "version: {build}"
}
```

Checked 2026-11-18: release `b12527`, asset
`llama-b12527-bin-macos-arm64.tar.gz` (17.3 MB, `sha256:8b2defad631db0ccc576d274bd3ff3bb0bf394deb255b76413b5226b0b865921`),
unpacks to `llama-b12527/llama-server` beside its dylibs. The newest
release flagged "latest" is `v1.0.0` (2026-11-13) and has no macOS asset;
the `b<build>` tags do. Confidence: high (listed with `gh api`, archive
listed with `tar -tzf`).

### Pilot (not run; needs the owner's go-ahead)

1. Merge the branch, redeploy the three agents from main (`host upgrade`).
2. Workstation: a `llamacpp-router` user unit from
   `RouterLaunch{binary: ~/AI/llama.cpp/build/bin/llama-server, port: 8012,
   preset: ~/.config/llmctl-next/llamacpp-router.ini, models_max: 2}`,
   `CUDA_VISIBLE_DEVICES=0,1`; adopt it with `--engine-kind llama_cpp_router
   --gpu-claim on_demand`, model `llamacpp-router`; `llamacpp import`.
   Port 8012: 8010 is llm-router's `extra_vllm_upstreams`, 8011 was a test unit.
3. Inside `longjob start --router`: `llamacpp add` the two Qwen models
   (`/mnt/storage/converted/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf`,
   `/mnt/storage/converted/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf`,
   each with its `mmproj-*.gguf`), declare a first need (`llamacpp need`),
   `load --evict --wait` each, one request each, `unload`; then `run
   vllm-engine` and see it plan the router models out.
4. m5: `llamacpp install llama-swap b<build> --recipe …`, `lanes macro`,
   `llamacpp lane` on a Qwen GGUF under `~/Models/huggingface/Qwen`, one
   request. mbp2: the same with an embedding GGUF from
   `~/Models/gguf/embeddings` and `--arg embedding`, one embedding request.

### Step 7 runbook — `gpt-oss-120b` moves into the router (prepared 2026-11-18, not run)

**When:** after the router pilot above passed, and on or after the first
zero-foreign-write day with M5b enabled (no earlier than 2026-11-20), so
the service swap does not count against the audit. Owner go-ahead first.
Cost: about half a day plus one GPU window.

Facts it rests on (read 2026-11-18, read-only):
`~/.config/systemd/user/gpt-oss-120b.service` (an llmctl-next managed
unit), its wrapper `~/bin/llama-server-guarded`, llm-router's `gptoss`
backend in `~/AI/llm-router/config/router.yaml`, and the agent's records:
recipe `source_build` (`~/AI/llama.cpp`, `build/bin/llama-server`), last
measured holdings 28.1 GiB on gpu 0 and 27.5 GiB on gpu 1 (`llmctl-next
vram`, 2026-11-18 05:23).

1. **Window.** `longjob start --router -- …` for the whole step; stop
   `gpt-oss-120b` if it runs (`llmctl-next stop gpt-oss-120b --wait`).
2. **Preset section**, its launch arguments unchanged (the router sets
   host, port and alias = the section name):

   ```bash
   llmctl-next llamacpp add llamacpp-router gpt-oss-120b \
     --model /home/yannik/models/gpt-oss-120b/gpt-oss-120b-MXFP4.gguf \
     --arg ctx-size=131072 --arg parallel=1 --arg n-gpu-layers=999 \
     --arg n-cpu-moe=6 --arg tensor-split=56,44 --arg ubatch-size=512 \
     --arg batch-size=2048 --arg threads=16 --arg flash-attn=on \
     --arg jinja=true --arg slot-save-path=/home/yannik/.cache/gpt-oss/slots \
     --arg cache-reuse=256
   ```

   Review the plan, re-run with `--confirm <digest> --wait`. The rendered
   section is the fixture `rust/crates/llmctl-core/tests/fixtures/llamacpp/gpt-oss-120b.ini`
   (it round-trips and validates in the core tests). `--no-webui` is a
   router-level flag: put it in the router unit's argv, not the section.
   The slot dir must exist and be writable by the router unit
   (`mkdir -p ~/.cache/gpt-oss/slots`; the old unit's `ExecStartPre`).
3. **Facts and the guard** (the guard's checks become the pre-load check;
   the router starts model processes with its own binary, so the wrapper
   no longer runs):

   ```bash
   llmctl-next llamacpp need llamacpp-router gpt-oss-120b --mib 0=28774,1=28160 --min-free 29500
   ```

   `--mib` is the measured holding of today's unit; the first load
   replaces it with the router's own measurement. Shards and `.incomplete`
   downloads are checked on every load (the model is one file). Measure the
   host-RAM share of `--n-cpu-moe 6` (the RSS of the model process during
   the first load) and declare it with `--host-ram <GiB>`.
4. **Companion hooks** for M5 only (they retire at M6 with
   `ds4-llmctl-sync`). Write `~/.local/state/llmctl-next-m4a/llamacpp/llamacpp-router.hooks.json`
   by hand (no API registers a command):

   ```json
   {"hooks": [
     {"model": "gpt-oss-120b", "name": "ds4-adopt", "when": "post_load", "optional": true,
      "timeout_secs": 1260,
      "argv": ["/usr/bin/env", "DSV4_PORT=8012", "DSV4_ALIAS=gpt-oss-120b", "DSV4_LLMCTL_WAIT_S=1200",
               "DSV4_LLMCTL_UNIT=llamacpp-router.service", "/home/yannik/bin/ds4-llmctl-sync", "adopt"]},
     {"model": "gpt-oss-120b", "name": "ds4-detach", "when": "post_unload", "optional": true,
      "argv": ["/usr/bin/env", "DSV4_PORT=8012", "DSV4_ALIAS=gpt-oss-120b",
               "/home/yannik/bin/ds4-llmctl-sync", "detach"]}
   ]}
   ```

   `optional: true` matches the unit's `-` prefix. A failed load runs the
   detach as part of its rollback. `llamacpp models llamacpp-router` lists
   the hooks it read; a file that does not validate refuses loads.
5. **Version pin.** Give the router workload the same `source_build`
   recipe (`recipe set llamacpp-router …`, it pins in the same step), then
   `llmctl-next version llamacpp-router --repin`. The router must run a
   build with router mode (the workstation checkout `257813839` has it).
6. **llm-router** — the change to make in `~/AI/llm-router/config/router.yaml`
   (not made here; back the file up first):

   ```yaml
   - name: gptoss
     served_model: gpt-oss-120b
     upstream: http://127.0.0.1:8012          # was :8005 (the old unit)
     unit: llamacpp-router                     # was gpt-oss-120b
     llmctl_next: llamacpp-router
     llmctl_next_model: gpt-oss-120b           # NEW key, needs the code change below
     start_cmd: ["/home/yannik/.local/state/llmctl-next-m4a/llmctl-next.bin",
                 "--data-dir", "/home/yannik/.local/state/llmctl-next-m4a",
                 "llamacpp", "load", "llamacpp-router", "gpt-oss-120b", "--wait", "--evict"]
     # gpus, runtime, timeouts, aliases unchanged
   ```

   llm-router's handoff today runs `llmctl-next run <workload> --wait
   --force` (`llm_router/switcher.py`, `_start_by_llmctl_next`). With
   `llmctl_next_model` set it must instead make sure the router runs
   (`run llamacpp-router --wait` when stopped) and then run `llamacpp load
   <workload> <model> --wait --evict`; rc 3/6 keep today's rule (fall back
   to `start_cmd` only when llmctl-next refused before changing anything).
   `--evict` stops `vllm-engine` (both cards) the way llm-router's own
   switch would; switching back (`qwen27b`) is a `vllm-engine` run, which
   now plans the router's loaded models out first. Health: llm-router polls
   the backend's `/v1/models`; the router forwards a loaded model's entry
   there. Apply with `~/AI/llm-router` restart; regenerate the gateway
   lanes if `gen-local-lanes.py` reads `upstream`.
7. **Switch test** (the done line for step 7): through llm-router,
   `qwen27b` → `gptoss` → `qwen27b`, one request answered on `gptoss`; the
   operation shows the `ds4-adopt`/`ds4-detach` phases; `llamacpp load` of
   a stand-in model whose second shard is missing (a temp dir, added and
   removed again) is refused by the pre-load check; `audit` shows no
   unexplained write.
8. **Way back.** Keep `gpt-oss-120b.service` disabled, not deleted, for one
   week: restore the `router.yaml` backup, `llamacpp remove llamacpp-router
   gpt-oss-120b`, `run gpt-oss-120b`. After the week, `forget gpt-oss-120b`
   and delete the unit.
