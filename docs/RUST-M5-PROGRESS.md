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
