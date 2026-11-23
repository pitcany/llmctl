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
| 2026-11-19 | **M5b enabled** (owner go-ahead, 11:11 PST, after the baseline day ended 10:32 PST). Baseline count accepted: workstation 0, mbp2 0, m5 4, all llama-swap's own lane swaps (request loads, one TTL unload, one from the M5d pilot), none by a legacy writer. Workstation: the `llmctl_next:` section above appended to `~/.config/llmctl/settings.yaml` (not committed to llmctl-config: the file holds a token; backup in the session scratchpad). m5: `next_enabled: 1` in `~/.config/llmctl/config.yaml` (backup `config.yaml.pre-m5b` next to it). mbp2 stays off (its llama-swap is not agent-owned). `llamacpp-router.service` is not in `units:` (no Python writer touches it) | Python `load_delegation()` reads the section; with `LLMCTL_NEXT_CMD=echo`: restart `vllm-engine.service` → `restart vllm-engine --wait`, stop `gpt-oss-120b` → `stop gpt-oss-120b --wait`, start `ollama-gpu1.service` → `run ollama-gpu1 --wait`; `llamacpp-router` and read verbs keep the systemctl path; the guard's real `cmd` answers `status`; `llmctl status` rc 0; `:8088` answers; m5 `llmctl config` shows `next_enabled 1 file` and the agent lists 13 owned lanes. No real write was sent through either guard |
| 2026-11-20 | **M5b day + drills.** Day (to 11:11 PST): legacy writers 0, but 11 foreign `vllm-engine` stops/starts from three mathlab GPU studies (`~/AI/mathlab/runs/wsp-*-20261119/run_study.sh`, raw `systemctl --user`), and the agent logged no delegated operation: the guards were not exercised. Owner chose drills instead of a second day. Before them: `llm-unit` start/stop/restart of owned units now go through the same Python guard (AI `a31d514c`, `-I` fix `4554005c`), so longjob and every llm-unit caller is covered. **Drill 1a passed:** `llmctl stop` → agent `Stop vllm-engine`, 0 foreign. **Drill 1b failed:** `llmctl run qwen27b` (via llm-router's switcher, which hands vLLM starts to the agent since M4c) → 409 `definition_drift: environment_files`: Python rewrites `vllm-engine.env` with its own render before the start, the agent sees a new digest and refuses, Python restores the old file. A second blocker: `run qwen27b --preset --dry-run` → 409 `gpu_occupied` (pid 3404093, `speaches` uvicorn, 538 MiB on GPU 0, no unit). So no delegated start of `vllm-engine` can succeed today. Engine restored with `LLMCTL_NEXT_DISABLED=1 llm-unit start vllm-engine` (down 19:39–19:44 UTC; that start is the one foreign write). **Drill 4 passed:** m5 Mac `llmctl start/stop qwen4b-m5` → two agent `LaneSwap` ops, 0 foreign. Drills 2, 3, 5 held: each would need an agent start of `vllm-engine` and hit `gpu_occupied` | `ops` and `audit --all --since 2026-11-20T19:38:52Z`; inventory digest of `vllm-engine.env` equals the live file after the restore; llm-unit tests 27 passed (5 new ones fail on the old script) |
| 2026-11-20 | **Workstation M5b turned off** (owner decision, after the drills): `llmctl_next.enabled: false` in `~/.config/llmctl/settings.yaml`. Python llmctl and `llm-unit` take the systemctl path again; the `llm-unit` delegation code stays (same switch). m5 `next_enabled: 1` stays on (its drill passed). Not affected: llm-router's M4c vLLM handoff, which still asks the agent and so still meets `gpu_occupied` while `speaches` holds GPU 0. Turn back on after: (1) Python preset runs go through `llmctl-next run <preset> --preset`; (2) a decision on `speaches` (agent tolerance or move it off the card); then rerun drills 1b, 2, 3, 5 | `load_delegation()` → enabled False, `vllm-engine` not delegated; the `llm-unit` snippet returns 100 (systemctl path); `llmctl status` rc 0; router lists `qwen27b` |
| 2026-11-20 | **`speaches` moved to CPU** (owner decision for blocker 2): `CUDA_VISIBLE_DEVICES=-1` and `WHISPER__INFERENCE_DEVICE=cpu` in `~/.harbor/services/speaches/override.env`, container recreated with `harbor up --skip-port-check speaches` (only it was recreated). JARVIS uses it for Kokoro TTS only (STT is Hermes' local whisper): TTS answers 200, about 5 s for one sentence of about 4.6 s audio. **Blocker 2 is not cleared:** `run vllm-engine --dry-run` now refuses on `rustdesk` (system unit, 614 MiB, started 02:30 PST) and `sunshine` (500 MiB, 08:15 PST) on GPU 0. Both use NVENC; moving them off the GPU means software encoding. `ops.rs` `check_gpu_occupancy` allows only units the plan touches or the VRAM fit keeps | `nvidia-smi` compute apps: rustdesk, sunshine, two vLLM workers (no speaches); container `CUDA_VISIBLE_DEVICES=-1`; TTS 200 twice |
| 2026-11-20 | **GPU tolerances** (`11b52bd`; owner: keep `speaches` on the GPU and tolerate it): `speaches` back on GPU 0 (override.env restored; warm Kokoro TTS 3.7 s vs 5 s on CPU). Inventory `gpu_tolerated` entries match a full unit identity or a command-line substring (`GpuOccupant.cmdline` from `/proc/<pid>/cmdline`), each with a VRAM cap; above the cap the process is foreign again and the refusal names the cap. Verbs `tolerate`, `tolerate add NAME --unit|--cmdline --max-mib N`, `tolerate remove NAME`; API `/v1/gpu/tolerated[/add|/remove]`. Workstation agent redeployed (sha256 `38e316d1…`, old binary `llmctl-next.bin.pre-11b52bd`); entries: rustdesk `system/rustdesk.service` 1024, sunshine `user/sunshine.service` 1024, speaches cmdline `speaches.main:create_app` 2048 (inventory revision 47). **Blocker 2 cleared:** `run vllm-engine --dry-run` and `run qwen27b --preset --dry-run` now plan instead of refusing `gpu_occupied`. No real start was run. Blocker 1 (Python preset run rewrites the env) still open; workstation M5b stays off | e2e `tolerated_gpu_occupants_let_a_run_through_until_one_outgrows_its_cap` (refused before tolerance, bad requests, duplicate, run passes, over-cap refusal names pid and cap, removal); core `gpu_tolerance_matches_by_unit_or_cmdline_and_validates_its_shape`; `cmdline_text` unit test; workspace suite 302 passed, 1 timing flake in `coord` lock test (passes 3/3 alone) |
| 2026-11-20 | **Blocker 1 re-diagnosed and fixed on the router path.** The 19:39 failure was not Python rewriting the env: llm-router handed the switch to the agent, the agent refused (`gpu_occupied`, exit 4), the router fell back to its own path (stop_first stopped strata, gpt-oss-120b, ollama-gpu1; copied the `qwen38` profile into `vllm-engine.env`; restarted through the guarded `restart_cmd`), and the agent refused that env (`definition_drift`). llm-router `c6d6941`: new `switch.llmctl_next_refusal_final` (default off): an agent refusal (4, 6) fails the switch with the agent's reason and changes nothing; 3 (agent away) and 5 (not enrolled) still fall back. Turn it on with M5b. **Live:** `llmctl-next run vllm-engine --profile-env …env.qwen38 --model qwen3.8-27b --wait` (the router's exact handoff) succeeded 20:27:52–20:29:46 UTC: profile applied, digest re-pinned to `9c93…`, verified; 0 foreign writes; live file = pinned digest. Still open: `llmctl run --direct` (and Python's fallbacks when the router is down or does not know the model) writes Python's own env render before the restart; with M5b on the agent refuses it before anything changes (safe, but the run fails) | router tests 23 in `test_switcher_llmctl_next.py` (6 new; 4 failed before the change), full router suite 369 passed + 1 pre-existing failure also on base (`test_every_backend_profile_is_tracked_in_the_repo`); agent op `Run vllm-engine` Succeeded, phases=6 |
| 2026-11-20 | **`llmctl run --direct` delegated** (`0fa0304`): when the engine unit is delegated, the orchestrator runs `llmctl-next run <preset> --preset --workload <w> --wait` and writes no env (the agent would refuse llmctl's own render as drift); one-off overrides (`--tp/--max-len/--gpu-mem/--gpus/--tq`) are refused with the way out. Deployed into `~/AI/.venv` (`llmctl 0.3.0+0fa0304c`, from the local repo; llmctl-api restarted; the llmctl-stack pin waits for a push). **Live, M5b on for this one call:** `llmctl run qwen27b --direct --yes` → agent ops `Stop vllm-engine` 20:46:19, `Run vllm-engine` (preset) 20:46:33, `Run ollama` 20:48:08 UTC, all Succeeded; 0 foreign writes; env file = pinned digest `9c93…`; M5b off again after | orchestrator tests 4 new (3 failed before); Python suite in 4 shards 1,824 passed, the one failure (ruff E501 on these lines) fixed, ruff clean |
| 2026-11-20 | **M5b back on** (owner go-ahead, 12:52 PST): `llmctl_next.enabled: true` in `~/.config/llmctl/settings.yaml`; llm-router `847da33` sets `switch.llmctl_next_refusal_final: true`, router restarted (loads `config/router.yaml`). m5 `next_enabled: 1` unchanged. Watch `audit --all` over normal use | guard maps `vllm-engine.service` restart and `ollama` stop to the agent, `llamacpp-router` stays on systemctl (`LLMCTL_NEXT_CMD=echo`); `llm-unit restart vllm-tp` → agent; router config parses with the flag true; `audit --all` since 20:46 UTC: 0 on all three hosts |
| 2026-11-21 | **M5b audit, first night** (09:17 UTC, ~12.5 h after the re-enable): 0 foreign writes on workstation, mbp2 and m5 since 20:52 UTC. No agent operations either, so the guards were not exercised overnight. The 19 findings in the full `audit --all` all predate the re-enable (workstation: 11 mathlab raw systemctl + the drill-1b restore at 19:42; m5: llama-swap request/TTL swaps, expected). Clean-day count starts at 20:52 UTC 2026-11-20; the first full day closes 20:52 UTC 2026-11-21 | `audit --all --since 2026-11-20T20:52:00Z` → 0; `ops` newest entries 20:48 (workstation), 19:43 (m5), 11-19 (mbp2); vllm-engine user-scope active, `/v1/models` serves qwen3.8-27b |
| 2026-11-21 | **M5b clean day 1 of 7 closed** (20:52 UTC; read 21:09 UTC): workstation 0, mbp2 0, m5 3 foreign writes since the re-enable. The 3 on m5 are llama-swap request and TTL swaps (11:46, 11:55, 12:31 UTC; the same daily rhythm as 11-19 and 11-20), expected legacy-free. Legacy writers: 0. Caveat: no agent operation ran on any host in the 24 h, so the guards saw no real traffic; the day proves nothing bypassed them, not that they work under load | `audit --all --since 2026-11-20T20:52:00Z` → 3 (all `actor=llama-swap` or lane-only on m5); `ops` newest entries unchanged since 11-20 20:48 (workstation), 19:43 (m5), 11-19 (mbp2) |
| 2026-11-21 | **Drills 2, 3, 5 passed live with M5b on** (owner go-ahead; engine away twice, 21:12–21:14 and 21:14–21:16 UTC). **Drill 3 (take-gpus):** `llmctl take-gpus qwen3coder30b --take-gpus --drain-mib 2000` → agent `Stop vllm-engine` 21:12:16, GPU 1 drained to 20 MiB. **Drill 2 (router switch):** a gateway request for `local` (`ccl --ask local`) → router `model.switch.begin to=qwen27b from_=none` 21:12:41, `model.switch.llmctl_next workload=vllm-engine`, agent `Run vllm-engine` 21:12:44, `switch.done` 21:14:18, answered in 98 s; llm-heal ran at 21:13:29 mid-switch and left the engine alone. **Drill 5 (longjob window):** `longjob start m5b-drill5 --stop-engine` → lease taken, agent `Stop vllm-engine` 21:14:41, body 20 s, agent `Run vllm-engine` 21:15:16, serving 21:16:53, lease free. All five drills have now passed (1a, 1b via the router fix, 2, 3, 4, 5). Note: a direct POST to :4100 needs the router bearer token (401); drives go through the gateway. Afterwards the ~/AI session approved its own queued `io-qwen27b-profile` (`longjob go` 21:15:57 UTC), an engine-stopping job on the same longjob path | `ops`: four new agent operations, all Succeeded; `audit --all --since 2026-11-21T21:12:00Z` → 0 on all three hosts; router `/health` upstream_ok, `qwen3.8-27b`; router journal lines quoted above; job log `~/.local/state/long-jobs/m5b-drill5/log` |
| 2026-11-22 | **M5b clean day 2 of 7 closed** (20:52 UTC; read 2026-11-23 00:52 UTC): workstation 0, mbp2 0, m5 17 foreign writes in the day window. Legacy writers: 0. 13 on m5 are llama-swap request and TTL swaps (04:44–19:46 UTC), expected legacy-free. 4 are pid changes of `ollama` and `llama-swap` with no actor (a known limit: a manager restart records as a pid change): 04:07 UTC follows a clean m5 reboot at 20:06 PST 11-21 (shutdown stall report `shutdown_stall_2026-11-21-200616`), 15:35 UTC follows the console login at 07:35 PST after an m5 kernel panic at ~01:58 PST (`watchdog timeout: no checkins from watchdogd in 90 seconds`, macOS 27.0.1 26A434; not caused by the agent, cause not investigated). Unlike day 1, the agents ran real operations in the window: workstation 35 (drills, Stop/Run cycles of `vllm-engine`, one `ollama-gpu1` stop), m5 28 lane operations, all Succeeded, with 0 foreign writes on the workstation (corrected 2026-11-22: the first version of this row listed only the last 6 workstation operations). Day 3 so far: m5 9 llama-swap swaps; the 00:04–00:05 UTC Stop/Run of vllm-engine is an agent operation (a `ccl --check` test through `llm-backend-guard`, logged in the vault) | `audit --all --since 2026-11-21T20:52:00Z` → workstation 0, mbp2 0, m5 26 (17 before 20:52 UTC); `ops` newest entries; m5 `last`, `/Library/Logs/DiagnosticReports/panic-full-2026-11-22-073511.0002.panic` |
| 2026-11-18 | **Router item 4 — not done, owner-accepted gap**: `qwen3coder30b` maps to owned `ollama-gpu1`, but `run ollama-gpu1` does not preload `qwen3-coder:30b`; the router's Ollama readiness reads `/api/ps`, so a handover breaks the lane. Stayed on the script; its take-gpus stops now delegate (item 2), its final `systemctl start ollama-gpu1.service` remains a foreign write when that lane is used | reasoning in B4 of the workflow matrix; revisit with M5c catalog work |

### Left to do in M5b

0. ~~Turn M5b back on~~ **done 2026-11-20** (row above): `llmctl_next.enabled: true` in `~/.config/llmctl/settings.yaml` and `llmctl_next_refusal_final: true` under `switch:` in llm-router's `router.yaml` (restart llm-router).

1. ~~Enable the workstation `llmctl_next:` section and m5 `next_enabled: 1`~~
   **done 2026-11-19 11:11 PST** (row above).
2. One day of normal use with `audit` on: the foreign-write count for
   owned workloads should fall to zero.
   Window: 2026-11-19 11:11 PST → 2026-11-20 11:11 PST. Count legacy
   writers only: llama-swap's own request loads and TTL unloads on m5 stay
   foreign by design, and the router's `qwen3coder30b` start of
   `ollama-gpu1` stays one (item 4 gap).

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
| 2026-11-22 | **Vendor `load` verb** (item 4, the `unload` mirror; branch `worktree-m5-leftovers` off main, **not deployed — the no-redeploy pilot window runs to 2026-11-27, and the qwen3coder30b router handover waits for `fix/run-rollback-order`**): `llmctl-next load <workload> <model> [--keep-alive N] [--force] [--wait]`. Core: `OpKind::Load`, `PlanStep::LoadModel`, `plan_load` (authority, Ollama-only `has_engine_load`, one model not the service identity, `keep_alive 0` refused, must be observed Running; rollback = one unload). Daemon `vendor_load.rs`: the catalog check (an enabled record of this engine naming the model) and the VRAM fit (measured `vram.json` record first, else the catalog's `estimated_vram_gb` split across the service's cards, 1 GiB headroom, free read from `gpu_free_mib`, **fail closed on every unknown**), re-planned under the admission lock with fresh observations; `--force` skips the fit only. Executor: load-only `POST /api/generate` (default `keep_alive -1`, the qwen3coder30b-start script's choice), confirmed through `/api/ps`, then the measured need recorded into `vram.json` (the free-VRAM drop, else the engine's own `size_vram`) — the next load plans with the real figure. Wire `POST /v1/loads`, `EngineReport.supports_load`, exit 4 `load_refused` | core `plan_load` planner test; `engine_api` stand-in tests (load+confirm, size_vram, already-loaded, dead port); e2e with a stand-in Ollama server (stopped service, uncataloged model exit 4, too-small card exit 4, happy path + `vram.json` record, repeat load, `supports_load`) |
| 2026-11-22 | **Load verb review fixes** (independent review of `779accb`; branch `fix/m5-leftovers-load-rollback` off `worktree-m5-leftovers`, not deployed): (1) a load that failed after admission always ended `needs_attention` — `run_recovery` refused the plan's own `UnloadModel` rollback ("an unload is not a restoration step"), so every engine error, unreachable Ollama or confirm timeout blocked all later admissions on the host and could leave the model resident with `keep_alive -1`; the rollback now runs the unload effect (unloading a model that never loaded is the asked-for end state). (2) A model matched by its catalog `name` was sent to Ollama under that name; the planner now prefers a `source` match and always plans with the record's `source`. The 7 minor findings of the same review, fixed in `018d77d` (daemon), `c8f12be` (TUI), `395c760` (CLI): the recorded need uses the free-VRAM drop only within 25 % of Ollama's `size_vram`, else `size_vram` (`vendor_load::measured_need`); one process-wide lock around every `vram.json` read-modify-write, and `write_json_atomic` uses a tmp name per write; a model the engine already holds skips the fit (the planner asks the engine for its resident list); each confirm `/api/ps` read is bounded by what is left of the 30 s window, and a load Ollama never answers says it timed out; the TUI scans every answering host when the catalog has no row; a sync answer reopens the preview only when no other modal is open, and a declined or refused sync clears the pending preset; a panicking `status --all` task becomes that host's error | e2e `a_failed_load_rolls_back_cleanly_and_the_engine_gets_the_source_name` (stand-in Ollama answers 500 for one model: op `failed`, recovery `restored`, next load admitted; `load svc coder` sends `qwen3-coder:30b`, `vram.json` key uses the source). Fails without each fix (`NeedsAttention` vs `Failed`; `["coder"]` sent). Minor fixes: the same e2e test also covers the no-op reload on a full card (allowed when resident, refused when not); unit tests `need_tests` (drop vs `size_vram`) and `a_silent_load_says_timed_out_and_the_confirm_window_holds`; TUI flows `a_late_sync_answer_never_replaces_an_open_modal`, `an_empty_catalog_can_still_be_scanned` (both fail on the old `app.rs`). `status --all` still has no test (it needs an ssh bridge). Workspace suite after the minor fixes: 344 passed, 1 failed (the known `bridge_subcommand_relays_http_to_the_socket` flake; 5/5 alone and the e2e binary 56/56 on rerun), 32 ignored. Full workspace suite `--no-fail-fast`: 341 passed, 0 failed, 32 ignored |
| 2026-11-18 | **Router item 4 revisited (from M5b)** — still not handed over, now with a smaller gap. The catalog names `qwen3-coder:30b` on the `ollama` store (it is one of the 53 rows the real-registry import maps), so "which model, on which store" is answered. What is still missing is the load itself: `run ollama-gpu1` starts the service but loads no model, and the router's Ollama readiness waits on `/api/ps`. Needed: a `load <workload> <model>` verb for vendor services (the mirror of `unload`: Ollama `POST /api/generate` with `keep_alive`, confirmed through `/api/ps`, the model checked against the catalog, the GPU claim planned as for M4c's on-demand claim), then the router backend can switch to `llmctl_next_workload: ollama-gpu1` plus that load. Until then the lane stays on `qwen3coder30b-start`; its final `systemctl start ollama-gpu1.service` remains the one known foreign write | read-only: router.yaml backend `qwen3coder30b` (`start_cmd` script, `unit: ollama-gpu1.service`), catalog rehearsal above |
| 2026-11-18 | **M5c check (this branch, not deployed)**: cargo workspace suite green (`--no-fail-fast`: core 52 + planner 27 + swap 11 + parity 3, daemon 93, client 4, e2e 42, lanes 5, presets/catalog/records 1 each); real-systemd spike suite `--ignored --test-threads=1` 22/22; full Python suite in 4 shards 1,820 passed, 1 skipped. Known flake, not from this branch: `e2e::bridge_subcommand_relays_http_to_the_socket` fails about 1 run in 10 on the base commit `bd90afa` too (the daemon closes a raw request unanswered) | logs in the session; base-commit check in a throwaway worktree |
| 2026-11-18 | **Merged and deployed** (M5c, M5d, M5e merged into main `75dfd5c`, `cd6db43`, `2dbddfd`, pushed): workstation agent binary swapped (release build of `2dbddfd`, sha256 `2ea13335…`; old binary kept as `llmctl-next.bin.pre-m5cde` in the m4a data dir; `llmctl-next.service` restarted 20:56 PST); mbp2 `host upgrade --source` (built on mbp2, sha256 `bacba625…`, agent restarted 20:58); m5 `host upgrade --binary` with mbp2's build (same sha256, restarted 21:15). Not yet run: `preset settle --write`, `catalog import`, any M5d load | `status` on the workstation unchanged line for line; `status --all` all three hosts seen 0 s ago; `preset list` answers (5 presets `ready`); `model list` answers on both Macs (0 models, nothing imported); `audit --all`: workstation 0, mbp2 0, m5 1 (the 2026-11-19 00:07 UTC lane load of `qwen35b-m5`, before the deploy) |
| 2026-11-18 | **`preset settle --write` run on the workstation** (21:33 PST): three comment lines — `vllm-engine.env.qwen38` and `.previous` line 9 now name the real preset `qwen27b.yaml` (not the missing `qwen3.8-27b.yaml`); `router.yaml` line 103 says the `qwen27b` backend runs from vllm-muse, not vllm-laguna. Committed `~/AI` `9d69f81f`, llm-router `0340362`; previous bytes in `*.llmctl-previous` (untracked). **Found:** `preset list` shows `qwen27b` as `ready`, not `serving`, while vllm-engine serves it — the active env has the same keys/values as the profile and the inventory's model and env path match, so the `active` test in `llmctl-daemon/src/presets.rs:368` is suspect (not investigated further) | re-run dry run: "every comment already says so (7 files read)"; each file differs from its backup by one line; vllm-engine Running/Ready, llm-router `/health` ok on qwen27b, `audit` 0 |
| 2026-11-18 | **Fixed: served preset shown as `ready`** (`129da67`): `preset list` compared the active env with the render as ordered lists (`parse_env` keeps file order); the live `vllm-engine.env` has `VLLM_TRUST_REMOTE` and `VLLM_EXTRA` elsewhere. Now `env_matches` = `diff_env(...).is_empty()`, the same test drift uses. Workstation agent redeployed with it (sha256 `274d4c82…`, 21:52 PST); the Macs keep `bacba625…` (they hold no presets) | new unit test failed on the old comparison, passes now; daemon 97 + core suites + `presets_e2e` green; live `preset list`: `qwen27b serving`, other four `ready`; `audit` 0 |
| 2026-11-19 | **`catalog import` dry run on all three agents** (bundle from `llmctl export-registry`: 74 models, 2 profiles; nothing imported): workstation `--as yannik-desktop` new 53, no_store 1, unmapped 20, 2 new profiles; m5 `--as Yanniks-MacBook-Pro.local` (exact hostname match) new 13, no_store 1, unmapped 60; mbp2 `--as macbookpro.lan` (by elimination: its hostname is `Freddie`) new 7, no_store 1, unmapped 66. The no_store row is the same on every host: `gpt-oss-120b-mxfp4`, whose GGUF is in no declared store | dry-run reports read per host; `model list` still 0 on the Macs after the runs (dry run wrote nothing) |
| 2026-11-19 | **`catalog import` applied on all three agents** (owner go-ahead; fresh `llmctl export-registry` bundle, same 74 models and 2 profiles as the dry run; each report re-run first and identical to the dry run): workstation `--as yannik-desktop` 54 models (53 new + the no_store row), m5 `--as Yanniks-MacBook-Pro.local` 14, mbp2 `--as macbookpro.lan` 8; unmapped rows kept for the export (20 / 60 / 66); 2 profiles each; catalog revision 1 on every host. The Python DB is not written. `gpt-oss-120b-mxfp4` is cataloged with no store (no store declared for `~/models/gpt-oss-120b`) | `model list` before: 0 on every host; after: 54 / 14 / 8; a second dry run says `present 54/14/8`, 0 new (idempotent); workstation `catalog export` gives back the bundle's exact 74 rows and 2 profiles; `audit --all`: workstation 0, mbp2 0, m5 4 (all llama-swap lane changes from before the import, the pilot's among them) |

### Left to do in M5c

1. Deploy (owner): ~~merge, install the agent build on the workstation~~, ~~then `preset settle --write`~~ (both done 2026-11-18, rows above) to put
   the pin's words (vllm-muse, vLLM 0.30.0) into the env header, the
   profiles and router.yaml's comment; `preset list` should show every
   preset `ready` or `serving`.
2. Import the real registry on each agent: workstation
   `catalog import <bundle> --as yannik-desktop`, the Macs with
   `--host m5|mbp2 … --as Yanniks-MacBook-Pro.local|macbookpro.lan`
   (dry run done 2026-11-19 on all three, row above; m5 = the
   `Yanniks-MacBook-Pro.local` entry, mbp2 = `macbookpro.lan`). Apply with
   `--confirm` (owner). **Done 2026-11-19** (row above). Optionally declare
   a store for `~/models/gpt-oss-120b` (the one no-store row).
3. Owner decision 7: install `hf` on m5 (not done here: no host changes
   in this session).
4. Router item 4: ~~the vendor `load` verb~~ **built 2026-11-22** (row
   above; merge + deploy after the pilot window). What remains is the
   router handover itself: llm-router `qwen3coder30b` →
   `llmctl_next: ollama-gpu1` + `load ollama-gpu1 qwen3-coder:30b --wait`
   as the start path — with `fix/run-rollback-order`, not before.
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
  "version_pattern": "build {build},"
}
```

Checked 2026-11-18: release `b12527`, asset
`llama-b12527-bin-macos-arm64.tar.gz` (17.3 MB, `sha256:8b2defad631db0ccc576d274bd3ff3bb0bf394deb255b76413b5226b0b865921`),
unpacks to `llama-b12527/llama-server` beside its dylibs. The newest
release flagged "latest" is `v1.0.0` (2026-11-13) and has no macOS asset;
the `b<build>` tags do. Confidence: high (listed with `gh api`, archive
listed with `tar -tzf`).

### Pilot (run 2026-11-18/19, results below)

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

#### Pilot results (2026-11-19)

| Step | Result | Evidence |
|---|---|---|
| 2. Router unit | `llamacpp-router.service` (user unit, started, not enabled) on :8012, `--models-max 2 --no-models-autoload`; adopted as a managed workload (inventory revision 44); `llamacpp import` round trip equivalent; both Qwen models added by admitted operations (preset revision 3) | adopt checks all passed incl. served_identity; `/models` lists both `unloaded`; a request for an unloaded model returned HTTP 400 "model is not loaded" and nothing loaded it |
| 3a. Qwen3.5-0.8B beside vllm-engine (`longjob m5d-pilot-a`, 22:05 PST) | fit check said yes (1.5 GiB per card needed, 4.2/5.1 GiB free); loaded in 21 s; measured 1.5 GiB gpu 0, 1.4 GiB gpu 1; answer `pong` (200 completion tokens: thinking on); unloaded | `nvidia-smi` before/after: llama-server 1506/1406 MiB while loaded, gone after; vllm-engine untouched |
| 3b. Qwen3.6-35B-A3B with `--evict` (`longjob m5d-pilot-b`, 03:01 PST) | dry run without `--evict` refused (21 GiB declared, 4–5 GiB free); `load --evict` stopped vllm-engine and loaded in 182 s (37.8 GB GGUF read cold); measured 20.2 GiB gpu 0, 18.0 GiB gpu 1; answer `pong` (thinking off), 119 tok/s; `run vllm-engine --dry-run` with the 35B loaded planned `unload qwen3.6-35b-a3b from llamacpp-router` before `start vllm-engine`; unload, then `run vllm-engine` ready in 95 s | vllm-engine down 03:01:01–03:05:53; afterwards vllm `:8003/health` 200, llm-router `:4100/health` 200, VLLM workers 26.0 GB per card |
| 4. m5 | **Bug found and fixed:** every stage on macOS failed `io: df failed` (GNU-only `df -B1 --output=avail`); fixed in `80312df` (`df -Pk`), Macs redeployed (mbp2 build, sha256 `041e1d59…`). **Recipe fixed:** b12527 prints `version: 1.0.0-dev (build 12527, …)`, so the pattern is now `build {build},` (the stager refused the old pattern and promoted nothing). Then b12527 staged into `~/local-llm/engines/llama.cpp/llama-swap.llamacpp-b12527-9389c37c`, macro `llamacpp` set, lane `qwen3-embed-llamacpp-m5` (Qwen3-Embedding-0.6B Q8, `--embedding`, ttl 300) answered a 1024-dim embedding in 6 s cold. Lane removed afterwards | **Found:** the lane was in no matrix set, so its first request evicted the preloaded `mimo9b-m5` and `qwen35b-m5` (about 50 s out at 02:51 PST; the reload on removal preloaded them again, both `ready` 02:52). fixed the same day: lane plans now warn (row below) |
| 4. mbp2 | same release staged, macro set, lane `nomic-code-embed-llamacpp-mbp2` (nomic-embed-code Q8, 7.5 GB, `--embedding`) answered a 3584-dim embedding in 11.5 s cold; lane removed afterwards (nothing else was running) | `/running` empty before and after; each write confirmed `configuration reloaded` and committed to the config repo |

The `llamacpp` macro and the staged b12527 stay on both Macs for real lanes.

| Date | Change | Evidence |
|---|---|---|
| 2026-11-19 | **Lane plans warn about a lane outside every matrix set** (`1d6b633`): `swap::lanes_outside_sets`; `LanePlanResponse.warnings` (new wire field, empty is omitted); the plan names each lane the change strands (a new lane, or one whose last set or var went), the lanes running now, and the `lanes matrix` way out; `lanes …` and `llamacpp lane` print `WARNING:` lines. A warning, not a refusal. All three agents redeployed with it (workstation sha256 `a2c0725b…`, old binary kept as `llmctl-next.bin.pre-1d6b633`; Macs `0160d9da…` built on mbp2) | core `lanes_outside_every_matrix_set_are_named`; e2e `a_new_lane_outside_every_matrix_set_is_warned_about` (failed before the change: no warning; an edit that strands nothing has none; the write still goes through); full workspace suite green; live preview on m5 (no `--confirm`): WARNING names `flash-m5, mimo9b-m5`; mbp2 (no matrix) shows none; `audit` 0 after the restart |

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

## M5e — minimal multi-host TUI

**Shape (decision 5, recorded here):** both — a new crate
`rust/crates/llmctl-tui` (state, worker, screens, tests) and a thin
`llmctl-next tui [--read-only] [--bench N]` subcommand
(`llmctl-cli/src/tui_cmd.rs`, three lines in `main.rs`) that builds the
host list: the local agent over its socket, every enrolled host over the
SSH bridge with the `status --all` cache. The crate keeps M5c/M5d merge
conflicts in `main.rs` to those three lines.

**How it stays a client and never stalls:** `App` is pure state (keys and
answers in, requests out); a worker runs each request as a tokio task with
every agent call bounded (reads by `LLMCTL_NEXT_STATUS_TIMEOUT_S`, default
10 s; writes by 30 s, expiry = "outcome unknown" → lost-ACK check by key,
as in `cmd_mutate`). The loop never awaits an agent. Hosts are asked
concurrently and each snapshot is drawn as it lands, so a sleeping Mac
costs nothing but its own STALE line (cached observations, never
"stopped"); it is re-dialled at most every 60 s on its own (each ssh
attempt can DarkWake it), `r` dials now. The other screens ask only hosts
whose last status answered.

**Writes:** Run (workloads until M5c's presets merge): enter/S/T → the
agent's plan preview (stop targets, steps, rollback, notes) → `y` admits
exactly that revision and those stop targets after the protocol check →
live progress. A `stale_plan` refusal shows the fresh plan and asks
again; nothing re-admits on its own. Ops: `C` cancel, `A` acknowledge,
each behind y/n and only where the state allows it. `--read-only` refuses
all of them.

**Router models (M5d):** the Hosts screen lists, under each workload, the
models its engine reports loaded (`/v1/workloads/<w>/engine` `loaded`, and
llama-swap's running lanes). A llama.cpp router's models appear there and
on the GPU screen with no TUI change; the snapshot test carries one.

| Date | Screen / item | Verified |
|---|---|---|
| 2026-11-18 | **`d` Hosts**: per host agent version, platform, last seen, revision; per workload process, readiness, intent (`stopped (manual)` from the agent's JSON), served model ✓/~, ownership, freshness; STALE header + cached rows for an unreachable host; engine-loaded models as `└ loaded` rows (`f75986b`, `a08c484`) | snapshot `hosts`; `hosts_screen_shows_stale_never_stopped`; `a_sleeping_host_shows_stale_from_the_cache_within_the_timeout` (silent fake, 300 ms bound, cache read, never "stopped", next auto refresh skips it); live against the pilot agent (6 workloads, `ollama └ loaded qwen3-embedding:0.6b`) |
| 2026-11-18 | **`o` Operations**: running first then newest, phases with marks, result, recovery, `[ack pending]`; `C`/`A` behind y/n | snapshot `ops`, `confirm`; `cancel_asks_first_and_no_sends_nothing`, `acknowledge_asks_first_and_only_for_terminal_unacknowledged_ops`, `read_only_never_offers_a_write`; live: 43 operations listed with details |
| 2026-11-18 | **`l` Logs**: picker of workloads and their lanes; workload logs paged by cursor with follow (`f`), PgUp/PgDn; lane logs from llama-swap's buffer; enter on Hosts jumps here | snapshot `logs`; `logs_follow_pages_by_cursor_and_lane_logs_come_from_the_lane` (cursor paging, no duplicates, pause stops polling); lane log in the GPU flow test; live: vllm-engine following |
| 2026-11-18 | **`g` GPUs/memory**: per card free VRAM, enrolled vs foreign pids, measured records (workstation); per llama-swap workload the budget, loaded-lane peak sum, over-budget sets, running lanes (read by dry-run unload, admits nothing); engines' loaded models | snapshot `gpu`; `gpu_screen_gathers_lanes_and_engine_models_and_hosts_show_them_as_rows` (dry run admitted nothing); live: both cards, two VRAM records, LM Studio and Ollama rows |
| 2026-11-18 | **`a` Audit**: per host "no foreign writes" or findings with before → after and actor | snapshot `audit`; `audit_screen_lists_each_hosts_findings`; live: workstation 0 findings |
| 2026-11-18 | **`w` Run**: workloads with plan preview → confirm → progress (presets pending M5c) | snapshots `run`, `preview`, `preview_refused`; `run_previews_then_admits_exactly_that_plan_and_follows_it` (decline sends nothing; admit carries rev 42 + stops; health before admit); `a_stale_plan_is_refused_and_the_fresh_one_must_be_confirmed_again` (one admit, nothing ran, no auto-retry, second admit carries rev 43 + new stops) |
| 2026-11-18 | **`m` Catalog**: marked pending until M5c merges | snapshot `catalog` |
| 2026-11-18 | **Control characters** (security review of `f75986b`): agent text could put escapes on the terminal through ratatui's paragraph path; a last pass over each frame replaces control characters, log lines cleaned on arrival (`50d5591`) | `agent_text_never_puts_control_characters_on_the_terminal` (failed before the fix) |
| 2026-11-22 | **`w` Run on presets** (leftover 1; branch `worktree-m5-leftovers`, **not deployed — no-redeploy window to 2026-11-27**): the Run screen lists each host's presets (state `serving`/`ready`/`drift`/`no-profile`, the CLI's vocabulary) above the workloads. Enter on a preset resolves it on the agent (read-only `preset_run`: dry-run, no sync), plans the run **with the resolved profile** (`PlanRequest.profile` now flows through the TUI's Plan/Admit and the stale-refused fresh plan), and the preview names the profile it would switch to. A `preset_drift` refusal offers `s`: a y/n confirm writes the sync (`preset sync`, comments kept), then the preview re-plans; `n` declines, `--read-only` refuses it like every write. `S`/`T` on a preset row say a preset is always a run | snapshots `run`, `preview_preset`, `preview_drift` (regenerated and reviewed); flows `a_preset_previews_resolves_its_profile_and_admits_with_it` (admit log carries `profile=…`; resolve writes nothing), `a_drifted_preset_offers_the_sync_which_replans_and_then_admits` (decline sends nothing; sync precedes the single admit) |
| 2026-11-22 | **`m` Catalog** (leftover 1): records per host grouped by store (revision, disabled and bundle-only counts in the header), a details pane (id, engine, origin, path, extra fields), and the writes behind confirmations: `E`/`D` enable/disable (`y`/`n`; capitals — the lowercase `d`/`a` are the Hosts and Audit screen keys, a collision the first test run caught), `s` scans the selected host's stores (a read — `add: false` — allowed read-only), and `R` records a scanned-but-uncataloged weight (every field comes from the scan; no text input needed). Weights `rm` stays a shell verb: its preview-and-token flow is CLI-shaped. After a catalog write the screen re-asks on its own (`catalog_pace` reset) | snapshot `catalog`; flows `the_catalog_screen_lists_scans_and_writes_behind_confirmations` (disable ask/decline/apply, scan, `R` records the found weight) |
| 2026-11-22 | **Small items** (leftover 3): the GPU screen no longer prints `free ?` per card when the host has no VRAM answer (the Macs' unified memory) — the free column is omitted and one dim line points at the lane budget below; `free ?` stays only when the answer exists but misses a card (a real unknown). `status --all` asks the enrolled hosts concurrently (tokio tasks, the TUI's bounded-call shape, per-host timeout unchanged, output order still the enrollment order, cache writes and STALE fallback unchanged) | snapshot `gpu` (regenerated; unchanged); `status --all`: no test calls it (the review found the earlier claim here wrong) |
| 2026-11-18 | **No stall over 100 ms**: every fake call 2 s, one host silent, keys every ~60 ms for 3 s | `slow_and_silent_agents_never_stall_a_frame`: key→frame max 2.9 ms, idle pass max 2.6 ms (debug build) |
| 2026-11-18 | **Live, all three agents, read-only** (release build, `tui --read-only` in tmux): workstation and m5 answered on every screen (m5: 4 workloads, llama-swap budget 104 GiB with `mimo9b-m5` loaded, lane_swap operations, 0 findings); mbp2 did not answer (asleep, last seen 5 h before) and showed STALE with its cached rows on Hosts and "STALE, skipped" on Ops/Audit/GPU, while the other hosts stayed live | mbp2's own screens not yet seen live: re-check when it is awake |
| 2026-11-18 | **Regression suites** with the TUI in the workspace | `cargo test --workspace`: 245 passed, 0 failed; `systemd_spike -- --ignored --test-threads=1`: 22 passed |

### Latency (2026-11-18, workstation, same machine, headless 160x50)

Same two measures on both TUIs: a screen key to the next idle frame, and a
status refresh from request to answer drawn. Rust: `llmctl-next tui
--bench 20` (release build; screens d o l g a). Python: Textual
`run_test` on `MissionControlApp` from `~/AI/.venv` (screens d g l s u;
`r` on the dashboard until `render_data` ran):
`~/AI/.venv/bin/python -I rust/crates/llmctl-tui/bench/bench_py_tui.py 20`
(`-I` so the installed llmctl is measured, not a checkout).

| Measure | Python TUI | llmctl-next tui (local agent only) | llmctl-next tui (all three hosts) |
|---|---|---|---|
| key → frame, n=100 | p50 197 ms · p95 931 ms · max 1416 ms | p50 0.21 ms · p95 0.46 ms · max 0.56 ms | p50 0.24 ms · p95 0.52 ms · max 0.68 ms |
| status refresh, n=20 | dashboard p50 613 ms · p95 1135 ms | hosts p50 174 ms · p95 209 ms | p50 246 ms · p95 10 003 ms ¹ |
| longest loop pass while answers pending | — | 0.91 ms | 3.41 ms |

¹ The bench waits for every host; mbp2 was asleep, so rounds that dialled
it ended at the 10 s status bound. The screen itself drew workstation and
m5 at ~250 ms and never waited (the slow-API test proves the same with
fakes). The Python number includes its own work (systemd, nvidia-smi,
the registry DB) on the UI path; the Rust numbers are the agent's
`/v1/status` (~160–170 ms, mostly the engine probes) plus drawing.

### Left to do in M5e

1. ~~After M5c merges to main: `w` Run on presets and `m` Catalog~~
   **built 2026-11-22** (rows above) on branch `worktree-m5-leftovers`;
   merge after `cargo test --workspace` runs green, deploy after the
   no-redeploy window (2026-11-27).
2. See mbp2's screens live once it is awake (read-only).
3. ~~Small: the Macs' `free ?` per card; `status --all` serial fetch~~
   **built 2026-11-22** (row above).
