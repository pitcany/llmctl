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
| 2026-11-18 | **No stall over 100 ms**: every fake call 2 s, one host silent, keys every ~60 ms for 3 s | `slow_and_silent_agents_never_stall_a_frame`: key→frame max 2.9 ms, idle pass max 2.6 ms (debug build) |
| 2026-11-18 | **Live, all three agents, read-only** (release build, `tui --read-only` in tmux): workstation and m5 answered on every screen (m5: 4 workloads, llama-swap budget 104 GiB with `mimo9b-m5` loaded, lane_swap operations, 0 findings); mbp2 did not answer (asleep, last seen 5 h before) and showed STALE with its cached rows on Hosts and "STALE, skipped" on Ops/Audit/GPU, while the other hosts stayed live | mbp2's own screens not yet seen live: re-check when it is awake |
| 2026-11-18 | **Regression suites** with the TUI in the workspace | `cargo test --workspace`: 245 passed, 0 failed; `systemd_spike -- --ignored --test-threads=1`: 22 passed |

### Latency (2026-11-18, workstation, same machine, headless 160x50)

Same two measures on both TUIs: a screen key to the next idle frame, and a
status refresh from request to answer drawn. Rust: `llmctl-next tui
--bench 20` (release build; screens d o l g a). Python: Textual
`run_test` on `MissionControlApp` from `~/AI/.venv` (screens d g l s u;
`r` on the dashboard until `render_data` ran), script kept at
`scratchpad/bench_py_tui.py` of the M5e session — rerun it from any
directory with `~/AI/.venv/bin/python -I`.

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

1. After M5c merges to main: `w` Run on presets (enter = plan preview,
   confirm, progress — the workload path is built and tested) and `m`
   Catalog. Both are marked pending on screen now.
2. See mbp2's screens live once it is awake (read-only).
3. Small: on the Macs the GPU screen prints `free ?` for each card (no
   VRAM endpoint there); `status --all` in the CLI still asks hosts one
   after another — it could use the TUI's concurrent bounded call.
