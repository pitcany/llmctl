# Deployment and rollback plan — control-plane coordination

Prepared November 3, 2026 (America/Los_Angeles), continuing
[NEXT-AGENT-PLAN-2026-11-03](NEXT-AGENT-PLAN-2026-11-03.md). **Nothing below has
been executed.** Each release needs explicit approval; commits and pushes need
approval first because the stack installs llmctl from a pinned, pushed Git SHA.

## Implementation ledger

| Package | Files | Evidence (regression first) | Tests (isolated) | Installed | Live validation | Remaining limits |
| --- | --- | --- | --- | --- | --- | --- |
| A — handoff fixes reviewed | llmctl `config.py`, `adapters/vllm.py`, `engine_load.py`, `validate.py`, `systemd.py`, `api/routes_sessions.py` (+tests, docs) | All 7 `roles()` consumers checked: adopt skips unreachable units, lane-derived rows yield to settings by port, `install-vllm-unit` can now target the user-scope engine 2 | Full llmctl suite with all changes: 1,677 passed, 2 skipped (no `llm_models_config` in the dev venv; no `node` on the isolated PATH), 314 s, temp HOME/XDG/DB/lock/model caches, inert systemctl/sudo/docker/ssh shims; ruff and `git diff --check` clean. Candidate wheel sha256 `a89222f7…` (working tree, not yet a commit); plugin 114/114 and llm-models-config 106/106 on candidate and on installed baseline | No (`16dc60f`) | None | `lanes.gather_state` still probes the main engine as system scope (`is_active(vllm.unit_name, False)`); harmless today (main engine is a system unit) |
| B — shared GPU ownership | llmctl `services/op_lock.py` (`handed_off`), `cli_swap.py`; router `op_lock.py`, `config.py`, `switcher.py`; heal `llm-heal.py`, `heal.yaml` | 6 llmctl, 9 router, 4 heal tests failed before the change (CLI held the flock across `/admin/switch`; router restarted vLLM while another process held the flock, or for a model engine 2 had meanwhile started; healer restarted vLLM under a live lease) | llmctl lock/swap/ops/regression 74 passed; router 261 passed; heal + maint 101 passed (1 pre-existing failure, below); cross-implementation interop both ways | No | None | Raw `systemctl`, systemd auto-restart, Mac services and hand-run start scripts stay outside; rollback restart in the router is not health-waited after the lock is released (unchanged behaviour) |
| C — idle revert safe for active requests | router `idle.py`, `app.py`, `switcher.py` (`veto`) | 8 router tests failed before (response longer than the timeout, request accepted mid-switch, per-instance accounting, cancel/error/drop, end-to-end stream) | Included in router 261 | No | None | Non-vLLM lanes have no engine gauges: direct traffic to them is invisible (documented; use a lease). Default 3,600 s unchanged |
| D — deployment | this document | — | — | **R1 deployed 2026-11-03 19:47 PST**: `~/AI/.venv` = `84ed67f2` (both pins, `uv.lock` only llmctl lines), `llmctl-api` restarted (PID 3727568, 0 restarts) | `/health` 200 in 2 s; `llmctl status` shows `vllm-engine2` + 7 plugin lanes; plugin tests 220/220 on the installed venv; 0 error lines in api/reconcile journals; router unchanged (`qwen27b`) | R2, R3 need approval. `llmctl-reconcile.timer` found stopped since 18:19:43 (explicit stop, before R1); left stopped pending owner decision. Backup: `~/AI/logs/staged-2026-11-03-control-plane/LAST_BACKUP` |
| P1 (Nov 4) — review fixes | router `idle.py`, `switcher.py`, `app.py`, `config.py`; heal test | Review reproduced 2 defects (engine-2 activity leak; request after veto met the restart), then an independent review reproduced 4 more (resident CPU Ollama blocked every revert; `current` skipped the drain gate; abort after `stop_first` stranded the GPUs; engine-2 model restarted :8003) | Router 293 passed, heal 50 passed, ruff clean | **R2 deployed 2026-11-04 ~04:40 PST**: live `~/AI/llm-router` `4c66a7b` → `e0377a0` (code `4a02f56` + enable `e0377a0`), `llm-router` restarted. **R3 deployed ~04:45**: live `~/AI` `14c900bb` → `3993e2c8` (healer branch rebased onto `14c900bb`; unrelated benchmark files and the uncommitted R1 pins untouched) | Router `/health` 200 after 2 s with the same backend `qwen27b`; `/running` has `idle_revert` and `last_rollback: null`; one completion through LiteLLM `local` succeeded and was released (`in_flight: {}`); op lock free; healer dry run clean with `/usr/bin/python3` (the plan's bare `python3` is brew 3.14 without `yaml`) | Planned check "`llmctl run <current>` reports Already serving" was NOT run: its preview showed it would restart `vllm-engine` (~10 min). With `defer`, a GPU-resident Ollama model or a Strata lane without gauges blocks idle revert (by decision). Rollback: `git -C ~/AI/llm-router revert --no-edit e0377a0 4a02f56` + restart `llm-router`; `git -C ~/AI revert --no-edit 3993e2c8` |
| P2 llmctl half | `presets/{sources,migrate,yaml_edit,store}.py`, `cli.py`, tests | DevLab run `20261104-043018-one-preset-store-ee2e`; 11 hidden tests fail on base | 1,700 passed under bwrap; hidden 11/11; plugin 114 + llm-models-config 106 against the candidate and again on the installed venv | **Deployed 2026-11-04 ~05:40 PST**: branch pushed `84ed67f..660510e`; both pins → `660510e2`, `uv.lock` llmctl lines only, `uv sync --locked --inexact`, `llmctl-api` restarted (PID 4064514, 0 restarts) | `/health` 200 in 2 s; `presets sources` on live: 4 presets, 0 collisions, 0 problems, both preset dirs unchanged; `settings qwen27b --json` 25 fields from the preset, 2 defaults; 0 API journal errors | The live preset migration (`presets migrate --apply`) has not been run. Rollback: restore the 3 files from `~/AI/logs/staged-2026-11-03-control-plane/backup-p2-20261104-053749/`, `uv sync --locked --inexact`, restart `llmctl-api` |
| E, F | not started | — | — | — | — | — |

The previously failing `~/AI/tests/test_llm_heal.py::test_lanes_come_from_the_shipped_router_yaml`
was fixed in R3 (it hard-coded the retired muse lane's 600 s allowance).

## Where the staged work lives

| Repo | Base | Staged as | Durable copy |
| --- | --- | --- | --- |
| `~/Projects/llmctl` | `b72b874` (= `origin/main`) | uncommitted working tree (prior handoff fixes + `handed_off`) | the working tree itself |
| `~/AI/llm-router` (own repo; service runs from this working tree) | `4c66a7b` | worktree branch `feat/shared-op-lock-2026-11-03` | `~/AI/logs/staged-2026-11-03-control-plane/llm-router.patch` |
| `~/AI` (heal runs from this working tree every 2 min) | `950c9135` | worktree branch `feat/heal-op-lock-2026-11-03` | `~/AI/logs/staged-2026-11-03-control-plane/ai-heal.patch` |

The worktrees are in this session's scratchpad under `/tmp`; the patch copies
survive a reboot (then `git worktree prune` in both repos).

## Scheduling constraint

A GPU window is booked for **2026-11-03 ~22:00 to 11-04 02:00** (hf2local Qwen3.6-35B;
it stops `vllm-engine` and Ollama). Do not run R2/R3 or any live acceptance
inside it; R1 (API restart only) is safe outside an active swap. A router restart
during that window would also reset its idle clock and holds mid-experiment.

## Release order and why

1. **R1 — llmctl** (candidate + `handed_off`). Safe alone.
2. **R2 — llm-router** (lock + in-flight idle). Its lock must never be enabled
   with an llmctl that holds the flock but lacks `handed_off` (`b72b874`):
   routed `llmctl run` would then wait 30 s and fail. Installed `16dc60f` has
   no flock and is also safe, but R1 first is simplest.
3. **R3 — llm-heal** (lock + lease deferral + `llmctl run` pattern). Takes
   effect on the next timer tick; no restart.

Each can be rolled back independently. R2 without its config line
(`op_lock_file`) still ships the idle in-flight protection.

## R1 — llmctl control-plane package

**Contents:** `b72b874` (operation lock, durable operations, benchmark
provenance — already reviewed in the audit) + second-engine inventory/scope/
settings fixes + `handed_off`. No DDL pending: the live DB already has the
`operations` table and `benchmarks.manifest` column (checked read-only), and
`16dc60f` ignores both, so the database stays compatible for rollback.

**Impact:** `llmctl-api` restart (TimeoutStopSec 30 s; a few seconds of :8088
unavailability). No engine, router, LiteLLM or Mac restart.

**Needs approval for:** commit on branch `fix/control-plane-coordination-2026-11-03`,
push to `origin` (the pin needs a pushed SHA), editing two pins in `~/AI`, `uv
lock`/`uv sync`, restarting `llmctl-api`.

Preflight (read-only):
```bash
pgrep -fc 'llmctl (run|swap)' || true            # expect 0: no swap in flight
test -e ~/.local/share/llmctl/operations.lock && cat ~/.local/share/llmctl/operations.lock  # expect absent (old build)
curl -s 127.0.0.1:4100/health                    # record serving model
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

Steps:
1. Commit (three commits: inventory/scope/settings fixes; `handed_off`; docs) and push the branch. Record `SHA`.
2. Back up state:
   ```bash
   B=~/AI/logs/staged-2026-11-03-control-plane/backup-$(date +%Y%m%d-%H%M%S); mkdir -p "$B"
   sqlite3 ~/.local/share/llmctl/llmctl.sqlite3 ".backup '$B/llmctl.sqlite3'"   # online backup API, WAL-safe
   cp -p ~/AI/uv.lock ~/AI/packages/llmctl-stack/pyproject.toml ~/AI/packages/llm-models-config/pyproject.toml "$B/"
   ```
3. Replace `16dc60f612849fbc4345c69db584effb58ed2016` with `SHA` in
   `~/AI/packages/llmctl-stack/pyproject.toml` and
   `~/AI/packages/llm-models-config/pyproject.toml`.
4. `cd ~/AI && uv lock --upgrade-package pitcany-llmctl && git diff --stat uv.lock` — expect only the llmctl entry to change; stop if anything else moves.
5. `uv sync --locked --inexact` (`--inexact` as in the stack's documented bump: a plain
   exact sync would remove packages in `~/AI/.venv` that the lock does not list), then from `/tmp`: `~/AI/.venv/bin/llmctl --version` must print `0.3.0+<SHA[:8]>`, and `~/AI/.venv/bin/python -c "import llmctl.services.op_lock as m; print(m.__file__, hasattr(m, 'handed_off'))"` must point into `~/AI/.venv` and print `True`.
6. `systemctl --user restart llmctl-api`.

Verify: `systemctl --user show llmctl-api -p ActiveState -p NRestarts`; `curl -s 127.0.0.1:8088/health`;
`llmctl status` lists `vllm-engine2` (inactive is expected) and plugin lanes;
`llmctl sessions --no-fresh`; journal of `llmctl-api` and `llmctl-reconcile` for
10 minutes shows no `LookupError`/migration error; router still serves the same model.

Rollback triggers: API not healthy within 60 s, reconcile errors, missing plugin
lanes, CLI import errors. Rollback: restore the two pyproject files and
`uv.lock` from `$B`, `uv sync --locked --inexact`, check `llmctl --version` prints
`+16dc60f6`, restart `llmctl-api`. Do not restore the DB backup over a live DB
(the new build writes only `operations` rows, which the old build ignores); keep
`$B` as evidence.

## R2 — llm-router

**Contents:** `op_lock.py`; `SwitchCfg.op_lock_file`/`op_lock_wait_s`
(default off); revalidation under ownership; per-upstream in-flight accounting,
`track_response`, veto under ownership, vLLM-gauge veto; `/running.idle_revert`;
`config/router.yaml` gains `op_lock_file` + `op_lock_wait_s: 30`.

**Impact:** `llm-router` restart. Requests through :4100 fail for a few seconds;
uvicorn drains open requests on SIGTERM, but systemd kills after 90 s, so a
longer stream would be cut. In-memory router state resets (failure cooldowns,
min-hold, idle clock: the loaded lane is re-seen and gets a fresh 3,600 s). No
vLLM restart.

**Needs approval for:** commits on the router branch, fast-forwarding the
live `~/AI/llm-router` main to it, restarting `llm-router`.

Preflight: R1 verified; `curl -s 127.0.0.1:8003/metrics | grep -E '^vllm:num_requests_(running|waiting)'`
shows 0/0 (quiet); `curl -s 127.0.0.1:4100/running` shows no `switching`;
`python3 ~/AI/heal/llm-heal.py pause 20m` (healer must not react to the restart).

Steps:
1. Commit in the worktree (lock; idle in-flight; config enable as its own commit).
2. `git -C ~/AI/llm-router merge --ff-only feat/shared-op-lock-2026-11-03` (live tree has only untracked backups; record old HEAD `4c66a7b`).
3. `~/AI/llm-router/.venv/bin/python -c "from llm_router.config import load_config; c=load_config('/home/yannik/AI/llm-router/config/router.yaml'); print(c.switch.op_lock_file)"`.
4. `systemctl --user restart llm-router`.

Verify: `/health` 200 with the same `active_backend`; `/running` contains
`idle_revert`; one routed completion via LiteLLM succeeds; `llmctl run <current
model>` reports "Already serving" (exercises `handed_off` + router lock without a
swap); the lock file shows a `llm-router` or `cli` holder only during such calls.

Rollback triggers: router unhealthy, 503s naming the op lock without a real
holder, any unexpected switch. Rollback: `git -C ~/AI/llm-router revert --no-edit`
the merged commits (non-destructive) and restart `llm-router`. Fastest partial
rollback: delete the two `op_lock_*` lines from router.yaml and restart (keeps
the in-flight protection).

## R3 — llm-heal

**Contents:** op-lock deferral/holding, live-lease deferral, `llmctl run` busy
pattern, `op_lock_file` in `heal.yaml`, tests.

**Impact:** none immediate; next 2-minute tick uses it. **Needs approval for:**
commits on the `~/AI` branch and fast-forwarding live `~/AI` main.

Steps: `python3 ~/AI/heal/llm-heal.py pause 10m`; commit; `git -C ~/AI merge --ff-only feat/heal-op-lock-2026-11-03`
(live `~/AI` has unrelated modified benchmark files: a fast-forward does not
touch them, verify with `git status` before and after); `python3 ~/AI/heal/llm-heal.py run --dry-run`
must list findings without errors; `python3 ~/AI/heal/llm-heal.py resume`.

Verify: `llm-heal status` fresh after the next tick; `~/AI/logs/heal/state.json`
updated; no new `healer_error` events.

Rollback: `git -C ~/AI revert --no-edit <merge commits>` (or remove the
`op_lock_file:` line from `heal.yaml` to disable only the lock). Takes effect on
the next tick.

## Separately approved live acceptance (after R1–R3)

Not covered by these releases: competing `llmctl run --direct` vs a client
request for another lane; long stream across an idle tick (temporarily set
`idle_revert_s` low on a test lane only, then restore 3,600); TP=2 takeover;
failed-start rollback; healer deferral during a real swap. Each restores the
original serving model and respects leases.

**Run 2026-11-04 09:21–09:46 PST:** competing request, long stream, TP=2
takeover and failed-start rollback pass. Healer deferral was not exercised live.
Record and findings: [LIVE-ACCEPTANCE-2026-11-04.md](LIVE-ACCEPTANCE-2026-11-04.md).
