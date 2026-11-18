# M5 workflow matrix — what a day looks like, per host

Status 2026-11-18: written in M5a from the facts in
[RUST-M5-PLAN.md](RUST-M5-PLAN.md) (checked read-only the same day) plus a
read of the tools named below. Every row names the llmctl-next path that
exists today or the M5 phase that closes the gap. The exit check walks this
table: *every row has a passing llmctl-next path on its host, or an
owner-accepted gap.*

Sources read: `llmctl/cli.py` (Python CLI surface), `llmctl/tui/`
(screens), `~/local-llm/bin/llmctl` on m5 (Mac CLI, usage block and
`cmd_*` functions), `~/AI/packages/llmctl-stack/llmctl_stack/take_gpus.py`,
`~/AI/tests/test_llm_heal.py` + `test_mac_heal.py` (healer behavior since
M4), the live inventory `~/.local/state/llmctl-next-m4a/inventory.json`
(revision 43, six managed workloads), and `llmctl-next --help` verbs.

## Legend

- ✅ path exists in llmctl-next today
- ◐ partial (named limitation) · ❌ gap → the M5 phase that closes it
- Hosts: **WS** workstation, **m5**, **mbp2** (Mac agents enrolled, protocol 1)

## Daily reads (no writes)

| # | Workflow | Today | llmctl-next | Status |
|---|---|---|---|---|
| R1 | One look at the whole day: workloads, readiness, GPUs | Python TUI `d` (`llmctl tui`, 111 uses), `llmctl status` (72) | `status` / `status --all` (all three hosts, freshness) | ✅ CLI; TUI Hosts screen M5e |
| R2 | Health of engines/endpoints | `llmctl health` (106) | `status` readiness + `vram` | ◐ no aggregated pass/fail yet; TUI M5e |
| R3 | Sessions / what ran | `llmctl sessions`, `:8088` dashboard | `ops`/`op` (durable operations) | ◐ Python session rows (incl. adopted ones from `ds4-llmctl-sync`) stay in the Python DB until M6 |
| R4 | GPU/memory occupancy | `llmctl status`, `nvidia-smi` | `status` per-card views (enrolled vs foreign), `vram` | ✅ |
| R5 | Mac memory budget, loaded lanes, swap pressure | Mac `llmctl status`/`memory`/`budget` | `lanes memory`, `lanes status` | ◐ `lanes status` exists; budget verb parity folded into M5c item 4 |
| R6 | Logs of a workload/lane | `journalctl`, Mac `llmctl logs [-f]` | `logs [--follow]`, `lanes logs` | ✅ |
| R7 | Foreign writes since yesterday | none (invisible) | `audit [--since]` | ❌ M5a (this phase) |

## Switching and lifecycle (writes)

| # | Workflow | Today | llmctl-next | Status |
|---|---|---|---|---|
| W1 | Switch the default vLLM profile | TUI Run screen, `llmctl swap/vllm` (11/24), `ccl` via llm-router | `run --profile-env` (planned, confirmed, rolled back); llm-router hands the three workloads over already | ◐ `run <preset>` by name ❌ M5c item 1 |
| W2 | Load/stop a lane (Strata, gpt-oss) | llm-router switch path | `lanes swap/stop` (admitted, confirmed by llama-swap) | ✅ (remaining router backends hand over in M5b item 4) |
| W3 | Stop a workload for a while (must stay down) | `llmctl stop` (4) | `stop` with manual-stop protection (survives eviction/rollback) | ✅ |
| W4 | Restart/unload a model | `llmctl` restart, Ollama evictions | `restart`, `unload` with in-flight safety | ✅ |
| W5 | Start a lane/model on a Mac | Mac `llmctl start MODEL` | `lanes swap` on the Mac agents | ◐ budget guard as write-blocker M5c item 4 |
| W6 | `take-gpus` before a heavy job | `llmctl-stack take-gpus <lane> --take` (stops units itself) | — | ❌ delegates to llmctl-next in M5b item 2 |
| W7 | gpt-oss-120b start/stop | own unit, llm-router `gptoss` backend | `run/stop gpt-oss-120b` (owned) | ✅ until it moves into the llama.cpp router (M5d step 7) |

## Config, presets, catalog (writes)

| # | Workflow | Today | llmctl-next | Status |
|---|---|---|---|---|
| C1 | Edit/list presets | `llmctl presets` (14) | — | ❌ M5c item 1 (same files, comments kept) |
| C2 | New model into the catalog (download) | `llmctl model add/hf get` | `model get`, `stores add` | ◐ list/show/add/edit/scan ❌ M5c item 3 |
| C3 | Remove a model from disk+catalog | `llmctl delete-model` | `model rm` | ✅ |
| C4 | Router aliases | `llmctl aliases/set-alias` | — | ❌ covered with C1 (M5c: `~/.config/llmctl/router_aliases.json` read/write) |
| C5 | Model dirs | `~/.config/llmctl/model_dirs.yaml` | `stores` | ◐ parity with the YAML file in M5c item 3 |
| C6 | Mac registry (models.yaml) | Mac `llmctl download/discover/list/info` | lane store per workload + `lanes import` | ◐ catalog verbs M5c item 3; `hf` on m5 (decision 7) M5c |

## Background writers (things that act without the owner)

| # | Actor | Today | M5 shape |
|---|---|---|---|
| B1 | Python llmctl CLI/TUI/`:8088` | `sessions/swap_jobs/units` → systemd directly for non-handed-over workloads | guard: owned workload → llmctl-next, refusal (not fallback) when agent down (M5b item 1) |
| B2 | llmctl-stack `take-gpus` | `systemctl --user stop` / `llm-unit stop` / `sudo -n systemctl stop` | `llmctl-next stop` for owned units (M5b item 2) |
| B3 | Mac llmctl (m5) | `llama-swap` loads through its own path (`start/stop` on the router) | owned lanes → `llmctl-next lanes swap/stop` (M5b item 3) |
| B4 | llm-router | own switch path for not-handed-over backends | hand the remaining owned-workload backends over (M5b item 4) |
| B5 | llm-heal / mac-heal | asks the agent's `intent` before repairs (since M4), then repairs through systemctl/launchctl itself; fail open | repairs that go through the agent are operations; direct repairs show as findings in `audit` (M5a) — full delegation is not in M5 |
| B6 | llmctl-warm | never loads a lane (M4c); `--guarded` no-ops when the router is down | no change needed in M5 |
| B7 | `ds4-llmctl-sync` | records adopted rows in the Python session table only; starts/stops nothing | kept all M5; retires at M6 with the Python reader |
| B8 | `llmctl-reconcile.timer` | writes session rows only | not a lifecycle write; untouched |

## M4 carry-overs folded into this matrix

1. **Enroll an unadoptable workload / change authority by verb** — today
   `adopt` refuses what it cannot prove; the verb pair lands in M5c item 5.
   Rows W2/W7 gain models with no systemd unit of their own.
2. **Llama-swap matrix verb** (several lanes loaded on purpose, budget as a
   write guard) — M5c item 4; row W5/R5.
3. **Profiles with the VRAM fit** — `run --profile-env` plans against the
   fit since M4c; preset parity (M5c item 1) makes it the default path for
   W1.
4. **Release archives on launchd** — recipe exists; a real launchd install
   happens with llama.cpp on the Macs (M5d item 6).
5. **Per-lane in-flight** — `lanes swap` counts router in-flight since M4c;
   the llama.cpp router models get the same safety in M5d item 2.

## What the pilot day uses (M5g)

A counted day touches, at minimum: one read row (R1/R4/R5/R7), one write
row (W1–W5), and `audit` clean (R7) on every host that day. Rows still ❌
at pilot time must be owner-accepted gaps and say so here.
