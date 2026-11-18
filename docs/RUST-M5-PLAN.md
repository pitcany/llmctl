# Rust successor — M5 plan (daily-use replacement)

Status 2026-11-18: **plan agreed — all seven owner decisions taken, each
as recommended** (end of this file). M3's sleep/wake drill is done
([report](RUST-M3-SLEEP-DRILL-2026-11-18.md)). **Added the same day at
the owner's request: M5d, llama.cpp on the Macs and llama.cpp serving
several models**; its decisions 8–10 were taken as recommended. Nothing of M5 is
built yet.
Parent: [RUST-SUCCESSOR-PLAN.md](RUST-SUCCESSOR-PLAN.md) (milestone table,
M5 row; "Outcome and scope"; "Cutover and reversal"). Builds on M4
([plan and exit check](RUST-M4-PLAN.md)): every day-one engine has an
adapter, pins, swaps with rollback, request safety and manual-stop
protection, and llm-router already hands three workloads to llmctl-next.

M5 deliverable (parent): *minimal multi-host TUI, config/preset parity,
catalog operations, migration/reversal rehearsal.* Exit criterion,
verbatim: *Daily workflows pass across workstation and Macs without
legacy lifecycle writes; at least seven consecutive days of recorded
pilot use.*

## What M5 means in practice

At the end of M5 the owner runs the day from llmctl-next: the TUI to look
and act, `run`/`stop` from the shell, presets and the model catalog edited
through it, on all three hosts. The Python llmctl and the Mac llmctl stay
installed, but every lifecycle write they would make for an owned
workload either goes through llmctl-next or does not happen. The agent
proves that with its own record for seven days in a row.

**Not in M5:** quiescing and removing the old controllers, switching
clients for good, and the final import (M6); the web dashboard and the
benchmark UI (later, per the parent plan); SGLang and FreeToken (M7).
The old tools keep running in M5 so that reversal is a config change, not
a restore.

## Facts that shape the plan (checked 2026-11-18, read-only)

**How the owner uses llmctl today** (shell history on the workstation,
`llmctl <verb>` counts): `tui` 111, `health` 106, `status` 72, `vllm` 24,
`presets` 14, `swap` 11, `doctor` 6, `stop` 4, `sessions` 4, `run` 3.
The TUI is the daily interface; status and health come next; switching
mostly happens inside the TUI (Run screen) or through `ccl`/llm-router.
Confidence: medium (history covers one shell; the Macs and the TUI's own
key presses are not counted).

**Python TUI** (`llmctl/tui/`, 6.5 k lines, Textual): eleven screens —
Dashboard `d`, Run `w`, Presets `p`, Units `u`, Models `m`, Profiles `f`,
Sessions `s`, GPUs `g`, Events `l`, Doctor `o`, Benchmarks `b`.

**Rust today:** `llmctl-cli` has `status`, `run`/`stop`/`restart`/`unload`,
`logs`, `ops`/`op`/`cancel`/`ack`, `intent`, `vram`, `lanes`
(swap/stop/memory/logs/import/…), `model` (get/rm), `stores`, `recipe`,
`version`, `install`/`upgrade`, `adopt`/`own`/`release`/`forget`, `host`
(add/upgrade/remove/list) and `bridge`. **No TUI crate**, no `ratatui`
dependency. No preset or catalog verbs: `run --profile-env` takes a raw
env file.

**Live ownership:** workstation agent `llmctl-next.service`, inventory
revision 43, six workloads (`vllm-engine`, `llmster`, `ollama` running;
`gpt-oss-120b`, `ollama-gpu1` manually stopped; `strata-fast` stopped).
Agents on m5 and mbp2 answer (`host list`: agent 0.1.0, protocol 1).
llm-router hands `vllm-engine` profiles, `strata-fast` and `gpt-oss-120b`
to llmctl-next; its other backends still use the router's own path.

**Configuration and catalog the Python side owns:**

| What | Where | Size |
|---|---|---|
| Presets | `~/.config/llmctl/presets/*.yaml` | 5 files (qwen27b family) |
| Settings | `~/.config/llmctl/settings.yaml` (private repo `llmctl-config`; holds a token) | 1 |
| Router profile envs | `~/AI/services/vllm-engine.env.<profile>` | one per vLLM profile |
| Model registry | `~/.local/share/llmctl/llmctl.sqlite3`, table `models` | 142 rows |
| Launch profiles | same DB, table `profiles` | 2 rows |
| Model dirs | `~/.config/llmctl/model_dirs.yaml` | 1 |
| Router aliases | `~/.config/llmctl/router_aliases.json` | 1 |

`router-profiles-check.timer` already compares router profile envs with
presets every hour (read-only).

**Who can still write lifecycle without llmctl-next** (the list the
"without legacy lifecycle writes" criterion is about):

| Actor | Write path | Today |
|---|---|---|
| Python llmctl CLI and TUI | `services/sessions.py`, `swap_jobs.py`, `units.py` → `systemctl` | `llmctl run` goes through the router for the three handed-over workloads; other paths call systemd directly |
| llmctl-api dashboard `:8088` | `POST /sessions/start`, `/{id}/stop`, `/{id}/restart`, `/swap/run`, `/swap/stop` | same service layer as the CLI |
| llmctl-stack `take-gpus` | `systemctl --user stop <unit>`, or `llm-unit stop` / `sudo -n systemctl stop` | `llmctl-stack take-gpus <lane> --take`, and router lanes started with `--take-gpus` (llm-heal's notes) |
| llm-router | own switch path for backends that are not handed over | three handed over |
| llm-heal / mac-heal | restarts; ask llmctl-next's intent first since M4 | fail open if the agent is down |
| Mac llmctl (`~/local-llm/bin/llmctl` on m5) and llmctl-warm | llama-swap loads | `warm` never loads a lane (M4c) |
| `llmctl-reconcile.timer` | writes session rows in the Python DB, not unit state | not a lifecycle write |

**Hazards carried from M4 that M5 touches:** three vLLM versions named for
`vllm-engine` and an env header that names a preset file that does not
exist (preset parity has to settle one source); mbp2 disk 93 % full (blocks
catalog downloads there); m5 has no `hf` (blocks `model get hf:` there).

## Design decisions (delegated defaults, per the parent plan)

1. **The agent keeps its own record of foreign writes.** Every observation
   pass compares a managed workload's state with the journal. A change of
   running/stopped, pid or served model that no admitted operation explains
   becomes a `foreign_transition` finding (host, workload, before, after,
   time, and the actor when systemd/launchd names one). `llmctl-next audit
   [--since]` and `GET /v1/audit` list them. This is the evidence for "no
   legacy lifecycle writes", and the pilot clock reads it. Healer repairs
   that went through llmctl-next are operations, not findings.
2. **Legacy writers delegate; they are not removed.** For a workload
   llmctl-next owns, the Python service layer (CLI, TUI, dashboard), the
   llmctl-stack `take_gpus` and the Mac llmctl call llmctl-next instead of
   systemd/launchd/llama-swap. Exit 3/6 (agent down, protocol) is a refusal
   with a clear message, not a silent fallback to the old write. Workloads
   llmctl-next does not own keep the old path. Reversal = one setting
   (`llmctl_next: off`) in each tool.
3. **Presets stay in their files.** llmctl-next reads and writes the same
   `presets/*.yaml`, `settings.yaml` and router profile envs, in the same
   format, keeping comments (port of `yaml_edit.py`'s rules). No second
   copy, so reversal needs no export for presets. A preset resolves to a
   workload plus a profile env plus a served model; `run <preset>` is the
   daily verb, `--profile-env` stays for scripts.
4. **The catalog is imported, with a reverse export.** The 142 registry
   rows and the launch profiles move into agent records per host (model,
   engine kind, path or ref, store, size, enabled). The Python DB is not
   written by Rust. A reverse export writes a registry bundle the Python
   `import-registry` accepts, so catalog edits made during the pilot
   survive a reversal. Weights stay in the M4 stores; the catalog points at
   them.
5. **The TUI is a client, nothing more.** A `ratatui` crate
   (`llmctl-tui`, binary `llmctl-next tui`) that talks only to the local
   agent API and, through the bridge, to the Mac agents. It never probes
   engines or runs commands itself. Every action shows the plan's impact
   preview and asks before it runs, as the CLI does.

## Remaining work, in order

### M5a. Workflow matrix and the foreign-write record (~1–2 days)

1. `docs/RUST-M5-WORKFLOWS.md`: every daily workflow on each host, from
   the facts above plus the Mac llmctl and the healers — what the owner
   does, which tool does it today, the llmctl-next path or the gap. Fold
   in M4's carry-overs that are daily needs (enroll verb, authority
   change, matrix verb, profiles with the VRAM fit, per-lane in-flight).
2. Decision 1: `foreign_transition` findings in the observer, `audit` verb
   and API, kept in the journal DB with a retention bound.
3. Same check for lanes on the Macs (a lane loaded or unloaded with no
   operation behind it).

Tests: daemon tests with a fake supervisor that flips a unit behind the
agent's back (finding), and one that flips it through an operation (no
finding); agent restart keeps findings. **Live:** read-only run of `audit`
on all three hosts for one day before anything is delegated — the
baseline count of foreign writes.

### M5b. Delegation of the legacy writers (~2–3 days)

1. Python llmctl: one guard in the service layer (`sessions.start/stop/
   restart`, `swap_jobs`, `units`) — owned workload → llmctl-next CLI
   with `--wait`, same exit-code map as llm-router uses. CLI, TUI and the
   `:8088` dashboard all get it, since they share the layer.
2. llmctl-stack `take_gpus` → `llmctl-next stop` for owned units.
3. Mac llmctl on m5 (`~/local-llm/bin/llmctl`): load/unload of an owned
   llama-swap lane → `llmctl-next lanes swap/stop`.
4. llm-router: hand over the remaining backends that map to owned
   workloads (one router change, as in M4c).

Tests: Python tests for the guard with a stand-in `llmctl-next`
(delegated, refused when the agent is down, old path for unowned units);
llmctl-stack and router tests the same way. **Live:** one day of normal
use with `audit` on; the foreign-write count should fall to zero for
owned workloads.

### M5c. Preset/config parity and catalog operations (~3–4 days)

1. Preset reader/writer (decision 3), `preset list/show/diff/edit/new`,
   `run <preset>`. Settle `vllm-engine`'s version source here: the pin is
   the truth, the env header and router comment are rewritten to match.
2. **Parity check:** for each preset, the launch the Python side would
   render (unit env, args, served name) equals what llmctl-next renders.
   Reuse `tests/test_vllm_env_parity.py` and
   `tests/test_preset_save_keeps.py` as fixtures; run both
   implementations on the same sanitised inputs.
3. Catalog (decision 4): `catalog import` (dry run first, report of rows
   that do not map), `model list/show/add/edit/enable/disable` per host,
   `model scan` over the declared stores, `catalog export` for reversal.
   Existing `model get/rm` and `stores` stay.
4. Mac lanes: a verb for the llama-swap matrix (M4 carry-over 2), so the
   memory budget becomes a guard on writes, not only a report.
5. Enroll an unadoptable workload and change a record's authority by verb
   (M4 carry-over 1).

Tests: round-trip fixtures for every preset file (read → write → byte
compare apart from intended changes); parity fixtures; import of a copy of
the real registry into a temp agent with the mapping report checked;
export → Python `import-registry` in a temp DB → same rows.

### M5d. llama.cpp on every host, several models per server (~3–4 days)

Why: the parent plan requires llama.cpp on "compatible Linux/macOS
hosts". M4 met that on Linux only, because no Mac lane used
`llama-server` at the time. The owner asked on 2026-11-18 for llama.cpp
on m5 and mbp2, and for llama.cpp serving several models.

Facts (checked 2026-11-18, read-only):

| Host | llama-server | Router mode | GGUF models on disk |
|---|---|---|---|
| workstation | source build 10705 (CUDA), `~/AI/llama.cpp/build/` | yes | `Qwen3.5-0.8B-GGUF`, `Qwen3.6-35B-A3B-GGUF`, `gpt-oss-120b`, Qwen3.8-Flash-Next quants under `~/models` |
| m5 (M5 Max, 128 GiB) | brew 1.0.0 (build 12359), no lane uses it | yes | Qwen GGUFs under `~/Models/huggingface/Qwen`; GLM and DeepSeek files under `~/Models/gguf` (ds4's, not checked for llama.cpp) |
| mbp2 (M1 Max, 64 GiB) | brew 1.0.0 (build 12359), no lane uses it | yes | embedding GGUFs under `~/Models/gguf/embeddings`; LM Studio's GGUFs; 123 GiB free |

**Router mode** is llama.cpp's own multi-model server: `llama-server`
started without `-m` serves the models of `--models-dir` and/or a preset
INI (`--models-preset`, one section per model with its arguments).
`GET /models` lists each model with `status` (`unloaded`, `loading`,
`loaded`), `POST /models/load` and `/models/unload` take
`{"model": id}`, requests are routed by their `model` field,
`--models-max` caps loaded models (default 4), `--no-models-autoload`
stops a request from loading a model by itself, and `GET /models?reload=1`
re-reads the sources (source: `tools/server/README.md`, "Using multiple
models", in the workstation checkout `257813839`).

Design (defaults; decisions 8–10 below):

1. **A `llamacpp-router` workload with model records.** Each model is a
   typed record (id, GGUF file or shard directory, `mmproj`, arguments
   such as context and GPU layers, `load-on-startup`) and llmctl-next
   renders the preset INI from them. Same rules as llama-swap lanes:
   import and round-trip first, every write admitted, validated and
   atomic, then `?reload=1` and confirm, previous file kept for rollback.
2. **Verbs** in the lanes shape: `llamacpp models <workload>`,
   `llamacpp load|unload <workload> <model>` (admitted operations,
   confirmed by `GET /models`), `llamacpp logs`. Request safety as for
   lanes: in-flight counts from llm-router; a per-model count from the
   server where it has one.
3. **Loads only through llmctl-next** (decision 9): the router runs with
   `--no-models-autoload`, so the VRAM fit (workstation) or the memory
   budget (Macs) runs before every load, and a request for an unloaded
   model fails fast instead of loading behind the planner.
   `--models-max` follows the fit.
4. **Workstation: llama.cpp becomes a general engine, not one model.**
   Today llama.cpp serves only `gpt-oss-120b` there. M5d adds one user
   unit `llamacpp-router` on the CUDA build that serves any GGUF in the
   stores. A model load claims cards like the M4c on-demand claim, so it
   is planned against `vllm-engine` and the others (evict or refuse).
   `gpt-oss-120b` stays its own unit in M5 (decision 10). Candidate
   models (sizes on disk, 2026-11-18):

   | Model | Size | Fits the two cards (2 × 32 GiB) |
   |---|---|---|
   | `Qwen3.5-0.8B-GGUF` (`/mnt/storage/converted`) | 1 GB | yes (pilot) |
   | `Qwen3.6-35B-A3B-GGUF` (`/mnt/storage/converted`) | 37 GB | yes (pilot) |
   | Qwen3.8-Flash-Next quants (`~/models`, `/mnt/storage/models`) | 89–104 GB | no: needs llama.cpp's CPU offload of MoE experts into the 128 GiB RAM; speed not measured, Strata serves them today |
   | `Maya-M` (`/mnt/storage/models`) | 110 GB | no: same as above |

   A model that needs CPU offload carries its offload arguments in its
   record, and the fit counts its host-RAM share too; the first one is
   measured before it is offered in the TUI.
5. **Macs** (decision 8): llama.cpp as a llama-swap lane engine, one
   `llama-server -m <gguf>` per lane. llama-swap is already the Macs'
   multi-model scheduler, and `lanes memory` sums every lane on the one
   unified-memory pool. `lanes swap/stop/version` work unchanged.
6. **Install and pin:** a `release-archive` recipe for the macOS arm64
   llama.cpp release on both Macs (versioned directory, checksum, pin,
   rollback). That closes M4 carry-over 4 (release archives on launchd).
   brew's 1.0.0 stays installed but no lane runs it. The workstation keeps
   its source build directories.

Tests: a stand-in router endpoint (load, unload, status, reload, a load
that fails) for the verbs; preset INI round-trip fixtures; a refused load
when the fit or the budget says no; a request for an unloaded model does
not load it. **Pilot:** workstation router with `Qwen3.5-0.8B` and
`Qwen3.6-35B-A3B` loaded and unloaded by verb, each answering one request,
planned against `vllm-engine` (`longjob … --router`); m5 a llama.cpp lane
on a Qwen GGUF; mbp2 a llama.cpp lane on an embedding GGUF (an embedding
request answered). The workflow matrix (M5a) and the catalog (M5c) list
router models and llama.cpp lanes like any other model.

### M5e. Minimal multi-host TUI (~4–5 days)

Screens, matched to the daily ones in the Python TUI (decision 5):

| Key | Screen | Content |
|---|---|---|
| `d` | Hosts | three hosts, agent version, last seen, each workload's state/readiness/intent/freshness |
| `w` | Run | presets and workloads; enter = plan preview (stops, evictions, in-flight, rollback), confirm, then live progress |
| `o` | Operations | running and recent operations, phases, recovery details, `ack`/`cancel` |
| `l` | Logs | workload log and lane log, follow mode |
| `g` | GPUs / memory | card occupancy (workstation), memory budget and loaded lanes (Macs) |
| `m` | Catalog | models per host and store; add/rm through the same previews |
| `a` | Audit | foreign-write findings (decision 1) |

Not in the first TUI: Benchmarks, Doctor, Profiles editing (presets edit
from the shell in M5). Add a screen only if the workflow matrix shows a
daily need.

Tests: TUI state tests against a fake API (render snapshots per screen,
confirm/cancel paths, a stale plan refused); a slow-API test proving no UI
stall over 100 ms (the parent plan's budget), measured p50/p95 against the
Python TUI on the same machine.

### M5f. Migration and reversal rehearsal (~1–2 days)

On temporary state, both directions, as the parent plan's "Cutover and
reversal" section describes:

1. Forward: copy the Python DB, presets and settings into a temp dir;
   `catalog import` into a temp agent; run a disposable unit through a
   preset; check what is running against the agent's record.
2. Reverse: drain the temp agent, `catalog export`, Python
   `import-registry` into the temp DB, set `llmctl_next: off`, check the
   Python side sees the same running state and the edits made in step 1.
3. Write the procedure down (`docs/RUST-M5-REHEARSAL.md`) — M6 runs it
   for real.

### M5g. Seven-day pilot (calendar time, at least 7 days)

Daily use from llmctl-next on all three hosts. A day counts when (a) the
owner used llmctl-next (TUI or CLI) on that day and (b) `audit` shows no
unexplained foreign write on any host for owned workloads. A day that
fails (b) restarts the count after the cause is fixed. A short daily line
goes into `docs/RUST-M5-PILOT.md` (date, workflows used, operations,
findings, anything that needed the old tools).

## Exit checklist (maps to the parent criterion)

| Criterion | Evidence required |
|---|---|
| Minimal multi-host TUI | the M5e screens work against all three agents; UI budget measured |
| Config/preset parity | every preset round-trips and renders the same launch as Python; `run <preset>` used in the pilot |
| Catalog operations | real registry imported with a mapping report; add/rm/scan used live; export accepted by Python |
| llama.cpp on every host, several models | router models loaded and unloaded by verb on the workstation; a llama.cpp lane on m5 and on mbp2 from a pinned release archive; each answered a request |
| Migration/reversal rehearsal | both directions pass on temp state; procedure written |
| Daily workflows across hosts | every row of the workflow matrix has a passing llmctl-next path on its host, or an owner-accepted gap |
| No legacy lifecycle writes | `audit` shows zero unexplained foreign writes for owned workloads across the pilot |
| Seven consecutive days | seven counted days in `RUST-M5-PILOT.md` |

## Owner decisions (2026-11-18, all as recommended)

The owner took all seven as recommended: M5a–M5f (now M5a–M5g, with
M5d added on 2026-11-18); a `ratatui` TUI;
legacy writers delegate and refuse when the agent is down; presets edited
in place; catalog imported with a reverse export; a bad pilot day restarts
the count; `hf` installed on m5 in M5c.

1. **Split M5 into M5a–M5f** as above, a check at the end of each.
   *Recommended.*
2. **TUI technology:** a new Rust `ratatui` crate (recommended), or point
   the existing Python Textual TUI at the llmctl-next API (faster, but
   keeps Python in the daily path that M6 removes).
3. **Legacy writers during the pilot:** delegate to llmctl-next with a
   refusal when the agent is down (recommended), or disable their write
   paths for owned workloads outright (stricter, more friction if the
   agent is down).
4. **Presets:** llmctl-next edits the existing YAML files in place
   (recommended), or keeps its own copy with an export (two sources during
   the pilot).
5. **Catalog:** import with reverse export (recommended), or llmctl-next
   reads and writes the Python SQLite directly (no export, but two
   programs writing one DB).
6. **Pilot day rule:** a day counts only with zero unexplained foreign
   writes, and a bad day restarts the count (recommended), or bad days are
   logged and explained without restarting.
7. **m5 `hf`:** install `huggingface_hub`'s CLI on m5 in M5c
   (recommended), or keep HF downloads workstation-only and copy over.

### Added 2026-11-18 with M5d (taken the same day, all as recommended)

8. **llama.cpp on the Macs:** as llama-swap lanes, one `llama-server`
   per model (recommended: llama-swap already schedules every Mac engine
   and the memory budget sums its lanes), or a router-mode `llama-server`
   as its own launchd job (same verbs on every host, but two schedulers
   on one memory pool, which the budget would have to sum together).
9. **Autoload:** off, loads only through llmctl-next (recommended: the
   fit and the budget run before every load), or on (a request loads its
   model by itself; convenient, but behind the planner's back).
10. **`gpt-oss-120b`:** stays its own unit in M5 (recommended: it is
   handed over by llm-router today and works), or becomes a model of the
   workstation router.
