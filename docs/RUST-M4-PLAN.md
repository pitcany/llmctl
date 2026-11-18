# Rust successor — M4 plan (required engines and coordination)

Status 2026-11-17: plan agreed — all eight owner decisions taken, each as
recommended (end of this file). **M4a done 2026-11-17 — see
[the pilot note](RUST-M4A-PILOT-2026-11-17.md). M4b steps 1–5 done:
[progress](RUST-M4B-PROGRESS-2026-11-17.md). M4c: all seven items have a
first version: [progress](RUST-M4C-PROGRESS.md). Exit check 2026-11-17:
1 row met, 4 partly, 1 not met. The other five rows closed on
2026-11-18, the last two at 02:13 PST: **all six rows are met**, each
with limits that are written down — see "Exit check" below. **M4 was
closed by the owner on 2026-11-18 (02:56 PST)**; what carries over is
listed at the end of the exit check.**
Parent: [RUST-SUCCESSOR-PLAN.md](RUST-SUCCESSOR-PLAN.md) (milestone table,
M4 row; "Engine support and ownership"). Builds on M2 (one owned Linux
workload, staged vLLM upgrade — [plan](RUST-M2-PLAN.md)) and M3 (Mac agent,
SSH bridge, llama-swap adopt/own/release — [plan](RUST-M3-PLAN.md),
[pilot](RUST-M3-PILOT-2026-11-17.md)). Engine/host targets:
[capability matrix](RUST-M0-CAPABILITY-MATRIX.md).

M4 deliverable (parent): *all day-one engine adapters/installers, model
acquisition, GPU/memory placement, swaps/rollback, router handoff and
companion hooks.* Exit criterion, verbatim: *Capability matrix passes
installation, version pinning, launch, readiness, stop, logs and
upgrade/recovery tests on applicable hosts; request safety/manual-stop
protection pass.*

## Scope

M4 is larger than M2 and M3 together, so it runs as three sub-milestones,
each with its own pilot and exit rows (owner decision 1, decided):

- **M4a — installs and versions.** One installer model for every engine
  kind; version facts, pins, staged upgrade and rollback per kind.
- **M4b — engine adapters and llama-swap.** Launch, readiness, served-model
  identity, stop and logs for each engine on each host; llama-swap lanes as
  managed records on the Macs.
- **M4c — coordination.** Model acquisition, GPU/memory fit, swaps with
  rollback, the llm-router handoff, companion hooks, request safety and
  manual-stop protection against the other actors.

**Not in M4:** the TUI, preset/config parity and catalog UI (M5); switching
clients or the router to llmctl-next for daily use and quiescing Python
llmctl (M6); SGLang and FreeToken (M7; SGLang is retired on the workstation
since 2026-11-05 anyway); LiteLLM config generation (neither codebase does
it today; `gen-local-lanes.py` stays the owner).

## Facts that shape the plan (checked 2026-11-17, read-only)

Three sweeps: the workstation, both Macs, and both codebases. Sources are
the commands and files named.

**Rust today.** Adapters are per *supervisor*, not per engine:
`systemd.rs` (user units only; system scope is refused by design since
M2), `launchd.rs`, `unitdef.rs`/`plistdef.rs`, and the `FakeEngine`.
`upgrade.rs` stages `uv venv` + `uv pip install pkg==ver` and is
Python-generic in shape, but `version.rs:validate_offline` imports
vllm/torch and `env_prefix` defaults to `VLLM`. There is no installer
trait, no engine-kind enum, no model-store type, no VRAM-MiB fit (only
card-overlap and foreign-occupancy guards in `ops.rs`), and no interop with
the legacy host flock. Recovery already never restarts a manual stop
(`recover.rs`).

**Python llmctl today** (`llmctl/services/engines.py` and neighbours):
installs vLLM (unpinned `pip install vllm`, in place), llama.cpp (latest
CUDA release, or a staged source build — the only staged install), Strata
(git clone + `setup.sh`) and Ollama (`curl install.sh | sh`, sudo). No
installer for LM Studio, MLX engines or ds4. It records versions, never
pins. It has VRAM fit (`vram_fit.py`), placement (`vllm_slots.py`,
`session_swap.py`), swap with rollback (`vllm_orchestrator.py`,
`lane_swap.py`), companion plugins (`plugins.py`), model add/remove over HF,
Ollama and `lms get` (`model_ops.py`, no hf2local). **Request safety and
manual-stop protection are not in Python llmctl:** draining and the
idle-revert veto live in llm-router; pause/resume lives in llm-heal.

**Workstation (2× RTX 5090).**

| Engine | Install / version | Supervisor | Notes |
|---|---|---|---|
| vLLM | miniconda envs: `vllm-muse` 0.30.0 (live), `vllm-serve` 0.25.1, `vllm-laguna` | **system** `vllm-engine.service` (TP=2, :8003); user `vllm-engine2` (:8010, M2 pilot) | launcher `~/AI/scripts/vllm-launcher.sh` + env file; **three sources of truth for the version** (env file: muse; `settings.yaml`: serve; `router.yaml` comment: laguna) |
| llama.cpp | source checkout `~/AI/llama.cpp`, three CUDA builds (`build/` = b10705, `build-b10679/`, `build-b10280/`) | user `gpt-oss-120b.service` (:8005) via `~/bin/llama-server-guarded` | build dirs are already version-per-directory |
| Ollama | vendor tarball 0.33.1 in `/usr/local` | **system** `ollama.service` + drop-in; user `ollama-gpu1` (:11435, bwrap GPU pin) | models `~/.ollama/models` |
| LM Studio | `llmster` 0.0.18-1 under `~/.lmstudio/llmster/<ver>/` | **none** (session scope, PPID 1) | `lms` at `~/.lmstudio/bin/lms` |
| Strata | git clones `~/AI/strata` (v0.1.44.1) and `~/AI/strata-0.2.8` (prebuilt `strata-dsv41` engine) | five user units, all inactive; GLM-5.3 via `~/AI/maya-src` | JSON lane files |
| ds4 | `~/Work/ds41-workstation-74aa754`, fork `pitcany/ds41-m5` branch `workstation/glm-ram-tier`, built `ds4-server` | **none** | nothing launches it |

Coordination actors: llm-router (:4100; swaps by copying a profile env onto
`vllm-engine.env`, `sudo -n systemctl restart vllm-engine`, waits for
`/v1/models`, rolls back; holds the flock `~/.local/share/llmctl/operations.lock`
and honours the lane lease and `min_hold_s`), llm-heal (every 2 min,
`mode: act`, restarts vLLM/Ollama via sudo and lanes via `systemctl --user`),
longjob (lane lease), hf2local (writes `/mnt/storage/converted`; vLLM serves
from `~/models/hf2local/`), companion hook `~/bin/ds4-llmctl-sync`
(`ExecStartPost`/`ExecStopPost` on Strata, Maya and gpt-oss units).
`HF_HOME=~/AI/cache/huggingface`.

**Macs.**

| | m5 (128 GiB, wired limit 115 GiB via LaunchDaemon) | mbp2 (64 GiB, wired limit unset, disk 93 % full) |
|---|---|---|
| llama-swap | v230, `-watch-config`, 30 KB config, matrix + preload, 14 lanes | v230, `-watch-config`, 3.3 KB config, 2 lanes |
| config owner | **hand-edited**, committed to `pitcany/AI-mac`; `history/` backups; `engines.lock` lists known-good engine versions + restore commands | **hand-edited**, own git repo `pitcany/llama-swap-m1` |
| MTPLX | venvs 2.11.0 and 2.12.0 (Python 3.14) | venv 2.11.2 (Python 3.12) |
| mlx-vlm | venv 0.6.6 | — |
| mlx-serve | versioned tarball dirs 26.10.1 (live), 26.9.5 (rollback) | — |
| ds4 | `~/Work/ds41-1017`, fork `main` bd450f8, `ds4-server` (no `--version`) | — |
| llama.cpp | brew 1.0.0 + a 0.4.0-dev build; **no lane uses llama-server** | same |
| Ollama | brew + own plist `com.pitcany.ollama`; **runs 0.34.3, brew has 0.40.16 installed (never restarted)** | the GUI app 0.40.16 under the vendor agent |
| LM Studio | app 0.4.28 | app 0.4.26 |
| actors | mac-heal (kickstarts llama-swap, Ollama, standby, forwarder), llmctl-warm, lms-stray-watch | mac-heal (kickstarts llama-swap — **conflicts with llmctl-next once it owns the label**; restarts Ollama by pkill + `open`; LM Studio via `lms server`) |

**Correction to the M3 plan.** M3 recorded that the Mac llmctl generates
llama-swap's `config.yaml` from its registry. It does not: `lib/llmconf.py`
only stores the config's *path*, and `lib/swapmatrix.py` is a port of
llama-swap's matrix solver used to *predict* evictions. Both configs are
hand-edited and committed to git. Taking over "generation" therefore means
taking over hand edits, not replacing a generator.

## Design decisions (delegated defaults, per the parent plan)

1. **One installer model, five recipe kinds.** Every managed engine
   instance has an `InstallRecipe` of one kind, and every kind installs into
   a new versioned directory, never in place:
   - `python-env` (vLLM, MTPLX, mlx-vlm, Strata's venv): `uv venv` +
     pinned packages — generalise `upgrade.rs`, move the vLLM import check
     into a per-recipe `validate` step.
   - `release-archive` (llama.cpp prebuilt, mlx-serve, Strata's prebuilt
     engine, `llmster`): URL + version + checksum where supplied, unpacked
     to `<root>/<engine>/<version>/`.
   - `source-build` (ds4, llama.cpp from source): git URL + commit + build
     command; the build runs in a staging dir and the binary is checked
     before the directory is promoted (the Python llama.cpp source path
     already does this).
   - `vendor-app` (Ollama, LM Studio): llmctl-next records and pins the
     version it observes and owns the service, but install/upgrade is an
     explicit, owner-run step with a tested backup/reinstall path (the
     parent plan's "cannot coexist" rule). No silent vendor auto-update
     counts as an upgrade.
   - `external` (generic OpenAI endpoints, anything not yet recipe-backed):
     observe-only, as today.
2. **Version facts per kind:** package metadata (python-env), the archive's
   version + checksum (release-archive), the git SHA plus the binary's own
   `--version` where it has one (source-build; ds4 has none, so its SHA is
   the identity), the running server's reported version (vendor-app —
   which is how the stale m5 Ollama gets caught). A mismatch between the
   pin and the running version is a drift finding, never auto-corrected.
3. **Launch definitions point at a version directory,** so promote and
   rollback swap one path in the unit/plist/lane and restart through an
   admitted operation (the M2 staged-upgrade flow, generalised).
4. **llama-swap lanes become typed records** on the Mac agent (engine kind,
   recipe/version, model path, args, ttl, groups/matrix membership), and
   llmctl-next renders `config.yaml` from them. Import first: the current
   hand-edited file must round-trip to equivalent YAML before any write
   (the M2/M3 "compare configuration" rule). Because both instances run
   with `-watch-config`, every write is an admitted operation: validate
   (port the matrix rules from `swapmatrix.py`), write atomically, then
   confirm llama-swap accepted it (`/running` + its log), and keep the
   previous file for rollback. The git commit to the config repo stays as
   history (owner decision 3, decided).
5. **Shared model stores are separate records** from engine installs, with
   an owner per root (HF cache, `~/models`, `/mnt/storage/*`, `~/Models`,
   `~/.ollama/models`, `~/.lmstudio/models`). Removing an engine never
   removes weights; removing weights refuses while a lane or preset
   references them.
6. **The legacy host flock is honoured.** Every Rust GPU operation on the
   workstation takes `~/.local/share/llmctl/operations.lock` (the flock
   llm-router and Python llmctl share) and the lane lease, so the three
   controllers cannot overlap during M4–M5.

## Remaining work, in order

### M4a. Installs and versions (~3–4 days)

1. `llmctl-core`: `EngineKind`, `InstallRecipe` (five kinds), `VersionFact`;
   extend `VersionRef`. Inventory schema bump with a migration test.
2. Generalise `upgrade.rs`/`version.rs`: recipe-supplied validate step,
   no `VLLM` defaults outside the vLLM recipe; gc stays per workload.
3. `release-archive` and `source-build` installers: staging dir, checksum
   where supplied, binary check, promote = rename, rollback = previous dir;
   provenance and build log kept with the version.
4. `vendor-app` observer: version from the running server (Ollama
   `/api/version`, `lms version`/`llmster` dir), installer provenance;
   upgrade = plan + owner confirmation + backup path, no automatic run.
5. `version` / `install` / `upgrade` CLI verbs over the existing API.

Tests: unit + fake-engine e2e for each kind (stage, verify, promote,
rollback, interrupted stage, gc); disposable spikes — a tiny archive and a
tiny C source build on Linux and on mbp2. **Pilot:** stage, promote and
roll back a real llama.cpp build on the workstation (the `gpt-oss-120b`
unit, inactive today) and a real mlx-serve version on m5 (26.10.1 ↔ 26.9.5
already on disk); record ds4's pin (fork SHA) without rebuilding.

### M4b. Engine adapters and llama-swap (~4–5 days)

1. Per-engine adapter on top of the supervisor: readiness URL and served
   model identity (vLLM/llama-server/MTPLX/mlx-serve `/v1/models`; Ollama
   `/api/ps`; LM Studio `lms ps`/`/v1/models`), logs, engine-level unload
   where the engine has one (Ollama, LM Studio).
2. Linux units: llama.cpp (`gpt-oss-120b`), Strata (one lane), `llmster`
   under a new user unit (decision 6), Ollama (`ollama-gpu1` first). ds4
   stays Mac-only in M4 (decision 5); the workstation build is untouched.
3. Move the two system units to user units (decision 4): `vllm-engine`
   (TP=2, :8003) and the main `ollama.service`, in one GPU window queued
   through `longjob --stop-engine`. Same env file and launcher; update
   llm-router's restart command, llm-heal's targets and the sudoers rule
   together; keep the system units disabled (not deleted) as the rollback
   path for one week.
4. Mac lanes: import llama-swap lanes as records (decision 4 above),
   round-trip both configs, then one managed lane per engine kind on m5
   (MTPLX, mlx-vlm, mlx-serve, ds4) and the two lanes on mbp2.
5. Mac vendor services: Ollama (m5 brew plist; mbp2 app agent — both kept,
   decision 8) and LM
   Studio — observe + start/stop through their own controls; record the
   console-login limit.

Tests: capability-matrix rows per engine × host (launch, readiness, stop,
logs) as disposable spikes first; llama-swap config round-trip fixtures
from both Macs (sanitised); a rejected matrix never reaches the file.

### M4c. Coordination (~4–5 days)

1. **Model acquisition:** HF snapshot into a declared store, `ollama pull`,
   `lms get`; adopt hf2local outputs as existing weights rather than
   re-implementing conversion (decision 7). Download progress through
   the operation journal; disk-space check first (mbp2 is at 93 %).
2. **Fit and placement:** port the Python VRAM fit (`fits`,
   `free_mib_now`, recorded VRAM per preset) into the planner; on Macs a
   unified-memory budget (`budget_gib` 104 / 46 from the Mac llmctl config,
   wired limit) instead of cards. "Unknown is never free" stays.
3. **Swaps with rollback:** a switch plan = stop/evict conflicts, start the
   target, wait ready, roll back to the previous set on failure — the same
   shape as llm-router's swap and `plan_run`'s eviction, extended to
   profiles (env-file sets) and llama-swap groups.
4. **Router handoff** (decision 2): llm-router keeps its
   `/admin/switch` front door in M4; for workloads llmctl-next owns, the
   router calls the llmctl-next API instead of `systemctl`, under the
   shared flock. llmctl-next reports in-flight counts from the router
   before any stop.
5. **Companion hooks:** declared pre/post hooks per workload, run inside the
   operation with bounded output (Ollama unload before vLLM, `ds4-llmctl-sync`
   adopt/detach, `lms-free-llms.sh`), with their undo run on rollback.
6. **Request safety:** a stop or swap of a serving workload asks the router
   for in-flight requests and waits up to a bound (or refuses without
   `--force`); a manual stop records intent so a swap's rollback never
   restarts it.
7. **Manual-stop protection against other actors:** llm-heal, mac-heal and
   llmctl-warm must not restart or load a workload that llmctl-next owns
   and that is manually stopped. Smallest change: each healer reads
   llmctl-next's intent (one local API call) and skips owned workloads;
   until then, owned workloads are on the healers' exclude lists. mbp2's
   mac-heal llama-swap target is the first case.

Tests: planner tests for fit/eviction/rollback; fake-engine e2e for request
safety (in-flight > 0 blocks) and manual-stop (rollback skips it); a drill
where a healer runs during an owned manual stop and does nothing.
**Pilot:** one real swap on the workstation through the router with
rollback forced, and one Mac lane swap that evicts by matrix.

## Exit checklist (maps to the parent criterion)

| Criterion | Evidence required |
|---|---|
| Installation | each recipe kind installs a real engine on an applicable host into a versioned dir with provenance |
| Version pinning | pin recorded per instance; drift between pin and running version reported (m5 Ollama is a live example) |
| Launch / readiness / stop / logs | matrix rows pass for vLLM, llama.cpp, Ollama, LM Studio, MTPLX, mlx-vlm, mlx-serve, Strata, ds4 on their hosts |
| Upgrade / recovery | staged upgrade + rollback per kind; interrupted stage recovers; vendor-app backup/reinstall path tested once |
| Request safety | in-flight requests block or delay a stop/swap; proven with a real request on a real engine |
| Manual-stop protection | a manual stop survives a swap rollback, an agent restart and a healer run |

## Exit check (2026-11-17, 21:40 PST)

Each row checked against the pilot notes, the test suites and live output
(`status`, `version`, `logs`, `lanes show`, `intent` on the workstation, m5
and mbp2). **Result: 1 row met, 4 partly, 1 not met. M4 has not passed its
exit yet.**

| Criterion | Verdict | Evidence | Missing |
|---|---|---|---|
| Installation | **Partly** on 2026-11-17 (2 of 5 kinds on a real engine). **Met since 2026-11-18**, see the fourth update below | `python_env`: vLLM 0.32.0 staged into a versioned env on `vllm-engine2` (M2 window 2). `source_build`: llama.cpp `a9d43ecc3` built, validated, manifest and build log kept (M4a). `release_archive`: real-systemd test `release_archive_stages_promotes_verifies_and_rolls_back` with a tiny archive | `release_archive` has installed no real engine (the mlx-serve dirs on m5 were already there) and has no launchd run. `vendor_app` installs nothing by design (it prints an owner-run plan) and that plan was never carried out. `external` is a record only |
| Version pinning | **Not met** on 2026-11-17 (2 of 10 enrolled workloads pinned). **Met since 2026-11-18**, see the update below | Pinned: `gpt-oss-120b` (binary digests), m5 `ollama` (vendor fact 0.34.3). Drift is reported: test `vendor_app_version_is_observed_pinned_and_drifts`; M4a saw m5 Ollama 0.34.3 against brew 0.40.16 | `version` says "no recipe; version pinning is off" for `vllm-engine`, `ollama`, `ollama-gpu1`, `llmster`, `strata-fast` and for `llama-swap` on both Macs. m5 `ds4` has a recipe with the fork SHA but `pinned: null`. The 15 lanes have no version records (an engine version is a macro path). m5 Ollama shows no drift today because its pin was set to the running 0.34.3 |
| Launch / readiness / stop / logs | **Partly** on 2026-11-17 (6 of 9 engines on a host). **Met since 2026-11-18** (9 of 9), see the third update below | Workstation, through admitted operations: vLLM (`vllm-engine`), llama.cpp (`gpt-oss-120b`, last 21:15), Ollama (`ollama-gpu1`), LM Studio (`llmster`), Strata (`strata-fast`). m5: MTPLX lanes loaded and evicted by `lanes swap` (20:22). `logs` answers for all six workstation workloads and for `llama-swap` (both Macs) and `ollama` (m5) | mlx-serve: restarted by a config reload and health-checked, never loaded by `lanes swap`. mlx-vlm and ds4 lanes: never loaded through llmctl-next; `logs ds4` is `unsupported` (the record has no plist). mbp2's MTPLX lanes: records and a rewrite only. m5 Ollama: never stopped or started (a restart would upgrade it). LM Studio on both Macs and Ollama on mbp2 are not enrolled. Lanes have no logs of their own (llama-swap's log). The per-row disposable spikes of M4b open item 5 were not written |
| Upgrade / recovery | **Partly** on 2026-11-17. **Met since 2026-11-18**, see the fourth update below | `python_env`: promote, rollback, and a promote killed mid-load that rolled back by itself (M2 window 2). `source_build`: stage → promote → rollback on `gpt-oss-120b` (M4a). `release_archive`: real-systemd test. Interrupted stage: never promotable and removed by gc (real-systemd tests). mlx-serve 26.10.1 → 26.9.5 on m5 as a lane macro change | `release_archive` upgrade and rollback on a real engine. The vendor-app backup/reinstall path "tested once": never run. The mlx-serve swap back was not run |
| Request safety | **Met** | Live 19:05: `restart vllm-engine` without `--force` refused with 12 real requests in flight after the 20 s wait; the engine's pid did not change. e2e `stops_wait_for_in_flight_requests_or_refuse_without_force`; lane swaps wait on llama-swap's in-flight count (e2e) | The "requests finish inside the wait, then the stop goes ahead" path is covered by tests only |
| Manual-stop protection | **Partly** on 2026-11-17. **Met since 2026-11-18**, see the second update below | Agent restart: `ollama-gpu1` was stopped at 14:06 and still carries the mark after every daemon restart since (last 21:01). Swap rollback: e2e `a_manual_stop_is_reported_survives_eviction_and_is_not_restored`. Healers: llm-heal's check against the real agent answers correctly; 58 + 16 healer tests | No live rollback past a manual stop. No live healer drill where it matters: llm-heal does not repair a cleanly stopped lane anyway, so its 2-minute runs prove nothing; the case the plan names (mac-heal kickstarting a manually stopped `llama-swap` on mbp2) was not run |

### Update 2026-11-18, 00:56 PST: version pinning is met

`6d33af6`, `0dfb880`. Every enrolled workload and every lane has a pin, and
`version <name>` / `lanes version <workload>` report drift against it
(exit 4). Read live from all three hosts after the last deploy, drift 0
everywhere:

| Host | Instance | Recipe kind | Pinned |
|---|---|---|---|
| workstation | `vllm-engine` | python_env | vllm 0.30.0, torch 2.13.0, transformers 5.17.0, the env's freeze digest, the launcher script |
| workstation | `strata-fast` | python_env | the venv (numpy 2.5.3, cublas 13.0.2.14, cuda-runtime 13.0.96, freeze digest), `server.py`, the engine binary and its library |
| workstation | `ollama` | vendor_app | running 0.33.1, the binary |
| workstation | `ollama-gpu1` | vendor_app | the binary. The running version is open: it is manually stopped |
| workstation | `llmster` | vendor_app | 0.0.18-1, the binary |
| workstation | `gpt-oss-120b` | source_build | as before (binary digest, `--version` output, guard script) |
| m5 | `llama-swap` | vendor_app | version 230, the binary |
| m5 | `ds4` | source_build | commit `bd450f844e37`, the `ds4-server` digest |
| m5 | `ollama` | vendor_app | running 0.34.3 (as before) |
| mbp2 | `llama-swap` | vendor_app | version 230, the binary |
| m5 | 13 lanes | per lane | mtplx 2.12.0 (5 lanes) and 2.11.0 (1), mlx-vlm 0.6.6, the mlx-serve binary, `ds4-server`, the helper scripts, the M1 forwarder script |
| mbp2 | 2 lanes | per lane | mtplx 2.11.2 |

What it took (`6d33af6`):

- **A latent refusal, fixed.** A vendor app pinned through its HTTP probe
  could not be started again once stopped: the start guard read "no
  answer" as drift. An unanswered HTTP probe is no longer drift and is
  never stored as a pin.
- **Pins for an install llmctl-next does not manage** (`ds4` on m5,
  `llama-swap` on mbp2): the facts live on the record.
- **Lane version records.** A lane's install is what its shell line runs:
  every program the command names and, for an engine in a Python env, the
  packages plus the content of the engine package's own files (mtplx on m5
  carries a patch that a reinstall of the same version reverts). A version
  swap written through `lanes macro` moves the pins; a hand edit does not.
- **`version <name> --repin`** and **`lanes version <workload> --pin`**
  accept a change and list what moved. `pin_files` on a workload pins
  extra files (a binary under `/usr/local`, an engine a script starts).

Evidence that drift is reported: real-systemd test
`vendor_app_version_is_observed_pinned_and_drifts` (now also: stop the
pinned app, no drift, start it again); e2e
`version_repin_records_the_pin_and_names_what_changed`; lanes e2e
`lane_versions_are_pinned_drift_is_reported_and_a_macro_swap_repins`.

Limits, said plainly:

- No drift was provoked on a real install (that would mean altering one).
  m5 Ollama is pinned to the 0.34.3 it runs, so hazard 1 (brew holds
  0.40.16) shows as drift only after the restart that upgrades it.
- `ollama-gpu1`: only its binary is pinned until it next serves (the agent
  pins the running version at its next start, or `--repin` does).
- The `deepseek41-1017-mxfp4-m5` lane starts through `serve.sh`, so its
  pin is that script; the `ds4-server` it execs is pinned on the `ds4`
  record and by the `glm53-m5` lane.
- Recipes are still attached by editing the inventory file with the agent
  stopped; there is no verb for it.
- On systemd a managed workload whose version facts drift is refused at
  start (llm-router then falls back to its own switch). launchd checks
  the program digest only.

### Update 2026-11-18, 01:23 PST: manual-stop protection is met

Two drills, both as `longjob` jobs, 01:16–01:23. The plan asks that a
manual stop survives a swap rollback, an agent restart and a healer run;
the agent restart was shown on 2026-11-17 (`ollama-gpu1`, stopped 14:06,
mark still there after every restart since).

**Swap rollback (`m4-drill-rollback`, workstation, 5 min 02 s).**
`ollama-gpu1` carried a manual stop. It was started from outside
(`systemctl --user start`), beside a `vllm-engine` restarted the same way,
because its unit only starts with 26 GB free on GPU 1. Then `llmctl-next
run vllm-engine --profile-env …qwen27b-unc-awq --model drill-wrong-model`:

- the plan named both as stop targets and kept the mark (`record intent:
  ollama-gpu1 → Stopped { manual: false }` → "kept: manually stopped");
- verification failed as intended (`endpoint serves qwen3.8-27b-unc-awq,
  expected drill-wrong-model`);
- the rollback restored the env, started `vllm-engine`, verified
  `qwen3.8-27b`, and skipped both `ollama-gpu1` steps: "skipped: manually
  stopped". `recovery: restored`, 189 s.

End state: default profile serving, `ollama-gpu1` inactive and still a
manual stop, lease and lock free. The default model was away about 5 min.
The switch and its rollback also passed the start guard with the new
version pins in place.

**Healer run (`m4-drill-macheal`, mbp2, 5 min 57 s).** `llmctl-next --host
mbp2 stop llama-swap` (launchd bootout, intent manual stop), then six
healer runs while it was down:

- **llm-heal** on the workstation, three runs (01:17, 01:19, 01:21), each:
  `stopped mac:mbp2-swap manual_stop manually stopped in llmctl-next on
  mbp2; left alone`. That is the intent check, asked over SSH.
- **mac-heal** on mbp2, three runs, each: `absent llama-swap … is not
  loaded (booted out on purpose?); left alone`. A repair would have been
  due on the second run.

`llama-swap` stayed down for all six and came back with `llmctl-next run
llama-swap` (pid 70780 → 2328, health OK). No lane was loaded on mbp2
before the drill.

Said plainly: mac-heal left it alone by its **own** rule for a job that is
not loaded, which comes before its llmctl-next check. A llmctl-next stop
on launchd always boots the job out, so mac-heal's intent check is a
second line that this path never reaches; it is covered by its tests
(`~/AI/tests/test_mac_heal.py`) only. llm-heal's check is the one that ran live.

### Update 2026-11-18, 01:54 PST: launch / readiness / stop / logs is met

Jobs `m4-lanes-m5` and `m4-lanes-mbp2` (01:51–01:52), each lane through
`llmctl-next --host <mac> lanes swap llama-swap <lane> --wait`, then one
real chat request through llama-swap, then the log through `llmctl-next
logs llama-swap`:

| Host | Lane | Engine | Load | Evicted, as predicted | Request |
|---|---|---|---|---|---|
| m5 | `qwen27b-vision-m5` | mlx-vlm 0.6.6 | 7 s | `qwen35b-m5` | answered |
| m5 | `glm53-m5` | ds4 | 14 s | `mimo9b-m5`, `qwen27b-vision-m5` | answered |
| m5 | `mimo9b-m5` | mlx-serve 26.9.5 | 3 s | `glm53-m5` | answered |
| mbp2 | `qwen4b-m1` | mtplx 2.11.2 | 26 s | nothing | answered |

Every operation succeeded and each lane's stop was the next swap's
eviction, confirmed against the matrix. m5 was put back on `mimo9b-m5` +
`qwen35b-m5` by a fourth swap; mbp2 had nothing loaded before, and
`qwen4b-m1` leaves by its own 10-minute ttl. With the six engines shown on
2026-11-17 that is nine of nine: vLLM, llama.cpp, Ollama, LM Studio and
Strata on the workstation; MTPLX, mlx-vlm, mlx-serve and ds4 on the Macs.

**Vendor apps on the Macs, enrolled observe-only** (`f2c6b6e`; backups
`inventory.json.bak-m4-vendors`): `lm-studio` on m5 and mbp2, `ollama` on
mbp2. Each is observed (`status`), answers `engine` where its server is
on, and is pinned: LM Studio 0.4.28+2 (m5) and 0.4.26+4 (mbp2), Ollama
0.40.16 (mbp2). They are `on_demand`, so no plan stops or unloads them.
LM Studio has no fixed launchd label (`application.<bundle id>.<n>.<n>`,
new numbers at every launch); the record names the stable part and the
adapter finds the running instance.

Limits, said plainly:

- A lane has no stop of its own in llmctl-next: it leaves by eviction or
  by its ttl. Its log is llama-swap's log (the lines that name it).
- `deepseek41-1017-mxfp4-m5` (the other ds4 lane, started through
  `serve.sh`) was not loaded; `logs ds4` is still unsupported.
- The three observe-only records refuse `stop` and `run` (by design) and
  have no log (`logs`: no plist to take the paths from). Starting and
  stopping them stays with their own controls; llmctl-next does not call
  those. On m5 the LM Studio app runs but its server is off, so it shows
  running and not ready.
- mbp2's Ollama job (`com.ollama.ollama`) reports no program path, so its
  served identity shows as unknown (`~`).
- m5 Ollama was still never stopped or started (hazard 1).
- The per-row disposable spikes of M4b open item 5 were not written; the
  rows above are live runs.

### Update 2026-11-18, 02:13 PST: installation and upgrade / recovery are met

**Vendor-app path, once: Ollama on m5, 0.34.3 → 0.40.16** (ends hazard 1).

1. *Backup.* brew had already removed the 0.34.3 keg, so there was no
   way back on disk. The official v0.34.3 release (`ollama-darwin.tgz`,
   sha256 `2c45865f…3d36e`, checked) now lies in
   `~/local-llm/runtimes/ollama-0.34.3`. It was run on a spare port with
   `OLLAMA_NOPRUNE=1` against the model store: version 0.34.3, all 4
   models listed, an embedding answered. The path is in the recipe
   (`backup_path`) and `install plan ollama` prints it.
2. *Reinstall.* The recorded command's first half, `brew reinstall
   ollama` (auto-update off, so it stayed on 0.40.16).
3. *Restart* through llmctl-next. **The first `restart ollama` failed**
   and left Ollama down for 13 s (02:05:17–02:05:30) until `run ollama`:
   see the fix below. On the fixed agent the same restart took 1 s.
4. *Drift, live on a real install:* `version ollama` exit 4, `version.fact`
   pinned 0.34.3, now 0.40.16; `install plan ollama` the same.
5. *Check:* 4 models, an embedding and a `llama3.2:3b` generation answered.
6. *Accept:* `version ollama --repin` listed the one change; drift 0.

**Bug found and fixed (`7a10a5c`).** A launchd stop read "still loaded"
right after `bootout`, which returns while the process is shutting down.
A restart then failed at its stop step and its rollback left the service
stopped. The stop now asks launchd until the job is gone. Real-launchd
test `a_job_that_exits_slowly_is_stopped_and_restarted`; suite 6 of 6 on
mbp2. Not checked: that the new test fails on the old code.

**A real release-archive install, upgrade and rollback** (workstation,
systemd). A pilot unit `llamacpp-release` ran the official llama.cpp CPU
release on port 8011 with `gemma-3-270m` (CPU only, `on_demand`, no GPU
and no neighbour touched):

| Step | Command | Result |
|---|---|---|
| install | `install stage llamacpp-release 12500`, `upgrade promote … 12500` | fetched from GitHub, sha256 verified against the recipe, unpacked to `~/AI/llama.cpp-releases/llamacpp-release-12500-0552b75a` with `staged.json` (url, archive digest, validation); unit restarted from it; served model and `--version` (build 12500) verified; a chat request answered |
| upgrade | the same for `12503` | `…-12503-480dd892`, build 12503 verified, request answered |
| refusal | `install stage … 12502` | "no sha256 is declared for llamacpp-release 12502"; nothing fetched |
| rollback | `upgrade rollback llamacpp-release` | back on the 12500 directory, verified, request answered |

It needed `b35d63a`: a recipe had one checksum for every version, which
no real release series can satisfy; `checksums` maps version to sha256.

The pilot unit was removed afterwards (record, unit file); the two
versioned directories, their archives, logs and manifests stay in
`~/AI/llama.cpp-releases` (217 MB). Backups: `inventory.json.bak-m4-archive`
(before), `…-pilot` (with the record).

Row by row:

- **Installation — met.** `python_env` (vLLM, M2), `source_build`
  (llama.cpp, M4a), `release_archive` (above), `vendor_app` (installs
  nothing by design; its owner-run plan was carried out on m5). `external`
  is a record with nothing to install.
- **Upgrade / recovery — met.** Staged upgrade and rollback for
  `python_env`, `source_build` and now `release_archive`; an interrupted
  stage is never promotable and is removed by gc (real-systemd tests);
  the vendor-app backup and reinstall path was run once.

Limits, said plainly:

- The release-archive run was on a pilot unit made for it, not on a
  production lane, and only on systemd. No launchd workload has a
  release-archive recipe (the Mac engines of that kind run as lanes,
  where a version swap is a macro path change).
- The 0.34.3 backup was proven to run and to read the model store; the
  service itself was not switched back to it.
- `release` refuses after a promote (the original unit file points at the
  first directory, the pin at the promoted one). The pilot record was
  removed by an inventory edit instead.
- The mlx-serve swap back to 26.10.1 on m5 was not run.

**To pass: nothing is left on the table.** All six rows are met as read
here; the limits in the four updates are the honest remainder.

### Closed 2026-11-18, 02:56 PST

The owner closed M4 with those limits accepted, then asked for the five
carried-over items that matter most in daily use. They were done the same
night (next section). What still carries over:

1. **Enrolling a workload that cannot be adopted** (a vendor app record,
   an observe-only record) and changing a record's authority are still
   inventory edits. `recipe set`, `forget` and `stores add` cover the rest.
2. **The matrix cannot be changed by a verb**, so the memory budget is a
   report today; its write guard waits for that.
3. **`deepseek41-1017-mxfp4-m5`** was never loaded through llmctl-next.
4. **Release archives on launchd**, and `release` after a promote.
5. **Profiles combined with the VRAM fit; per-lane in-flight counts.**
6. ~~**From M3:** the sleep/wake drill (blocked by Acronis' sleep lock).~~
   Done 2026-11-18: [drill report](RUST-M3-SLEEP-DRILL-2026-11-18.md).

### After the close: five carried-over items (2026-11-18, 03:26 PST)

| Item | What exists now | Commits | Live |
|---|---|---|---|
| No verb for recipes or records | `recipe show/set/clear` (set pins in the same step), `forget <workload> [--yes]` (preview first; refused for a running managed workload) | `2d603ae` | workstation: recipe of `llmster` re-applied by verb, `forget vllm-engine` refused |
| Lanes: no stop, no log | `lanes stop <workload> <lane>` (an admitted operation, held by requests in flight), `lanes logs <workload> <lane>` (llama-swap's per-lane buffer) | `78f1bcf` | m5: `qwen4b-m5` loaded then stopped, `mimo9b-m5` log read |
| Mac vendor apps observe-only | authority `vendor_command`: the app's own start and stop commands and its log; running = the endpoint answers | `7dc5403`, `f2c6b6e` | m5: LM Studio's server started, log read, stopped (it was off before). mbp2: Ollama.app stopped and started (down 9 s) |
| A failed restart leaves it stopped | the rollback of a restart starts the workload again and verifies it; if that fails too: `needs_attention` | `3102b7d` | not provoked on a real service; e2e with injected faults |
| M4c gaps | `stores add/remove`; `model rm` (preview, token, never while referenced); `lanes memory` (budget and per-lane peaks, summed over every matrix set) | `5770cf2`, `c246571`, `10b6584` | both Macs: four stores each; mbp2: a tiny model downloaded and removed; m5: weights two lanes run were refused; m5 budget 104 GiB from the Mac registry: 6 sets fit (largest 102G), 3 `risky*` sets over on purpose |

The healers follow: mac-heal and llm-heal ask llmctl-next about an app
without a launchd job by its workload name (`~/AI` `120f8548`,
`4eba5128`; 74 healer tests).

Found on the way: an agent started by launchd has a short `PATH` and did
not find `hf`, so every download size was "unknown" on the Macs
(`10b6584`). m5 has no `hf` at all: `model get hf:` does not work there
yet.

Checks at the end: workspace suites (e2e 40, lanes 4, core 31, planner
27, swap 11, daemon 84), real-systemd 22 of 22, real-launchd 7 of 7 on
mbp2, clippy at the 24-warning baseline, macOS type check.

Limits, said plainly:

- `model rm` deletes for good; the only guards are the reference scan and
  the token. The scan reads workload records, launch commands,
  environments, env files with the profiles beside them, and lanes. It
  does not know about benchmark scripts, Hermes configs or anything else
  that names a model path.
- A `directory` store downloads to `<store>/<name>`; the Macs keep
  `~/Models/huggingface/<org>/<name>`. `model rm path:<org>/<name>`
  works there; `model get` into that store would not match the layout.
- LM Studio's `stop` stops its server, not the app. Ollama's `stop`
  quits the app.
- Lane peaks are the registry's figures (measured or estimated), not
  measured by llmctl-next.

## Hazards found by the sweep (not M4 work, owner's call)

1. m5 Ollama runs 0.34.3 while brew installed 0.40.16 — the next restart
   (or a mac-heal kickstart) silently upgrades it.
   **Closed 2026-11-18:** upgraded on purpose and re-pinned at 0.40.16;
   a 0.34.3 binary is kept in `~/local-llm/runtimes/ollama-0.34.3`.
2. mbp2's disk is 93 % full (144 GiB free) — blocks any model download there.
3. Three different vLLM versions are named for `vllm-engine`; the env
   header points at a preset file (`qwen3.8-27b.yaml`) that does not exist.
4. `llmster` on the workstation has no supervisor; a logout or crash leaves
   it down with nothing to restart it.

## Owner decisions (2026-11-17, all as recommended)

1. **Split M4 into M4a/M4b/M4c**, a pilot each.
2. **Router:** llm-router stays the swap front door through M4 and calls
   llmctl-next for the workloads llmctl-next owns, under the shared flock.
3. **llama-swap config:** llmctl-next renders it from lane records after a
   round-trip import, and still commits each change to the config repo.
4. **System units:** `vllm-engine` and the main `ollama` move to user units
   (M4b step 3); no system-scope adapter, no new sudo path.
5. **ds4:** Mac-only for M4 (pin and manage it on m5).
6. **`llmster` on Linux:** runs under a user unit owned by llmctl-next.
7. **Model acquisition:** hf2local stays the converter; llmctl-next adopts
   its finished outputs as weights in a store.
8. **Ollama on the Macs:** keep brew + plist on m5 and the app on mbp2, as
   two vendor-app paths.
