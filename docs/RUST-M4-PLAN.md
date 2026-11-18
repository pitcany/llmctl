# Rust successor — M4 plan (required engines and coordination)

Status 2026-11-17: plan agreed — all eight owner decisions taken, each as
recommended (end of this file). **M4a done 2026-11-17 — see
[the pilot note](RUST-M4A-PILOT-2026-11-17.md). M4b steps 1–5 done:
[progress](RUST-M4B-PROGRESS-2026-11-17.md). M4c: all seven items have a
first version: [progress](RUST-M4C-PROGRESS.md). Exit check 2026-11-17:
1 row met, 4 partly, 1 not met. Version pinning closed 2026-11-18, so it
now stands at 2 met, 4 partly — see "Exit check" below.**
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
| Installation | **Partly** (2 of 5 kinds on a real engine) | `python_env`: vLLM 0.32.0 staged into a versioned env on `vllm-engine2` (M2 window 2). `source_build`: llama.cpp `a9d43ecc3` built, validated, manifest and build log kept (M4a). `release_archive`: real-systemd test `release_archive_stages_promotes_verifies_and_rolls_back` with a tiny archive | `release_archive` has installed no real engine (the mlx-serve dirs on m5 were already there) and has no launchd run. `vendor_app` installs nothing by design (it prints an owner-run plan) and that plan was never carried out. `external` is a record only |
| Version pinning | **Not met** on 2026-11-17 (2 of 10 enrolled workloads pinned). **Met since 2026-11-18**, see the update below | Pinned: `gpt-oss-120b` (binary digests), m5 `ollama` (vendor fact 0.34.3). Drift is reported: test `vendor_app_version_is_observed_pinned_and_drifts`; M4a saw m5 Ollama 0.34.3 against brew 0.40.16 | `version` says "no recipe; version pinning is off" for `vllm-engine`, `ollama`, `ollama-gpu1`, `llmster`, `strata-fast` and for `llama-swap` on both Macs. m5 `ds4` has a recipe with the fork SHA but `pinned: null`. The 15 lanes have no version records (an engine version is a macro path). m5 Ollama shows no drift today because its pin was set to the running 0.34.3 |
| Launch / readiness / stop / logs | **Partly** (6 of 9 engines on a host) | Workstation, through admitted operations: vLLM (`vllm-engine`), llama.cpp (`gpt-oss-120b`, last 21:15), Ollama (`ollama-gpu1`), LM Studio (`llmster`), Strata (`strata-fast`). m5: MTPLX lanes loaded and evicted by `lanes swap` (20:22). `logs` answers for all six workstation workloads and for `llama-swap` (both Macs) and `ollama` (m5) | mlx-serve: restarted by a config reload and health-checked, never loaded by `lanes swap`. mlx-vlm and ds4 lanes: never loaded through llmctl-next; `logs ds4` is `unsupported` (the record has no plist). mbp2's MTPLX lanes: records and a rewrite only. m5 Ollama: never stopped or started (a restart would upgrade it). LM Studio on both Macs and Ollama on mbp2 are not enrolled. Lanes have no logs of their own (llama-swap's log). The per-row disposable spikes of M4b open item 5 were not written |
| Upgrade / recovery | **Partly** | `python_env`: promote, rollback, and a promote killed mid-load that rolled back by itself (M2 window 2). `source_build`: stage → promote → rollback on `gpt-oss-120b` (M4a). `release_archive`: real-systemd test. Interrupted stage: never promotable and removed by gc (real-systemd tests). mlx-serve 26.10.1 → 26.9.5 on m5 as a lane macro change | `release_archive` upgrade and rollback on a real engine. The vendor-app backup/reinstall path "tested once": never run. The mlx-serve swap back was not run |
| Request safety | **Met** | Live 19:05: `restart vllm-engine` without `--force` refused with 12 real requests in flight after the 20 s wait; the engine's pid did not change. e2e `stops_wait_for_in_flight_requests_or_refuse_without_force`; lane swaps wait on llama-swap's in-flight count (e2e) | The "requests finish inside the wait, then the stop goes ahead" path is covered by tests only |
| Manual-stop protection | **Partly** | Agent restart: `ollama-gpu1` was stopped at 14:06 and still carries the mark after every daemon restart since (last 21:01). Swap rollback: e2e `a_manual_stop_is_reported_survives_eviction_and_is_not_restored`. Healers: llm-heal's check against the real agent answers correctly; 58 + 16 healer tests | No live rollback past a manual stop. No live healer drill where it matters: llm-heal does not repair a cleanly stopped lane anyway, so its 2-minute runs prove nothing; the case the plan names (mac-heal kickstarting a manually stopped `llama-swap` on mbp2) was not run |

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

**To pass, in order of weight** (item 1, the pins, is done):

1. **Mac lanes and vendor services** (row 3, 1–2 hours, evicts the
   always-on lanes on m5 for some minutes): load the mlx-vlm lane, one ds4
   lane and mlx-serve by `lanes swap`; load one lane on mbp2; enroll LM
   Studio on both Macs and Ollama on mbp2 observe-only.
2. **Manual-stop drills** (row 6, about 20 minutes): mac-heal on mbp2 with
   `llama-swap` manually stopped; one forced rollback on the workstation
   with a manually stopped neighbour in the eviction set.
3. **A real `release_archive` install and the vendor-app path once** (rows
   1 and 4, about 2 hours; owner's call): m5 Ollama 0.34.3 → 0.40.16 is the
   natural vendor-app case and ends hazard 1.

## Hazards found by the sweep (not M4 work, owner's call)

1. m5 Ollama runs 0.34.3 while brew installed 0.40.16 — the next restart
   (or a mac-heal kickstart) silently upgrades it.
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
