# Inference inventory and dispositions (P0)

Snapshot: 2026-11-04 04:03 PST, desktop `yannik-desktop`. Produced for P0 of
[the consolidation plan](CONSOLIDATION-PLAN-2026-11-04.md). This is an audit
artifact, not a database or a source of truth: the owners named below remain
authoritative. Nothing was moved, restarted, installed or committed.

Evidence labels: **obs** = observed live this snapshot (systemd state, passive
HTTP GET, `/proc/<pid>/exe`, `ss -ltnp` program names, SQLite `mode=ro`);
**src** = read from source/config files; **doc** = taken from comments or docs,
not verified; **unverified** = depends on content this audit may not read.

Restrictions that shape the gaps: environment launch files (`*.env*`,
including every router `profile_env`, `vllm-engine.env`, `vllm-engine2.env`,
`llama-server.env`, `litellm.env`, `~/.config/llm-router/env`) and
`~/.config/llmctl/settings.yaml` (mode 600) were **not read**. Process
command lines and environments were not inspected. Launch equivalence that
depends on those files is marked *unverified (env file not audited)*.
`llmctl router-profiles` (the existing drift check) reads profile env files, so
it was not run either; running it is the operator's step (see §6).

## 1. Serving instances and profiles (desktop)

State key: P = production, S = intentional standby (configured, stopped,
started on demand), X = experimental, H = historical, U = unknown.

### 1a. vLLM (`vllm-engine` :8003, `vllm-engine2` :8010)

Shared facts for all four rows (obs/src unless noted):

- Runtime: launcher `~/AI/scripts/vllm-launcher.sh` reads `VLLM_*` from the
  unit's `EnvironmentFile`; interpreter chosen by `VLLM_PYTHON` (default
  conda `vllm-serve`). Running `vllm-engine` exe (obs): `~/miniconda3/envs/vllm-muse/bin/python3.12`.
  Installed vLLM: `vllm-muse` 0.30.0, `vllm-serve` 0.25.1, `vllm-laguna` 0.25.1+cu129.
  Observed serving `/version` on :8003: **0.30.0** (obs). Presets set
  `python_root: vllm-muse` (src). router.yaml comment saying "vllm-laguna" is stale (doc).
- Units: `vllm-engine.service` **system** scope, alias `vllm-tp.service`,
  `Restart=on-failure`, `StartLimitBurst=3/1800s`, `Conflicts=llama-server.service`.
  `vllm-engine2.service` **user** scope, disabled, same launcher, `EnvironmentFile=vllm-engine2.env`.
  Unit file copies in `~/AI/services/` are byte-identical to the installed copies (obs).
- Readiness: `GET /v1/models` (router `health_path`); llmctl load from `/metrics`.
- Logs: journald (`journalctl -u vllm-engine`, `journalctl --user -u vllm-engine2`).
- Recovery owners: systemd `Restart=on-failure`; llm-heal (2-min timer);
  router rollback on failed switch; manual `recover-vllm.sh` / `restore-vllm`.
- Authoritative editable source: llmctl preset YAML in **legacy**
  `~/.config/llm-models/<alias>.yaml` (canonical `~/.config/llmctl/presets/`
  exists but is **empty**, so `migrate_legacy_presets()` is a no-op and legacy
  is the only effective source; obs/src).
- Derived files: router `profile_env` copy per backend (`~/AI/services/vllm-engine.env.<profile>`);
  active `vllm-engine.env` / `vllm-engine2.env` (written by llmctl
  `vllm_orchestrator` render, by the router's profile copy on switch, and by
  `scripts/vllm-ab.sh` edits); `vllm-engine.env.previous` (router rollback copy).
- Precedence (src, `llmctl/presets/store.py`): canonical dir loaded first, legacy
  dir overrides; `save_preset` writes legacy in place when only legacy holds the alias.
- Deployment source: router.yaml in live tree `~/AI/llm-router` (HEAD `4c66a7b`);
  presets are unversioned user config.

| ID / router name | Served name, aliases | GPUs / TP | Endpoint | Launch entry | State | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen27b` (preset `qwen27b`, profile `vllm-engine.env.qwen38`) | `qwen3.8-27b`; alias `Qwen/Qwen3.8-27B-FP8`; LiteLLM `local*`, `qwen27b`, `qwen27b-nothink` | both, TP=2 | :8003 | router switch (`restart_cmd sudo -n systemctl restart vllm-engine`) or `llmctl run qwen27b` | **P** — loaded since 2026-11-04 03:36 (obs); router `default_backend` | Keep preset; **generate** profile env from preset (blocker: env equivalence unverified) |
| `qwen27b-uncensored` (profile `qwen38-unc`) | `qwen3.8-27b-uncensored`; alias `orcarouter/Qwen3.8-27B-Uncensored-FP8` | both, TP=2 | :8003 | same | S | Same as above |
| `qwen27b-awq` (profile `qwen27b-awq`) | `qwen3.8-27b-awq` | one card, TP=1; placed by `llmctl run` on engine 1 or 2 | :8003 / :8010 | `llmctl run qwen27b-awq` (placement); router only if no engine serves it (then stops every lane) | S, X (quality unmeasured, doc) | Same; engine-2 placement keeps llmctl as owner |
| `qwen27b-unc-awq` (profile `qwen27b-unc-awq`) | `qwen3.8-27b-unc-awq`; alias `twolven/Qwen3.8-27B-abliterated-AWQ-MTP` | one card | :8003 / :8010 | same | S, X | Same |
| profile `vllm-engine.env.qwen38-pp` | unknown (not opened) | unknown | — | no router backend references it (src) | H/U | **Retire** (archive) after confirming no script uses it |
| `vllm-engine2` slot | whatever llmctl last placed; `vllm-engine2.env` last written 2026-10-22, **untracked in git** (obs) | one card | :8010 | llmctl `vllm_slots.place` only | S (inactive, obs) | Keep (llmctl-owned); content unverified |

### 1b. Command-backed lanes (router.yaml `start_cmd`)

All are **user** units, never enabled, `Restart=no` except gpt-oss
(`on-failure`). Each unit's `ExecStartPost` runs `~/bin/ds4-llmctl-sync adopt`,
which registers an ADOPTED llmctl session (track-only; `adopt` does not take
the GPU operation lock, src). Start scripts live in `~/AI/services/lanes/`
(symlinked from `~/bin`) and own the GPU hand-over (`--take-gpus`: stop
`vllm-engine`, evict Ollama, stop conflicting lanes). Units also carry
`Conflicts=` lists and `ExecStartPre` free-VRAM floors. Readiness: router
`/v1/models` probe with per-lane `health_timeout_s`. Logs: journald user unit.
Recovery: llm-heal (lane units from router.yaml), else manual.
llmctl surfaces them from router.yaml via `llmctl-stack/router_lanes.py`
(status `source=lane`), except `gptoss` and `qwen35b`, which come from
`settings.yaml` managed units (status `source=settings`, obs).

| Router name | Served name | Engine / runtime path (src) | GPUs | Port | Unit | Launch config | State | Disposition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flash` | `qwen3.8-flash-next-strata` | Strata, `~/AI/strata/.venv` | 0,1 | 8007 | `qwen3.8-flash-next-strata` | `~/AI/strata/strata-lane.json` → **`strata-unsloth-ud-q4_k_xl.2gpu.res.json`** (obs); router.yaml comment says `.2gpu.json` | S | Keep; fix the comment or the symlink (decision for owner) |
| `flash-fast` | `qwen3.8-flash-next-fast-strata` | Strata, `~/AI/strata/.venv` | 0,1 | 8009 | `qwen3.8-flash-next-fast-strata` | `~/AI/services/strata-fast-lane.json` → `strata-atomic-ad427.json` | S | Keep |
| `flash-uncensored` | `qwen3.8-flash-next-uncensored-strata` | Strata, `~/AI/strata/.venv` | 0,1 | 8008 | `qwen3.8-flash-next-uncensored-strata` | `~/AI/services/strata-uncensored-lane.json` → `strata-navin-ad427.json` | S | Keep |
| `dsv41` | `deepseek-v4.1-flash` | Strata 0.2.8, `~/AI/strata-0.2.8/.venv` | 0,1 + ~100 GB RAM | 8018 | `deepseek-v4.1-flash-strata` (+ `deepseek-v41-readiness.timer` enabled) | `~/AI/services/dsv41-lane.json` → `strata-0.2.8/…-2.00bpw.json` | S | Keep |
| (`dsv41-q`) | — | Strata 0.2.8 | 0,1 | — | `deepseek-v4.1-flash-dsv41-q-strata.service` (repo file, not installed) | `dsv41-q-lane.json` | H (cold storage, doc) | Keep as archived lane per lanes/README |
| `glm53` | `glm-5.3-flash` | Maya, `~/AI/maya-src/.venv` | 0,1 | 8006 | `glm-5.3-flash` | `~/AI/maya-src/maya-lane.json` | S | Keep; retire stale `~/.config/llm-models/profiles/glm-5.3-flash-cpumoe.yaml` (llama.cpp era) |
| `gptoss` | `gpt-oss-120b` | llama.cpp via `~/bin/llama-server-guarded` | 0,1 | 8005 | `gpt-oss-120b` (Restart=on-failure) | flags in unit ExecStart; **also** SQL profiles `gpt-oss-120b-cpumoe`, `gpt-oss-120b-swap` and `~/AI/services/gpt-oss-120b-cpumoe.profile.yaml` | S | Keep unit as the launch definition; **reference/retire** the duplicate profiles after checking `default_profile_id` links |
| `qwen3coder30b` | `qwen3-coder:30b` | Ollama (`/usr/local/bin/ollama` 0.33.1, bwrap hides GPU0) | 1 | 11435 | `ollama-gpu1` | unit env + `coder1-start` (symlinks `qwen3coder1-start`, `gemma1-start`) | S | Keep; fix stale settings row `qwen35b → ollama-gpu1` |

### 1c. Other desktop serving paths

| Instance | Owner / scope | Endpoint | Observed | State | Disposition |
| --- | --- | --- | --- | --- | --- |
| Ollama `ollama.service` | system unit (+ `ollama.service.d` drop-in); repo copy `~/AI/services/ollama.service` **differs** from installed (obs) | :11434 (all interfaces) | v0.33.1; loaded `llama3.2-3b-cpu` (0 VRAM); 18 tags incl. embeddings and `:cloud` proxies (obs) | P (LiteLLM `embed*`, `tiny`, `ocloud-*`, `qwen3.6-35b-cpu` call it **directly**, not via the router) | Keep; reconcile repo vs installed unit |
| LM Studio daemon `llmster` | **no unit**; runs in login `session-11.scope` (obs) | 127.0.0.1:1234 | reachable; lists `qwen3-coder`, `llama3.2`, `qwen3.6`… (obs) | U | Decide: keep as unmanaged, or bring under a unit; not a retirement without consumer check |
| `llama-server.service` + `scripts/llama.sh`, `llama-server-launcher.sh`, `llama-server.env` | system unit, disabled, inactive (obs); `vllm-engine` declares `Conflicts=` with it | unknown (env) | — | H/U | **Retire** candidate; blocker: consumers of `llama.sh` / WebUI preset builder unknown; system unit change needs approval |
| `sglang-qwen38-nvfp4.service` + `sglang-qwen38-*` scripts | repo file only, **not installed** (obs); base env file absent, `.env.local` present | :30000 (unit) | registry has 1 SGLANG model | H | **Retire** (archive) |
| `llmctl-gateway.service` (`llmctl gateway`, 127.0.0.1:9000) | user unit, disabled | :9000 | inactive (obs) | H/U | Retire candidate (second front door); blocker: confirm nothing points at :9000 |
| `llmctl-api.service` | user unit, `~/AI/.venv/bin/llmctl serve` | :8088 | active, llmctl 0.3.0+84ed67f2 | P | Keep |

### 1d. Front door and routing

| Component | Owner | Endpoint | Version / state (obs) | Notes |
| --- | --- | --- | --- | --- |
| LiteLLM | user unit `litellm`, `~/AI/llm-router/.venv-litellm` | 0.0.0.0:4000 | litellm 1.100.1, active | Lane routes **generated** from router.yaml by `litellm/gen-local-lanes.py` (marked block in `config.yaml`, applied by `apply.sh`). 24 routes → router :4100; 12 → Ollama :11434 directly; Mac routes via forwarder ports (below). No route targets :8003/:8010 directly (obs). |
| llm-router | user unit `llm-router`, live tree `~/AI/llm-router` @ `4c66a7b` (+ untracked router.yaml backups) | 127.0.0.1:4100 | active 57 h; `qwen27b` loaded; 2298 requests (7 err); rejected unknown_model 3, no_live_backend 39 | Single lane registry (`config/router.yaml`), read by llmctl-stack, llm-heal and gen-local-lanes. Staged coordination changes: see §5. |
| llm-heal | user timer `llm-heal.timer` (2 min) → `~/AI/heal/llm-heal.py`, config `heal/heal.yaml` | — | timer active | Reads router.yaml lanes plus its own `extra_lanes` (ollama-gpu1, ollama), `front_doors`, **`mac_repair`** (repairs Mac launchd services over SSH). |
| llmctl reconcile | `llmctl-reconcile.timer` → `llmctl reconcile` | — | active (re-enabled since the Nov 3 snapshot) | Updates session tracking. |

## 2. Mac routes (known; not inspected beyond passive GETs)

Desktop side: `llm-mesh-forwarder.service` (`~/AI/mesh/llm-mesh-forwarder.py`,
config `~/.config/llm-gateway/forwarder.json`) listens on loopback ports and
picks the first healthy upstream (LAN first, Tailscale second). Mac-side
lifecycle is launchd/apps on each Mac (src: heal.yaml `mac_repair`); repair is
also attempted from the desktop healer via SSH.

| Route | Listen | Upstreams (forwarder.json) | Selected (state 04:00) | GET `/v1/models` | LiteLLM names | Mac owner (src/doc) |
| --- | --- | --- | --- | --- | --- | --- |
| m5-swap | :18099 | lan **10.0.0.110**:8099, tailscale | tailscale | 200 | `fast`, `vision`, `code`, `*-m5`, `top`, `flash-m5`… | launchd `com.pitcany.llama-swap` |
| m5-ollama | :18434 | lan 10.0.0.110:11434, tailscale | tailscale | 200 | `qwen3embed*-m5`, `nomicembed-m5` | launchd `com.pitcany.ollama` |
| m5-fleet | :18100 | lan 10.0.0.110:8100, tailscale | tailscale | 401 (auth expected) | `fleet-*` | fleetd (unknown owner) |
| m5-lms | :11234 | lan 10.0.0.110:1234, tailscale | none | not probed | none found | LM Studio |
| mbp2-swap | :28099 | lan 10.0.0.100:8099, tailscale | lan | 200 | `qwen35b-m1` | launchd llama-swap |
| mbp2-ollama | :28434 | lan 10.0.0.100:11434, tailscale | lan | not probed | none found | Ollama.app |
| mbp2-lms | :21234 | lan 10.0.0.100:1234, tailscale | none | **503** | `glm46v-m1`, `nomicembed-m1` | `llmster` |

Unresolved: forwarder uses **10.0.0.110** for M5 while the operator notes list
M5 as **10.0.0.144**; M5 routes are currently on Tailscale. The `-m1` suffix
maps to mbp2 ports. Both are owner questions, not P0 actions.

## 3. Supporting artifacts

### 3a. Preset and profile stores

| Artifact | Contents (obs) | Readers | Writers | Disposition |
| --- | --- | --- | --- | --- |
| `~/.config/llm-models/*.yaml` (legacy, effective) | 4 presets: `qwen27b`, `qwen27b-uncensored`, `qwen27b-awq`, `qwen27b-unc-awq` (all `shortcuts: []`, `schema_version: 1`); `_tui.yaml` (TUI state) | llmctl (`load_all`), llm-models-config, `llmctl-stack/router_profiles.py` via llmctl, `scripts/run-quant-ab.sh` (path), `packages/ask/.../discover.py` (path) | llmctl `save_preset`, llm-models-config store | Keep data; migrate to canonical in P2.4 with manifest |
| `~/.config/llmctl/presets/` (canonical) | **empty** directory (created 2026-10-21) | llmctl | llmctl | Target store |
| `~/.config/llm-models/_shortcuts.yaml` | **absent** | llm-models-config | llm-models-config | No shortcut overlay to carry over |
| `~/.config/llm-models/legacy/vserve/` | 9 old presets (not loaded: nested) | none found | none | Archive (H) |
| `~/.config/llm-models/{_pruned-*, _archived-offnvme, _backups, _bak-*}` | retired presets and backups | none (prefix/nesting excludes them) | manual | Keep as history |
| `~/.config/llm-models/profiles/` | `deepseek-v4-flash-cpumoe.yaml`, `glm-5.3-flash-cpumoe.yaml` (profile exports) | unknown | `llmctl profile export` (inferred) | H — both workloads changed engines; retire after P2.5 review |
| SQLite `profiles` | 3: `tp2-awq-70b` (70B lanes removed, H), `gpt-oss-120b-cpumoe`, `gpt-oss-120b-swap` | llmctl | llmctl | Keep table; review the 3 rows (P2.5) |
| SQLite `models` | 142 rows: 68 DELETED, 32 MISSING, 41 DISCOVERED, 1 REGISTERED; hosts: desktop 62, two Mac hostnames 44, NULL 36 | llmctl | discovery/import | Keep (history); host NULL rows need attribution before any cross-host linking |
| SQLite `sessions` | 2: RUNNING ADOPTED `qwen3.8-27b` on :8003 recorded with unit **`vllm-tp.service`** (alias); STOPPED OWNED :8000 | llmctl | llmctl, `ds4-llmctl-sync`, reconcile | Keep; alias must stay resolvable |
| `~/.config/llmctl/settings.yaml` | not read (mode 600). `llmctl status` shows managed-unit rows `vllm-engine`, `vllm-engine2`, **`gptoss`** (also a router lane), **`qwen35b` → `ollama-gpu1`** (lane retired 2026-11-01; router name is `qwen3coder30b`) | llmctl | operator | Reference router.yaml for lanes; drop stale `qwen35b` row (owner edit) |

### 3b. Router `profile_env` files (`~/AI/services/`, names only)

Used: `vllm-engine.env.qwen38`, `.qwen38-unc`, `.qwen27b-awq`, `.qwen27b-unc-awq`.
Unused by router.yaml: `.qwen38-pp`. Active/derived: `vllm-engine.env`,
`vllm-engine.env.previous`, `vllm-engine2.env` (untracked). Backups:
7 × `vllm-engine.env.bak-ab-*` (2026-11-01/02, from `vllm-ab.sh`). Legacy name:
`vllm-tp.env` + 6 backups (still referenced by `scripts/apply-spec-rollout.sh`,
`scripts/pp-vs-tp-ab.sh`, `scripts/git-hooks/check-live-config.sh`,
`agent-bench/runs/e5e6.sh` and tests). Content of all: **unverified (env file not audited)**.

### 3c. Units (installed vs repo copies)

| Unit | Scope | Installed form | Repo copy | Enabled / active (obs) |
| --- | --- | --- | --- | --- |
| vllm-engine (+alias vllm-tp) | system | file | `~/AI/services/vllm-engine.service`, identical | enabled / active |
| ollama | system | file + drop-in | repo copy **differs** | enabled / active |
| llama-server | system | file | repo copy **differs** | disabled / inactive |
| llm-cpu-perf, llm-thp | system | file | not compared | enabled / exited (host tuning) |
| vllm-engine2, ollama-gpu1, 4 Strata lanes, glm-5.3-flash, gpt-oss-120b | user | **regular-file copies** | identical copies in `~/AI/services/` | disabled / inactive |
| harbor-tools-healthcheck | user | symlink → repo | — | enabled |
| llm-heal | user | symlink → `~/AI/heal/systemd/` | — | timer active |
| llm-mesh-forwarder | user | file | `~/AI/mesh/systemd/`, identical | active |
| llm-router, litellm, llmctl-api, llmctl-gateway | user | file | no repo copy found | active / active / active / disabled |
| deepseek-v41-readiness (.timer) | user | file | not compared | timer enabled |

### 3d. Shims and lifecycle scripts

- `~/AI/bin/`: `llmctl` (→ `~/AI/.venv/bin/llmctl`), `llmc` (retiring), `modellab`,
  `llm-backend-guard` (a separate "one backend owns the GPUs" guard for
  claude-local), `rag`, `llm-usage`. `~/AI/.venv/bin` also exposes `llm-models`
  (no shim).
- `~/bin` → `~/AI/services/lanes/*` symlinks: `strata-{start,stop,status,logs,gpus,fast,uncensored}`,
  `glm53-{start,stop,status}`, `gptoss-start`, `coder1-start` (+`gemma1-start`,
  `qwen3coder1-start`), `dsv41`, `dsv41-q`. `~/bin/llama-server-guarded` is a
  **regular-file copy** identical to `~/AI/scripts/llama-server-guarded`.
- vLLM lifecycle helpers: `services/lanes/restore-vllm`, `services/lanes/vllm-on-gpus`,
  `scripts/recover-vllm.sh`, `scripts/vllm-ab.sh`, `scripts/vllm-ab-queue.sh`,
  `scripts/run-quant-ab.sh` (uses `llmctl swap`).
- `~/AI/packages/llmctl/`: empty leftover directory (only `.claude/`), since
  llmctl moved to its own repo (`41229f6a`).

## 4. Retirement check: `llmc` (pitcany-llmc 0.1.0) and `llm-models-config` 0.1.0

| Consumer type | Finding (2026-11-04) |
| --- | --- |
| Python imports outside the two packages | none in `~/AI` (excluding the packages), llmctl, llmctl-stack, router, heal (rg) |
| Package dependencies | `pitcany-llmc` depends on `llm-models-config` (workspace); `llm-models-config` pins `pitcany-llmctl @ 84ed67f` (second pin site, uncommitted bump) |
| llmctl tests | `tests/test_preset_schema_parity.py` (`importorskip`) |
| Command shims | `~/AI/bin/llmc`; `llm-models` only in `~/AI/.venv/bin` |
| Shell/script callers of the commands | none (rg over scripts, services, heal, router, plugin, bin) |
| Scheduled jobs | none in crontab, user units or timers |
| CI / release | `.github/workflows/release.yml` (tags `llmc-v*`, `llm-models-config-v*`) and `wheels.yml` (build + smoke `llmc --help`, `llm-models --help`); both also reference a **`gpu-models`** package that no longer exists |
| Data consumers of the legacy directory (not the package) | `scripts/run-quant-ab.sh`, `packages/ask/src/ask/discover.py`, llmctl itself |
| Lockfile | `uv.lock` entries `pitcany-llmc` and `llm-models-config` (editable) |
| Mac usage | unknown (not inspected) |

Unique functionality that would be retired (record honestly in P2.6): llmc's
LLM-Compressor planner, emitters and benchmark harness; llm-models-config's
`list/show/migrate` CLI, standalone renderer and shortcut overlay (unused here).

## 5. Package pins and staged changes (separate from benchmark work)

- Installed llmctl: 0.3.0+84ed67f2 in `~/AI/.venv` (API pid 3727568).
- `~/AI` @ `14c900bb`, uncommitted control-plane edits: `packages/llmctl-stack/pyproject.toml`,
  `packages/llm-models-config/pyproject.toml`, `uv.lock` (llmctl pin → `84ed67f`).
  Also untracked: `services/vllm-engine2.env`.
- Staged, uncommitted, not deployed: router branch `feat/shared-op-lock-2026-11-03`
  (worktree under the session scratchpad, base `4c66a7b`); healer branch
  `feat/heal-op-lock-2026-11-03` (base `950c9135`). Patch copies:
  `~/AI/logs/staged-2026-11-03-control-plane/`.
- Unrelated working-tree changes in `~/AI` (not part of this work):
  `benchmarks/claude-quota/statusline.jsonl`, `benchmarks/runs/20261030-dsv41-optimize/*`,
  untracked `benchmarks/runs/ab-admission-20261101/*`, `benchmarks/runs/ab-cell-20261102/*`;
  untracked router.yaml backups in `~/AI/llm-router/config/`.

## 6. Baseline counts (for reduction targets later; no targets set here)

| Measure | Count | Detail |
| --- | --- | --- |
| Conflicting preset aliases | **0** | canonical dir empty; 4 legacy aliases equal their router backend names |
| Independently editable launch definitions for live (P/S) workloads | **13** | 4 presets + 4 router profile envs (hand-synced copies of the same workloads); gptoss: unit + 2 SQL profiles + 1 profile YAML export = 4; Strata/Maya/Ollama lanes have 1 each |
| Duplicated unit-file copies | **10** | 8 user units + `vllm-engine` + `llm-mesh-forwarder` installed as copies of repo files (identical today); `ollama.service` copy already drifted |
| Duplicate GPU hand-over implementations | **11** | router `stop_first` + `evict_ollama`; llmctl `lane_swap`/run; 7 lane start scripts (`strata-start`, `strata-fast`, `strata-uncensored`, `dsv41`, `glm53-start`, `gptoss-start`, `coder1-start`); `restore-vllm`; `llm-backend-guard`. Unit `Conflicts=` lists are a 12th, declarative layer |
| `vllm-engine` restart paths | **7** + systemd | router switch, llmctl (run/vllm/restart), llm-heal, `restore-vllm`, `recover-vllm.sh`, `vllm-ab.sh`, `vllm-ab-queue.sh`; only llmctl takes the operation lock today (router and heal: staged); `vllm-ab.sh` uses the lane lease and pauses heal |
| Unmanaged relevant processes/units | **3** | LM Studio `llmster` (no unit); `llama-server.service` (disabled, outside llmctl and router); `sglang-qwen38-nvfp4.service` (file only). Ollama :11434 models are unmanaged by design |
| Configs whose effective source cannot be explained | **3 unverified + 4 contradictions** | Unverified: active `vllm-engine.env` vs `profile qwen38` (A/B edits since 11-01); `vllm-engine2.env` content; profile envs vs presets. Contradictions: `strata-lane.json` → `.res` variant vs router.yaml comment; settings row `qwen35b`; M5 address `.110` vs `.144`; router.yaml "vllm-laguna" vs observed `vllm-muse` |

Operator step to close the first unverified item without exposing contents:
run `llmctl router-profiles` (no `--write`) yourself; exit 0 = in sync.

## 7. Top dispositions

| # | Candidate | Disposition | Replacement | Blockers | Live approval |
| --- | --- | --- | --- | --- | --- |
| 1 | Router profile envs `vllm-engine.env.{qwen38,qwen38-unc,qwen27b-awq,qwen27b-unc-awq}` | **Generate** from presets (existing `router-profiles --write`); retire `qwen38-pp` | llmctl preset + render | env equivalence unverified; router copies profile → active env on switch, so a generated profile changes the next switch | yes (affects next swap) |
| 2 | `llmc` + `llm-models-config` | **Retire** (P2) | llmctl presets/CLI | CI workflows, `uv.lock`/workspace, parity test, `~/AI/bin/llmc`, second llmctl pin; Mac usage unknown | yes for production uninstall only |
| 3 | gptoss duplicate profiles (2 SQL + 1 YAML export) | **Reference** the unit/lane; retire profiles | router lane `gptoss` + unit | registry `default_profile_id` links; which path `llmctl run gpt-oss-120b-mxfp4` uses | no (config only) |
| 4 | Historical paths: `llama-server.service`/`llama.sh`, sglang unit + scripts, `vllm-tp.env*`, `tp2-awq-70b` profile, `profiles/*-cpumoe.yaml`, `legacy/vserve`, `llmctl-gateway.service` | **Retire** (archive, reversible) | none needed / existing lanes | callers of `vllm-tp.env` in 4 scripts; `llama.sh` and :9000 consumers unknown; system unit needs sudo | yes for system units |
| 5 | Unit-file copies (repo vs installed) | **Generate** (install from repo, e.g. symlink like `llm-heal`) | repo unit as single source | `ollama.service` drift + drop-in; system scope needs sudo | yes |

## 8. Findings that affect P1 / P2 scope

P1:
- Idle revert of a TP=2 default also stops `ollama-gpu1` and `vllm-engine2`
  (`stop_first`) and **evicts Ollama :11434**, which LiteLLM calls directly for
  embeddings. Footprint (P1.3) must include these; :11434 has no router-side
  activity signal.
- Direct engine traffic exists by design (router.yaml: Hermes on :8003);
  lane start scripts, `restore-vllm`, `recover-vllm.sh` and `vllm-ab.sh` act
  without the operation lock. Document them as external boundaries (P1.5) or
  wrap them in P3.
- llmctl records the engine as `vllm-tp.service`; any unit matching added in
  P1 must treat aliases as one unit.

P2:
- No shortcut overlay exists; P2.3 needs no shortcut support today.
- llmctl itself writes the legacy directory (`save_preset`), so retiring
  llm-models-config does not end legacy writes; P2.4 must change llmctl too.
- CI (`release.yml`, `wheels.yml`) must drop `llmc`, `llm-models-config` and the
  already-missing `gpu-models`. Remove the empty `~/AI/packages/llmctl/`.
- Path consumers `run-quant-ab.sh` and `ask/discover.py` hard-code
  `~/.config/llm-models`; update them with the migration.
