# Rust successor — M2 plan (what is left)

Status 2026-11-16: **M2 done** (A–D implemented, pilot windows 1–3 passed; see the [pilot report](RUST-M2-PILOT-2026-11-16.md)). Parent:
[RUST-SUCCESSOR-PLAN.md](RUST-SUCCESSOR-PLAN.md) (milestone table, M2 row).
Code: `rust/` (`llmctl-next`), see [rust/README.md](../rust/README.md).

M2 is "one owned Linux workload". Its exit criterion, verbatim from the parent
plan: *disposable-unit tests then controlled real-model pilot pass
startup/stop failures, bounded recovery, daemon restart and upgrade rollback.*

## Done (2026-11-16, all on `main`)

| Piece | Commits | Proof |
|---|---|---|
| systemd adapter + per-workload engine routing | `0f4bad2`, `d15d9f0` | run → Ready → stop; early exit; wrong served model |
| Adoption: import via D-Bus, systemd round-trip equivalence, pinned digest, drift guard | `f7ac5c8`, `fd22a74` | adopt → stop → run → daemon restart; stale confirm; drift blocks run; inline secret refused |
| `restart` as its own plan kind | `4cd6f13` | new MainPID + re-verify; stopped unit refused |
| `logs` from the journal, follow by cursor | `d86d642`, `5786628` | output, run markers, follow across a restart, bad cursor → 400 |
| Pilot windows 1 and 2 on `vllm-engine2` + their fixes | `1a5d123` `43f673d` `8109ae8` `8544662` `193816f` `f3c923c` `c6e0f51` | [RUST-M2-PILOT-2026-11-16.md](RUST-M2-PILOT-2026-11-16.md) |
| A: foreign GPU occupancy guard | 2026-11-16 | nvidia-smi + `/proc/<pid>/cgroup`; foreign pid/unit → `gpu_occupied` (409) before anything is journaled; probe failure refuses; status shows per-card enrolled vs foreign; e2e: refuse-before-journal, enrolled-conflict allowed, stale enrolled unit refused |
| B: bounded recovery | 2026-11-16 | watcher (only background loop), opt-in `recovery` policy, SQLite budget surviving daemon restart, exhaustion → `needs_attention` marker blocking until acked; systemd retries first (`auto-restart` → report only; a start after start-limit does `reset-failed`); manual stop never recovered; e2e: recover-once, budget-exhaust + block + survive restart, manual-stop untouched; spike: SIGKILL → systemd burns retries → start-limit → one recovery succeeds |
| B: stop-timeout honesty | 2026-11-16 | `LLMCTL_NEXT_SYSTEMD_EFFECT_TIMEOUT_MS` configurable effect timeout; spike: TERM-trapped unit with `TimeoutStopSec=infinity` → `Unconfirmed`, op fails, process observed honestly |
| C: `own` / `release` + `apply_definition` primitive | 2026-11-16 | drop-ins refused (listed); rendered unit written atomically (never through a symlink — the link itself is replaced, repo copy untouched); `daemon-reload` + re-import prove or restore byte-for-byte; MainPID unchanged; owned-file digest is a drift short circuit; `release` restores the original byte-identical / relinks and sets ownership external; corrupted render (fault hook) restored + refused |
| D: version pin + staged upgrade with rollback | 2026-11-16 | `VersionRef` pinned at `definition_schema: 2` (interpreter, package versions, CUDA, `pip freeze` lock digest, non-system `Exec*` digests); legacy pins re-pinned on load; version drift guard (`version.*` fields); `upgrade stage` (uv env side by side, ≥10 GiB check, install log, manifest-last so an interrupted stage is never promotable) + offline validation; `upgrade promote` = admitted `upgrade` operation (stop → ApplyDefinition → start → readiness → identity → VerifyVersion) with automatic rollback to N; `upgrade rollback/list/gc`; fakeengine 1.0/2.0/2.0.1-broken wheel-index spike: stage → validate → promote (serves 2.0, history `promoted`) → rollback (serves 1.0) → broken never promoted → manifest-less stage never promotable → in-place env change = drift |

Evidence: 65 tests + 16 opt-in disposable-unit tests
(`cargo test -p llmctl-cli --test systemd_spike -- --ignored --test-threads=1`).

## Facts that shape the rest (checked 2026-11-16, read-only)

- **The main engine is out of M2 scope.** `vllm-engine.service` (:8003, TP=2,
  both cards) is a *system* unit; the adapter refuses system scope by design
  (parent plan: system-unit operations need an explicit authorized path, the
  daemon never runs as root). M2 stays in user scope.
- **The pilot candidate is `vllm-engine2.service`**: user unit, vLLM (a
  day-one engine with a strong `/v1/models` identity), one card, :8010,
  currently inactive. `ollama-gpu1` is weaker (Ollama lists every pulled
  model, not the loaded one); `gpt-oss-120b` needs both cards.
- **Every candidate unit already retries itself**: `Restart=on-failure`,
  `StartLimitBurst=3` (vllm-engine2: 15 s delay, 30 min window). Bounded
  recovery must not stack a second retry loop on top.
- **The vLLM version is invisible to today's drift guard.** `vllm-engine2`
  runs `~/AI/scripts/vllm-launcher.sh`, which execs
  `${VLLM_PYTHON:-~/miniconda3/envs/vllm-serve/bin/python}`. *Correction
  2026-11-16:* `vllm-engine2.env` sets `VLLM_PYTHON` to the `vllm-muse` env
  (vllm 0.30.0, torch 2.13.0), not the `vllm-serve` default (0.25.1), so
  the interpreter must be read from the env file, never assumed from the
  launcher. The real recipe therefore needs `apply: "env"`. The pinned
  digest covers argv and the `EnvironmentFile=` contents, but not the
  launcher script's contents or the Python env. An in-place
  `pip install -U vllm` or a launcher edit is not detected.
- **Other actors can stop or rewrite `vllm-engine2`:**
  - llm-router stops it before every switch (`switch.stop_first`);
  - Python llmctl places presets on it and generates its env file
    (`services/vllm_slots.py`);
  - `llmctl take-gpus` and lane scripts free its card.

  The router and llm-heal both honour the lane lease (`_read_lease` /
  `active_lease`); llm-heal fails open on a stale lease.
- **llmctl-next cannot see GPU users it has not enrolled.** Its conflict
  planner only knows enrolled workloads. A run onto a card held by
  `vllm-engine` or a stray process would be planned as conflict-free.

## Remaining work, in order

### A. Foreign GPU occupancy guard (prerequisite for any real pilot)

Before a plan starts a unit, ask the GPUs who is on them:

1. Probe `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv`
   plus the index → UUID map; map each pid to its unit through
   `/proc/<pid>/cgroup`.
2. Any process on a target card that is not the target itself or an enrolled
   workload already planned to stop → refuse with a new `gpu_occupied` (409,
   exit 4) listing pid, unit and memory. GPU probe failure → refuse
   (unknown is never "free").
3. Add the probe result to `status` (per card: enrolled vs foreign users).

Tests: CSV and cgroup parsing (unit tests); a fake-engine GPU probe with a
foreign occupant → run refused, nothing journaled. Size: ~2 h.

### B. Bounded recovery

Recovery decides; observation only reports. Policy:

1. **systemd retries first.** While a unit is `activating/auto-restart`
   llmctl only reports "systemd retrying (n/burst)". llmctl acts only when
   systemd has given up (`failed`, e.g. `start-limit-hit`) or the unit has
   `Restart=no`.
2. **llmctl's slow budget**, per workload, persisted in SQLite so a daemon
   restart does not reset it. Default: 2 recoveries per 60 min, then
   `needs_attention` (which already blocks mutations until acked).
3. **A recovery is an ordinary admitted operation** (`run` with key
   `recover-<workload>-<n>`), so it gets admission, the drift guard, the GPU
   guard (A), the journal and rollback for free.
4. **Never recover when:**
   - intent is a manual stop or maintenance;
   - the observation is `Unknown` (manager or host unreachable);
   - the definition drifted;
   - a foreign process holds the card;
   - an operation is active, or one is unacknowledged `needs_attention`;
   - the daemon started less than one observation interval ago.
5. **Opt-in per workload** (`recovery: {max_attempts, window_secs}` in the
   inventory). Off for adopted workloads until the owner turns it on.
6. **A watcher task** observes enrolled workloads every 15 s. It is the only
   new background loop.

Tests (also covers the stop-timeout row of the exit checklist):
- fake engine: unit dies → recovered once; keeps dying → budget exhausted →
  `needs_attention`, no further attempts; manual stop / `Unknown` / drift →
  never touched; budget survives a daemon restart.
- systemd spike: a unit that fails until a counter file allows it. systemd
  burns its 3 retries, llmctl recovers once after `start-limit-hit`, and
  the journal shows both loops in order.

Size: ~half a day.

### C. Write and own the service definition

Today adoption pins the definition and leaves the unit file as the
authority. Full ownership (parent plan) means llmctl holds the one service
definition. Add `llmctl-next own NAME` (managed workloads only):

1. **Preconditions:**
   - no drop-ins (`DropInPaths` empty; refuse and list them otherwise);
   - definition not drifted;
   - no active operation.
2. **Write**, under the admission lock:
   - record the current fragment (path, sha256, and the symlink target when
     it is a link; the P3 units are links into `~/AI/services/`);
   - replace the *link or file itself* with the rendered unit (atomic
     rename, header naming llmctl-next and the digest). Never write through
     a symlink into the repo copy.
3. **Prove**: `daemon-reload`, re-import, and the effective definition must
   equal the pinned one. Otherwise restore the recorded file/link, reload,
   and refuse. The unit is not restarted (equivalent definition), so a
   running engine keeps its MainPID.
4. **Track**: ownership gains `service_file: {path, digest, previous}`; the
   drift check compares the file digest first (cheap), then the effective
   definition.
5. **Primitive for D**: `apply_definition(workload, next)` writes N+1 with the
   same prove-or-restore steps and keeps N as the rollback target.
6. **`llmctl-next release NAME`** restores the recorded original
   byte-for-byte (or the link) and sets ownership back to external. This is
   the hand-back path to Python llmctl after the pilot.

Tests (spike):
- own a running unit: file replaced, MainPID unchanged, no drift;
- own a symlinked unit: link replaced, repo target untouched;
- an injected non-equivalent render is restored and refused;
- release restores the original byte-identical;
- a unit with a drop-in is refused.

Size: ~half a day.

### D. Version pin and staged upgrade with rollback

Version ownership is not auto-update (parent plan). For the vLLM recipe:

1. **Version reference** on the managed workload:
   - interpreter path and env prefix;
   - versions of `vllm`, `torch` and CUDA as reported by the env;
   - a lock digest (sorted `pip freeze` or `conda-meta` digest);
   - content digests of non-system `Exec*` paths (the launcher script).

   This extends `LaunchDefinition`/ownership. Add a `definition_schema`
   field and re-pin on load, so older pinned digests are not reported as
   false drift.
2. **Drift guard** covers it: an edited launcher or an in-place package
   change → `definition_drift`.
3. **Stage**: create a new env at a versioned path
   (`~/.local/share/llmctl-next/envs/vllm-<version>-<hash>`) with `uv`;
   check free disk first (~10 GB); keep the install log and provenance.
   Never touch the serving env. An interrupted stage is discarded, never
   promoted.
4. **Validate offline**: imports, reported version, and the served model's
   config loads under the new version.
5. **Promote** (one card means downtime, and the plan says so):
   1. stop;
   2. `apply_definition` N+1 (new interpreter) through C;
   3. start, readiness, served-identity check, one smoke completion.

   Any failure → automatic rollback to N (old definition, old env, start,
   verify). A failed rollback → `needs_attention`.
6. **History and gc**: record from, to, operation id and result. Old envs
   stay until an explicit `gc`. Shared weights and HF caches are never
   touched. Version-specific compile caches are left alone.

Tests: a tiny local package `fakeengine` 1.0 / 2.0 / 2.0-broken built into a
temp wheel index (serves `/v1/models` and reports its version):
- stage → validate → promote → explicit rollback;
- the broken version fails validation and is never promoted;
- an interrupted stage leaves nothing promotable;
- an in-place change → drift.

Size: ~1–2 days.

### E. Real-model pilot (two windows, owner present)

**Window 1 — lifecycle, after A–C. Done 2026-11-16 ([pilot 2026-11-16](RUST-M2-PILOT-2026-11-16.md)).** It runs as a `longjob` GPU job, whose
lease keeps the router and llm-heal from switching underneath it.

1. **Pre-checks:**
   - `vllm-engine` is off the pilot card;
   - (the Python `llmctl-reconcile.timer` is safe to leave on: it only
     flips session rows, never starts or stops units; verified in
     `SessionService.reconcile`);
   - no Python preset is placed on engine2 during the window.
2. Adopt `vllm-engine2` with its already-installed one-card model (preview,
   confirm), then `own` it.
3. run / logs / restart / stop with real load times.
4. **Drills:**
   - kill the main PID (systemd retries, then bounded recovery);
   - SIGKILL the daemon during `AwaitReadiness` (vLLM loads take minutes).
     It must reconcile honestly, and the engine must survive in its own
     cgroup;
   - recovery exhaustion → `needs_attention`;
   - start failure: the GPU guard refusing while a foreign process holds
     the card.
5. `release`, then confirm Python llmctl and the router drive it as before.
6. Write up as `docs/RUST-M2-PILOT-<date>.md`.

**Window 2 — upgrade, after D. Done 2026-11-16 (see the pilot report).** Stage vLLM 0.32.0 (decision 3) for the
pilot env, promote it, run a smoke check, force a rollback, then release.

## Exit checklist (maps to the parent criterion)

- [x] Startup failures: early exit, wrong model, GPU occupied — disposable (all three: early exit `unit_that_exits…`, wrong model, `gpu_occupied` e2e + the live card refused while a docker process held it); **pilot pass** ([pilot 2026-11-16](RUST-M2-PILOT-2026-11-16.md): early exit + GPU occupied; wrong model disposable only) **Pilot pass:** early exit (window 1), GPU occupied (window 1), wrong model (window 3).
- [x] Stop failures: timeout → `Unconfirmed` handled honestly — disposable — and **pilot pass** (window 3: real vLLM stop with a 100 ms effect timeout → `unconfirmed`, no fake success)
  (`a_stop_that_outlives_the_effect_timeout_is_unconfirmed`: TERM-trapped
  unit, `TimeoutStopSec=infinity`, effect timeout via
  `LLMCTL_NEXT_SYSTEMD_EFFECT_TIMEOUT_MS`)
- [x] Bounded recovery: budgeted, coordinated with `Restart=`, respects manual stop — disposable (e2e budget/survival + spike start-limit drill); **pilot pass** ([pilot 2026-11-16](RUST-M2-PILOT-2026-11-16.md))
- [x] Daemon restart: mid-operation SIGKILL reconciles, engine survives — disposable (done previously); **pilot pass** ([pilot 2026-11-16](RUST-M2-PILOT-2026-11-16.md))
- [x] Upgrade rollback: failed promotion rolls back to the pinned N — disposable (fakeengine drill: automatic rollback on failure verified in the first promote attempts, explicit `upgrade rollback` green); **pilot pass** (window 2: promote 0.30.0 → 0.32.0 verified on the running process, explicit rollback, failed promote rolled back automatically — [pilot](RUST-M2-PILOT-2026-11-16.md))

All rows pass on disposable units and in the real-model pilot. The hand-back path (`release` after an upgrade) is also proven on the real unit (window 3).

## Owner decisions (2026-11-16)

1. **Pilot model: decided (2026-11-16), the default.** `qwen27b-awq`
   (served name `qwen3.8-27b-awq`) on `vllm-engine2`, GPU 1, :8010. This is
   the preset already in `vllm-engine2.env` (placed 2026-10-22), so no
   Python re-placement is needed. GPU 1 is held by `vllm-engine` (TP=2,
   both cards), so the window needs `vllm-engine` stopped for its length.
   The time slot is set when the owner says go.
2. **Recovery default: decided, opt-in per workload.** It stays off for
   adopted workloads until enabled in the inventory.
3. **Window 2 upgrade target: decided (2026-11-16, owner delegated),
   vLLM 0.30.0 → 0.32.0.** The earlier rule assumed the wrong env (see the
   correction under Facts). `vllm-engine2` runs `vllm-muse`: vllm 0.30.0,
   torch 2.13.0, CUDA 13.0 runtime (driver 595.99, CUDA 13.2). On PyPI,
   0.31.0 and 0.32.0 pin the same `torch==2.13.0`; 0.33.0 (2026-11-02) moves
   to torch 2.14.0. 0.32.0 (2026-10-18) is therefore the newest release that
   changes only vLLM, so a failed promote points at vLLM and not at a torch
   or CUDA jump. No 0.30.x/0.31.x/0.32.x patch releases exist. Staged side
   by side with uv and never installed in place; `vllm-muse` is the
   rollback target N. Free disk on `/`: 221 GB (stage needs ~10 GB).
4. **After the pilot: decided, hand back.** `release` returns
   `vllm-engine2` to Python llmctl; llmctl-next takes over daily use at M5.
