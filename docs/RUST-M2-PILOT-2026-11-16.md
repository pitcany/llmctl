# Rust successor — M2 pilot, windows 1 and 2 (2026-11-16)

Plan: [RUST-M2-PLAN.md](RUST-M2-PLAN.md) § E, window 1 (lifecycle).
Owner present; started on the owner's "start window 1".

## Setup

| | |
|---|---|
| Workload | `vllm-engine2.service` (user unit) → `engine2`, GPU 1, :8010 |
| Model | `qwen3.8-27b-awq` (router profile `qwen27b-awq`, already placed in `vllm-engine2.env`) |
| Engine env | `vllm-muse`: vLLM 0.30.0, torch 2.13.0 (CUDA 13.0 runtime) |
| Window | `longjob` job `m2-pilot-w1 --stop-engine`, `vllm-engine` stopped 11:54–12:18 PST (~24 min; serving again 1 min 35 s after the window closed); the lane lease kept the router and llm-heal off the cards |
| Binary | `llmctl-next` release build at `1a5d123` (window), `8109ae8` (own/release after the window) |
| Data dir | `~/.local/state/llmctl-next-pilot` (fresh inventory: host GPUs 0 and 1, no workloads) |
| Evidence | `~/.local/state/llmctl-next-pilot/evidence/` (numbered files below, plus `daemon.log`) |
| Inventory knobs | `readiness_timeout_secs: 600`, `recovery: {max_attempts: 2, window_secs: 3600, min_interval_secs: 30}` |

## Results

| # | Step | Result | Evidence |
|---|---|---|---|
| 1 | Adopt (preview, confirm, `--allow-unverified` because the unit was stopped) | pass, but exposed finding F1 | `02`, `03` |
| 2 | `run engine2 --wait` | Ready + identity verified in **181 s**; smoke completion `"pilot ok"` | `05`–`07` |
| 3 | `logs` | run marker + vLLM output through the journal | `08` |
| 4 | `restart --wait` | new MainPID (3703079 → 3912963), re-verified, 147 s | `09` |
| 5 | Kill drill: SIGKILL MainPID, then every systemd retry | systemd retried twice (15 s apart), then `failed` (start-limit); llmctl showed `gave_up`, admitted `recover-engine2-1` **13 s** later; Ready + verified 216 s after that; no second attempt | `10` |
| 6 | Exhaustion: same again, plus kill llmctl's recovery during load | systemd gave up; recovery 2 of 2 failed ("process exited before readiness — different from a timeout") and was rolled back (stop); then one `needs_attention` marker (`recover-stalled-…`, "2 recoveries in the last 3600s (budget 2)"), no third attempt; a fresh run was refused `blocked_needs_attention` (409, exit 4); after `ack` the watcher raised no new marker | `11`–`14` |
| 7 | GPU guard: foreign torch process holding 1 GiB on GPU 1 | `run` refused `gpu_occupied` naming the pid and 1522 MiB, exit 4, **nothing journaled** (ops count 5 → 5) | `15` |
| 8 | SIGKILL the daemon during `AwaitReadiness` | engine kept loading in its own cgroup (MainPID 475214 unchanged); restarted daemon reconciled the op as `Interrupted` with the observed state ("external effects complete; only observation steps remained"), never a fake success; engine reached Ready + verified at the same MainPID | `16`, `17` |
| 9 | Manual stop | stopped, GPU 1 freed (18 MiB), intent `stopped/manual`; no recovery in the following 20 s | `18` |
| 10 | `own` with the window-1 pin | **refused** (F1 safety net): lists the six directives the pin did not carry; link untouched | `19` |
| 11 | Re-adopt (fixed importer), `own`, `release` (after the window, unit stopped) | `own` replaced the symlink with the rendered file (all settings + `[Install]`), repo copy untouched (sha256 `9ffe2340…` before and after), systemd reports the same RestartUSec 15 s, TimeoutStartUSec 15 min, StartLimit 3/30 min; `release` restored the link; unit file state back to `linked` | `20`, `21` |

Bonus during step 11: the GPU guard refused a dry-run plan while
`vllm-engine` was reloading on GPU 1, naming `pid 726394 in unit
system/vllm-engine.service` — the system-scope mapping works on the real
engine.

## Findings (all fixed during the pilot)

**F1 — adoption dropped six unit settings silently (`43f673d`).** The real
unit sets `RestartSec=15`, `TimeoutStartSec=900`, `StartLimitBurst=3`,
`StartLimitIntervalSec=1800`, `SyslogIdentifier=` and `After=`. The importer
never read them, so the round-trip check compared only what it knew and
passed. `own` would have written a unit with systemd defaults (100 ms
restart delay, 5 starts per 10 s), changing the very retry loop bounded
recovery defers to. Fix: import and render them (optional fields, so older
pins keep their digest and the diff ignores what a pin does not carry);
`own` copies `[Install]` verbatim and refuses any directive the pin does not
carry (step 10 is that refusal on the real unit).

**F2 — the fixed 120 s readiness budget is shorter than a real load
(`1a5d123`).** Measured loads: 181 s, 147 s, 216 s, 154 s (weights alone
78–116 s). Fix: per-workload `readiness_timeout_secs` in the inventory
(default stays 120 s).

**F3 — a second `run` at the same revision replayed the first
(`8109ae8`).** After step 6 left engine2 stopped, `run engine2` printed
"replaying existing operation … (same idempotency key)" and exited 0
without starting anything: the default key was `run-engine2-r3` and the
revision had not moved. Fix: every verb's default key is unique per
invocation (as restart already was); `--key` still retries.

## Notes and small gaps (not fixed)

- The `blocked_needs_attention` hint says `llmctl-next op ack ID`; the
  command is `llmctl-next ack ID`.
- `status` shows the recovery policy (`recover=2/3600s`), not attempts
  used; the count is only in the `recoveries` table.
- The rendered unit now lists implicit dependencies explicitly
  (`After=-.mount app.slice basic.target …`, `Requires=app.slice
  basic.target`): equivalent, noisier.
- A process outside any service (a shell, a `.scope`) shows as `no unit`.
- `own` on a *running* real engine (MainPID unchanged) was not repeated
  here; it needs the GPU window and is covered by the disposable spike.
  The real unit was owned while stopped.
- Eight clippy warnings predate this pilot (from the A–D commits).
- Three stray `target/debug/llmctl-next` test daemons from earlier
  sessions were running under `/tmp/tmp.*` (fake engines); left alone.

## Hand back

`release` returned `vllm-engine2.service` to its original link and the
workload to `external`; the pilot daemon was stopped. `vllm-engine` came
back through the `longjob` wrapper (serving `qwen3.8-27b` at 12:18:13,
lane lease free), and Python `llmctl status` lists `vllm-engine2` with its
own env file as before. Nothing in Python llmctl, the router or
the env files was changed.

## Exit checklist rows covered (pilot)

- Startup failures — early exit (recovery killed during load), GPU occupied: **pilot pass**. Wrong model: disposable only.
- Bounded recovery — budgeted, systemd retries first, manual stop respected: **pilot pass**.
- Daemon restart — mid-operation SIGKILL reconciles, engine survives: **pilot pass**.
- Stop failures (timeout → `Unconfirmed`): disposable only (a real vLLM stop took < 60 s).
- Upgrade rollback: **pilot pass** in window 2 (below).

---

# Window 2 — staged upgrade with rollback (2026-11-16)

Plan: § E window 2; target decided in the plan's owner decision 3 (vLLM
0.30.0 → 0.32.0, same `torch==2.13.0`).

## Before the window (no GPU, `vllm-engine` serving)

Reading the upgrade path for an `apply: "env"` recipe found two defects that
would have made the pilot pass falsely; both were fixed first (`8544662`):

**F4 — the env override did not apply.** Promote set an inline
`Environment=VLLM_PYTHON=<staged>`, but systemd lets `EnvironmentFile=`
override `Environment=`, and `vllm-engine2.env` sets `VLLM_PYTHON` itself:
the unit would have started the old env. Fix: an llmctl-owned override env
file listed *last* (a later env file wins), content-digest pinned.

**F5 — `VerifyVersion` proved nothing.** It asked the *staged* interpreter
for its own package versions, so promote would report `vllm=0.32.0` while
0.30.0 served. Fix: a process in the unit's cgroup must run the staged
interpreter (argv[0] or files mapped from its env), in addition to the
package check. Stage also builds the venv on the replaced env's Python
minor version (3.12) instead of whatever uv finds first.

Then: re-adopt `engine2` with the recipe (`vllm`, interpreter
`vllm-muse/bin/python`, `packages: [vllm]`, `apply: env`, `model_config` =
the model dir), daemon restart pinned version facts (schema 2: vllm 0.30.0,
lock digest, launcher digest), `own`, and **stage 0.32.0 in 26 s** (uv cache)
— validated offline: `vllm 0.32.0`, `torch 2.13.0+cu130`, CUDA 13.0, model
config loads (`qwen3_5`). Evidence `30`–`32`.

## In the window (`m2-pilot-w2 --stop-engine`, 12:27–12:44 PST, ~17 min)

| # | Step | Result | Evidence |
|---|---|---|---|
| 1 | Baseline `run` | 0.30.0 from `vllm-muse` (argv[0]), `/version` 0.30.0, smoke "baseline ok", 181 s | `33` |
| 2 | `upgrade promote engine2 0.32.0 --wait` | stop → apply → start → ready → identity → `VerifyVersion` all Done; engine argv[0] = the staged env's python; `/version` **0.32.0**; smoke "upgraded ok"; override env file listed after the unit's; 263 s | `34` |
| 3 | `upgrade rollback engine2 --wait` | back to `vllm-muse`, `/version` 0.30.0, override gone, `VerifyVersion vllm=0.30.0` Done, 204 s | `35` |
| 4 | Promote again with the new engine SIGKILLed 25 s into its load | promote **failed** at readiness ("process exited before readiness"); automatic rollback: stop → old definition → start → ready → identity, all applied; `vllm-muse` 0.30.0 serving; history row `failed`; 173 s | `36`, `37` |
| 5 | stop, `release` | stop ok; **release refused** (F7) — original link restored by hand from the recorded target; repo copy sha256 `9ffe2340…` unchanged | `38`, `39` |

## Findings in the window (fixed)

**F6 — a second promote of the same version replayed the first
(`f3c923c`).** After step 3, promoting 0.32.0 again returned the first
promote's `Succeeded` (exit 0) under key `upgrade-engine2-0.32.0` while
0.30.0 kept serving, and the completion poller marked the history row
`promoted`. Same class as F3. Rollback keyed on the history row had the same
flaw. Fix: unique keys per invocation. (A rolled-back or promoted stage must
be restaged before another promote: by design; 30 s from the uv cache.)

**F7 — release after an upgrade could not hand back the original
(`c6e0f51`).** Every definition change (promote, rollback) replaced the
ownership record's release target with the current llmctl render, reusing
the original's backup slot. `release` then restored the 0.32.0 render,
failed its equivalence check, and wrote back an llmctl render *without*
`[Install]` (the unit turned `static`). For a regular-file original it
would have overwritten the only backup of the original bytes. Fix:
definition changes undo to their own slot; the release target recorded by
`own` is carried forward; release's fallback keeps `[Install]`. The
disposable upgrade spike now re-promotes, rolls back again and checks that
`release` restores the original unit byte for byte (it fails on the old
code with the pilot's exact error).

Smaller: the README said `upgrade stage NAME --version X` (it is
positional); the interrupted-stage marker was named `engine2-0.32.staging`
(`with_extension` ate the version suffix) — both fixed in `193816f`.

## Gaps (not fixed)

- `promote --wait` can return before the completion poller marks the
  history row; an immediate `rollback` is refused ("no promoted upgrade")
  for a moment.
- History keeps one row per staged env id: restaging the same version
  overwrites the earlier promote/rollback record.
- Release after an upgrade is proven on the disposable unit only; on the
  real unit the hand-back in this window was the manual relink (exactly the
  recorded target).
- The staged env (8.0 GB, `~/.local/state/llmctl-next-pilot/envs/`) is
  kept; `upgrade gc` would remove it.
