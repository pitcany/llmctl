# Rust successor — M2 pilot, window 1 (2026-11-16)

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
- Upgrade rollback: window 2 (vLLM 0.30.0 → 0.32.0), not yet run.
