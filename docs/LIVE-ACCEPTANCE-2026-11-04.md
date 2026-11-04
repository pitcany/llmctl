# Live acceptance — control-plane coordination (R1–R3, P1)

Status: **RUN 2026-11-04 09:21–09:46 PST** (approved by the owner). 4 of 5 pass; T5
(healer deferral) not exercised live. Results are at the end of this file; the plan below is
unchanged except where the Results section records a deviation.
Scope: the five tests listed under "Separately approved live acceptance" in
[DEPLOYMENT-PLAN-2026-11-03.md](DEPLOYMENT-PLAN-2026-11-03.md). Each test restores the
original serving model (`qwen27b`, TP=2) and respects leases.

## Starting state (verified 08:38–08:45)

| Item | Value |
| --- | --- |
| Router | `~/AI/llm-router` @ `e0377a0`, `/running`: `qwen27b` loaded, no lease, pause, broken, cooldown, rollback; `in_flight: {}` |
| Engine | system `vllm-engine`, Qwen3.8-27B-FP8, TP=2, ~27 GB per card, 0 % utilisation |
| Lock | `~/.local/share/llmctl/operations.lock` free |
| Switch cost | qwen27b start measured at 97 s (2026-11-04 03:34:54 → 03:36:31) |
| Settings | `min_hold_s: 300`, `op_lock_wait_s: 30`, `idle_revert_s: 3600`, `idle_poll_s: 30`, `idle_revert_unknown_load: defer` (global only, no per-lane override) |
| Lanes used | `qwen27b-awq` (vLLM, TP=1, same unit and port :8003); `qwen3coder30b` (Ollama on GPU 1, :11435, `qwen3coder1-start --take-gpus`) |

### Constraint found while planning: the systemd start limit

`vllm-engine` has `StartLimitIntervalSec=1800` and `StartLimitBurst=3`. Every
start counts, successful or not, including `Restart=on-failure` retries. This
plan needs 4–5 starts within about 30 min. The rollback failure on 2026-11-01
15:38:47 (`restart_error=Job for vllm-engine.service failed.`, raised within
milliseconds of the switch beginning) matches a start-limit refusal. Mitigation
for the session: run `sudo -n systemctl reset-failed vllm-engine` (allowed by
sudoers) before every planned start. The router does not do this itself; see T4.

## Preflight (≈5 min, no disruption)

1. Snapshot `/running`, `nvidia-smi`, `llm-heal status`, and the
   `sha256sum` of `~/AI/services/vllm-engine.env` (hash only; the file is not read).
2. Back up `config/router.yaml` to `~/AI/logs/staged-2026-11-03-control-plane/live-acceptance-20261104/`.
3. Confirm `git -C ~/AI/llm-router status` is clean and no benchmark or eval process is running.

Local inference (router, LiteLLM `local`, shims) is interrupted for the whole
session, about 45 min.

## T1 — competing `llmctl run --direct` vs a client request, plus T5 healer deferral (≈5 min)

1. `reset-failed`; start `llmctl run qwen27b-awq --direct -y` in the background.
   It holds the flock through the engine restart.
2. +5 s: a client request for `qwen3-coder:30b` through :4100 (max_tokens 8).
3. +10 s: `/usr/bin/python3 ~/AI/heal/llm-heal.py run`.

Pass:
- The client gets a 503 after ≤30 s that names the llmctl holder.
- No `model.switch.begin` is logged, and `cooldown` stays empty.
- The healer logs that repairs are held for the lock and restarts nothing.
- llmctl finishes with `qwen27b-awq` serving on one card, and the router reports it.

## T3 — long stream across an idle revert (≈12 min)

1. Set `idle_revert_s: 120` and `idle_poll_s: 10` in router.yaml; restart `llm-router` (no engine restart).
2. Stream to `qwen3.8-27b-awq` with `max_tokens: 16000` and `ignore_eos: true`, so it runs about 4 min, longer than `idle_revert_s`.

Pass:
- While streaming: `idle_revert.in_flight` is non-empty and no switch begins.
- The stream finishes with `finish_reason: length` and no error or truncation.
- After it ends, the revert to `qwen27b` happens on its own (allowing for min_hold), and `model.switch.done to=qwen27b` follows.
- Restore: `git checkout config/router.yaml` (after confirming the diff is only those 2 lines), restart the router, and confirm 3,600.

## T2 — TP=2 takeover (≈6 min)

1. `reset-failed`; `POST /admin/switch {"backend":"qwen3coder30b","force":true}`
   (force skips only min_hold and the failure cooldown). vllm-engine stops, and Ollama loads on GPU 1.
2. Send one request to `qwen3-coder:30b` and check that it is served from GPU 1.
3. `POST /admin/switch {"backend":"qwen27b","force":true}`.

Pass:
- `ollama-gpu1` is stopped and GPU 1 freed before vLLM starts.
- vLLM comes up with TP=2 (both cards ~27 GB).
- One completion through LiteLLM `local` succeeds.

## T4 — failed-start rollback (≈8 min)

1. `reset-failed`. Create `~/AI/services/vllm-engine.env.acc-badstart` (mode 0600, written
   from scratch, with `VLLM_MODEL=/nonexistent/acc-badstart` and `VLLM_SERVED_NAME=acc-badstart`).
2. Add a temporary backend `acc-badstart` (that profile, upstream :8003,
   `health_timeout_s: 180`); restart the router.
3. `POST /admin/switch {"backend":"acc-badstart","force":true}`.

Pass:
- The switch fails.
- The active env's sha256 equals the preflight hash.
- `last_rollback` is populated with the restart issued (rc 0).
- `qwen27b` serves again and one completion through `local` succeeds.

Expected risk: the `Restart=on-failure` retries can use up the 3-start burst, so the
rollback restart is refused (the 2026-11-01 failure). If that happens, record it
as a finding and recover with
`sudo -n systemctl reset-failed vllm-engine && sudo -n systemctl restart vllm-engine`
(the env is already restored). The fix is a follow-up, not part of this session.

Cleanup: `git checkout config/router.yaml`, restart the router, delete the bad env file.

## Final restore check

All of the following must hold:
- `/running`: `qwen27b`, no lease, pause, broken or cooldown entries.
- Active env sha256 equals the preflight hash.
- router.yaml is clean at `e0377a0`, with `idle_revert_s` 3,600.
- `ccl --ask local` answers.
- `llm-heal status` is clean.
- A final `reset-failed` has run, so later starts are not refused.

**Abort rule:** if any step leaves the engine without a serving model for more
than 10 min past its expected time, stop testing and run
`POST /admin/switch {"backend":"qwen27b","force":true}` (after `reset-failed`).

## Results

Run by Claude session llmctl-c7. Raw logs were kept in the session scratchpad. The router.yaml
backup is at `~/AI/logs/staged-2026-11-03-control-plane/live-acceptance-20261104/`.
Auth: test clients sourced `ROUTER_API_KEY` from `~/.config/llm-router/env` into their own
environment (the value was not printed). Peer session ai-64 held its GPU embedding
`--apply` for the window.

| Test | Verdict | Evidence |
| --- | --- | --- |
| Preflight | PASS | 09:21:50. `qwen27b` serving, env sha256 `b04ed4f5…` (= `vllm-engine.env.qwen38`), no lease/pause/broken/cooldown, healer clean, `NRestarts=0` |
| T1 llmctl side | PASS | `llmctl run qwen27b-awq --direct -y` held the flock (`fuser`: pid 3950843 at +5 s); finished rc 0 in 233 s on GPU 1; the router then reported `qwen27b-awq`; no cooldown, broken entry or rollback |
| T1 client side (first try) | INVALID | Client got `401 missing or invalid bearer token`: the router requires `ROUTER_API_KEY`, and the plan had left out auth |
| T1b competing client (re-run) | PASS | `flock` held the lock 50 s; a request for `qwen3-coder:30b` got **503 after 30.03 s**: "another GPU operation is active … waited 30s"; `model.switch.op_lock_busy`, no `switch.begin`, cooldown empty, AWQ lane kept serving |
| T3 long stream across idle revert | PASS | `idle_revert_s` 120 / `idle_poll_s` 10. Stream 139 s (> 120 s), 6,493 chunks, `finish: length`, no error. `/running` showed `in_flight {:8003: 1}` and `deferred: 1 request(s) in flight` at every sample. `idle.revert.begin` at 09:33:30, **121 s after the stream ended**; `switch.done to=qwen27b` 107 s later. Config restored to 3,600 / 30 (matches backup) |
| T2 via router (first try) | BLOCKED (pre-existing defect) | `/admin/switch qwen3coder30b` failed in 66 ms: `qwen3coder1-start: no backend named 'qwen3coder1'`. Nothing was stopped; qwen27b kept serving |
| T2b manual lane + takeover | PARTIAL | Lane brought up by hand under the flock, but the coder model did not load (request sent before Ollama listened; GPU 1 at 17 MiB). Takeover itself OK (93.6 s) |
| T2c TP=2 takeover | PASS | `qwen3-coder:30b` resident on GPU 1 (30.6 GB, `size_vram` 31.5 GB); router `/running` showed `qwen3coder30b`. `/admin/switch qwen27b force` stopped `ollama-gpu1` first, then started TP=2 in 115 s; GPUs 26.8 / 26.0 GB, `ollama-gpu1` inactive; `ccl --ask local` OK |
| T4 with nonexistent path | GUARDED | The router's `model_available` preflight refused in 33 ms ("refusing to swap + restart"); nothing changed |
| T4b failed-start rollback | PASS | Profile pointed at a dir with `config.json` `{}`. vLLM crashed (`Unrecognized model … model_type`); the router reported the crash at +23 s; `model.switch.rollback restored=…vllm-engine.env restarted=True restart_error=None`; env sha256 back to `b04ed4f5…`; `last_rollback.complete: true`; qwen27b serving at +119 s; `ccl` OK. **The start limit was not hit**: the router caught the crash before systemd's own retry (2 starts) |
| T5 healer deferral | NOT EXERCISED | The healer ran at +11 s (T1) and +15 s (T4b, router holding the lock, pid 937356). Both times it made no repair: in T1 it judged the engine "idle by choice", in T4b "loading (4 s of a 10 m allowance)". The lock deferral (`llm-heal.py:440`, `self.busy`) is only reported when a repair is planned, so the deferral path was not reached. Covered by unit tests only |
| Final restore | PASS | 09:46. `qwen27b` serving; env sha256 = preflight; router.yaml = `e0377a0` (= backup), `idle_revert_s` 3600; no lease/pause/broken/cooldown/rollback; lock free; healer ok; `reset-failed` done; T4 fixtures deleted |

### Findings

1. **`qwen3coder30b` lane cannot start (pre-existing).** `start_cmd` is
   `~/bin/qwen3coder1-start --take-gpus`, and the script takes its lane name from its own filename
   (`qwen3coder1`). Commit `58ea349` (one naming grammar, hard cutover) renamed the backend to
   `qwen3coder30b`. **Fixed 2026-11-04 ~09:55:** link `qwen3coder30b-start` added (`~/AI`
   `8557b8f6`, `~/bin`), `start_cmd` pointed at it (router `9373e71`, router restarted),
   RUNBOOK note `72afa21e`. Verified: `qwen3coder30b-start --show-model` resolves
   `qwen3-coder:30b` from router.yaml. A live switch to the lane was not re-run.
2. **The op-lock holder text can be stale.** In T1b the 503 named `pid 3950843 … run
   qwen3.8-27b-awq, 311s ago`, an llmctl that had already exited. The real holder was a plain
   `flock` (pid 67651). The router reads metadata left in the lock file by the last writer
   without checking that the pid is alive. Cosmetic for llmctl/router/healer holders, which
   rewrite the metadata, but misleading for any other holder.
3. **The start limit is a latent risk, not hit here.** `StartLimitBurst=3` per 1,800 s counts
   every start. Three planned switches in 30 min, or a crash loop, will make the next
   restart (including a rollback restart) fail, as on 2026-11-01 15:38:47. The router does
   not run `reset-failed` before restarting. Candidate fix: the router (it already has sudoers
   rights) runs `systemctl reset-failed vllm-engine` before a switch or rollback restart, or
   the unit's burst is raised.
4. **Minor log wording.** `model.switch.rollback to=acc-badstart` names the *failed* target,
   not the restored one. `previous` / `from_` is `none` after a router restart, even when a
   lane is loaded.
5. **T5 needs a different setup.** To prove the deferral live, the healer has to want a repair
   (engine `failed`) while the lock is held, which means extra downtime and start-limit risk.
   Not attempted.
