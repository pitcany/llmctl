# Rust successor — M4b step 3: system → user units (plan, 2026-11-17)

Parent: [RUST-M4-PLAN.md](RUST-M4-PLAN.md) (M4b step 3, owner decision 4:
`vllm-engine` and the main `ollama` move to user units owned by
llmctl-next; no system-scope adapter, no new sudo path).
Progress so far: [RUST-M4B-PROGRESS-2026-11-17.md](RUST-M4B-PROGRESS-2026-11-17.md).

**Status: draft for the owner. Nothing changed yet.**

## Why the plan needs this step

The M4 plan said: one GPU window, update llm-router's restart command,
llm-heal's targets and the sudoers rule together. A sweep on 2026-11-17
(read-only) shows that is not enough:

- **~40 live scripts** call systemd on these units by name, in seven shapes
  (counts across `~/AI`, `~/bin`, `~/.local/bin`, workflow-skills, llmctl):
  `sudo systemctl stop vllm-tp` ×32, `sudo systemctl restart vllm-engine`
  ×18, `sudo systemctl start vllm-tp` ×14, unprivileged `systemctl
  is-active vllm-engine|vllm-tp` ×23, `reset-failed` ×19, `sudo -n …` ×19,
  `… ollama` ×19; plus `journalctl -u vllm-engine|ollama` in 8 files.
  The old `vllm-tp` alias is still most of the traffic.
- **Variable-unit callers:** longjob (`sudo -n systemctl <verb> <unit>`,
  used by every `--stop-engine` job), llm-router (`restart_cmd` /
  `reset_failed_cmd` in `config/router.yaml`), hf2local (`gpu_handoff.py`,
  `doctor.py`), llmctl-stack (`take_gpus.py`, already `--user` for its
  lanes), Python llmctl (`integrations/systemctl.py`, its single
  chokepoint).
- **Already scope-aware:** llm-heal (`heal.yaml` has `scope:` per unit;
  `ollama-gpu1` is `scope: user` today).

If only the units moved, every one of those callers would either fail
(no system unit active) or — worse — `sudo systemctl start vllm-engine`
would start the *disabled* system unit next to the user unit: two vLLMs
reserving the same cards. `disable` does not stop a manual start, and
`mask` is not in the sudoers rule.

Other facts that shape the plan:

- `Linger=yes` for yannik: user units start at boot without a login.
- `vllm-engine` already runs `User=yannik`; its `Conflicts=` /
  `After=llama-server.service` names a unit that no longer exists (drop).
- System `ollama` runs as user `ollama`; `~/.ollama/models` is
  `ollama:ollama 2775` (setgid, group-writable) and yannik is in group
  `ollama`, so a user-scope Ollama can read every blob (644/664) and write
  new ones. `ollama-gpu1` (user unit) already shares that store.
- Only `math-shim.service` (user unit) orders on `ollama.service`; in user
  scope that becomes a real ordering after the move (better than today).

## Design

### 1. One wrapper: `llm-unit`

`~/AI/bin/llm-unit <verb> <unit> [args]` — the only way scripts touch these
units. Verbs: `start stop restart is-active [--quiet] is-failed status show
reset-failed enable disable` plus `journal <unit> [journalctl args]` and
`scope <unit>`.

- **Scope from one file**, `~/.config/llm-unit/scopes`
  (`vllm-engine=system`, `ollama=system` at first). Units not listed: user
  scope if `systemctl --user cat` finds them, else system.
- **Aliases resolved first:** `vllm-tp` → `vllm-engine`.
- system → `sudo -n systemctl <verb> <unit>` for mutating verbs (the
  existing sudoers lines), plain `systemctl` for reads; user →
  `systemctl --user <verb> <unit>`. `journal` → `journalctl -u` vs
  `journalctl --user-unit`. Exit codes pass through unchanged, so
  `if llm-unit is-active --quiet vllm-engine` keeps working.
- **Interlock:** `start`/`restart` refuse when the same unit is active in the
  *other* scope (exit 3, one-line reason). This makes "two vLLMs" impossible
  through the wrapper during and after the flip.
- `llm-unit audit` lists every raw `systemctl|journalctl` call on the
  managed units under the known roots (backups, run logs and archives
  excluded). It is the 3a exit check and the 3b preflight.
- Shell, ~80 lines, a bats-style test file driving it against a fake
  `systemctl` on `PATH`.

### 2. Phase 3a — migrate callers, no window, no behaviour change (~1 day)

The scopes file still says `system`, so every migrated caller does exactly
what it did before. Order (each repo its own commit, tests run per repo):

1. Install `llm-unit` + tests (AI repo).
2. Programs with config: llm-router (`router.yaml` `restart_cmd` /
   `reset_failed_cmd` → `["llm-unit", …]`), longjob (`systemctl()` →
   `llm-unit`), hf2local (`gpu_handoff`, `doctor` — the doctor's sudoers
   check becomes "llm-unit can act"), Python llmctl
   (`integrations/systemctl.py`), llmctl-stack. llm-heal stays on its
   `scope:` field (flipped in 3b) — or calls `llm-unit`; owner's pick.
3. Scripts: mechanical rewrite of the seven shapes (`sudo [-n] systemctl
   <verb> <unit>` → `llm-unit <verb> <unit>`, `systemctl is-active …` →
   `llm-unit is-active …`, `journalctl -u <unit>` → `llm-unit journal
   <unit>`), shown as a diff per repo before it is applied. `vllm-tp`
   becomes `vllm-engine` in the same pass (the unit's own comment plans to
   drop the alias).
4. Exit: `llm-unit audit` prints nothing; the suites of llm-router,
   hf2local, llmctl-stack, llmctl and workflow-skills pass; a normal
   `longjob --stop-engine` job and a router switch still work (they now go
   through the wrapper, still system scope).

Docs/artifacts (`driving-llmctl`, the two fleet pages) are updated after
3b, through the artifact audit.

### 3. Phase 3b — the flip, one GPU window (~45 min)

One script, `~/AI/scripts/unit-scope-flip.sh user|system`, does both
directions, so the rollback is the same tested code. Forward (`user`),
inside `longjob queue unit-scope-flip --stop-engine --stop ollama`:

1. Preflight: `llm-unit audit` empty; both system units inactive (longjob
   stopped them); the user unit files parse (`systemd-analyze --user
   verify`).
2. Install the user units from the AI repo:
   - `services/vllm-engine.user.service` — the system unit minus
     `User=/Group=` and the dead `llama-server` lines; same
     `EnvironmentFile`, launcher, `KillMode=mixed`, start limits;
     `[Install] Alias=vllm-tp.service`, `WantedBy=default.target`.
   - `services/ollama.user.service` — `/usr/local/bin/ollama serve` with
     the `coexist.conf` environment (`OLLAMA_MODELS`, `OLLAMA_HOST`,
     `NUM_GPU`, `KEEP_ALIVE`, `MAX_LOADED_MODELS`), `Restart=always`.
3. `sudo -n systemctl disable vllm-engine ollama` (both verbs are already
   in the sudoers rule — no new sudo path); the system units stay on disk.
4. Flip the scopes file to `user`; `systemctl --user enable` both.
5. longjob's end-of-job restart (`llm-unit start …`) now starts the
   **user** units and waits until they answer — the same handoff as every
   window.
6. Verify: `:8003/v1/models` and `:11434/api/version` answer; llm-router
   status sees the engine; llm-heal (scope flipped to `user` in
   `heal.yaml`) reports both ok; a Hermes local cron fires; the interlock
   refuses `llm-unit start` in the old scope.
7. Adopt both into llmctl-next (`--engine-kind vllm`,
   `--engine-kind ollama --model ollama`): adoption only observes, no
   restart.

**Rollback** (any time during the week): `unit-scope-flip.sh system` —
stop the user units, flip the file back, `sudo -n systemctl enable
--now` the system units, disable the user units. Models Ollama pulled
while in user scope stay readable by the `ollama` user (644, group
`ollama`).

After a week without rollback, the owner (by hand) removes the system
units' sudoers lines and masks or deletes the system units.

## Risks

| Risk | Guard |
|---|---|
| A caller the sweep missed starts the system unit after the flip | `llm-unit audit` at 3a exit and 3b preflight; system units disabled; the interlock covers every wrapped caller; owner masks after a week |
| Ollama as yannik cannot write the store | group `ollama` + setgid dir checked today; preflight writes and removes a test file as yannik |
| Boot order: user manager starts later than system units did | `Linger=yes`; Hermes and llm-router already tolerate a late engine (they do during every window) |
| llm-router flock / lane lease semantics change | unchanged: the wrapper only changes *how* systemctl is called, not when |

## Owner decisions

1. Wrapper name and home: `llm-unit` in `~/AI/bin` (recommended).
2. llm-heal: flip its `scope:` field (recommended, smallest) or route it
   through `llm-unit`.
3. Rename `vllm-tp` callers to `vllm-engine` in the 3a pass (recommended)
   or keep the alias indefinitely.
4. Which llmctl-next daemon adopts the moved units in 3b: the workstation
   pilot daemon (as for step 2, recommended until M6) or the main agent.
5. After the week: mask (keeps the file) or delete the system units.
