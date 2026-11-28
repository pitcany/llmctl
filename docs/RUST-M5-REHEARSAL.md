# Rust successor — M5f rehearsal (migration and reversal on temporary state)

Status 2026-11-27: **passed three times in a row**, 33 of 33 checks each
time, with the Python code of this branch. With the Python build that is
deployed now (`55c83de`) it stops at check R5: that build has no
`import-registry --update-models` (finding 1). The owner asked for this
rehearsal before M6b ("Do a rehearsal before M6b", 2026-11-27). It closes
M5 gap 6 and the M5 exit row "Migration/reversal rehearsal".

Plan text: [RUST-M5-PLAN.md](RUST-M5-PLAN.md), M5f. M6c extends this
rehearsal ([RUST-M6-PLAN.md](RUST-M6-PLAN.md)).

## How to run it

```bash
python3 scripts/rehearsal_m5f.py --python-llmctl <a Python llmctl with --update-models>
```

Options: `--agent-bin` (default: the workstation agent's binary file),
`--python-llmctl` (default `~/AI/.venv/bin/llmctl`), `--db`, `--presets`,
`--as-host`, `--keep` (keep the temp dir). Exit 0 = each check passed. One
run takes about 10 s.

## What the rehearsal touches

- It reads the Python DB through the SQLite backup API (source opened
  read-only) and copies the preset files. It reads no `settings.yaml` of
  the host. It writes its own settings file, with no token.
- It makes one systemd user unit, `llmctl-next-rehearsal-<id>.service`, in
  `~/.config/systemd/user/`. The unit runs a stand-in engine: a Python
  HTTP server that reads `VLLM_PORT` and `VLLM_SERVED_NAME` from its
  `EnvironmentFile` and answers `/v1/models`. The script removes the unit
  at the end.
- It starts its own agent (the same binary as the live agent) on a temp
  data dir with `LLMCTL_NEXT_OP_LOCK=off`, `LLMCTL_NEXT_LANE_LEASE=off` and
  `LLMCTL_NEXT_ROUTER=off`. The workload is on the notional GPU 9. Thus
  the temp agent cannot block the live agent, and it stops no real engine.
- The Python tool runs with `LLMCTL_CONFIG_DIR`, `LLMCTL_DB_URL`,
  `LLMCTL_OP_LOCK` and the `XDG_*` dirs in the temp dir, with
  `LLMCTL_PLUGINS=none` and `LLM_UNIT_BIN=none`.

Checked after the runs of 2026-11-27: the live serving signature is
the same before and after (`2e267a06…`), `audit` shows 0 foreign writes,
`engine-pins-check` is ok, and no temp dir, unit or process is left.

## Forward: Python state into a temp agent

| Check | What it proves | Result 2026-11-27 |
|---|---|---|
| F1 | A consistent DB snapshot, with a sha256 | pass (`0ec8fae4…`) |
| F2 | Python `export-registry` of the snapshot | pass: 74 models, 2 profiles |
| F3 | The temp agent answers | pass |
| F4, F5 | `catalog import --as <host>`: dry run, then `--confirm` with the report's token | pass: 54 rows for this host, 20 unmapped rows kept |
| F6 | A second dry run finds nothing new | pass: 54 `present` |
| F7 | The disposable unit serves `reh/model-a` | pass |
| F8, F9 | `adopt` with `--engine-kind vllm --gpus 9 --readiness-timeout 60`: each preview check passes; the confirm keeps the pid | pass |
| F10 | The agent reads the five copied presets and two rehearsal presets | pass: 7 listed |
| F11 | `preset sync` writes each profile env | pass |
| F12 to F14 | `run reh-b --wait`: stop, apply the profile, start, verify. The unit has a new pid and serves `reh/model-b`. The active env equals the profile; the old env is kept as `.llmctl-previous` | pass |
| F15 | The serving signature agrees with the agent's record | pass |
| F16, F17 | Two catalog edits: `model add`, and `model edit` of an imported row | pass |
| F18 to F20 | Delegate state: the agent stops the unit; Python `llmctl start-unit` is redirected, the agent runs it (2 new operations), the unit serves again | pass |

The 54 rows are `no_store` here because the temp agent declares no store.
On the live agent the same import gave 53 `new` and 1 `no_store`.

## Reverse: back to the Python tool

| Check | What it proves | Result 2026-11-27 |
|---|---|---|
| R1 | Drained: no operation that is not terminal | pass |
| R2 | Signature before the reversal agrees with the agent | pass |
| R3, R4 | `catalog export` gives back each row of the bundle, plus the added row | pass: 75 models, 2 profiles |
| R5 | Python `import-registry --update-models` into the DB copy | pass: `Imported 1 models, updated 1 models` |
| R6 | Python sees the added model | pass |
| R7 | Python sees the edit of an imported row | pass (failed before finding 1 was fixed) |
| R8 | The agent stops; the engine keeps its pid and its model | pass |
| R9 | Negative drill: delegation on, agent down. Python `start-unit` is refused and nothing starts | pass (exit 1, the message names the reversal setting) |
| R10, R11 | `llmctl_next.enabled: false`: Python `start-unit` starts the unit with `systemctl --user` | pass |
| R12 | Python `status --json` shows the same unit, port and served model | pass |

## Findings

1. **A reversal lost each edit of an existing catalog row** (fixed on
   this branch). The agent's export held the edit. Python
   `import-registry` skipped the row, because it skips each row whose
   `(runtime, source or name)` exists, and in a reversal each row exists.
   Only new rows arrived. The earlier test imported into an empty DB, so
   it did not show this. Fix: `import-registry --update-models` updates a
   row that has the bundle row's id (the agent keeps the ids). The default
   is not changed. Test:
   `test_a_reversal_into_the_db_it_came_from_needs_update_models`.
2. **The live reversal needs that Python build first.** R1 and R2 of the
   M6 plan say "run `catalog export` and Python `import-registry`". Use
   `import-registry --update-models`, and install the Python build that
   has it in `~/AI/.venv` before the cutover (M6d step 6). On the Macs the
   Python tool is an editable install, so a pull is enough.
3. **`adopt` warns when no `--readiness-timeout` is given.** The preview
   check `readiness_timeout` is then `warn`, not `pass`. M6e already
   passes the flag for each lane.

## What this rehearsal does not cover

M6c must add these, in the container:

- The cutover switches of M6b (`llmctl_next.mode: cutover`, the router's
  `llmctl_next_only`, the healers, the Mac `next_only`).
- `history import`.
- A legacy `llmctl run <preset>` after the reversal. Only `start-unit` is
  used here as the legacy write.
- llm-router, `llm-heal`, the lane scripts and the Macs.
- A real engine and real cards.
- Isolation from the host's service manager: this rehearsal uses the
  host's systemd user manager with one disposable unit, as the
  real-systemd test suite does. M6c runs in a Debian 12 container.

## Procedure for the real reversal (from the checks above)

1. Drain: `llmctl-next ops` shows no operation that is not terminal.
2. Take the serving signature (`scripts/serving_signature.py`).
3. `llmctl-next catalog export <file>`.
4. `llmctl import-registry <file> --update-models`. Read the line
   `Imported N models, updated M models`.
5. Set `llmctl_next.enabled: false` in `settings.yaml`.
6. Stop the agent. No engine stops.
7. Compare `llmctl status --json` with the signature of step 2.
8. Do one write with the Python tool, on a unit that can be away.
