# Rust successor — M6 progress log

Work log for [RUST-M6-PLAN.md](RUST-M6-PLAN.md). Newest status first.

**Status 2026-11-28:** M6c passed: the cutover, R1 and R2 ran twice in a
row from a fresh Debian 12 container, 75 of 75 checks each at `fb4a2cc`
([RUST-M6-REHEARSAL.md](RUST-M6-REHEARSAL.md), `scripts/rehearsal_m6c.sh`).
Three findings: R2 must run `catalog export` before the agent stops (a
procedure change); a delegated `start` of a running unit restarted it
(fixed in `llm-unit`, `~/AI` `1752e34f`, branch `fix/llm-unit-start-noop`);
`audit` missed a raw stop right after an agent stop and run (fixed in the
agent, `5ac909b`). Nothing merged to `main`, nothing deployed. Next: merge
llmctl `m6c-rehearsal` and `~/AI` `fix/llm-unit-start-noop`, then M6d.

**Status 2026-11-27:** M6b repository work is built for all eight items.
Nothing is merged, pushed or deployed, and every switch is off. Next: the
review of these branches, then M6c (rehearsal of the cutover in a Debian
12 container).

## M6b — code for the cutover

The branches. Each change ships with its switch off.

| Repository | Branch (tip) | Items |
|---|---|---|
| llmctl | `m6b-cutover` | 1 (Python), 4, 5 follow-ups, 7, catalog delete |
| llm-router | `feat/llmctl-next-only` (`8f6fe60`) | 2 |
| `~/AI` | `feat/m6b-cutover-healers-scripts` (`9848837e`) | 1 (take-gpus), 3, 6 |
| local-llm (on m5) | `m6b-next-only` (`c02eec0`) | 8 |

| Item | What is built | Switch | Tests |
|---|---|---|---|
| 1 Python cutover mode | `llmctl_next.mode: cutover`: a write verb on a unit outside the map is refused with the `llmctl-next` verb to use; `LLMCTL_NEXT_DISABLED=1` is refused; the delegate timeout is 4,200 s (two 1,800 s readiness windows and the stop). Written by the Codex M6b session, imported as `a8c8047`. `take-gpus` has no `ImportError` fallback (`aa18911d`). `llm-unit`: the guard decides about `LLMCTL_NEXT_DISABLED`; a guard that cannot be imported is a refusal (exit 3); `kill`, `enable` and `disable` ask the guard first | `settings.yaml` `llmctl_next.mode: cutover` | Python suite 1,847 in 4 shards; `test_take_gpus` 19; `test_llm_unit` 24 |
| 2 llm-router | `switch.llmctl_next_only`: each non-zero agent exit fails the switch with its reason, also 3 and 5; the router never runs `stop_first`, the Ollama eviction, the env copy, `restart_cmd` or a backend's `start_cmd`. The idle revert is a switch, so it is covered. The config check refuses the flag until every backend names its workload | `router.yaml` `switch.llmctl_next_only: true` | router switch tests 41; full router suite 387 passed, 1 failed (`test_every_backend_profile_is_tracked_in_the_repo`, fails on `main` too) |
| 3 Healers | `llm-heal`: `llmctl_next_report_only` holds every repair of a unit the agent lists in `intent --json` (also the GPU fallback start of the default engine); `mac_repair.report_only` holds the SSH kickstart of a job the Mac's agent answers for. `mac-heal`: `llmctl_next_report_only`. Only a workload with an intent counts as owned (the agent's recovery does not act on intent none or unmanaged, so those stay repairable). Each fails open: an agent that does not answer owns nothing | `heal.yaml`, `mac-heal-<node>.json` | 79 healer tests |
| 4 Agent, small | `audit --unexplained` (class `engine_scheduler` for a llama-swap lane change with an unchanged pid; Codex, `a8c8047`). Data dir default `~/.local/state/llmctl-next` (decision 3). The enrollment store and status cache follow `--data-dir` (else the socket's dir); while the data dir has no `hosts.json`, the pre-M6 `llmctl-next-dev/hosts.json` is read with a note. Version 0.2.0 (decision 13). `status --json` carries `pid` and `engine_kind` (`efc970d`) | none (code) | Rust workspace suite |
| 5 Vendor `load` | built in M5 (on `main` since `ccabf75`). The router handoff for `qwen3coder30b` is M6e item 3 | — | — |
| 6 Scripts | the lane scripts start and stop their unit with `$HOME/AI/bin/llm-unit`; the gate test `tests/test_lane_scripts_use_llm_unit.py` fails on a raw `systemctl` write in `services/lanes` (it found 7 on `main`); `ds4-llmctl-sync` is a no-op while `~/.local/state/llmctl/ds4-llmctl-sync.off` exists | the flag file | gate 3 tests |
| 7 History import | `llmctl-next history import <copy> [--as HOST] [--confirm T]` and `history list`; journal table `legacy_history`, ids `py:<host>:<id>`; a live DB (with `-wal`) is refused; a changed source row is reported and the first import kept; nothing reads the table for admission, recovery or audit (`939c554`) | none (verb) | unit test, `history_e2e`; dry run of a copy of the live Python DB: 34 rows `new` (2 sessions, 32 operations), the live DB path refused |
| 8 Mac `next_only` | `next_only: 1` turns delegation on and refuses a start or stop of a lane the agent does not own, `stop` with no lane, `benchmark`, `measure`, and the `doctor --fix` router restarts | `~/.config/llmctl/config.yaml` `next_only: 1` | 6 new tests (4 fail on the old script); local-llm suite 509 passed, 1 skipped |

Also on `m6b-cutover`: `llmctl-next model delete` (owner: "Add a catalog
delete", `df4d113`), M5 gap 2 passed live (`6cfe42d`), and the catalog
e2e agent reads no presets of the owner.

### Facts found during M6b that M6d needs

1. The workstation's host enrollments are in
   `~/.local/state/llmctl-next-dev/hosts.json` and `hosts/`, not in the
   live data dir. Copy them into the live data dir at the deploy. Until
   then the new CLI reads them from the old place.
2. The Python build with `import-registry --update-models` and the
   cutover mode must be in `~/AI/.venv` before the switch (M6d step 6).
3. The lane scripts call `llm-unit` by its full path: a user unit has no
   `~/AI/bin` on `PATH`. They call it only in cutover mode (review fix 1).

### Review fixes (2026-11-27)

Seven findings of the code review. Each new behaviour is off until its
switch is set. Commits in `~/AI` are on `feat/m6b-cutover-healers-scripts`,
the others on `m6b-cutover`.

| # | Commit | What changed | Test |
|---|---|---|---|
| 1 | `~/AI` `e9facc4a` | The lane scripts deadlocked with every switch off: `llm-unit` sent an enrolled unit to the agent while the caller held the host lock. `services/lanes/_unit.sh` `lane_unit` now uses `llm-unit` only in cutover mode and `systemctl --user <verb> <unit>` in any other mode. New verb `llm-unit mode` prints `cutover` or `delegate` | `tests/test_lane_scripts_use_llm_unit.py` (the six scripts use the helper; three `lane_unit` tests), `test_mode_*` in `tests/test_llm_unit.py` |
| 2 | `1dbb007` | Cutover mode guarded only `llmctl run <preset>`. Now `lane_swap.unit_write` is the one place for a unit write: `llmctl stop <lane>`, the TUI swap helpers and the rollback go to the agent for an enrolled unit and are refused for any other. `llmctl run <enrolled lane>` is one `llmctl-next run <workload> --wait`. A session start is refused. `install_unit` does not `enable --now` | `tests/test_cutover_writes.py` (12 tests: agent, refusal, old argv) |
| 3 | `1f31e0f` | `history import --confirm` refused a WAL-mode `.backup` copy: the dry run had created `-wal` and `-shm` beside it. The snapshot is opened `file:<path>?mode=ro&immutable=1` | `history_e2e.rs` `a_wal_mode_backup_copy_is_imported_and_nothing_is_written_beside_it` |
| 4 | `~/AI` `2dbe07ea` | `llm-unit` refused every unit when the guard could not be imported, also with the switches off. It now reads the cutover switch itself (line scan of `settings.yaml`): cutover and no guard = exit 3, else `systemctl` as before | `tests/test_llm_unit.py` (import failure and missing guard Python, both modes) |
| 5 | `~/AI` `0055fc82` | The healers repaired the agent's engines when the agent did not answer. With report-only on and no usable answer they hold every engine repair and still repair front doors. Off: unchanged | `test_report_only_holds_*`, `test_report_only_still_repairs_a_front_door_*` in `tests/test_llm_heal.py` and `tests/test_mac_heal.py` |
| 6 | `~/AI` `153bb70f` | `take-gpus` ran `ollama stop <model>`. In cutover mode with `ollama.service` enrolled it runs `llmctl-next unload <workload> <model> --wait`; a refusal is a failed take | `test_cutover_unload*` in `packages/llmctl-stack/tests/test_take_gpus.py` |
| 7 | `9405286` | Any data dir read the pre-M6 `hosts.json`, so a temp agent bridged to the real Macs. Only the default dir and `~/.local/state/llmctl-next-m4a` read it now | `enroll.rs` `only_the_default_and_the_live_workstation_dir_read_the_pre_m6_store` |

Also `~/AI` `5829eea3`: the llmctl-stack suite tests the checkout. Before,
a git worktree tested the copy that `~/AI/.venv` has installed.

Suites after the fixes: llmctl Python 1861 passed; Rust workspace 371
passed, 32 ignored, and `bridge_subcommand_relays_http_to_the_socket`
failed once (the known flake; 3 of 3 passed alone); the five `~/AI` test
files 148 passed.

Facts for M6d from the fixes:

1. Deploy `llm-unit` and `services/lanes/_unit.sh` together. An old
   `llm-unit` has no `mode` verb: `lane_unit` then uses `systemctl`.
2. In cutover mode `llmctl run <lane>` needs the lane's unit in
   `llmctl_next.units`. The agent plans the stops; llmctl stops nothing.
3. Report-only with a stopped agent holds every engine repair. The alert
   says "llmctl-next did not answer".

### Open in M6b

- `llm-unit` start, stop or restart with extra options (`--no-block`)
  keeps the systemctl path, also in cutover mode.
- The intent-only verb and `ConflictedBy` (M6a findings 5 and 6) are not
  built.
- The stray `llmster` (M6a finding 2) is fixed: the test suite started it
  through the real `lms` CLI. `llmster` keeps its manual stop.
