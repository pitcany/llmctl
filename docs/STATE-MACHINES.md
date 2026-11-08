# State machines in llmctl

The vocabularies below are the states llmctl moves things through. Each one is
a `StrEnum`, so a member compares equal to its stored string: `status ==
SessionStatus.RUNNING` and `status == "running"` are the same test, JSON and
SQLite keep the plain string, and the rendered word does not change.

Rule for new code: compare against the member, never the string literal. The
string literals that were scattered across the CLI, TUI and API were the
implicit state machines this document replaced (2026-11-08).

`tests/test_state_vocabularies.py` checks that every member below is named in
this file and that the predicates agree with the sets.

## 1. Session lifecycle — `SessionStatus` (`llmctl/db.py`)

Stored in `sessions.status`. Written only through
`llmctl.services.sessions.set_status`, which logs at DEBUG when a move is
outside `SESSION_TRANSITIONS`; it never refuses, so the table documents the
machine without being able to break it.

| From | To | Where |
| --- | --- | --- |
| `planned` | `starting` | `_launch_record` (start) |
| `planned` | `stopped`, `failed` | `stop` on a planned row; launch without an adapter |
| `starting` | `running` | owned/adopted endpoint answers (`reconcile`, `_on_spawn`) |
| `starting` | `failed` | owned process exits before READY |
| `starting` | `stopped` | `stop`; adopted unit stopped |
| `starting` | `degraded` | process survived a stop attempt (`_mark_survivor`) |
| `running` | `degraded` | endpoint stops answering while the pid lives |
| `running` | `stopped` | `stop`; process gone |
| `degraded` | `running` | endpoint answers again |
| `degraded` | `stopped` | `stop`; process gone |
| `stopped` | `running` | adopted unit came back on its own (`_reconcile_adopted`) |
| `stopped`, `failed`, `running`, `degraded`, `starting` | `planned` | `restart` re-plans an owned session |
| `stopped`, `failed`, `running`, `degraded` | `starting` | `restart` of an adopted unit |
| `unknown` | any | reconcile could not classify the row |

Sets, in `llmctl/db.py`:

- `SESSION_ACTIVE_STATES` = `starting`, `running`, `degraded` — probed by reconcile, routed by the gateway (`SessionStatus.is_active`).
- `SESSION_TERMINAL_STATES` = `stopped`, `failed` — `cleanup` may purge them (`SessionStatus.is_terminal`).
- `stopping` is declared for API clients but no code path enters it; `stop` goes straight to `stopped` or `degraded`.

Surfaces: `cli.py::_print_session_outcome`, TUI `screens_sessions._STATUS_COLOR`
and `_BAD_STATUSES`, `screens_models._after_start`, `_data` running count,
`stop_serving`.

## 2. Mutating operation — `OperationPhase` (`llmctl/services/operations.py`)

Stored in `operations.phase`. Set by `OperationStore.set_phase` and the
`mark_phase` helpers while a swap or stop runs.

`preparing` → `stopping` → `loading` → `checking_readiness` → (`restoring`) →
one of the terminal phases `completed`, `failed`, `interrupted`,
`needs_attention`.

- `TERMINAL_PHASES` / `TERMINAL_PHASE_VALUES` / `OperationPhase.is_terminal` are the one definition of "over".
- `needs_attention` is `failed` plus a restore that also failed: a human has to put the lanes back.
- `interrupted` is a record whose owner process died; `reconcile` finalizes it.

Surfaces: TUI `screens_swap` and `screens_dashboard` operation panels, web UI
job status line.

## 3. Swap job — `SwapJobStatus` (`llmctl/services/swap_jobs.py`)

What the browser, CLI and TUI see of a running swap: a three-state projection
of the operation phase, built by `SwapJobStatus.from_phase` and
`from_exit_code`.

| Operation phase | Job status |
| --- | --- |
| any non-terminal phase | `running` |
| `completed` | `succeeded` |
| `failed`, `interrupted`, `needs_attention` | `failed` |

The API models `SwapJobSummary` and `SwapJobDetail` type `status` with this
enum, so the OpenAPI schema lists the three values.

## 4. Lane picker row — `LaneState` (`llmctl/services/lanes.py`)

Derived on every read by `SwapState.state_of`, never stored.

- `serving` — the lane answers (served model, running unit, or live GPU session).
- `loading` — the lane's unit is active but not yet serving.
- `weights gone` — the catalog row's weights were removed; `llmctl run` refuses it.
- idle — the empty string, so the State column stays blank (`LaneState.IDLE`).

## Classification vocabularies (not lifecycles)

These label a thing at one moment; nothing moves between the values.

- `HealthState` (`schemas.py`): `ok`, `degraded`, `unavailable`, `unknown`, `not_used` — runtime health on the dashboard.
- `LinkageState` (`services/preset_loader.py`): `explicit`, `auto`, `missing`, `unlinked` — how a preset found its registry model.
- `ModelStatus` (`db.py`): `discovered`, `registered`, `missing`, `deleted`.
- GPU placement modes (`scheduler.VALID_GPU_MODES`, `gpu_arg.GPU_MODES`): `auto`, `balanced`, `most-free`, `least-used`, plus `explicit` and `cpu` as parse results.
- Benchmark execution: `live` / `mock` (`services/benchmarks.py`).
- systemd `ActiveState` strings (`integrations/systemctl.py`, `adapters/vllm_systemd.py`): external vocabulary, read as-is.
