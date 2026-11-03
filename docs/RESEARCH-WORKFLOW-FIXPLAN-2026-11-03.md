# llmctl Research Workflow Fix Plan

Date: 2026-11-03  
Status: implementation and review follow-up on `fix/research-workflow-2026-11-03`; deferred acceptance checks remain (see Completion record)

Reviewed commit: `555f56c229f5ab7e7afb5236b143168cc7c14edd`

The three priorities are trustworthy benchmark comparisons, recoverable model switching, and a clearer research workflow. Implement them in that order: benchmark conclusions need sound measurements, and interface progress needs reliable operation state.

This is a handoff for a future implementation session. The review covered source code and isolated behavioral checks, not a live production switching exercise or visual usability study. Recheck the relevant implementation against this plan before editing; later changes may supersede individual findings.

## Evidence and verification baseline

The review used `graphify-out/graph.html`, which contains 222 community nodes and 1,598 aggregated edges. The relevant communities are:

| Area | Graph communities |
| --- | --- |
| Benchmarks | Benchmarks Service & API; TUI Benchmarks Screen; Settings & TUI Data Layer |
| Switching | Swap Jobs API; Run/Swap CLI Front End; Stop Serving Command; Lanes State Gathering |
| Interface | TUI App Shell & Test Isolation; TUI Presets Screen; Web UI Routes & Static Page |

The graph identifies paths to inspect, not proof of a defect. The file was already untracked during review; preserve it and do not assume it will be present in another checkout.

The full existing test suite completed with **1,493 passed and 1 skipped in 249.87 seconds**. The skipped module, `tests/test_preset_schema_parity.py`, requires unavailable `llm_models_config.schema`. A previous socket-permission failure disappeared when the tests ran with local socket access. The existing suite is green; the issues below require additional behavioral coverage.

The completed run used:

```bash
timeout --signal=INT --kill-after=15s 900s .venv/bin/python -m pytest tests -v --tb=short -o faulthandler_timeout=45 -o faulthandler_exit_on_timeout=true --junitxml=/tmp/llmctl-full-suite-20261103.xml > /tmp/llmctl-full-suite-20261103.log 2>&1
```

The log and XML are temporary artifacts and may not survive until the next session. Tests redirect registry/settings/state into temporary locations and disable live plugin discovery through `tests/conftest.py`; preserve that isolation. Do not restart or switch production models to validate this plan.

## 1. Make benchmark results suitable for scientific comparison

### Problem and evidence

- With the default prompt list and `concurrency=8`, an injected HTTP client observed **one request**, while the result recorded concurrency as eight. `_run_live` creates one task per prompt; `_resolve_prompts` defaults to one prompt.
- The TUI throughput comparison displayed a green `+10.0` for a live result at 60 tokens/s against a synthetic baseline at 50 tokens/s. The delta helpers check numeric availability, not comparability.
- When the server omits usage, `_stream_one` substitutes whitespace word counts without recording the token-count source per sample.
- The request schema defaults `require_live` to false. The TUI explicitly opts into strict live behavior, but other callers can still receive a successful synthetic result when no endpoint resolves. An endpoint that resolves but fails during the request already produces a failure record; preserve that distinction.
- Records retain useful samples and parameters, but do not capture a deliberate immutable experiment manifest. `rerun` reconstructs a request using selected parameters and resolves its target again; that is not a guarantee of reproducing the original environment.

Primary files: `llmctl/services/benchmarks.py`, `llmctl/schemas.py`, `llmctl/db.py`, `llmctl/api/routes_benchmarks.py`, `llmctl/cli.py`, `llmctl/tui/_data.py`, and `llmctl/tui/screens_benchmarks.py`.

### Implementation steps

- [ ] **1A — Correct execution mode and workload semantics.** Make ordinary benchmark requests require live execution across CLI, API, TUI, and rerun. Keep synthetic execution explicit and clearly labeled. Preserve deprecated flags with documented compatibility behavior. Treat missing endpoints as failures rather than fabricated measurements.
- [ ] Add a validated workload-size or repetition setting separate from concurrency. Define concurrency as the maximum number of simultaneous requests, record both requested and achieved concurrency, and flag underfilled workloads. Keep the total workload fixed across a sweep so changing concurrency does not silently change the experiment. Do not duplicate prompts implicitly without recording the policy.
- [ ] **1B — Record measurement provenance.** Store whether token counts came from server usage, a matching tokenizer, or an estimate. Label estimates in every result surface. Avoid inventing a token for an empty response. Separate endpoint health-check latency from generation TTFT. Define TTFT as the first relevant generated content and document handling of reasoning/tool chunks.
- [ ] Preserve successful and failed request outcomes within a partially failed batch. Report request counts, errors, wall-clock aggregate throughput, and per-request timing. Add explicit warm-up and repetition policies; exclude warm-ups from reported measurements. Compute latency/TTFT percentiles with sample counts and an insufficient-sample indication.
- [ ] **1C — Capture an immutable manifest.** Store a versioned, allowlisted snapshot of resolved model/served identity, available model revision, engine version, launch configuration, hardware identity, workload hash, and effective generation parameters. Record unknown fields as unknown. Distinguish runner-host GPU telemetry from serving-host telemetry; do not attribute local GPU readings to a remote endpoint. Exclude credentials and secret environment values.
- [ ] Preserve legacy records without fabricating missing provenance. Add export of results plus manifest. Make rerun show configuration drift and reuse the stored workload; fail clearly if the original target cannot be resolved. A new run must have its own identity and a link to its source run.
- [ ] **1D — Gate comparisons.** Put comparison eligibility in a shared service/helper rather than only in TUI rendering. Never present live-versus-synthetic, failed, or incompatible benchmark kinds as an improvement. Compare workload, generation settings, measurement method, and hardware context; explain differences. Allow the intended experimental variable, such as model or engine, to differ. Do not require identical manifests for a model comparison.

### Acceptance and tests

- [ ] A controlled server/barrier test proves eight overlapping requests when the workload supports concurrency eight; an undersized workload records its lower achieved concurrency and a warning.
- [ ] A missing endpoint fails in CLI, API, and TUI live mode; explicit synthetic mode remains deterministic and labeled.
- [ ] Estimated counts and empty responses cannot silently become precise token measurements.
- [ ] A partially failed batch retains successful samples and reports its failure rate.
- [ ] Synthetic/live and incompatible-kind comparisons never receive green improvement indicators. Compatible model comparisons still work.
- [ ] Manifest/export round-trips preserve effective settings; rerun detects drift; old rows remain readable with unknown provenance.

Extend `tests/test_benchmarks_systemd.py`, `tests/test_benchmarks_kinds.py`, `tests/test_cli_bench_synthetic.py`, and `tests/test_tui_benchmarks.py`. Add focused tests for manifest migration/export and comparison eligibility where appropriate. Use fake endpoints and injected clocks; do not depend on a loaded GPU model.

Deliver 1A/1B before the larger manifest and comparison work. Keep each behavioral change reviewable.

## 2. Coordinate and recover model switching across processes

### Problem and evidence

`SwapJobs` stores jobs and a `threading.Lock` in memory. Two independently constructed managers accepted simultaneous jobs in an isolated check; a fresh manager could not retrieve the first manager's job. The API creates a per-app registry. The TUI also invokes switching functions directly.

Existing previews, readiness checks, and engine-specific rollback are valuable. The gap is durable operation ownership across interfaces and process restarts. This is a demonstrated coordination limitation, not evidence of an observed production collision.

Primary files: `llmctl/services/swap_jobs.py`, `llmctl/api/routes_swap.py`, `llmctl/api/app.py`, `llmctl/cli_swap.py`, `llmctl/services/lane_swap.py`, `llmctl/services/session_swap.py`, `llmctl/services/stop_serving.py`, `llmctl/services/vllm_orchestrator.py`, `llmctl/services/sessions.py`, and `llmctl/tui/screens_swap.py`.

### Implementation steps

- [ ] **2A — Establish one operation boundary.** Introduce a shared operation service for CLI, TUI, and API. Inventory all llmctl paths that mutate the same managed resources, including direct start/stop/restart paths, so an advanced command cannot silently bypass coordination. Preserve existing low-level engine adapters and rollback logic.
- [ ] Start with a conservative host-wide interprocess lock for managed GPU lifecycle changes. Acquire ownership before gathering the actionable state, then revalidate the preview before stopping anything. Define the lock's configured scope and file permissions. Read-only status operations must remain available.
- [ ] Avoid nested-lock deadlocks: the current API launches the CLI as a child process. Choose a single execution owner, or explicitly transfer operation ownership to that child. Do not independently acquire the same non-reentrant lock at both levels. External tools/systemd remain outside llmctl's lock, so probe actual state before mutations.
- [ ] **2B — Persist operation state.** Add a versioned operation record with stable ID, target, previous state, owner identity, phase, timestamps, and recovery outcome. Persist intent before side effects and outcomes afterward. Keep logs bounded and sanitized. Separate the fact that an operation failed from the fact that a model is now serving after restoration.
- [ ] Use explicit phases such as preparing, stopping, loading, checking readiness, restoring, completed, and needs attention. Add request idempotency keys: repeating a key with the same request returns the existing operation; reusing it for a different request is refused.
- [ ] **2C — Recover conservatively.** On process restart, reconcile unfinished records against process identity, units, and endpoints before admitting conflicting work. Account for a child that outlives its API parent and for PID reuse. Do not infer that an operation ended solely because its parent or lock disappeared.
- [ ] Reattach to a verified surviving operation where possible. Otherwise record an interrupted or needs-attention outcome with concrete recovery guidance. Do not blindly replay stop/start side effects, claim restoration before readiness, or add cancellation by killing a worker without cleanup semantics.
- [ ] **2D — Share structured progress.** Keep existing route compatibility where practical, but expose stable operation IDs and structured phases to all interfaces. Preserve recovery history after API restart and browser refresh. Make status and logs available without waiting for the mutating operation to finish.

### Acceptance and tests

- [ ] Separate OS processes cannot execute conflicting GPU mutations concurrently. Cover API-versus-CLI, CLI-versus-CLI, and stop-versus-start, not just threads sharing one object.
- [ ] A repeated idempotent request cannot start a second switch.
- [ ] Simulated interruption at each phase retains a discoverable operation and yields a reconciled outcome. A surviving child prevents a conflicting new operation.
- [ ] Readiness and rollback tests continue to pass. Failed restoration remains visibly different from successful restoration and from successful target startup.
- [ ] Restart/reattach tests use temporary state, fake units, and fake endpoints without changing live services.

Extend `tests/test_swap_jobs.py`, `tests/test_api_swap.py`, `tests/test_api_swap_stop.py`, `tests/test_api_ui_job_reattach.py`, `tests/test_lane_swap.py`, `tests/test_session_swap.py`, `tests/test_cli_swap.py`, and `tests/test_tui_swap.py`. Add dedicated multiprocess ownership and interrupted-operation recovery tests.

## 3. Organize the interface around research tasks

### Problem and evidence

The TUI exposes eleven peer screens. Its home view emphasizes inventory counts and a "Safe mode (label only)" card. The web swap page has useful serving details, but no model search/filter controls, and its job view primarily shows status plus raw logs.

The design priority is to answer four questions with less navigation: what is ready, what should I run, what will switching interrupt, and how does the resulting experiment compare? These are source-level design findings; validate the proposed layout with rendered screens and task walkthroughs during implementation.

Primary files: `llmctl/tui/app.py`, `llmctl/tui/screens_dashboard.py`, `llmctl/tui/screens_swap.py`, `llmctl/tui/screens_models.py`, `llmctl/tui/screens_benchmarks.py`, `llmctl/tui/_data.py`, and `llmctl/api/static/{index.html,app.js,app.css}`.

### Implementation steps

- [ ] **3A — Establish the information hierarchy.** Build the landing view around Serving now, Current operation, and Recent experiments. Keep serving identity, endpoint, readiness, and observation time together. Move inventory totals and configuration flags to secondary detail. Retain advanced screens, existing shortcuts, and truthful state distinctions.
- [ ] Distinguish catalog availability, loading, confirmed readiness, degradation, and stale observations. Show text/icons alongside color. A last successful observation must not appear fresh after refresh failures.
- [ ] **3B — Improve model selection.** Add text search and engine/state filters. Use a detail panel for fit information, launch settings, endpoints, and recent benchmark evidence. Display unknown fit information honestly. Keep selection anchored to a stable model/target ID across refreshes and filters, not a mutable row number.
- [ ] Carry the selected identity and configuration through select → preview → run → benchmark → compare. Revalidate before acting and keep the consequences of switching visible in the preview. Do not obscure advanced settings or silently change presets.
- [ ] **3C — Explain long operations.** Consume the structured phases from plan 2. Show current phase, elapsed time, readiness evidence, and any restoration outcome above expandable logs. Label historical load times as historical rather than promising an exact completion time. Reattach after refresh/restart.
- [ ] **3D — Improve experiment inspection.** Use the manifest and comparison eligibility from plan 1. Give each result an inspection view with sample counts, distributions, provenance, errors, and comparison differences. In the benchmark table, make Enter inspect the record and give rerun a separate clearly labeled action; update keyboard help and tests together.
- [ ] Implement the richer research workflow in the TUI first. Keep the web interface focused on serving, searchable selection, and operation progress unless a later scope explicitly adds a full web experiment workspace. Reuse shared state and comparison logic across surfaces.

### Acceptance and tests

- [ ] With a fixture catalog of at least 150 entries, an operator can locate a model, understand what switching stops, launch it, verify readiness, benchmark it, and inspect a comparison without losing selection/context.
- [ ] Validate keyboard-only use at an 80-column terminal size and the web interface at a narrow viewport (for example 375 pixels). Essential actions and status remain reachable without clipped text or focus loss.
- [ ] Cover empty, loading, stale, failed, restored, and needs-attention states with rendered examples. Color is never the only status signal.
- [ ] Refresh/filter changes cannot retarget an action to a different model. Enter on a benchmark inspects it without launching another workload.
- [ ] Preserve existing destructive-action guards, preview disclosure, and browser reattachment behavior.

Extend the existing TUI benchmark, chrome/cards, refresh robustness, swap, and action-guard tests, plus `tests/test_api_ui.py`, `tests/test_api_ui_actions.py`, and `tests/test_api_ui_job_reattach.py`. Add rendered layout checks and a short task walkthrough; DOM-string assertions alone do not establish usability.

## Implementation handoff

1. Read applicable repository instructions, inspect the working tree, and confirm whether any findings have already been fixed. Preserve existing user changes and the graph output.
2. Start with plan 1A: reproduce the underfilled-concurrency and implicit-synthetic behaviors using isolated tests, then change the service/schema and callers together.
3. Complete benchmark measurement corrections before adding the manifest and comparison presentation. Establish shared operation ownership before adding persistence/recovery, then expose progress to the UI.
4. Prototype plan 3 layouts using fixture data while backend work is underway; integrate real progress and comparison indicators only after their contracts are reliable.
5. Run the narrow relevant tests after each substantive change. Run the full suite after integration, with local socket access and test isolation equivalent to the verified baseline. Record actual outcomes and unresolved gaps here.
6. Add migrations and user-guide changes alongside behavior changes. Keep changes focused; do not introduce unrelated refactors or a frontend framework migration. Commit only when requested and never push unasked.

## Completion record

Implemented 2026-11-03 on branch `fix/research-workflow-2026-11-03`, one commit per slice:

| Slice | Commit | Contents |
| --- | --- | --- |
| 1A | `7608741` | strict-live default (`require_live=True`; `False` keeps the deprecated mock fallback), validated `repetitions` workload setting recorded via `prompt_count`/`total_requests`/`repetition_policy`, requested+achieved concurrency with underfill warnings, fixed-workload sweeps, rerun replays the stored workload and preserves its mode |
| 1B | `dfcbc7e` | per-sample + record token-count provenance (`server_usage`/`estimate`/`mixed`/`synthetic`), zero-token empty responses, TTFT = first content (reasoning/tool chunks → `first_token_ms`), health latency separated from TTFT, per-request failure retention with request counts, p50/p90/p99 with sample counts + insufficient-sample flag, warm-up exclusion |
| 1C | `cc68460` | versioned allowlisted manifests (`services/benchmark_manifest.py`, `BenchmarkRecord.manifest`, in-place migration), unknown-as-unknown, runner vs serving host telemetry split, env/credentials excluded, export (API + CLI `--export`), rerun links its source and records drift; legacy rows read as unknown provenance |
| 1D | `ccc67d0` | `services/benchmark_compare.comparison_verdict` gates deltas (failed runs, live-vs-synthetic, kind, workload hash/size, generation settings, token-count method, serving hardware; model/engine differences allowed); TUI renders `n/c` with a per-row explanation |
| 2A | `e05e74e` | `services/op_lock.py`: host-wide flock (0600, `LLMCTL_OP_LOCK`), reentrant per thread, cross-thread/process `LockBusy` with holder identity; wrapped CLI `run` (whole mutation incl. rollback), swap/session service cores, `SessionService` start (non-dry-run)/stop/restart/start_unit, TUI unit stop/restart; API maps `LockBusy` → 409; the API-spawned CLI child owns the lock |
| 2B/2C/2D | `89ef7ca` | `OperationRecord` + `services/operations.py`: intent before side effects, explicit phases, terminal states incl. `needs_attention` (failed rollback stays distinct), bounded sanitized output, idempotency keys (same request returns the operation; different request refused), API→CLI child handover via env, conservative reconcile at SwapJobs construction (pid + start-time + cmdline verification; surviving children block conflicting work; nothing replayed); routes expose `operation_id`/phase/recovery |
| 3A–3D | `2ad015e` | dashboard = Serving now / Current operation / Recent experiments (icon+text states, historical load labeled, inventory+flags demoted); models search (`/`) + engine filter + detail panel with unknown-labeled fit info and latest benchmark; id-anchored selection on models and swap screens; swap screen operation panel; TUI swaps write operation records; benchmarks Enter=inspect (samples, percentiles, provenance, manifest, drift, comparison verdict) with `r` as the separate re-run action; web filter box + job phase (app.js line cap raised to 470, dated) |

**Tests:** full suite green after implementation (1,578 passed, 1 skipped — `test_preset_schema_parity` still needs the unavailable `llm_models_config.schema`). New modules: `test_benchmark_manifest.py`, `test_benchmark_compare.py`, `test_op_lock.py` (incl. real multiprocess exclusion and no-interleave proofs), `test_operations.py` (idempotency, interruption at each exit class, PID-reuse, surviving child, API restart), `test_tui_landing.py`, `test_tui_models_select.py`; benchmark/TUI suites extended accordingly.

**Migration/compatibility decisions:** `require_live` default flipped with `False` as the documented legacy escape hatch; `bench --dry-run` still warns and maps to synthetic; new columns (`benchmarks.manifest`, `operations` table) are added in place, legacy rows preserved with unknown provenance; `SwapJobs.start()` kept its historical return contract (`start_with_operation` is the extended entry point).

**Known gaps (deliberate):**
- Manifest `engine.version` and model revision are recorded as `unknown` (no probing wired).
- Warm-up is settable via CLI/API schema, not yet in the TUI launch modal.
- The 80-column keyboard-only walkthrough, 150-model fixture walkthrough, and 375px web viewport checks are covered only indirectly by navigation/render tests, not as a scripted walkthrough suite.
- Web phase display is one line (current phase), not a full phase timeline.
- Systemd-triggered restarts outside llmctl remain outside the lock by design; mutating paths probe state before acting.

### Review follow-up on 2026-11-03

The independent review at `124b9a1` reproduced four integration gaps despite the passing suite. The following corrections are now implemented:

- [x] Hold the TUI operation lock across the entire swap, including the interval between pausing sessions and starting the replacement. Finalize exceptions and cancellation as interrupted, or needs attention when restoration was interrupted. A busy lock must not create an unfinished record.
- [x] Persist browser job IDs and child exit codes with operation records. A new API instance discovers unfinished jobs and serves the original polling URL, including output, phase, recovery, and completion. Reconcile recovered owners again during polling so a child that subsequently dies does not appear permanently running. Idempotent responses include the existing job ID.
- [x] Forward failed-restoration labels from the CLI lane/session callers and TUI failure formatter into the operation recorder. These outcomes persist as `needs_attention` with `recovery.not_restored` instead of a plain failure with empty recovery data.
- [x] Share generation defaults between benchmark execution, stored parameters, manifests, and workload hashes. New default requests record `max_tokens=256` and `temperature=0.0`; exports retain those values. Explicitly supplying those same defaults no longer changes the workload hash.

Regression coverage is in `tests/test_research_swap_regressions.py`, `tests/test_operations.py`, and `tests/test_benchmark_manifest.py`. The initial 12 regression cases all failed before the fixes. The related switching/API/TUI/benchmark run passed 151 tests; additional interruption, migration, and invalid-input checks passed in a subsequent 59-test focused run.

Final verification: **1,595 passed, 1 skipped in 355.08 seconds**, with **85% overall line coverage**. The same optional schema-parity dependency remains unavailable. Ruff passed on all changed Python files, and `git diff --check` passed. The coverage run reported 1,998 warnings, chiefly unclosed SQLite connections; connection-resource cleanup remains separate follow-up work. The first coverage attempt selected individual importable modules and failed during collection in SQLAlchemy; rerunning with directory-based `--cov=llmctl` completed successfully. Startup database/config/state were redirected to temporary locations.

Temporary verification artifacts: `/tmp/llmctl-research-followup.log`, `/tmp/llmctl-research-followup.xml`, and `/tmp/llmctl-research-followup-coverage.xml`. These paths may not survive a later session. No production service change was made during this follow-up.

Compatibility: nullable `operations.job_id` and `operations.job_exit_code` columns are added by the existing migration mechanism. Older operation history is preserved. Job IDs lost before this change cannot be reconstructed from old rows that never stored them. Existing benchmark manifests are immutable; missing provenance in historical records is not retroactively filled in. The deferred items above remain open.
