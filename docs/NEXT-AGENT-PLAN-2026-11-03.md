# Next-agent implementation and deployment plan

Prepared November 3, 2026. Repository: `/home/yannik/Projects/llmctl`.

## 1. Objective and instructions

Make the existing llmctl stack more cohesive and reliable by extending what is
already implemented. Treat source and fresh runtime observations as authoritative.
Do not restart the architecture investigation or assume that lifecycle management,
GPU scheduling, routing, recovery, benchmarking or Mac integration is missing.

Start by reviewing the existing uncommitted fixes and deployment difference. Then
work through the packages below in order, with small reviewable changes. Prepare
deployment independently of the remaining engineering work; a pending deployment
approval need not stop safe source changes or isolated testing.

Honor the user's operating instructions:

- Read before editing; preserve unrelated changes and match surrounding code.
- Never read or expose `~/.secrets`, `.env*`, `~/.ssh`, `~/.aws`, API keys or
  process environments. Do not copy credentials or private logs into reports.
- Commit only when explicitly asked; never push unasked. This plan is not a
  commit, installation, restart, remote-access or destructive-action approval.
- Obtain approval before disruptive/destructive work: live model swaps, failure
  injection, service restarts, package replacement in the running stack, model
  deletion, configuration migrations and tests that preempt workloads.
- Continue reversible source changes and isolated tests without repeated approval
  questions. Prepare a concrete diff, validation and rollback plan before asking
  for deployment approval.
- Do not introduce another gateway, registry, supervisor, scheduler daemon or
  watchdog. Do not consolidate the vLLM environments by upgrading them globally.

For device work, read
`/home/yannik/.local/share/three-machine-monitor/SYSTEMS_KNOWLEDGE.md` and the
relevant `discovery/` report. These dated references do not grant new access.
Use configured private routes; do not discover or administer unrelated devices.

## 2. Read these first

1. [Audit and capability map](CONTROL-PLANE-AUDIT-2026-11-03.md).
2. [Operational workflow and rollout boundaries](OPERATIONS.md).
3. The current `git diff` and the new regression files listed below.
4. [Research-workflow fix plan](RESEARCH-WORKFLOW-FIXPLAN-2026-11-03.md), especially
   its implementation record. Several originally unchecked items were subsequently
   implemented; do not use checkbox state alone to decide what is missing.
5. `graphify-out/GRAPH_REPORT.md` for navigation. Some edges are inferred; verify
   relationships in source. Preserve the entire existing untracked directory.

The audit explains the implemented/partial/observed distinction. Keep that
distinction in subsequent reports, especially for rollback and remote health.

## 3. Exact handoff state

### Source and deployment

At handoff, repository HEAD is `b72b874`, with the prior agent's uncommitted
changes described below. Recheck status before editing; another agent or the
user may have progressed since this document was written.

The workstation shim `/home/yannik/AI/bin/llmctl` executes
`/home/yannik/AI/.venv/bin/llmctl`. The installed package is a non-editable
Git install pinned to **`16dc60f612849fbc4345c69db584effb58ed2016`**. The plugin
at `/home/yannik/AI/packages/llmctl-stack/pyproject.toml` pins that same revision.
Both deployed and checkout versions say **0.3.0**.

The checkout already contains a shared GPU operation lock, durable operation
records/recovery and improved benchmark provenance. These are absent from the
installed generation. Selected installed files matched the old pin; installed
`services/op_lock.py` and `services/operations.py` were absent. There were 47
changed files between the pin and HEAD before this audit's additional fixes.

**Import resolution must be checked outside the checkout.** Running the deployed
Python executable from this repository can import the local source and falsely
suggest production is current. Installation metadata, an outside-checkout import
path, selected source comparisons and running-process evidence should agree.
An existing database table is not proof the running API uses its new feature.

### Changes already implemented, tested and left uncommitted

| Files relative to this repository | Completed change |
| --- | --- |
| `llmctl/config.py` | Include optional second engine in shared managed roles; reserve its role names against shadowing. |
| `llmctl/adapters/vllm.py` | Discover/probe configured vLLM roles, including engine 2, while excluding other runtimes. |
| `llmctl/services/engine_load.py` | Use shared roles instead of separately appending engine 2. |
| `llmctl/services/validate.py` | Respect configured user scope when checking managed-unit activity. |
| `llmctl/services/systemd.py` | Generate units in configured user/system scope, with appropriate install target and User field. |
| `llmctl/api/routes_sessions.py` | Pass application settings consistently to planning and other session services, matching start/stop/restart. |
| `tests/test_two_engine_inventory.py` | Seven cases for inventory, discovery, role conflicts, scope and generation. |
| `tests/test_api_plan_settings.py` | API preview and dry-run start use the same injected endpoint. |
| `tests/test_validate_service.py` | Test double accepts the existing runner's user-scope argument; assertions unchanged. |
| `README.md`, `docs/USER-GUIDE.md`, `CHANGELOG.md` | Document the above and replace obsolete single-engine-only descriptions. |
| `docs/CONTROL-PLANE-AUDIT-2026-11-03.md`, `docs/OPERATIONS.md` | Audit evidence, remaining gaps, preservation boundaries and operational workflow. |

Do not reimplement these changes or discard them to get a clean tree. Review
them, repair only demonstrated defects, and carry them into the selected release.

### Verification already performed

- Reproduced each fixed defect before its fix with isolated tests.
- Related tests: **111 passed**.
- Full suite: **1,671 passed, 1 skipped**, 2,039 warnings, 388.91 seconds;
  **86% statement coverage**. The existing skipped module requires the optional
  `llm_models_config.schema` dependency, absent from the environment.
- `ruff check .` and `git diff --check` passed.
- Full-suite testing used temporary home/data/state/model caches, disabled stack
  plugins, and inert service-command shims. Test-created example alias overlays
  were removed. No installed package, live configuration or workload was changed.
- Optional temporary evidence may still exist at `/tmp/llmctl-audit-suite.txt`
  and `/tmp/llmctl-audit-coverage.json`; do not rely on their persistence.

### Last passive runtime observations

At 18:31 Pacific on November 3, router `qwen27b` and served model `qwen3.8-27b`
remained healthy. API/router/vLLM process IDs were unchanged from the earlier
observation; their restart counters remained zero. Recheck rather than relying
on those process IDs for control operations.

- Two RTX 5090s, 32,607 MiB each; substantial VRAM allocation on both.
- Main vLLM system unit on :8003; configured second engine is an inactive user
  unit on :8010. Inactive is expected until a workload uses it.
- vLLM HTTP version was 0.30.0; generic `vllm-serve` environment was 0.25.1.
  `vllm-muse` and `vllm-laguna` are separate intentional environments.
- API :8088, router :4100, LiteLLM at `172.18.0.1:4000`; standalone llmctl
  gateway :9000 disabled intentionally.
- Router default `qwen27b`, idle revert 3,600s, poll 30s, minimum hold 300s;
  failure cooldown, leases, broken-lane guards and rollback already implemented.
- Reconcile timer runs every 30s. Existing healer is enabled in `act` mode,
  uses two consecutive observations, 120s settle time and a two-action/six-hour
  per-target cap. Do not install another recovery service.
- Mesh selected Tailscale for M5 and LAN for M1 serving lanes. Two LM Studio
  forwarded lanes were unavailable; their intended state was not established.
- Five legacy presets existed under `~/.config/llm-models`, no canonical YAML
  files under `~/.config/llmctl/presets`. Do not migrate them automatically.

## 4. Work order and deliverables

| Package | Priority | Dependency | Deliverable |
| --- | --- | --- | --- |
| A. Confirm handoff and release candidate | P1 | None | Reviewed fixes, current deployment delta, isolated package/plugin compatibility evidence. |
| B. Coordinate existing lifecycle owners | P1 | A's source review | Shared ownership contract and tested integration across actual mutation paths. |
| C. Protect active requests from idle reversion | P1 | B's ownership contract | Per-backend activity tracking and race-safe idle decisions. |
| D. Deploy a bounded release | P1 | A; B/C if included in that release; approval | Verified package/process parity and rollback record. Can deploy A separately if its limits are explicit. |
| E. Consolidate configuration and diagnostics | P2 | Stable behavior from A–C | Drift reporting and explicit ownership without mass migration. |
| F. Generalize hybrid policy where needed | P3 | B/C plus verified workload requirements | Small optional policy extension using existing router/slots, preserving defaults. |

Do not bundle every package into one release. In particular, an inventory/API
release does not need to wait for a general scheduler, but must not be described
as solving router/healer coordination.

## 5. Package A — confirm the handoff and prepare deployment

1. Inspect Git status/diff in llmctl and, before editing them, the router and AI
   repositories. Identify their actual Git roots and applicable instructions.
   Do not assume a nested directory is an independent repository.
2. Read the completed fixes and tests. Check all `ManagedUnitsConfig.roles()`
   consumers for implications of the added role. Preserve the current unit names,
   ports and scope; do not reinstall existing units to change inventory.
3. Recheck the installed package outside the checkout, plugin pin, API service
   working directory/executable, and selected installed source files. Do not
   print full process command lines or environment settings.
4. Review the complete old-pin-to-candidate delta, including DB migrations,
   child-process operation ownership and benchmark serialization. Keep old
   sessions, profiles and benchmark rows readable; do not fabricate provenance.
5. Test the stack plugin with the candidate in an isolated environment. Include
   lane import, status/router screen, pause/resume, companion registration and
   post-start hook interfaces. Do not execute real companion or post-start actions.
6. Build or otherwise identify a reproducible candidate through existing package
   tooling. Record source revision plus patch identity or artifact digest; do not
   refer only to “0.3.0” or a moving `main`. Record the plugin compatibility choice.
   If a Git pin requires a new commit, ask before committing; do not invent a pin.
7. Prepare the exact deployment and rollback steps from package D. Do not run
   the production installer yet. Continue B/C while approval is pending.

Acceptance: an outside-checkout check distinguishes installed and candidate
code; existing fixes remain green; plugin compatibility is demonstrated; the
release contents and limitations are explicit and reproducible.

## 6. Package B — coordinate the existing lifecycle owners

### Confirmed problem and source entry points

llmctl's new lock coordinates its own CLI/TUI/API/session/swap mutations.
llm-router has a separate process-local asyncio lock and restarts systemd
directly for profile switches. Healer actions also include direct systemd
operations. These do not all share llmctl's ownership boundary. The audit found
a potential race in source, not evidence of an observed collision.

Read these paths together:

- llmctl: `llmctl/services/op_lock.py`, `operations.py`, `swap_jobs.py`,
  `sessions.py`, `lane_swap.py`, `session_swap.py`, `stop_serving.py`,
  `lane_switcher.py`; `llmctl/cli.py`, `cli_swap.py`, API swap routes.
- Router: `/home/yannik/AI/llm-router/llm_router/switcher.py`, `app.py`,
  `idle.py`, `config.py`, plus the actual configured start-script paths.
- Healer: `/home/yannik/AI/heal/llm-heal.py`, its documented gate/action logic.
- Plugin: `/home/yannik/AI/packages/llmctl-stack/llmctl_stack/router_lanes.py`
  and existing command integration.

### Implementation steps

1. Produce a short call/ownership table covering: direct CLI run, router-delegated
   CLI run, API child run, TUI mutations, session stop/restart, router requests,
   idle revert, healer recovery and command-lane launch scripts. Identify where
   each path can stop a service, evict Ollama, rewrite launch state or restart.
2. Write a failing integration test for at least one direct-mutation/router race
   using fake engines and real subprocess synchronization. Do not rely only on
   two asyncio tasks in one process.
3. Choose the smallest common ownership mechanism using existing components.
   Prefer extending the current operation boundary or routing cooperating actions
   through an existing owner; no new lock daemon, Redis or scheduler framework.
4. Define who owns a delegated operation for its entire mutation/rollback phase.
   Document acquisition order and how progress, cancellation, timeout and
   surviving-child identity are represented. A caller must not hold an exclusive
   lock while waiting for an HTTP/subprocess callee that needs that same lock.
5. Revalidate real state under ownership before taking action. Preserve
   idempotency, role resolution, configured lock paths, leases, heal-broken
   refusals, cooldowns, minimum holds, stop-survivor handling and rollback details.
6. Integrate cooperating router/healer/script paths incrementally. Use bounded
   refusal/defer semantics and useful holder information. Keep read-only status
   responsive while a large model loads. Do not block the async event loop with
   synchronous flock waits.
7. Be explicit about what remains outside coordination: an administrator's raw
   systemctl action and systemd's own autonomous restart behavior are not magically
   covered by an application lock. Preserve/recheck these boundaries rather than
   claim universal exclusion.

The current lock has thread ownership semantics. If using async worker threads,
do not acquire in one arbitrary worker and release/reenter from another. Test the
chosen bridge rather than assuming thread-local reentrancy works across awaits.
Do not add an unauthenticated “skip lock” header or environment bypass to solve
delegation. Operation identity and authorization must remain trustworthy.

### Acceptance tests

| Scenario | Required result |
| --- | --- |
| Direct mutation overlaps router switch on the same cards | Only one mutation proceeds; the other waits within budget or refuses clearly. |
| CLI delegates to router | Completes without recursive-lock deadlock; one logical owner/outcome. |
| API parent dies, CLI child survives | Conflicting work remains refused; status can reattach to verified child identity. |
| Owner dies or PID is reused | Stale ownership is reconciled safely; no blind replay or wrong-process control. |
| Healer meets an operator operation or active lease | Defers without overriding intentional state. |
| A start fails and rollback is underway | Ownership remains held through restoration; failures retain `needs_attention` evidence. |
| Custom lock/data directory | CLI, API, children and router integration agree on the same boundary. |
| Busy operation with status requests | Read-only health/status remains bounded and available. |

Extend `tests/test_op_lock.py`, `test_operations.py`, `test_research_swap_regressions.py`
and the router's `test_switch_lock.py`, `test_admin_switch.py`, lease, rollback
and cooldown tests. Preserve conservative host-wide serialization initially;
per-GPU concurrent mutation is a later optimization, not a prerequisite.

## 7. Package C — make idle reversion safe for active requests

### Confirmed problem

`IdleRevert.touch()` records arrival time. It does not track completion, open
streams or active requests, and the current watcher is not a full per-backend
multi-slot policy. A long generation or direct-to-engine traffic can be mistaken
for inactivity. The current one-hour setting reduces frequency; it is not a
proof of safety.

Start in `/home/yannik/AI/llm-router/llm_router/{idle,app,proxy,admission,switcher}.py`
and `tests/test_idle_revert.py`, `test_proxy_stream_status.py`, `test_admission.py`,
`test_current_backend.py`, `test_switcher_extra_vllm.py`.

### Implementation steps

1. Add failing fake-clock tests for a response longer than the idle timeout and
   for a request arriving between idle evaluation and switch ownership acquisition.
2. Track activity per concrete backend/serving instance or resource footprint,
   not only a global last-request time. Define how aliases resolving to the same
   serving instance share accounting and how two slots remain independent.
3. Protect accepted/queued work, including time spent in admission. Increment
   at the appropriate accepted-request boundary and decrement exactly once.
   Completion or cancellation should update the idle baseline.
4. For streaming, retain activity until the response body is exhausted or closed.
   Returning a StreamingResponse is not completion. Handle disconnects, upstream
   exceptions, timeouts, rejection before forwarding and cancellation in `finally`.
5. Recheck activity and current serving identity under B's ownership immediately
   before stopping an engine. Make request admission/activity registration and
   idle mutation obey a coherent ordering so the check cannot race a new request.
6. Preserve pause, lease, broken-lane, cooldown and hold guards. An idle revert
   must not get an operator-force exemption accidentally.
7. Handle traffic bypassing the router explicitly. Existing engine running/waiting
   metrics can veto reversion; unknown load is not proof of idleness. Decide and
   document a conservative behavior for engines lacking reliable activity
   visibility, using existing leases or explicit policy where necessary.
8. Keep the 3,600s default and existing default-model behavior. Expose minimal
   diagnostic fields through existing status/metrics only if they help explain
   why reversion is deferred. Do not log request bodies.

Acceptance: no request or open stream is cut by an automatic idle decision;
counters recover on cancellation/error; a quiet non-default lane still reverts;
default-only operation is unchanged; two-slot and direct-traffic limitations are
tested or explicitly refused rather than silently assumed safe.

## 8. Package D — controlled deployment and rollback

This is an approval-gated package. Prepare everything below first. Approval to
deploy the control-plane package is not approval for live model-swap experiments.

### Before asking for approval

1. Identify exact candidate artifacts/pins, included fixes, remaining gaps and
   plugin compatibility. Determine whether router/healer changes are included
   and which processes would need refreshing.
2. Inspect migration compatibility and prepare a consistent SQLite backup method
   using the SQLite backup API. Do not copy only the main file while WAL writes
   continue. Identify configurations to preserve without reading credentials.
3. Verify the existing jobs/operations are quiescent before restarting the old
   API. Its older in-memory job behavior must not be treated as durable.
4. Prepare the rollback package, previous plugin pin and a recovery procedure
   for database changes. Do not overwrite a live database with a backup while
   processes are writing, or discard new operational records without a decision.
5. State exact service impact, expected interruption, verification and rollback
   triggers. Request approval for this concrete release, not blanket fleet access.

### After approval

1. Capture a fresh passive baseline and take the planned recoverable backups.
2. Install through the existing stack package process; avoid an unreviewed
   production editable install or unrelated dependency upgrades.
3. Refresh only affected control processes. A llmctl inventory/API-only release
   should not require restarting vLLM, Ollama or Mac engines. Router/healer source
   changes require their own deliberate rollout consideration.
4. Verify installed import path and source/artifact identity outside the checkout;
   check CLI/API/plugin parity. Validate health, second-engine visibility and
   read-only status. Keep dry-run API parity checks on isolated records/state.
5. Monitor for DB migration errors, missing plugin lanes, auth/route regressions,
   deadlocks and unexpected serving changes. Roll back on demonstrated regression
   using the prepared process, preserving incident evidence.
6. Record what was actually installed/restarted and the final serving state.
   Keep deployment and performance conclusions separate.

### Separately approved live acceptance

Schedule controlled tests for one-card coexistence, TP=2 takeover, failed-start
rollback, stop/restart readiness, long streams, competing clients and recovery
after failure. Use known small/appropriate test workloads, protect existing
leases and restore the original serving configuration. Mac sleep/wake or route
failure tests require explicit scope; do not restart an unavailable Mac lane
merely because a probe failed. Preserve benchmark artifacts and measurement dates.

## 9. Package E — consolidate facts and improve diagnosis

Keep this incremental and mostly observational until a real mismatch is shown.

1. **Configuration ownership:** distinguish YAML presets, SQL runtime profiles,
   router backend references, rendered launch files and engine installation
   records. They have different consumers; do not merge them merely because all
   contain model-related fields. Add missing cross-reference/drift checks to
   existing doctor/validate/plugin facilities rather than create a new registry.
2. **Legacy presets:** current loading gives legacy files precedence. If migration
   is justified, first produce an effective-value and collision report, with
   explicit duplicate/alias behavior and rollback. No move/delete before approval.
3. **Scope and identity:** compare configured unit scopes to actual managers,
   especially generic plugin lanes. Preserve the valid `vllm-tp.service` alias.
   Do not rewrite IDs, null host fields or names without evidence of harm.
4. **Runtime versions:** show serving version separately from launch-binary
   version where relevant. Existing environments may intentionally serve different
   presets. Preserve the working launchers and hardware compatibility choices.
5. **Router failures:** seven 502s were observed historically; recheck current
   counters and correlate bounded, sanitized operational evidence before changing
   retry/backoff. Never print raw prompts, credentials or full command lines.
6. **Remote integration:** verify mesh-state freshness, selected route and owner
   before diagnosing a Mac failure. Avoid duplicate desktop/Mac repair actions.
7. **Optional gateway:** keep :9000 disabled on this workstation. Before its use
   elsewhere, test excluding STARTING sessions from routable readiness and resolve
   the inert `router.auto_start` setting honestly. Do not implement a second
   on-demand lifecycle just to make that flag functional here.

Acceptance: operators can determine which component owns a model, endpoint,
launch profile and recovery action; diagnostic changes do not mutate state
unexpectedly or suggest starting intentionally idle engines.

## 10. Package F — extend hybrid policy only after safety work

The user wants selected models warm and others on demand with idle unloading.
The existing policy already keeps one default warm by reverting to it. Preserve
that useful default; generalization is optional configuration, not a rewrite.

1. Establish a concrete desired warm set using existing workloads and model
   profiles. Do not invent model preferences. Determine whether the set fits
   simultaneously on the two GPUs with actual context/KV requirements.
2. Define what “warm” means: protected while resident, eager at startup, restored
   after a competing workload, or all three. Define precedence for manual stops,
   maintenance pause, leases, resource shortfall and explicit operator requests.
3. Extend existing router policy/configuration only as needed. Distinguish
   reverting to a warm default from unloading a lane to leave its GPU free.
   Reuse existing stop/start/placement/rollback operations.
4. Default missing policy fields to current behavior. Invalid or impossible
   warm sets should produce an actionable validation result, not an eviction loop.
5. Preserve on-demand queuing/readiness, bounded startup failure behavior and
   per-backend idle accounting. Track in-flight work before eviction and respect
   unknown/manual GPU consumers. Low GPU utilization is not idle ownership.
6. Start with deterministic placement and conservative serialization. Reuse
   current scheduler estimates, per-card free memory and safety margins. Do not
   treat combined free VRAM as proof of one-card fit or introduce preemptive
   optimization before basic coexistence is validated.

Acceptance tests: fitting warm set, insufficient capacity, one-card/TP=2 conflict,
timeout disabled, per-lane timeout, active stream, direct-backend activity,
manual stop, lease, pause, startup failure and restart reconciliation. Legacy
configuration must retain the one-warm-default behavior.

## 11. Testing rules and practical isolation

Use the existing Python environments and tests; no new test framework is needed.
Read each repository's fixtures before running tests outside restrictions.

- llmctl has autouse fixtures for plugins, config/data/state and operation locks,
  but inherited model-cache variables can still expose real models. Explicitly
  redirect `HF_HOME`, `OLLAMA_MODELS` and `LMSTUDIO_MODELS_DIR` to temporary paths.
- Set `LLMCTL_CONFIG_DIR` to a temporary directory containing copies of the
  example YAML files, not directly to the repository's `configs/`. A prior run
  wrote test alias overlays there; they were removed. Keep all generated state
  disposable and verify Git status afterward.
- Isolate collection-time DB initialization as well as per-test fixtures. Disable
  stack plugins with `LLMCTL_PLUGINS=none`; point `LLMCTL_DB_URL` and
  `LLMCTL_OP_LOCK` at temporary files. Isolate test HOME/XDG paths deliberately.
- For broad suites, use inert command shims for systemctl/sudo/docker/ssh and
  inspect tests that intentionally spawn processes. Unit tests must not invoke
  actual model installers, start scripts, post-start hooks or heal actions.
- Router fixtures provide a synthetic test API key; never use a production key.
  Keep command runners, evictions and HTTP engine probes fake. Use injected
  clocks/barriers instead of waiting an hour to exercise idle behavior.
- Async tests stalled inside the restricted execution environment during the
  audit, with an idle worker and waiting event loop. Properly isolated runs
  outside that restriction completed normally. Do not weaken assertions to work
  around that environment behavior. Seek execution approval only if needed.
- The prior full suite took about 6.5 minutes with coverage. Allow a reasonable
  completion budget while continuing progress updates; do not interrupt a healthy
  run merely because it is slower than a unit-test subset.

Run the smallest relevant tests after each substantive change. Before a release,
run the full affected repository suites, plugin compatibility tests, lint and
`git diff --check`. Record actual pass/fail/skip counts and environmental limits.
Do not promise the prior 1,671 count if new tests or code have changed it.

## 12. Next-agent reporting and completion criteria

Keep a small implementation ledger with columns: package, files, evidence,
tests, installed revision, live validation, remaining limitations. Update the
operational workflow as behavior changes; do not overwrite historical snapshots
as if they described today's runtime.

Each handoff should answer:

1. What already worked and was preserved?
2. What changed in source, and which regression proved the need?
3. Which source changes are actually deployed, to which environment/process?
4. What tests and passive/live checks ran, with their actual results?
5. Which workload or configuration changes still need approval?
6. What is the next smallest useful step?

For the initial continuation, success is a reviewed release candidate containing
the existing fixes, a verified deployment plan, and progress on the demonstrated
ownership/idle gaps with isolated tests. Do not declare the entire infrastructure
reliable solely because unit tests pass. Conversely, do not replace working
components merely to make the architecture look uniform.
