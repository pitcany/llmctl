# llmctl consolidation and reliability plan

Prepared November 4, 2026 (America/Los_Angeles). Planning only: this document
does not authorize installation, service changes, migrations, commits or pushes.

This is the current forward plan. It replaces the work ordering and completion
criteria in [the November 3 handoff](NEXT-AGENT-PLAN-2026-11-03.md), while retaining
its useful technical detail. The [audit](CONTROL-PLANE-AUDIT-2026-11-03.md) and
[deployment record](DEPLOYMENT-PLAN-2026-11-03.md) remain dated evidence. Their
statements about installed versions and timer state are not current facts.

## Outcome

Make llmctl the normal way to inspect, configure and operate the existing
inference stack. Consolidation must reduce independently maintained definitions
and competing lifecycle paths, not merely document their existence.

An operator should be able to identify a workload, see where its effective
configuration comes from, preview its GPU/service impact, start or stop it,
check readiness, inspect logs and understand recovery through existing llmctl
surfaces. Adding a supported workload should not require manually repeating its
launch parameters in multiple configurations.

User decision: preserve the configured **qwen27b warm default initially**, and
make lifecycle policy configurable. This is a future policy default, not an
instruction to start qwen27b now or override an intentional stop.

User decision: **retire both `llmc` (`pitcany-llmc`) and `llm-models-config`**.
Preserve recoverable code/history and useful presets, shortcuts and benchmark
artifacts. Carry functionality into existing llmctl modules only where a real
workflow needs it. Do not migrate or maintain the entire compression planner
merely to preserve the second package's consumer. This supersedes the earlier
proposal to migrate `llmc` onto llmctl. This decision sets implementation scope;
this planning step does not uninstall anything or remove live configuration.

## 1. Verified starting point and limits

- llmctl checkout and installed package are at `84ed67f`. The inventory,
  user-scope, API-settings and delegation-handoff changes are already committed.
  The local remote-tracking branch matches; remote state was not refreshed.
- Router coordination and idle accounting are uncommitted in branch
  `feat/shared-op-lock-2026-11-03` in a temporary worktree. Healer coordination
  is uncommitted in `feat/heal-op-lock-2026-11-03` in another temporary worktree.
  Neither is deployed. Resolve their paths with `git worktree list`; durable
  patch copies exist under `~/AI/logs/staged-2026-11-03-control-plane/`.
- November 4 review: 98 targeted llmctl tests, 61 router tests and 6 healer
  tests passed. Two additional isolated router regressions failed: activity
  leaks when a request is redirected to engine 2, and an idle restart can race
  a request after its last veto. Do not accept the old green suite as closure.
- Unknown engine-load measurements currently permit idle reversion in the
  staged code. This does not meet the prior conservative-activity requirement.
- Latest passive check: API/router processes and reconcile/heal timers active;
  both vLLM engines inactive; router has no active backend. This does not prove
  an incident. Recheck intent and state before any future deployment.
- Broad consolidation and general lifecycle policy have not been implemented.
- `pitcany-llmc` 0.1.0 is installed with a `~/AI/bin/llmc` shim. The targeted
  search found no callers in llmctl, inspected service definitions, operational
  scripts or benchmark scripts beyond its own launcher. This is no identified
  operational dependency, not proof it was never used. Recheck consumers before
  retirement, including scheduled jobs and relevant integrations.

Source examined for this plan establishes two specific consolidation seams:

1. `llmctl.presets.schema.Model` already owns the shared vLLM preset schema.
   However, llmctl loads canonical and legacy directories with legacy precedence,
   while `llm-models-config` still treats the legacy directory as its store.
   Retire the legacy writer while preserving preset/shortcut data; then migrate
   live presets only after llmctl's effective values and precedence are verified.
2. `llmctl-stack/router_lanes.py` already imports command backends from
   `router.yaml`. It intentionally excludes profile-based vLLM backends, which
   llmctl obtains from presets. Preserve that integration and investigate the
   remaining profile-reference duplication; do not create another lane list.

## 2. Target ownership and filesystem structure

These are proposed ownership decisions for incremental implementation, not a
claim that every existing artifact already complies.

| Concern | Owner and target | Concrete consolidation |
| --- | --- | --- |
| Operator actions and previews | Existing llmctl CLI/TUI/API and services | All routine workload operations resolve through the same existing planning/lifecycle functions. |
| Host control settings | Existing `~/.config/llmctl/settings.yaml` | Keep host defaults, managed slots and unit scope here; detect conflicts with lane metadata. |
| Model identity and history | Existing llmctl SQLite registry | Preserve IDs, host attribution, sessions and benchmark history; use existing `model_ref` links where identity is verified. |
| vLLM model-specific presets | Existing llmctl schema/store; target `~/.config/llmctl/presets/` | One writable definition per preset; retire `llmc` and `llm-models-config`, preserving useful data and required workflows. Legacy paths remain compatible during data migration. |
| Reusable runtime profiles | Existing SQL profiles and export/import | Retain their cross-runtime purpose; explicit references/precedence connect them to model-specific configuration where needed. No automatic YAML-to-SQL flattening. |
| Launch rendering | Existing llmctl preset loader and vLLM renderer | One authoritative managed-launch rendering path. Preserve extra arguments, parsers, speculation, runtime selection and absent-value semantics. |
| Engine installations | Existing engine records and per-preset runtime paths | Discover/adopt valid installations rather than reinstall. Keep vLLM environments separate where required by workloads. |
| Request routing and command lanes | Existing router configuration and stack plugin | Routing aliases/policy remain here; reuse imported lanes. Profile launch artifacts should be derived from verified preset definitions where parity is proven. |
| Process lifetime | Existing system/user systemd units and llmctl-owned sessions | Retain correct scope and aliases; remove duplicate supervisors only after proving equivalence and switching consumers. |
| Specialized launch scripts | Existing owning source repository | Keep necessary engine-specific setup; turn duplicate operator shortcuts into compatible llmctl entry points where safe. |
| Recovery and reconciliation | Existing healer, systemd policies, reconcile timer | Shared coordination, manual-stop protection and bounded recovery; no additional watchdog. |
| Front door and Mac connectivity | Existing LiteLLM and mesh/Mac owners | Preserve endpoint names and fallback. Expose provenance/health through llmctl without moving Mac lifecycle ownership to Linux. |

Physical layout after convergence:

- `~/Projects/llmctl`: portable control-plane source, schemas and lifecycle code.
- `~/AI` and its existing repositories/packages: workstation integrations,
  engine-specific launch sources, router, healer and deployment records.
- `~/.config/llmctl`: host settings, canonical presets and existing engine metadata.
- `~/.local/share/llmctl`: existing registry/history/operation state.
- Existing engine environments, model directories, unit paths and serving ports
  remain in place unless an individually justified migration is approved.

Do not introduce a new persisted inventory database. The inventory below is an
audit artifact; operational views should be derived from existing owners.

## 3. Work packages and dependency order

| Package | Purpose | Completion evidence |
| --- | --- | --- |
| P0 — baseline and concrete inventory | Name every in-scope definition, consumer and mutation path | Dated inventory plus per-artifact keep/link/generate/wrap/retire decisions. |
| P1 — close staged reliability defects | Make coordination and idle protection safe | New regressions pass, existing guards retained, release boundary documented. |
| P2 — unify configuration resolution and retire both extra packages | Preserve required workflows in llmctl and remove unused tooling | Clean stack installation and retained-workflow tests pass without either package; effective configuration has provenance. |
| P3 — consolidate launchers and service ownership | Make llmctl the routine management entry point | Pilot workload managed end to end without repeated launch definitions. |
| P4 — migrate existing definitions | Apply the agreed structure one workload at a time | Before/after parity, fewer independent definitions, tested rollback. |
| P5 — configurable lifecycle and GPU policy | Preserve qwen27b default; support optional warm/on-demand policy | Resource-aware policy and request-safety acceptance matrix. |
| P6 — controlled rollout and operations | Verify deployed behavior and finish cleanup | Exact installed versions, live acceptance record and updated workflow. |

P0 comes first. P1 and P2 source work can proceed independently after P0.
P3 can be prepared alongside P2; migration requires its compatibility work.
P5 depends on P1 and the ownership decisions in P2/P3. P6 applies to each small
release, not only the end. A large policy feature must not delay useful
consolidation that has already passed its own acceptance criteria.

## 4. P0 — inventory and decisions, not another architecture survey

Reuse the audit and `graphify-out/` to find source; verify inferred relationships.
Read applicable instructions and the device knowledge/discovery records.

Produce `docs/INFERENCE-INVENTORY.md` with one row per serving instance/profile
and linked rows for supporting artifacts. Include:

- Host, stable existing model/profile IDs, served name and aliases.
- Runtime/interpreter or binary path, installed version and observed serving
  version separately; verification date and evidence source.
- GPUs/TP, endpoint, unit/process owner, system/user scope, launch entry point,
  readiness probe, log location and recovery owner.
- Authoritative editable fields, derived files, actual readers/writers,
  preset/profile references, precedence and deployment source.
- State: production, experimental, intentional standby, historical, or unknown.
  Unknown is not permission to remove an artifact.
- Disposition: keep, reference, generate, compatibility wrapper, or retire;
  rationale, blockers, rollback and whether live approval is required.

Inventory configured inference scripts in their known directories and trace
callers in units, router, plugin, CLI and relevant benchmarks. Do not interpret
an unreferenced script or old filename as proof of obsolescence. Record current
package pins and staged changes separately from unrelated benchmark work.

Measure the baseline: conflicting preset aliases, independently maintained
launch definitions, duplicate lifecycle entry points, unmanaged relevant units,
and configurations whose effective source cannot be explained. Set concrete
reduction targets after enumeration; do not invent counts now.

Acceptance: every configured desktop serving path and known Mac route has an
owner, launch chain and evidence/unknown status; each proposed retirement names
its replacement and all identified consumers. No files moved or services changed.

## 5. P1 — finish the existing safety work

Preserve and review the staged changes rather than start over. Recreate the two
November 4 reproductions in the router suite; `/tmp/llmctl-review-regressions.py`
is an optional temporary reference, not a durable dependency.

1. Fix activity ownership when the actual upstream changes. Cover buffered
   responses, ordinary streams and delayed loading streams, including admission,
   disconnect, cancellation and upstream failure. Completion must release the
   actual instance exactly once; no counters remain after work ends.
2. Close the interval between idle evaluation and destructive work. Request
   admission and switching need a coherent reservation/ordering contract across
   awaited stop, eviction and unit-state calls. A final check alone is insufficient
   if a later await permits new work on a resource about to be stopped.
3. Compute the complete affected GPU/service footprint. A default TP=2 revert
   must account for work on both cards, not only the lane being watched. Test
   queued work and a busy second slot as well as aliases of one upstream.
4. Unknown activity visibility must defer automatic destructive idle actions by
   default. If a router-only traffic assumption is supported, make it explicit
   and opt-in. A metrics snapshot alone cannot guarantee safety for new direct
   engine traffic; document the required routing/lease contract.
5. Verify router/healer/direct CLI coordination, delegation without deadlock,
   custom lock paths, surviving jobs, cancellation, and restoration after failure.
   Revalidate stale observations under ownership. Explicitly decide when rollback
   is considered complete and how failed recovery remains visible.

Acceptance: multiprocess lock tests and deterministic async race tests pass;
status remains responsive; automatic actions cannot cut protected work. Keep
raw administrator actions and systemd autonomous restarts documented as external
boundaries. No live failure injection during source validation.

## 6. P2 — one preset implementation; retire llmc and llm-models-config

Extend `llmctl/presets/{paths,store}.py`, `services/preset_loader.py`, the existing
profile service and doctor/validate/plugin seams as needed. Retire both
`~/AI/packages/llmc` and `~/AI/packages/llm-models-config`. Preserve a recoverable
source revision or patch archive, including relevant uncommitted work, before
source removal. Do not create another installed legacy package to hold the code.

1. Build a side-effect-free inspection path using existing structures. It must
   not initialize/migrate the database, create preset symlinks, seed profiles or
   run discovery imports. Report missing/unreadable evidence explicitly.
2. Expose effective source and precedence for each relevant field. Preserve
   current behavior first: legacy preset precedence, explicit preset overrides,
   settings defaults and SQL parameter precedence. Flag collisions instead of
   silently selecting a newly preferred value.
3. Inventory existing presets, shortcut overlays and benchmark artifacts without
   deleting or rewriting them. Reuse llmctl's existing inspection, planning,
   preset and benchmark capabilities for retained workflows. Preserve aliases,
   supported extras and absent-versus-default values. Add shortcut-overlay support
   only where needed to retain actual configured aliases. Do not port the unused
   compression-ranking knowledge base, calibration pipeline or planner-driven
   preset generator by default; retain their source/history for future retrieval.
4. Stage canonical-directory migration with a dry-run manifest and collision
   report. No reader may lose a preset after another tool edits it. Avoid symlink
   cycles and hidden deletion through save operations. Do not simply reverse
   lookup precedence or delete the legacy directory.
5. Keep reusable SQL profiles distinct. Use verified model references and
   explicit provenance rather than name-only guessing across hosts. Define a
   shared launch value once when two paths describe the same workload; do not
   globally synchronize independent experimental profiles.
6. Check remaining imports, command invocations and scheduled jobs before removal.
   For any actual consumer, identify its needed behavior and migrate that narrow
   workflow to existing llmctl capabilities. Record unique functionality being
   retired, such as heuristic compression rankings and KB calibration, honestly;
   do not claim llmctl provides an equivalent without evidence.
7. Remove unused `llmc`/`llm-models` launchers; retain a small compatibility wrapper
   only for an identified caller and an equivalent command. Do not silently map
   commands with different semantics. Document replacements and retired commands.
8. Remove both packages' source, workspace/dependency references, lockfile entries,
   obsolete CI/release targets and stale documentation in a reviewable change.
   Keep remaining llmctl pins consistent with the approved release. Replace the
   optional cross-package schema-parity test with relevant legacy-data/retained-
   behavior coverage, rather than leave it silently skipped. Remove tests solely
   for retired features; preserve tests for every behavior that remains supported.
9. Verify from a clean isolated installation with both distributions absent.
   Search maintained source, launchers and build/release configuration for remaining
   imports or distribution requirements; explain intentional historical mentions.
   Production uninstallation belongs to an approved release with rollback, not
   to a routine environment sync during development.

Acceptance: presets resolve consistently across llmctl CLI/TUI/API and dependent
launch rendering, with no remaining required writer in the retired packages. Tests
cover duplicate aliases, broken links, rollback, interrupted writes, custom
paths, null/default semantics and any configured shortcuts. The retained stack
works with both retired distributions absent. Existing benchmark artifacts remain
intact and attributable, with a documented way to inspect historical results.
Read-only diagnostics leave filesystem/database state unchanged. Package
retirement and migration of live preset files are distinct steps: do not delete
legacy data with the packages.

## 7. P3/P4 — consolidate scripts, services and live definitions

Choose one representative vLLM preset and one command-backed workload from the
verified inventory. The command-backed pilot tests the existing plugin boundary;
do not choose an active workload solely because it is convenient.

For each pilot:

1. Trace preview -> control action -> launch script/unit -> readiness -> session
   tracking -> logs -> stop/restart -> recovery. Identify duplicated facts and
   lifecycle policy embedded in scripts.
2. Keep engine-specific setup in its current implementation where necessary.
   Move duplicated policy into the established owner. Keep low-level router
   launch helpers separate from operator compatibility wrappers: a wrapper that
   delegates to llmctl must not create a router -> wrapper -> router loop or
   recursively acquire an incompatible lock.
3. Preserve system/user scope, service aliases, ports, GPU placement, parser and
   runtime choices. Reference the existing unit from llmctl rather than install
   a second unit for the same process. Validate ambiguous port-only matches.
4. Use router-imported command lanes as today. For vLLM profile references,
   generate launch artifacts from a verified preset where supported; preserve
   experimental/manual profiles until equivalence is demonstrated.
5. Surface status, readiness, configured versus observed version, log access,
   pause/lease state and recovery through existing llmctl surfaces. Retain engine
   logs/journald; no new logging backend is required.
6. Prepare an exact migration manifest: old/new paths, consumers updated,
   effective-value differences, compatibility wrappers and rollback. Back up
   configuration and use a consistent SQLite backup if state changes are involved.
7. After approval, migrate one pilot, verify it, and pause expansion on regression.
   Archive superseded definitions reversibly; defer deletion until consumers and
   rollback no longer require them. Repeat only for enumerated in-scope workloads.

Acceptance: the pilot can be managed through llmctl without editing several
independent launch definitions. Previous aliases and intended workflows still
work. The measured duplication count falls; registry IDs, benchmark history and
remote routes remain intact. A documented inventory alone does not complete P4.

## 8. P5 — hybrid lifecycle on the existing two-GPU machinery

Preserve the current qwen27b default and 3,600-second idle-revert behavior for
legacy configuration. Add optional policy through the existing router and
llmctl/plugin integration; no additional scheduler service.

Design decisions to encode and test:

- Distinguish warm intent, on-demand loading, return-to-default and unload-to-free-
  capacity. Document whether warm means eager startup, retention, restoration,
  or a combination. A warm intent is conditional on authorized capacity.
- Manual stop and maintenance pause suppress automatic revival until explicitly
  cleared. Active leases and protected requests constrain eviction. Recovery
  restores intended operation, not every catalog entry.
- Use existing per-card placement and estimates, actual runtime allocations,
  context/KV requirements and headroom. Do not infer fit from summed free VRAM
  or idleness from low utilization. Identify unknown GPU consumers explicitly.
- Treat TP=2 as a conflict with both slots. Prefer deterministic placement and
  conservative serialization; no opportunistic eviction optimizer in this release.
- Invalid/overcommitted warm sets produce a clear validation result and no
  restart loop. Startup/readiness failures remain bounded with visible rollback.
- Preserve historical benchmark provenance; capture exact effective configuration
  for new measurements. Mac routes retain their existing lifecycle owners.

Acceptance matrix: legacy default, fitting/oversized warm set, one-slot/TP=2
conflicts, per-lane idle timeout and disabled timeout, queued/streaming/direct
traffic, unknown telemetry, manual stop, pause, lease, failed startup and restart
reconciliation. Additional warm-model choices remain configuration decisions;
do not invent them for the user.

## 9. P6 — releases, verification and final workflow

Use small releases: corrected coordination; llmctl preset/shortcut compatibility
and diagnostics; retirement of `llmc` and `llm-models-config`; pilot consolidation;
remaining approved data migrations; optional policy. Any required consumer
replacement ships before package removal. Retain an installable prior release
and recoverable source for rollback throughout the transition.
Each release identifies exact source/artifact revisions, dependency pins,
processes affected and compatibility with the preceding release.

Before requesting live approval, finish the diff, isolated tests, deployment
steps, backup/rollback procedure and expected interruption. Inspect current jobs,
leases and workloads again. Staged router/healer work cannot be considered
deployed because llmctl alone was updated. Validate imports outside the checkout.

Testing uses existing tooling, temporary config/data/model-cache paths, disabled
production plugins and fake commands/transports. Check subprocess paths before
running tests. Do not execute real installers, hooks or start scripts in unit
tests. Run narrow regressions per change, then full affected suites, cross-package
tests, lint and diff checks for each release. Report real failures/skips.

Live acceptance requires separate, bounded approval for model switches,
coexistence, rollback/failure tests and remote changes. Restore the agreed
starting workload/intent afterwards. Passive HTTP health is not proof of
successful generation, recovery or failover.

Update `docs/OPERATIONS.md` and a current implementation ledger. Preserve dated
audit snapshots. The final workflow must explain:

1. Where to edit a model/profile, how aliases resolve, and how to preview changes.
2. How to start/stop/restart, see readiness/GPU ownership, and retrieve logs.
3. How warm/on-demand policy interacts with manual stops, leases and maintenance.
4. How to diagnose drift, blocked recovery, unavailable Macs and version mismatch.
5. How to deploy/roll back a package or individual migrated workload.

## 10. Boundaries and definition of done

Never read or expose prohibited credential locations, `.env*` files, API keys,
process environments or private request logs. Environment launch-file contents
were not audited. Where launch equivalence cannot be proven from permitted
source/configuration and observable metadata, mark it unverified and defer that
migration; do not bypass the restriction with a helper that reads those files.
Request a sanitized, permitted representation only when necessary.

Preserve unrelated work, `graphify-out/`, model weights, installed runtimes,
registry/history, endpoints and remote ownership. No commits/pushes without a
separate request. No disruptive/destructive actions without approval. Source
implementation starts only when requested after this planning step.

Completion requires all of the following, with explicit exceptions agreed:

- All in-scope workloads have traceable configuration and lifecycle ownership.
- Preset readers/writers converge; compatibility paths do not create drift.
- Both `llmc` and `llm-models-config` are retired, and clean installation/integration
  tests pass without them. Required workflows use llmctl; presets, configured
  shortcuts, benchmark artifacts and recoverable source/history are preserved.
  No wholesale planner migration was added merely to keep unused tooling alive.
- Agreed duplicate definitions/entry points are actually consolidated, with
  before/after counts and tested replacement paths.
- Routine operations work through llmctl, including specialized and remote lanes
  within their existing ownership boundaries.
- Required safety and configurable lifecycle tests pass; live validation states
  exactly what was exercised and what remains unverified.
- Deployed revisions match the approved release and documentation explains the
  resulting workflow, rollback and remaining limitations.

Next implementation step: P0's concrete inventory/disposition table, followed by
P1 regression fixes and P2 compatibility work. Do not resume the old plan at
deployment or declare consolidation complete merely because diagnostics improve.
