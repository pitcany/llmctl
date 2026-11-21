# Rust successor assessment — November 20, 2026

Reviewed checkout: `9b60d08c5474475a60907857ca15d3d8524b7f8d`.
Reviewer: Codex. This is an assessment, not an implementation or deployment.

Follow-up: the three reproduced findings now have working-tree fixes and regression
tests. See [implementation and verification](RUST-REVIEW-FIXES-2026-11-20.md).
The findings below describe the reviewed commit; they do not describe a deployment.

## Assessment

Continue the Rust architecture. The implementation now has a useful separation
between pure planning, host execution, transport, and UI. Durable operation
records, ownership transfer, version pins, staged upgrades, rollback, remote host
handling, and legacy delegation address the original reasons for a successor.
The implementation is substantially beyond the initial fake-engine prototype.

Stabilize it before M6 cutover. Three isolated probes reproduced gaps in lock
ownership, request identity, and audit attribution. The seven-day audit gate must
not be treated as proof that no external writes occurred until its false-negative
case is addressed and its observation coverage is stated clearly.

Scope: source and commit review, M2–M5 documentation, relevant vault records,
default Rust tests, targeted Python tests, and three temporary fake-engine probes.
No installed inference service was started, stopped, upgraded, or reconfigured.
Deployment and Mac drill results cited below are recorded evidence from earlier
work, not live verification performed during this review. This was a targeted
review of critical boundaries, not an exhaustive audit of every adapter.

## Reproduced findings

### P1 — delegated operation can outlive the host lock

`rust/crates/llmctl-daemon/src/ops.rs:437` skips taking a lock when the claimed
caller currently holds it. At `ops.rs:1090`, background execution receives that
absent lock and continues independently of the client. There is no transfer of
the caller's lock descriptor or continued ownership enforcement.

Reproduction against a temporary daemon and fake engine:

1. Hold the temporary operation flock in a caller process.
2. Admit a delayed `run coder` with `llmctl-lock-holder` naming that process.
3. Release the caller's lock while the operation is admitted/executing. This
   models lock release on caller failure or an early-returning caller.
4. Acquire the same flock through a second independent file description.
5. Keep that second hold until the agent's operation finishes.

Observed: the second controller acquired the lock and the original operation
completed with `succeeded` while the second controller still held it.

The same live agent blocks a second admission through its own API; that does not
protect against another legacy controller operating under the shared flock.
The caller contract asks it to retain the lock until terminal status, but a caller
crash cannot honor that promise. This conflicts with durable execution after a
client disappears.

Required direction: the executor must own a durable lock hold, including rollback.
Use a real ownership transfer or restructure delegation so the agent acquires
ownership before external effects. A one-time PID check is insufficient. Add a
regression where the delegating process dies during execution and another
controller attempts to take the lock.

### P2 — idempotency matches only operation kind and workload

`rust/crates/llmctl-daemon/src/ops.rs:950` returns an existing operation whenever
its kind and workload match. It does not compare the requested profile or other
material operation parameters.

Reproduction: complete `run coder` with a key, then reuse that key for the same
workload with a different profile source and expected model.

Observed: HTTP 200, `replayed: true`, and the old successful operation ID. The
changed request is not rejected. This can make a profile change appear successful
although the returned operation describes the earlier request.

Required direction: persist a canonical fingerprint of the material request and
compare it before replay. Include engine-specific parameters such as profile/model,
unload target, and upgrade target as applicable. Distinguish those from retry-only
transport metadata. Cover changed-payload reuse for every operation family.

### P1 — audit can suppress an unrelated external stop

`rust/crates/llmctl-daemon/src/audit.rs:162` treats a transition as explained if any
recent operation touches the workload. It does not require an executed effect
that explains the direction or identity of the observed transition.

Reproduction against a managed fake workload:

1. Start `coder`; call status to establish a running baseline with no findings.
2. Submit another successful `run coder` while it is already running.
3. Before the next status comparison, externally stop it in the fake world.
4. Call status and audit.

Observed: status reported `Stopped`, but audit returned zero findings. As a control,
a subsequent external start with a new PID and no intervening operation produced
one finding. The audit was functioning; the recent no-op run suppressed the stop.

Required direction: attribute transitions to actual recorded effects and resulting
identities, not just a workload touched during the sampling interval. Add this
no-op case, opposite-direction changes, and acknowledgement-only updates to tests.
Treat observation-based audit as incomplete coverage: polling can miss transient
changes even after this bug is fixed.

## Additional limits to resolve or retain explicitly

- Request protection is best effort. `ops.rs:486` allows the operation when router
  accounting is unavailable. A timeout or malformed response does not establish
  that no requests exist. The count check also does not itself reserve/drain the
  router's request-admission path. The router's own switches have draining logic;
  direct agent operations need a shared contract too. This behavior is documented
  in `inflight.rs`; this review did not run a real traffic-interruption drill.
- The TUI still renders preset and catalog placeholders (`llmctl-tui/src/ui.rs:212`
  and `:518`) while the corresponding backend work exists. Since M5 identifies
  the TUI as the main daily interface, this is a replacement-readiness gap.
- Source size is now about 43.5k lines across crate `src/*.rs`, including inline
  tests. `ops.rs` is 2,281 lines and CLI `main.rs` is 2,987. Rust has not removed
  orchestration complexity. Stabilize shared admission/execution contracts before
  adding more operation kinds; extract modules where that makes those contracts
  easier to review. A broad rewrite is unnecessary.
- Entry documentation is stale: `rust/README.md` still opens with first-batch/M2
  status; M5 plan/progress headers retain older snapshots. Dated detail is useful,
  but one current summary should identify completed work and remaining blockers.

## What is working well

- The core/daemon/client/TUI boundaries are meaningful. The extra CLI crate avoids
  a dependency cycle rather than introducing another lifecycle implementation.
- Tests exercise real daemon processes, Unix sockets, SQLite and process death.
  These are stronger evidence for the design than adapter mocks alone.
- M2–M5 pilot reports record failures, diagnoses and corrective tests. The November
  20 router fallback incident demonstrates why controlling every legacy writer
  matters; the documented refusal-final change is a useful correction.
- The TUI uses bounded background calls and stale-host views. The M5 report records
  much better key-to-frame latency. That measurement was not repeated here and
  reflects architecture/work allocation as well as language choice.
- M4's closure with limits is explicitly recorded in ADR-053. Preserve those limits
  as acceptance work rather than interpreting milestone closure as universal
  engine/host coverage.

## Verification performed

Toolchain observed: `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.

| Check | Result |
| --- | --- |
| `cargo test --workspace --locked` | Failed: 277 passed before termination, 1 failed, 32 ignored. Failure: `coord::tests::the_lock_excludes_a_second_holder_and_names_the_first`, `coord.rs:285`, lock remained busy immediately after drop. This failure is already mentioned in M5 progress. |
| `cargo test --workspace --locked -- --test-threads=1` | Passed: 303 passed, 0 failed, 32 ignored. Serial success does not establish the parallel failure's cause. |
| Targeted Python guard and vLLM orchestration tests | 46 passed in 0.74 s. |
| `cargo fmt --all -- --check` | Failed; formatting differences in 85 files on the installed toolchain. |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | Failed in core with six lint diagnostics. This run stops before proving the rest of the workspace clean. |
| Three isolated review probes | All three findings above reproduced; only fake engines and temporary state used. |

Ignored service/remote drills were not enabled. No macOS build or live serving
test was performed. No production defect fix is included in this review.

Temporary evidence: `/tmp/llmctl-review-repros.py`,
`/tmp/llmctl-review-repros.json`, `/tmp/llmctl-assessment-tests.log`,
`/tmp/llmctl-assessment-serial.log`, `/tmp/llmctl-assessment-fmt.log`,
`/tmp/llmctl-assessment-clippy.log`, `/tmp/llmctl-assessment-python.log`.
The steps and results above remain the durable record if temporary files expire.

## Recommended next sequence

1. Fix and regression-test delegated lock ownership, request fingerprinting and
   audit attribution. Resolve or explicitly narrow the request-safety guarantee.
2. Restore reliable parallel test runs and establish a pinned formatting/lint gate.
3. Connect presets/catalog to the TUI and update the current-status summary.
4. Run the required pilot with the corrected audit and named coverage limits,
   including caller death, agent restart, remote disconnect and rollback.
5. Reassess M6 cutover from that evidence. Keep further feature expansion behind
   this acceptance work.

Artifact documentation check: the local `~/AI/docs/ARTIFACT-INDEX.md` identifies
the published **Driving llmctl** guide. Pending verification: reconcile that guide
with delegated `run --direct`, one-off override refusals and Rust ownership. No
published page was inspected or updated during this review.

Confidence: high for the local source findings and reproduced outcomes; medium
for fleet readiness inferred from earlier deployment reports. Live deployment
state and unexamined adapter paths remain outside this assessment.
