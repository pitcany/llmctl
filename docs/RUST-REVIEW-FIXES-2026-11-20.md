# Rust review fixes — November 20, 2026

The three reproduced findings in [the assessment](RUST-REVIEW-2026-11-20.md)
are fixed in the working trees. Base llmctl commit: `9b60d08`. No application
commit, deployment, service restart, or live inference mutation was performed.

Follow-up: the owner subsequently authorized commits and deployment. See
[deployment and real-host acceptance](RUST-REVIEW-DEPLOYMENT-2026-11-20.md)
for the installed commits, corrected Python packaging, and verification results.

## Executor lock ownership

The daemon always acquires its own host flock before admission and retains it
through execution and rollback. A caller PID no longer bypasses the lock.
Dropping the lock explicitly unlocks it before closing the descriptor; a
short-lived fork inheriting the descriptor cannot extend ownership.

Python llmctl releases its existing hold for delegated lifecycle and preset
commands, then reacquires it before continuing. The companion change in
`~/AI/llm-router` does the same for lane and profile delegation. Reacquisition
failures propagate before fallback or further mutation. The router tracks the
hold by async task; a child task cannot release its parent's lock.

Deploy the daemon and these caller changes together. An old caller that retains
its lock will be refused by the corrected daemon. No PID header can authorize
execution without the executor's own hold.

Regression coverage includes a real delegating CLI killed during execution,
an independent flock contender, false and true PID claims, successful and failed
caller handoffs, and failure to reacquire before fallback.

## Material request identity

Admission persists a SHA-256 fingerprint with the operation in the same SQLite
insert. It covers kind, workload, force, profile, and explicit specialized inputs
(unload model, lane/router edits and requests, acquisition request, upgrade or
rollback target). Generated plan IDs, changing observations and lifecycle
confirmation metadata do not change the fingerprint.

An exact retry returns the existing operation. A changed request conflicts.
The fingerprint survives daemon restart. Existing journals receive a nullable
column; old rows remain readable, but their keys cannot be replayed when the
original request identity is unknown. Inspect such operations by ID/key before
submitting a new action. Do not automatically retry with a new key.

## Audit evidence

The executor records timestamped post-effect observations for workloads affected
by completed state-changing phases, including rollback. It captures them while
holding the host lock and saves them with the terminal result. No-op plans do not
create effect evidence. Startup reconciliation does not fabricate old evidence.

Audit compares transitions against these observations and their timestamps.
Acknowledging an operation does not refresh its evidence. An active operation
preserves the previous baseline until execution settles. Unknown observations
cannot explain a transition.

Regression coverage includes a no-op followed by a foreign stop, a successful
restart followed by a foreign stop before sampling, an acknowledgement of old
evidence, and status sampling during a restart.

Polling still cannot prove the absence of every external write. Unobserved
transient changes and external changes indistinguishable from a captured result
remain outside its attribution guarantees. Missing evidence is conservative;
it does not authorize suppressing a finding.

## Cancellation race found during verification

A parallel test accepted a cancellation that a later progress write erased.
The existing separate read/check/write did not protect that interleaving.
SQLite now preserves `cancel_requested` atomically when stale `executing`
progress arrives. A deterministic database regression failed before this fix.

## Verification

- `cargo test --workspace --locked`: **314 passed, zero failed, 32 ignored**
  with default parallel execution.
- Targeted Python delegation/orchestration: **51 passed**.
- Router handoff/host-lock tests in its actual source tree: **41 passed**.
- Broader Python run including the operation-lock suite: **67 passed, one failed**
  at the existing PID-substring assertion described below.
- `git diff --check`: clean in both working trees.

All daemon tests use isolated temporary data and fake engines. Live systemd/GPU/macOS/remote tests
remain opt-in and were not run. Existing repository formatting and strict Clippy
failures from the assessment remain outside these fixes.

The Python operation-lock suite exposed an existing assertion defect:
`test_separate_process_blocks_until_release` searches for the parent's PID as a
substring of the entire error. Parent PID `4` matches child PID `14`, although the
reported owner is correct. The assertion was left unchanged.

## Remaining work

Deploy and run the real-host handoff/pilot acceptance separately. Continue the
request-draining, TUI preset/catalog, and format/lint work identified in the
assessment. This does not reopen the accepted M4 milestone or claim M6 readiness.

Published artifact follow-up: check Driving llmctl against this lock handoff and
legacy-key behavior when updating its Rust delegation guidance. The artifact
index was checked; no published page was changed.
