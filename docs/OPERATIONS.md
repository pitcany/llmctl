# Operating the existing llmctl stack

This workflow extends the existing services. The November 3, 2026
[audit](CONTROL-PLANE-AUDIT-2026-11-03.md) records what was verified on this
workstation, including important differences between installed and checkout code.

## Identify the version and state first

The normal workstation shim `~/AI/bin/llmctl` executes
`~/AI/.venv/bin/llmctl`. This is a pinned package, not an editable install of
`~/Projects/llmctl`. At audit time both reported version 0.3.0, but the installed
commit was `16dc60f` and the checkout was `b72b874` plus the audit fixes. A version
string alone cannot show whether a fix is deployed. The stack plugin pins its
tested llmctl revision in `~/AI/packages/llmctl-stack/pyproject.toml`.

Use the existing inspection surfaces:

```bash
llmctl status
llmctl health --json
llmctl metrics --json
llmctl sessions --no-fresh --json
```

`status` includes plugin router state; `metrics` reports engine load rather than
inferring it from allocated VRAM. A model in `models`/`ls` is cataloged; a model
listed by a server may merely be available; a loaded model occupies memory;
successful generation is a separate observation. An inactive optional lane is
normal. Missing health evidence must not be called a successful generation test.

`sessions --no-fresh` reads persisted state. `sessions` without that flag and
`reconcile` can update session tracking, and ordinary CLI initialization can
create/migrate tables. `doctor` and `validate` are diagnostics, but their loading
paths can initialize storage or migrate legacy preset symlinks. For a strict
read-only audit use passive HTTP GETs, `systemctl show` and SQLite `mode=ro`.

## Preview and change serving models

```bash
llmctl ls
llmctl run NAME --dry-run
llmctl run NAME
llmctl status
llmctl metrics --json
```

Replace NAME with an exact model/preset/lane identifier shown by llmctl. Inspect
the preview's affected workloads, GPUs, endpoint and readiness budget before
confirming the run. Normal swaps use llm-router when available; `--direct` is an
explicit fallback with different coordination boundaries, not the default way
to operate. Avoid simultaneous direct operations and request-driven switches.
Do not use `--force` to dismiss missing weights, port conflicts or fit failures.

For the existing two-card workstation:

- A TP=2 preset needs both cards and can evict one-card workloads.
- One-card presets can coexist through the already configured second engine on
  :8010. Actual placement and conflicts are determined by the preview.
- New inventory behavior exposes that configured engine even when it is idle;
  it does not start it or change the placement policy.
- Unit generation now preserves `user_scope`; an existing unit does not need to
  be regenerated merely to gain inventory visibility.

Stop/restart through llmctl with an unambiguous subject:

```bash
llmctl stop NAME --dry-run
llmctl stop NAME
llmctl restart SESSION --systemd
llmctl logs SESSION --lines 100
```

Use `--systemd` for a restart only when the session is adopted from systemd;
owned sessions use `llmctl restart SESSION`. Stopping an adopted session by
session selector likewise requires `--systemd`; a lane or preset NAME goes
through the serving-target path. Verify the resulting endpoint and status.
Do not equate “restart issued” or STARTING with ready. If rollback fails, retain
its recovery details and inspect the serving state before retrying.

## Keep the current lifecycle and recovery policy

The desktop router currently returns to `qwen27b` after a non-default lane has
been idle for 3,600 seconds. It polls every 30 seconds, uses a 300-second
minimum hold, and has failure cooldown, lease and broken-lane guards. Ollama
has its own resident-model lifecycle and is evicted before applicable vLLM
swaps. Mac lifecycle remains with the existing Mac services and transport.

Do not shorten the idle timeout until request completion/cancellation and
in-flight protection are implemented in the existing router. Requests sent
directly to engine ports bypass its arrival clock. Per-model warm/TTL policies
and simultaneous slots need explicit per-backend ownership before automation.

The reconcile timer updates tracking every 30 seconds; the healer runs on its
existing two-minute timer. An inactive oneshot with result `success` is normal.
Inspect `llm-heal status` and existing incident evidence when recovery is held.
An intentionally stopped engine should stay stopped. Preserve cooldowns and
systemd start limits; repeated forced retries can exhaust the restart budget.

### Coordination changes staged on November 3 (not deployed)

These exist in source and isolated tests only; the running router, healer and
installed llmctl still behave as described above until the
[deployment plan](DEPLOYMENT-PLAN-2026-11-03.md) is approved and executed.

- **One GPU ownership boundary.** llm-router (optional `switch.op_lock_file`)
  and llm-heal (optional `op_lock_file`) join llmctl's existing flock,
  `~/.local/share/llmctl/operations.lock`. Order: router `asyncio.Lock`, then
  the flock, held through restart and rollback; the router re-probes and
  rechecks pause/broken/lease once it owns the GPUs. Busy longer than
  `op_lock_wait_s` (30 s) is a 503 naming the holder, never a failure
  cooldown. Waiting polls with `await asyncio.sleep`; `/health` and
  `/running` stay responsive.
- **Delegation.** `llmctl run` (router path) releases its hold only for the
  `POST /admin/switch` call and takes it back to restore paused lanes; if
  someone else kept it, the command stops and names the holder. Therefore the
  router lock must not be enabled with an llmctl that has the flock but lacks
  this handoff (`b72b874`); the installed `16dc60f` has no flock and is safe.
- **Healer.** GPU repairs are held while the flock is held or a router lane
  lease is live (same freshness/PID rules as the router). The healer holds
  the flock while it restarts an engine itself, and releases it for
  `llmctl ...` repairs, which take it on their own. `llmctl run` now matches
  its busy-process patterns too.
- **Idle revert.** Activity is counted per upstream from acceptance (switch
  wait and KV hold included) until the response body finishes, fails, is
  cancelled or is dropped; the revert never runs over in-flight work, and the
  clock restarts at completion. The check is repeated under switch ownership
  with no await before the restart. vLLM `num_requests_running/waiting` vetoes
  for traffic outside the router; engines without those gauges are "unknown"
  and do not veto (use a lane lease for direct work on them). `GET /running`
  gains `idle_revert.in_flight` and `idle_revert.deferred`. The 3,600 s
  default is unchanged; do not shorten it until this is deployed.
- **Still outside coordination:** a raw `systemctl` by an administrator,
  systemd's own `Restart=on-failure`, Mac-side services, and start scripts run
  by hand. Mutating paths still probe real state after taking ownership.

## Configuration and benchmark discipline

Keep canonical model facts in the existing preset/profile/registry mechanisms.
Keep command-lane definitions in router.yaml, which the installed plugin already
imports. Before migrating legacy presets, compare effective values: the current
store loads legacy files after canonical files, so legacy aliases can override
canonical ones. Do not delete old files based solely on directory names.

Keep vLLM environments separate where presets need different versions. Check the
serving `/version` endpoint when benchmarking; the binary configured for an
ad-hoc launch can differ from the running unit's interpreter. Use existing
benchmark manifests, export and comparison checks after deploying the current
checkout. Preserve old results as historical evidence; do not manufacture absent
provenance or compare synthetic and live results as performance improvements.

## Controlled deployment and acceptance

The audit fixes are source changes only. Updating the desktop package also brings
in the checkout's earlier operation-lock, persistent-job and benchmark changes;
review that complete delta and plugin compatibility before deploying it.

For an approved maintenance window:

1. Record the current package pin, serving models, unit scopes, endpoints, GPU
   allocation and lease state. Save recoverable copies of configuration and a
   consistent SQLite backup using its backup mechanism (not a lone database
   file copy while WAL writes are active).
2. Update the existing deployment/package pin through the stack's established
   installation process. Retain a rollback package and matching database backup.
   Avoid an unreviewed editable install over production.
3. Restart the control-plane API deliberately so it imports the new package.
   Do not restart inference engines just to refresh inventory. Confirm that
   the CLI, API and plugin use the intended revision.
4. Check status/health and the second-engine row. Compare API preview/start
   settings using an isolated dry-run record, without scheduling a real model.
5. Separately authorize and schedule live acceptance: two one-card workloads,
   TP=2 takeover, failed-start rollback and recovery under a known test workload.
   Exercise long streams and competing requests when the router coordination
   fixes are ready. Validate Mac fallback without assuming an offline laptop is
   broken. Record actual results and restore the original workload.

No installation, service restart or disruptive acceptance test was performed as
part of the source audit and regression fixes.
