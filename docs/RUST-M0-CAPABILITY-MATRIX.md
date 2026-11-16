# M0: engine/host capability matrix and open questions

Status: first cut, 2026-11-15. Companion to
[RUST-SUCCESSOR-PLAN.md](RUST-SUCCESSOR-PLAN.md) ("M0: contracts and design
spikes"). Rows record what the repo/plan already establish; **open**
questions are the ones that need the actual machines or the owner's
decisions — they are surfaced here rather than silently deferred.

## Engine → host → ownership path

| Engine | Linux (workstation) | macOS (Macs) | Install/version ownership path | State |
| --- | --- | --- | --- | --- |
| vLLM | Required (day one) | out of scope | Managed Python env + pinned version; systemd unit; env-file launch definition (existing `vllm-engine.service` pattern) | Established in Python today; Rust adapter = M2 pilot | 
| llama.cpp | Required | Required (LM Studio-independent serving) | Managed prebuilt `llama-server` binary matched to CUDA (Linux) / Metal builds (macOS); launchd on macOS | Python does Linux today; macOS + Rust = M3 |
| Ollama | Required | Required (present on Macs) | Official installer under management; engine-owned model lifecycle via its API | Python does Linux; verify unattended macOS service control (open Q3) |
| LM Studio | Required | Required | Vendor app install + `lms` CLI service controls; verify unattended capabilities before claiming parity | **Open Q3** |
| MLX | — | Required | **Open Q1**: which serving package counts as "the MLX server" (mlx-lm server, lmstudio-mlx, custom) must be identified before an adapter exists | Blocked on Q1 |
| Strata | Required | unknown | Python repo knows the installer; Rust needs the recipe re-verified on the target host | **Open Q4** |
| Custom servers | Required (owner's named servers) | as applicable | **Open Q2**: recipes (install/build/run/health/stop) must be declared by the owner; not inferred | Blocked on Q2 |
| Generic OpenAI endpoints | Connection/health only | same | Observe-only `ControlAuthority` (implemented in `llmctl-core`); lifecycle needs an enrolled host + known recipe | Type support done; enrollment = M3 |
| SGLang, FreeToken | Later | Later | Import records without lifecycle claims | Deferred by plan |

## Cross-cutting ownership rules (decided, encoded in the plan)

- One authoritative version/environment ref, launch definition, lifecycle
  policy and upgrade history per managed instance; systemd/launchd stay the
  supervisors.
- Version ownership ≠ auto-update: pin, stage, validate, drain, promote
  explicitly, keep rollback. No in-place mutation of a serving Python env.
- Shared model caches are tracked separately from engine installs; removal
  never deletes shared weights.
- External endpoints stay externally owned even when connections are
  managed; unsupported operations are declared blockers, not pretend-owns.

## Open questions (owner input or machine access needed)

1. **MLX serving identity (blocks the MLX adapter):** which implementation
   is "the MLX server" on the Macs — `mlx-lm` server, LM Studio's MLX
   runtime, or a custom runner? Needs a look at what actually serves on
   mbp2 / the M5.
2. **Custom-server recipes:** the owner's named servers need declared
   install/build/run/health/stop recipes. Which servers are day-one?
3. **LM Studio unattended capabilities:** does `lms` (or the app's headless
   mode) support install/start/stop/model-load without a logged-in GUI
   session on macOS? Required before claiming parity.
4. **Strata host support:** verify the installer/upgrade path on the
   intended host(s) and whether macOS is in scope.
5. **launchd process identity (M3 spike):** PID identity, service domains,
   login-session behavior and restart persistence must be validated on a
   real Mac; Linux mocks cannot establish this (plan, verification
   strategy). The `UnitScopeMarker::Launchd` type exists; behavior is
   unverified.
6. **YAML reader for config parity (M5):** `serde_yaml` is archived; choose
   a maintained reader against current docs before implementing preset
   parity. Not blocking M1–M3.

## Spikes completed in this batch (M0/M1 overlap)

- **Unix-socket HTTP server/client:** axum over `UnixListener` (0600),
  hyper client with a Unix connector — exercised by every e2e test.
- **SQLite execution boundary:** rusqlite behind one bounded worker thread;
  every journal write its own transaction; crash-consistency validated by
  the SIGKILL/restart e2e tests.
- **systemd lifetime:** deferred to M2 by design (the plan adopts existing
  named units first; the fake engine stands in until then). The
  `Engine` port (`crates/llmctl-daemon/src/engine.rs`) is where the
  systemd/launchd adapters land.

## Compatibility fixtures (frozen contract surfaces)

The wire DTOs in `rust/crates/llmctl-client/src/wire.rs` are the first
frozen fixtures of the successor's own API (protocol 1): error codes, op
statuses (`admitted executing cancel_requested succeeded failed interrupted
needs_attention canceled`), phase statuses (`pending running
effect_pending done failed skipped`), and CLI exit codes
(0 ok · 1 failed · 3 no-daemon · 4 conflict · 5 unknown · 6 invalid).
Python-verb JSON fixtures (`ls`/`status`/`run`/`stop`/`logs`/`doctor`)
get frozen when the compatibility importer lands (M4/M5).
