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
| LM Studio | Required | Required | Vendor app install + `lms` CLI; headless core `llmster` via `lms daemon up/down/status` (LM Studio ≥ 0.4.0) | **Q3 answered 2026-11-16** (needs a console login on macOS; see below) |
| MLX | — | Required | **Q1 answered 2026-11-16:** no single MLX server. Engines run as recipes under llama-swap: `mlx-serve` (ddalcu/mlx-serve, native Zig binary, versioned runtime dirs), MTPLX, `mlx-vlm` server; plus LM Studio's MLX runtime | Unblocked: model as llama-swap-managed recipes (M3) |
| Strata | Required | out of scope (owner, 2026-11-16: cannot run on macOS) | Python repo knows the installer; Rust needs the recipe re-verified on the workstation | **Q4: Linux only**; installer re-verification remains (M3) |
| Custom servers | ds4 (CUDA build in `~/Work/ds41-workstation-*`, no unit yet) | ds4 (Metal, `~/Work/ds41-1017/mac-local/serve.sh`), MTPLX | **Q2 answered 2026-11-16:** day one = antirez's **ds4** (owner's fork `pitcany/ds41-m5`; C, build from source) and **MTPLX** (`youssofal/mtplx`, pip, MLX; Macs only). Lifecycle comes from llama-swap's config; only install/version recipes remain | Install/version: M3 |
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

## Open questions — answered 2026-11-16

Checked read-only on mbp2 (macOS 27.2.1) and the M5 (macOS 27.0.1), plus
one disposable launchd agent on mbp2 (created, exercised, booted out).

1. **MLX serving identity — answered.** There is no single "MLX server".
   On both Macs **llama-swap** (launchd agent `com.pitcany.llama-swap`,
   `:8099`, `KeepAlive` + `RunAtLoad`) starts every model engine on demand
   as its own child process:
   - `mlx-serve` (ddalcu/mlx-serve, native Zig binary, no Python) — M5,
     26.10.1, with the previous runtime dir (26.9.5) kept for rollback;
   - **MTPLX** (`mtplx` pip package, dedicated venvs: 2.11.0 and 2.12.0 on
     mlx 0.32.2) — the multi-token-prediction MLX engine;
   - `mlx-vlm` server (dedicated venv, 0.6.6) — the vision lane;
   - **ds4** (`serve.sh`, GGUF DeepSeek v4.1);
   - plus LM Studio's own MLX runtime, separately.

   Consequence for M3: the Mac "MLX adapter" is a set of **recipes under
   llama-swap**, and llama-swap is the supervised unit. Versioned runtime
   dirs and per-engine venvs already match the plan's staged-upgrade model.
2. **Custom-server recipes — owner answer.** Day one: **ds4** (antirez's
   engine; owner fork `pitcany/ds41-m5` on the M5, a CUDA build in
   `~/Work/ds41-workstation-*` on the workstation) and **MTPLX** (Macs).
   *Refined 2026-11-16 (owner):* because llmctl-next supervises llama-swap
   on the Macs, run / health / stop are already declared per engine in
   llama-swap's config (13 entries, each with `cmd`, `proxy`,
   `checkEndpoint`, `ttl`), so no separate lifecycle recipes are needed for
   ds4 or MTPLX. What remains is **install and version**. *Decided
   2026-11-16 (owner): llmctl-next owns engine versions under llama-swap.*
   Per engine: build ds4 from a pinned git sha of the fork, create an MTPLX
   venv at a pinned version (likewise `mlx-serve` runtime dirs and the
   `mlx-vlm` venv) — the same stage → promote → rollback flow as the M2
   vLLM pilot, where "apply" rewrites that engine's llama-swap entry
   (`cmd` / macro path) instead of a systemd unit, and llama-swap's
   `-watch-config` picks it up. The workstation has no llama-swap; running ds4 there would
   need a systemd unit (the M2 adoption path).
3. **LM Studio unattended capabilities — answered, with one limit.** Since
   0.4.0, LM Studio ships `llmster`, a headless daemon (`lms daemon
   up/down/status`), plus `lms server start/stop/status`, `lms load/unload`,
   `lms get` and `lms runtime` for models and runtimes. The M5 runs it
   headless today (`LM Studio --run-as-service`, v0.4.25, parented to
   launchd, started from a terminal session); mbp2 runs the GUI app
   (v0.4.26). `lms` works over SSH against a running daemon. `lms daemon
   down` stops only `llmster`, never the GUI app. The limit is macOS's,
   not LM Studio's: the documented no-GUI startup task is for Linux; on a
   Mac with a GUI the service starts at *login*. So "unattended" on macOS
   means "a user is logged in" (next item). Install stays a vendor-app step.
4. **Strata host support — owner answer.** Linux only (cannot run on
   macOS). Re-verifying the installer on the workstation remains (M3).
5. **launchd process identity — answered by a disposable-agent spike.**
   - Identity: label + domain + `runs` (incremented on every spawn, like a
     systemd invocation id) + PID + process start time (`ps -o lstart`).
     That tuple is safe against PID reuse.
   - Exit cause is reported (`last terminating signal = Killed: 9`, `last
     exit code`).
   - With `KeepAlive`, SIGKILL **and** SIGTERM are both respawned (runs went
     1→4 across kill, `kickstart -k` and `launchctl kill SIGTERM`). A stop
     must be `launchctl bootout`, which unloaded the agent and ended its
     process. A *durable* manual stop also needs `launchctl disable`,
     because an agent in `~/Library/LaunchAgents` with `RunAtLoad` comes back
     at the next login (not exercised: it writes launchd's persistent
     override database).
   - Domain: every agent runs in `gui/501`; bootstrapping into `user/501`
     fails for the user (I/O error 5). `gui/501` exists only while the user
     is logged in at the console: the M5 has been logged in since Oct 6 and
     has no auto-login set. After a reboot nothing runs until someone logs
     in. Not exercised (disruptive): behaviour across logout and reboot.
6. **YAML reader — answered: `serde-saphyr`.** Checked on crates.io and
   RustSec (2026-11-16):

   | Crate | Latest | Last release | Status |
   |---|---|---|---|
   | `serde_yaml` | 0.9.34+deprecated | 2024-03 | archived |
   | `serde_yml` | 0.0.13 | 2026-05 | **RUSTSEC-2025-0068: unsound and unmaintained**, repo archived |
   | `serde_yaml_ng` | 0.10.0 | 2024-05 | fork on the unmaintained `unsafe-libyaml` |
   | `serde_norway` | 0.9.42 | 2024-12 | fork on `unsafe-libyaml-norway`; quiet for ~2 years |
   | `yaml-rust2` | 0.13.0 | 2026-09 | maintained, no serde |
   | `saphyr` | 0.3.4 | 2026-11 | maintained parser, no serde |
   | **`serde-saphyr`** | 2.1.1 | 2026-11 | serde on the pure-Rust saphyr parser; no advisories; ~15 M recent downloads, 648 dependents; repo active |

   Choose `serde-saphyr`; fallback `serde_norway` if a serde_yaml-identical
   API is ever needed. The configs to read (`settings.yaml`,
   `model_dirs.yaml`, `router.yaml`) use no anchors, merge keys or tags, so
   any of them parses today; pin the choice with fixtures at M5.

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
