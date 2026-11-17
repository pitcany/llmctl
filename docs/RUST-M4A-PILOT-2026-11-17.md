# Rust successor — M4a pilot (installs and versions, 2026-11-17)

Parent: [RUST-M4-PLAN.md](RUST-M4-PLAN.md), sub-milestone M4a. Code:
commits `8153d7c`..`ed009ef` on `main` (core types, five-kinds
installers, vendor observer, CLI/API, hardened-unit pin coverage, four
pilot-driven fixes).

## What was built

| Piece | Evidence |
|---|---|
| `EngineKind`, `InstallRecipe` (5 kinds), `VersionFact`, inventory migration | `llmctl-core` tests incl. legacy-file migration (`inventory.rs`) |
| Kind-aware probe / validate / definition rewrite | `version.rs` tests; no `VLLM` default outside the vLLM recipe |
| `release-archive` + `source-build` installers (staging dir, checksum, binary check, promote = rename, manifest last) | systemd spikes: tiny-archive and tiny-C-source tests (`systemd_spike.rs`, `--ignored`) |
| `vendor-app` observer + owner-run plan (never executed) | spike: pin, drift finding, plan output, stage refusal |
| `version` / `install` / `upgrade` CLI verbs, `/v1/installs/plan`, `/v1/workloads/{name}/version` | e2e suite green (24 tests) + spikes |
| Hardened-unit pin coverage (Documentation, ExecStopPost, KillSignal, Standard*, State/CacheDirectory, sandboxing, ReadWritePaths) | `own` on `gpt-oss-120b` passes the scratch-unit round trip |

## Pilot — workstation (llama.cpp, gpt-oss-120b)

Target: the `gpt-oss-120b` user unit (inactive, b10705 in
`~/AI/llama.cpp/build`). Daemon: dev data dir
`~/.local/state/llmctl-next-m4a`. Recipe: `source_build`
(git ggml-org/llama.cpp, arch 120, tools-only, `$ORIGIN` rpath),
version `a9d43ecc3` (upstream master 2026-11-17).

- **One-line unit edit** (backed up): `Environment=LLAMA_GUARD_REAL_BIN=`
  now names the versioned binary — the definition points at a version
  directory (M4 decision 3). The lane's exec path rides this env var;
  promote swaps it.
- **Stage**: `install stage gpt-oss-120b a9d43ecc3` cloned, built
  (CUDA, tools-only) and validated offline — binary exists, executable,
  `--version` contains the commit (`1.0.0-dev build 12492`). Build log
  and manifest (`staged.json`, incl. the 40-char commit) kept with the
  version dir.
- **Promote** (window 5, longjob `--stop-engine --router`, speaches
  container stopped/restarted around it): admitted operation — stop,
  apply definition (one env-var path swap), start, readiness, served
  model `gpt-oss-120b` ✓, binary version `a9d43ecc3` ✓; curl :8005
  confirmed serving. Version report showed the fact pin move with the
  promote.
- **Rollback** (window 6): Succeeded — definition back to `build/`,
  served model re-verified, the running binary reported
  `0.3.0-dev (build 10705, commit 257813839)` (the original), then a
  daemon `stop` restored the pre-pilot inactive state. Full chain:
  **stage → promote → rollback → stop**, each an admitted operation
  with its journal row.
- GPU refusal evidence: window 1 refused the promote while vllm held
  the cards (`gpu_occupied`, nothing journaled) — the never-free guard
  doing its job on a real engine.

Pilot-driven fixes (each committed):

1. A refused admission no longer marks the stage `failed` (it poisoned
   a healthy build into a full restage).
2. Definition rewrite maps the primary executable exactly across
   differing install layouts (live `build/bin` vs staged
   `<id>/build/bin`); env-var paths under the recipe root swap too.
3. systemd starts are queued `--no-block`: `ExecStartPost` (the
   ds4-sync hook waits up to 20 min) held `systemctl start` open far
   past the effect timeout; the daemon's own AwaitReadiness is the
   came-up authority.
4. Rollback version check stays fail-closed (the promote stashes the
   old binary's actual `--version` output) and a rollback whose
   recovery restored the world settles the row back to `promoted`
   (retryable) — both from the automated security review of the
   commits.

Recipe lessons (llama.cpp source builds): pin
`-DCMAKE_CUDA_ARCHITECTURES=120` (cmake 3.22 has no `native`),
`-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON` (promote renames the dir; a baked
absolute rpath dangles), `-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF`
(brew binutils `ld` fails linking test binaries), and force
`PATH=/usr/bin:/bin:/usr/local/cuda/bin` in the build command (brew
ld breaks the tools' links).

## Pilot — m5

Agent enrolled with the M4a binary (built on mbp2 from the workstation
tree — m5 has no cargo; relayed via the workstation since m5↔mbp2 scp
was refused). Inventory records:

| Workload | Recipe | Fact |
|---|---|---|
| `ds4` | `source_build` (git pitcany/ds41-m5, ref `bd450f844e3717daafc5c5c79f044f610b3e45a1`, no rebuild — the pin record the plan asks for) | fork SHA recorded |
| `mlxserve` | `release_archive` (root `~/local-llm/runtimes`) | 26.10.1 live / 26.9.5 rollback present on disk |
| `ollama` | `vendor_app` (`http_json /api/version`) | **running 0.34.3** — the plan's live drift example (brew has 0.40.16; the next restart silently upgrades it) |

`install plan ollama` over the bridge prints the observed version, the
pin, drift, and the owner-run reinstall path; `install stage` refuses.

**Not done in M4a** (blocked on M4b by owner decision 4): the actual
mlx-serve 26.10.1 ↔ 26.9.5 swap — mlx-serve runs as a llama-swap lane,
and lane records with a round-trip import come first; hand-editing
`config.yaml` is exactly what the decision forbids. The mbp2 tiny
archive/C spikes are also open (the new spikes are systemd-shaped;
the launchd flavor lands with M4b's Mac adapter work).

## Open items for M4b

- The pilot daemon stays up on `~/.local/state/llmctl-next-m4a`,
  owning `gpt-oss-120b` (rollback restored the original definition;
  `release` hands it back if wanted). The staged `a9d43ecc3` dir stays
  under `~/AI/llama.cpp/` until `upgrade gc`.
- Settle's row semantics after a restored rollback are now
  promote-preserving, but the promote row of a *failed-and-restored*
  promote still reads `failed` — the M2 wording survives; revisit if
  M4b hits it.
