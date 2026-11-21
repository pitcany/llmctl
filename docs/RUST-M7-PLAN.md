# Rust successor — M7 plan, first half (distribution)

Status 2026-11-21: **draft, owner decisions open** (end of this file).
Nothing of M7 is built.
Parent: [RUST-SUCCESSOR-PLAN.md](RUST-SUCCESSOR-PLAN.md) (milestone table,
M7 row). Runs while the M5 pilot counts its seven days
([progress](RUST-M5-PROGRESS.md)); M6 (cutover) is not started.

M7 deliverable (parent): *portable installation/releases/docs, clean-host
CI, dependency/license checks, SGLang/FreeToken expansion.* Exit
criterion, verbatim: *Clean supported Linux/macOS control-plane
install/remove without workstation paths or a Python requirement for the
Rust agent; engine Python dependencies remain engine-specific.*

This plan covers the **distribution half** only. SGLang and FreeToken
come after M6 and get their own plan.

## Why now, and the rules that make it safe

The owner asked on 2026-11-21 to use the M5 pilot wait for M7. The owner
set three rules the same day:

1. **Distribution only.** No new engines.
2. **No agent redeploy** on the workstation, mbp2 or m5 before the M5
   exit (earliest 12:52 PST 2026-11-27). All M7 work is repo work, temp
   agents and throwaway hosts.
3. **Clean-host tests run locally:** Docker containers on the workstation
   and a clean macOS environment on mbp2. No hosted CI.

The 2026-11-20 review says to keep feature expansion behind the M5/M6
acceptance work ([review](RUST-REVIEW-2026-11-20.md), "Recommended next
sequence", step 5). Distribution changes no daily-use behavior and is
not deployed in the pilot window, so it does not compete with that work.
M5 audit checks and M6 work come first when they are due.

## What M7 (distribution) means in practice

At the end, a person with a supported Linux or macOS machine and no copy
of this workstation can:

1. Download one archive and check its `SHA256SUMS`.
2. Run `llmctl-next agent install`. The agent starts as a user service.
3. Run `llmctl-next status` and manage an engine they enroll.
4. Run `llmctl-next agent uninstall`. Nothing of llmctl-next stays
   behind except the data dir, and that goes too with `--purge`.

No step needs Python, Rust, Homebrew, `~/AI`, `~/local-llm` or this
owner's user name. Engines that are Python programs (vLLM, the Strata
lanes) keep their own Python, installed by their recipes, as today.

**Not in this half:** SGLang and FreeToken; publishing a GitHub release
(decision 7); deploying to the three hosts before the M5 exit; moving the
workstation agent from its `llmctl-next-m4a` data dir to the standard
layout (M6 or later); Linux on arm64, Windows, notarization, Homebrew
taps and `.deb`/`.rpm` packages.

## Facts that shape the plan (checked 2026-11-21, read-only)

**Workspace:** `rust/` has five crates (`llmctl-core`, `-daemon`,
`-client`, `-cli`, `-tui`), version 0.1.0, `license = "MIT"`,
`rust-version = "1.85"`. `Cargo.lock` has 194 packages and **no TLS
crate** (no `openssl`, `native-tls`, `rustls` or `ring`); SQLite is
`rusqlite` with `bundled`. A static `x86_64-unknown-linux-musl` build is
therefore possible. No `rust-toolchain.toml`: builds use whatever
toolchain each host has (workstation 1.99.1, mbp2 has its own).

**How agents are installed today (all by hand or by `host` verbs):**

| Host | Binary | Data dir | Service | Installed by |
|---|---|---|---|---|
| workstation | `~/.local/state/llmctl-next-m4a/llmctl-next.bin` | `~/.local/state/llmctl-next-m4a` | `llmctl-next.service` (user), linked from `~/AI/services/user/` | `cp` + `systemctl --user` by hand (steps in the unit's header) |
| mbp2 | `~/.local/bin/llmctl-next` | `~/.local/state/llmctl-next` | LaunchAgent, `enroll::AGENT_LABEL` | `host add/upgrade mbp2 --source` (rsync + `cargo build` on the Mac) |
| m5 | same as mbp2 | same | same | `host upgrade m5 --binary` (mbp2's build) |

There is **no Linux install verb**: `host add` is macOS-only
(`enroll.rs`: `ioreg`, `launchctl`). The unit file is hand-written.

**Workstation paths in non-test code** (`grep`, test fixtures excluded):

| Where | Path | Used for |
|---|---|---|
| `llmctl-daemon/src/presets.rs:91` | `~/AI/llm-router/config/router.yaml` | router `profile_env` backends |
| `llmctl-client/src/lease.rs:118` | `~/AI/logs/lane-lease.json` | longjob lane lease |
| `llmctl-client/src/bridge.rs:24` | `~/.local/bin/llmctl-next --data-dir ~/.local/state/llmctl-next` | remote bridge command (standard layout, not workstation-only) |
| `llmctl-daemon/src/acquire.rs:53` | `/opt/homebrew/bin`, `/usr/local/bin` | tool search on macOS (generic, keep) |
| `llmctl-next.service` | `PATH=…/AI/bin:/home/linuxbrew/…` | unit file, not code |

Presets come from `~/.config/llmctl/presets` (the Python llmctl's dir).
On a clean host the dir is absent and the agent has no presets. That is
correct behavior and needs only a test.

**Programs the agent starts** (`Command::new` in `crates/*/src`):
`systemctl`, `journalctl` (Linux); `launchctl` (macOS); `nvidia-smi`
(GPU hosts); `git`, `curl`, `uv` (engine installers and recipes);
`ssh`, `rsync` (CLI, remote hosts). No `python`/`python3` call. `uv`
installs engine Python and is engine-specific, which the exit criterion
allows.

**Tools on the workstation:** Docker 29.10 (server up). Missing:
`cargo-deny`, `cargo-about`, `cargo-audit`, `cargo-zigbuild`, `cross`,
`zig`, `podman`. Only the `x86_64-unknown-linux-gnu` target is installed.
mbp2: macOS 27.2.1, arm64, `cargo` and `rustc` in `~/.cargo/bin`.

**Repo and CI:** `pitcany/llmctl` is **public**. GitHub Actions works on
it (`test.yml`, Python only: ruff + pytest on 3.12/3.13; last run green
2026-11-21). The Rust workspace has no CI job. Releases `v0.1.0`–`v0.3.0`
are the Python llmctl. (The Actions billing stop in memory applies to
`pitcany/AI`, not to this repo.) The owner still chose local clean-host
tests (rule 3); Actions stays an option for later.

## Steps

Each step ends with a check. Effort figures are working time for one
session doing the work; the macOS steps add waiting on mbp2 builds.

### M7a. Remove workstation defaults (~1 day)

1. Add an agent config file, `<data-dir>/agent.toml` (decision 4), with
   optional keys `router_config` (path to router.yaml) and `lane_lease`
   (path to the lease file). Absent key = that integration is off, and
   `status` says so in one line.
2. `presets.rs` and `lease.rs` read the keys. No path under `~/AI` stays
   in non-test code.
3. Before the change reaches the workstation (after the M5 exit), write
   both keys with today's values into the m4a data dir. Add a startup
   warning: if the key is absent **and** the old default file exists,
   log `router_config not set; ~/AI/llm-router/config/router.yaml exists
   and is ignored`. This makes a missed config step visible instead of
   silent.
4. Add a gate test: scan `crates/*/src` (outside `#[cfg(test)]` and
   `tests/`) for `/home/`, `/Users/`, `AI/`, `local-llm`; fail on a hit
   not in an allowlist with a reason per entry.

Check: workspace suite green; a temp agent with an empty `HOME` starts,
answers `status`, lists zero presets and reports both integrations off;
the gate test fails on a planted path and passes after.

### M7b. `agent install` / `agent uninstall` / `agent status` (~1.5–2 days)

One code path for both OSes and for remote hosts:

| Verb | Linux | macOS |
|---|---|---|
| `agent install [--prefix ~/.local] [--data-dir …]` | copy self to `<prefix>/bin/llmctl-next` (atomic rename); write `~/.config/systemd/user/llmctl-next.service` (generated, minimal `PATH`); `daemon-reload`, `enable --now`; print the `loginctl enable-linger` line if lingering is off | copy self; write the LaunchAgent plist (`enroll::agent_plist`, moved to a shared module); `env -i launchctl bootstrap` |
| `agent uninstall [--purge]` | `disable --now`, remove unit and binary; `--purge` removes the data dir | `bootout`, remove plist and binary; `--purge` removes data dir and `~/Library/Logs/llmctl-next` |
| `agent status` | installed?, binary path and sha256, service state, socket answers, version | same |

Rules:
- **Install never touches an existing agent it did not write.** If a
  unit or plist with the agent's name exists and points to a different
  binary or data dir (the workstation's m4a unit is the case), refuse
  with the paths and the way out. No `--force` in M7.
- **Uninstall stops no engine.** It stops only the agent. Engines the
  agent ran keep running, as today when `llmctl-next.service` stops.
  The output says which workloads were owned and are now unmanaged.
- Every file written is listed in `<data-dir>/install-manifest.json`
  with its sha256. Uninstall removes exactly those files and refuses to
  remove a file whose sha256 changed (the owner edited it), naming it.
- `host add` and `host upgrade --binary` call `agent install` on the
  remote host after copying the binary, in place of their own
  bootstrap. `host upgrade --source` stays as is.

Check: unit tests for unit/plist render and manifest handling; temp-HOME
tests for refuse-on-foreign-unit, refuse-on-edited-file, uninstall
leaves the engines; the clean-host runs in M7e.

### M7c. Release builds (~1 day)

1. Add `rust/rust-toolchain.toml` pinning one stable version (decision:
   the version current at the time, e.g. 1.99.1), so every host builds
   with the same compiler.
2. `scripts/release.sh <version>` (repo root):
   - Linux: `x86_64-unknown-linux-musl`, static (decision 2), built in a
     pinned Docker image so the workstation's own libraries do not leak
     in. Check: `file` says "statically linked"; it runs in a bare
     `debian:12-slim` and `alpine` container.
   - macOS: `aarch64-apple-darwin`, built on mbp2 over ssh (same rsync
     path as `host upgrade --source`), ad-hoc signed by the linker as
     today. Minimum macOS set by `MACOSX_DEPLOYMENT_TARGET` (decide the
     floor when the build runs; record it).
   - Each archive `llmctl-next-<version>-<target>.tar.gz` holds the
     binary, `LICENSE`, `THIRD-PARTY-LICENSES.md` (M7d), `INSTALL.md`.
   - `SHA256SUMS` over all archives; a `BUILDINFO` file with git sha,
     toolchain, `Cargo.lock` sha256, image digest and build host.
   - `llmctl-next --version` prints version and git sha.
3. Output goes to `rust/dist/` (gitignored). Nothing is uploaded
   (decision 7).

Check: two builds of the same commit give archives with the same file
list and the same `--version` line; byte-identical builds are not
required in M7 (recorded as a known limit).

### M7d. Dependency and license gate (~0.5 day)

1. `cargo-deny` with `rust/deny.toml`: licenses allowlist (MIT,
   Apache-2.0, BSD-2/3-Clause, ISC, Unicode-3.0, Zlib; anything else
   fails with the crate named), `bans` (duplicate major versions warn),
   `sources` (crates.io only), `advisories` (RustSec database; network
   needed, so it runs in the gate, not in unit tests).
2. `cargo-about` generates `THIRD-PARTY-LICENSES.md` for the archives.
3. MSRV check: build with the 1.85 toolchain in Docker, or raise
   `rust-version` to what actually builds. The field must be true.
4. `cargo fmt --check` and `cargo clippy -- -D warnings` with the pinned
   toolchain (the review's step 2 asks for a pinned lint gate).
5. `scripts/check.sh` runs 1–4 plus the workspace test suite (in shards,
   as the Python suite) and prints one PASS/FAIL line per gate.

Check: `check.sh` passes on the workstation; a planted GPL-3.0 test
dependency in a scratch branch fails the license gate.

### M7e. Clean-host tests, local (~1–1.5 days)

`scripts/clean-host-test.sh linux|macos <archive>`; no GPU and no real
engine needed (the `fake` engine adapter is enough).

**Linux** (Docker on the workstation, systemd as PID 1, cgroup v2):
images `debian:12` and `ubuntu:24.04`, each with systemd and a plain
user with lingering on, **no Python, no Rust, no curl preinstalled**
beyond what the image ships. Steps:

1. Record the file list of the user's home and `/etc`.
2. Copy the archive in, check `SHA256SUMS`, unpack, `agent install`.
3. `agent status` = running; `status` answers; enroll a `fake`
   workload; `run` it, `stop` it; `ops` shows both operations.
4. `agent uninstall --purge`; the file list equals step 1.
5. Run `agent install` twice (idempotent) and `uninstall` on a host with
   nothing installed (clean refusal).

**macOS** (decision 6): the same steps on mbp2 as a user with no
Homebrew, no Rust and no `~/local-llm` in its environment, over ssh,
`env -i` for every step.

Plus one negative test per OS: a pre-existing hand-written unit/plist
for the agent is left untouched and install refuses (the workstation
case).

Check: both scripts pass on the M7c archives; the run log goes to
`docs/RUST-M7-CLEAN-HOST-<date>.md`.

### M7f. Install documentation (~0.5 day)

1. `docs/INSTALL.md` (Simplified Technical English): supported hosts
   table, download and check, install, first `status`, enroll an
   existing systemd unit or launchd job, upgrade, uninstall, what the
   agent runs (`systemctl`/`journalctl`/`launchctl`; `nvidia-smi` on GPU
   hosts; `git`/`curl`/`uv` only for engine installers), where logs and
   data live, how to report a bug.
2. Update `rust/README.md` (install section points to `INSTALL.md`;
   remove "M2 has started" status text that is out of date).
3. Check `~/AI/docs/ARTIFACT-INDEX.md`: if the published **Driving
   llmctl** guide describes install, add a dated note or update it.

Check: a dry read of `INSTALL.md` against the M7e logs: every command in
the document appears in a passing clean-host run.

## When the work reaches the three hosts

Not in the pilot window (rule 2). After the M5 exit, the order depends
on decision 5. In both cases the workstation config from M7a step 3 is
written before the new binary starts, and the existing m4a unit stays as
it is (install refuses to replace it by design). Moving the workstation
to the standard layout is a separate, later step with its own reversal.

## Exit checklist (maps to the parent criterion, distribution half)

| Criterion | Evidence required |
|---|---|
| Clean Linux install/remove | M7e Linux passes on `debian:12` and `ubuntu:24.04` from the release archive; file list after `--purge` equals before |
| Clean macOS install/remove | M7e macOS passes on the clean environment on mbp2 |
| No workstation paths | M7a gate test green; clean-host runs have no `~/AI` or `~/local-llm` |
| No Python requirement for the agent | clean-host images have no Python; install, status, run/stop and uninstall pass |
| Engine Python stays engine-specific | no change to recipes; `uv` remains the only Python entry, inside engine installers |
| Releases | M7c archives, `SHA256SUMS`, `BUILDINFO`, `--version` with git sha |
| Dependency/license checks | `check.sh` gates pass; `THIRD-PARTY-LICENSES.md` in every archive |
| Docs | `INSTALL.md` matches the passing clean-host runs |
| Pilot untouched | M5 `audit --all` unchanged by M7 work; no agent redeploy before the M5 exit |

## Owner decisions (open — recommendation first)

1. **Split into M7a–M7f** as above, a check at the end of each, about
   5.5–6.5 working days in total. *Recommended.*
2. **Linux binary:** static `musl` (recommended: one file, runs on any
   glibc or none, no TLS or system SQLite to link), or `glibc` built in
   an old-distro container (slightly faster allocator, but a glibc floor
   to state and test).
3. **Install layout for new installs:** user scope on both OSes,
   `~/.local/bin/llmctl-next` and `~/.local/state/llmctl-next`, user
   systemd unit / LaunchAgent (recommended: the same layout the Macs
   already use), or system-wide `/usr/local` with a system unit (needs
   root, and the agent drives user units today).
4. **Workstation-only defaults:** move to `agent.toml` keys, absent =
   off, with a startup warning when the old file exists (recommended),
   or keep the `~/AI` defaults when the file exists (no config step at
   deploy, but the exit criterion's "without workstation paths" fails a
   strict reading).
5. **Branch policy:** keep M7 on a branch `m7a-distribution` until M6
   exits, so M6 deploys the build the pilot proved (recommended), or
   merge to `main` as each step passes (no rebase work, but M6 then
   deploys untested-in-pilot code or must pin an older commit).
6. **Clean macOS environment:** a throwaway local user account on mbp2
   that the owner creates with `sudo` (recommended: minutes to set up,
   no download; it shares `/opt/homebrew`, so the test strips `PATH`
   with `env -i`), or a Tart macOS VM on mbp2 (truly clean, but a
   25–50 GB image and more setup).
7. **Publishing:** build archives locally only; publishing a GitHub
   release (public repo) is a separate owner go-ahead after M6
   (recommended), or publish a pre-release at the end of M7.
