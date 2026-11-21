# Rust successor — M7 plan, first half (distribution)

Status 2026-11-21: **plan agreed — all eight owner decisions taken, each
as recommended** (end of this file).
Reviewed the same day by an independent read-only review (Claude Fable
5.2 subagent); its findings are folded in below (one blocker, ten
should-fix). Nothing of M7 is built.
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
   and a clean macOS environment on mbp2. No hosted CI. This is an
   owner-approved deviation from the parent's "clean-host CI": the local
   scripts must run from a fresh clone at a recorded commit.

The review added a fourth rule, needed to keep rule 2:

4. **No real service-manager call on the three hosts from M7 code or
   tests.** `systemctl --user` and `launchctl gui/<uid>` address the real
   user manager whatever `HOME` is. A temp-`HOME` test of `agent
   uninstall` on the workstation or mbp2 would stop the live pilot agent.
   Service-manager calls sit behind a trait, faked in unit tests; real
   calls run only inside the Docker containers and as the throwaway
   macOS user.

The 2026-11-20 review says to keep feature expansion behind the M5/M6
acceptance work ([review](RUST-REVIEW-2026-11-20.md), "Recommended next
sequence", step 5). Distribution changes no daily-use behavior and is
not deployed in the pilot window, so it does not compete with that work.
M5 audit checks and M6 work come first when they are due.

## What M7 (distribution) means in practice

At the end, a person with a supported host (x86_64 Linux with a systemd
user manager, or arm64 macOS at or above the recorded floor) and no copy
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
layout (M6 or later); rewiring `host add` / `host upgrade` onto `agent
install` (after M6, see M7b); the workspace-wide `cargo fmt` commit
(after M6, see M7d); Linux on arm64, Windows, notarization, Homebrew taps
and `.deb`/`.rpm` packages.

## Facts that shape the plan (checked 2026-11-21, read-only)

**Workspace:** `rust/` has five crates (`llmctl-core`, `-daemon`,
`-client`, `-cli`, `-tui`), version 0.1.0, `license = "MIT"`,
`rust-version = "1.85"`. `Cargo.lock` has 194 packages and **no TLS
crate** (no `openssl`, `native-tls`, `rustls` or `ring`); SQLite is
`rusqlite` with `bundled`, the only C dependency. A static
`x86_64-unknown-linux-musl` build is therefore possible (the build image
needs a musl C compiler for SQLite). No `rust-toolchain.toml`.

**`rust-version = "1.85"` is not true.** The lock holds `uuid 1.29.2`
and `darling 0.24.1`; the review reports they need 1.89 and 1.88. M7d
raises the field instead of testing 1.85.

**Lint state** ([review](RUST-REVIEW-2026-11-20.md) lines 149–150):
`cargo fmt --check` fails in 85 files; `cargo clippy -D warnings` fails
in core with six diagnostics.

**How agents are installed today (all by hand or by `host` verbs):**

| Host | Binary | Data dir | Service | Installed by |
|---|---|---|---|---|
| workstation | `~/.local/state/llmctl-next-m4a/llmctl-next.bin` | `~/.local/state/llmctl-next-m4a` | `llmctl-next.service` (user), a symlink to `~/AI/services/user/llmctl-next.service` | `cp` + `systemctl --user` by hand (steps in the unit's header) |
| mbp2 | `~/.local/bin/llmctl-next` | `~/.local/state/llmctl-next` | LaunchAgent `com.pitcany.llmctl-next` with `--host-id mbp2` | `host add/upgrade mbp2 --source` (rsync to `llmctl-next-src/`, `cargo build` on the Mac, move into `~/.local/bin`) |
| m5 | same as mbp2 | same | same, `--host-id m5` | `host upgrade m5 --binary` (mbp2's build) |

There is **no Linux install verb**: `host add` is macOS-only
(`enroll.rs`: `ioreg`, `launchctl`). No install has a manifest.
`host add` without `--binary`/`--source` reuses a remote binary when the
**last whitespace token** of its `--version` output equals the local
version (`enroll.rs:361-363`).

**Owner-specific defaults in non-test code** (test fixtures excluded):

| Where | Default | Existing override |
|---|---|---|
| `llmctl-daemon/src/presets.rs:91` | `~/AI/llm-router/config/router.yaml` | `LLMCTL_ROUTER_CONFIG` (`presets.rs:83`) |
| `llmctl-client/src/lease.rs:118` | `~/AI/logs/lane-lease.json` | `LLMCTL_NEXT_LANE_LEASE` (`lease.rs:16`; `off` disables) |
| — read by the daemon (`coord.rs:81-85`) **and the CLI** (`holder_ancestor`, `lease.rs:142-149`, called from `main.rs:763`, `tui_cmd.rs:21`) | | |
| `llmctl-daemon/src/app.rs:173` | host id `"workstation"` | `--host-id` |
| `enroll.rs:19`, `records.rs:79`, `plistdef.rs:581` | label `com.pitcany.llmctl-next` | none (renaming breaks the Macs) |

Generic defaults, kept: `~/.config/llmctl/presets` and
`~/.config/llm-models` (`presets.rs:72`), `settings.yaml`
(`presets.rs:78`), `~/.local/share/llmctl/operations.lock`
(`oplock.rs:25`), `/opt/homebrew/bin` and `/usr/local/bin`
(`acquire.rs:53`), the standard layout in `bridge.rs:24`. On a clean
host the preset dirs are absent and the agent has no presets: correct
behavior that needs only a test.

**Programs the agent starts** (`Command::new` and helpers in
`crates/*/src`, corrected by the review):

| Group | Programs |
|---|---|
| Linux service manager | `systemctl`, `journalctl`, `busctl` (`unitdef.rs`) |
| macOS service manager | `launchctl`, `plutil` (`plistdef.rs`), `id` (`launchd.rs:71`), `env` (`launchd.rs:850`) |
| Disk and archives | `df`, `du` (`weights.rs:281`), `tar`, `unzip`, `sh` (`upgrade.rs`) |
| GPU hosts | `nvidia-smi` |
| Engine installers and engines | `git`, `curl`, `uv` (bare, `upgrade.rs:322`), `hf` (`acquire.rs:149`), `lms` (`engine_api.rs:321`), `ollama` (`weights.rs:400`) |
| Engine interpreter | `<engine env>/bin/python -I -c …` (`version.rs:912`, `lane_version.rs:160`): the engine's own Python, to read its version |
| Owner hooks | programs named in hook files (`hooks.rs:30`) |
| CLI only | `ssh`, `rsync` |

The agent itself never needs a system Python. The engine-interpreter
call runs only for an engine that has one, which the exit criterion
allows ("engine Python dependencies remain engine-specific").

**Tools on the workstation:** Docker 29.10 (server up). Missing:
`cargo-deny`, `cargo-about`, `cargo-audit`, `cargo-zigbuild`, `cross`,
`zig`, `podman`. Only the `x86_64-unknown-linux-gnu` target is installed.
mbp2: macOS 27.2.1, arm64, `cargo` and `rustc` in `~/.cargo/bin`.

**Repo and CI:** `pitcany/llmctl` is **public**. GitHub Actions works on
it (`test.yml`, Python only; last run green 2026-11-21). The Rust
workspace has no CI job. Releases `v0.1.0`–`v0.3.0` are the Python
llmctl. The Actions billing stop applies to `pitcany/AI`, not to this
repo. The owner still chose local clean-host tests (rule 3).

## Steps

Each step ends with a check. Effort figures are working time for one
session doing the work; the macOS steps add waiting on mbp2 builds.
Order: M7d (report-only) first, so every later step runs against the
gate.

### M7d. Dependency and license gate, report-only (~0.5–1 day)

1. `cargo-deny` with `rust/deny.toml`: licenses allowlist (MIT,
   Apache-2.0, BSD-2/3-Clause, ISC, Unicode-3.0, Zlib; anything else
   fails with the crate named; the review checked 157 Linux packages
   against it), `bans` (duplicate major versions warn), `sources`
   (crates.io only), `advisories` (RustSec; network needed). An
   **unmaintained ignore list** with a reason per entry
   (`serde_yaml 0.9.34+deprecated`, `paste`: expected findings).
   macOS-only crates are checked on mbp2 once.
2. Raise `rust-version` to the lowest version the lock builds with
   (`cargo +<ver> check --locked`). No Docker MSRV build.
3. `cargo fmt --check` and `cargo clippy -- -D warnings` run
   **report-only** in M7. The 85-file format commit lands on `main` as
   one mechanical commit right after the M6 exit; the six clippy fixes
   go with it. Doing either on the M7 branch would conflict with every
   M5/M6 fix on `main` (decision 5).
4. `scripts/check.sh` runs 1–3 plus the workspace test suite (in shards)
   and prints one PASS/FAIL/REPORT line per gate.

Check: `check.sh` gives PASS for licenses, sources and tests, REPORT for
fmt/clippy; a planted GPL-3.0 dependency in a scratch branch fails the
license gate.

### M7a. Remove workstation defaults (~1 day)

1. Add an agent config file, `<data-dir>/agent.toml` (decision 4), with
   optional keys `router_config` and `lane_lease`. Precedence: env
   override (`LLMCTL_ROUTER_CONFIG`, `LLMCTL_NEXT_LANE_LEASE`) first,
   then the key, then nothing. Absent = that integration is off, and
   `status` says so in one line.
2. **One reader for daemon and CLI.** `holder_ancestor()` (CLI, under a
   longjob job) and `coord.rs` (daemon) resolve the lease path through
   the same function. Otherwise longjob-driven calls lose the
   under-lease header after cutover and are refused.
3. Before the change reaches the workstation (after the M5 exit), write
   both keys with today's values into the m4a data dir. Add a startup
   warning: if the key is absent **and** the old default file exists,
   log `router_config not set; ~/AI/llm-router/config/router.yaml exists
   and is ignored`.
4. Gate test: scan `crates/*/src` (outside `#[cfg(test)]` and `tests/`)
   for `/home/`, `/Users/`, `AI/`, `local-llm`, `workstation`,
   `pitcany`; fail on a hit not in an allowlist with a reason per
   entry. The label `com.pitcany.llmctl-next` is allowlisted as a vendor
   id. The `"workstation"` host-id default becomes the machine's
   hostname (short form), with `--host-id` still winning; the workstation
   keeps its id because its unit passes none today and the M6 deploy
   adds `--host-id workstation`.

Check: workspace suite green; a temp agent with an empty `HOME` starts,
answers `status`, lists zero presets and reports both integrations off;
a test proves `holder_ancestor()` honours the key; the gate test fails on
a planted path and passes after.

### M7b. `agent install` / `agent uninstall` / `agent status` (~2 days)

| Verb | Linux | macOS |
|---|---|---|
| `agent install [--prefix ~/.local] [--data-dir …] [--host-id …]` | copy self to `<prefix>/bin/llmctl-next` (atomic rename); write `~/.config/systemd/user/llmctl-next.service` (generated; `PATH` = `<prefix>/bin` + system dirs, so a bare `uv` resolves); `daemon-reload`, `enable --now`; print the `loginctl enable-linger` line if lingering is off | copy self; write the LaunchAgent plist (`enroll::agent_plist`, moved to a shared module, `--host-id` passed through); `env -i launchctl bootstrap` |
| `agent uninstall [--purge]` | `disable --now`, remove unit, the `default.target.wants/` link and binary; `--purge` removes the data dir | `bootout`, remove plist and binary; `--purge` removes data dir and `~/Library/Logs/llmctl-next` |
| `agent status` | installed?, binary path and sha256, service state, socket answers, version | same |

Rules:
- **Service-manager calls go through a trait** (rule 4). Unit tests use
  a fake. A hidden `--service-name` / `--label` override lets real-call
  tests in Docker and as the macOS test user use their own names.
- **Install never touches an agent it did not write.** A unit or plist
  with the agent's name that points to a different binary or data dir
  (the workstation's m4a unit) is refused, with the paths and the way
  out. No `--force` in M7.
- **Adopt rule for pre-manifest installs:** an install in the standard
  layout with no manifest (the Macs today) is adopted only by an
  explicit `agent install --adopt`, which writes the manifest from what
  is on disk and changes nothing else. Not run on mbp2/m5 before the M5
  exit.
- **Uninstall with no manifest takes no service action** and refuses,
  naming what it found. It never runs `disable`, `stop` or `bootout` on
  an install it cannot prove it wrote.
- **Uninstall stops no engine.** It stops only the agent and names the
  workloads that are now unmanaged.
- The manifest (`<data-dir>/install-manifest.json`) lists every file
  with its sha256 **and every directory it created**. Uninstall removes
  those files, then those directories if empty, and refuses to remove a
  file whose sha256 changed (the owner edited it), naming it.
- **Binary upgrades update the manifest.** `agent install` over its own
  install replaces the binary and the manifest entry. `host upgrade
  --source` keeps its current behavior in M7; after M6 it calls the
  same update.
- **`host add` / `host upgrade --binary` are not rewired in M7.** They
  are the M6 deploy path and cannot be tested against the real Macs
  before the pilot exit. Rewiring them onto `agent install` is the first
  step after M6.

Check: unit tests (fake service manager) for unit/plist render, manifest
files + dirs, adopt, refuse-on-foreign-unit, refuse-without-manifest,
refuse-on-edited-file, uninstall leaves the engines; real calls only in
M7e.

### M7c. Release builds (~1 day)

1. Add `rust/rust-toolchain.toml` pinning one stable version (the one
   current when the step runs). Pre-install that toolchain on mbp2 first
   (`rustup toolchain install`), so the next `host upgrade --source`
   after the pilot does not stall on a download.
2. `scripts/release.sh <version>` (repo root):
   - Linux: `x86_64-unknown-linux-musl`, static (decision 2), built in a
     pinned Docker image with a musl C compiler. Check: `file` says
     "statically linked"; it runs in a bare `alpine` container.
   - macOS: `aarch64-apple-darwin`, built on mbp2 over ssh in its **own
     directory `llmctl-next-release-src/`**. It never calls
     `build_on_host` and never writes `~/.local/bin` (that would be a
     redeploy). Ad-hoc signed by the linker as today. Minimum macOS from
     `MACOSX_DEPLOYMENT_TARGET`; record the floor.
   - Each archive `llmctl-next-<version>-<target>.tar.gz` holds the
     binary, `LICENSE`, `INSTALL.md`.
   - `SHA256SUMS` over all archives; `BUILDINFO` with git sha,
     toolchain, `Cargo.lock` sha256, image digest and build host.
   - `--version` stays one line, `llmctl-next <version>` (the parser in
     `enroll.rs:361-363` reads its last token). The git sha goes into a
     new `llmctl-next version --json`.
3. Output goes to `rust/dist/` (gitignored). Nothing is uploaded
   (decision 7). `THIRD-PARTY-LICENSES.md` (cargo-about) is generated
   when publishing is approved, not in M7.

Check: archives build from a fresh clone at a recorded commit;
`SHA256SUMS` verifies; `version --json` shows the commit; `--version`
output unchanged in shape.

### M7e. Clean-host tests, local (~1–1.5 days)

`scripts/clean-host-test.sh linux|macos <archive>`; no GPU and no real
engine (the `fake` engine adapter is enough).

**Linux:** one `debian:12` image with systemd as PID 1, run **without
`--privileged`**, private cgroup namespace, no host D-Bus socket; the
image adds `systemd`, `dbus-user-session`, `libpam-systemd` and a plain
user with lingering on. **No Python, no Rust, no curl.** The script
asserts that `busctl`, `df` and `du` exist (the agent needs them). Plus a
bare `alpine` run of the static binary (`--version`, `status` against a
temp data dir, no service manager).

1. Record the file and directory list of the user's home and `/etc`.
2. Copy the archive in, check `SHA256SUMS`, unpack, `agent install`.
3. `agent status` = running; `status` answers; enroll a `fake`
   workload; `run` it, `stop` it; `ops` shows both operations.
4. `agent uninstall --purge`; the list equals step 1, except entries the
   script names as allowed (none expected; any is a finding).
5. `agent install` twice (idempotent); `uninstall` with nothing
   installed (clean refusal, no service call); a hand-written unit for
   the agent is left untouched and install refuses (the workstation
   case).

**macOS** (decision 6): the same steps on mbp2 as a throwaway user.
**Open point, check first:** `launchctl bootstrap gui/<uid>` probably
needs that user to have a GUI login session; over ssh alone the domain
may not exist (review, moderate confidence). If so, the user logs in once
by fast user switching, and `INSTALL.md` states the requirement; the
alternative to test is the `user/<uid>` domain. The macOS run proves
install/remove and no Homebrew/Rust; it does **not** prove "no Python"
(`/usr/bin/python3` ships with the OS stub). "No Python" is proven on
Linux.

Check: both scripts pass on the M7c archives from a fresh clone; the run
log goes to `docs/RUST-M7-CLEAN-HOST-<date>.md`.

### M7f. Install documentation (~0.5 day)

1. `docs/INSTALL.md` (Simplified Technical English): supported hosts
   (x86_64 Linux with a systemd user manager; arm64 macOS at or above
   the floor, with a GUI login if M7e confirms it), download and check,
   install, first `status`, enroll an existing systemd unit or launchd
   job, upgrade, adopt an existing install, uninstall, the programs table
   above, where logs and data live, how to report a bug.
2. Update `rust/README.md` (install section points to `INSTALL.md`;
   remove the out-of-date "M2 has started" status text).
3. Check `~/AI/docs/ARTIFACT-INDEX.md`: if the published **Driving
   llmctl** guide describes install, add a dated note or update it.

Check: every command in `INSTALL.md` appears in a passing clean-host run.

## When the work reaches the three hosts

Not in the pilot window (rules 2 and 4). After the M5 exit and M6
(decision 5):

1. Format commit on `main` (M7d step 3), then rebase the M7 branch.
2. Merge M7. Write the workstation `agent.toml` keys (M7a step 3) before
   the new binary starts; add `--host-id workstation` to its unit.
3. `agent install --adopt` on mbp2 and m5 (writes manifests only).
4. Rewire `host add` / `host upgrade` onto `agent install` (M7b), tested
   against one Mac first.

The workstation m4a unit stays as it is (install refuses to replace it
by design). Moving it to the standard layout is a separate step with its
own reversal.

## Exit checklist (maps to the parent criterion, distribution half)

| Criterion | Evidence required |
|---|---|
| Supported hosts defined | `INSTALL.md` table: x86_64 Linux + systemd user manager; arm64 macOS ≥ recorded floor |
| Clean Linux install/remove | M7e Linux passes on `debian:12` from the release archive; file and directory list after `--purge` equals before |
| Clean macOS install/remove | M7e macOS passes as the throwaway user on mbp2 |
| Clean-host "CI" | owner-approved local scripts, run from a fresh clone at a recorded commit (rule 3) |
| Upgrade / adopt | M7b tests: install over own install updates the manifest; `--adopt` writes a manifest for a pre-manifest install and changes nothing else |
| No workstation paths | M7a gate test green (paths plus `workstation`, `pitcany`); clean-host runs have no `~/AI` or `~/local-llm` |
| No Python requirement for the agent | Linux image has no Python; install, status, run/stop and uninstall pass |
| Engine Python stays engine-specific | no change to recipes; the agent's only Python calls are an engine's own interpreter and `uv` inside installers |
| Releases | M7c archives, `SHA256SUMS`, `BUILDINFO`, `version --json` with git sha |
| Dependency/license checks | `check.sh` license, sources and advisories gates pass (ignore list reasoned); fmt/clippy reported |
| Docs | `INSTALL.md` matches the passing clean-host runs |
| Pilot untouched | M5 `audit --all` unchanged by M7 work; no agent redeploy and **no service-manager call** on the three hosts before the M5 exit |

## Owner decisions (2026-11-21, all as recommended)

The owner took all eight as recommended. The review agreed with the first
seven, with the amendments noted; decision 8 was added after it.

**Owner action needed before M7e (macOS):** create the throwaway user on
mbp2 (`sudo`), and log it in once by fast user switching if M7e confirms
launchd needs a GUI session.

1. **Split into M7d, M7a, M7b, M7c, M7e, M7f** in that order, a check at
   the end of each, about **6.5–7.5 working days** (raised from 5.5–6.5
   after the review). *Recommended.*
2. **Linux binary:** static `musl` (recommended: one file, no TLS, DNS
   or NSS use; the only C code is bundled SQLite), or `glibc` built in an
   old-distro container (a glibc floor to state and test).
3. **Install layout for new installs:** user scope on both OSes,
   `~/.local/bin/llmctl-next` and `~/.local/state/llmctl-next`, user
   systemd unit / LaunchAgent (recommended: the agent drives user units,
   and the Macs already use this layout), or system-wide `/usr/local`
   with a system unit (needs root).
4. **Workstation-only defaults:** `agent.toml` keys, env overrides keep
   precedence, daemon and CLI share one reader, absent = off with a
   startup warning when the old file exists (recommended), or keep the
   `~/AI` defaults when the file exists (fails a strict reading of
   "without workstation paths").
5. **Branch policy:** keep M7 on a branch `m7a-distribution` until M6
   exits, so M6 deploys the build the pilot proved; rebase at each step;
   the format commit and the `host` rewire stay off the branch
   (recommended), or merge to `main` as each step passes (M6 then
   deploys code the pilot did not run, or pins an older commit).
6. **Clean macOS environment:** a throwaway local user on mbp2 that the
   owner creates with `sudo` and logs in once by fast user switching if
   M7e confirms launchd needs it (recommended: minutes to set up; it
   cannot prove "no Python", Linux does that), or a Tart macOS VM on
   mbp2 (truly clean, but a 25–50 GB image and more setup).
7. **Publishing:** build archives locally only; publishing a GitHub
   release (public repo) and generating `THIRD-PARTY-LICENSES.md` is a
   separate owner go-ahead after M6 (recommended), or publish a
   pre-release at the end of M7.
8. **Host-id default** (added after the review; not reviewed): the
   `"workstation"` default becomes the short hostname, and the M6 deploy
   adds `--host-id workstation` to the workstation unit (recommended:
   a clean host gets a true id), or keep `"workstation"` and allowlist
   it in the gate (no deploy step, but every new host is "workstation"
   until `--host-id` is set). Risk of the recommendation: if the deploy
   step is missed, the workstation agent answers to a new id and its
   records no longer match; M7a adds a startup refusal when the data
   dir's recorded host id differs from the resolved one.
