# M7e clean-host test — Linux, 2026-11-21

Plan: [RUST-M7-PLAN.md](RUST-M7-PLAN.md) (M7e). Script:
`scripts/clean-host-test.sh linux <archive>`. Archive:
`llmctl-next-0.1.0-x86_64-unknown-linux-musl.tar.gz` from release
`0.1.0-f0d04dfc7b69` (commit `f0d04df`, M7c). Run at 2026-11-22T00:46:51Z
(16:46 PST 2026-11-21) on the workstation's Docker 29.10.

**Result: run 1 27/27 PASS; run 2 (after the empty-inventory change) 29/29
PASS.** macOS half not run yet (needs the throwaway mbp2 user).

## Environment

- Base `debian:12` (`debian@sha256:268fb24c39bca92d326dd83ea3abebe4a3f7751c804ffef23bb6d857dd343ff8`)
  plus `systemd`, `systemd-sysv`, `dbus`, `dbus-user-session`,
  `libpam-systemd`, `procps`; user `tester` (uid 1000) with lingering on.
- Started **without `--privileged`**: `--cgroupns=private`,
  `--security-opt writable-cgroups=true` (Docker 25+), `--tmpfs /run`,
  `--tmpfs /run/lock`, no host D-Bus socket. Without `writable-cgroups`,
  systemd 252 exits at start (`Failed to create /init.scope control group:
  Read-only file system`).
- No `python3`, `python`, `curl`, `cargo` or `rustc` in the image
  (checked). `busctl`, `df`, `du`, `tar` present (checked).
- Every real `systemctl --user` call of `agent install`/`uninstall` ran
  inside this container (plan rule 4). None ran on the workstation.

## Checks

| # | Check | Result |
|---|---|---|
| 1 | user manager runs for the lingering user | PASS |
| 2 | no python3 / python / curl / cargo / rustc | PASS |
| 3–6 | busctl, df, du, tar present | PASS |
| 7 | `SHA256SUMS` verifies the archive | PASS |
| 8 | `agent install --host-id clean-box` (from the unpacked archive in /tmp) | PASS |
| 9 | unit at `~/.config/systemd/user/llmctl-next.service` | PASS |
| 10 | unit enabled | PASS |
| 11 | `agent status`: `service: active`, `socket: answers (agent 0.1.0)` | PASS |
| 12 | `status`: `host clean-box`, `integrations: router config: off … · lane lease: off …` | PASS |
| 13–14 | `run coder --wait`, `stop coder --wait` (fake engine) | PASS |
| 15 | `ops --json` lists both | PASS |
| 16 | second `agent install` → `agent upgraded` | PASS |
| 17 | unit still passes `--host-id clean-box` (upgrade keeps the id) | PASS |
| 18 | agent answers after the upgrade | PASS |
| 19 | `agent uninstall --purge` | PASS |
| 20 | unit gone from the user manager | PASS |
| 21 | **file and directory list of `/home/tester` and `/etc` after `--purge` equals the list before install** | PASS |
| 22–23 | uninstall with nothing installed: `no install manifest … nothing removed, no service stopped`, non-zero exit | PASS |
| 24 | install refuses a hand-written unit at the same path | PASS |
| 25 | that unit unchanged (sha256) | PASS |
| 26 | no binary installed by the refused install | PASS |
| 27 | bare `alpine:3.22`: `daemon` + `status` with a temp data dir, no service manager | PASS |

Uninstall output (check 19): removed the binary and the unit, then the
directories the install created (`~/.config/systemd/user/`,
`~/.config/systemd/`, `~/.config/`, `~/.local/bin/`, `~/.local/state/`,
`~/.local/`) and, with `--purge`, the data dir; and printed
`now unmanaged (still in their last state; no engine was stopped): coder
(Stopped), helper (Stopped), big (Stopped)`.

## Run 2 — empty inventory (commit `eb1da4f`, release `0.1.0-eb1da4fb43fb`)

Run at 2026-11-22T01:30:06Z after the owner chose an empty inventory for
fresh installs (finding 1 below). Same image and flags. **29/29 PASS.**
Changes to the checks:

- Before the snapshot, a user unit `sleeper.service` (`/bin/sleep
  infinity`) is written and started: an existing unit to enroll, the way
  a real host gets workloads.
- `agent install` prints `inventory: empty, GPU slots [0] (enroll an
  existing unit with llmctl-next adopt)` (no NVIDIA card in the
  container, so one slot for the host's memory).
- `status`: `host clean-box (Linux) · inventory revision 1 · 0 workloads`.
- `adopt sleeper.service --name sleeper --model none --port 9 --gpus 0`
  (preview, no `--confirm`): exit 0, imports `/bin/sleep` through the
  unit's D-Bus definition (`busctl`), prints the confirm command with a
  digest and `--allow-unverified` (the unit serves no endpoint); still 0
  workloads after.
- The fake-workload run/stop checks are gone (no sample workloads exist).

## Run 3 — INSTALL.md as written (2026-11-22, commit `27c182a`)

Run at 2026-11-22T16:40:31Z on release `0.1.0-27c182a4fbd1`, with the
scripts from a fresh clone at that commit. **78/78 PASS**: the 29 checks
of run 2 and 49 checks of the install guide (`clean-host-guide.sh`).

- The guide runner takes `INSTALL.md` from the archive and runs each
  tagged command block as written. An untagged block or a block that did
  not run is a failure.
- New coverage: the printed adopt confirm command, an enrolled service
  that keeps running through `--purge`, three upgrades in a row, uninstall
  with the data kept, install over kept data, `agent install --adopt`.
- The image now has `polkitd`, so the user can run
  `loginctl enable-linger` as the guide says.
- Negative control (a trial archive with a planted wrong command, a
  leftover file, an untagged block and a block that never runs): each
  one failed the test.

**macOS: not run.** The account `llmctltest` does not exist on mbp2 yet
(`scripts/mac-test-user.sh`, run by the owner). The macOS mode refused
the owner's account (exit 2). Its document parsing, tree comparison and
bash 3.2 syntax were rehearsed on mbp2 with no service-manager call. The
GUI-domain question is open until the run.

## macOS run 1 — no desktop session (2026-11-22T16:53:10Z)

Release `0.1.0-27c182a4fbd1`, user `llmctltest` (uid 503, not an admin),
macOS 27.2.1. The user's console login had ended at 08:51 PST (`last`),
so the run stopped with exit 3 as designed.

- **The plan's open point is confirmed:** with ssh alone, and after a
  logout, `gui/503` does not exist. `agent install` fails there.
- The failed install uninstalls with no agent files left (checked).
- The test left its own `/tmp/llmctl-guide.*` directory on an early exit.
  Fixed.
- The full macOS run needs the user logged in, left by fast user
  switching.

## macOS run 2 — final series (2026-11-22, commit `6a6951b`)

Release `0.1.0-6a6951b05a45`, scripts from a fresh clone at that commit,
user `llmctltest` with a desktop session. A series of five runs was fixed
before it started. Every run is listed.

| Run (UTC) | Result | Detail |
|---|---|---|
| 1 (17:36:44) | 49/50 | tree check: `~/Library/MediaAnalysis` (Apple photo analysis) |
| 2 (17:36:59) | 49/50 | tree check: Adobe GrowthSDK database side files |
| 3 (17:37:13) | **50/50** | 2 changed paths in allowed system stores |
| 4 (17:37:27) | **50/50** | 2 changed paths in allowed system stores |
| 5 (17:37:41) | **50/50** | **tree after `--purge` identical, no allowed path needed** |

- Linux on the same release and scripts: **78/78**.
- Upgrade stress on the same archive: **150 upgrades, 0 failed**.
- After the series the test account had no job, process, agent file or
  temp directory left.
- The binary links only `CoreFoundation`, `libiconv` and `libSystem`
  (`otool -L`), `minos 13.0`. The test user's PATH is
  `/usr/bin:/bin:/usr/sbin:/sbin`. Homebrew is installed system-wide on
  mbp2, so its absence is shown by linkage and PATH only.
- macOS 13 to 26 were not tested.
- Pilot untouched: `audit --all --since 2026-11-22T15:56:00Z` shows 0
  foreign writes on the three hosts; the mbp2 agent kept pid 60385 and
  binary `273f2847742447b2`.

**The tree check is noisy on macOS.** With no llmctl running, 3 of 6 idle
20 s windows failed it (Apple stores created in a new account's first
hour). The allowed roots are `Containers`, `Daemon Containers`,
`Application Scripts`, `Trial`, `IdentityServices/delete`, `Caches`,
`Metadata` and `Biome` under `~/Library`. A path that names llmctl is
never allowed. Do not widen the list to get a pass: read the printed
paths and run again.

**Earlier runs the same day**, on `ad18687` (the fix below): 2 of 8 passed
50/50. Four failed the tree check on system files. Two failed a race in
the test's fixture, fixed in `b292692`.

## Findings

1. ~~**A fresh agent bootstraps the dev sample inventory**~~ **resolved in
   run 2** (owner: empty inventory; `eb1da4f`): three fake
   workloads (`coder`, `helper`, `big`) and two GPUs, on a host with no
   GPU (`status`: `3 workloads`). Useful for this test; wrong for a real
   clean install. Owner decision open: `agent install` seeds an empty
   inventory (no workloads, GPUs from detection), or the sample stays
   until a workload is enrolled.
2. Cosmetic: uninstall prints the purged data dir after the created
   directories (report order), so `~/.local/state/llmctl-next/` appears
   after `~/.local/`. The removal itself is correct (check 21).
3. **`llmctl-next status` after an install finds no agent** (run 3): the
   CLI's default data dir is `~/.local/state/llmctl-next-dev`, and
   `agent install` uses `~/.local/state/llmctl-next`. `INSTALL.md` sets
   `LLMCTL_NEXT_DATA_DIR`. The default is not changed: owner decision
   open.
4. Minor: uninstall with the data kept, install again, then `--purge`
   leaves the empty directories `~/.local` and `~/.local/state` (the
   second install did not create them, so it does not record them).
   Reproduced in the container. Stated in `INSTALL.md`.
5. **Bug, fixed in `ad18687`: a macOS upgrade could leave the agent
   down.** `launchctl bootout` returns before launchd releases the job.
   The next bootstrap then fails with `5: Input/output error`. Seen in 1
   of 30 real upgrades, and in 2 of 3 under heavy load. The retry works
   at once. `agent install` now waits for the job to go and retries only
   that error.
6. **The same race is on `main` in `host add` / `host upgrade`**
   (`enroll.rs`: bootout, then bootstrap, no wait, no retry). M6 deploys
   the Macs with `host upgrade`. Not changed here: owner decision open.

Raw transcript kept outside the repo (`llmctl-clean-host.Kgf1Rf.log`).
