# M7e clean-host test — Linux, 2026-11-21

Plan: [RUST-M7-PLAN.md](RUST-M7-PLAN.md) (M7e). Script:
`scripts/clean-host-test.sh linux <archive>`. Archive:
`llmctl-next-0.1.0-x86_64-unknown-linux-musl.tar.gz` from release
`0.1.0-f0d04dfc7b69` (commit `f0d04df`, M7c). Run at 2026-11-22T00:46:51Z
(16:46 PST 2026-11-21) on the workstation's Docker 29.10.

**Result: 27 of 27 checks PASS.** macOS half not run yet (needs the
throwaway mbp2 user).

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

## Findings

1. **A fresh agent bootstraps the dev sample inventory**: three fake
   workloads (`coder`, `helper`, `big`) and two GPUs, on a host with no
   GPU (`status`: `3 workloads`). Useful for this test; wrong for a real
   clean install. Owner decision open: `agent install` seeds an empty
   inventory (no workloads, GPUs from detection), or the sample stays
   until a workload is enrolled.
2. Cosmetic: uninstall prints the purged data dir after the created
   directories (report order), so `~/.local/state/llmctl-next/` appears
   after `~/.local/`. The removal itself is correct (check 21).

Raw transcript kept outside the repo (`llmctl-clean-host.Kgf1Rf.log`).
