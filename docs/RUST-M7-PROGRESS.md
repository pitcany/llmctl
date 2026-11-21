# Rust successor — M7 progress (distribution half)

Plan: [RUST-M7-PLAN.md](RUST-M7-PLAN.md) (agreed 2026-11-21, ADR-060 in
the vault). Branch: `m7a-distribution` (stays off `main` until M6 exits,
decision 5). Rules: repo-only, no agent redeploy and no real
service-manager call on the workstation, mbp2 or m5 before the M5 exit.

| Date | Step | Result | Evidence |
|---|---|---|---|
| 2026-11-21 | **M7d gate, report-only** | `rust/deny.toml` (cargo-deny 0.22.0): licenses allowlist MIT, Apache-2.0, BSD-2/3-Clause, Unicode-3.0, Zlib (ISC dropped: no crate uses it); graph covers `x86_64-unknown-linux-gnu`, `x86_64-unknown-linux-musl`, `aarch64-apple-darwin`; bans warn on duplicates (`syn`, `unicode-width`, `hashbrown`); sources crates.io only. All workspace crates now `publish = false` (path dependencies are then allowed wildcards). `rust-version` raised **1.85 → 1.89** (`uuid 1.29.2` declares 1.89; `cargo +1.89 check --locked --workspace --all-targets` passes). `scripts/check.sh [--no-tests] [--offline]`: one PASS/FAIL/REPORT line per gate, exit 1 on a FAIL | `check.sh`: licenses, sources, bans, advisories, msrv **PASS**; fmt **REPORT** (1318 diff hunks); clippy **REPORT** (`llmctl-core` stops at 6 errors: 2 needless borrows, 2 large enum variants, too many arguments, empty line after doc comment); tests 325 passed, 1 failed, 32 ignored. Planted GPL-3.0 path dependency in a copy: `licenses FAILED`, exit 4 |

## Advisories ignored (each with a reason in `deny.toml`)

The review expected `serde_yaml` and `paste` to be flagged; neither was.
These two were:

| ID | Crate | Why ignored | Way out |
|---|---|---|---|
| RUSTSEC-2026-0459 (CVE-2026-119375) | `libsqlite3-sys 0.30.1` (bundled SQLite < 3.54, via `rusqlite 0.32.1`) | Stale page-cache memory is returned only for a crafted db + `-wal` pair. The agent opens only databases in its own data dir, which only its user can write. | `rusqlite 0.41` (`libsqlite3-sys 0.39`, SQLite 3.54.x; declares Rust 1.89). An API upgrade across nine minor versions: **owner follow-up, not in M7d** |
| RUSTSEC-2026-0543 | `chrono 0.4.45` | Unmaintained, no vulnerability, no safe upgrade. | Move to `jiff` or `time`: a workspace-wide change, not part of M7 |

## Known flake seen in the M7d run

`e2e::recovery_budget_exhausts_blocks_and_survives_the_daemon_restart`
failed once in the full parallel run (`crates/llmctl-cli/tests/e2e.rs:2420`:
`recover-coder-2` still `Executing` when the test asserted two failed
recoveries) and passed 3/3 alone. No code changed in M7d (manifest
metadata and `deny.toml` only), so this is the review's open "reliable
parallel test runs" item, not an M7 regression. The test was not edited.
