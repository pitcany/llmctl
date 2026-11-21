# Rust successor — M7 progress (distribution half)

Plan: [RUST-M7-PLAN.md](RUST-M7-PLAN.md) (agreed 2026-11-21, ADR-060 in
the vault). Branch: `m7a-distribution` (stays off `main` until M6 exits,
decision 5). Rules: repo-only, no agent redeploy and no real
service-manager call on the workstation, mbp2 or m5 before the M5 exit.

| Date | Step | Result | Evidence |
|---|---|---|---|
| 2026-11-21 | **M7d gate, report-only** | `rust/deny.toml` (cargo-deny 0.22.0): licenses allowlist MIT, Apache-2.0, BSD-2/3-Clause, Unicode-3.0, Zlib (ISC dropped: no crate uses it); graph covers `x86_64-unknown-linux-gnu`, `x86_64-unknown-linux-musl`, `aarch64-apple-darwin`; bans warn on duplicates (`syn`, `unicode-width`, `hashbrown`); sources crates.io only. All workspace crates now `publish = false` (path dependencies are then allowed wildcards). `rust-version` raised **1.85 → 1.89** (`uuid 1.29.2` declares 1.89; `cargo +1.89 check --locked --workspace --all-targets` passes). `scripts/check.sh [--no-tests] [--offline]`: one PASS/FAIL/REPORT line per gate, exit 1 on a FAIL | `check.sh`: licenses, sources, bans, advisories, msrv **PASS**; fmt **REPORT** (1318 diff hunks); clippy **REPORT** (`llmctl-core` stops at 6 errors: 2 needless borrows, 2 large enum variants, too many arguments, empty line after doc comment); tests 325 passed, 1 failed, 32 ignored. Planted GPL-3.0 path dependency in a copy: `licenses FAILED`, exit 4 |
| 2026-11-21 | **`rusqlite` 0.32.1 → 0.41.1** (owner go-ahead; fixes RUSTSEC-2026-0459) | `libsqlite3-sys` 0.30.1 → 0.39.1 (bundled SQLite 3.54.x), `hashlink` 0.9 → 0.12; new lock entries `rsqlite-vfs`, `sqlite-wasm-rs` (wasm targets only). No source change: `llmctl-daemon/src/db.rs` builds unchanged with no warnings. The `deny.toml` ignore is removed. 0.43 exists; 0.41 is the release the advisory names, so the step stops there. SQLite's file format is backward compatible, so the live data dirs need no migration; not tested against a copy of a live data dir (a temp daemon on it could make service-manager calls, rule 4) | `cargo test --workspace --locked --no-fail-fast`: 326 passed, 0 failed, 32 ignored; `check.sh --no-tests`: licenses, sources, bans, **advisories**, msrv (1.89) PASS |
| 2026-11-21 | **M7a: host integrations from `agent.toml`** | New `llmctl_client::agent_config` (`toml` 1.2, parse+serde only): optional keys `router_config`, `lane_lease` in `<data-dir>/agent.toml`, `~/` expanded, unknown keys and bad values refused (an invalid file stops the agent's start). Precedence: env override (`LLMCTL_ROUTER_CONFIG`, `LLMCTL_NEXT_LANE_LEASE`) → key → off. **One resolution for both sides:** the daemon installs the config at `App::open` (`coord::lease_path`, `presets::router_config_path` read it); the CLI's `holder_ancestor(data_dir)` loads the same file from `--data-dir` (or the parent of `--socket`). `lease::default_path` (the `~/AI` lease path) is gone; `router_config_path()` is now `Option` (no router = no router profiles, no settle entry). Start-up warning when a key is unset but the pre-M7 file exists (`… exists and is ignored`). `status` prints `integrations: router config: … · lane lease: …` (new optional `StatusResponse.integrations`, omitted when empty, so older clients are unaffected). Gate test `llmctl-core/tests/no_owner_paths.rs`: scans `crates/*/src` outside comments and `#[cfg(test)]` items for `/home/`, `/Users/`, `AI/`, `local-llm`, `workstation`, `pitcany`; six allowlisted hits, each with a reason (3× the launchd vendor label, 2× the pre-M7 warning paths, 1× the daemon's `workstation` default); stale allowlist entries also fail. **Host id (decision 8) not changed, owner asked:** the id is bound once in the journal (`meta host_id`) and a bound id beats the default, so live agents never change id; changing the daemon default would touch 27 daemon-spawn sites in the tests. Proposed instead: `agent install` (M7b) writes `--host-id <short hostname>` into the unit/plist it generates | unit: `agent_config` 2, `holder_ancestor_reads_the_lane_lease_key_from_agent_toml`, `the_lane_lease_comes_from_agent_toml_for_real_engines_on_linux`; e2e `host_integrations_are_off_without_agent_toml_and_named_with_it` (empty `$HOME` → both off, 0 presets, no warning; pre-M7 files → warning; keys → status names them), `an_invalid_agent_toml_stops_the_start`; gate: 2 tests pass, a planted `/home/…` const fails it. `check.sh`: licenses, sources, bans, advisories, msrv PASS; fmt REPORT 1330 hunks (1318 before; +12 from edits inside unformatted files, new files formatted); clippy REPORT (same 6 in core); tests 333 passed, 1 failed, 32 ignored — the 1 is `bridge_subcommand_relays_http_to_the_socket`, the known ~1-in-10 flake on `main` too, 3/3 alone |

## Advisories found by the gate

The review expected `serde_yaml` and `paste` to be flagged; neither was.
These two were:

| ID | Crate | Why ignored | Way out |
|---|---|---|---|
| ~~RUSTSEC-2026-0459 (CVE-2026-119375)~~ **fixed by the `rusqlite` 0.41 row above; ignore removed** | `libsqlite3-sys 0.30.1` (bundled SQLite < 3.54, via `rusqlite 0.32.1`) | Stale page-cache memory is returned only for a crafted db + `-wal` pair. The agent opens only databases in its own data dir, which only its user can write. | `rusqlite 0.41` (`libsqlite3-sys 0.39`, SQLite 3.54.x; declares Rust 1.89). An API upgrade across nine minor versions: **owner follow-up, not in M7d** |
| RUSTSEC-2026-0543 | `chrono 0.4.45` | Unmaintained, no vulnerability, no safe upgrade. | Move to `jiff` or `time`: a workspace-wide change, not part of M7 |

## Known flake seen in the M7d run

`e2e::recovery_budget_exhausts_blocks_and_survives_the_daemon_restart`
failed once in the full parallel run (`crates/llmctl-cli/tests/e2e.rs:2420`:
`recover-coder-2` still `Executing` when the test asserted two failed
recoveries) and passed 3/3 alone. No code changed in M7d (manifest
metadata and `deny.toml` only), so this is the review's open "reliable
parallel test runs" item, not an M7 regression. The test was not edited.
