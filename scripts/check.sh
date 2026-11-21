#!/usr/bin/env bash
# Quality gate for the llmctl-next Rust workspace (M7d, docs/RUST-M7-PLAN.md).
#
# Prints one line per gate: PASS, FAIL or REPORT. REPORT gates never fail
# the run (fmt and clippy stay report-only until the format commit after
# M6). Exit code: 0 when no gate failed, 1 otherwise.
#
# Usage: scripts/check.sh [--no-tests] [--offline]
#   --no-tests  skip the workspace test suite
#   --offline   skip the advisories gate (it fetches the RustSec database)
#
# Needs: cargo, cargo-deny (cargo install cargo-deny --locked), and the
# toolchain named by rust-version in rust/Cargo.toml (rustup toolchain
# install <ver> --profile minimal). A missing tool is a FAIL, not a skip.
set -u

repo=$(cd "$(dirname "$0")/.." && pwd)
ws="$repo/rust"
logdir=$(mktemp -d "${TMPDIR:-/tmp}/llmctl-check.XXXXXX")
run_tests=1
offline=0
for arg in "$@"; do
  case "$arg" in
    --no-tests) run_tests=0 ;;
    --offline) offline=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

failed=0
# gate NAME MODE CMD...  (MODE: fail | report)
gate() {
  local name=$1 mode=$2; shift 2
  local log="$logdir/$name.log"
  if (cd "$ws" && "$@") >"$log" 2>&1; then
    printf 'PASS    %-12s\n' "$name"
  elif [ "$mode" = report ]; then
    printf 'REPORT  %-12s %s\n' "$name" "$(summary "$name" "$log")"
  else
    printf 'FAIL    %-12s %s\n' "$name" "$(summary "$name" "$log")"
    failed=1
  fi
}

# One short line explaining a non-pass, from the gate's own log.
summary() {
  local name=$1 log=$2 msg
  case "$name" in
    fmt) msg="$(grep -c '^Diff in ' "$log") diff hunks" ;;
    # clippy stops at the first crate that fails; quote cargo's own count.
    clippy) msg=$(grep -E '^error: could not compile' "$log" | head -1)
            [ -n "$msg" ] || msg="$(grep -cE '^warning(\[|: [^`])' "$log") warnings" ;;
    tests) msg=$(grep -E '^test result: FAILED|^error: test failed|^error:' "$log" | head -1) ;;
    *) msg=$(grep -E '^error(\[|:)' "$log" | head -1) ;;
  esac
  printf '%s (log: %s)' "${msg:-see log}" "$log"
}

msrv=$(sed -n 's/^rust-version = "\(.*\)"$/\1/p' "$ws/Cargo.toml")

command -v cargo-deny >/dev/null || { echo "FAIL    cargo-deny   not installed (cargo install cargo-deny --locked)"; failed=1; }
if command -v cargo-deny >/dev/null; then
  gate licenses fail cargo deny --locked check licenses
  gate sources fail cargo deny --locked check sources
  gate bans fail cargo deny --locked check bans
  if [ "$offline" = 1 ]; then
    printf 'REPORT  %-12s skipped (--offline)\n' advisories
  else
    gate advisories fail cargo deny --locked check advisories
  fi
fi

# The declared rust-version must really build the workspace.
gate msrv fail env CARGO_TARGET_DIR="$ws/target/msrv" \
  cargo "+$msrv" check --locked --workspace --all-targets

gate fmt report cargo fmt --all -- --check
gate clippy report cargo clippy --workspace --all-targets --locked -- -D warnings

if [ "$run_tests" = 1 ]; then
  gate tests fail cargo test --workspace --locked --no-fail-fast
fi

echo "logs: $logdir"
exit "$failed"
