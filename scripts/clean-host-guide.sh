#!/usr/bin/env bash
# The part of the clean-host test (M7e, M7f) that runs on the clean host.
# clean-host-test.sh copies it there; do not start it by hand.
#
# Usage: clean-host-guide.sh linux|macos <dir with archive and SHA256SUMS>
#
# It takes INSTALL.md from the archive and runs each tagged command block
# as written, then checks upgrade, uninstall and the refusals.
#
# It makes real service-manager calls, so it first proves where it is (plan
# rule 4): user `tester` in a container, or the non-admin throwaway user
# `llmctltest` on macOS. No option or variable turns this guard off.
# Exit: 0 all pass, 1 a check failed, 2 guard or usage, 3 no GUI domain.
#
# Keep it bash 3.2 compatible (macOS /bin/bash).
set -uo pipefail

os=${1:-}
dl=${2:-}

refuse() { echo "REFUSED  $*" >&2; exit 2; }

# ---- guard: nothing above or inside it calls a service manager --------------
case "$os" in
  linux)
    [ "$(uname -s)" = Linux ] || refuse "not Linux"
    [ "$(id -un)" = tester ] || refuse "not the container user tester"
    [ -f /.dockerenv ] || refuse "not inside a container"
    [ "$HOME" = /home/tester ] || refuse "HOME is not /home/tester"
    ;;
  macos)
    [ "$(uname -s)" = Darwin ] || refuse "not macOS"
    [ "$(id -un)" = llmctltest ] || refuse "not the throwaway user llmctltest"
    [ "$HOME" = /Users/llmctltest ] || refuse "HOME is not /Users/llmctltest"
    [ "$(id -u)" -gt 501 ] || refuse "uid is the uid of a primary account"
    case " $(id -Gn) " in *" admin "*) refuse "the test user must not be an admin" ;; esac
    ;;
  *) refuse "usage: linux|macos <dir>" ;;
esac
[ -d "$dl" ] || refuse "no directory $dl"

uid=$(id -u)
USER=$(id -un); export USER
bin="$HOME/.local/bin/llmctl-next"
data="$HOME/.local/state/llmctl-next"
failed=0
work=$(mktemp -d /tmp/llmctl-guide.XXXXXX)
trap 'rm -rf "$work"' EXIT
log="$work/log"

if [ "$os" = linux ]; then
  target=x86_64-unknown-linux-musl
  svc="$HOME/.config/systemd/user/llmctl-next.service"
  sleeper="$HOME/.config/systemd/user/sleeper.service"
else
  target=aarch64-apple-darwin
  label=com.pitcany.llmctl-next
  svc="$HOME/Library/LaunchAgents/$label.plist"
  sleeper="$HOME/Library/LaunchAgents/com.example.sleeper.plist"
fi

pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; failed=1; }
# check NAME CMD...: PASS when CMD exits 0. A failure shows the output.
check() {
  local what=$1; shift
  if "$@" >"$log" 2>&1; then pass "$what"; else fail "$what"; sed 's/^/      | /' "$log" | tail -15; fi
}
# has TEXT CMD...: CMD exits 0 and prints TEXT.
has() {
  local text=$1 o; shift
  o=$("$@" 2>&1) || { echo "$o"; return 1; }
  echo "$o"
  grep -qF -- "$text" <<<"$o"
}
no() { if "$@"; then return 1; fi; }
digest() {
  if [ "$os" = linux ]; then sha256sum; else shasum -a 256; fi
}
cli() { LLMCTL_NEXT_DATA_DIR="$data" "$bin" "$@"; }

# launchctl output is only ever filtered or discarded: `print` shows the
# environment.
agent_gone() {
  if [ "$os" = linux ]; then
    no systemctl --user cat llmctl-next.service
  else
    no launchctl print "gui/$uid/$label" >/dev/null 2>&1
  fi
}
sleeper_runs() {
  if [ "$os" = linux ]; then
    systemctl --user is-active sleeper.service
  else
    launchctl print "gui/$uid/com.example.sleeper" 2>/dev/null | grep -q '^[[:space:]]*state = running'
  fi
}

snapshot() {
  if [ "$os" = linux ]; then
    find "$HOME" /etc -xdev 2>/dev/null | LC_ALL=C sort
  else
    find "$HOME" /private/etc -xdev 2>/dev/null | LC_ALL=C sort
  fi
}
# same_tree BEFORE: the tree equals BEFORE. macOS writes into a logged-in
# account all the time: controls with no llmctl running saw 10 changes in
# 25 s and 186 in 40 s (2026-11-22). Only the roots below are allowed, each
# seen changing in those controls or the first run. A fresh account also
# creates other Apple stores in its first hour: a run can fail on one
# (null control: 3 of 6 idle windows). Do not widen the list for that; read
# the paths the failure prints and run again. Anything else,
# and any path that names llmctl or the test service, fails. LaunchAgents,
# Preferences, Application Support, Logs and all outside ~/Library stay
# strict.
same_tree() {
  local changed path bad=0 allowed=0
  snapshot >"$work/after"
  changed=$(diff "$1" "$work/after" | sed -n 's/^[<>] //p')
  [ -n "$changed" ] || return 0
  while IFS= read -r path; do
    # Match names below the home: the home itself is /Users/llmctltest.
    case "${path#"$HOME"/}" in
      *llmctl*|*com.example.sleeper*) echo "left: $path"; bad=1; continue ;;
    esac
    if [ "$os" = macos ]; then
      case "${path#"$HOME"/Library/}" in
        Containers/*|"Daemon Containers"/*|"Application Scripts"/*|Trial/*|IdentityServices/delete*|\
        Caches/*|Metadata/*|Biome/*)
          allowed=$((allowed + 1)); continue ;;
      esac
    fi
    echo "changed: $path"; bad=1
  done <<<"$changed"
  echo "allowed: $allowed paths in macOS system stores"
  return "$bad"
}
no_product_files() {
  local p
  for p in "$bin" "$svc" "$data" "$HOME/Library/Logs/llmctl-next"; do
    if [ -e "$p" ]; then echo "left: $p"; return 1; fi
  done
}

# ---- the commands in INSTALL.md -------------------------------------------------
name=$(cd "$dl" && for f in llmctl-next-*-"$target".tar.gz; do echo "${f%.tar.gz}"; done)
[ -f "$dl/$name.tar.gz" ] || refuse "no $target archive in $dl"
doc="$work/INSTALL.md"
tar -xzOf "$dl/$name.tar.gz" "$name/INSTALL.md" >"$doc" 2>/dev/null \
  || { fail "the archive has INSTALL.md"; exit 1; }
pass "the archive has INSTALL.md"

block() {
  awk -v name="$1" -v os="$os" '
    $0 == "<!-- run: " name " " os " -->" || $0 == "<!-- run: " name " both -->" { want = 1; next }
    want && /^```bash$/ { inside = 1; next }
    inside && /^```$/ { exit }
    inside { print }
  ' "$doc"
}
every_block_tagged() {
  awk '/^```bash$/ { if (prev !~ /^<!-- run: [a-z-]+ (linux|macos|both) -->$/) { print "untagged block at line " NR; bad = 1 } }
       { prev = $0 }
       END { exit bad }' "$doc"
}
ran=" "
# guide NAME DIR: run the block as written, in DIR, after the shell block.
guide() {
  local name=$1 dir=$2 body setup=""
  ran="$ran$name "
  body=$(block "$name")
  if [ -z "$body" ]; then echo "INSTALL.md has no block $name for $os"; return 1; fi
  case "$name" in check|install|upgrade) ;; *) setup=$(block shell) ;; esac
  (cd "$dir" && env -u LLMCTL_NEXT_DATA_DIR bash -eo pipefail -c "$setup
$body")
}
every_block_ran() {
  local n bad=0
  for n in $(sed -nE "s/^<!-- run: ([a-z-]+) ($os|both) -->\$/\1/p" "$doc"); do
    [ "$n" = shell ] && continue
    case "$ran" in *" $n "*) ;; *) echo "not run: $n"; bad=1 ;; esac
  done
  return "$bad"
}
check "each command block in INSTALL.md has a test tag" every_block_tagged

# ---- clean start ----------------------------------------------------------------
if no_product_files; then
  pass "clean start: no agent files"
else
  fail "clean start: no agent files"
  exit 1
fi

guide check "$dl" >"$log" 2>&1 && pass "guide: check and unpack" \
  || { fail "guide: check and unpack"; cat "$log"; exit 1; }
pkg="$dl/$name"

if [ "$os" = macos ]; then
  check "no Rust toolchain in this account" no test -e "$HOME/.cargo"
  if ! launchctl print "gui/$uid" >/dev/null 2>&1; then
    echo "NOTE  gui/$uid does not exist: the user has no desktop session"
    check "install without a GUI domain fails" no "$pkg/llmctl-next" agent install
    [ -x "$bin" ] && check "the failed install can be uninstalled" "$bin" agent uninstall --purge
    check "no agent files left by the failed install" no_product_files
    echo "== STOPPED: log llmctltest in, switch back by fast user switching (no logout), then run again"
    exit 3
  fi
  pass "gui/$uid exists"
fi

cleanup() {
  [ -x "$bin" ] && "$bin" agent uninstall --purge >/dev/null 2>&1
  if [ "$os" = macos ]; then
    launchctl bootout "gui/$uid/$label" >/dev/null 2>&1
    launchctl bootout "gui/$uid/com.example.sleeper" >/dev/null 2>&1
    rm -f "$sleeper"
  fi
  rm -rf "$work"
}
trap cleanup EXIT

# An existing service to enroll. Written before the snapshot.
mkdir -p "$(dirname "$sleeper")"
if [ "$os" = linux ]; then
  if [ ! -f "$sleeper" ]; then
    printf '[Unit]\nDescription=clean-host test unit\n[Service]\nExecStart=/bin/sleep infinity\n' >"$sleeper"
  fi
  systemctl --user daemon-reload
  systemctl --user start sleeper.service
else
  cat >"$sleeper" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.example.sleeper</string>
<key>ProgramArguments</key><array><string>/bin/sleep</string><string>86400</string></array>
<key>RunAtLoad</key><true/>
</dict></plist>
EOF
  env -i /bin/launchctl bootstrap "gui/$uid" "$sleeper"
fi
# A fresh launchd job reads `xpcproxy` for a moment before `running`.
sleeper_starts() {
  local n=0
  until sleeper_runs; do
    n=$((n + 1))
    if [ "$n" -ge 50 ]; then return 1; fi
    sleep 0.2
  done
}
check "test service runs" sleeper_starts
sleeper_sum=$(digest <"$sleeper")
snapshot >"$work/before"

# ---- path A: install, enroll, upgrade, purge ------------------------------------
check "guide: install" guide install "$pkg"
check "guide: first status" guide first-status "$HOME"
check "agent answers" has "socket: answers" cli agent status
check "status: 0 workloads" has "0 workloads" cli status
check "status: integrations off" has "router config: off" cli status
check "without the shell block, status finds no agent (as INSTALL.md says)" \
  no env -u LLMCTL_NEXT_DATA_DIR "$bin" status
if [ "$os" = linux ]; then
  check "guide: keep the agent on after logout" guide linger "$HOME"
fi
check "guide: logs" guide logs "$HOME"

preview=$(guide preview "$HOME" 2>&1); preview_rc=$?
ran="${ran}preview "
echo "$preview" | sed 's/^/      > /'
check "guide: enroll preview exits 0" test "$preview_rc" = 0
check "preview read the service definition" grep -q '/bin/sleep' <<<"$preview"
check "preview changed nothing" has "0 workloads" cli status
confirm=$(sed -n '/^to transfer ownership:$/{n;p;}' <<<"$preview" | sed 's/^ *//')
confirm_printed() {
  case "$confirm" in "llmctl-next adopt "*) ;; *) echo "no confirm command printed"; return 1 ;; esac
  (cd "$HOME" && bash -eo pipefail -c "$(block shell)
$confirm")
}
check "the printed confirm command enrolls it" confirm_printed
check "guide: status and ops after enroll" guide after-enroll "$HOME"
check "status: 1 workload" has "1 workload" cli status
check "enrolled service file unchanged" test "$sleeper_sum" = "$(digest <"$sleeper")"

svc_sum=$(digest <"$svc")
for i in 1 2 3; do
  check "guide: upgrade in place, round $i" guide upgrade "$pkg"
  check "agent answers after upgrade $i" has "socket: answers" cli agent status
done
check "upgrade keeps the service file (host id)" test "$svc_sum" = "$(digest <"$svc")"
check "upgrade keeps the inventory" has "1 workload" cli status

check "guide: uninstall --purge" guide purge "$HOME"
check "agent gone from the service manager" agent_gone
check "enrolled service still runs" sleeper_runs
check "files after --purge equal the list before" same_tree "$work/before"
grep '^allowed:' "$log" | sed 's/^/NOTE  /'

# ---- path B: uninstall with the data kept -----------------------------------------
check "install again" guide install "$pkg"
check "guide: uninstall, data kept" guide uninstall "$HOME"
check "binary and service file removed" no test -e "$bin" -o -e "$svc"
check "inventory kept" test -f "$data/inventory.json"
check "install over kept data" guide install "$pkg"
check "agent answers on kept data" has "socket: answers" cli agent status

# ---- path C: record an install that has no manifest -----------------------------
rm -f "$data/install-manifest.json"
svc_sum=$(digest <"$svc")
check "guide: record an older install" guide adopt-install "$HOME"
check "recorded as adopted" has "installed (adopted)" cli agent status
check "record changed no service file" test "$svc_sum" = "$(digest <"$svc")"
check "purge the adopted install" guide purge "$HOME"
check "no agent files left" no_product_files
check "agent gone again" agent_gone

# ---- refusals -----------------------------------------------------------------
check "uninstall with nothing installed refuses" no "$pkg/llmctl-next" agent uninstall
mkdir -p "$(dirname "$svc")"
echo "hand-written" >"$svc"
check "install refuses a hand-written service file" no "$pkg/llmctl-next" agent install
check "hand-written file unchanged" test "$(cat "$svc")" = hand-written
check "refused install wrote no binary" no test -e "$bin"
rm -f "$svc"

# check runs its command in this shell, so `ran` holds every guide block.
check "each command block for $os was run" every_block_ran

if [ "$failed" = 0 ]; then echo "== GUIDE ALL PASS"; else echo "== GUIDE FAILURES"; fi
exit "$failed"
