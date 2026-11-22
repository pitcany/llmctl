#!/usr/bin/env bash
# Clean-host test of a llmctl-next release archive (M7e,
# docs/RUST-M7-PLAN.md). Owner-approved local stand-in for hosted CI.
#
# Usage: scripts/clean-host-test.sh linux <archive.tar.gz> [--keep]
#   linux   Debian 12 container with systemd as PID 1 and a lingering user
#           (`tester`), started without --privileged: private cgroup
#           namespace, writable cgroups (Docker >= 25), no host D-Bus.
#           The image has no Python, no Rust and no curl.
#   --keep  leave the container running for inspection.
#   (macos: to be added; needs the throwaway mbp2 user.)
#
# Every real service-manager call runs inside the container (plan rule 4).
# One PASS/FAIL line per check; exit 1 when any check failed. The full
# transcript goes to the log file named at the end.
set -uo pipefail

mode=${1:-}
archive=${2:-}
keep=0
[ "${3:-}" = "--keep" ] && keep=1
[ "$mode" = linux ] || { sed -n '2,15p' "$0"; exit 2; }
[ -f "$archive" ] || { echo "no archive: $archive" >&2; exit 2; }
archive=$(cd "$(dirname "$archive")" && pwd)/$(basename "$archive")
name=$(basename "$archive" .tar.gz)
image=llmctl-clean-debian12
ctr=llmctl-clean-$$
log=$(mktemp "${TMPDIR:-/tmp}/llmctl-clean-host.XXXXXX.log")
failed=0

say() { printf '%s\n' "$*" | tee -a "$log"; }
# check NAME CMD...: PASS when CMD exits 0.
check() {
  local what=$1; shift
  if "$@" >>"$log" 2>&1; then say "PASS  $what"; else say "FAIL  $what"; failed=1; fi
}
# as tester, with the user manager's environment.
ux() {
  docker exec -u tester -w /home/tester \
    -e XDG_RUNTIME_DIR=/run/user/1000 \
    -e DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    "$ctr" "$@"
}
# out CMD...: run as tester, echo the output into the log, print it.
out() { ux "$@" 2>&1 | tee -a "$log"; return "${PIPESTATUS[0]}"; }
snapshot() {
  docker exec "$ctr" sh -c 'find /home/tester /etc -xdev 2>/dev/null | sort'
}
cleanup() {
  if [ "$keep" = 1 ]; then say "kept container: $ctr"; else docker rm -f "$ctr" >/dev/null 2>&1; fi
}
trap cleanup EXIT

say "== clean-host test: $name ($(date -u +%Y-%m-%dT%H:%M:%SZ))"

# ---- image and container ----------------------------------------------------
dockerfile=$(cat <<'EOF'
FROM debian:12
RUN apt-get update \
 && apt-get install -y --no-install-recommends systemd systemd-sysv dbus dbus-user-session libpam-systemd procps \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 -s /bin/bash tester \
 && mkdir -p /var/lib/systemd/linger && touch /var/lib/systemd/linger/tester
STOPSIGNAL SIGRTMIN+3
CMD ["/lib/systemd/systemd"]
EOF
)
docker pull -q debian:12 >/dev/null
say "base: $(docker image inspect debian:12 --format '{{index .RepoDigests 0}}')"
printf '%s\n' "$dockerfile" | docker build -q -t "$image" - >>"$log" 2>&1 \
  || { say "FAIL  image build"; exit 1; }
docker run -d --name "$ctr" --cgroupns=private --security-opt writable-cgroups=true \
  --tmpfs /run --tmpfs /run/lock "$image" >>"$log" 2>&1 || { say "FAIL  container start"; exit 1; }
state=""
for _ in $(seq 1 60); do
  state=$(docker exec "$ctr" systemctl is-system-running 2>/dev/null)
  case "$state" in running|degraded) break ;; esac
  sleep 1
done
say "system: $state"
for _ in $(seq 1 30); do
  ux systemctl --user is-system-running >/dev/null 2>&1 && break
  sleep 1
done
check "user manager runs for the lingering user" ux systemctl --user is-system-running

# ---- the clean image ----------------------------------------------------------
check "no python3 / python / curl / cargo / rustc in the image" \
  docker exec "$ctr" sh -c '! command -v python3 && ! command -v python && ! command -v curl && ! command -v cargo && ! command -v rustc'
for tool in busctl df du tar; do
  check "agent dependency present: $tool" docker exec "$ctr" sh -c "command -v $tool"
done

before=$(snapshot)

# ---- download and check -------------------------------------------------------
docker exec "$ctr" mkdir -p /tmp/dl
docker cp "$archive" "$ctr:/tmp/dl/" >>"$log" 2>&1
sums="$(dirname "$archive")/SHA256SUMS"
if [ -f "$sums" ]; then
  docker cp "$sums" "$ctr:/tmp/dl/SHA256SUMS" >>"$log" 2>&1
  check "SHA256SUMS verifies the archive" \
    docker exec -w /tmp/dl "$ctr" sh -c "grep ' $name.tar.gz\$' SHA256SUMS | sha256sum -c -"
else
  say "FAIL  no SHA256SUMS next to the archive"; failed=1
fi
docker exec "$ctr" sh -c "chown -R tester /tmp/dl && cd /tmp/dl && su tester -c 'tar -xzf $name.tar.gz'"
bin=/tmp/dl/$name/llmctl-next
inst=/home/tester/.local/bin/llmctl-next
data=/home/tester/.local/state/llmctl-next

# ---- install --------------------------------------------------------------------
check "agent install" bash -c "$(declare -f ux); ctr=$ctr; ux $bin agent install --host-id clean-box | grep -q 'agent installed'"
check "unit written where systemd looks" ux test -f /home/tester/.config/systemd/user/llmctl-next.service
check "unit is enabled" ux systemctl --user is-enabled llmctl-next.service
check "agent status: active and answering" bash -c "$(declare -f ux); ctr=$ctr; s=\$(ux $inst agent status); echo \"\$s\"; echo \"\$s\" | grep -q 'service: active' && echo \"\$s\" | grep -q 'socket: answers'"
check "status: host clean-box, integrations off" bash -c "$(declare -f ux); ctr=$ctr; s=\$(ux $inst --data-dir $data status); echo \"\$s\"; echo \"\$s\" | grep -q 'host clean-box' && echo \"\$s\" | grep -q 'integrations: router config: off'"

# ---- operate a workload (the fake engine of the sample inventory) ------------
check "run coder --wait" ux "$inst" --data-dir "$data" run coder --wait
check "stop coder --wait" ux "$inst" --data-dir "$data" stop coder --wait
check "ops shows the run and the stop" bash -c "$(declare -f ux); ctr=$ctr; o=\$(ux $inst --data-dir $data ops --json); echo \"\$o\"; [ \$(echo \"\$o\" | grep -c '\"workload\": \"coder\"') -ge 2 ]"

# ---- idempotent install ------------------------------------------------------
check "second install upgrades in place" bash -c "$(declare -f ux); ctr=$ctr; ux $bin agent install | grep -q 'agent upgraded'"
check "host id kept by the upgrade" bash -c "$(declare -f ux); ctr=$ctr; ux grep -q -- '--host-id clean-box' /home/tester/.config/systemd/user/llmctl-next.service"
check "agent still answers after the upgrade" bash -c "$(declare -f ux); ctr=$ctr; ux $inst agent status | grep -q 'socket: answers'"

# ---- uninstall --------------------------------------------------------------
check "agent uninstall --purge" ux "$inst" agent uninstall --purge
check "unit gone from the user manager" bash -c "$(declare -f ux); ctr=$ctr; ! ux systemctl --user cat llmctl-next.service"
after=$(snapshot)
if [ "$before" = "$after" ]; then
  say "PASS  file and directory list after --purge equals before"
else
  say "FAIL  file and directory list after --purge differs:"
  diff <(printf '%s\n' "$before") <(printf '%s\n' "$after") | tee -a "$log"
  failed=1
fi
check "uninstall with nothing installed refuses (no manifest)" bash -c "$(declare -f ux); ctr=$ctr; ux $bin agent uninstall 2>&1 | grep -q 'no install manifest'"
check "...and exits non-zero" bash -c "$(declare -f ux); ctr=$ctr; ! ux $bin agent uninstall"

# ---- a hand-written unit is left alone (the workstation case) ----------------
ux mkdir -p /home/tester/.config/systemd/user
ux sh -c 'printf "[Service]\nExecStart=/opt/other/llmctl-next.bin daemon\n" > /home/tester/.config/systemd/user/llmctl-next.service'
sum_before=$(ux sha256sum /home/tester/.config/systemd/user/llmctl-next.service)
check "install refuses a hand-written unit" bash -c "$(declare -f ux); ctr=$ctr; ! ux $bin agent install"
sum_after=$(ux sha256sum /home/tester/.config/systemd/user/llmctl-next.service)
check "hand-written unit unchanged" test "$sum_before" = "$sum_after"
check "no binary installed by the refused install" bash -c "$(declare -f ux); ctr=$ctr; ! ux test -e $inst"

# ---- bare alpine: the static binary without any service manager -------------
abin_dir=$(mktemp -d)
tar -xzf "$archive" -C "$abin_dir"
check "alpine: daemon + status with a temp data dir" docker run --rm -v "$abin_dir/$name:/x:ro" alpine:3.22 sh -c \
  '/x/llmctl-next --data-dir /tmp/d daemon --host-id alpine >/tmp/d.log 2>&1 & for i in $(seq 1 50); do [ -S /tmp/d/control.sock ] && break; sleep 0.2; done; /x/llmctl-next --data-dir /tmp/d status'
rm -rf "$abin_dir"

if [ "$failed" = 0 ]; then say "== ALL PASS"; else say "== FAILURES (see log)"; fi
say "log: $log"
exit "$failed"
