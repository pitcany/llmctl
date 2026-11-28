#!/usr/bin/env bash
# M6c rehearsal: the cutover and both reversals (R1, R2) in a fresh
# Debian 12 container with systemd as PID 1 (docs/RUST-M6-REHEARSAL.md).
#
# Usage: scripts/rehearsal_m6c.sh [--runs N] [--keep] [--agent-bin FILE]
#   --runs N        fresh containers, one after the other (default 2).
#   --keep          leave the last container running for inspection.
#   --agent-bin F   a static (musl) llmctl-next to use; default: build one
#                   from HEAD into rust/dist/m6c-<sha>/ (cached).
#   --dev           take scripts/rehearsal_m6c.py from the working tree (to
#                   develop it); a recorded run never uses it.
# Env: LLMCTL_M6C_AI_REF (default origin/main of ~/AI),
#      LLMCTL_M6C_ROUTER_REF (default origin/main of ~/AI/llm-router),
#      LLMCTL_M6C_LOCAL_LLM ("host:repo", default m5:local-llm),
#      LLMCTL_M6C_LOCAL_LLM_REF (default origin/master).
#
# What it reads on this host: the committed sources (git archive, never a
# working tree), a backup copy of the Python DB (SQLite backup API, source
# opened read-only) and the preset files. No settings.yaml, no token, no
# env file. Every service-manager call runs inside the container, which has
# no network. Exit 0 = every check of every run passed.
set -uo pipefail

runs=2
keep=0
agent_bin=""
dev=0
while [ $# -gt 0 ]; do
  case "$1" in
    --runs) runs=$2; shift ;;
    --keep) keep=1 ;;
    --agent-bin) agent_bin=$2; shift ;;
    --dev) dev=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

repo=$(cd "$(dirname "$0")/.." && pwd)
sha=$(git -C "$repo" rev-parse HEAD)
short=${sha:0:12}
image=llmctl-m6c-rehearsal
work=$(mktemp -d "${TMPDIR:-/tmp}/llmctl-m6c.XXXXXX")
log=$work/rehearsal.log
ctr=""
trap '[ -n "$ctr" ] && [ "$keep" = 0 ] && docker rm -f "$ctr" >/dev/null 2>&1; rm -rf "$work/src"' EXIT
say() { printf '%s\n' "$*" | tee -a "$log"; }
die() { say "FAIL  $*"; exit 1; }

say "== M6c rehearsal at llmctl $short ($(date -u +%Y-%m-%dT%H:%M:%SZ))"

# ---- the agent: one static build of HEAD --------------------------------------
if [ -z "$agent_bin" ]; then
  out=$repo/rust/dist/m6c-$short
  agent_bin=$out/llmctl-next
  if [ ! -x "$agent_bin" ]; then
    say "building a static llmctl-next from $short (rust:1.100.0-alpine)"
    rm -rf "$out/src" && mkdir -p "$out/src"
    git -C "$repo" archive "$sha" rust | tar -x -C "$out/src"
    docker run --rm -v "$out/src:/src" -v llmctl-release-cargo-registry:/usr/local/cargo/registry \
      -e LLMCTL_GIT_SHA="$sha" -w /src/rust rust:1.100.0-alpine \
      sh -c "apk add --no-cache musl-dev >/dev/null && cargo build --release --locked -p llmctl-cli --bin llmctl-next && chown -R $(id -u):$(id -g) /src" \
      >"$out/build.log" 2>&1 || die "agent build (see $out/build.log)"
    cp "$out/src/rust/target/release/llmctl-next" "$agent_bin"
    rm -rf "$out/src"
  fi
fi
file "$agent_bin" | grep -q static || die "not a static binary: $agent_bin"
say "agent: $agent_bin sha256 $(sha256sum "$agent_bin" | cut -c1-16)"

# ---- the sources under test, from git objects ----------------------------------
ai_ref=${LLMCTL_M6C_AI_REF:-origin/main}
router_ref=${LLMCTL_M6C_ROUTER_REF:-origin/main}
mac_spec=${LLMCTL_M6C_LOCAL_LLM:-m5:local-llm}
mac_ref=${LLMCTL_M6C_LOCAL_LLM_REF:-origin/master}
src=$work/src
mkdir -p "$src"/{llmctl,AI,llm-router,local-llm,fixtures/presets}
git -C "$repo" archive "$sha" | tar -x -C "$src/llmctl"
if [ "$dev" = 1 ]; then
  cp "$repo/scripts/rehearsal_m6c.py" "$src/llmctl/scripts/"
  say "DEV: scripts/rehearsal_m6c.py from the working tree"
fi
git -C ~/AI archive "$ai_ref" bin heal services packages/llmctl-stack | tar -x -C "$src/AI" \
  || die "git archive ~/AI $ai_ref"
git -C ~/AI/llm-router archive "$router_ref" | tar -x -C "$src/llm-router" \
  || die "git archive llm-router $router_ref"
ssh -o BatchMode=yes -o ConnectTimeout=10 "${mac_spec%%:*}" \
  "git -C ${mac_spec#*:} archive $mac_ref bin lib registry" | tar -x -C "$src/local-llm" \
  || die "git archive local-llm $mac_spec $mac_ref"
say "sources: llmctl $short, ~/AI $(git -C ~/AI rev-parse --short "$ai_ref"), llm-router $(git -C ~/AI/llm-router rev-parse --short "$router_ref"), local-llm $(ssh -o BatchMode=yes "${mac_spec%%:*}" "git -C ${mac_spec#*:} rev-parse --short $mac_ref")"
cp "$agent_bin" "$src/llmctl-next"

# Fixtures: the Python DB through the backup API, and the preset files.
python3 - "$HOME/.local/share/llmctl/llmctl.sqlite3" "$src/fixtures/llmctl.sqlite3" <<'EOF' || die "DB backup"
import sqlite3, sys
s = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
d = sqlite3.connect(sys.argv[2])
s.backup(d)
assert d.execute("pragma integrity_check").fetchone()[0] == "ok"
d.execute("pragma journal_mode=delete")
d.close(); s.close()
EOF
for p in "$HOME"/.config/llmctl/presets/*.yaml; do
  case "$(basename "$p")" in _*) ;; *) cp "$p" "$src/fixtures/presets/" ;; esac
done
say "fixtures: DB sha256 $(sha256sum "$src/fixtures/llmctl.sqlite3" | cut -c1-16), $(ls "$src/fixtures/presets" | wc -l) presets"

# ---- the image: Debian 12 (bookworm) with systemd and Python 3.12 --------------
cat >"$work/Dockerfile" <<'EOF'
FROM python:3.12-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends systemd systemd-sysv dbus dbus-user-session libpam-systemd procps sqlite3 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 -s /bin/bash tester \
 && mkdir -p /var/lib/systemd/linger && touch /var/lib/systemd/linger/tester
# Dependencies first (a cached layer), then the code under test.
COPY src/llmctl/pyproject.toml /deps/llmctl.toml
COPY src/llm-router/requirements.txt /deps/router.txt
RUN python -m venv /opt/venv \
 && python -c "import tomllib; print(chr(10).join(tomllib.load(open('/deps/llmctl.toml','rb'))['project']['dependencies']))" >/deps/llmctl.txt \
 && /opt/venv/bin/pip install -q --no-cache-dir -r /deps/llmctl.txt -r /deps/router.txt
COPY src /src
RUN /opt/venv/bin/pip install -q --no-cache-dir --no-deps /src/llmctl /src/llm-router \
 && install -m 0755 /src/llmctl-next /usr/local/bin/llmctl-next
STOPSIGNAL SIGRTMIN+3
CMD ["/lib/systemd/systemd"]
EOF
say "building image $image"
docker build -q -t "$image" -f "$work/Dockerfile" "$work" >>"$log" 2>&1 || die "image build (see $log)"

failed=0
for run in $(seq 1 "$runs"); do
  ctr=llmctl-m6c-$$-$run
  say ""
  say "== run $run of $runs: fresh container $ctr"
  docker run -d --name "$ctr" --network none --cgroupns=private --security-opt writable-cgroups=true \
    --tmpfs /run --tmpfs /run/lock "$image" >>"$log" 2>&1 || die "container start"
  for _ in $(seq 1 60); do
    case "$(docker exec "$ctr" systemctl is-system-running 2>/dev/null)" in running|degraded) break ;; esac
    sleep 1
  done
  for _ in $(seq 1 30); do
    docker exec -u tester -e XDG_RUNTIME_DIR=/run/user/1000 "$ctr" systemctl --user is-system-running >/dev/null 2>&1 && break
    sleep 1
  done
  docker exec -u tester -w /home/tester \
    -e XDG_RUNTIME_DIR=/run/user/1000 \
    -e DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    -e PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    "$ctr" python3 /src/llmctl/scripts/rehearsal_m6c.py --as-host "$(hostname)" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || failed=1
  if [ "$run" -lt "$runs" ] || [ "$keep" = 0 ]; then docker rm -f "$ctr" >/dev/null 2>&1; ctr=""; fi
done
[ -n "$ctr" ] && say "kept container: $ctr"
if [ "$failed" = 0 ]; then say "== ALL $runs RUNS PASSED"; else say "== NOT PASSED"; fi
cp "$log" "$repo/rust/dist/m6c-rehearsal-$short.log" 2>/dev/null && say "log: $repo/rust/dist/m6c-rehearsal-$short.log"
exit "$failed"
