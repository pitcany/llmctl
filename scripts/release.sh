#!/usr/bin/env bash
# Build release archives of llmctl-next (M7c, docs/RUST-M7-PLAN.md).
#
# Usage: scripts/release.sh [--linux-only | --mac-only] [--force] [COMMIT]
#   COMMIT  a committed revision (default HEAD). The build uses
#           `git archive COMMIT`, never the working tree, so uncommitted
#           changes are not in a release.
#   --force replace an existing output directory.
#
# Output: rust/dist/<version>-<short sha>/ (gitignored). Nothing is uploaded.
#   llmctl-next-<version>-x86_64-unknown-linux-musl.tar.gz
#   llmctl-next-<version>-aarch64-apple-darwin.tar.gz
#   SHA256SUMS, BUILDINFO
#
# Linux: static musl binary, built in the pinned rust:<channel>-alpine
#   image (Docker; channel from rust/rust-toolchain.toml).
# macOS: built over ssh on $LLMCTL_RELEASE_MAC (default mbp2) in its own
#   directory ~/llmctl-next-release-src/<sha>/, removed afterwards. It never
#   writes ~/.local/bin and never touches the agent there (no redeploy).
#   Minimum macOS: $MACOS_FLOOR (MACOSX_DEPLOYMENT_TARGET).
#
# Each archive holds llmctl-next, LICENSE and INSTALL.md (when the commit
# has docs/INSTALL.md). Archives are packed with sorted names, fixed owner
# and the commit's time, so the same binaries give the same archives.
set -euo pipefail

MACOS_FLOOR=13.0
mac_host=${LLMCTL_RELEASE_MAC:-mbp2}
do_linux=1
do_mac=1
force=0
commit_arg=HEAD
for arg in "$@"; do
  case "$arg" in
    --linux-only) do_mac=0 ;;
    --mac-only) do_linux=0 ;;
    --force) force=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) commit_arg=$arg ;;
  esac
done

say() { printf '== %s\n' "$*"; }
die() { printf 'release: %s\n' "$*" >&2; exit 1; }

repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
commit=$(git rev-parse --verify "${commit_arg}^{commit}") || die "not a commit: $commit_arg"
short=${commit:0:12}
ctime=$(git show -s --format=%ct "$commit")
version=$(git show "$commit:rust/Cargo.toml" | sed -n 's/^version = "\(.*\)"$/\1/p' | head -1)
channel=$(git show "$commit:rust/rust-toolchain.toml" | sed -n 's/^channel = "\(.*\)"$/\1/p')
[ -n "$version" ] || die "no workspace version in rust/Cargo.toml at $short"
[ -n "$channel" ] || die "no channel in rust/rust-toolchain.toml at $short"
lock_sha=$(git show "$commit:rust/Cargo.lock" | sha256sum | cut -d' ' -f1)
has_install_md=0
git cat-file -e "$commit:docs/INSTALL.md" 2>/dev/null && has_install_md=1

out="$repo/rust/dist/$version-$short"
if [ -e "$out" ]; then
  [ "$force" = 1 ] || die "$out exists (pass --force to replace it)"
  rm -rf "$out"
fi
mkdir -p "$out"
work=$(mktemp -d "${TMPDIR:-/tmp}/llmctl-release.XXXXXX")
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/files"
git show "$commit:LICENSE" >"$work/files/LICENSE"
[ "$has_install_md" = 1 ] && git show "$commit:docs/INSTALL.md" >"$work/files/INSTALL.md"

say "llmctl-next $version at $short (rust $channel)"
buildinfo="$out/BUILDINFO"
{
  echo "name: llmctl-next"
  echo "version: $version"
  echo "git_sha: $commit"
  echo "commit_time: $(date -u -d "@$ctime" +%Y-%m-%dT%H:%M:%SZ)"
  echo "toolchain: $channel"
  echo "cargo_lock_sha256: $lock_sha"
  echo "built_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [ "$has_install_md" = 1 ] || echo "note: docs/INSTALL.md not in this commit; archives carry no INSTALL.md"
} >"$buildinfo"

# pack TARGET BINARY: a reproducible tar.gz in $out.
pack() {
  local target=$1 bin=$2
  local name="llmctl-next-$version-$target"
  local d="$work/pkg/$name"
  mkdir -p "$d"
  install -m 0755 "$bin" "$d/llmctl-next"
  install -m 0644 "$work/files/LICENSE" "$d/LICENSE"
  [ "$has_install_md" = 1 ] && install -m 0644 "$work/files/INSTALL.md" "$d/INSTALL.md"
  tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$ctime" \
    -C "$work/pkg" -cf - "$name" | gzip -n -9 >"$out/$name.tar.gz"
  echo "binary_sha256 $target: $(sha256sum "$bin" | cut -d' ' -f1)" >>"$buildinfo"
}

if [ "$do_linux" = 1 ]; then
  image="rust:$channel-alpine"
  say "linux: pulling $image"
  docker pull -q "$image" >/dev/null
  digest=$(docker image inspect "$image" --format '{{index .RepoDigests 0}}')
  mkdir -p "$work/linux"
  git archive "$commit" rust | tar -x -C "$work/linux"
  say "linux: building in $digest"
  # RUSTUP_TOOLCHAIN: use the image's compiler as is (the toolchain file's
  # extra components are for development, not for this build).
  docker run --rm \
    -v "$work/linux:/src" \
    -v llmctl-release-cargo-registry:/usr/local/cargo/registry \
    -e RUSTUP_TOOLCHAIN="$channel" \
    -e LLMCTL_GIT_SHA="$commit" \
    -e LLMCTL_BUILD_HOST="docker $digest" \
    -w /src/rust "$digest" \
    sh -c "apk add --no-cache musl-dev >/dev/null \
      && cargo build --release --locked -p llmctl-cli --bin llmctl-next \
      && chown -R $(id -u):$(id -g) /src"
  lbin="$work/linux/rust/target/release/llmctl-next"
  file "$lbin" | grep -q 'static' || die "linux binary is not statically linked: $(file "$lbin")"
  echo "linux_image: $digest" >>"$buildinfo"
  pack x86_64-unknown-linux-musl "$lbin"
fi

if [ "$do_mac" = 1 ]; then
  rdir="llmctl-next-release-src/$commit"
  say "macos: building on $mac_host in ~/$rdir"
  git archive "$commit" rust | ssh "$mac_host" "rm -rf ~/$rdir && mkdir -p ~/$rdir && tar -x -C ~/$rdir"
  ssh "$mac_host" "cd ~/$rdir/rust \
    && export PATH=\"\$HOME/.cargo/bin:\$PATH\" MACOSX_DEPLOYMENT_TARGET=$MACOS_FLOOR LLMCTL_GIT_SHA=$commit \
    && export LLMCTL_BUILD_HOST=\"\$(hostname -s) macOS \$(sw_vers -productVersion)\" \
    && cargo build --release --locked -p llmctl-cli --bin llmctl-next 2>&1 | tail -2"
  mkdir -p "$work/mac"
  ssh "$mac_host" "cat ~/$rdir/rust/target/release/llmctl-next" >"$work/mac/llmctl-next"
  mac_info=$(ssh "$mac_host" "echo \"\$(hostname -s) macOS \$(sw_vers -productVersion) \$(uname -m)\"")
  ssh "$mac_host" "rm -rf ~/$rdir"
  file "$work/mac/llmctl-next" | grep -q 'Mach-O 64-bit.*arm64' \
    || die "macOS binary is not arm64 Mach-O: $(file "$work/mac/llmctl-next")"
  echo "macos_build_host: $mac_info" >>"$buildinfo"
  echo "macos_deployment_target: $MACOS_FLOOR" >>"$buildinfo"
  pack aarch64-apple-darwin "$work/mac/llmctl-next"
fi

(cd "$out" && sha256sum ./*.tar.gz | sed 's| \./| |' >SHA256SUMS)
say "done: $out"
ls -l "$out"
