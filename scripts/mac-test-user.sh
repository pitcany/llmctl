#!/usr/bin/env bash
# Create or delete the throwaway macOS user of the M7e clean-host test
# (docs/RUST-M7-PLAN.md, decision 6). The owner runs this with sudo on the
# Mac. It touches the account `llmctltest` and no other account.
#
# Usage: sudo /bin/bash mac-test-user.sh create <public key file>
#        sudo /bin/bash mac-test-user.sh delete
#
# create: a standard (non-admin) user, allowed to log in over ssh with that
#         key. macOS asks for a new password: it is needed for the one
#         desktop login by fast user switching.
# delete: removes that account and its home. Log the user out first.
set -euo pipefail

user=llmctltest
full="llmctl clean-host test"
group=com.apple.access_ssh

die() { echo "mac-test-user: $*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] || die "run this on the Mac"
[ "$(id -u)" = 0 ] || die "run this with sudo"

case "${1:-}" in
  create)
    key=${2:-}
    [ -f "$key" ] || die "usage: create <public key file>"
    [ "$(grep -c . "$key")" = 1 ] && grep -qE '^(ssh-ed25519|ssh-rsa|ecdsa-sha2-[a-z0-9-]+) ' "$key" \
      || die "$key is not one public key"
    if id "$user" >/dev/null 2>&1; then die "$user exists already; nothing changed"; fi
    echo "Choose a password for $user when macOS asks."
    sysadminctl -addUser "$user" -fullName "$full" -home "/Users/$user" -password -
    # sysadminctl can exit 0 after a failure: check the result.
    id "$user" >/dev/null 2>&1 || die "macOS did not create $user"
    case " $(id -Gn "$user") " in *" admin "*) die "$user became an admin; delete it" ;; esac
    [ -d "/Users/$user" ] || createhomedir -c -u "$user" >/dev/null
    [ -d "/Users/$user" ] || die "no home for $user"
    # The group exists only when Remote Login is limited to some users.
    if dseditgroup -o read "$group" >/dev/null 2>&1; then
      dseditgroup -o edit -a "$user" -t user "$group"
    fi
    install -d -m 700 -o "$user" -g staff "/Users/$user/.ssh"
    install -m 600 -o "$user" -g staff "$key" "/Users/$user/.ssh/authorized_keys"
    echo "created $user (uid $(id -u "$user"), not an admin)"
    echo "next: log in as \"$full\" once by fast user switching, then switch back without logging it out"
    ;;
  delete)
    id "$user" >/dev/null 2>&1 || die "no user $user"
    dscl . -read "/Users/$user" RealName | grep -qF "$full" \
      || die "$user was not created by this script; nothing changed"
    if pgrep -u "$(id -u "$user")" -x Finder >/dev/null; then
      die "$user is logged in to the desktop; log it out first"
    fi
    if dseditgroup -o read "$group" >/dev/null 2>&1; then
      dseditgroup -o edit -d "$user" -t user "$group" || true
    fi
    sysadminctl -deleteUser "$user"
    if id "$user" >/dev/null 2>&1; then
      die "macOS did not delete $user; run this from Terminal on the Mac, not over ssh"
    fi
    echo "deleted $user"
    ;;
  *)
    sed -n '2,12p' "$0"
    exit 2
    ;;
esac
