#!/bin/bash
# secret-scan.sh — refuse commits/pushes that carry credentials.
# Modes: lines (stdin) | staged (pre-commit) | push (pre-push, stdin=ref lines)
# Escape hatch: SECRETS_ALLOW=1 git commit/push ...
#
# v3.2 (2026-10-17): (a) findings via temp file — v2's flag died inside a
# pipeline subshell; (b) batched grep matching — v2 spawned subprocesses
# per-line and couldn't finish full-history scans; (c) stdin consumed only
# in `lines` mode — v3's top-level `cat` ate stdin before push-mode's
# `while read` ran, so push scans were silently empty.
set -u
[ "${SECRETS_ALLOW:-0}" = "1" ] && exit 0

ALLOW_RE='XXXXX|xxxxx|your[-_a-z0-9]|<redacted>|changeme|placeholder|REDACTED|e\.g\.|\.\.\.$'
# Placeholder VALUES: angle-bracket pointers ("<see-...>") are always safe —
# kconfig credential lines carrying them are placeholders, not secrets.
VALUE_PLACEHOLDER_RE='="<[^"]*>"'

SECRET_RE='tskey-auth-[A-Za-z0-9]+-[A-Za-z0-9]{20,}|sk-(ant|proj|svcacct)-[A-Za-z0-9_-]{20,}|CONFIG_[A-Z0-9_]*(PASS|PASSWORD|SECRET|TOKEN|API_KEY|AUTH_KEY|PRIVATE_KEY)[A-Z0-9_]*="[^"]+|-----BEGIN [A-Z ]*PRIVATE KEY-----'

RESULT="$(mktemp /tmp/secret-scan.XXXXXX)"
STREAM="$(mktemp /tmp/secret-stream.XXXXXX)"
trap 'rm -f "$RESULT" "$STREAM"' EXIT
: > "$RESULT"

scan_stream_file() {
  # $1 = file of candidate lines (already '+ '-stripped where applicable)
  grep -Ev "$ALLOW_RE" "$1" 2>/dev/null \
    | grep -Ev "$VALUE_PLACEHOLDER_RE" 2>/dev/null \
    | grep -En "$SECRET_RE" >> "$RESULT" || true
}

MODE="${1:-lines}"
case "$MODE" in
  lines)
    cat > "$STREAM"
    scan_stream_file "$STREAM"
    ;;
  staged)
    git diff --cached -U0 --diff-filter=ACM -- . ':!*managed_components*' ':!.githooks/*' 2>/dev/null \
      | grep '^+' | grep -v '^+++' > "$STREAM" || true
    scan_stream_file "$STREAM"
    ;;
  push)
    while read -r lref lsha rref rsha; do
      [ -n "${lsha:-}" ] || continue
      if [[ "${rsha:-}" =~ ^0+$ ]]; then
        range="$lsha"     # new remote branch: entire history is about to go public
      elif git merge-base --is-ancestor "$rsha" "$lsha" 2>/dev/null; then
        range="${rsha}..${lsha}"
      elif git cat-file -e "$rsha" 2>/dev/null; then
        range="${rsha}...${lsha}"
      else
        range="$lsha"     # unknown remote sha: fail safe, scan everything
      fi
      # Range BEFORE the pathspec separator (v3's bug: after '--' git read it
      # as a pathspec and scanned nothing).
      git log -p --no-ext-diff --format= "$range" -- . ':!*managed_components*' ':!.githooks/*' 2>/dev/null \
        | grep '^+' | grep -v '^+++' > "$STREAM" || true
      scan_stream_file "$STREAM"
    done
    ;;
esac

COUNT=$(wc -l < "$RESULT" | tr -d ' ')
if [ "$COUNT" -gt 0 ]; then
  echo "" >&2
  echo "secret-scan: BLOCKED — $COUNT credential-looking line(s) would be committed/pushed." >&2
  head -5 "$RESULT" | cut -c1-160 | sed 's/^/  SECRET? /' >&2
  echo "  Fix: move values into a git-ignored overlay (sdkconfig.credentials / .env)," >&2
  echo "  or 'git rm --cached <file>' for files that must stay untracked." >&2
  echo "  Secret already in history? git-filter-repo BEFORE the first push." >&2
  echo "  False positive? SECRETS_ALLOW=1 git commit/push ..." >&2
  exit 1
fi
exit 0