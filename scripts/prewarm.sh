#!/usr/bin/env bash
#
# prewarm.sh — post-install Metal-JIT pre-warm for qwen38-mtp-server.
#
# Runs the real server binary once with --prewarm-exit: the full startup
# (model load + the startup shape-warm, i.e. the exact decode-family Metal
# kernels a scored request uses), which populates Metal's built-in
# per-user JIT cache on THIS machine, then writes a version-keyed
# provenance manifest and exits. Every subsequent normal boot validates
# against the manifest and warns loudly on mismatch; a mismatch degrades
# to a cold JIT by Metal's own content-keyed cache — never a stale hit.
# See docs/PRE-WARM.md.
#
# The pre-warm must run as the SAME user, on the SAME machine, with the
# SAME deployment flags (model, head, --spec-draft-n-max, forced k) as the
# server it is preparing: the manifest is compared field-by-field at boot.
#
# usage:
#   scripts/prewarm.sh [serve-args...]   run the pre-warm (args are forwarded
#                                        to `serve`; pass the SAME flags the
#                                        deployment runs with). The HTTP port
#                                        is never bound (the run exits before
#                                        serving).
#   scripts/prewarm.sh check [serve-args...]
#                                        validate the current build against
#                                        the manifest (no model load);
#                                        exit 0 match / 1 mismatch / 2 no manifest
#
# Fails loudly (non-zero exit + log tail) if the run does not reach readyz
# or the manifest is not written.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1
BIN="$REPO_ROOT/.build/release/qwen38-mtp-server"

cmd="run"
if [ "${1:-}" = "check" ]; then
  cmd="check"
  shift
elif [ "${1:-}" = "run" ]; then
  shift
fi
case "$cmd" in
  check)
    exec env QWEN_STARTUP_TRACE=1 "$BIN" serve --prewarm-check "$@"
    ;;
esac

if [ ! -x "$BIN" ]; then
  echo "release binary missing at $BIN — building (swift build -c release)..." >&2
  swift build -c release
fi

MANIFEST="${QWEN_PREWARM_MANIFEST:-$HOME/.qwen38-mtp/prewarm-manifest.json}"
DUCD="$(getconf DARWIN_USER_CACHE_DIR 2>/dev/null || true)"
METAL_CACHE="${DUCD%/}/com.apple.metal"
LOG="$(mktemp -d /tmp/qwen38-prewarm.XXXXXX)/run.log"

metal_size_kb() {
  if [ -d "$METAL_CACHE" ]; then
    du -sk "$METAL_CACHE" 2>/dev/null | awk '{print $1}'
  else
    echo 0
  fi
}

SIZE_BEFORE="$(metal_size_kb)"
T0="$(perl -MTime::HiRes -e 'printf "%.3f", Time::HiRes::time()')"

set +e
env QWEN_STARTUP_TRACE=1 "$BIN" serve --prewarm-exit "$@" > "$LOG" 2>&1
RC=$?
set -e

T1="$(perl -MTime::HiRes -e 'printf "%.3f", Time::HiRes::time()')"
WALL="$(awk -v a="$T0" -v b="$T1" 'BEGIN { printf "%.1f", b - a }')"
SIZE_AFTER="$(metal_size_kb)"

if [ "$RC" -ne 0 ]; then
  echo "PREWARM FAILED (server exit $RC). Log tail:" >&2
  tail -25 "$LOG" >&2
  exit "$RC"
fi
if [ ! -f "$MANIFEST" ]; then
  echo "PREWARM FAILED: manifest not written at $MANIFEST. Log tail:" >&2
  tail -25 "$LOG" >&2
  exit 1
fi

echo "pre-warm complete: wall ${WALL}s (incl. manifest write), Metal cache ${SIZE_BEFORE}KB -> ${SIZE_AFTER}KB"
echo "manifest: $MANIFEST"
python3 - "$MANIFEST" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
print("  binary   " + m["binary_sha256"][:16] + "…")
print("  metallib " + m["metallib_sha256"][:16] + "… (" + m["metallib_path"] + ")")
print("  weights  " + m["model_weight_digest"][:16] + "… (" + m["model_path"] + ")")
print("  head     " + m["mtp_head_digest"][:16] + "… (" + m["mtp_head_path"] + ")")
print("  k/nmax   " + str(m["forced_draft_k"]) + "/" + str(m["spec_draft_n_max"])
      + "  os " + m["macos_build"] + "  hw " + m["hardware_id"])
PYEOF
echo "next boot: the server logs 'Prewarm check: provenance MATCH' (or a loud"
echo "MISMATCH warning) ~10 s after readyz — check the startup log."
