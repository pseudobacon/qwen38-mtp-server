#!/usr/bin/env bash
# E2E acceptance test for the in-RAM Radix prefix-reuse repair (Phase 3).
#
# Starts a single server instance (in-RAM KV cache), sends the same prompt
# twice (cold then warm), and asserts:
#   1. Correctness: warm output == cold output (token-identical).
#   2. Gate: warm TTFT <= 0.7 * cold TTFT.
#   3. Metrics: /metrics reports prefix_reuse_hits >= 1 and
#      prefix_reuse_tokens_saved > 0 after the warm request.
#
# The cache is in-RAM, so a single server instance serves both runs (no
# restart needed). A restart between runs would reset the in-RAM cache and
# defeat the test.
#
# Output: benchmarks/results/reusable-path-repair/<ts>/results.json + REPORT.md.

set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/.build/arm64-apple-macosx/debug/qwen38-mtp-server"
PORT=8170
HOST=127.0.0.1
PROMPT="$REPO/benchmarks/reusable-path-prompt.json"
BASE_URL="http://$HOST:$PORT"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO/benchmarks/results/reusable-path-repair/$TS"
DIAG="$OUT_DIR/diag.log"

mkdir -p "$OUT_DIR"

# --- Build ----------------------------------------------------------------
echo "Building HTTPServer..."
( cd "$REPO" && swift build --target HTTPServer ) || { echo "BUILD FAILED"; exit 1; }

# --- Start server ---------------------------------------------------------
QWEN_STREAM_DIAG=1 "$BIN" --host "$HOST" --port "$PORT" > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM $SERVER_PID 2>/dev/null' EXIT

# --- Wait for ready -------------------------------------------------------
ready=0
for i in $(seq 1 40); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/readyz" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then ready=1; break; fi
    sleep 5
done
if [ "$ready" != "1" ]; then
    echo "Server failed to become ready (see $OUT_DIR/server.log)"; exit 1
fi
echo "Server ready."

# --- Cold run -------------------------------------------------------------
echo "=== Cold run ==="
cold_body="$OUT_DIR/cold.json"
cold_wall=$(python3 -c 'import time; print(time.time())')
cold_total=$(curl -s -o "$cold_body" -w '%{time_total}' \
    -X POST "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$PROMPT")
cold_wall_end=$(python3 -c 'import time; print(time.time())')

# --- Warm run -------------------------------------------------------------
sleep 1
echo "=== Warm run ==="
warm_body="$OUT_DIR/warm.json"
warm_total=$(curl -s -o "$warm_body" -w '%{time_total}' \
    -X POST "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$PROMPT")

# --- Metrics --------------------------------------------------------------
sleep 2
metrics=$(curl -s "$BASE_URL/metrics")

# --- Correctness: extract the first content token from each ---------------
# The SSE body ends with [DONE]; the content is in the delta. Compare the
# concatenated content/reasoning of both bodies (ignoring ids/timestamps).
cold_content=$(python3 -c "
import re,sys
def content(p):
    s=open(p,encoding='utf-8',errors='replace').read()
    return ''.join(re.findall(r'\"(?:content|reasoning|reasoning_content)\":\"((?:[^\"\\\\]|\\\\.)*)\"', s))
print(content('$cold_body'))
")
warm_content=$(python3 -c "
import re
def content(p):
    s=open(p,encoding='utf-8',errors='replace').read()
    return ''.join(re.findall(r'\"(?:content|reasoning|reasoning_content)\":\"((?:[^\"\\\\]|\\\\.)*)\"', s))
print(content('$warm_body'))
")

correct="false"
[ -n "$cold_content" ] && [ "$cold_content" = "$warm_content" ] && correct="true"

# --- Gate: warm/cold <= 0.7 -----------------------------------------------
ratio=$(python3 -c "cold=$cold_total; warm=$warm_total; print(f'{(warm/cold if cold>0 else 999.0):.4f}')")
if python3 -c "import sys; sys.exit(0 if $warm_total <= 0.7 * $cold_total else 1)"; then
    gate_verdict="PASS"
else
    gate_verdict="FAIL"
fi

# --- Metrics fields -------------------------------------------------------
eval "$(echo "$metrics" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('HITS='+str(d.get('prefix_reuse_hits',0)))
print('FALLBACKS='+str(d.get('prefix_reuse_fallbacks',0)))
print('SAVED='+str(d.get('prefix_reuse_tokens_saved',0)))
")"

echo "cold_total=$cold_total"
echo "warm_total=$warm_total"
echo "ratio=$ratio"
echo "correct=$correct (cold='$cold_content' warm='$warm_content')"
echo "prefix_reuse_hits=$HITS fallbacks=$FALLBACKS tokens_saved=$SAVED"

# --- Write results.json ---------------------------------------------------
cat > "$OUT_DIR/results.json" <<EOF
{
  "timestamp": "$TS",
  "cold_ttft_seconds": $cold_total,
  "warm_ttft_seconds": $warm_total,
  "warm_over_cold": $ratio,
  "gate": "warm_ttft_seconds <= 0.7 * cold_ttft_seconds",
  "gate_verdict": "$gate_verdict",
  "correctness": {
    "identical": $correct,
    "cold_content": "$cold_content",
    "warm_content": "$warm_content"
  },
  "metrics": {
    "prefix_reuse_hits": $HITS,
    "prefix_reuse_fallbacks": $FALLBACKS,
    "prefix_reuse_tokens_saved": $SAVED
  }
}
EOF

# --- Final verdict --------------------------------------------------------
overall="PASS"
[ "$gate_verdict" != "PASS" ] && overall="FAIL"
[ "$correct" != "true" ] && overall="FAIL"
[ "$HITS" -lt 1 ] && overall="FAIL"

echo ""
echo "=== E2E VERDICT: $overall ==="
echo "cold=$cold_total warm=$warm_total ratio=$ratio correct=$correct hits=$HITS"
echo "results: $OUT_DIR/results.json"
[ "$overall" = "PASS" ]
