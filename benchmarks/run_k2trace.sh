#!/bin/bash
# K2 decomposition — Phase 3 external Metal System Trace of ONE default k=2
# greedy request (xctrace attach only, W2 style; no in-graph instrumentation
# beyond the host-side trace lines).
#
# Server env:
#   QWEN_MTP_STEP_TRACE=1            STEP-TRACE four-phase lines + MTP-STEP-
#                                    SUMMARY (server)
#   MLX_QWEN_MTP_TRACE=1             mtp-trace five-way split + mtp-anchor
#                                    absolute mach-uptime anchors (engine)
#   MLX_QWEN_MTP_TRACE_PATH=<file>   trace lines to a file (not stderr)
# The mtp-anchor lines carry absolute uptime-ns anchors per round phase; the
# xctrace GPU interval start-times are ns-since-boot, i.e. the SAME timebase,
# so host phase windows and GPU command-buffer intervals can be intersected
# directly (see analyze_k2trace.py).
#
# Default config: no QWEN_MTP_DRAFT_K, no --spec-draft-n-max -> pinned k=2.
# max_tokens 256 keeps the trace small (W2 rule).
#
# usage: run_k2trace.sh [port] [trace-out] [max_tokens]
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1
PORT=${1:-18099}
TRACE_OUT=${2:-/tmp/k2-profile.trace}
MAX_TOKENS=${3:-256}
PROMPT_FILE=$SERVER/benchmarks/prompts/essay-1024.txt

SERVER_BIN=$SERVER/.build/release/qwen38-mtp-server
STDERR_LOG=/tmp/k2trace.stderr
STDOUT_LOG=/tmp/k2trace.stdout
TRACE_FILE=/tmp/k2trace-mtp-trace.log
RESPONSE=/tmp/k2trace.response.json
REQUEST=/tmp/k2trace.request.json
rm -f "$STDERR_LOG" "$STDOUT_LOG" "$RESPONSE" "$REQUEST" "$TRACE_OUT" "$TRACE_FILE"

BIN_SHA256=$(shasum -a 256 "$SERVER_BIN" 2>/dev/null | awk '{print $1}')
# kern.boottime prints "usec NNN sec NNN" — the seconds field is $4.
BOOT_EPOCH=$(sysctl -n kern.boottime | awk '{print $4}')

pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
sleep 1

echo "[k2trace] binary sha256=$BIN_SHA256 boot_epoch=$BOOT_EPOCH"

QWEN_MTP_STEP_TRACE=1 MLX_QWEN_MTP_TRACE=1 MLX_QWEN_MTP_TRACE_PATH="$TRACE_FILE" \
  "$SERVER_BIN" serve --port "$PORT" --model ./weights \
  > "$STDOUT_LOG" 2> "$STDERR_LOG" &
SRV=$!

echo "[k2trace] server pid=$SRV, waiting for readyz"
CODE=000
for i in $(seq 1 150); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz" 2>/dev/null)
  if [ "$CODE" = "200" ]; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo '{"error":"SERVER DIED during load"}'
    exit 1
  fi
  sleep 2
done
if [ "$CODE" != "200" ]; then
  echo "{\"error\":\"readyz never 200 (last=$CODE)\"}"
  kill $SRV 2>/dev/null
  exit 1
fi

echo "[k2trace] server ready, pid=$SRV; attaching xctrace"

/tmp/benchvenv/bin/python - "$PROMPT_FILE" "$REQUEST" "$MAX_TOKENS" <<'PYEOF'
import json, sys
prompt = open(sys.argv[1]).read()
body = {
    "model": "qwen3.8-27b-mtp",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(sys.argv[3]),
    "temperature": 0,
    "top_k": 1,
    "mtp_enabled": True,
    "enable_thinking": False,
    "stream": False,
}
open(sys.argv[2], "w").write(json.dumps(body))
PYEOF

xctrace record --template 'Metal System Trace' --attach "$SRV" \
  --output "$TRACE_OUT" &
XTR=$!
sleep 4

echo "[k2trace] sending request (max_tokens=$MAX_TOKENS)"
WALL_SECONDS=$(curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @"$REQUEST" -o "$RESPONSE" -w '%{time_total}')
CURL_RC=$?
echo "[k2trace] request done wall=$WALL_SECONDS s rc=$CURL_RC"

sleep 2
kill -INT $XTR 2>/dev/null
for i in $(seq 1 60); do
  kill -0 $XTR 2>/dev/null || break
  sleep 1
done
kill $XTR 2>/dev/null

kill $SRV 2>/dev/null
pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
wait $SRV 2>/dev/null

echo "[k2trace] step summary:"
grep "MTP-STEP-SUMMARY" "$STDERR_LOG" || echo "[k2trace] NO STEP SUMMARY"
echo "[k2trace] head/draft lines:"
grep -E "MLXLM: MTP (head selected|draft depth)" "$STDOUT_LOG"
echo "[k2trace] mtp-trace lines: $(grep -c 'mtp-trace:' $TRACE_FILE 2>/dev/null || echo 0)"
echo "[k2trace] mtp-anchor lines: $(grep -c 'mtp-anchor:' $TRACE_FILE 2>/dev/null || echo 0)"

# export the tables the bucketing analysis needs
xctrace export --input "$TRACE_OUT" --toc --output /tmp/k2trace-toc.xml \
  || echo "[k2trace] TOC EXPORT FAILED"
xctrace export --input "$TRACE_OUT" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
  --output /tmp/k2trace-gpu-intervals.xml \
  || echo "[k2trace] GPU INTERVALS EXPORT FAILED"
xctrace export --input "$TRACE_OUT" \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-execution-points"]' \
  --output /tmp/k2trace-exec-points.xml \
  || echo "[k2trace] EXEC POINTS EXPORT FAILED"

ls -la "$TRACE_OUT" /tmp/k2trace-*.xml 2>/dev/null
echo "[k2trace] done"
exit 0