#!/bin/bash
# W2 — external Metal System Trace of ONE greedy request (no in-graph
# instrumentation; xctrace attach only, per the W2 hard rule).
#
# Starts the release server with the DEFAULT config (fusions ON, compiled
# decode ON, MLX_QWEN_QMV_VERIFY unset = default ON), waits for /readyz
# (weights + warm-up complete), attaches xctrace to the running pid, sends
# one essay request (max_tokens 256 to keep the trace small), waits for the
# response, finalizes the trace (SIGINT), and stops the server.
#
# usage: run_w2trace.sh [port] [trace-out] [max_tokens]
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1
PORT=${1:-18099}
TRACE_OUT=${2:-/tmp/w2-profile.trace}
PROMPT_FILE=$SERVER/benchmarks/prompts/essay-1024.txt
MAX_TOKENS=${3:-256}

SERVER_BIN=$SERVER/.build/release/qwen38-mtp-server
STDERR_LOG=/tmp/w2-trace.stderr
STDOUT_LOG=/tmp/w2-trace.stdout
RESPONSE=/tmp/w2-trace.response.json
REQUEST=/tmp/w2-trace.request.json
rm -f "$STDERR_LOG" "$STDOUT_LOG" "$RESPONSE" "$REQUEST" "$TRACE_OUT"

BIN_SHA256=$(shasum -a 256 "$SERVER_BIN" 2>/dev/null | awk '{print $1}')

echo "[w2trace] binary sha256=$BIN_SHA256"

# never run against a stale server holding the port
pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
sleep 1

QWEN_MTP_STEP_TRACE=1 "$SERVER_BIN" serve --port "$PORT" --model ./weights \
  > "$STDOUT_LOG" 2> "$STDERR_LOG" &
SRV=$!

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

echo "[w2trace] server ready, pid=$SRV; attaching xctrace"

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

echo "[w2trace] sending request (max_tokens=$MAX_TOKENS)"
WALL_SECONDS=$(curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @"$REQUEST" -o "$RESPONSE" -w '%{time_total}')
CURL_RC=$?
echo "[w2trace] request done wall=$WALL_SECONDS s rc=$CURL_RC"

# give the last kernels a moment to retire, then finalize the trace
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

echo "[w2trace] step summary:"
grep "MTP-STEP-SUMMARY" "$STDERR_LOG" || echo "[w2trace] NO STEP SUMMARY"
echo "[w2trace] trace file:"
ls -la "$TRACE_OUT" 2>/dev/null || echo "[w2trace] TRACE MISSING"
exit 0
