#!/bin/bash
# B4 TTFT benchmark driver: start the release server, run prefix_ttft.py
# (multi-turn exact-prefix reuse vs cold-miss control), then stop the server.
#
# usage: run_prefix_ttft.sh [port]
#   Emits the TTFT JSON from prefix_ttft.py on stdout.
set -u
PORT=${1:-18099}
SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1

SERVER_BIN=$SERVER/.build/release/qwen38-mtp-server
if [ ! -x "$SERVER_BIN" ]; then
  echo '{"error":"release binary missing — run: swift build -c release --product qwen38-mtp-server"}'
  exit 1
fi

# never run against a stale server holding the port
pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
sleep 1

"$SERVER_BIN" serve --port "$PORT" --model ./weights >/tmp/prefix-ttft.stdout 2>/tmp/prefix-ttft.stderr &
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

/tmp/benchvenv/bin/python "$SERVER/benchmarks/prefix_ttft.py" "$PORT"
RC=$?

kill $SRV 2>/dev/null
pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
wait $SRV 2>/dev/null
exit $RC
