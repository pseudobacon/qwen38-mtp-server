#!/usr/bin/env bash
# run_radix_ssd_restart.sh
#
# Restart benchmark for the Radix SSD persistence tier.
#
#   Phase 1 (warm): start server (SSD on) -> request A (cold prefill, fills RAM)
#                   -> request A again (in-RAM hit, measure TTFT_warm) -> SIGTERM
#                   (graceful flush to SSD).
#   Phase 2 (disk): restart server (skeleton restored from SSD) -> request A
#                   (disk hit -> lazy load, measure TTFT_disk) -> request B
#                   (no match, cold re-prefill, measure TTFT_cold) -> SIGTERM.
#
# Acceptance: TTFT_disk <= 2 * TTFT_warm  AND  TTFT_disk <= 0.7 * TTFT_cold.
#
# Usage: bash benchmarks/run_radix_ssd_restart.sh
set -u

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

BIN=".build/arm64-apple-macosx/debug/qwen38-mtp-server"
PORT="${PORT:-8123}"
CACHE_DIR="${CACHE_DIR:-/tmp/radix-ssd-bench-$$}"
HOST="127.0.0.1"
BASE="http://$HOST:$PORT"
MODEL="qwen3.8-27b-mtp"
PROMPT_TOKENS="${PROMPT_TOKENS:-2048}"

# A fixed ~2048-token prompt (1 token ~ 4 chars -> ~8 KB of text).
build_content() {
    local n="$1" out=""
    local base="The quick brown fox jumps over the lazy dog near the old mill. "
    while [ ${#out} -lt $((n * 4)) ]; do
        out="$out$base"
    done
    printf '%s' "$out"
}
CONTENT="$(build_content "$PROMPT_TOKENS")"

# Build a chat-completions request body (max_tokens=1: we only measure TTFT).
request_body() {
    local file="$1"
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":1,"stream":true,"temperature":0}\n' \
        "$MODEL" "$CONTENT" > "$file"
}
request_body_b() {
    local file="$1"
    # A different ~2048-token prompt (B): same length as A (fair cold prefill)
    # but different content (never matches A's cached prefix).
    local n="$PROMPT_TOKENS" out=""
    local base="The slow lazy dog trots past the old mill near the river bank. "
    while [ ${#out} -lt $((n * 4)) ]; do out="$out$base"; done
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":1,"stream":true,"temperature":0}\n' \
        "$MODEL" "$out" > "$file"
}

wait_ready() {
    local tries=0
    while [ $tries -lt 180 ]; do
        local code
        code="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/readyz" 2>/dev/null || true)"
        [ "$code" = "200" ] && return 0
        tries=$((tries + 1))
        sleep 1
    done
    echo "FATAL: server did not become ready on :$PORT" >&2
    return 1
}

start_server() {
    "$ROOT/$BIN" \
        --host "$HOST" --port "$PORT" \
        --kv-ssd-enabled \
        --kv-ssd-cache-dir "$CACHE_DIR" \
        --kv-ssd-cache-gb 1 \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
}

stop_server() {
    [ -n "${SERVER_PID:-}" ] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    local tries=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        [ $tries -ge 60 ] && { kill -9 "$SERVER_PID" 2>/dev/null; break; }
        tries=$((tries + 1))
        sleep 1
    done
    SERVER_PID=""
}

t() {  # t <reqfile> -> TTFT proxy: time_total for a max_tokens=1 request
       # (the stream ends right after the single token is produced, so total
       # time == prefill + one decode step == time to first token)
    curl -s -o /dev/null -w '%{time_total}\n' \
        -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d @"$1" 2>/dev/null
}

REQ_A="/tmp/radix-ssd-A.$$.json"
REQ_B="/tmp/radix-ssd-B.$$.json"
SERVER_LOG="/tmp/radix-ssd-server.$$.log"
request_body "$REQ_A"
request_body_b "$REQ_B"
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

echo "=== Radix SSD restart benchmark ==="
echo "prompt tokens ~$PROMPT_TOKENS   port=$PORT   cache=$CACHE_DIR"

# ---- Phase 1: warm (in-RAM hit) -------------------------------------------
SERVER_PID=""
start_server
wait_ready || { stop_server; exit 1; }
echo "[warm] server ready"
t "$REQ_A" >/dev/null          # cold prefill, fills RAM
sleep 1
WARM="$(t "$REQ_A")"
echo "[warm] TTFT_warm=${WARM}s"
stop_server
echo "[warm] flushed + stopped"

# ---- Phase 2: disk (lazy load) + cold -------------------------------------
start_server
wait_ready || { stop_server; exit 1; }
echo "[disk] server ready (skeleton restored)"
DISK="$(t "$REQ_A")"
echo "[disk] TTFT_disk=${DISK}s"
COLD="$(t "$REQ_B")"
echo "[cold] TTFT_cold=${COLD}s"
stop_server

# ---- Report ----------------------------------------------------------------
echo ""
echo "=== Results ==="
echo "TTFT_warm (in-RAM hit):   ${WARM}s"
echo "TTFT_disk (SSD lazy):     ${DISK}s"
echo "TTFT_cold (re-prefill):   ${COLD}s"
awk -v warm="$WARM" -v disk="$DISK" -v cold="$COLD" 'BEGIN{
  r2 = (disk <= 2*warm) ? "PASS" : "FAIL";
  r30 = (disk <= 0.7*cold) ? "PASS" : "FAIL";
  printf "disk/warm  = %.2fx   (need <= 2.0x): %s\n", disk/warm, r2;
  printf "disk/cold  = %.2fx   (need <= 0.7x): %s\n", disk/cold, r30;
  printf "OVERALL: %s\n", (r2=="PASS" && r30=="PASS") ? "PASS" : "FAIL"; }'

rm -f "$REQ_A" "$REQ_B"
