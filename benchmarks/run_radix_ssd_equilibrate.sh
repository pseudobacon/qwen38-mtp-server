#!/usr/bin/env bash
# run_radix_ssd_equilibrate.sh
#
# Task 6 Phase 2: re-measurement WITH EQUILIBRATION.
#
#   ONE full warm-up restart cycle, DISCARDED (equilibrates thermals + the OS
#   page cache), then N >= 7 measured restart cycles.
#
#   Per measured cycle, record:
#     - TTFT_warm  (in-RAM hit)
#     - TTFT_disk  (SSD lazy-load after restart)  <-- absolute, gates G2
#     - TTFT_cold  (re-prefill, no match)
#     - disk/warm  ratio                           (informational, G3)
#     - lazy-load duration in ms (server log "Lazy-loaded radix prefix ... in X ms")
#       so disk TTFT is decomposed into lazy-load vs. the rest.
#
#   Gate (see GATE-RESPEC.md, judged on the MEDIAN of the measured cycles):
#     G1: disk/cold <= 0.7x
#     G2: absolute post-restart disk TTFT <= 1.5 s
#     G3: (informational) median disk/warm + range
#
# Usage: bash benchmarks/run_radix_ssd_equilibrate.sh [N_MEASURED]
set -u

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

BIN=".build/arm64-apple-macosx/debug/qwen38-mtp-server"
PORT="${PORT:-8123}"
HOST="127.0.0.1"
BASE="http://$HOST:$PORT"
MODEL="qwen3.8-27b-mtp"
PROMPT_TOKENS="${PROMPT_TOKENS:-2048}"
N_MEASURED="${1:-8}"

# Persistent cache dir: the warm-up cycle's flush is what the measured cycles
# restore from (the real post-restart scenario). NOT cleared between cycles.
CACHE_DIR="/tmp/radix-ssd-equib-$$"

build_content() {
    local n="$1" out=""
    local base="The quick brown fox jumps over the lazy dog near the old mill. "
    while [ ${#out} -lt $((n * 4)) ]; do out="$out$base"; done
    printf '%s' "$out"
}
CONTENT="$(build_content "$PROMPT_TOKENS")"

request_body() {
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":1,"stream":true,"temperature":0}\n' \
        "$MODEL" "$CONTENT" > "$1"
}
request_body_b() {
    local n="$PROMPT_TOKENS" out=""
    local base="The slow lazy dog trots past the old mill near the river bank. "
    while [ ${#out} -lt $((n * 4)) ]; do out="$out$base"; done
    printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":1,"stream":true,"temperature":0}\n' \
        "$MODEL" "$out" > "$1"
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

SERVER_PID=""
# NOTE (Task 6 launch fix): the `--kv-ssd-*` CLI flags crash `app.execute()`
# (Vapor command dispatch rejects them: `.unknownCommand` / `.unknownInput`).
# The server can only be launched with `--host` / `--port` as CLI flags; the
# SSD tier is configured via the equivalent env vars (QWEN_KV_SSD_*). This is
# the only launch form that keeps the server alive to serve requests.
start_server() {
    QWEN_KV_SSD_ENABLED=1 \
    QWEN_KV_SSD_CACHE_DIR="$CACHE_DIR" \
    QWEN_KV_SSD_CACHE_GB=1 \
        "$ROOT/$BIN" \
        --host "$HOST" --port "$PORT" \
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
t() {  # TTFT proxy: time_total for a max_tokens=1 request
    curl -s -o /dev/null -w '%{time_total}\n' \
        -X POST "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d @"$1" 2>/dev/null
}

cleanup() { stop_server; rm -f "$REQ_A" "$REQ_B" "$SERVER_LOG"; }
trap cleanup EXIT

REQ_A="/tmp/radix-ssd-equib-A.$$.json"
REQ_B="/tmp/radix-ssd-equib-B.$$.json"
SERVER_LOG="/tmp/radix-ssd-equib-server.$$.log"
request_body "$REQ_A"
request_body_b "$REQ_B"
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR"

echo "=== Radix SSD equilibration benchmark ==="
echo "prompt tokens ~$PROMPT_TOKENS   port=$PORT   cache=$CACHE_DIR"
echo "warm-up cycles: 1 (discarded)   measured cycles: $N_MEASURED"
echo ""

# run_one_cycle -> prints "<warm> <disk> <cold> <lazy_ms>" (0 for any failed
# parse). A "cycle" = Phase 1 (warm) + Phase 2 (disk + cold), i.e. two
# restarts, matching run_radix_ssd_restart.sh's structure.
run_one_cycle() {
    # Phase 1: warm (fresh process -> cold prefill fills RAM -> warm hit -> flush)
    start_server
    wait_ready || { stop_server; return 1; }
    t "$REQ_A" >/dev/null          # cold prefill, fills RAM
    sleep 1
    local warm
    warm="$(t "$REQ_A")"
    stop_server                    # graceful flush to SSD
    # Phase 2: disk (restart -> skeleton restored -> request A disk hit -> lazy
    # load) + cold (request B, no match -> re-prefill).
    start_server
    wait_ready || { stop_server; return 1; }
    local disk
    disk="$(t "$REQ_A")"
    local cold
    cold="$(t "$REQ_B")"
    stop_server
    local lazy
    lazy="$(grep -oE 'from SSD tier in [0-9]+(\.[0-9]+)? ms' "$SERVER_LOG" 2>/dev/null | head -1 | awk '{print $(NF-1)}')"
    echo "${warm:-0} ${disk:-0} ${cold:-0} ${lazy:-0}"
}

# ---- warm-up cycle (DISCARDED) --------------------------------------------
echo "[warmup] running 1 discarded cycle (equilibrates thermals + page cache) ..."
run_one_cycle >/dev/null
echo "[warmup] done"
echo ""

# ---- measured cycles -------------------------------------------------------
DATA_FILE="/tmp/radix-ssd-equib-data.$$"
: > "$DATA_FILE"
for i in $(seq 1 "$N_MEASURED"); do
    echo "[cycle $i/$N_MEASURED] measuring ..."
    line="$(run_one_cycle)"
    read -r w d c l <<EOF2
$line
EOF2
    echo "  cycle $i: warm=${w}s disk=${d}s cold=${c}s lazy_ms=${l}"
    echo "$w $d $c $l" >> "$DATA_FILE"
done

# ---- report (median + full range) -----------------------------------------
echo ""
echo "=== Raw measured cycles (order: warm disk cold lazy_ms) ==="
cat "$DATA_FILE"
echo ""
echo "=== Median + full range (measured cycles only) ==="
awk '
function median(a, n,   i, j, t) {           # sorts a[1..n] in place
    for (i = 2; i <= n; i++) { t = a[i]
        for (j = i - 1; j >= 1 && a[j] > t; j--) a[j+1] = a[j]
        a[j+1] = t }
    return (n % 2) ? a[(n+1)/2] : (a[n/2] + a[n/2+1]) / 2
}
function summarize(name, a, n,   i, med, mn, mx, extra) {
    med = median(a, n); mn = a[1]; mx = a[n]
    extra = (name == "lazy") ? "  (ms)" : ""
    printf "%-10s median=%.4f  range=[%.4f, %.4f]  n=%d%s\n", name, med, mn, mx, n, extra
    if (name == "dwarm") { dwarm_med = med; return }
    if (name == "disk")  { disk_med  = med; return }
    if (name == "cold")  { cold_med  = med; return }
}
{
    warm[NR] = $1 + 0; disk[NR] = $2 + 0; cold[NR] = $3 + 0; lazy[NR] = $4 + 0
    dwarm[NR] = ($1 + 0 > 0) ? ($2 / $1) : 0
    n = NR
}
END {
    summarize("warm",  warm,  n)
    summarize("disk",  disk,  n)
    summarize("cold",  cold,  n)
    summarize("lazy",  lazy,  n)
    summarize("dwarm", dwarm, n)
    g1 = (disk_med <= 0.7 * cold_med) ? "PASS" : "FAIL"
    g2 = (disk_med <= 1.5)            ? "PASS" : "FAIL"
    printf "\n=== GATE (on the MEDIAN of %d measured cycles) ===\n", n
    printf "G1  disk/cold = %.3fx  (need <= 0.7x)  : %s\n", disk_med / cold_med, g1
    printf "G2  abs disk  = %.3fs  (need <= 1.5s)  : %s\n", disk_med, g2
    printf "G3  disk/warm = %.3fx  (informational)\n", dwarm_med
    printf "OVERALL: %s\n", ((g1 == "PASS" && g2 == "PASS") ? "PASS" : "FAIL")
}
' "$DATA_FILE"

rm -f "$DATA_FILE"
