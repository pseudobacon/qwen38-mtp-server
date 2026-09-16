#!/bin/bash
# A/B benchmark runner for the compact-space rejection walk
# (docs/compact-rejection-rfc.md §7).
#
# Usage:
#   benchmarks/run_compact_rejection_ab.sh <baseline_binary> <feature_binary> [rounds]
#
# Protocol (per docs/speculative-sampling-rfc.md §6 / user spec):
#   - Fixed prompt (benchmarks/prompts/compact-rejection-prompt.json),
#     temp 0.7, top_p 0.95, max_tokens 300, stream=true.
#   - QWEN_MLX_SEED fixed so each binary's sampling stream is reproducible.
#   - MLX_QWEN_MTP_TRACE=1 per round -> per-round verify_ms + active_bytes.
#   - Interleaved A/B, >= 5 runs per side, >= 60 s cooldown between runs.
#   - Metrics: decode tok/s (server log), response wall time, per-round
#     verify_ms (trace), peak RSS (sampler); active_bytes (feature trace).
#
# Output: benchmarks/results/compact-rejection/<timestamp>/results.csv
set -u

BIN_A=${1:?usage: run_compact_rejection_ab.sh <baseline_binary> <feature_binary> [rounds]}
BIN_B=${2:?missing <feature_binary>}
ROUNDS=${3:-5}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PROMPT="$ROOT/benchmarks/compact-rejection-prompt.json"
OUT="$ROOT/benchmarks/results/compact-rejection/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
CSV="$OUT/results.csv"
PORT=8000
COOLDOWN_S=60
export QWEN_MLX_SEED=20260916

echo "timestamp" "side" "binary" "decode_tps" "wall_s" "chunks" "peak_rss_mb" "rounds" "round_us_mean" "active_mb_first" "active_mb_last" "active_mb_delta"
> "$CSV"

# --- helpers -------------------------------------------------------------

wait_ready() {
    local pid=$1
    for _ in $(seq 1 300); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "server died during startup" >&2
            return 1
        fi
        if curl -s -m 2 "http://127.0.0.1:$PORT/readyz" | grep -q '"status":"ready"'; then
            return 0
        fi
        sleep 1
    done
    echo "server never became ready" >&2
    return 1
}

run_once() {
    local side=$1 bin=$2 n=$3
    local logf="$OUT/server_${side}${n}.log"
    local trace="$OUT/rounds_${side}${n}.tsv"
    local rssfile="$OUT/rss_${side}${n}.txt"

    MLX_QWEN_MTP_TRACE=1 MLX_QWEN_MTP_TRACE_PATH="$trace" \
        QWEN_PORT=$PORT "$bin" > "$logf" 2>&1 &
    local pid=$!

    # Peak-RSS sampler (100 ms cadence).
    (
        while kill -0 "$pid" 2>/dev/null; do
            ps -o rss= -p "$pid" 2>/dev/null
            sleep 0.1
        done
    ) > "$rssfile" &
    local sampler=$!

    if ! wait_ready "$pid"; then
        kill -9 "$pid" 2>/dev/null
        return 1
    fi

    # Fixed non-greedy request; time the full stream.
    local start=$(date +%s.%N)
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d @"$PROMPT" -o "$OUT/resp_${side}${n}.json"
    local end=$(date +%s.%N)
    local wall=$(echo "$end $start" | awk '{printf "%.3f", $1 - $2}')
    local chunks=$(grep -c '^data:' "$OUT/resp_${side}${n}.json" 2>/dev/null || echo 0)

    # Give the server a beat to flush the final trace line, then stop it.
    sleep 2
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    kill "$sampler" 2>/dev/null

    # Server logs don't emit a tokens/s line at INFO level; derive it from
    # the streamed chunk count over wall time (TTFT bias ~1%, identical on
    # both sides). The per-round round_us from the trace is the primary
    # cost metric anyway.
    local tps=$(grep -o 'tokens/s=[0-9.]*' "$logf" | tail -1 | cut -d= -f2)
    if [ -z "$tps" ] && [ -n "$wall" ]; then
        tps=$(awk -v c="${chunks}" -v w="${wall}" 'BEGIN { if (w > 0) printf "%.2f", c / w }')
    fi
    local peak_rss=$(sort -n "$rssfile" | tail -1)
    [ -z "$peak_rss" ] && peak_rss=0
    local peak_rss_mb=$((peak_rss / 1024))
    local round_us active_first active_last
    # mtp-trace lines are space-separated key=value; round_us is the per-round
    # wall time in microseconds. active_bytes is present only in builds with
    # the compact-walk change (baseline shows NA).
    if [ -f "$trace" ]; then
        round_us=$(grep -o 'round_us=[0-9]*' "$trace" | cut -d= -f2 \
            | awk '{s+=$1; n++} END {if (n) printf "%.1f", s/n}')
        active_first=$(grep -o 'active_bytes=[0-9]*' "$trace" | head -1 | cut -d= -f2)
        active_last=$(grep -o 'active_bytes=[0-9]*' "$trace" | tail -1 | cut -d= -f2)
    else
        round_us=""; active_first=""; active_last=""
    fi
    local active_delta=""
    if [ -n "$active_first" ] && [ -n "$active_last" ]; then
        active_delta=$(( (active_last - active_first) / 1048576 ))
        active_first=$((active_first / 1048576))
        active_last=$((active_last / 1048576))
    fi
    local rounds_count
    rounds_count=$(grep -c 'mtp-trace:' "$trace" 2>/dev/null || echo 0)

    local line="$(date +%Y-%m-%dT%H:%M:%S) $side $(basename "$bin") ${tps:-NA} ${wall} ${chunks} ${peak_rss_mb} ${rounds_count} ${round_us:-NA} ${active_first:-NA} ${active_last:-NA} ${active_delta:-NA}"
    echo "$line"
    echo "$line" >> "$CSV"
}

# --- main: interleaved A/B with cooldowns --------------------------------

for i in $(seq 1 "$ROUNDS"); do
    echo "=== round $i/$ROUNDS: A (baseline) ==="
    run_once A "$BIN_A" "$i"
    echo "cooldown ${COOLDOWN_S}s"; sleep "$COOLDOWN_S"
    echo "=== round $i/$ROUNDS: B (feature) ==="
    run_once B "$BIN_B" "$i"
    echo "cooldown ${COOLDOWN_S}s"; sleep "$COOLDOWN_S"
done

# --- summary -------------------------------------------------------------

echo ""
echo "=== summary (median decode tok/s) ==="
for side in A B; do
    med=$(awk -F' ' -v s="$side" '$2 == s && $4 != "NA" {print $4}' "$CSV" \
        | sort -n | awk '{a[NR]=$1} END {if (NR % 2) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}')
    echo "side=$side median_decode_tps=$med runs=$(awk -v s="$side" '$2==s' "$CSV" | wc -l | tr -d ' ')"
done
echo "results: $CSV"
