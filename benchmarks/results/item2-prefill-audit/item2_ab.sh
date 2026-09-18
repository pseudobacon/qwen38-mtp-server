#!/bin/bash
# Item 2 — _PREFILL divergence audit + prefill-win decision.
# The fused GDN prefill kernel engages per forward chunk (S <= 4096). Under
# MLX_CHUNKED_PREFILL (pc=2048) the GDN sees S=2048/forward at ALL lengths, so
# _PREFILL engages at 8K/16K/32K alike. The incumbent 32K stream family is
# 6576c099 (verify-agnostic; verify is bit-exact).
#
#   32K:  prefill off (incumbent) vs on (new family), 6 reps each, interleaved.
#         MLX_QWEN_TOP2_GAP_TRACE=1 logs the primary top-2 gap per round; the
#         harness also saves raw completion_ids for the divergence audit.
#   8K / 16K: dense (NO MLX_CHUNKED_PREFILL -> GDN sees S=8192/16384 > 4096,
#         kernel does NOT engage) -> expect bit-exact with the incumbent.
set -u
RES=/Users/cwong/ai/qwen38-mtp-server/benchmarks/results/item2-prefill-audit
cd /Users/cwong/ai/qwen38-mtp-server || exit 1
P32=benchmarks/prompts/longctx-32k.txt
P8=benchmarks/prompts/longctx-8k.txt
P16=benchmarks/prompts/longctx-16k.txt
PORT=18097

run() { # tag env port prompt expect-head max-tokens cache-state
  bash benchmarks/run_cell.sh "$1" "$2" "$3" "$4" "" "$5" "$6" "$7" > "$RES/$1.json" 2>/dev/null
  cp "/tmp/mtp-bench-$1.stderr" "$RES/$1.stderr" 2>/dev/null
}

# --- 32K: prefill off (O) vs on (P), 6 reps interleaved, gap-trace on ---
for r in 1 2 3 4 5 6; do
  if [ $((r % 2)) -eq 1 ]; then
    run "P32O-r$r" "MLX_CHUNKED_PREFILL=1 MLX_QWEN_TOP2_GAP_TRACE=1" "$PORT" "$P32" q4 128 MISS
    run "P32P-r$r" "MLX_CHUNKED_PREFILL=1 MLX_QWEN_TOP2_GAP_TRACE=1 MLX_QWEN_FUSED_GDN_PREFILL=1" "$PORT" "$P32" q4 128 MISS
  else
    run "P32P-r$r" "MLX_CHUNKED_PREFILL=1 MLX_QWEN_TOP2_GAP_TRACE=1 MLX_QWEN_FUSED_GDN_PREFILL=1" "$PORT" "$P32" q4 128 MISS
    run "P32O-r$r" "MLX_CHUNKED_PREFILL=1 MLX_QWEN_TOP2_GAP_TRACE=1" "$PORT" "$P32" q4 128 MISS
  fi
  pmset -g therm > "$RES/therm-32-r$r.txt" 2>/dev/null || true
done

# --- 8K / 16K: DENSE (no chunked prefill), prefill off vs on, 3 reps each ---
for L in 8 16; do
  P="$P8"; [ "$L" = "16" ] && P="$P16"
  for r in 1 2 3; do
    run "P${L}KO-r$r" "MLX_QWEN_TOP2_GAP_TRACE=1" "$PORT" "$P" q4 128 MISS
    run "P${L}KP-r$r" "MLX_QWEN_TOP2_GAP_TRACE=1 MLX_QWEN_FUSED_GDN_PREFILL=1" "$PORT" "$P" q4 128 MISS
  done
done
echo "DONE"
