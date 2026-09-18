#!/bin/bash
# Item 1 (clean end-to-end) — verify-width fusion A/B at 8K (shorter prefill,
# decode-dominant, less re-heating than 32K). The verify fusion is bit-exact and
# context-independent (per-verify decode path), so 8K measures the same -1%.
# A = default (fused verify ON), B = MLX_QWEN_FUSED_GDN=0 (off). 6 reps interleaved.
set -u
RES=/Users/cwong/ai/qwen38-mtp-server/benchmarks/results/item1-verify-flip
cd /Users/cwong/ai/qwen38-mtp-server || exit 1
P8=benchmarks/prompts/longctx-8k.txt
PORT=18097
run() { bash benchmarks/run_cell.sh "$1" "$2" "$3" "$4" "" "$5" "$6" "$7" > "$RES/$1.json" 2>/dev/null; cp "/tmp/mtp-bench-$1.stderr" "$RES/$1.stderr" 2>/dev/null; }
for r in 1 2 3 4 5 6; do
  if [ $((r % 2)) -eq 1 ]; then
    run "A8-r$r" "" "$PORT" "$P8" q4 128 MISS
    run "B8-r$r" "MLX_QWEN_FUSED_GDN=0" "$PORT" "$P8" q4 128 MISS
  else
    run "B8-r$r" "MLX_QWEN_FUSED_GDN=0" "$PORT" "$P8" q4 128 MISS
    run "A8-r$r" "" "$PORT" "$P8" q4 128 MISS
  fi
  pmset -g therm > "$RES/therm-8-r$r.txt" 2>/dev/null || true
done
echo "DONE"
