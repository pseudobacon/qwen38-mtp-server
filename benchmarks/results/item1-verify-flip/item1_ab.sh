#!/bin/bash
# Item 1 — verify-width fused-GDN default flip: 32K A/B, 6 reps, interleaved,
# rotating start, MISS, chunked (dense 32K overflows).
#   A (default): MLX_CHUNKED_PREFILL=1                    -> verify ON  (flipped default)
#   B (rollback): MLX_CHUNKED_PREFILL=1 MLX_QWEN_FUSED_GDN=0 -> verify OFF (eager)
# Both are bit-exact (verify kernel is bit-exact), so A and B share one stream
# family; the gates are (1) that stream == registered 6576c099, (2) A faster than
# B in >=4/5 paired reps (the -1% transfers), (3) rollback knob engages (banner OFF
# in B, ON in A).
set -u
RES=/Users/cwong/ai/qwen38-mtp-server/benchmarks/results/item1-verify-flip
cd /Users/cwong/ai/qwen38-mtp-server || exit 1
PROMPT=benchmarks/prompts/longctx-32k.txt
PORT=18097
rm -f "$RES"/A-r*.json "$RES"/B-r*.json "$RES"/A-r*.banner "$RES"/B-r*.banner "$RES"/therm-r*.txt

run_rep() {
  local tag=$1 env=$2
  bash benchmarks/run_cell.sh "$tag" "$env" "$PORT" "$PROMPT" "" q4 128 MISS > "$RES/$tag.json" 2>/dev/null
  grep "fused GDN prework" "/tmp/mtp-bench-$tag.stdout" > "$RES/$tag.banner" 2>/dev/null || echo "(no banner)" > "$RES/$tag.banner"
}

for r in 1 2 3 4 5 6; do
  if [ $((r % 2)) -eq 1 ]; then
    run_rep "A-r$r" "MLX_CHUNKED_PREFILL=1"
    run_rep "B-r$r" "MLX_CHUNKED_PREFILL=1 MLX_QWEN_FUSED_GDN=0"
  else
    run_rep "B-r$r" "MLX_CHUNKED_PREFILL=1 MLX_QWEN_FUSED_GDN=0"
    run_rep "A-r$r" "MLX_CHUNKED_PREFILL=1"
  fi
  pmset -g therm > "$RES/therm-r$r.txt" 2>/dev/null || echo "(therm unavailable)" > "$RES/therm-r$r.txt"
done
echo "DONE"
