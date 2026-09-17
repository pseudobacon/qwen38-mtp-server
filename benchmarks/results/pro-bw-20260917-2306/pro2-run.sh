#!/bin/bash
# PRO2 — in-pipeline round replay. FullBench width matrix (M=1,3) + serial,
# compared to the 81.6 ms in-pipeline tEvalAvg anchor.
set -u
cd /Users/cwong/ai/qwen38-mtp-server
OUT=benchmarks/results/pro-bw-20260917-2306
FB=/Users/cwong/ai/mlx-swift-lm/.build/debug/FullBench
echo "[2026-09-17 23:18:26] PRO2 start. FB=$FB"
echo "=== FullBench width matrix (M=1,3), prime=2048 ==="
$FB --model ./weights --prime 2048 --widths 1,3 --reps 12 --warmup 3 2>$OUT/pro2-widths.err | tee $OUT/pro2-widths.txt
echo "=== FullBench serial (M=1), prime=2048 ==="
$FB --model ./weights --prime 2048 --serial --reps 12 --warmup 3 2>$OUT/pro2-serial.err | tee $OUT/pro2-serial.txt
echo "[2026-09-17 23:18:26] PRO2 DONE"
