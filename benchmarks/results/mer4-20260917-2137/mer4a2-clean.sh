#!/bin/bash
# MER4A.2 clean re-run of the decision region (M=512/1024), no contention.
set -u
OUTDIR=/Users/cwong/ai/qwen38-mtp-server/benchmarks/results/mer4-20260917-2137
cd /Users/cwong/ai/mlx-swift-lm
B=.build/release/qmvbench
W=/Users/cwong/ai/qwen38-mtp-server/weights
LOG() { echo "[$(date '+%F %T')] $*" | tee -a "$OUTDIR/mer4a2-clean.log"; }
LOG "clean re-run start. binary=$(shasum -a 256 $B | cut -d' ' -f1)"
for rep in 1 2; do
  LOG "clean rep=$rep: FFN M=512,1024 (batch=128 wall=90)"
  $B --weights $W --ffn-prefill --ffn-pair --ffn-ms 512,1024 --ffn-batch 128 --ffn-wall 90 \
    > "$OUTDIR/mer4a2-clean-rep$rep.json" 2>&1
done
LOG "clean re-run DONE"
