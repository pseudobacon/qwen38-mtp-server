#!/bin/bash
# MER4A — qmvbench sustained re-baseline (v0.32.2).
set -u
OUTDIR=/Users/cwong/ai/qwen38-mtp-server/benchmarks/results/mer4-20260917-2137
cd /Users/cwong/ai/mlx-swift-lm
B=.build/release/qmvbench
W=/Users/cwong/ai/qwen38-mtp-server/weights
LOG() { echo "[$(date '+%F %T')] $*" | tee -a "$OUTDIR/mer4a-run.log"; }

LOG "MER4A start. binary=$(shasum -a 256 $B | cut -d' ' -f1)"
pmset -g therm 2>/dev/null | tee -a "$OUTDIR/mer4a-thermal-pre.log"

# --- MER4A.1: verify-width M=1..9 (routed vs fallback, sustained) ---
LOG "MER4A.1: verify-width M=1..9 (warmup=50 timed=500 blocks=3 throughput=200)"
$B --weights $W --ms 1,2,3,4,5,6,7,8,9 --warmup 50 --timed 500 --blocks 3 --throughput 200 \
  > "$OUTDIR/mer4a1-verify-m1-9.json" 2>&1
LOG "MER4A.1 done"

# --- MER4A.2: FFN M-curve (interleaved DVFS-fair, 2 reps at M=512/1024) ---
for rep in 1 2; do
  LOG "MER4A.2 rep=$rep: FFN M=512,1024 (batch=128 wall=60)"
  $B --weights $W --ffn-prefill --ffn-pair --ffn-ms 512,1024 --ffn-batch 128 --ffn-wall 60 \
    > "$OUTDIR/mer4a2-ffn-m512-1024-rep$rep.json" 2>&1
done
LOG "MER4A.2: FFN M=2048 (batch=64 wall=60)"
$B --weights $W --ffn-prefill --ffn-pair --ffn-ms 2048 --ffn-batch 64 --ffn-wall 60 \
  > "$OUTDIR/mer4a2-ffn-m2048.json" 2>&1
LOG "MER4A.2: FFN M=4096 (batch=32 wall=60)"
$B --weights $W --ffn-prefill --ffn-pair --ffn-ms 4096 --ffn-batch 32 --ffn-wall 60 \
  > "$OUTDIR/mer4a2-ffn-m4096.json" 2>&1
LOG "MER4A.2: FFN M=8192 (batch=16 wall=60)"
$B --weights $W --ffn-prefill --ffn-pair --ffn-ms 8192 --ffn-batch 16 --ffn-wall 60 \
  > "$OUTDIR/mer4a2-ffn-m8192.json" 2>&1

# --- MER4A.3: bit-exactness / tolerance spot-checks at the FFN widths ---
LOG "MER4A.3: ffn-check M=512,1024,2048 (dequant reference)"
$B --weights $W --ffn-prefill --ffn-check --ffn-ms 512,1024,2048 \
  > "$OUTDIR/mer4a3-ffn-check.json" 2>&1

pmset -g therm 2>/dev/null | tee -a "$OUTDIR/mer4a-thermal-post.log"
LOG "MER4A DONE"
