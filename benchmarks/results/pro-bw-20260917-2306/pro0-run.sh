#!/bin/bash
# PRO0 — headline refresh on current main (MER4.0-controlled protocol, cache state per rep).
set -u
cd /Users/cwong/ai/qwen38-mtp-server
OUT=benchmarks/results/pro-bw-20260917-2306
CELL=benchmarks/run_cell.sh
PFENV="MLX_CHUNKED_PREFILL=1 QWEN_PREFILL_CHUNK_SIZE=2048"
LOG() { echo "[2026-09-17 23:08:21] $*" | tee -a "$OUT/pro0.log"; }
LOG "PRO0 start. RUNID=pro-bw-20260917-2306"

# --- 1. Decode: essay-1024 + specdec-800, 6 reps, INTERLEAVED (DVFS-fair) ---
for r in 1 2 3 4 5 6; do
  LOG "decode essay r$r (MISS)"
  bash $CELL pro0-essay-r$r "QWEN_MTP_DRAFT_K=2" 18099 benchmarks/prompts/essay-1024.txt "" q4 1024 MISS > "$OUT/pro0-essay-r$r.json" 2>$OUT/pro0-essay-r$r.err
  LOG "decode specdec r$r (MISS)"
  bash $CELL pro0-specdec-r$r "QWEN_MTP_DRAFT_K=2" 18099 benchmarks/prompts/specdec-800.txt "" q4 1024 MISS > "$OUT/pro0-specdec-r$r.json" 2>$OUT/pro0-specdec-r$r.err
done

# --- 2. Prefill walls at pc=2048 (max_tokens=16 so wall ~ prefill; MISS) ---
for s in 8 16 32 96; do
  LOG "prefill longctx-${s}k (MISS, max_tokens=16)"
  bash $CELL pro0-prefill-${s}k "$PFENV" 18099 benchmarks/prompts/longctx-${s}k.txt "" q4 16 MISS > "$OUT/pro0-prefill-${s}k.json" 2>$OUT/pro0-prefill-${s}k.err
done

# --- 3. TTFT pair: cold MISS vs RAM-HIT (max_tokens=16, wall ~ TTFT) ---
LOG "ttft cold MISS (max_tokens=16)"
bash $CELL pro0-ttft-miss "QWEN_MTP_DRAFT_K=2" 18099 benchmarks/prompts/essay-1024.txt "" q4 16 MISS > "$OUT/pro0-ttft-miss.json" 2>$OUT/pro0-ttft-miss.err
LOG "ttft RAM-HIT (max_tokens=16)"
bash $CELL pro0-ttft-ramhit "QWEN_MTP_DRAFT_K=2" 18099 benchmarks/prompts/essay-1024.txt "" q4 16 RAM-HIT > "$OUT/pro0-ttft-ramhit.json" 2>$OUT/pro0-ttft-ramhit.err

LOG "PRO0 DONE"
