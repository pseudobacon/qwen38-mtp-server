#!/bin/bash
# Phase 4 gate session (post-fix): essay deep-k cells k in {5,6,8}.
#
# After the SDPA exactness chunk fix in attentionWithCacheUpdate, verify
# widths 6..8 must be deterministic per config and in the serial family (no
# gross early corruption — the W3 pre-fix signature was divergence at the
# first drafted token of the first verify round). 6 reps per cell, rep 1
# warmup. All cells declare EXPECT_HEAD=q4. The trailing single h-* cells
# re-gate the headline hashes (the fix must not perturb default-config
# streams; their timing is not measured here).
#
# usage: run_deepk.sh
set -u
DIR=/Users/cwong/ai/qwen38-mtp-server/benchmarks
cd /Users/cwong/ai/qwen38-mtp-server || exit 1
mkdir -p benchmarks/results
OUT=benchmarks/results/deepk.jsonl
: > "$OUT"

run_cell() {
  local TAG=$1 ENVSPEC=$2 PROMPT=$3 EXTRA=${4:-}
  echo "[deepk] cell $TAG env=[$ENVSPEC] extra=[$EXTRA] prompt=$PROMPT" >&2
  "$DIR/run_cell.sh" "$TAG" "$ENVSPEC" 18099 "$PROMPT" "$EXTRA" q4 >> "$OUT" 2>"$OUT".cellerr
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[deepk] cell $TAG failed rc=$rc — stopping" >&2
    exit $rc
  fi
  rm -f "$OUT".cellerr
}

ESSAY=benchmarks/prompts/essay-1024.txt
SPECDEC=benchmarks/prompts/specdec-800.txt
MAX8="--spec-draft-n-max 8"

for rep in 1 2 3 4 5 6; do
  run_cell d5-essay-r$rep "QWEN_MTP_DRAFT_K=5" "$ESSAY" "$MAX8"
  run_cell d6-essay-r$rep "QWEN_MTP_DRAFT_K=6" "$ESSAY" "$MAX8"
  run_cell d8-essay-r$rep "QWEN_MTP_DRAFT_K=8" "$ESSAY" "$MAX8"
done
run_cell h-essay-regate "" "$ESSAY"
run_cell h-specdec-regate "" "$SPECDEC"
echo "[deepk] done -> $OUT" >&2
