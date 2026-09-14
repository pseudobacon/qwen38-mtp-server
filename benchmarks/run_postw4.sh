#!/bin/bash
# Post-W4 queue timing session (Phases 2 + 3 + serial probes for Phase 5).
#
# Six cells, 6 reps each, interleaved per rep (headline / k=2 / serial on each
# fixture). Rep 1 of each cell is warmup (excluded from the median). All cells
# declare EXPECT_HEAD=q4 (the post-W4 default is the 4-bit tree; a missing tree
# fails startup loudly, and the cell-level gate backstops the head state).
#
#   h-*    headline, default config (cost-model depth; the post-W4 default)
#   k2-*   pinned depth 2 (QWEN_MTP_DRAFT_K=2) — Phase 3 evaluation
#   s-*    serial probe (--spec-draft-n-max 0) — Phase 5 in-pipeline floor
#
# usage: run_postw4.sh
set -u
DIR=/Users/cwong/ai/qwen38-mtp-server/benchmarks
cd /Users/cwong/ai/qwen38-mtp-server || exit 1
mkdir -p benchmarks/results
OUT=benchmarks/results/postw4.jsonl
: > "$OUT"

run_cell() {
  local TAG=$1 ENVSPEC=$2 PROMPT=$3 EXTRA=${4:-}
  echo "[postw4] cell $TAG env=[$ENVSPEC] extra=[$EXTRA] prompt=$PROMPT" >&2
  "$DIR/run_cell.sh" "$TAG" "$ENVSPEC" 18099 "$PROMPT" "$EXTRA" q4 >> "$OUT" 2>"$OUT".cellerr
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[postw4] cell $TAG failed rc=$rc — stopping" >&2
    exit $rc
  fi
  rm -f "$OUT".cellerr
}

ESSAY=benchmarks/prompts/essay-1024.txt
SPECDEC=benchmarks/prompts/specdec-800.txt
K2="QWEN_MTP_DRAFT_K=2"
SERIAL="--spec-draft-n-max 0"

for rep in 1 2 3 4 5 6; do
  run_cell h-essay-r$rep "" "$ESSAY"
  run_cell k2-essay-r$rep "$K2" "$ESSAY"
  run_cell s-essay-r$rep "" "$ESSAY" "$SERIAL"
  run_cell h-specdec-r$rep "" "$SPECDEC"
  run_cell k2-specdec-r$rep "$K2" "$SPECDEC"
  run_cell s-specdec-r$rep "" "$SPECDEC" "$SERIAL"
done
echo "[postw4] done -> $OUT" >&2