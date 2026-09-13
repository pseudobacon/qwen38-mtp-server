#!/bin/bash
# Driver for the fusion-diagnosis benchmark (Item A) and the §0
# verification/resolution runs. Every run goes through run_cell.sh, which reads
# the prompt from a fixture file at request time. Results are appended as JSONL
# under benchmarks/results/<phase>.jsonl (truncated at phase start).
#
# usage: run_matrix.sh <verify|resolve|itemA>
set -u
PHASE=${1:-}
DIR=/Users/cwong/ai/qwen38-mtp-server/benchmarks
cd /Users/cwong/ai/qwen38-mtp-server || exit 1
mkdir -p benchmarks/results
OUT=benchmarks/results/$PHASE.jsonl
: > "$OUT"

run_cell() {
  local TAG=$1 ENVSPEC=$2 PROMPT=$3
  echo "[matrix] cell $TAG env=[$ENVSPEC] prompt=$PROMPT" >&2
  "$DIR/run_cell.sh" "$TAG" "$ENVSPEC" 18099 "$PROMPT" >> "$OUT" 2>"$OUT".cellerr
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[matrix] cell $TAG failed rc=$rc — stopping" >&2
    exit $rc
  fi
  rm -f "$OUT".cellerr
}

ESSAY=benchmarks/prompts/essay-1024.txt
SPECDEC=benchmarks/prompts/specdec-800.txt

OFF="MLX_QWEN_FUSED_QKV=0 MLX_QWEN_FUSED_SWIGLU=0"
QKV1="MLX_QWEN_FUSED_QKV=1 MLX_QWEN_FUSED_SWIGLU=0"
SW1="MLX_QWEN_FUSED_QKV=0 MLX_QWEN_FUSED_SWIGLU=1"
BOTH="MLX_QWEN_FUSED_QKV=1 MLX_QWEN_FUSED_SWIGLU=1"

case "$PHASE" in
  verify)
    # §0 provenance check on both pinned fixtures, all-fusion-off env
    # (fusion is bit-exact, so the hashes hold under any fusion setting).
    run_cell verify-essay-off "$OFF" "$ESSAY"
    run_cell verify-specdec-off "$OFF" "$SPECDEC"
    ;;
  resolve)
    run_cell resolve-specdec-off "$OFF" "$SPECDEC"
    ;;
  itemA)
    # 6 reps of the interleaved cell order A0,A3,A1,A2; first rep per cell is warmup
    for rep in 1 2 3 4 5 6; do
      run_cell A0-r$rep "$OFF" "$ESSAY"
      run_cell A3-r$rep "$BOTH" "$ESSAY"
      run_cell A1-r$rep "$QKV1" "$ESSAY"
      run_cell A2-r$rep "$SW1" "$ESSAY"
    done
    ;;
  *)
    echo "unknown phase: $PHASE (want verify|resolve|itemA)" >&2
    exit 2
    ;;
esac
echo "[matrix] phase $PHASE done -> $OUT" >&2
