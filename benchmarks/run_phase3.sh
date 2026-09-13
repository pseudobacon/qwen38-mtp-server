#!/bin/bash
# Phase 3 — dual-fixture re-baseline + compiled-path ablation (one session).
#
# Cells (interleaved B1 -> B2 -> B3 on every rep so thermal drift hits cells
# evenly):
#   B1  essay-1024.txt    default env              re-baseline, comparable to A0
#   B2  specdec-800.txt   default env              first specdec measurement on
#                                                   the current binary
#   B3  essay-1024.txt    MLX_COMPILED_DECODE=0    ablation: compiled activation
#                                                   micro-fusions + QK-RoPE fast
#                                                   path + compiled decode
#                                                   segments OFF (bit-exact eager
#                                                   fallbacks)
#
# Protocol (pinned, no deviations): release binary, port 18099, greedy
# (temperature 0, enable_thinking false, max_tokens 1024, finish_reason length),
# prompt read from the pinned fixture file, QWEN_MTP_STEP_TRACE=1, fresh server
# per cell (run_cell.sh: pkill + /readyz poll), 6 reps per cell with rep 1
# discarded as warmup (5 measured), no parallel builds/tests during timing,
# no rebuild of the binary mid-matrix.
#
# Thermal discipline: `pmset -g therm` is captured before every rep, appended
# to .tmp/phase3-run.log (human-readable) and .tmp/phase3-thermal.log
# (machine-readable THERMAL marker blocks, merged into the per-rep JSONL records
# afterwards).
#
# Determinism gate per rep: essay cells must reproduce 599/1086/426 (stream hash
# 949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe); the specdec
# cell must reproduce 645/1008/380 (stream hash
# 139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac). Any
# mismatch stops the matrix immediately; partial results are kept as evidence
# and must never be reported as performance numbers.
#
# Results: one JSON line per rep, appended per cell to
#   benchmarks/results/rebaseline-essay.jsonl      (B1)
#   benchmarks/results/rebaseline-specdec.jsonl    (B2)
#   benchmarks/results/ablation-compiled-off.jsonl (B3)
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
DIR=$SERVER/benchmarks
cd "$SERVER" || exit 1
mkdir -p benchmarks/results .tmp

RUNLOG=.tmp/phase3-run.log
THERMALLOG=.tmp/phase3-thermal.log
: > "$RUNLOG"
: > "$THERMALLOG"

OUT_B1=benchmarks/results/rebaseline-essay.jsonl
OUT_B2=benchmarks/results/rebaseline-specdec.jsonl
OUT_B3=benchmarks/results/ablation-compiled-off.jsonl
: > "$OUT_B1"; : > "$OUT_B2"; : > "$OUT_B3"

ESSAY=benchmarks/prompts/essay-1024.txt
SPECDEC=benchmarks/prompts/specdec-800.txt

DEFAULT_ENV=""
ABLATION_ENV="MLX_COMPILED_DECODE=0"

ESSAY_HASH=949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe
SPECDEC_HASH=139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac

log() {
  echo "[$(date '+%F %T')] $*" >> "$RUNLOG"
  echo "[phase3] $*" >&2
}

# Capture the thermal state before a rep; append the pmset output to the run
# log and record a machine-readable block in the thermal log.
thermal_snapshot() {
  local tag=$1
  local n
  n=$(grep -c '^THERMAL ' "$THERMALLOG" 2>/dev/null)
  n=${n:-0}
  n=$((n + 1))
  {
    echo "THERMAL $n $tag $(date '+%F %T')"
    pmset -g therm 2>&1
  } >> "$THERMALLOG"
  log "thermal snapshot $n before $tag (pmset -g therm):"
  pmset -g therm 2>&1 | while IFS= read -r l; do log "  therm: $l"; done
}

# Per-rep determinism gate: the last JSONL record of the cell must reproduce
# the fixture's accepted/proposed/rounds, stream hash, and finish reason.
gate_check() {
  local tag=$1 out=$2 expected_hash=$3 acc=$4 prop=$5 rounds=$6
  /tmp/benchvenv/bin/python - "$tag" "$out" "$expected_hash" "$acc" "$prop" "$rounds" <<'PYEOF'
import json, sys

tag = sys.argv[1]
out = sys.argv[2]
expected_hash = sys.argv[3]
acc, prop, rounds = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])

lines = [ln for ln in open(out) if ln.strip()]
if not lines:
    print(f"DETERMINISM GATE FAIL {tag}: no record in {out}", file=sys.stderr)
    sys.exit(1)
rec = json.loads(lines[-1])
checks = {
    "accepted": (rec.get("accepted"), acc),
    "proposed": (rec.get("proposed"), prop),
    "rounds": (rec.get("rounds"), rounds),
    "stream_hash": (rec.get("stream_hash"), expected_hash),
    "finish_reason": (rec.get("finish_reason"), "length"),
    "completion_tokens": (rec.get("completion_tokens"), 1024),
}
bad = [(k, v, e) for k, (v, e) in checks.items() if v != e]
if bad:
    print(f"DETERMINISM GATE FAIL {tag}: "
          + "; ".join(f"{k}={v} expected {e}" for k, v, e in bad), file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
print(f"DETERMINISM GATE PASS {tag}: {acc}/{prop}/{rounds} "
      f"hash {expected_hash[:8]}...")
PYEOF
}

run_cell() {
  local tag=$1 envspec=$2 prompt=$3 out=$4 exp_hash=$5 acc=$6 prop=$7 rounds=$8
  thermal_snapshot "$tag"
  log "cell $tag env=[$envspec] prompt=$prompt"
  "$DIR/run_cell.sh" "$tag" "$envspec" 18099 "$prompt" >> "$out" 2>"$out".cellerr
  local rc=$?
  rm -f "$out".cellerr
  if [ $rc -ne 0 ]; then
    log "cell $tag failed rc=$rc — STOP"
    exit $rc
  fi
  if ! gate_check "$tag" "$out" "$exp_hash" "$acc" "$prop" "$rounds" >> "$RUNLOG" 2>&1; then
    log "DETERMINISM GATE FAIL on $tag — STOP; partial results kept as evidence"
    exit 3
  fi
  log "cell $tag ok"
}

log "phase 3 matrix start: 3 cells x 6 reps, interleaved B1->B2->B3"
for rep in 1 2 3 4 5 6; do
  run_cell "B1-r$rep" "$DEFAULT_ENV"  "$ESSAY"   "$OUT_B1" "$ESSAY_HASH"   599 1086 426
  run_cell "B2-r$rep" "$DEFAULT_ENV"  "$SPECDEC" "$OUT_B2" "$SPECDEC_HASH" 645 1008 380
  run_cell "B3-r$rep" "$ABLATION_ENV" "$ESSAY"   "$OUT_B3" "$ESSAY_HASH"   599 1086 426
done

log "phase 3 matrix complete"
echo "[phase3] done -> $OUT_B1 $OUT_B2 $OUT_B3" >&2
exit 0
