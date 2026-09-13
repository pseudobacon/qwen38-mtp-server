#!/bin/bash
# Item D — A/B: route the MTP verify pass through the candidate QMV kernel.
#
# Same release binary in both cells; the only difference is the env var:
#   D0  essay-1024.txt    default env (knob OFF)
#                          pre-Item D behavior: 3-D verify always falls back
#                          to the incumbent; M=1 decode routes through the
#                          candidate kernel
#   D1  essay-1024.txt    MLX_QWEN_QMV_VERIFY=1 (knob ON)
#                          3-D verify with B*L in 2..9 reshapes to 2-D and
#                          routes through the candidate kernel; M=1 (decode
#                          and B*L==1) falls back to the incumbent
#
# Fusion defaults stay ON in both cells (MLX_QWEN_FUSED_QKV / MLX_QWEN_FUSED_
# SWIGLU unset) — the A/B isolates the verify-path QMV routing only.
#
# Protocol (pinned, no deviations): release binary, port 18099, greedy
# (temperature 0, enable_thinking false, max_tokens 1024, finish_reason length),
# prompt read from the pinned fixture file, QWEN_MTP_STEP_TRACE=1, fresh server
# per cell (run_cell.sh: pkill + /readyz poll), 6 reps per cell with rep 1
# discarded as warmup (5 measured), interleaved with alternating start cell
# (D0->D1, D1->D0, ...) so thermal drift hits both cells evenly, no parallel
# builds/tests during timing, no rebuild of the binary mid-matrix.
#
# Thermal discipline: `pmset -g therm` is captured before every rep, appended
# to .tmp/itemd-run.log (human-readable) and .tmp/itemd-thermal.log
# (machine-readable THERMAL marker blocks).
#
# Determinism gate per rep: the essay cell must reproduce 599/1086/426 (stream
# hash 949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe) in
# BOTH cells — a hash mismatch in either cell stops the matrix immediately;
# partial results are kept as evidence and must never be reported as
# performance numbers. The binary_sha256 must be identical across all D0/D1
# records (same binary, env-var-only difference).
#
# Engagement gate (post-run): D0 records must show zero routed 3-D widths in
# qmvVerifyRouted (the knob never engaged); D1 records must show a routed
# width histogram over 2..9 proportional to the observed verify depths — a
# near-zero routed share in D1 invalidates the cell as a test of the routing.
# qmvVerifyMaterialized must be read: a non-trivial copy count is a result
# (copies may be eating the kernel win), not noise.
#
# Results: one JSON line per rep, appended per cell to
#   benchmarks/results/itemd-D0-essay.jsonl
#   benchmarks/results/itemd-D1-essay.jsonl
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
DIR=$SERVER/benchmarks
cd "$SERVER" || exit 1
mkdir -p benchmarks/results .tmp

RUNLOG=.tmp/itemd-run.log
THERMALLOG=.tmp/itemd-thermal.log
: > "$RUNLOG"
: > "$THERMALLOG"

OUT_D0=benchmarks/results/itemd-D0-essay.jsonl
OUT_D1=benchmarks/results/itemd-D1-essay.jsonl
: > "$OUT_D0"; : > "$OUT_D1"

ESSAY=benchmarks/prompts/essay-1024.txt

D0_ENV=""
D1_ENV="MLX_QWEN_QMV_VERIFY=1"

ESSAY_HASH=949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe

log() {
  echo "[$(date '+%F %T')] $*" >> "$RUNLOG"
  echo "[itemd] $*" >&2
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

log "item D matrix start: D0/D1 x 6 reps, interleaved with alternating start cell"
for rep in 1 2 3 4 5 6; do
  if [ $((rep % 2)) -eq 1 ]; then
    run_cell "D0-r$rep" "$D0_ENV" "$ESSAY" "$OUT_D0" "$ESSAY_HASH" 599 1086 426
    run_cell "D1-r$rep" "$D1_ENV" "$ESSAY" "$OUT_D1" "$ESSAY_HASH" 599 1086 426
  else
    run_cell "D1-r$rep" "$D1_ENV" "$ESSAY" "$OUT_D1" "$ESSAY_HASH" 599 1086 426
    run_cell "D0-r$rep" "$D0_ENV" "$ESSAY" "$OUT_D0" "$ESSAY_HASH" 599 1086 426
  fi
done

log "item D matrix complete"
echo "[itemd] done -> $OUT_D0 $OUT_D1" >&2
exit 0