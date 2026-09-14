#!/bin/bash
# W4 — MTP draft-head quantization A/B: BF16 pinned head vs 4-bit group-64
# `q4` head (MLX_QWEN_MTP_HEAD_QUANT=1), default draft policy (k=3 adaptive,
# no QWEN_MTP_DRAFT_K, no --spec-draft-n-max), 6 reps per cell (rep 1 warmup,
# 5 measured), interleaved with rotating start cell, fresh server per cell
# (run_cell.sh), port 18099, greedy (temperature 0, enable_thinking false,
# max_tokens 1024, finish_reason length), QWEN_MTP_STEP_TRACE=1, pmset -g
# therm before every rep, no parallel work, no mid-matrix rebuild. Binary
# SHA must be identical across all cells (single release binary; env-only
# difference).
#
# Determinism gate (STRONG, per rep): the committed 1024-token stream must be
# the fixture's hash (essay 949b9423…, specdec 139acb9d…) in EVERY cell and
# rep — the 4-bit head may change acceptance/rounds but NEVER the committed
# stream (target-verify guarantee). Also gated: finish_reason=length,
# completion_tokens=1024, phaseSumOK, qmvVerifyMaterialized=0, and
# head_selected matching the cell's intended state (engagement proof). Any
# failure stops the matrix; partial results are kept as evidence, never
# reported as performance.
#
# Results: one JSON line per rep, per-cell files
#   benchmarks/results/w4-ab-essay-{bf16,q4}.jsonl
#   benchmarks/results/w4-ab-specdec-{bf16,q4}.jsonl
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
DIR=$SERVER/benchmarks
cd "$SERVER" || exit 1
mkdir -p benchmarks/results .tmp

RUNLOG=.tmp/w4-run-${W4_MODE:-essay}.log
THERMALLOG=.tmp/w4-thermal-${W4_MODE:-essay}.log
: > "$RUNLOG"
: > "$THERMALLOG"

MODE=${W4_MODE:-essay}   # essay | specdec
REPS=${W4_REPS:-6}

if [ "$MODE" = "essay" ]; then
  PROMPT=benchmarks/prompts/essay-1024.txt
  FIX_HASH=949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe
else
  PROMPT=benchmarks/prompts/specdec-800.txt
  FIX_HASH=139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac
fi

CELLS="bf16 q4"

env_of() {
  case $1 in
    bf16) echo "" ;;
    q4) echo "MLX_QWEN_MTP_HEAD_QUANT=1" ;;
  esac
}

# Which substring of the load-time log line proves the intended head loaded.
head_expect() {
  case $1 in
    bf16) echo "BF16 pinned" ;;
    q4) echo "4-bit quantized" ;;
  esac
}

log() {
  echo "[$(date '+%F %T')] $*" >> "$RUNLOG"
  echo "[w4] $*" >&2
}

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
  log "thermal snapshot $n before $tag:"
  pmset -g therm 2>&1 | while IFS= read -r l; do log "  therm: $l"; done
}

# W4 per-rep gate: committed stream bit-exact, intended head actually loaded,
# phase-sum valid, zero materializations. Acceptance/rounds are FREE.
gate_check() {
  local tag=$1 out=$2 expected_hash=$3 expect_head=$4
  /tmp/benchvenv/bin/python - "$tag" "$out" "$expected_hash" "$expect_head" <<'PYEOF'
import json, sys

tag = sys.argv[1]
out = sys.argv[2]
expected_hash = sys.argv[3]
expect_head = sys.argv[4]

lines = [ln for ln in open(out) if ln.strip()]
if not lines:
    print(f"GATE FAIL {tag}: no record in {out}", file=sys.stderr)
    sys.exit(1)
rec = json.loads(lines[-1])
mat = rec.get("qmvVerifyMaterialized")
if mat not in (0, 0.0):
    print(f"GATE FAIL {tag}: qmvVerifyMaterialized={mat}, expected 0", file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
checks = {
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
head = rec.get("head_selected", "")
if expect_head not in head:
    print(f"ENGAGEMENT GATE FAIL {tag}: head_selected={head!r}, expected substring "
          f"{expect_head!r} — the intended head did not load", file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
ps_ok = rec.get("phaseSumOK")
ps_delta = rec.get("phaseSumDeltaMs")
if ps_ok is not True:
    print(f"PHASE-SUM GATE FAIL {tag}: phaseSumOK={ps_ok} "
          f"phaseSumDeltaMs={ps_delta} (cell voided)", file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
print(f"GATE PASS {tag}: {rec.get('accepted')}/{rec.get('proposed')}/"
      f"{rec.get('rounds')} hash {expected_hash[:8]}... head={head} "
      f"fusion={rec.get('fusion_summary')} phaseSumDeltaMs={ps_delta}")
PYEOF
}

run_cell() {
  local cell=$1 rep=$2
  local tag="w4${cell}-r${rep}"
  local envspec expect out
  envspec=$(env_of $cell)
  expect=$(head_expect $cell)
  out="benchmarks/results/w4-ab-${MODE}-${cell}.jsonl"
  thermal_snapshot "$tag"
  log "cell $tag env=[$envspec]"
  # EXPECT_HEAD=$cell double-gates at the cell level (run_cell fails loudly
  # before the request if the wrong head loaded); gate_check re-checks.
  "$DIR/run_cell.sh" "$tag" "$envspec" 18099 "$PROMPT" "" "$cell" >> "$out" 2>"$out".cellerr
  local rc=$?
  rm -f "$out".cellerr
  if [ $rc -ne 0 ]; then
    log "cell $tag failed rc=$rc — STOP"
    exit $rc
  fi
  if ! gate_check "$tag" "$out" "$FIX_HASH" "$expect" >> "$RUNLOG" 2>&1; then
    log "GATE FAIL on $tag — STOP; partial results kept as evidence"
    exit 3
  fi
  log "cell $tag ok"
}

# Interleaved reps with rotating start cell so thermal drift spreads evenly.
NC=$(echo $CELLS | wc -w | tr -d ' ')
for rep in $(seq 1 $REPS); do
  start=$(( (rep - 1) % NC ))
  for i in $(seq 0 $((NC - 1))); do
    idx=$(( (start + i) % NC ))
    cell=$(echo $CELLS | cut -d' ' -f$((idx + 1)))
    run_cell "$cell" "$rep"
  done
done

log "w4 A/B ($MODE) complete"
log "binary/provenance check:"
for cell in $CELLS; do
  f="benchmarks/results/w4-ab-${MODE}-${cell}.jsonl"
  sha=$(grep -o '"binary_sha256": "[a-f0-9]*"' "$f" 2>/dev/null | sort -u | head -1)
  log "  $cell $sha"
done
echo "[w4] done ($MODE)" >&2
exit 0
