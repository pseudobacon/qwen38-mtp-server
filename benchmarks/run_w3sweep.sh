#!/bin/bash
# W3 — MTP draft-depth sweep: k ∈ {1,2,3,4,6,8} (k=3 is the current value:
# adaptive cost-model policy, offered depth = server default 3 — no env,
# no flag), essay-1024, 6 reps per cell (rep 1 warmup, 5 measured),
# interleaved with rotating start cell, fresh server per cell (run_cell.sh),
# port 18099, greedy (temperature 0, enable_thinking false, max_tokens 1024,
# finish_reason length), QWEN_MTP_STEP_TRACE=1, pmset -g therm before every
# rep, no parallel work, no mid-sweep rebuild. Binary SHA must be identical
# across all cells (single release binary; env/CLI-only differences).
#
# Cell configuration (verified against Qwen38MTPBlockSession.draftPolicy
# and ServerConfig 2026-09-14):
#   k1: QWEN_MTP_DRAFT_K=1                     depth = min(offered 3, 1) = 1
#   k2: QWEN_MTP_DRAFT_K=2                     depth = min(offered 3, 2) = 2
#   k3: (no env, no flag)                      adaptive, offered 3, cap 3  <-- current value
#   k4: --spec-draft-n-max 8, QWEN_MTP_DRAFT_K=4   depth = min(8, 4) = 4
#   k6: --spec-draft-n-max 8, QWEN_MTP_DRAFT_K=6   depth = min(8, 6) = 6
#   k8: --spec-draft-n-max 8, QWEN_MTP_DRAFT_K=8   depth = min(8, 8) = 8
#
# Determinism gate (STRONG, per rep): the committed 1024-token stream must be
# the fixture's hash (essay 949b9423…, specdec 139acb9d…) in EVERY cell and
# rep — accepted/proposed/rounds WILL vary with k and are logged, not gated.
# Also gated: finish_reason=length, completion_tokens=1024, phaseSumOK
# (phase-sum validation), qmvVerifyMaterialized=0. Any failure stops the
# sweep; partial results are kept as evidence, never reported as performance.
#
# Results: one JSON line per rep, per-cell files
#   benchmarks/results/w3-draftk-essay-k<K>.jsonl
#   benchmarks/results/w3-draftk-specdec-k<K>.jsonl
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
DIR=$SERVER/benchmarks
cd "$SERVER" || exit 1
mkdir -p benchmarks/results .tmp

RUNLOG=.tmp/w3-run-${W3_MODE:-essay}.log
THERMALLOG=.tmp/w3-thermal-${W3_MODE:-essay}.log
: > "$RUNLOG"
: > "$THERMALLOG"

MODE=${W3_MODE:-essay}   # essay | specdec
REPS=${W3_REPS:-6}

if [ "$MODE" = "essay" ]; then
  PROMPT=benchmarks/prompts/essay-1024.txt
  FIX_HASH=949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe
else
  PROMPT=benchmarks/prompts/specdec-800.txt
  FIX_HASH=139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac
fi

# k -> envspec extraargs
env_of() {
  case $1 in
    1) echo "QWEN_MTP_DRAFT_K=1" ;;
    2) echo "QWEN_MTP_DRAFT_K=2" ;;
    3) echo "" ;;
    4) echo "QWEN_MTP_DRAFT_K=4" ;;
    6) echo "QWEN_MTP_DRAFT_K=6" ;;
    8) echo "QWEN_MTP_DRAFT_K=8" ;;
  esac
}
extra_of() {
  case $1 in
    4|6|8) echo "--spec-draft-n-max 8" ;;
    *) echo "" ;;
  esac
}

# W3 2026-09-14: the original set {1,2,3,4,6,8} was cut to {1,2,3,4} after
# the depth-5/6 determinism gate stopped the matrix (verify width 6 commits
# a wrong first-round token; see the W3 section in progress.md and the
# benchmarks/results/w3-probe-*.jsonl records). 6/8 stay invalid until the
# engine bug is fixed.
CELLS=${W3_CELLS:-"1 2 3 4"}

log() {
  echo "[$(date '+%F %T')] $*" >> "$RUNLOG"
  echo "[w3] $*" >&2
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

# W3 per-rep gate: committed stream must be bit-exact; acceptance/rounds are
# FREE (they vary with k). Phase-sum + zero materializations still mandatory.
gate_check() {
  local tag=$1 out=$2 expected_hash=$3
  /tmp/benchvenv/bin/python - "$tag" "$out" "$expected_hash" <<'PYEOF'
import json, sys

tag = sys.argv[1]
out = sys.argv[2]
expected_hash = sys.argv[3]

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
ps_ok = rec.get("phaseSumOK")
ps_delta = rec.get("phaseSumDeltaMs")
if ps_ok is not True:
    print(f"PHASE-SUM GATE FAIL {tag}: phaseSumOK={ps_ok} "
          f"phaseSumDeltaMs={ps_delta} (cell voided)", file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
print(f"GATE PASS {tag}: {rec.get('accepted')}/{rec.get('proposed')}/"
      f"{rec.get('rounds')} hash {expected_hash[:8]}... "
      f"depthDist={rec.get('depthDist')} phaseSumDeltaMs={ps_delta}")
PYEOF
}

run_cell() {
  local k=$1 rep=$2
  local tag="w3k${k}-r${rep}"
  local envspec extra out
  envspec=$(env_of $k)
  extra=$(extra_of $k)
  out="benchmarks/results/w3-draftk-${MODE}-k${k}.jsonl"
  thermal_snapshot "$tag"
  log "cell $tag env=[$envspec] extra=[$extra]"
  "$DIR/run_cell.sh" "$tag" "$envspec" 18099 "$PROMPT" "$extra" >> "$out" 2>"$out".cellerr
  local rc=$?
  rm -f "$out".cellerr
  if [ $rc -ne 0 ]; then
    log "cell $tag failed rc=$rc — STOP"
    exit $rc
  fi
  if ! gate_check "$tag" "$out" "$FIX_HASH" >> "$RUNLOG" 2>&1; then
    log "DETERMINISM GATE FAIL on $tag — STOP; partial results kept as evidence"
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
    k=$(echo $CELLS | cut -d' ' -f$((idx + 1)))
    run_cell "$k" "$rep"
  done
done

log "w3 sweep ($MODE) complete"
log "binary/provenance check:"
for k in $CELLS; do
  f="benchmarks/results/w3-draftk-${MODE}-k${k}.jsonl"
  sha=$(grep -o '"binary_sha256": "[a-f0-9]*"' "$f" 2>/dev/null | sort -u | head -1)
  log "  k=$k $sha"
done
echo "[w3] done ($MODE)" >&2
exit 0