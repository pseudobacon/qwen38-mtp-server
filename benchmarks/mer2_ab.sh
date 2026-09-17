#!/bin/bash
# MER2 — same-session incumbent (v0.31.6) vs upgraded (v0.32.2) decode A/B.
#
# Two BINARIES (not env-only): incumbent from engine main (0.31.6 pin), upgraded
# from the feature branch (v0.32.2 pin). Each cell swaps its binary+metallib into
# .build/release/ (mer2_run_cell_bin.sh) and runs one greedy 1024-token request
# through run_cell.sh.
#
# Pinned protocol:
#   - fixtures essay-1024 AND specdec-800 (pinned, SHA recorded), greedy,
#     max_tokens 1024, enable_thinking false, EXPECT_HEAD=q4 fail-loud,
#     port 18099, fresh server per cell.
#   - 6 reps per cell, rep 1 discarded as warmup, 5 measured.
#   - interleave inc/upg with ROTATING START within each fixture-block.
#   - pmset -g therm before every rep; no parallel work; no mid-matrix rebuild.
#   - per-rep gates: committed stream hash (registry), phaseSumOK,
#     completion_tokens=1024, finish_reason=length, fusion engagement.
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
DIR=$SERVER/benchmarks
cd "$SERVER" || exit 1
RUNID=${MER2_RUNID:-$(date +%Y%m%d-%H%M)}
OUTDIR=$SERVER/benchmarks/results/mlx-v0322-merge-$RUNID
mkdir -p "$OUTDIR"
RUNLOG=$OUTDIR/run.log
THERMALLOG=$OUTDIR/thermal.log
: > "$RUNLOG"; : > "$THERMALLOG"

MODE=${MER2_MODE:-all}   # essay | specdec | all
REPS=${MER2_REPS:-6}

INC_BIN=/tmp/mer2/incumbent
UPG_BIN=/tmp/mer2/upgraded

prompt_of() {
  case $1 in
    essay) echo "benchmarks/prompts/essay-1024.txt" ;;
    specdec) echo "benchmarks/prompts/specdec-800.txt" ;;
  esac
}
hash_of() {
  case $1 in
    essay) echo "949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe" ;;
    specdec) echo "139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac" ;;
  esac
}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNLOG"; }

thermal_snapshot() {
  local tag=$1 n
  n=$(grep -c '^THERMAL ' "$THERMALLOG" 2>/dev/null); n=${n:-0}; n=$((n + 1))
  { echo "THERMAL $n $tag $(date '+%F %T')"; pmset -g therm 2>&1; } >> "$THERMALLOG"
}

gate_check() {
  local tag=$1 out=$2 expected_hash=$3
  /tmp/benchvenv/bin/python - "$tag" "$out" "$expected_hash" <<'PYEOF'
import json, sys
tag, out, expected_hash = sys.argv[1], sys.argv[2], sys.argv[3]
lines = [ln for ln in open(out) if ln.strip()]
if not lines:
    print(f"GATE FAIL {tag}: no record", file=sys.stderr); sys.exit(1)
rec = json.loads(lines[-1])
if "error" in rec:
    print(f"GATE FAIL {tag}: {rec['error']}", file=sys.stderr); print(json.dumps(rec)); sys.exit(1)
mat = rec.get("qmvVerifyMaterialized")
if mat not in (0, 0.0, None):
    print(f"GATE FAIL {tag}: qmvVerifyMaterialized={mat}", file=sys.stderr); sys.exit(1)
checks = {
    "stream_hash": (rec.get("stream_hash"), expected_hash),
    "finish_reason": (rec.get("finish_reason"), "length"),
    "completion_tokens": (rec.get("completion_tokens"), 1024),
}
bad = [(k, v, e) for k, (v, e) in checks.items() if v != e]
if bad:
    print(f"DETERMINISM GATE FAIL {tag}: " + "; ".join(f"{k}={v} expected {e}" for k, v, e in bad),
          file=sys.stderr)
    print(json.dumps(rec)); sys.exit(1)
if rec.get("phaseSumOK") is not True:
    print(f"PHASE-SUM GATE FAIL {tag}: phaseSumOK={rec.get('phaseSumOK')} "
          f"delta={rec.get('phaseSumDeltaMs')}", file=sys.stderr)
    print(json.dumps(rec)); sys.exit(1)
# engagement: the fused backbone must be active (swiGLU 64, qkv 16, gdn 48)
fus = rec.get("fusion_summary", "")
if "swiGLU 64" not in fus:
    print(f"ENGAGEMENT GATE FAIL {tag}: fusion_summary={fus!r} (expected swiGLU 64)", file=sys.stderr)
    sys.exit(1)
print(f"GATE PASS {tag}: {rec.get('accepted')}/{rec.get('proposed')}/{rec.get('rounds')} "
      f"hash {expected_hash[:8]}... ttlt={rec.get('ttlt')} stepAvg={rec.get('stepAvg')} "
      f"depthAvg={rec.get('depthAvg')} fusion={fus}")
PYEOF
}

run_cell() {
  local cell=$1 rep=$2 mode=$3
  local tag="mer2-${mode}-${cell}-r${rep}"
  local bindir
  case $cell in
    inc) bindir=$INC_BIN ;;
    upg) bindir=$UPG_BIN ;;
  esac
  local out="$OUTDIR/mer2-${mode}-${cell}.jsonl"
  local prompt=$(prompt_of "$mode")
  local fixhash=$(hash_of "$mode")
  thermal_snapshot "$tag"
  log "cell $tag (bin=$cell rep=$rep) start"
  "$DIR/mer2_run_cell_bin.sh" "$cell" "$bindir" "$tag" "" 18099 "$prompt" q4 1024 >> "$out" 2>"$out".cellerr
  local rc=$?
  rm -f "$out".cellerr
  if [ $rc -ne 0 ]; then
    log "cell $tag failed rc=$rc — STOP"; exit $rc
  fi
  if ! gate_check "$tag" "$out" "$fixhash" >> "$RUNLOG" 2>&1; then
    log "GATE FAIL on $tag — STOP; partial results kept as evidence"; exit 3
  fi
  log "cell $tag ok"
}

CELLS="inc upg"
NC=2
# fixture list: MODE=all runs both fixtures; else the single fixture
if [ "$MODE" = "all" ]; then MODES="essay specdec"; else MODES="$MODE"; fi
for m in $MODES; do
  log "=== fixture block: $m ==="
  for rep in $(seq 1 $REPS); do
    start=$(( (rep - 1) % NC ))
    for i in $(seq 0 $((NC - 1))); do
      idx=$(( (start + i) % NC ))
      cell=$(echo $CELLS | cut -d' ' -f$((idx + 1)))
      run_cell "$cell" "$rep" "$m"
    done
  done
done

log "mer2 A/B ($MODE) complete"
# provenance
for m in $MODES; do
  for cell in $CELLS; do
    f="$OUTDIR/mer2-${m}-${cell}.jsonl"
    [ -f "$f" ] || continue
    sha=$(grep -o '"binary_sha256": "[a-f0-9]*"' "$f" 2>/dev/null | sort -u | head -1)
    log "  $m/$cell $sha"
  done
done
echo "[mer2] done ($MODE)" >&2
exit 0
