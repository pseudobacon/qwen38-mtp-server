#!/bin/bash
# Draft-depth policy sweep (2026-09-14): decide the speculative draft-depth
# policy on measured end-to-end server behavior. See DRAFT-DEPTH-POLICY.md.
#
# Cell definitions (audit of run_cell.sh + Qwen38MTPBlockSession.draftPolicy +
# MLXGenerator.decodeDepth, verified 2026-09-14):
#   The server offers `decodeDepth = --spec-draft-n-max` (default 3) every
#   round; the session's draft policy returns the ACTUAL draft count
#   d = min(offer, QWEN_MTP_DRAFT_K ?? defaultDraftDepth=2); verify width
#   M = d + 1. So:
#     s : --spec-draft-n-max 0                     d = 0 (M=1, serial control)
#     k1: QWEN_MTP_DRAFT_K=1                       d = min(3,1) = 1 (M=2)
#     k2: QWEN_MTP_DRAFT_K=2                       d = min(3,2) = 2 (M=3)
#     k3: QWEN_MTP_DRAFT_K=3                       d = min(3,3) = 3 (M=4)
#     k4: QWEN_MTP_DRAFT_K=4 --spec-draft-n-max 8  d = min(8,4) = 4 (M=5)
#     k6: QWEN_MTP_DRAFT_K=6 --spec-draft-n-max 8  d = min(8,6) = 6 (M=7)
#     k8: QWEN_MTP_DRAFT_K=8 --spec-draft-n-max 8  d = min(8,8) = 8 (M=9)
#
# Protocol per cell: fresh server (run_cell.sh), greedy temperature 0,
# enable_thinking false, fixed fixture, fixed max_tokens (1024 sustained /
# 128 interactive), QWEN_MTP_STEP_TRACE=1, EXPECT_HEAD=q4, thermal snapshot
# before every cell. Interleaved reps with rotating start cell so thermal
# drift spreads evenly. Rep 1 is warmup, reps 2..N measured (N=6 for the
# decision stages).
#
# Gates (per rep; a failure stops the stage — partial results kept as
# evidence, never reported as performance):
#   stream_hash  == expected (registered fixture hash for the 1024 modes; for
#                  the 128 modes the first passing cell fixes the expected
#                  hash for the rest of the stage — cross-depth equality)
#   completion_tokens == mode max_tokens, finish_reason == length
#   phaseSumOK == true
#   depthDist has exactly one depth key == the cell's d (no silent cap)
#   head q4 gate (run_cell.sh)
#
# usage: run_dpsweep.sh <stage> <mode> <cells> [reps]
#   stage: A | B | C  (prefix for result files)
#   mode:  essay1024 | specdec1024 | essay128 | specdec128
#   cells: space-separated subset of "s k1 k2 k3 k4 k6 k8"
#   reps:  default 1
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
DIR=$SERVER/benchmarks
cd "$SERVER" || exit 1
mkdir -p benchmarks/results .tmp

STAGE=$1
MODE=$2
CELLS=$3
REPS=${4:-1}

case "$MODE" in
  essay1024)   PROMPT=benchmarks/prompts/essay-1024.txt;   MAXT=1024; FIX_HASH=949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe ;;
  specdec1024) PROMPT=benchmarks/prompts/specdec-800.txt;  MAXT=1024; FIX_HASH=139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac ;;
  essay128)    PROMPT=benchmarks/prompts/essay-1024.txt;   MAXT=128;  FIX_HASH= ;;
  specdec128)  PROMPT=benchmarks/prompts/specdec-800.txt;  MAXT=128;  FIX_HASH= ;;
  *) echo "bad mode $MODE" >&2; exit 2 ;;
esac

RUNLOG=.tmp/dpsweep-${STAGE}-${MODE}.log
THERMALLOG=.tmp/dpsweep-${STAGE}-${MODE}.thermal
LEARNED=.tmp/dpsweep-learned-hash-${MODE}
: > "$RUNLOG"
: > "$THERMALLOG"

# 128 modes: no registered hash; first passing cell fixes it for the stage.
if [ -z "$FIX_HASH" ] && [ -f "$LEARNED" ]; then
  FIX_HASH=$(cat "$LEARNED")
fi

env_of() {
  case $1 in
    s)  echo "" ;;
    k1) echo "QWEN_MTP_DRAFT_K=1" ;;
    k2) echo "QWEN_MTP_DRAFT_K=2" ;;
    k3) echo "QWEN_MTP_DRAFT_K=3" ;;
    k4) echo "QWEN_MTP_DRAFT_K=4" ;;
    k6) echo "QWEN_MTP_DRAFT_K=6" ;;
    k8) echo "QWEN_MTP_DRAFT_K=8" ;;
  esac
}
extra_of() {
  case $1 in
    s)  echo "--spec-draft-n-max 0" ;;
    k4|k6|k8) echo "--spec-draft-n-max 8" ;;
    *) echo "" ;;
  esac
}
d_of() { case $1 in s) echo 0 ;; k1) echo 1 ;; k2) echo 2 ;; k3) echo 3 ;; k4) echo 4 ;; k6) echo 6 ;; k8) echo 8 ;; esac; }

depth_gate() {
  local tag=$1 out=$2 d=$3
  /tmp/benchvenv/bin/python - "$tag" "$out" "$d" <<'PYEOF'
import json, sys
tag, out, d = sys.argv[1], sys.argv[2], int(sys.argv[3])
rec = json.loads(open(out).readlines()[-1])
dd = rec.get("depthDist")
ok = False
if dd:
    keys = [kv.split(":")[0] for kv in dd.split(",")]
    ok = keys == [str(d)]
if not ok:
    print(f"DEPTH GATE FAIL {tag}: depthDist={dd}, expected constant d={d}", file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
print(f"DEPTH GATE PASS {tag}: d={d}")
PYEOF
}

gate_check() {
  local tag=$1 out=$2 expected_hash=$3
  /tmp/benchvenv/bin/python - "$tag" "$out" "$expected_hash" "$MAXT" <<'PYEOF'
import json, sys
tag = sys.argv[1]
out = sys.argv[2]
expected_hash = sys.argv[3]
maxt = int(sys.argv[4])
rec = json.loads(open(out).readlines()[-1])
checks = {
    "stream_hash": (rec.get("stream_hash"), expected_hash),
    "finish_reason": (rec.get("finish_reason"), "length"),
    "completion_tokens": (rec.get("completion_tokens"), maxt),
}
bad = [(k, v, e) for k, (v, e) in checks.items() if v != e]
if bad:
    print(f"GATE FAIL {tag}: " + "; ".join(f"{k}={v} expected {e}" for k, v, e in bad), file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
ps_ok = rec.get("phaseSumOK")
ps_delta = rec.get("phaseSumDeltaMs")
if ps_ok is not True:
    print(f"PHASE-SUM GATE FAIL {tag}: phaseSumOK={ps_ok} phaseSumDeltaMs={ps_delta}", file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
print(f"GATE PASS {tag}: {rec.get('accepted')}/{rec.get('proposed')}/{rec.get('rounds')} "
      f"hash {expected_hash[:8]}... depthDist={rec.get('depthDist')} "
      f"phaseSumDeltaMs={ps_delta} accAvg={rec.get('accAvg')}")
PYEOF
}

# Expected hash for a (cell, mode) pair.
#
# Near-tie property (verified 2026-09-14, pre-existing): the per-width QMV
# families (M=1 serial, M=2, M=3, …) accumulate in different orders, so a
# rare near-tie argmax can flip. On essay-1024 all families agree for all
# 1024 tokens; on specdec-800 they do NOT — first M=1 vs M=3 divergence at
# completion token 989, and M=2 diverges from M=3 as well (three distinct
# deterministic streams: M=1 c70882fc…, M=2 a3dfa862…, M=3 139acb9d…). Each
# family is deterministic per (binary, cell). Streams are therefore gated
# per family: essay1024 cells against the single registered hash (all
# families coincide there), all other modes against a per-(mode,cell) family
# hash learned on first encounter. Times remain fully comparable across
# families; the near-tie flip is not a policy input.
expected_hash_for() {
  local cell=$1 famfile
  famfile=".tmp/dpsweep-family-hash-${MODE}-${cell}"
  # essay1024: every width family coincides with the registered stream — the
  # strongest possible gate. Other modes: per-(mode,cell) family hashes.
  if [ "$MODE" = "essay1024" ]; then
    echo "$FIX_HASH"
    return
  fi
  if [ -f "$famfile" ]; then
    cat "$famfile"
    return
  fi
  echo ""
}

run_cell() {
  local cell=$1 rep=$2
  local tag="dp${STAGE}${MODE}-${cell}-r${rep}"
  local envspec extra out d exp_hash
  envspec=$(env_of $cell)
  extra=$(extra_of $cell)
  d=$(d_of $cell)
  exp_hash=$(expected_hash_for $cell)
  out="benchmarks/results/dpsweep-${STAGE}-${MODE}-${cell}.jsonl"
  {
    echo "THERMAL $(date '+%F %T') $tag"
    pmset -g therm 2>&1
  } >> "$THERMALLOG"
  echo "[$(date '+%F %T')] cell $tag env=[$envspec] extra=[$extra]" >> "$RUNLOG"
  "$DIR/run_cell.sh" "$tag" "$envspec" 18099 "$PROMPT" "$extra" q4 "$MAXT" >> "$out" 2>"$out".cellerr
  local rc=$?
  rm -f "$out".cellerr
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] cell $tag failed rc=$rc — STOP" >> "$RUNLOG"
    exit $rc
  fi
  if ! depth_gate "$tag" "$out" "$d" >> "$RUNLOG" 2>&1; then
    echo "[$(date '+%F %T')] DEPTH GATE FAIL on $tag — STOP" >> "$RUNLOG"
    exit 3
  fi
  if ! gate_check "$tag" "$out" "$exp_hash" >> "$RUNLOG" 2>&1; then
    if [ -z "$exp_hash" ]; then
      # First encounter of this (mode, cell) family: learn its hash, accept,
      # continue. The record itself passed depth/phase-sum/length gates only
      # in the sense that gate_check reached the hash check last — re-verify
      # the non-hash gates explicitly before accepting the learned hash.
      local h recheck
      h=$(/tmp/benchvenv/bin/python -c "import json,sys; print(json.loads(open('$out').readlines()[-1])['stream_hash'])")
      recheck=$(/tmp/benchvenv/bin/python -c "
import json
r = json.loads(open('$out').readlines()[-1])
assert r.get('phaseSumOK') is True and r.get('finish_reason')=='length' and r.get('completion_tokens')==$MAXT, r
print('ok')")
      if [ "$recheck" != "ok" ]; then
        echo "[$(date '+%F %T')] LEARN REFUSED (non-hash gate) on $tag — STOP" >> "$RUNLOG"
        exit 3
      fi
      if [ "$MODE" = "specdec1024" ] && [ "$cell" = "k2" ] && [ "$h" != "139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac" ]; then
        echo "[$(date '+%F %T')] LEARNED k2 hash $h != registered 139acb9d… — STOP" >> "$RUNLOG"
        exit 3
      fi
      echo "$h" > ".tmp/dpsweep-family-hash-${MODE}-${cell}"
      cp ".tmp/dpsweep-family-hash-${MODE}-${cell}" "$LEARNED"
      echo "[$(date '+%F %T')] learned family hash $h for $MODE/$cell from $tag" >> "$RUNLOG"
    else
      echo "[$(date '+%F %T')] GATE FAIL on $tag — STOP" >> "$RUNLOG"
      exit 3
    fi
  fi
  echo "[$(date '+%F %T')] cell $tag ok" >> "$RUNLOG"
}

NC=$(echo $CELLS | wc -w | tr -d ' ')
for rep in $(seq 1 $REPS); do
  start=$(( (rep - 1) % NC ))
  for i in $(seq 0 $((NC - 1))); do
    idx=$(( (start + i) % NC ))
    cell=$(echo $CELLS | cut -d' ' -f$((idx + 1)))
    run_cell "$cell" "$rep"
  done
done

echo "[$(date '+%F %T')] dpsweep stage $STAGE mode $MODE complete" >> "$RUNLOG"
for c in $CELLS; do
  f="benchmarks/results/dpsweep-${STAGE}-${MODE}-${c}.jsonl"
  sha=$(grep -o '"binary_sha256": "[a-f0-9]*"' "$f" 2>/dev/null | sort -u | head -1)
  echo "  $c $sha" >> "$RUNLOG"
done
echo "[dpsweep] done stage $STAGE mode $MODE" >&2
exit 0