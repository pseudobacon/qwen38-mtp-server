#!/bin/bash
# Post-flip (production default draft depth k = 2) final headline session +
# pre-timing gates. The engine default policy
# (Qwen38MTPBlockSession.draftPolicy) now pins k = 2 when QWEN_MTP_DRAFT_K
# is unset (post-W4 queue decision, Phases 2/3/5: k = 2 best measured on
# both fixtures under the full protocol). This driver verifies the flip and
# takes the final headline on the new default.
#
# Phases:
#   gate     One-shot verify-equivalent cells (run_matrix.sh verify style),
#            all declaring EXPECT_HEAD=q4:
#            - gate-default-essay / gate-default-specdec: default config
#              (no env, no extra args) — must reproduce the registered
#              (k = 2, q4) hashes 949b9423… / 139acb9d…
#            - gate-k3-essay / gate-k3-specdec: rollback knob
#              QWEN_MTP_DRAFT_K=3 — must reproduce the registered k = 3
#              hashes (the same two hashes on both fixtures).
#            The load-time log line "MLXLM: MTP draft depth: k=…" must state
#            the active depth (recorded in the JSONL `draft_depth` field and
#            gate-checked: k=2 default cells, k=3 rollback cells).
#   headline Default config, BOTH fixtures, 6 reps each interleaved per rep
#            (rep 1 per cell warmup, 5 measured), `pmset -g therm` before
#            every rep, single release binary across all cells, no parallel
#            builds/tests, no mid-matrix rebuild. Nothing else runs during
#            these cells.
#
# usage: run_k2default.sh <gate|headline>
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
DIR=$SERVER/benchmarks
cd "$SERVER" || exit 1
mkdir -p benchmarks/results .tmp

RUNLOG=.tmp/k2default-run.log
THERMALLOG=.tmp/k2default-thermal.log
: > "$RUNLOG"
: > "$THERMALLOG"

ESSAY=benchmarks/prompts/essay-1024.txt
SPECDEC=benchmarks/prompts/specdec-800.txt
ESSAY_HASH=949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe
SPECDEC_HASH=139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac
K3="QWEN_MTP_DRAFT_K=3"

log() {
  echo "[$(date '+%F %T')] $*" >> "$RUNLOG"
  echo "[k2default] $*" >&2
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

# Per-cell gate: the last JSONL record must reproduce the expected stream
# hash, commit 1024 tokens with finish_reason=length, pass the phase-sum
# validation and the head gate, and state the expected active draft depth
# (k=2 default cells, k=3 rollback cells).
gate_check() {
  local tag=$1 out=$2 expected_hash=$3 expected_k=$4
  /tmp/benchvenv/bin/python - "$tag" "$out" "$expected_hash" "$expected_k" <<'PYEOF'
import json, sys

tag = sys.argv[1]
out = sys.argv[2]
expected_hash = sys.argv[3]
expected_k = sys.argv[4]

lines = [ln for ln in open(out) if ln.strip()]
if not lines:
    print(f"GATE FAIL {tag}: no record in {out}", file=sys.stderr)
    sys.exit(1)
rec = json.loads(lines[-1])
if "error" in rec:
    print(f"GATE FAIL {tag}: {rec['error']}", file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
checks = {
    "stream_hash": (rec.get("stream_hash"), expected_hash),
    "finish_reason": (rec.get("finish_reason"), "length"),
    "completion_tokens": (rec.get("completion_tokens"), 1024),
    "phaseSumOK": (rec.get("phaseSumOK"), True),
}
bad = [(k, v, e) for k, (v, e) in checks.items() if v != e]
dd = rec.get("draft_depth", "")
if not dd.startswith("k=" + expected_k):
    bad.append(("draft_depth", dd, "k=" + expected_k + " prefix"))
if bad:
    print(f"DETERMINISM GATE FAIL {tag}: "
          + "; ".join(f"{k}={v!r} expected {e!r}" for k, v, e in bad), file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
print(f"GATE PASS {tag}: hash {expected_hash[:8]}… draft_depth={dd} "
      f"phaseSumDeltaMs={rec.get('phaseSumDeltaMs')}")
PYEOF
}

run_cell() {
  local TAG=$1 ENVSPEC=$2 PROMPT=$3 OUT=$4 EXP_HASH=$5 EXP_K=$6
  thermal_snapshot "$TAG"
  log "cell $TAG env=[$ENVSPEC] prompt=$PROMPT expect_head=q4 expect_k=$EXP_K"
  "$DIR/run_cell.sh" "$TAG" "$ENVSPEC" 18099 "$PROMPT" "" q4 >> "$OUT" 2>"$OUT".cellerr
  local rc=$?
  rm -f "$OUT".cellerr
  if [ $rc -ne 0 ]; then
    log "cell $TAG failed rc=$rc — STOP"
    exit $rc
  fi
  if ! gate_check "$TAG" "$OUT" "$EXP_HASH" "$EXP_K" >> "$RUNLOG" 2>&1; then
    log "DETERMINISM GATE FAIL on $TAG — STOP; partial results kept as evidence"
    exit 3
  fi
  log "cell $TAG ok"
}

PHASE=${1:-}
case "$PHASE" in
  gate)
    OUT=benchmarks/results/k2default-gate.jsonl
    : > "$OUT"
    log "gate start: 4 one-shot verify-equivalent cells (default + k3 rollback)"
    run_cell gate-default-essay  ""         "$ESSAY"   "$OUT" "$ESSAY_HASH"   2
    run_cell gate-default-specdec ""         "$SPECDEC" "$OUT" "$SPECDEC_HASH" 2
    run_cell gate-k3-essay        "$K3"      "$ESSAY"   "$OUT" "$ESSAY_HASH"   3
    run_cell gate-k3-specdec      "$K3"      "$SPECDEC" "$OUT" "$SPECDEC_HASH" 3
    log "gate complete"
    echo "[k2default] gate done -> $OUT" >&2
    ;;
  headline)
    OUT=benchmarks/results/k2default.jsonl
    : > "$OUT"
    log "headline start: default config (pinned k=2), both fixtures, 6 reps interleaved"
    for rep in 1 2 3 4 5 6; do
      run_cell d-essay-r$rep   "" "$ESSAY"   "$OUT" "$ESSAY_HASH"   2
      run_cell d-specdec-r$rep "" "$SPECDEC" "$OUT" "$SPECDEC_HASH" 2
    done
    log "headline complete"
    echo "[k2default] headline done -> $OUT" >&2
    ;;
  *)
    echo "unknown phase: $PHASE (want gate|headline)" >&2
    exit 2
    ;;
esac
exit 0
