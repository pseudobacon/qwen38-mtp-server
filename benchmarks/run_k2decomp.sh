#!/bin/bash
# K2 decomposition — Phase 2 algebraic measurement cells + Phase 4 A/B set.
#
# One driver run, one release binary, no parallel builds/tests, thermal
# snapshot before every cell, cells interleaved per rep so every config
# sees the same thermal trajectory (differences are taken within a rep).
#
# algebra: 4 configs x 2 fixtures x 6 reps = 48 cells, interleaved per rep:
#   s   --spec-draft-n-max 0                  (serial control, verify M=1)
#   k1  --spec-draft-n-max 3 QWEN_MTP_DRAFT_K=1   (verify M=2)
#   k2  --spec-draft-n-max 3 QWEN_MTP_DRAFT_K=2   (verify M=3)
#   k3  --spec-draft-n-max 3 QWEN_MTP_DRAFT_K=3   (verify M=4)
#   Gates: head q4, finish length, committed 1024, phaseSumOK, draft_depth
#   prefix; k2/k3 cells must reproduce the registered hashes
#   (949b9423... essay / 139acb9d... specdec); k1/s cells have no
#   registered hash — 6-rep in-session determinism is checked in summary.
#   QWEN_MTP_STEP_TRACE=1 is always on (run_cell.sh); MLX_QWEN_MTP_TRACE is
#   OFF for ranked algebra cells (the A/B phase proves it is zero-artifact
#   before the Phase 3 trace run turns it on).
#
# ab: k2-essay, 3 reps with MLX_QWEN_MTP_TRACE off vs 3 reps with it on
#   (trace file path set) — the zero-artifact A/B for the 5-way split +
#   mtp-anchor instrumentation used by the Phase 3 trace run.
#
# summary: prints the within-rep algebra table from the JSONL records.
#
# usage: run_k2decomp.sh <algebra|ab|summary>
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
DIR=$SERVER/benchmarks
cd "$SERVER" || exit 1
mkdir -p benchmarks/results .tmp

RUNLOG=.tmp/k2decomp-run.log
THERMALLOG=.tmp/k2decomp-thermal.log
ALG_OUT=benchmarks/results/k2decomp.jsonl
AB_OUT=benchmarks/results/k2decomp-ab.jsonl
: > "$RUNLOG"
: > "$THERMALLOG"

ESSAY=benchmarks/prompts/essay-1024.txt
SPECDEC=benchmarks/prompts/specdec-800.txt
ESSAY_HASH=949b9423bd851233c71abf4a701e1e8e50f7dfee8663818b886f1d065de7f0fe
SPECDEC_HASH=139acb9d30fee4749c873aaa42142481d888f53537a729e630c68ca8dcf49cac

log() {
  echo "[$(date '+%F %T')] $*" >> "$RUNLOG"
  echo "[k2decomp] $*" >&2
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
  log "thermal snapshot $n before $tag"
}

cell_gate() {
  local tag=$1 out=$2 expected_k=$3 expected_hash=$4
  /tmp/benchvenv/bin/python - "$tag" "$out" "$expected_k" "$expected_hash" <<'PYEOF'
import json, sys

tag = sys.argv[1]
out = sys.argv[2]
expected_k = sys.argv[3]
expected_hash = sys.argv[4]

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
    "finish_reason": (rec.get("finish_reason"), "length"),
    "completion_tokens": (rec.get("completion_tokens"), 1024),
    "phaseSumOK": (rec.get("phaseSumOK"), True),
}
bad = [(k, v, e) for k, (v, e) in checks.items() if v != e]
dd = rec.get("draft_depth", "")
if not dd.startswith("k=" + expected_k):
    bad.append(("draft_depth", dd, "k=" + expected_k + " prefix"))
if expected_hash and rec.get("stream_hash") != expected_hash:
    bad.append(("stream_hash", str(rec.get("stream_hash"))[:16] + "...",
                expected_hash[:16] + "..."))
if bad:
    print(f"GATE FAIL {tag}: "
          + "; ".join(f"{k}={v!r} expected {e!r}" for k, v, e in bad), file=sys.stderr)
    print(json.dumps(rec))
    sys.exit(1)
print(f"GATE PASS {tag}: draft_depth={dd} phaseSumDeltaMs={rec.get('phaseSumDeltaMs')} "
      f"hash={'checked' if expected_hash else 'recorded'}")
PYEOF
}

run_cell() {
  local TAG=$1 ENVSPEC=$2 EXTRA=$3 PROMPT=$4 OUT=$5 EXP_K=$6 EXP_HASH=$7
  thermal_snapshot "$TAG"
  log "cell $TAG env=[$ENVSPEC] extra=[$EXTRA]"
  "$DIR/run_cell.sh" "$TAG" "$ENVSPEC" 18099 "$PROMPT" "$EXTRA" q4 >> "$OUT" 2>"$OUT".cellerr
  local rc=$?
  rm -f "$OUT".cellerr
  if [ $rc -ne 0 ]; then
    log "cell $TAG failed rc=$rc — STOP"
    exit $rc
  fi
  if ! cell_gate "$TAG" "$OUT" "$EXP_K" "$EXP_HASH" >> "$RUNLOG" 2>&1; then
    log "GATE FAIL on $TAG — STOP; partial results kept as evidence"
    exit 3
  fi
  log "cell $TAG ok"
}

# One config group: env spec, extra serve args, expected k prefix, expected
# hash (empty = no external hash gate; in-session determinism checked later).
cfg_run() {
  local CFG=$1 ENVSPEC=$2 EXTRA=$3 EXP_K=$4
  local FIX OUT_HASH
  for FIX in essay specdec; do
    case $FIX in
      essay)   OUT_HASH=$ESSAY_HASH ;;
      specdec) OUT_HASH=$SPECDEC_HASH ;;
    esac
    # k1 and s cells have no registered hash: pass empty.
    if [ "$CFG" = "s" ] || [ "$CFG" = "k1" ]; then OUT_HASH=""; fi
    if [ "$FIX" = "essay" ]; then
      run_cell "${CFG}-essay"   "$ENVSPEC" "$EXTRA" "$ESSAY"   "$ALG_OUT" "$EXP_K" "$OUT_HASH"
    else
      run_cell "${CFG}-specdec" "$ENVSPEC" "$EXTRA" "$SPECDEC" "$ALG_OUT" "$EXP_K" "$OUT_HASH"
    fi
  done
}

PHASE=${1:-}
case "$PHASE" in
  algebra)
    : > "$ALG_OUT"
    log "algebra start: 4 configs x 2 fixtures x 6 reps interleaved per rep"
    for rep in 1 2 3 4 5 6; do
      cfg_run s   ""                    "--spec-draft-n-max 0" 0
      cfg_run k1  "QWEN_MTP_DRAFT_K=1"  "--spec-draft-n-max 3" 1
      cfg_run k2  "QWEN_MTP_DRAFT_K=2"  "--spec-draft-n-max 3" 2
      cfg_run k3  "QWEN_MTP_DRAFT_K=3"  "--spec-draft-n-max 3" 3
    done
    log "algebra complete: 48 cells -> $ALG_OUT"
    echo "[k2decomp] algebra done -> $ALG_OUT" >&2
    ;;
  ab)
    : > "$AB_OUT"
    log "ab start: k2-essay, 4 alternating (on, off) pairs — thermal drift
      cancels across adjacent cells; artifact = mean(off-on) - drift"
    for rep in 1 2 3 4; do
      run_cell "ab-on-r$rep"  \
        "QWEN_MTP_DRAFT_K=2 MLX_QWEN_MTP_TRACE=1 MLX_QWEN_MTP_TRACE_PATH=/tmp/k2decomp-ab-on-$rep.log" \
        "--spec-draft-n-max 3" "$ESSAY" "$AB_OUT" 2 "$ESSAY_HASH"
      run_cell "ab-off-r$rep" "QWEN_MTP_DRAFT_K=2" "--spec-draft-n-max 3" \
        "$ESSAY" "$AB_OUT" 2 "$ESSAY_HASH"
    done
    log "ab complete: 8 cells -> $AB_OUT"
    echo "[k2decomp] ab done -> $AB_OUT" >&2
    ;;
  summary)
    /tmp/benchvenv/bin/python - "$ALG_OUT" "$AB_OUT" <<'PYEOF'
import json, sys

alg_out, ab_out = sys.argv[1], sys.argv[2]

recs = []
for ln in open(alg_out):
    if ln.strip():
        recs.append(json.loads(ln))

# Group by (config, fixture) -> list of reps in run order.
groups = {}
for rec in recs:
    tag = rec["tag"]
    cfg, rest = tag.split("-", 1)
    fixture = rest.rsplit("-r", 1)[0]
    groups.setdefault((cfg, fixture), []).append(rec)

cfgs = ["s", "k1", "k2", "k3"]
fixs = ["essay", "specdec"]

print("=== K2 decomposition — Phase 2 within-rep algebra (engine stepAvg, ms) ===")
for fix in fixs:
    print(f"--- fixture {fix} ---")
    print(f"{'rep':>4} {'serial':>9} {'k1':>9} {'k2':>9} {'k3':>9} "
          f"{'k1-s':>7} {'k2-k1':>7} {'k3-k2':>7} "
          f"{'k2 tG/tE/tH/tC':>22}")
    deltas1, deltas2, deltas3 = [], [], []
    for rep in range(1, 7):
        row = []
        for cfg in cfgs:
            cell = groups[(cfg, fix)][rep - 1]
            row.append(cell["stepAvg"])
        d1, d2, d3 = row[1] - row[0], row[2] - row[1], row[3] - row[2]
        deltas1.append(d1); deltas2.append(d2); deltas3.append(d3)
        k2 = groups[("k2", fix)][rep - 1]
        phases = (f"{k2['tGraphBuildAvg']:.1f}/{k2['tEvalAvg']:.1f}/"
                  f"{k2['tHostReadAvg']:.1f}/{k2['tCacheStateAvg']:.1f}")
        print(f"{rep:>4} {row[0]:>9.1f} {row[1]:>9.1f} {row[2]:>9.1f} {row[3]:>9.1f} "
              f"{d1:>7.1f} {d2:>7.1f} {d3:>7.1f} {phases:>22}")
    n = len(deltas1)
    def m(x): return sum(x) / len(x)
    def sd(x):
        mu = m(x)
        return (sum((v - mu) ** 2 for v in x) / (len(x) - 1)) ** 0.5
    means = [m([r["stepAvg"] for r in groups[(cfg, fix)]]) for cfg in cfgs]
    print(f"mean  {means[0]:>9.1f} {means[1]:>9.1f} {means[2]:>9.1f} {means[3]:>9.1f}")
    print(f"delta k1-s: mean {m(deltas1):.1f} sd {sd(deltas1):.1f}  "
          f"min {min(deltas1):.1f} max {max(deltas1):.1f}")
    print(f"delta k2-k1: mean {m(deltas2):.1f} sd {sd(deltas2):.1f}  "
          f"min {min(deltas2):.1f} max {max(deltas2):.1f}")
    print(f"delta k3-k2: mean {m(deltas3):.1f} sd {sd(deltas3):.1f}  "
          f"min {min(deltas3):.1f} max {max(deltas3):.1f}")

print("=== in-session determinism (6 reps per config+fixture) ===")
bad = 0
for key in sorted(groups):
    hashes = [r.get("stream_hash") for r in groups[key]]
    ok = all(h == hashes[0] for h in hashes)
    if not ok:
        bad += 1
    if ok:
        print(f"{key[0]}-{key[1]}: OK {hashes[0][:16]}...")
    else:
        print(f"{key[0]}-{key[1]}: MISMATCH {[h[:8] for h in hashes]}")
print(f"determinism: {bad} mismatching groups")

print("=== in-session cross-checks (k1 vs serial, same fixture) ===")
for fix in fixs:
    h_k1 = groups[("k1", fix)][0]["stream_hash"]
    h_s = groups[("s", fix)][0]["stream_hash"]
    print(f"{fix}: k1 {'==' if h_k1 == h_s else '!='} serial "
          f"({h_k1[:8]}... vs {h_s[:8]}...)")

print("=== per-config phase means (ms) ===")
for fix in fixs:
    for cfg in cfgs:
        rs = groups[(cfg, fix)]
        n = len(rs)
        ph = {k: sum(r[k] for r in rs) / n for k in
              ("tGraphBuildAvg", "tEvalAvg", "tHostReadAvg", "tCacheStateAvg")}
        print(f"{fix} {cfg}: tG {ph['tGraphBuildAvg']:.1f} tE {ph['tEvalAvg']:.1f} "
              f"tH {ph['tHostReadAvg']:.1f} tC {ph['tCacheStateAvg']:.1f}")

try:
    ab = [json.loads(ln) for ln in open(ab_out) if ln.strip()]
    if ab:
        print("=== Phase 4 A/B: MLX_QWEN_MTP_TRACE on/off alternating (k2-essay) ===")
        # Sequence in run order: on1 off1 on2 off2 ...  Pair delta (off-on)
        # carries the artifact minus one step of thermal drift; the drift is
        # estimated from the overall sequence slope.
        seq = sorted(ab, key=lambda r: int(r["tag"].rsplit("r", 1)[1]))
        vals = [(r["tag"], r["stepAvg"]) for r in seq]
        for tag, v in vals:
            print(f"  {tag:<10} {v:8.1f}")
        pairs = []
        for i in range(0, len(vals) - 1, 2):
            if vals[i][0].startswith("ab-on") and vals[i + 1][0].startswith("ab-off"):
                pairs.append(vals[i + 1][1] - vals[i][1])
        xv = [v for _, v in vals]
        n = len(xv)
        drift = sum((xv[i + 1] - xv[i]) for i in range(n - 1)) / (n - 1)
        pair_mean = m(pairs) if pairs else float("nan")
        artifact = pair_mean - drift if pairs else float("nan")
        print(f"pair deltas (off-on): {' '.join(f'{v:+.1f}' for v in pairs)}  mean {pair_mean:+.2f}")
        print(f"sequence drift estimate: {drift:+.2f} ms/cell")
        print(f"artifact estimate (pair mean - drift): {artifact:+.2f} ms/round")
except FileNotFoundError:
    pass
PYEOF
    ;;
  *)
    echo "unknown phase: $PHASE (want algebra|ab|summary)" >&2
    exit 2
    ;;
esac
exit 0