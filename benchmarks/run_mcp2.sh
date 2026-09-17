#!/bin/bash
# MCP2 — prefill-chunk-size (pc) sweep, end-to-end, config-only.
#
# MCP1 (see mcp1-curve.md) measured the FFN per-token cost vs M and found an
# interior minimum at M=1024 (down_proj −73.7% vs M=512; FFN −42.9%), predicting
# optimal pc=1024. This sweep tests that prediction end-to-end on the real 32K
# prefill: does a larger --prefill-chunk-size cut the wall with zero source change?
#
# Pinned protocol (no deviations):
#   - single release binary (SHA recorded), both trees clean
#   - fresh server per (pc, rep) cell, port 18099, MLX_CHUNKED_PREFILL=1, greedy
#   - cells pc in {512, 1024, 2048} (4096 NOT added: MCP1 predicts the crossing
#     below 2048, at 1024)
#   - Phase A (hash gates, one-time per pc): 8K/16K bit-exact vs incumbent;
#     32K hash recorded (pc=512 must be the incumbent 97bc0d74)
#   - Phase B: 32K, 6 reps, rotating start cell, rep 1 discarded, 5 measured;
#     thermal captured before every rep
#   - per rep: prefill wall (eval-sync), per-phase (PF2), peak RSS, per-tile
#     buffer, content hash, admission
#
# Gate (to flip the default pc): mean 32K wall >= 5% better than pc=512 AND
#   >= 4/5 paired reps favor the new pc; 64K not regressed (Phase C, separate);
#   all determinism gates pass; admission verified.
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1
BIN=$SERVER/.build/release/qwen38-mtp-server
PORT=18099
FIXDIR=$SERVER/benchmarks/results/prefill-verify-2026-09-15
OUTDIR=$SERVER/benchmarks/results/mcp-20260917
mkdir -p "$OUTDIR"
RUNLOG=$OUTDIR/mcp2-run.log
REPS=$OUTDIR/mcp2-reps.jsonl
HASHG=$OUTDIR/mcp2-hashgates.jsonl
PY=/tmp/benchvenv/bin/python
NHEADS=24   # Qwen3.8-27B query heads (per-tile buffer = NHEADS * pc * L * 2)

PCs_arr=(512 1024 2048)
BIN_SHA=$(shasum -a 256 "$BIN" | cut -d' ' -f1)
FIX32_SHA=$(shasum -a 256 "$FIXDIR/req32k.json" | cut -d' ' -f1)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNLOG"; }
die() { log "FATAL: $* — STOP"; exit 1; }

baseline_hash() {
  case "$1" in
    8k)  echo 660dd1208737764c ;;
    16k) echo 2e583ad29dc28465 ;;
    32k) echo 97bc0d74846a043b ;;
    64k) echo 14b26f9f89f16da5 ;;
    *)   echo "" ;;
  esac
}

launch_server() {
  local pc=$1 tag=$2
  pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
  sleep 8   # allow GPU memory to be released before the next 15 GB load
  MLX_CHUNKED_PREFILL=1 QWEN_PREFILL_CHUNK_SIZE=$pc MLX_TRACE_PREFILL=1 \
    "$BIN" serve --port "$PORT" --model ./weights > /dev/null 2> "$OUTDIR/srv-${tag}.log" &
  local SRV=$!
  local code=000
  for i in $(seq 1 240); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz" 2>/dev/null)
    if [ "$code" = "200" ]; then echo "$SRV"; return 0; fi
    if ! kill -0 $SRV 2>/dev/null; then echo ""; return 1; fi
    sleep 2
  done
  kill $SRV 2>/dev/null
  echo ""; return 1
}

run_req() {
  local pc=$1 SRV=$2 tag=$3 len=$4
  local req="$FIXDIR/req${len}.json"
  local resp="$OUTDIR/resp-${tag}.json"
  local rss="$OUTDIR/rss-${tag}.csv"
  echo "t,rss_kb" > "$rss"
  curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$req" \
    -o "$resp" -w '%{time_total}' > "$OUTDIR/wall-${tag}.txt" &
  local CURL=$!
  while kill -0 $CURL 2>/dev/null; do
    echo "$(date +%s),$(ps -o rss= -p $SRV 2>/dev/null | tr -d ' ')" >> "$rss"
    sleep 1
  done
  wait $CURL
  cat "$OUTDIR/wall-${tag}.txt" 2>/dev/null | tr -d ' '
}

# parse_cell pc rep order tag len binsha fix32sha  -> one JSON line on stdout
parse_cell() {
  local pc=$1 rep=$2 order=$3 tag=$4 len=$5 binsha=$6 fix32sha=$7
  local srvlog="$OUTDIR/srv-${tag}.log"
  local resp="$OUTDIR/resp-${tag}.json"
  local rss="$OUTDIR/rss-${tag}.csv"
  local wall=$(cat "$OUTDIR/wall-${tag}.txt" 2>/dev/null | tr -d ' ')
  local thermal=$(pmset -g therm 2>/dev/null | tr '\n' ' ')
  $PY - "$pc" "$rep" "$order" "$tag" "$len" "$wall" "$thermal" "$resp" "$rss" "$srvlog" "$binsha" "$fix32sha" <<'PYEOF'
import json, sys, re, hashlib
pc, rep, order, tag, L, wall, thermal = sys.argv[1:8]
resp, rss, srvlog, binsha, fix32sha = sys.argv[8:13]
intL = int(L[:-1]) * 1000 if L.endswith("k") else int(L)
out = {"rep": int(rep), "pc": int(pc), "order": int(order), "len": L,
       "bin_sha": binsha, "fixture_sha32": fix32sha,
       "wall_s": float(wall) if wall else None, "thermal": thermal,
       "buffer_gb": round(24 * int(pc) * intL * 2 / 1e9, 4)}
try:
    r = json.load(open(resp))
    c = r["choices"][0]
    out["finish_reason"] = c.get("finish_reason")
    out["usage"] = r.get("usage")
    out["content_hash"] = hashlib.sha256(c["message"]["content"].encode()).hexdigest()
except Exception as e:
    out["parse_error"] = str(e)
peak = 0
for line in open(rss):
    line = line.strip()
    if line and line != "t,rss_kb":
        try: peak = max(peak, int(line.split(",")[1]))
        except ValueError: pass
out["peak_rss_gb"] = round(peak / (1024*1024), 3)
pf2 = None
for line in open(srvlog, errors="replace"):
    if "PF2 prefill" in line: pf2 = line.strip()
out["pf2"] = pf2
if pf2:
    kv = dict(re.findall(r"(\w+)=([0-9.\-]+)", pf2))
    for k in ("total_ms","ffn_ms","gdn_ms","attn_ms","sdpa_ms","qkv_ms",
              "oproj_ms","rope_ms","norm_ms","residual_ms"):
        if k in kv: out[k] = float(kv[k])
    prim = ["ffn_ms","gdn_ms","attn_ms","norm_ms","residual_ms"]
    out["phase_sum_ms"] = round(sum(out.get(p,0) for p in prim), 2)
    out["phase_sum_check"] = out.get("total_ms") is not None and \
        abs(out["phase_sum_ms"] - out["total_ms"]) <= max(1.0, 0.02*out["total_ms"])
json.dump(out, sys.stdout)
PYEOF
}

log "MCP2 start. binary=$BIN_SHA fixture32=$FIX32_SHA"
: > "$REPS"; : > "$HASHG"

# ---------- Phase A: hash gates (one-time per pc) ----------
for pc in "${PCs_arr[@]}"; do
  for L in 8k 16k 32k; do
    tag="hashg-pc${pc}-${L}"
    log "hashgate pc=$pc L=$L: launching"
    SRV=$(launch_server "$pc" "$tag"); [ -z "$SRV" ] && die "server died (pc=$pc $L)"
    wall=$(run_req "$pc" "$SRV" "$tag" "$L")
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    line=$(parse_cell "$pc" 0 0 "$tag" "$L" "$BIN_SHA" "$FIX32_SHA")
    echo "$line" >> "$HASHG"
    h=$($PY -c "import json,sys;print(json.loads(sys.argv[1]).get('content_hash','')[:16])" "$line")
    exp=$(baseline_hash "$L")
    if [ "$h" = "$exp" ]; then st=PASS; else st=DIFF; fi
    log "hashgate pc=$pc L=$L: hash=$h expected=$exp -> $st (wall=${wall}s)"
  done
done
log "Phase A done. per-pc 32K hashes in $HASHG"

# ---------- Phase B: 32K, 6 reps, rotating start cell ----------
for rep in 1 2 3 4 5 6; do
  off=$(( (rep-1) % 3 ))
  for i in 0 1 2; do
    pc=${PCs_arr[$(( (i+off) % 3 ))]}
    tag="rep${rep}-pc${pc}"
    log "rep=$rep order=$i pc=$pc: launching"
    SRV=$(launch_server "$pc" "$tag"); [ -z "$SRV" ] && die "server died (rep=$rep pc=$pc)"
    wall=$(run_req "$pc" "$SRV" "$tag" 32k)
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    [ -z "$wall" ] && die "no wall (rep=$rep pc=$pc)"
    line=$(parse_cell "$pc" "$rep" "$i" "$tag" "32k" "$BIN_SHA" "$FIX32_SHA")
    echo "$line" | tee -a "$REPS"
  done
done
log "MCP2 Phase B complete. Reps in $REPS"
log "DONE"
