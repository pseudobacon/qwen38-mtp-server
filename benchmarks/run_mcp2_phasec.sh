#!/bin/bash
# MCP2 Phase C — generalization of the winning pc (2048) past 32K.
#
# The 32K gate passed (pc=2048: −12.7%, 5/5 paired, bit-exact). Phase C confirms
# the improvement generalizes: 64K not regressed vs pc=512 (paired), and a short
# fixture (essay) still works at pc=2048 (pc > prompt len => single dense chunk).
#
# Pinned: same release binary, port 18099, MLX_CHUNKED_PREFILL=1, greedy, fresh
# server per cell, rep 1 discarded. 64K: 3 reps x {512,2048}, rotating. essay:
# 2 reps at pc=2048.
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1
BIN=$SERVER/.build/release/qwen38-mtp-server
PORT=18099
FIXDIR=$SERVER/benchmarks/results/prefill-verify-2026-09-15
OUTDIR=$SERVER/benchmarks/results/mcp-20260917
mkdir -p "$OUTDIR"
RUNLOG=$OUTDIR/mcp2-phasec.log
REPS=$OUTDIR/mcp2-phasec.jsonl
PY=/tmp/benchvenv/bin/python
BIN_SHA=$(shasum -a 256 "$BIN" | cut -d' ' -f1)

# build the short essay request (same template as req8k)
ESSAY_REQ=$OUTDIR/req-essay.json
[ -f "$ESSAY_REQ" ] || $PY - "$FIXDIR/req8k.json" "$SERVER/benchmarks/prompts/essay-1024.txt" "$ESSAY_REQ" <<'PYEOF'
import json, sys
tmpl, essay, outp = sys.argv[1:4]
d = json.load(open(tmpl))
d["messages"] = [{"role": "user", "content": open(essay).read()}]
json.dump(d, open(outp, "w"))
PYEOF
FIX_ESSAY_SHA=$(shasum -a 256 "$ESSAY_REQ" | cut -d' ' -f1)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNLOG"; }
die() { log "FATAL: $* — STOP"; exit 1; }

launch_server() {
  local pc=$1 tag=$2
  pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
  sleep 8
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
  kill $SRV 2>/dev/null; echo ""; return 1
}

run_req() {
  local pc=$1 SRV=$2 tag=$3 reqfile=$4
  local resp="$OUTDIR/resp-${tag}.json"
  local rss="$OUTDIR/rss-${tag}.csv"
  echo "t,rss_kb" > "$rss"
  curl -s --max-time 1200 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$reqfile" \
    -o "$resp" -w '%{time_total}' > "$OUTDIR/wall-${tag}.txt" &
  local CURL=$!
  while kill -0 $CURL 2>/dev/null; do
    echo "$(date +%s),$(ps -o rss= -p $SRV 2>/dev/null | tr -d ' ')" >> "$rss"
    sleep 1
  done
  wait $CURL
  cat "$OUTDIR/wall-${tag}.txt" 2>/dev/null | tr -d ' '
}

parse_cell() {
  local pc=$1 rep=$2 tag=$3 L=$4 binsha=$5 reqsha=$6
  local srvlog="$OUTDIR/srv-${tag}.log"
  local resp="$OUTDIR/resp-${tag}.json"
  local rss="$OUTDIR/rss-${tag}.csv"
  local wall=$(cat "$OUTDIR/wall-${tag}.txt" 2>/dev/null | tr -d ' ')
  local thermal=$(pmset -g therm 2>/dev/null | tr '\n' ' ')
  $PY - "$pc" "$rep" "$tag" "$L" "$wall" "$thermal" "$resp" "$rss" "$srvlog" "$binsha" "$reqsha" <<'PYEOF'
import json, sys, re, hashlib
pc, rep, tag, L, wall, thermal = sys.argv[1:7]
resp, rss, srvlog, binsha, reqsha = sys.argv[7:12]
intL = int(L[:-1]) * 1000 if L.endswith("k") else 0
out = {"rep": int(rep), "pc": int(pc), "len": L, "bin_sha": binsha, "req_sha": reqsha,
       "wall_s": float(wall) if wall else None, "thermal": thermal,
       "buffer_gb": round(24 * int(pc) * intL * 2 / 1e9, 4)}
try:
    r = json.load(open(resp))
    c = r["choices"][0]
    out["finish_reason"] = c.get("finish_reason")
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
    for k in ("total_ms","ffn_ms","gdn_ms","attn_ms","sdpa_ms","norm_ms","residual_ms"):
        if k in kv: out[k] = float(kv[k])
    prim = ["ffn_ms","gdn_ms","attn_ms","norm_ms","residual_ms"]
    out["phase_sum_ms"] = round(sum(out.get(p,0) for p in prim), 2)
    out["phase_sum_check"] = out.get("total_ms") is not None and \
        abs(out["phase_sum_ms"] - out["total_ms"]) <= max(1.0, 0.02*out["total_ms"])
json.dump(out, sys.stdout)
PYEOF
}

log "Phase C start. binary=$BIN_SHA essay_req=$FIX_ESSAY_SHA"
: > "$REPS"

# ---- 64K: 3 reps x {512,2048}, rotating ----
for rep in 1 2 3; do
  for pc in 512 2048; do
    tag="pc64-rep${rep}-pc${pc}"
    log "64k rep=$rep pc=$pc: launching"
    SRV=$(launch_server "$pc" "$tag"); [ -z "$SRV" ] && die "server died (64k rep=$rep pc=$pc)"
    wall=$(run_req "$pc" "$SRV" "$tag" "$FIXDIR/req64k.json")
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    [ -z "$wall" ] && die "no wall (64k rep=$rep pc=$pc)"
    line=$(parse_cell "$pc" "$rep" "$tag" "64k" "$BIN_SHA" "req64k")
    echo "$line" | tee -a "$REPS"
  done
done

# ---- essay (short) at pc=2048: 2 reps ----
for rep in 1 2; do
  tag="essay-rep${rep}-pc2048"
  log "essay rep=$rep pc=2048: launching"
  SRV=$(launch_server "2048" "$tag"); [ -z "$SRV" ] && die "server died (essay rep=$rep)"
  wall=$(run_req "2048" "$SRV" "$tag" "$ESSAY_REQ")
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  [ -z "$wall" ] && die "no wall (essay rep=$rep)"
  line=$(parse_cell "2048" "$rep" "$tag" "essay" "$BIN_SHA" "$FIX_ESSAY_SHA")
  echo "$line" | tee -a "$REPS"
done

log "Phase C complete."
log "DONE"
