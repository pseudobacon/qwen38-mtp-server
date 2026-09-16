#!/bin/bash
# Phase 2 (LCP) — re-profile at 32K/64K with the toggle-gated kernel
# extensions, plus 8K/16K bit-exact hash gates with both toggles on.
#
# Pinned protocol: same as run_lcp_p1.sh (release binary, greedy, pc=512,
# MLX_TRACE_PREFILL=1, 1 Hz RSS, content-hash gate against the pinned
# pc=512 baselines). Cells:
#   8K/16K: both toggles on (end-to-end bit-exactness at short lengths)
#   32K/64K: each toggle alone + both (per-kernel uplift attribution)
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1
BIN=$SERVER/.build/release/qwen38-mtp-server
PORT=18099
FIXDIR=$SERVER/benchmarks/results/prefill-verify-2026-09-15
OUTDIR=$SERVER/benchmarks/results/prefill-opt-20260915
mkdir -p "$OUTDIR"
RUNLOG=$OUTDIR/run-p2.log
: > "$RUNLOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$RUNLOG"; }

baseline_for() {
  case "$1" in
    8k)  echo 660dd1208737764c ;;
    16k) echo 2e583ad29dc28465 ;;
    32k) echo 97bc0d74846a043b ;;
    64k) echo 14b26f9f89f16da5 ;;
    *)   echo "" ;;
  esac
}

run_cell() {
  local tag=$1 len=$2 envspec=$3
  local req="$FIXDIR/req${len}.json"
  local srvlog="$OUTDIR/srv-${tag}.log"
  local resp="$OUTDIR/resp-${tag}.json"
  local rss="$OUTDIR/rss-${tag}.csv"

  log "cell $tag: env: $envspec"
  pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
  sleep 2

  env MLX_CHUNKED_PREFILL=1 QWEN_PREFILL_CHUNK_SIZE=512 MLX_TRACE_PREFILL=1 $envspec \
    "$BIN" serve --port "$PORT" --model ./weights > /dev/null 2> "$srvlog" &
  local SRV=$!

  local code=000
  for i in $(seq 1 300); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz" 2>/dev/null)
    if [ "$code" = "200" ]; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then
      log "cell $tag: SERVER DIED during load — STOP"
      exit 1
    fi
    sleep 2
  done
  if [ "$code" != "200" ]; then
    log "cell $tag: readyz never 200 (last=$code) — STOP"
    kill $SRV 2>/dev/null
    exit 1
  fi
  log "cell $tag: ready"

  echo "t,rss_kb" > "$rss"
  curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @"$req" -o "$resp" -w '%{time_total}' > "$OUTDIR/wall-${tag}.txt" &
  local CURL=$!
  while kill -0 $CURL 2>/dev/null; do
    echo "$(date +%s),$(ps -o rss= -p $SRV 2>/dev/null | tr -d ' ')" >> "$rss"
    sleep 1
  done
  wait $CURL
  local rc=$?
  local wall
  wall=$(cat "$OUTDIR/wall-${tag}.txt" 2>/dev/null | tr -d ' ')

  kill $SRV 2>/dev/null
  wait $SRV 2>/dev/null

  if [ $rc -ne 0 ]; then
    log "cell $tag: curl failed rc=$rc — STOP"
    exit 1
  fi

  /tmp/benchvenv/bin/python - "$resp" "$rss" "$srvlog" "$OUTDIR/summary-${tag}.json" <<'PYEOF'
import json, sys, hashlib

resp_path, rss_path, srvlog, out_path = sys.argv[1:5]
out = {}
resp = json.load(open(resp_path))
choice = resp["choices"][0]
out["finish_reason"] = choice.get("finish_reason")
out["usage"] = resp.get("usage")
content = choice["message"]["content"]
out["content_hash"] = hashlib.sha256(content.encode()).hexdigest()
peak = 0
for line in open(rss_path):
    line = line.strip()
    if line and line != "t,rss_kb":
        try:
            peak = max(peak, int(line.split(",")[1]))
        except ValueError:
            pass
out["peak_rss_gb"] = peak / (1024 * 1024)
pf2 = None
for line in open(srvlog, errors="replace"):
    if "PF2 prefill" in line:
        pf2 = line.strip()
out["pf2"] = pf2
json.dump(out, open(out_path, "w"), indent=2)
print(json.dumps(out))
PYEOF
  local pyrc=$?
  if [ $pyrc -ne 0 ]; then
    log "cell $tag: summary parse failed — STOP"
    exit 1
  fi

  local hash expected
  hash=$(/tmp/benchvenv/bin/python -c "import json,sys; print(json.load(open('$OUTDIR/summary-${tag}.json'))['content_hash'][:16])")
  expected=$(baseline_for "$len")
  if [ "$hash" != "$expected" ]; then
    log "BIT-EXACT GATE FAIL $tag: hash $hash != baseline $expected — STOP"
    exit 3
  fi
  log "cell $tag: wall=${wall}s hash=$hash == baseline PASS"
}

log "LCP Phase 2 matrix start (8 cells)"
run_cell "p2-8k-both"  8k  "MLX_QWEN_FUSED_RESIDUAL_3D=1 MLX_QWEN_FUSED_GDN_PREFILL=1"
run_cell "p2-16k-both" 16k "MLX_QWEN_FUSED_RESIDUAL_3D=1 MLX_QWEN_FUSED_GDN_PREFILL=1"
run_cell "p2-32k-res3d" 32k "MLX_QWEN_FUSED_RESIDUAL_3D=1"
run_cell "p2-32k-gdn"  32k "MLX_QWEN_FUSED_GDN_PREFILL=1"
run_cell "p2-32k-both" 32k "MLX_QWEN_FUSED_RESIDUAL_3D=1 MLX_QWEN_FUSED_GDN_PREFILL=1"
run_cell "p2-64k-res3d" 64k "MLX_QWEN_FUSED_RESIDUAL_3D=1"
run_cell "p2-64k-gdn"  64k "MLX_QWEN_FUSED_GDN_PREFILL=1"
run_cell "p2-64k-both" 64k "MLX_QWEN_FUSED_RESIDUAL_3D=1 MLX_QWEN_FUSED_GDN_PREFILL=1"
log "LCP Phase 2 matrix complete"
exit 0
