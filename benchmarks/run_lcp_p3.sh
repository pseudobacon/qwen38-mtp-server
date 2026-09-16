#!/bin/bash
# Phase 3 (LCP) — gated attention-path validation.
#
#   ENABLE_BIT_EXACT_ATTENTION=1 : reference dense attention
#                                  (8K/16K must match the pinned pc=512
#                                   baseline hashes; 32K/64K must be
#                                   rejected 507 — dense buffer infeasible)
#   ENABLE_BIT_EXACT_ATTENTION=0 : chunked/fused attention path
#                                  (pc=512 baseline, hashes must match)
#   ENABLE_BIT_EXACT=1           : strict unoptimized fallback — dense
#                                  attention AND all fusions disabled;
#                                  8K/16K outputs must still match the
#                                  baseline hashes (fusions are bit-exact)
#
# Same protocol as run_lcp_p1.sh (release binary, greedy, pc=512 default,
# MLX_TRACE_PREFILL=1, 1 Hz RSS, content-hash gate).
set -u
SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1
BIN=$SERVER/.build/release/qwen38-mtp-server
PORT=18099
FIXDIR=$SERVER/benchmarks/results/prefill-verify-2026-09-15
OUTDIR=$SERVER/benchmarks/results/prefill-opt-20260915
mkdir -p "$OUTDIR"
RUNLOG=$OUTDIR/run-p3.log
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

# run_cell <tag> <len> <env-spec> <expect-hash|507>
run_cell() {
  local tag=$1 len=$2 envspec=$3 expect=$4
  local req="$FIXDIR/req${len}.json"
  local srvlog="$OUTDIR/srv-${tag}.log"
  local resp="$OUTDIR/resp-${tag}.json"
  local rss="$OUTDIR/rss-${tag}.csv"
  local statusfile="$OUTDIR/status-${tag}.txt"

  log "cell $tag: env: $envspec expect: $expect"
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
  local hcode
  hcode=$(curl -s --max-time 900 -o "$resp" -w '%{http_code}' \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d @"$req")
  echo "$hcode" > "$statusfile"

  kill $SRV 2>/dev/null
  wait $SRV 2>/dev/null

  if [ "$expect" = "507" ]; then
    if [ "$hcode" = "507" ]; then
      log "cell $tag: HTTP 507 (dense infeasible rejection) PASS"
      cp "$resp" "$OUTDIR/summary-${tag}.json" 2>/dev/null
    else
      log "cell $tag: expected 507, got $hcode — STOP"
      exit 3
    fi
    return 0
  fi

  if [ "$hcode" != "200" ]; then
    log "cell $tag: expected 200, got $hcode — STOP"
    exit 3
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

  local hash
  hash=$(/tmp/benchvenv/bin/python -c "import json,sys; print(json.load(open('$OUTDIR/summary-${tag}.json'))['content_hash'][:16])")
  if [ "$hash" != "$expect" ]; then
    log "GATE FAIL $tag: hash $hash != expected $expect — STOP"
    exit 3
  fi
  log "cell $tag: hash=$hash == expected PASS"
}

log "LCP Phase 3 matrix start (8 cells)"
run_cell "p3-8k-dense"  8k  "ENABLE_BIT_EXACT_ATTENTION=1" "660dd1208737764c"
run_cell "p3-16k-dense" 16k "ENABLE_BIT_EXACT_ATTENTION=1" "2e583ad29dc28465"
run_cell "p3-8k-strict"  8k  "ENABLE_BIT_EXACT=1" "660dd1208737764c"
run_cell "p3-16k-strict" 16k "ENABLE_BIT_EXACT=1" "2e583ad29dc28465"
run_cell "p3-32k-dense"  32k "ENABLE_BIT_EXACT_ATTENTION=1" "507"
run_cell "p3-64k-dense"  64k "ENABLE_BIT_EXACT_ATTENTION=1" "507"
run_cell "p3-32k-chunk"  32k "ENABLE_BIT_EXACT_ATTENTION=0" "97bc0d74846a043b"
run_cell "p3-64k-chunk"  64k "ENABLE_BIT_EXACT_ATTENTION=0" "14b26f9f89f16da5"
log "LCP Phase 3 matrix complete"
exit 0
