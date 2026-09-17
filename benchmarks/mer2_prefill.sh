#!/bin/bash
# MER2 prefill cell: ONE 32K prefill per side (MLX_CHUNKED_PREFILL=1, pc=2048,
# MLX_TRACE_PREFILL=1, pinned 32K fixture). Captures prefill wall (eval-sync),
# per-phase (PF2), peak RSS, content hash, admission. The upgrade must not
# regress prefill beyond noise.
#
# usage: mer2_prefill.sh <cell> <bin-dir> <out-dir>
set -u
CELL=$1
BINDIR=$2
OUTDIR=$3
SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1
RELEASE=$SERVER/.build/release
PORT=18099
FIXDIR=$SERVER/benchmarks/results/prefill-verify-2026-09-15
REQ=$FIXDIR/req32k.json
PY=/tmp/benchvenv/bin/python

BIN_SRC="$BINDIR/qwen38-mtp-server"
METAL_SRC="$BINDIR/mlx.metallib"
[ -x "$BIN_SRC" ] || { echo "{\"error\":\"missing binary $BIN_SRC\"}"; exit 1; }
[ -f "$METAL_SRC" ] || { echo "{\"error\":\"missing metallib $METAL_SRC\"}"; exit 1; }
cp -f "$BIN_SRC" "$RELEASE/qwen38-mtp-server"
cp -f "$METAL_SRC" "$RELEASE/mlx.metallib"

TAG="mer2-prefill-${CELL}"
SRVLOG=$OUTDIR/srv-${TAG}.log
RESP=$OUTDIR/resp-${TAG}.json
RSS=$OUTDIR/rss-${TAG}.csv
WALLF=$OUTDIR/wall-${TAG}.txt
SUMF=$OUTDIR/summary-${TAG}.json
BIN_SHA=$(shasum -a 256 "$BIN_SRC" | cut -d' ' -f1)
FIX_SHA=$(shasum -a 256 "$REQ" | cut -d' ' -f1)

echo "[$(date '+%F %T')] prefill cell $TAG start (pc=2048, fix=$FIX_SHA)" | tee -a "$OUTDIR/run.log"
pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
sleep 8

MLX_CHUNKED_PREFILL=1 QWEN_PREFILL_CHUNK_SIZE=2048 MLX_TRACE_PREFILL=1 \
  "$RELEASE/qwen38-mtp-server" serve --port "$PORT" --model ./weights > /dev/null 2> "$SRVLOG" &
SRV=$!

code=000
for i in $(seq 1 300); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz" 2>/dev/null)
  if [ "$code" = "200" ]; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "{\"error\":\"SERVER DIED\"}" | tee -a "$SUMF"; exit 1; fi
  sleep 2
done
if [ "$code" != "200" ]; then echo "{\"error\":\"readyz never 200\"}" | tee -a "$SUMF"; kill $SRV 2>/dev/null; exit 1; fi
echo "[$(date '+%F %T')] prefill cell $TAG ready" | tee -a "$OUTDIR/run.log"

echo "t,rss_kb" > "$RSS"
curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @"$REQ" -o "$RESP" -w '%{time_total}' > "$WALLF" &
CURL=$!
while kill -0 $CURL 2>/dev/null; do
  echo "$(date +%s),$(ps -o rss= -p $SRV 2>/dev/null | tr -d ' ')" >> "$RSS"
  sleep 1
done
wait $CURL
rc=$?
wall=$(cat "$WALLF" 2>/dev/null | tr -d ' ')
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
if [ $rc -ne 0 ]; then echo "{\"error\":\"curl rc=$rc\"}" | tee -a "$SUMF"; exit 1; fi

$PY - "$RESP" "$RSS" "$SRVLOG" "$SUMF" "$CELL" "$BIN_SHA" "$FIX_SHA" "$wall" <<'PYEOF'
import json, sys, hashlib
resp_path, rss_path, srvlog, out_path, cell, bin_sha, fix_sha, wall = sys.argv[1:9]
out = {"cell": cell, "binary_sha256": bin_sha, "fixture_sha256": fix_sha}
try:
    out["wall_seconds"] = float(wall)
except Exception:
    pass
try:
    resp = json.load(open(resp_path))
    choice = resp["choices"][0]
    out["finish_reason"] = choice.get("finish_reason")
    out["usage"] = resp.get("usage")
    content = choice["message"]["content"]
    out["content_hash"] = hashlib.sha256(content.encode()).hexdigest()
except Exception as e:
    out["response_error"] = str(e)
peak = 0
for line in open(rss_path):
    line = line.strip()
    if line and line != "t,rss_kb":
        try: peak = max(peak, int(line.split(",")[1]))
        except ValueError: pass
out["peak_rss_gb"] = peak / (1024 * 1024)
pf2 = None
for line in open(srvlog, errors="replace"):
    if "PF2 prefill" in line:
        pf2 = line.strip()
out["pf2"] = pf2
# admission: capture the admission log line if present
for line in open(srvlog, errors="replace"):
    if "admission" in line.lower() or "KV" in line and "admit" in line.lower():
        out.setdefault("admission", line.strip())
json.dump(out, open(out_path, "w"), indent=2)
print(json.dumps(out))
PYEOF
echo "[$(date '+%F %T')] prefill cell $TAG done wall=${wall}s" | tee -a "$OUTDIR/run.log"
exit 0
