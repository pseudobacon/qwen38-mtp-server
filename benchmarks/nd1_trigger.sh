#!/bin/bash
# ND1 — characterize the incumbent (v0.31.6) specdec cold/warm trigger.
#
# Fresh incumbent server, specdec-800, greedy, N identical requests in ONE
# process. Per request record: stream hash, prefix-reuse hit/miss (from the
# /metrics prefix_reuse_hits counter delta — proven, not assumed), and the
# MTP-STEP-SUMMARY acceptance/rounds (engagement). Expected: r1 = 139a (miss),
# r2+ = 0688 (hit) if the trigger is the store-on-success prefix-cache hit.
#
# usage: nd1_trigger.sh <out-dir> <n-requests> <port> [cache-buster]
#   cache-buster: if "1", prefix each request with a unique leading token so
#     every request is a MISS (control 1).
set -u
OUTDIR=$1
N=${2:-10}
PORT=${3:-18097}
BUST=${4:-0}

SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1
RELEASE=$SERVER/.build/release
BINDIR=/tmp/mer2/incumbent
PROMPT="$SERVER/benchmarks/prompts/specdec-800.txt"
SRVLOG=$OUTDIR/nd1-server.stderr
SRVOUT=$OUTDIR/nd1-server.stdout
PY=/tmp/benchvenv/bin/python

# swap in the incumbent binary + colocated metallib
cp -f "$BINDIR/qwen38-mtp-server" "$RELEASE/qwen38-mtp-server"
cp -f "$BINDIR/mlx.metallib" "$RELEASE/mlx.metallib"
codesign --force --sign - "$RELEASE/qwen38-mtp-server" 2>/dev/null || true

pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
sleep 3

QWEN_MTP_STEP_TRACE=1 "$RELEASE/qwen38-mtp-server" serve --port "$PORT" --model ./weights \
  > "$SRVOUT" 2> "$SRVLOG" &
SRV=$!

CODE=000
for i in $(seq 1 150); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz" 2>/dev/null)
  if [ "$CODE" = "200" ]; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "SERVER DIED"; exit 1; fi
  sleep 2
done
[ "$CODE" = "200" ] || { echo "readyz never 200 (last=$CODE)"; kill $SRV 2>/dev/null; exit 1; }

# request body (specdec, greedy, 1024 tokens). Cache-buster prefixes a unique
# comment token per request so every request is a fresh prefill (a MISS).
REQ=$OUTDIR/nd1-req.json
RESP=$OUTDIR/nd1-resp.json
BASE_PROMPT=$(cat "$PROMPT")

hits_before() {
  curl -s "http://127.0.0.1:$PORT/metrics" | "$PY" -c 'import sys,json; d=json.load(sys.stdin); print(int(d.get("prefix_reuse_hits",0)))'
}

TABLE=$OUTDIR/nd1-trigger-table.tsv
echo -e "idx\tstream_hash\thit\tdelta\trounds\tacceptedPerStep\tdecodeS" > "$TABLE"

for i in $(seq 1 $N); do
  if [ "$BUST" = "1" ]; then
    # unique leading token per request → guaranteed MISS
    PROMPT="<!-- nd1-bust-$i -->$BASE_PROMPT"
  else
    PROMPT="$BASE_PROMPT"
  fi
  "$PY" -c 'import json,sys; open(sys.argv[1],"w").write(json.dumps({"model":"qwen3.8-27b-mtp","messages":[{"role":"user","content":sys.argv[2]}],"max_tokens":1024,"temperature":0,"top_k":1,"mtp_enabled":True,"enable_thinking":False,"stream":False}))' \
    "$REQ" "$PROMPT"
  hb=$(hits_before)
  curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$REQ" -o "$RESP"
  ha=$(hits_before)
  delta=$((ha - hb))
  hash=$( "$PY" - "$RESP" <<'PYEOF'
import json, sys
from tokenizers import Tokenizer
import hashlib
resp = json.load(open(sys.argv[1]))
content = resp["choices"][0]["message"]["content"]
tok = Tokenizer.from_file("/Users/cwong/ai/qwen38-mtp-server/weights/tokenizer.json")
ids = tok.encode(content).ids
print(hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest())
PYEOF
)
  hit=miss
  [ "$delta" -ge 1 ] && hit=HIT
  echo -e "$i\t${hash:0:12}\t$hit\t$delta\t-\t-\t-" >> "$TABLE"
done

kill $SRV 2>/dev/null
wait $SRV 2>/dev/null

# fill in the per-request MTP-STEP-SUMMARY (one per request, in order)
"$PY" - "$SRVLOG" "$TABLE" <<'PYEOF'
import sys, re
srvlog, table = sys.argv[1], sys.argv[2]
sums = []
for line in open(srvlog, errors="replace"):
    if "MTP-STEP-SUMMARY" in line:
        m = {}
        for kv in line.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                m[k] = v
        sums.append((m.get("rounds"), m.get("acceptedPerStep"), m.get("decodeSeconds")))
rows = [ln.rstrip("\n").split("\t") for ln in open(table)]
header = rows[0]
for j, r in enumerate(rows[1:]):
    if j < len(sums):
        r[4], r[5], r[6] = sums[j][0], sums[j][1], sums[j][2]
    rows[j+1] = r
with open(table, "w") as f:
    f.write("\n".join("\t".join(r) for r in rows) + "\n")
PYEOF

echo "=== ND1 trigger table ($OUTDIR/nd1-trigger-table.tsv) ==="
cat "$TABLE"
echo ""
echo "=== server head selection / fusion (stdout) ==="
grep -E "MTP head selected|fusion prepare|draft depth" "$SRVOUT" | head
exit 0
