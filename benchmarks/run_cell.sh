#!/bin/bash
# One benchmark cell run: start the release server with the given env, serve one
# greedy 1024-token request whose prompt is read from a fixture file at request
# time, parse the MTP step trace + response, compute the greedy stream hash,
# and emit one JSON line on stdout.
#
# usage: run_cell.sh <tag> <env-spec> <port> <prompt-file>
#   env-spec: space-separated VAR=VAL pairs (only the fusion/layout knobs vary)
set -u
TAG=$1
ENVSPEC=$2
PORT=${3:-18099}
PROMPT_FILE=$4

SERVER=/Users/cwong/ai/qwen38-mtp-server
cd "$SERVER" || exit 1

case "$PROMPT_FILE" in
  /*) : ;;
  *) PROMPT_FILE="$SERVER/$PROMPT_FILE" ;;
esac

FIXTURE_HASH=$(shasum -a 256 "$PROMPT_FILE" | awk '{print $1}')

# --- provenance: the binary actually launched + repo HEADs -----------------
ENGINE=/Users/cwong/ai/mlx-swift-lm
SERVER_BIN=$SERVER/.build/release/qwen38-mtp-server

BIN_SHA256=$(shasum -a 256 "$SERVER_BIN" 2>/dev/null | awk '{print $1}')
BIN_MTIME=$(stat -f '%Sm' -t '%Y-%m-%dT%H:%M:%S' "$SERVER_BIN" 2>/dev/null)
SERVER_HEAD=$(git -C "$SERVER" rev-parse --short HEAD 2>/dev/null)
ENGINE_HEAD=$(git -C "$ENGINE" rev-parse --short HEAD 2>/dev/null)
SERVER_DIRTY=$([ -z "$(git -C "$SERVER" status --porcelain 2>/dev/null)" ] && echo clean || echo dirty)
ENGINE_DIRTY=$([ -z "$(git -C "$ENGINE" status --porcelain 2>/dev/null)" ] && echo clean || echo dirty)

if [ -z "$BIN_SHA256" ]; then
  echo "{\"error\":\"release binary missing at $SERVER_BIN — build before benchmarking\"}"
  exit 1
fi

STDERR_LOG=/tmp/mtp-bench-$TAG.stderr
STDOUT_LOG=/tmp/mtp-bench-$TAG.stdout
RESPONSE=/tmp/mtp-bench-$TAG.response.json
REQUEST=/tmp/mtp-bench-$TAG.request.json

rm -f "$STDERR_LOG" "$STDOUT_LOG" "$RESPONSE" "$REQUEST"

# never run against a stale server holding the port
pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
sleep 1

ENV_PREFIX="QWEN_MTP_STEP_TRACE=1"
for kv in $ENVSPEC; do ENV_PREFIX="$ENV_PREFIX $kv"; done

env $ENV_PREFIX "$SERVER_BIN" serve --port "$PORT" --model ./weights \
  > "$STDOUT_LOG" 2> "$STDERR_LOG" &
SRV=$!

CODE=000
for i in $(seq 1 150); do
  CODE=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz" 2>/dev/null)
  if [ "$CODE" = "200" ]; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo '{"error":"SERVER DIED during load"}'
    exit 1
  fi
  sleep 2
done
if [ "$CODE" != "200" ]; then
  echo "{\"error\":\"readyz never 200 (last=$CODE)\"}"
  kill $SRV 2>/dev/null
  exit 1
fi

# request body: prompt read from the fixture file at request time (no inline copy)
/tmp/benchvenv/bin/python - "$PROMPT_FILE" "$REQUEST" <<'PYEOF'
import json, sys
prompt = open(sys.argv[1]).read()
body = {
    "model": "qwen3.8-27b-mtp",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1024,
    "temperature": 0,
    "top_k": 1,
    "mtp_enabled": True,
    "enable_thinking": False,
    "stream": False,
}
open(sys.argv[2], "w").write(json.dumps(body))
PYEOF

WALL_SECONDS=$(curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @"$REQUEST" -o "$RESPONSE" -w '%{time_total}')

curl_rc=$?
kill $SRV 2>/dev/null
pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null
wait $SRV 2>/dev/null

/tmp/benchvenv/bin/python - "$TAG" "$STDERR_LOG" "$RESPONSE" "$REQUEST" "$PROMPT_FILE" "$FIXTURE_HASH" "$WALL_SECONDS" \
  "$BIN_SHA256" "$BIN_MTIME" "$SERVER_HEAD" "$ENGINE_HEAD" "$SERVER_DIRTY" "$ENGINE_DIRTY" <<'PYEOF'
import json, sys, hashlib

tag, stderr_log, response, request, prompt_file, fixture_hash, wall_seconds = sys.argv[1:8]
bin_sha256, bin_mtime, server_head, engine_head, server_dirty, engine_dirty = sys.argv[8:14]
out = {
    "tag": tag,
    "fixture_hash": fixture_hash,
    "binary_sha256": bin_sha256,
    "binary_mtime": bin_mtime,
    "server_head": server_head,
    "engine_head": engine_head,
    "server_dirty": server_dirty,
    "engine_dirty": engine_dirty,
}

try:
    wall_seconds = float(wall_seconds)
except ValueError:
    wall_seconds = None
out["wall_seconds"] = wall_seconds

prompt_text = open(prompt_file).read()

try:
    req = json.load(open(request))
    out["prompt_chars"] = len(req["messages"][0]["content"])
except Exception:
    pass

# tokenization with the model's HF tokenizer (same tokenizer.json the server uses)
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/cwong/ai/qwen38-mtp-server/weights/tokenizer.json")
try:
    from jinja2 import Template
    tmpl = Template(open("/Users/cwong/ai/qwen38-mtp-server/weights/chat_template.jinja").read())
    rendered = tmpl.render(
        messages=req["messages"],
        enable_thinking=False,
        add_generation_prompt=True,
        tools=None,
    )
    out["prompt_tokens"] = len(tok.encode(rendered).ids)
except Exception:
    pass

summary = None
for line in open(stderr_log, errors="replace"):
    if "MTP-STEP-SUMMARY" in line:
        summary = line
if summary:
    for kv in summary.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v

step_ms, t_eval, t_graph, t_cache, t_read, n = [], [], [], [], [], 0
for line in open(stderr_log, errors="replace"):
    if "STEP-TRACE" not in line:
        continue
    n += 1
    vals = {}
    for kv in line.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            vals[k] = v
    def f(key, lst):
        try:
            lst.append(float(vals.get(key, "nan")))
        except ValueError:
            pass
    f("stepMs", step_ms)
    f("tEvalMs", t_eval)
    f("tGraphBuildMs", t_graph)
    f("tCacheStateMs", t_cache)
    f("tHostReadMs", t_read)
if n:
    out["stepTraceRounds"] = n
    out["stepAvg"] = sum(step_ms) / len(step_ms)
    out["tEvalAvg"] = sum(t_eval) / len(t_eval)
    out["tGraphBuildAvg"] = sum(t_graph) / len(t_graph)
    out["tCacheStateAvg"] = sum(t_cache) / len(t_cache)
    out["tHostReadAvg"] = sum(t_read) / len(t_read)
    # Phase-sum validation (mandatory since the Item D flush incident): the
    # four phase averages must add back to the round wall-time average.
    # The STEP-TRACE line emits stepMs = tail - t0, so a large delta means a
    # phase is missing/misattributed or work leaked outside the four windows
    # (e.g. a hidden eval-forcing call shifting GPU wait between phases).
    # A violation voids the cell.
    phase_sum = (out["tEvalAvg"] + out["tGraphBuildAvg"]
                 + out["tCacheStateAvg"] + out["tHostReadAvg"])
    out["phaseSumAvg"] = phase_sum
    out["phaseSumDeltaMs"] = abs(phase_sum - out["stepAvg"])
    tolerance = max(1.0, 0.01 * out["stepAvg"])
    out["phaseSumOK"] = out["phaseSumDeltaMs"] <= tolerance

try:
    resp = json.load(open(response))
    choice = resp["choices"][0]
    out["finish_reason"] = choice.get("finish_reason")
    out["usage"] = resp.get("usage")
    content = choice["message"]["content"]
    ids = tok.encode(content).ids
    out["completion_tokens"] = len(ids)
    out["stream_hash"] = hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()
except Exception as e:
    out["response_error"] = str(e)

if "decodeSeconds" in out and out["decodeSeconds"]:
    try:
        out["ttlt"] = 1024.0 / float(out["decodeSeconds"])
    except (TypeError, ValueError):
        pass

print(json.dumps(out))
PYEOF