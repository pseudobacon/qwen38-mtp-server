#!/bin/bash
# Item 4 — lazy SSD restore: eager vs lazy A/B on time-to-readyz.
# 5 restarts each, interleaved. SSD cleared per restart (so the delta isolates
# the weight-identity hash, not the prefix restore). The lazy path moves the
# ~8s hash off the critical startup path (QWEN_KV_SSD_LAZY=1).
set -u
RES=/Users/cwong/ai/qwen38-mtp-server/benchmarks/results/item4-lazy-ssd
cd /Users/cwong/ai/qwen38-mtp-server || exit 1
mkdir -p "$RES"
BIN=.build/release/qwen38-mtp-server

measure() { # tag envspec
  local tag=$1 envspec=$2
  rm -rf ~/.qwen38-mtp/kv-ssd 2>/dev/null
  local t0=$(python3 -c 'import time;print(time.time())')
  if [ -n "$envspec" ]; then
    env "$envspec" QWEN_STARTUP_TRACE=1 $BIN serve --port 18097 --model ./weights \
      > "$RES/$tag.out" 2> "$RES/$tag.err" &
  else
    QWEN_STARTUP_TRACE=1 $BIN serve --port 18097 --model ./weights \
      > "$RES/$tag.out" 2> "$RES/$tag.err" &
  fi
  local srv=$!
  for i in $(seq 1 180); do
    local code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:18097/readyz" 2>/dev/null)
    [ "$code" = "200" ] && break
    kill -0 $srv 2>/dev/null || break
    sleep 1
  done
  local t1=$(python3 -c 'import time;print(time.time())')
  kill $srv 2>/dev/null; pkill -f "qwen38-mtp-server serve --port 18097" 2>/dev/null
  wait $srv 2>/dev/null
  python3 -c "import json;open('$RES/$tag.json','w').write(json.dumps({'tag':'$tag','timeToReady_s':round($t1-$t0,2)}))"
  pmset -g therm > "$RES/therm-$tag.txt" 2>/dev/null || true
}

# Alternate the order so each mode gets the (cold-GPU) first position equally
# and the warm-GPU second position equally, removing the GPU first-position
# artifact. Odd r: EAGER then LAZY; even r: LAZY then EAGER.
for r in 1 2 3 4 5; do
  if [ $((r % 2)) -eq 1 ]; then
    measure "EAGER-r$r" ""
    measure "LAZY-r$r" "QWEN_KV_SSD_LAZY=1"
  else
    measure "LAZY-r$r" "QWEN_KV_SSD_LAZY=1"
    measure "EAGER-r$r" ""
  fi
done
echo "DONE"
