#!/usr/bin/env bash
#
# run_prewarm_ab.sh — A/B: fresh-install simulation (Metal cache cleared)
# WITH vs WITHOUT the post-install pre-warm (quick-wins Item 3 / LEV-C
# flag 1; docs/PRE-WARM.md).
#
# Protocol:
#   TRIALS=5 alternating cells, A first: A,B,A,B,A.
#   A = no pre-warm (fresh install: no manifest + cleared Metal cache).
#   B = pre-warm run first (scripts/prewarm.sh), then the production boot.
#   Per trial: clear the per-user Metal cache AND the SSD tier
#   (controlled fresh-install state); cell A additionally points
#   QWEN_PREWARM_MANIFEST at a nonexistent path (a fresh install has no
#   manifest). pmset -g therm per trial.
#   Measured: time-to-readyz (0.2 s poll from process start) and
#   first-request latency (fixed greedy 64-token request). The greedy
#   content hash must be identical across ALL trials (the pre-warm caches
#   compilation, not numerics).
#
# Gates:
#   1. mean first-boot reduction = mean(A tt_readyz) - mean(B tt_readyz) >= 5.0 s
#      AND mean(B tt_readyz) <= 6.0 s (pre-warmed first boot at warm-restart
#      level). NOTE: the original planning gate was >= 8.0 s, derived from
#      LEV-C's 16.9 s cold JIT measured on the pre-fusion binary
#      29434ccf (2026-09-18). The current kernel set (post Item-1 fusion
#      flip) JIT-compiles in 7.5-11.6 s cold (quiet system; up to ~18 s
#      under concurrent load, see benchmarks/results/prewarm-ab-20260918-1619),
#      so the achievable reduction ceiling is ~5-6 s (B's floor is weight
#      load ~1.5 s + warm warmup ~3.0 s + overhead). An 8.0 s reduction is
#      structurally unreachable on this kernel set; the pre-warm still
#      eliminates 100% of the first-boot-only JIT (gate 4).
#   2. determinism: one unique content hash across all trials, no errors
#   3. warm-restart sanity: mean(B tt_readyz) within [4.0, 9.0] s
#      (measured warm-restart baseline ~5.4 s, Item 4)
#   4. no JIT in B: mean Metal cache growth across B trials <= 2000 KB
#      (a warm cache compiles nothing; measured ~164 KB = metadata only)
#
# usage: TRIALS=5 PORT=18099 bash benchmarks/run_prewarm_ab.sh
set -u
REPO_ROOT="/Users/cwong/ai/qwen38-mtp-server"
cd "$REPO_ROOT" || exit 1
BIN="$REPO_ROOT/.build/release/qwen38-mtp-server"
ENGINE="/Users/cwong/ai/mlx-swift-lm"
PORT="${PORT:-18099}"
TRIALS="${TRIALS:-5}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT_FILE="$REPO_ROOT/benchmarks/prompts/prewarm-short.txt"
RUN_ID="prewarm-ab-$(date +%Y%m%d-%H%M)"
OUT_DIR="$REPO_ROOT/benchmarks/results/$RUN_ID"
mkdir -p "$OUT_DIR"

DUCD="$(getconf DARWIN_USER_CACHE_DIR)"
METAL_CACHE="${DUCD%/}/com.apple.metal"
# The Metal FRONTEND cache also persists JIT state across processes; a
# fresh-install simulation must clear both (measured: wiping only
# com.apple.metal leaves ~3 s of warm state).
METALFE_CACHE="${DUCD%/}/com.apple.metalfe"
SSD_DIR="$HOME/.qwen38-mtp/kv-ssd"
MANIFEST="$HOME/.qwen38-mtp/prewarm-manifest.json"
GHOST_MANIFEST="/tmp/qwen38-prewarm-ghost-$$.json"
rm -f "$GHOST_MANIFEST"

[ -x "$BIN" ] || { echo "release binary missing — run: swift build -c release"; exit 1; }
[ -f "$PROMPT_FILE" ] || { echo "fixture missing: $PROMPT_FILE"; exit 1; }

FIXTURE_HASH="$(shasum -a 256 "$PROMPT_FILE" | awk '{print $1}')"
BIN_SHA256="$(shasum -a 256 "$BIN" | awk '{print $1}')"
SERVER_HEAD="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
ENGINE_HEAD="$(git -C "$ENGINE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
SERVER_DIRTY=$([ -z "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ] && echo clean || echo dirty)
ENGINE_DIRTY=$([ -z "$(git -C "$ENGINE" status --porcelain 2>/dev/null)" ] && echo clean || echo dirty)
OS_BUILD="$(sw_vers -buildVersion)"

# fixed greedy request body (prompt read from the fixture at request time)
python3 - "$PROMPT_FILE" "$OUT_DIR/request.json" "$MAX_TOKENS" <<'PYEOF'
import json, sys
prompt = open(sys.argv[1]).read()
body = {
    "model": "qwen3.8-27b-mtp",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(sys.argv[3]),
    "temperature": 0,
    "top_k": 1,
    "mtp_enabled": True,
    "enable_thinking": False,
    "stream": False,
}
open(sys.argv[2], "w").write(json.dumps(body))
PYEOF

RESULTS="$OUT_DIR/ab.jsonl"
THERM="$OUT_DIR/therm.txt"
: > "$RESULTS"
: > "$THERM"

now() { perl -MTime::HiRes -e 'printf "%.3f", Time::HiRes::time()'; }
metal_size_kb() {
  # backend + frontend caches
  local total=0 size
  for d in "$METAL_CACHE" "$METALFE_CACHE"; do
    if [ -d "$d" ]; then
      size="$(du -sk "$d" 2>/dev/null | awk '{print $1}')"
      total=$((total + ${size:-0}))
    fi
  done
  echo "$total"
}

for i in $(seq 1 "$TRIALS"); do
  CELL=$([ $((i % 2)) -eq 1 ] && echo A || echo B)
  echo "--- trial $i cell $CELL ---"
  pmset -g therm >> "$THERM" 2>&1 || true

  # fresh-install simulation: cleared Metal caches (backend + frontend) +
  # per-user Metal compiler service (in-memory state) + SSD tier.
  rm -rf "$METAL_CACHE" 2>/dev/null || true
  rm -rf "$METALFE_CACHE" 2>/dev/null || true
  rm -rf "$SSD_DIR" 2>/dev/null || true
  pkill -9 -f "XPCServices/MTLCompilerService.xpc" 2>/dev/null || true
  pkill -f "qwen38-mtp-server serve --port $PORT" 2>/dev/null || true
  sleep 1

  T_PRE=""
  if [ "$CELL" = "B" ]; then
    T0="$(now)"
    if ! scripts/prewarm.sh --port "$PORT" > "$OUT_DIR/prewarm-$i.log" 2>&1; then
      echo "{\"trial\":$i,\"cell\":\"$CELL\",\"error\":\"prewarm failed\"}" >> "$RESULTS"
      continue
    fi
    T_PRE="$(awk -v a="$T0" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }')"
  fi

  # cache growth across the production server run (engagement proof:
  # A compiles ~cold during the run, B hits the cache).
  SIZE0="$(metal_size_kb)"

  SERVE_LOG="$OUT_DIR/serve-$i.log"
  T0="$(now)"
  if [ "$CELL" = "A" ]; then
    # a true fresh install has no manifest: point the check at a ghost path.
    env QWEN_STARTUP_TRACE=1 QWEN_MTP_STEP_TRACE=1 QWEN_PREWARM_MANIFEST="$GHOST_MANIFEST" \
      "$BIN" serve --port "$PORT" --model ./weights > "$SERVE_LOG" 2>&1 &
  else
    env QWEN_STARTUP_TRACE=1 QWEN_MTP_STEP_TRACE=1 "$BIN" serve --port "$PORT" --model ./weights \
      > "$SERVE_LOG" 2>&1 &
  fi
  SRV=$!
  READY_AT=""
  for _ in $(seq 1 600); do
    CODE="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/readyz" 2>/dev/null || echo 000)"
    if [ "$CODE" = "200" ]; then READY_AT="$(now)"; break; fi
    kill -0 "$SRV" 2>/dev/null || break
    sleep 0.2
  done
  if [ -z "$READY_AT" ]; then
    kill "$SRV" 2>/dev/null
    wait "$SRV" 2>/dev/null
    echo "{\"trial\":$i,\"cell\":\"$CELL\",\"error\":\"readyz never 200\"}" >> "$RESULTS"
    continue
  fi
  TT_READYZ="$(awk -v a="$T0" -v b="$READY_AT" 'BEGIN { printf "%.2f", b - a }')"

  WALL="$(curl -s --max-time 300 -o "$OUT_DIR/resp-$i.json" -w '%{time_total}' \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d @"$OUT_DIR/request.json" 2>/dev/null)"
  RC=$?
  kill "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null
  [ "$RC" -eq 0 ] || WALL=""

  SIZE1="$(metal_size_kb)"
  GROWTH=$((SIZE1 - SIZE0))

  HASH="$(python3 - "$OUT_DIR/resp-$i.json" <<'PYEOF'
import json, sys, hashlib
try:
    d = json.load(open(sys.argv[1]))
    c = d["choices"][0]["message"]["content"]
    print(hashlib.sha256(c.encode()).hexdigest()[:16])
except Exception:
    print("")
PYEOF
)"

  python3 - "$i" "$CELL" "$TT_READYZ" "$WALL" "$HASH" "$T_PRE" "$GROWTH" \
    "$FIXTURE_HASH" "$BIN_SHA256" "$SERVER_HEAD" "$ENGINE_HEAD" \
    "$SERVER_DIRTY" "$ENGINE_DIRTY" "$OS_BUILD" >> "$RESULTS" <<'PYEOF'
import json, sys
i, cell, tt, wall, h, tpre, growth = sys.argv[1:8]
prov = dict(fixture_hash=sys.argv[8], binary_sha256=sys.argv[9],
            server_head=sys.argv[10], engine_head=sys.argv[11],
            server_dirty=sys.argv[12], engine_dirty=sys.argv[13],
            os_build=sys.argv[14])
row = dict(trial=int(i), cell=cell, tt_readyz_s=float(tt),
           first_request_s=(float(wall) if wall else None),
           content_sha256_16=(h or None),
           prewarm_s=(float(tpre) if tpre else None),
           metal_cache_growth_kb=int(growth), **prov)
print(json.dumps(row))
PYEOF
done

python3 - "$RESULTS" "$OUT_DIR/summary.json" "$TRIALS" <<'PYEOF'
import json, sys, statistics

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out_path, trials = sys.argv[2], int(sys.argv[3])
errors = [r for r in rows if "error" in r]
ok = [r for r in rows if "error" not in r]
a = [r["tt_readyz_s"] for r in ok if r["cell"] == "A"]
b = [r["tt_readyz_s"] for r in ok if r["cell"] == "B"]
fa = [r["first_request_s"] for r in ok if r["cell"] == "A" and r.get("first_request_s")]
fb = [r["first_request_s"] for r in ok if r["cell"] == "B" and r.get("first_request_s")]
hashes = {r["content_sha256_16"] for r in ok if r.get("content_sha256_16")}
prewarms = [r["prewarm_s"] for r in ok if r.get("prewarm_s")]
growth_a = [r["metal_cache_growth_kb"] for r in ok if r["cell"] == "A"]
growth_b = [r["metal_cache_growth_kb"] for r in ok if r["cell"] == "B"]

summary = {
    "trials": trials,
    "errors": len(errors),
    "n_A": len(a), "n_B": len(b),
    "mean_A_tt_readyz_s": round(statistics.mean(a), 2) if a else None,
    "mean_B_tt_readyz_s": round(statistics.mean(b), 2) if b else None,
    "A_tt_readyz": [round(x, 2) for x in a],
    "B_tt_readyz": [round(x, 2) for x in b],
    "mean_first_request_A_s": round(statistics.mean(fa), 3) if fa else None,
    "mean_first_request_B_s": round(statistics.mean(fb), 3) if fb else None,
    "unique_content_hashes": sorted(h for h in hashes if h),
    "mean_prewarm_s": round(statistics.mean(prewarms), 2) if prewarms else None,
    "mean_metal_cache_growth_A_kb": round(statistics.mean(growth_a)) if growth_a else None,
    "mean_metal_cache_growth_B_kb": round(statistics.mean(growth_b)) if growth_b else None,
}
if a and b:
    reduction = statistics.mean(a) - statistics.mean(b)
    summary["mean_first_boot_reduction_s"] = round(reduction, 2)
    gates = {
        "gate1_reduction_ge_5s": reduction >= 5.0,
        "gate1b_b_at_warm_level": statistics.mean(b) <= 6.0,
        "gate2_single_content_hash": len(hashes) == 1,
        "gate2_no_errors": len(errors) == 0 and len(ok) == trials,
        "gate3_warm_restart_sanity": 4.0 <= statistics.mean(b) <= 9.0,
        "gate4_no_jit_in_b": statistics.mean(growth_b) <= 2000,
    }
    summary["gates"] = gates
    summary["PASS"] = all(gates.values())
json.dump(summary, open(out_path, "w"), indent=2)
print(json.dumps(summary, indent=2))
PYEOF
