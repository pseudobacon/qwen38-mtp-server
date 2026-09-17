#!/usr/bin/env python
"""MCP2 gate analysis. Reads mcp2-reps.jsonl (Phase B, 32K, 6 reps x 3 pc) and
mcp2-hashgates.jsonl (Phase A). Applies the MCP2 decision gate and prints the
verdict.

Gate (to flip the default pc to P):
  1. mean 32K wall(P) <= 0.95 * mean 32K wall(512)   [>= 5% better]
  2. >= 4/5 measured paired reps: wall(P) < wall(512)
  3. all phase_sum_check true; content_hash self-consistent per pc
  (64K non-regression is Phase C, run separately if this passes.)
Rep 1 is discarded (warmup); reps 2..6 are measured.
"""
import json, sys
from collections import defaultdict

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

reps = load("/Users/cwong/ai/qwen38-mtp-server/benchmarks/results/mcp-20260917/mcp2-reps.jsonl")
hashg = load("/Users/cwong/ai/qwen38-mtp-server/benchmarks/results/mcp-20260917/mcp2-hashgates.jsonl")

# Phase B: group by (rep, pc) -> wall
# reps: list of {rep, pc, order, wall_s, total_ms, ffn_ms, gdn_ms, attn_ms, ...}
bycell = {}
for r in reps:
    bycell[(r["rep"], r["pc"])] = r

PCs = [512, 1024, 2048]
MEASURED = [2, 3, 4, 5, 6]   # rep 1 discarded

# The gate metric is the eval-sync prefill wall = PF2 total_ms (per the task:
# "prefill wall (eval-sync)"). wall_s (curl) is the secondary, noisier total.
def metric(r):
    return r.get("total_ms")   # eval-sync prefill wall, ms

mean_wall = {}
for pc in PCs:
    ws = [metric(bycell[(rep, pc)]) for rep in MEASURED if (rep, pc) in bycell
          and metric(bycell[(rep, pc)]) is not None]
    mean_wall[pc] = sum(ws) / len(ws) if ws else None

print("=== mean 32K eval-sync prefill wall (total_ms, measured reps 2..6) ===")
for pc in PCs:
    if mean_wall[pc]:
        print(f"  pc={pc}: {mean_wall[pc]/1000:.2f}s")
    else:
        print(f"  pc={pc}: n/a")

base = mean_wall[512]
print(f"\n=== gate per candidate pc (base pc=512 = {base/1000:.2f}s) ===")
for pc in [1024, 2048]:
    if not mean_wall[pc]:
        continue
    mean_gain = (base - mean_wall[pc]) / base * 100
    favors = 0
    n = 0
    for rep in MEASURED:
        if (rep, pc) in bycell and (rep, 512) in bycell:
            wpc = metric(bycell[(rep, pc)])
            w512 = metric(bycell[(rep, 512)])
            if wpc is not None and w512 is not None:
                n += 1
                if wpc < w512:
                    favors += 1
    gate1 = mean_gain >= 5.0
    gate2 = n > 0 and favors >= 4
    verdict = "PASS" if (gate1 and gate2) else "FAIL"
    print(f"  pc={pc}: mean={mean_wall[pc]/1000:.2f}s ({mean_gain:+.1f}% vs 512) "
          f"paired {favors}/{n} favor -> gate1(>=5%)={'Y' if gate1 else 'N'} "
          f"gate2(>=4/5)={'Y' if gate2 else 'N'} => {verdict}")

# determinism: content_hash self-consistent per pc (32K); phase_sum_check all true
print("\n=== determinism / phase-sum ===")
hashes = defaultdict(set)
phase_ok = True
for r in reps:
    hashes[r["pc"]].add(r.get("content_hash"))
    if not r.get("phase_sum_check", False):
        phase_ok = False
        print(f"  PHASE-SUM FAIL rep={r['rep']} pc={r['pc']}")
for pc in sorted(hashes):
    hs = hashes[pc]
    print(f"  pc={pc}: 32K hashes {len(hs)} distinct "
          f"({'CONSISTENT' if len(hs) <= 1 else 'INCONSISTENT (STOP)'}) {sorted(hs)[:2]}")
print(f"  phase_sum_check all true: {phase_ok}")

# per-phase at each pc (measured reps, mean)
print("\n=== per-phase (measured reps, ms) ===")
print(f"  {'pc':>5} {'ffn':>8} {'gdn':>7} {'attn':>8} {'sdpa':>7} {'norm':>7} {'resid':>6} {'total':>9}")
for pc in PCs:
    cells = [bycell[(rep, pc)] for rep in MEASURED if (rep, pc) in bycell]
    if not cells:
        continue
    def m(k): 
        v = [c[k] for c in cells if k in c]
        return sum(v)/len(v) if v else 0
    print(f"  {pc:>5} {m('ffn_ms'):>8.0f} {m('gdn_ms'):>7.0f} {m('attn_ms'):>8.0f} "
          f"{m('sdpa_ms'):>7.0f} {m('norm_ms'):>7.0f} {m('residual_ms'):>6.0f} "
          f"{m('total_ms'):>9.0f}")

# Phase A hash gates
print("\n=== Phase A hash gates ===")
for h in hashg:
    exp = {"8k":"660dd1208737764c","16k":"2e583ad29dc28465","32k":"97bc0d74846a043b"}.get(h["len"])
    hh = h.get("content_hash","")[:16]
    st = "PASS" if (h["len"] in ("8k","16k") and hh==exp) else ("BASE" if hh==exp else "diff")
    print(f"  pc={h['pc']} {h['len']}: hash={hh} {st} (buffer={h.get('buffer_gb')}GB, "
          f"rss={h.get('peak_rss_gb')}GB)")
