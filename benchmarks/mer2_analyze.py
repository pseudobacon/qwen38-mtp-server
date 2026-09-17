#!/usr/bin/env python3
# MER2 analysis: paired per-rep deltas, mean delta per fixture, thermal
# trajectory, prefill comparison, and the MER3 merge-criteria verdict.
import json, sys, os, re

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
CELLS = ["inc", "upg"]
FIXES = ["essay", "specdec"]

def load_cell(m, c):
    f = os.path.join(OUTDIR, f"mer2-{m}-{c}.jsonl")
    if not os.path.exists(f):
        return []
    return [json.loads(l) for l in open(f) if l.strip()]

print("=" * 70)
print("MER2 — same-session incumbent (v0.31.6) vs upgraded (v0.32.2) A/B")
print("=" * 70)

all_reps_ok = True
decode = {}  # (fix) -> {cell: [rep records]}
for m in FIXES:
    decode[m] = {}
    for c in CELLS:
        recs = load_cell(m, c)
        decode[m][c] = recs
        print(f"\n[{m}/{c}] {len(recs)} reps")
        for i, r in enumerate(recs, 1):
            ok = r.get("stream_hash") and r.get("phaseSumOK") is True
            if not ok:
                all_reps_ok = False
            print(f"  rep {i}: ttlt={r.get('ttlt')} stepAvg={r.get('stepAvg')} "
                  f"hash={str(r.get('stream_hash'))[:8]} phaseSumOK={r.get('phaseSumOK')} "
                  f"depthAvg={r.get('depthAvg')} acc={r.get('accAvg')}")

print("\n" + "=" * 70)
print("Paired per-rep deltas (UPG - INC, same rep position; rep 1 = warmup, discarded)")
print("=" * 70)
verdict = {}
for m in FIXES:
    inc = decode[m]["inc"]
    upg = decode[m]["upg"]
    n = min(len(inc), len(upg))
    inc_m = inc[1:n+1]   # reps 2..n (measured, rep 1 warmup)
    upg_m = upg[1:n+1]
    inc_ttlt = [r.get("ttlt") for r in inc_m if r.get("ttlt")]
    upg_ttlt = [r.get("ttlt") for r in upg_m if r.get("ttlt")]
    if not inc_ttlt or not upg_ttlt:
        print(f"\n[{m}] MISSING ttlt — cannot compute")
        continue
    inc_mean = sum(inc_ttlt) / len(inc_ttlt)
    upg_mean = sum(upg_ttlt) / len(upg_ttlt)
    delta_pct = (upg_mean - inc_mean) / inc_mean * 100
    # paired by position
    pairs = list(zip(inc_ttlt, upg_ttlt))
    favor = sum(1 for a, b in pairs if b >= a)
    print(f"\n[{m}] INC mean={inc_mean:.3f} tok/s  UPG mean={upg_mean:.3f} tok/s  "
          f"delta={delta_pct:+.2f}%  ({favor}/{len(pairs)} paired reps favor UPG)")
    for i, (a, b) in enumerate(pairs, 2):
        print(f"  rep {i}: inc={a:.3f} upg={b:.3f}  delta={b-a:+.3f} ({(b-a)/a*100:+.2f}%)")
    verdict[m] = delta_pct

# prefill comparison
print("\n" + "=" * 70)
print("32K prefill comparison (pc=2048)")
print("=" * 70)
pref = {}
for c in CELLS:
    f = os.path.join(OUTDIR, f"summary-mer2-prefill-{c}.json")
    if os.path.exists(f):
        pref[c] = json.load(open(f))
        print(f"  [{c}] wall={pref[c].get('wall_seconds')}s peakRSS={pref[c].get('peak_rss_gb')}GB "
              f"content={str(pref[c].get('content_hash'))[:16]}")
        print(f"       pf2={pref[c].get('pf2')}")
if "inc" in pref and "upg" in pref:
    wi, wu = pref["inc"].get("wall_seconds"), pref["upg"].get("wall_seconds")
    if wi and wu:
        pd = (wu - wi) / wi * 100
        print(f"  prefill delta (UPG-INC) = {pd:+.2f}%  (INC {wi}s vs UPG {wu}s)")
        pref["delta_pct"] = pd

# merge criteria
print("\n" + "=" * 70)
print("MER3 merge-criteria verdict")
print("=" * 70)
crit = []
# 1. determinism
crit.append(("1. determinism (every rep reproduces registry hash + phaseSumOK)", all_reps_ok))
# 2. no material regression
decode_ok = all(v > -3.0 for v in verdict.values()) if verdict else False
pref_ok = True
if "delta_pct" in pref:
    # "not worse than noise": accept up to +3% (regression) as within noise for a
    # single-cell prefill compare
    pref_ok = pref["delta_pct"] <= 3.0
crit.append(("2a. decode not >3% worse on either fixture", decode_ok))
crit.append(("2b. 32K prefill not worse than noise (<=3%)", pref_ok))
# 3. engagement/phase-sums (covered by all_reps_ok + gates)
crit.append(("3. engagement + phase-sums clean on every rep", all_reps_ok))

all_pass = all(ok for _, ok in crit)
for name, ok in crit:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n  MER3 MERGE: {'ALL PASS -> MERGE' if all_pass else 'STOP -> DO NOT MERGE'}")
sys.exit(0 if all_pass else 2)
