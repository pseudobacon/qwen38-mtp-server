#!/usr/bin/env python3
"""Phase 3 post-processing.

Merges the pre-rep thermal snapshots (from .tmp/phase3-thermal.log, THERMAL
marker blocks) into the per-rep JSONL records of each cell file, then reports
per-cell mean / min / max over the 5 measured reps (reps 2-6; rep 1 is the
discarded warmup).

Usage: phase3_report.py <cell.jsonl> <cell-prefix> [<cell.jsonl> <prefix> ...]
       thermal log path is the fixed .tmp/phase3-thermal.log next to this
       script's parent benchmarks/ dir.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
THERMAL_LOG = os.path.join(HERE, os.pardir, ".tmp", "phase3-thermal.log")

FIELDS = [
    "avgStepMs", "tEvalAvg", "tGraphBuildAvg", "tCacheStateAvg",
    "tHostReadAvg", "wall_seconds", "decodeSeconds", "ttlt",
]


def load_thermal_blocks(path):
    blocks = []
    cur = None
    with open(path) as f:
        for line in f:
            m = re.match(r"^THERMAL (\d+) (\S+) (.+)$", line.rstrip("\n"))
            if m:
                if cur is not None:
                    blocks.append(cur)
                cur = [m.group(2), []]
            elif cur is not None:
                cur[1].append(line.rstrip("\n"))
    if cur is not None:
        blocks.append(cur)
    return blocks


def rep_of(tag):
    m = re.match(r"^\S+-r(\d+)$", tag)
    return int(m.group(1)) if m else None


def merge(cell_file, prefix, blocks):
    snaps = [text for tag, text in blocks if tag.startswith(prefix + "-r")]
    with open(cell_file) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if len(recs) > len(snaps):
        print(f"ERROR {cell_file}: {len(recs)} records but only {len(snaps)} "
              f"thermal snapshots for prefix {prefix!r}", file=sys.stderr)
        sys.exit(1)
    with open(cell_file, "w") as f:
        for i, rec in enumerate(recs):
            rec["thermal"] = "\n".join(snaps[i])
            f.write(json.dumps(rec) + "\n")
    return recs


def stats(recs):
    measured = [r for r in recs if (rep_of(r["tag"]) or 0) >= 2]
    if len(measured) != 5:
        print(f"WARNING {len(measured)}/5 measured reps available", file=sys.stderr)
    out = {"measured_reps": len(measured)}
    for k in FIELDS:
        vals = [r[k] for r in measured if isinstance(r.get(k), (int, float))]
        if vals:
            out[k] = (sum(vals) / len(vals), min(vals), max(vals))
    for k in ("accepted", "proposed", "rounds", "acceptedPerStep"):
        vals = {r.get(k) for r in measured}
        out[k] = vals.pop() if len(vals) == 1 else vals
    ttlt_wall = [1024.0 / r["wall_seconds"] for r in measured
                 if isinstance(r.get("wall_seconds"), (int, float)) and r["wall_seconds"]]
    if ttlt_wall:
        out["ttlt_wall"] = (sum(ttlt_wall) / len(ttlt_wall),
                            min(ttlt_wall), max(ttlt_wall))
    return out


def fmt(t):
    if t is None:
        return "n/a"
    if isinstance(t, (int, float)):
        return f"{t:.3f}"
    return str(t)


def main():
    argv = sys.argv[1:]
    blocks = load_thermal_blocks(THERMAL_LOG)
    cells = []
    for i in range(0, len(argv), 2):
        cell_file = os.path.join(HERE, argv[i]) if not argv[i].startswith("/") else argv[i]
        prefix = argv[i + 1]
        recs = merge(cell_file, prefix, blocks)
        cells.append((prefix, recs, stats(recs)))

    header = (f"{'cell':<10} {'field':<16} {'mean':>10} {'min':>10} {'max':>10}")
    print(header)
    print("-" * len(header))
    for prefix, recs, st in cells:
        gate = {r.get("stream_hash", "")[:8] for r in recs}
        print(f"{prefix:<10} {'gate':<16} {sorted(gate)}")
        for k in FIELDS + ["ttlt_wall", "accepted", "proposed", "rounds"]:
            if k in st:
                v = st[k]
                if isinstance(v, tuple):
                    print(f"{prefix:<10} {k:<16} {v[0]:>10.3f} {v[1]:>10.3f} {v[2]:>10.3f}")
                else:
                    print(f"{prefix:<10} {k:<16} {fmt(v):>10}")
        print()

    # in-session ablation delta: B3 (compiled OFF) vs B1 (default)
    if len(cells) >= 2:
        b1 = next((s for p, _, s in cells if p == "B1"), None)
        b3 = next((s for p, _, s in cells if p == "B3"), None)
        if b1 and b3:
            print("ablation delta B3 - B1 (compiled decode OFF - default):")
            for k in FIELDS + ["ttlt_wall"]:
                if k in b1 and k in b3:
                    d = b3[k][0] - b1[k][0]
                    pct = 100.0 * d / b1[k][0] if b1[k][0] else float("nan")
                    print(f"  {k:<16} {d:+10.3f}  ({pct:+.2f}%)")


if __name__ == "__main__":
    main()
