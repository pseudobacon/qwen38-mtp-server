#!/usr/bin/env python3
"""MER4B — in-pipeline effective BW for the essay decode tape.

in-pipeline BW = weights_streamed_GB / tEval_per_step_s.
Each MTP verify round streams the full 4-bit weight matrix (~14.4 GB) once.
Compare to the qmvbench sustained figure (310-355 GB/s).
"""
import json, sys

WEIGHTS_GB = 14.4  # 4-bit weight stream (from task context)

def analyze(cell_json, label):
    c = json.load(open(cell_json))
    t_eval_ms = c.get('tEvalAvg')
    step_avg_ms = c.get('stepAvg')
    if not t_eval_ms:
        print(f"{label}: no tEvalAvg in {cell_json}")
        return
    bw = WEIGHTS_GB / (t_eval_ms / 1000.0)
    print(f"{label}: tEvalAvg={t_eval_ms:.1f}ms  stepAvg={step_avg_ms:.1f}ms  "
          f"in-pipeline BW={bw:.1f} GB/s  (sustained ref 310-355 GB/s)")
    # phase sums
    print(f"  tGraphBuild={c.get('tGraphBuildAvg',0):.2f}ms  "
          f"tCacheState={c.get('tCacheStateAvg',0):.2f}ms  "
          f"tHostRead={c.get('tHostReadAvg',0):.3f}ms  "
          f"phaseSumDelta={c.get('phaseSumDeltaMs',0):.4f}ms")

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        label, path = arg.split(':', 1)
        analyze(path, label)
