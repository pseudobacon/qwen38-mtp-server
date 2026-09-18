# SCH1 — MTP Round Kill-Switch Ledger

**Run ID:** sch1-20260918-0041
**Date:** 2026-09-18
**Decision:** **STOP** — kill switch triggers. Addressable per-dispatch sync + round-structure overhead is **2.68 ms/round (3.2%)**, far below the 8 ms (~10%) threshold. **Do NOT proceed to SCH2 restructures.**

## Instrumentation

- **GPU side:** xctrace Metal System Trace (launched mode), `metal-gpu-intervals` table, server process (pid 6733), 36,644 GPU execution intervals.
- **Host side:** `QWEN_MTP_STEP_TRACE=1` + `MLX_QWEN_MTP_TRACE=1` → `mtp-anchor:` per-round mach-uptime phase stamps (t_verify_built → t_eval_done = eval window) + `mtp-trace:` per-round phase µs (eval_wall, verify_build, readout, commit, upkeep).
- **Join:** mach-uptime offset calibrated (median delta = 42 ms; total GPU-busy stable across 0–40 ms, 0.5% variation). Per-round GPU-busy = sum of server GPU-interval overlap with the round's [t_verify_built, t_eval_done] window.
- **Protocol:** steady-state k2 decode, essay-1024, MISS cache, greedy (temp 0, top_k 1), 1024 tokens, 457 decode rounds (rounds 5..461 steady-state).

## Ledger (median, 429 sane rounds)

| Category | Description | Median (ms) | % of round |
|---|---|---|---|
| **round total** | round_us | **84.6** | 100% |
| **(i) kernel exec** | GPU busy in eval window | **76.43** | 90.6% |
| **(ii) sync/idle** | eval wall − GPU busy (inter-CB gaps) | **1.224** | 1.4% |
| **(iii) host build/read** | verify_build + readout | **3.920** | 4.6% |
| **(iv) round-struct** | commit + upkeep | **1.451** | 1.7% |

- **GPU utilization in eval window: 97.4%** (matches prior K2 decomposition: 98.4%).
- **(ii)+(iv) = 2.675 ms < 8 ms → KILL SWITCH TRIGGERS.**

## Interpretation

The k2 round is **90.6% kernel execution** (GPU busy, useful work). The addressable overhead the lever #1 (batched/fused dispatch + state amortization across MTP steps) targets is:

- **(ii) sync/idle = 1.224 ms** — the inter-command-buffer gaps within the eval window (host waiting on GPU). This is the per-dispatch sync cost.
- **(iv) round-structure = 1.451 ms** — commit + upkeep (accept-walk, KV cache state update).

Together **2.68 ms/round (3.2%)**, an order of magnitude below the 8 ms (~10%) threshold. Even a perfect restructure that eliminates all of (ii)+(iv) saves only 3.2% of the round — not worth the complexity and correctness risk of a per-round dispatch restructure.

### What is NOT addressable by lever #1

- **(i) kernel exec = 76.43 ms (90.6%)** — the QMV/attention kernel execution. This is the actual useful work; not addressable by scheduling (it's the weight stream + compute).
- **(iii) host build/read = 3.920 ms (4.6%)** — the verify graph construction (verify_build) + readout. This is **host-side graph construction**, a separate lever (MLX graph building), NOT the per-dispatch sync. It is NOT in the kill-switch criterion. (If pursued separately, it's a different task with its own analysis.)

### Why the gap (176.6 GB/s) is NOT a scheduling artifact

The eval window is 97.4% GPU-busy. The 176.6 GB/s in-pipeline bandwidth (vs 310–355 GB/s sustained) is therefore **not** inter-CB gaps (only 1.22 ms/round idle) and **not** round-structure overhead (1.45 ms/round). It is the **M=3 verify geometry itself**: 3× activations/KV writes make the M=3 verify less memory-bound than M=1 (250 GB/s), and the QMV kernels at M=3 are the bottleneck. This is consistent with PRO2 (M=3 verify = 79.52 ms ≈ in-pipeline 83.2 ms) and PRO1 (kernel-mix interleave is free).

## Conclusion

- **STOP.** The per-round scheduling restructure (lever #1) is not worth it: 2.68 ms/round addressable (3.2% < 10%).
- The 176.6 GB/s in-pipeline bandwidth is an **M=3 verify geometry property**, not a scheduling artifact. It is not closable by batched/fused dispatch.
- No server/engine source changes (kill switch is a no-code STOP).
- The (iii) verify_build (3.92 ms, 4.6%) is a separate host-side graph-construction lever, out of scope for this task (and below the 8 ms threshold on its own).

## Verification

- `benchmarks/results/sch1-20260918-0041/sch1-ledger.csv` — per-round ledger (457 rounds).
- `benchmarks/results/sch1-20260918-0041/sch1-gpu-intervals.xml` — GPU intervals (xctrace export).
- `benchmarks/results/sch1-20260918-0041/sch1-mtp-trace.log` — mtp-anchor + mtp-trace lines.
- Analysis: `benchmarks/results/sch1-20260918-0041/sch1_final.py`.
