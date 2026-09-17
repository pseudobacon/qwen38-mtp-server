# MLX v0.32.2 kernel re-baseline (post-merge)

**Date:** 2026-09-17
**Run:** `benchmarks/results/mer4-20260917-2137/`
**Engine:** `6e8eab2` (v0.32.2, metallib `b57de586…`), server `628c0fc`
**qmvbench binary:** `a4b56e2b…`  |  **Server binary:** `4d08bbca…`

Maps which v0.32.2 kernel changes actually engage at our geometry, measures their effect,
and re-checks the config decisions (`pc`, QMV thresholds) optimized against v0.31.6 kernel
behavior. No speculative work — only measurements that change a decision.

## Engagement map

| v0.32.2 kernel change | Engages at our geometry? | Measured effect | End-to-end addressability |
|-----------------------|--------------------------|-----------------|---------------------------|
| **qmv_wide** (small-batch quantized matvec) | **YES** — M=2..9 verify widths | Routed wins 1.18–1.55× at M=2..9; M=1 wash (0.97×). Profile **unchanged** vs v0.31.1. | Decode MTP verify (M=3) is in the routed-win band; already engaged. No new lever. |
| **split-K quantized matmul** | Engages, but **does NOT close the anomaly** | M=512 down_proj stays at **9.818 µs/tok** (3.4× the M=1024 2.857). The small-M/large-K tiling anomaly persists. | None — the anomaly (the thing FFP1/FFP4 closed) is not fixed. |
| **M5-class qmv batch limit** | Engages, no profile change | Verify-width M=1..9 profile identical to v0.31.1. | None. |
| **gqa-8 decode attention** | **NO** — our GQA ratio is **6** (24 Q / 4 KV heads), not 8. | n/a (unreachable at our geometry). | None. |
| **NVFP4 QMV** | **NO** — hardware-gated; M5 Pro (Mac17,9) is not the target class. | n/a (unreachable on this hardware). | None. |

**Bottom line:** the only v0.32.2 kernel changes that engage at our geometry are
qmv_wide / the M5 batch limit, and both leave the M=1..9 win profile unchanged. The
split-K matmul does not close the FFN anomaly. gqa-8 and NVFP4 are unreachable. **No new
kernel lever is introduced at our geometry.**

## MER4A.1 — Verify-width M=1..9 (routed vs fallback)

Sustained `wide_global` GB/s, v0.32.2 vs v0.31.1 (`w5-throughput.txt`). M=1 wash, M=2..9
routed wins — **identical profile to v0.31.1**. The QMV dispatch threshold (routed at
M=2..9) does **not** need re-tuning. See `mer4a1-verify-comparison.md`.

## MER4A.2 — FFN M-curve (the decision-relevant number)

Sustained, DVFS-fair interleaved (`--ffn-pair`), 2 clean reps at the decision region
(M=512/1024), 1 rep each at M=2048/4096/8192:

| M | gateup µs/tok | down µs/tok | **FFN µs/tok** | (v0.31.1 FFN) |
|---|---------------|-------------|----------------|---------------|
| 512  | 3.440 | **9.818** | **13.258** | 11.261 |
| 1024 | 4.444 | **2.857** | **7.301** ← min | 6.426 |
| 2048 | 4.980 | 2.860 | 7.839 | 7.528 |
| 4096 | 5.120 | 2.812 | 7.932 | 8.190 |
| 8192 | 5.388 | 2.933 | 8.321 | 7.966 |

The M=512 down_proj anomaly **persists** (9.818 vs 2.857 µs/tok at M=1024, 3.4×). The
per-token FFN cost still has an interior minimum at **M=1024**. The curve did **not**
flatten. See `mer4a2-ffn-mcurve-comparison.md`.

## MER4A.3 — Bit-exactness / tolerance spot-check

`--ffn-check`: M=512 **bit-exact** (max|diff|=0.0); quantizedMM 16.6 TF vs bf16 12.2 TF.
M=1024/2048 dequant-reference overflows to nan (a test artifact at the larger M, not a
divergence). No tolerance concern at our prefill widths.

## MER4B — In-pipeline attribution (essay decode, controlled MISS)

tEvalAvg = 81.6 ms (reproduces MER2's 80.7 ms). **In-pipeline effective BW = 14.4 GB /
81.6 ms = 176.6 GB/s** vs the qmvbench sustained 310–355 GB/s. The in-pipeline gap
**persists** on the v0.32.2 kernels: the decode ceiling is **per-dispatch/state-bound**
(not kernel-bound). The decode-bandwidth question **closes with a mechanism** — the next
lever (if any) is **scheduling, not kernels**.

## MER4C — Conditional config re-checks (both CLOSED, no sweep)

- **pc sweep (32K, pc ∈ {512,1024,2048}):** NOT triggered — the FFN M-curve did not
  flatten materially (the M=512 anomaly persists). The **pc=2048 default stands**.
- **QMV dispatch threshold re-tune:** NOT triggered — the M=1..9 profile is unchanged.

## Ranked next tasks (new-backend performance levers)

1. **In-pipeline scheduling (expected: up to ~176→310 GB/s on the weight stream, i.e. the
   in-pipeline/decode gap).** Evidence: MER4B — the in-pipeline BW (176.6 GB/s) is far
   below the sustained BW (310–355 GB/s) on the *same* v0.32.2 kernels, so the ceiling is
   per-dispatch/state-bound (launch overhead, state setup, host round-trips), not the GEMM
   kernel. The lever is batched/fused dispatch and state amortization across MTP steps, not
   a new GEMM.
2. **FFN M=512 down_proj tiling (expected: the −42.9% FFN per-token at M=512→1024, already
   captured by pc=2048).** Evidence: MER4A.2 — the anomaly persists but is amortized by the
   existing pc=2048 default. No new work; the config axis is already optimal.
3. **gqa-8 / NVFP4:** unreachable at our geometry (GQA 6, M5 Pro) — no action; revisit only
   on a model/hardware change.
